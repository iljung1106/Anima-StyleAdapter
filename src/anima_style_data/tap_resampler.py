from __future__ import annotations

import math
import random
import time
import json
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .cradio import _load_cradio, _storage_dtype, _summary_features, preprocess_cradio_image
from .io import read_records, write_json, write_records


def select_tap_experiment_rows(
    rows: list[dict[str, Any]], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select artist-disjoint meta splits and a fixed number of images per artist."""
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_split = str(cfg.get("source_split", "train"))
    for row in rows:
        if row.get("split", "train") == source_split:
            by_style[str(row.get("style_id", row["artist"]))].append(row)

    images_per_artist = int(cfg["images_per_artist"])
    eligible = sorted(k for k, values in by_style.items() if len(values) >= images_per_artist)
    artist_count = int(cfg["artist_count"])
    if len(eligible) < artist_count:
        raise RuntimeError(f"Need {artist_count} eligible train artists, found {len(eligible)}")

    rng = random.Random(int(cfg["seed"]))
    artists = rng.sample(eligible, artist_count)
    rng.shuffle(artists)
    val_count = int(cfg["val_artists"])
    test_count = int(cfg["test_artists"])
    train_count = artist_count - val_count - test_count
    if train_count <= 0:
        raise ValueError("artist_count must exceed val_artists + test_artists")
    boundaries = (train_count, train_count + val_count)

    selected: list[dict[str, Any]] = []
    for artist_index, style_id in enumerate(artists):
        split = (
            "meta_train"
            if artist_index < boundaries[0]
            else "meta_val"
            if artist_index < boundaries[1]
            else "meta_test"
        )
        artist_rows = sorted(by_style[style_id], key=lambda row: int(row["id"]))
        chosen = rng.sample(artist_rows, images_per_artist)
        rng.shuffle(chosen)
        for sample_rank, row in enumerate(chosen):
            selected.append(
                {
                    **row,
                    "experiment_split": split,
                    "experiment_artist_index": artist_index,
                    "experiment_sample_rank": sample_rank,
                }
            )
    return selected


def _experiment_dir(destination: Path, cfg: dict[str, Any]) -> Path:
    path = destination / str(cfg.get("output_directory", "tap_resampler_experiment"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _materialize_local_feature_cache(
    rows: list[dict[str, Any]],
    source_dir: Path,
    local_dir: Path,
    feature_cfg: dict[str, Any],
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Repack the selected pilot tensors once so training avoids random NFS reads."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    manifest_dir = local_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    completed_rows: list[dict[str, Any]] = []
    shard_numbers: list[int] = []
    for path in sorted(manifest_dir.glob("part-*.parquet")):
        shard_numbers.append(int(path.stem.split("-")[-1]))
        completed_rows.extend(read_records(path))
    completed_ids = {int(row["id"]) for row in completed_rows}
    selected_ids = {int(row["id"]) for row in rows}
    if completed_ids - selected_ids:
        raise RuntimeError(
            f"Local feature cache contains {len(completed_ids - selected_ids)} stale images: "
            f"{local_dir}"
        )

    layers = sorted(int(layer) for layer in feature_cfg["layers"])
    native_globals = sorted(
        {
            int(str(variant["global"]).rsplit("_", 1)[-1])
            for variant in variants
            if str(variant.get("global", "none")).startswith("native_")
        }
    )
    needs_visual = any(
        str(variant.get("global", "none")) == "siglip_visual" for variant in variants
    )
    tensor_suffixes = [f"layer_{layer:02d}_spatial" for layer in layers]
    tensor_suffixes.extend(f"layer_{layer:02d}_siglip_cls" for layer in native_globals)
    if needs_visual:
        tensor_suffixes.append("siglip_visual")

    remaining = [row for row in rows if int(row["id"]) not in completed_ids]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remaining:
        grouped[str(row["feature_shard"])].append(row)
    shard_rows = int(feature_cfg.get("local_shard_rows", 512))
    shard_index = max(shard_numbers) + 1 if shard_numbers else 0
    tensor_buffer: dict[str, Any] = {}
    record_buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal tensor_buffer, record_buffer, shard_index
        if not record_buffer:
            return
        feature_path = local_dir / f"part-{shard_index:05d}.safetensors"
        temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
        save_file(tensor_buffer, temporary)
        temporary.replace(feature_path)
        for record in record_buffer:
            record["feature_shard"] = feature_path.name
        write_records(manifest_dir / f"part-{shard_index:05d}.parquet", record_buffer)
        completed_rows.extend(record_buffer)
        print(
            f"materialized local feature shard {shard_index} "
            f"({len(record_buffer)} images)",
            flush=True,
        )
        tensor_buffer = {}
        record_buffer = []
        shard_index += 1

    for source_shard, shard_rows_ in sorted(grouped.items()):
        with safe_open(source_dir / source_shard, framework="pt", device="cpu") as handle:
            for row in shard_rows_:
                image_id = int(row["id"])
                for suffix in tensor_suffixes:
                    key = f"{image_id}.{suffix}"
                    tensor_buffer[key] = handle.get_tensor(key).clone()
                record_buffer.append(dict(row))
                if len(record_buffer) >= shard_rows:
                    flush()
    flush()
    completed_rows.sort(key=lambda row: int(row["experiment_artist_index"]) * 100 + int(row["experiment_sample_rank"]))
    write_records(local_dir / "manifest.parquet", completed_rows)
    write_json(
        local_dir / "summary.json",
        {
            "images": len(completed_rows),
            "layers": layers,
            "native_global_layers": native_globals,
            "source_directory": str(source_dir),
            "storage_bytes": sum(path.stat().st_size for path in local_dir.glob("*.safetensors")),
        },
    )
    return completed_rows


def extract_tap_features(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    cfg = config["tap_resampler_experiment"]
    feature_cfg = cfg["features"]
    radio_cfg = {**config["cradio"], **feature_cfg.get("preprocess", {})}
    root = _experiment_dir(destination, cfg)
    selection_path = root / "selection.parquet"
    if selection_path.exists():
        rows = read_records(selection_path)
    else:
        rows = select_tap_experiment_rows(
            read_records(destination / "final_manifest.parquet"), cfg
        )
        write_records(selection_path, rows)

    feature_dir = root / "features"
    source_directory = feature_cfg.get("source_directory")
    if source_directory:
        source_dir = destination / str(source_directory)
        source_manifest = source_dir / "manifest.parquet"
        if not source_manifest.exists():
            raise FileNotFoundError(
                f"Production feature cache is not complete: {source_manifest}"
            )
        cached = {int(row["id"]): row for row in read_records(source_manifest)}
        missing = [int(row["id"]) for row in rows if int(row["id"]) not in cached]
        if missing:
            raise RuntimeError(
                f"Production feature cache is missing {len(missing)} selected images"
            )
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_rows = [
            {
                **row,
                **{
                    key: value
                    for key, value in cached[int(row["id"])].items()
                    if key
                    in {
                        "feature_shard",
                        "target_height",
                        "target_width",
                        "spatial_tokens",
                        "spatial_dim",
                        "storage_dtype",
                        "feature_signature",
                    }
                },
            }
            for row in rows
        ]
        local_cache_directory = feature_cfg.get("local_cache_directory")
        if local_cache_directory:
            output_rows = _materialize_local_feature_cache(
                output_rows,
                source_dir,
                Path(str(local_cache_directory)),
                feature_cfg,
                list(cfg["variants"]),
            )
        manifest_path = feature_dir / "manifest.parquet"
        write_records(manifest_path, output_rows)
        summary = {
            "artists": int(cfg["artist_count"]),
            "images": len(output_rows),
            "layers": sorted(int(layer) for layer in feature_cfg["layers"]),
            "source_directory": str(source_directory),
            "local_cache_directory": (
                str(local_cache_directory) if local_cache_directory else None
            ),
            "storage_bytes": 0,
            "manifest": str(manifest_path.resolve()),
        }
        write_json(feature_dir / "summary.json", summary)
        return summary

    manifest_dir = feature_dir / "manifests"
    manifest_path = feature_dir / "manifest.parquet"
    summary_path = feature_dir / "summary.json"
    if manifest_path.exists() and summary_path.exists():
        import json

        return json.loads(summary_path.read_text(encoding="utf-8"))

    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    completed_rows: list[dict[str, Any]] = []
    shard_numbers = []
    for path in sorted(manifest_dir.glob("part-*.parquet")):
        shard_numbers.append(int(path.stem.split("-")[-1]))
        completed_rows.extend(read_records(path))
    completed_ids = {int(row["id"]) for row in completed_rows}
    rows = [row for row in rows if int(row["id"]) not in completed_ids]
    model, device = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    adaptor_name = str(radio_cfg["adaptor_name"])
    layers = sorted({int(layer) for layer in feature_cfg["layers"]})
    if min(layers) < 0 or max(layers) >= len(model.model.blocks):
        raise ValueError(f"Tap layers must be in [0, {len(model.model.blocks) - 1}]")

    batch_size = int(feature_cfg.get("batch_size", radio_cfg["batch_size"]))
    shard_rows = int(feature_cfg.get("shard_rows", 16))
    max_open_buckets = int(feature_cfg.get("max_open_buckets", 16))
    storage_dtype = _storage_dtype(feature_cfg.get("storage_dtype", "float16"))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[radio_cfg["amp_dtype"]]
    buckets: dict[tuple[int, int], list[tuple[dict[str, Any], torch.Tensor, Any]]] = defaultdict(list)
    tensor_buffer: dict[str, torch.Tensor] = {}
    record_buffer: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = list(completed_rows)
    shard_index = max(shard_numbers) + 1 if shard_numbers else 0

    def flush_shard() -> None:
        nonlocal tensor_buffer, record_buffer, shard_index
        if not record_buffer:
            return
        path = feature_dir / f"part-{shard_index:05d}.safetensors"
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file({key: value.contiguous() for key, value in tensor_buffer.items()}, temporary)
        temporary.replace(path)
        for record in record_buffer:
            record["feature_shard"] = path.name
        write_records(manifest_dir / f"part-{shard_index:05d}.parquet", record_buffer)
        output_rows.extend(record_buffer)
        print(f"wrote tap feature shard {shard_index} ({len(record_buffer)} images)", flush=True)
        tensor_buffer = {}
        record_buffer = []
        shard_index += 1

    def run_batch(items: list[tuple[dict[str, Any], torch.Tensor, Any]]) -> None:
        images = torch.stack([item[1] for item in items]).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")
        ):
            final, intermediate = model.forward_intermediates(
                images,
                indices=layers,
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NLC",
                aggregation="sparse",
            )
        final_siglip, _ = _summary_features(final[adaptor_name])
        for item_index, (row, _, info) in enumerate(items):
            prefix = str(int(row["id"]))
            spatial_dim = None
            for layer, output in zip(layers, intermediate):
                spatial = output.features[item_index].detach().to("cpu", dtype=storage_dtype)
                summary = output.summary[item_index].detach().to("cpu", dtype=storage_dtype)
                spatial_dim = int(spatial.shape[-1])
                # C-RADIO summary concatenates teacher-specific CLS slots. The
                # first backbone-width slice is the SigLIP teacher slot.
                siglip_cls = summary.reshape(-1)[:spatial_dim]
                tensor_buffer[f"{prefix}.layer_{layer:02d}_spatial"] = spatial
                tensor_buffer[f"{prefix}.layer_{layer:02d}_siglip_cls"] = siglip_cls
            visual = F.normalize(final_siglip[item_index].float(), dim=-1)
            tensor_buffer[f"{prefix}.siglip_visual"] = visual.to("cpu", dtype=storage_dtype)
            record_buffer.append(
                {
                    **row,
                    "target_height": int(info.target_height),
                    "target_width": int(info.target_width),
                    "spatial_tokens": int((info.target_height // 16) * (info.target_width // 16)),
                    "spatial_dim": int(spatial_dim),
                    "siglip_visual_dim": int(visual.shape[-1]),
                    "storage_dtype": str(feature_cfg.get("storage_dtype", "float16")),
                }
            )
            if len(record_buffer) >= shard_rows:
                flush_shard()

    def flush_bucket(key: tuple[int, int]) -> None:
        items = buckets.pop(key)
        for offset in range(0, len(items), batch_size):
            run_batch(items[offset : offset + batch_size])

    for index, row in enumerate(rows):
        with Image.open(row["local_path"]) as image:
            array, info = preprocess_cradio_image(image, radio_cfg)
        key = (info.target_height, info.target_width)
        buckets[key].append((row, torch.from_numpy(array), info))
        if len(buckets[key]) >= batch_size:
            flush_bucket(key)
        elif len(buckets) > max_open_buckets:
            flush_bucket(max(buckets, key=lambda bucket: len(buckets[bucket])))
        if (index + 1) % 100 == 0:
            print(f"prepared tap images {index + 1}/{len(rows)}", flush=True)
    for key in list(buckets):
        flush_bucket(key)
    flush_shard()
    output_rows.sort(key=lambda row: int(row["experiment_artist_index"]) * 100 + int(row["experiment_sample_rank"]))
    write_records(manifest_path, output_rows)
    total_bytes = sum(path.stat().st_size for path in feature_dir.glob("*.safetensors"))
    summary = {
        "artists": int(cfg["artist_count"]),
        "images": len(output_rows),
        "layers": layers,
        "storage_bytes": total_bytes,
        "manifest": str(manifest_path.resolve()),
    }
    write_json(summary_path, summary)
    return summary


def _feature_storage_dir(
    destination: Path, root: Path, feature_cfg: dict[str, Any]
) -> Path:
    local_cache_directory = feature_cfg.get("local_cache_directory")
    if local_cache_directory:
        return Path(str(local_cache_directory))
    source_directory = feature_cfg.get("source_directory")
    return destination / str(source_directory) if source_directory else root / "features"


def _feature_manifest_dir(destination: Path, root: Path, feature_cfg: dict[str, Any]) -> Path:
    source_directory = feature_cfg.get("manifest_source_directory")
    return destination / str(source_directory) / "features" if source_directory else root / "features"


def _positional_grid(height: int, width: int, dim: int, device):
    import torch

    if dim % 4:
        raise ValueError("model_dim must be divisible by four")
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    frequencies = torch.exp(
        torch.linspace(0, math.log(1000.0), dim // 4, device=device)
    )
    return torch.cat(
        (
            torch.sin(x.reshape(-1, 1) * frequencies),
            torch.cos(x.reshape(-1, 1) * frequencies),
            torch.sin(y.reshape(-1, 1) * frequencies),
            torch.cos(y.reshape(-1, 1) * frequencies),
        ),
        dim=-1,
    )


def build_tap_resampler_model(
    *,
    taps: list[int],
    reconstruction_taps: list[int],
    spatial_dim: int,
    global_kind: str,
    global_dim: int,
    model_dim: int,
    latent_tokens: int,
    heads: int,
    resampler_layers: int,
    decoder_layers: int,
    style_dim: int,
    spatial_fusion: str = "weighted_sum",
    direct_style_tokens: bool = False,
    prototype_pool: str = "joint_flatten",
):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class CrossBlock(nn.Module):
        def __init__(self, *, self_attention: bool):
            super().__init__()
            self.self_attention = (
                nn.MultiheadAttention(model_dim, heads, batch_first=True)
                if self_attention
                else None
            )
            self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
            self.norm_self = nn.LayerNorm(model_dim)
            self.norm_cross_q = nn.LayerNorm(model_dim)
            self.norm_cross_kv = nn.LayerNorm(model_dim)
            self.norm_ff = nn.LayerNorm(model_dim)
            self.ff = nn.Sequential(
                nn.Linear(model_dim, model_dim * 4),
                nn.GELU(),
                nn.Linear(model_dim * 4, model_dim),
            )

        def forward(self, query, context, context_mask=None):
            if self.self_attention is not None:
                value = self.norm_self(query)
                query = query + self.self_attention(value, value, value, need_weights=False)[0]
            query = query + self.cross_attention(
                self.norm_cross_q(query),
                self.norm_cross_kv(context),
                self.norm_cross_kv(context),
                key_padding_mask=None if context_mask is None else ~context_mask,
                need_weights=False,
            )[0]
            return query + self.ff(self.norm_ff(query))

    class TapResampler(nn.Module):
        def __init__(self):
            super().__init__()
            self.taps = list(taps)
            self.reconstruction_taps = list(reconstruction_taps)
            self.global_kind = global_kind
            self.spatial_fusion = spatial_fusion
            self.direct_style_tokens = direct_style_tokens
            self.tap_norms = nn.ModuleDict(
                {str(layer): nn.LayerNorm(spatial_dim) for layer in taps}
            )
            if spatial_fusion == "concat_mlp":
                fusion_dim = spatial_dim * len(taps)
                self.spatial_projection = nn.Sequential(
                    nn.LayerNorm(fusion_dim),
                    nn.Linear(fusion_dim, model_dim * 2),
                    nn.GELU(),
                    nn.Linear(model_dim * 2, model_dim),
                )
                self.shared_spatial_projection = None
                self.tap_weights = None
                self.tap_projections = None
            elif spatial_fusion == "weighted_sum":
                self.spatial_projection = None
                self.shared_spatial_projection = nn.Linear(spatial_dim, model_dim)
                self.tap_weights = nn.Parameter(torch.zeros(len(taps)))
                self.tap_projections = None
            elif spatial_fusion == "projected_sum":
                self.spatial_projection = None
                self.shared_spatial_projection = None
                self.tap_projections = nn.ModuleDict(
                    {str(layer): nn.Linear(spatial_dim, model_dim) for layer in taps}
                )
                self.tap_weights = nn.Parameter(torch.zeros(len(taps)))
            else:
                raise ValueError(f"Unknown spatial fusion: {spatial_fusion}")
            self.global_projection = (
                nn.Sequential(nn.LayerNorm(global_dim), nn.Linear(global_dim, model_dim))
                if global_kind != "none"
                else None
            )
            self.queries = nn.Parameter(torch.empty(latent_tokens, model_dim))
            nn.init.normal_(self.queries, std=0.02)
            self.encoder = nn.ModuleList(
                [CrossBlock(self_attention=True) for _ in range(resampler_layers)]
            )
            self.decoder = nn.ModuleList(
                [CrossBlock(self_attention=True) for _ in range(decoder_layers)]
            )
            self.decoder_heads = nn.ModuleDict(
                {str(layer): nn.Linear(model_dim, spatial_dim) for layer in reconstruction_taps}
            )
            self.style_head = (
                None
                if direct_style_tokens
                else nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, model_dim),
                    nn.GELU(),
                    nn.Linear(model_dim, style_dim),
                )
            )
            self.style_projection = (
                nn.Identity()
                if direct_style_tokens and model_dim == style_dim
                else nn.Linear(model_dim, style_dim, bias=False)
                if direct_style_tokens
                else None
            )
            self.prototype_pool = prototype_pool
            if prototype_pool == "attention":
                self.prototype_query = nn.Parameter(torch.empty(style_dim))
                nn.init.normal_(self.prototype_query, std=style_dim**-0.5)
            elif prototype_pool == "joint_flatten":
                self.prototype_query = None
            else:
                raise ValueError(f"Unknown prototype pooling: {prototype_pool}")

        def encode(self, features, mask, global_feature=None):
            normalized = [self.tap_norms[str(layer)](features[layer]) for layer in self.taps]
            if self.spatial_fusion == "concat_mlp":
                context = self.spatial_projection(torch.cat(normalized, dim=-1))
            elif self.spatial_fusion == "projected_sum":
                weights = self.tap_weights.softmax(dim=0)
                context = sum(
                    weight * self.tap_projections[str(layer)](value)
                    for layer, weight, value in zip(self.taps, weights, normalized)
                )
            else:
                weights = self.tap_weights.softmax(dim=0)
                context = sum(
                    weight * self.shared_spatial_projection(value)
                    for weight, value in zip(weights, normalized)
                )
            context_mask = mask
            if self.global_projection is not None:
                projected = self.global_projection(global_feature).unsqueeze(1)
                context = torch.cat((projected, context), dim=1)
                context_mask = torch.cat(
                    (torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device), mask),
                    dim=1,
                )
            latent = self.queries.unsqueeze(0).expand(mask.shape[0], -1, -1)
            for block in self.encoder:
                latent = block(latent, context, context_mask)
            representation = (
                self.style_projection(latent)
                if self.direct_style_tokens
                else F.normalize(self.style_head(latent.mean(dim=1)), dim=-1)
            )
            # A fixed, non-affine LayerNorm prevents flow-only fine-tuning from
            # hiding unbounded token scale in learned normalization weights.
            return latent, F.layer_norm(representation, (style_dim,))

        def prototype_descriptor(self, representation):
            if representation.ndim != 3 or self.prototype_query is None:
                return _joint_token_descriptor(representation)
            normalized = F.layer_norm(
                representation.float(), (representation.shape[-1],)
            )
            scores = torch.einsum(
                "bnd,d->bn", normalized, self.prototype_query.float()
            ) / math.sqrt(normalized.shape[-1])
            pooled = torch.einsum("bn,bnd->bd", scores.softmax(dim=1), normalized)
            return F.normalize(pooled, dim=-1)

        def decode(self, latent, shapes, max_tokens):
            queries = latent.new_zeros((latent.shape[0], max_tokens, latent.shape[-1]))
            mask = torch.zeros((latent.shape[0], max_tokens), dtype=torch.bool, device=latent.device)
            for index, (height, width) in enumerate(shapes):
                grid = _positional_grid(height // 16, width // 16, model_dim, latent.device)
                queries[index, : len(grid)] = grid
                mask[index, : len(grid)] = True
            for block in self.decoder:
                queries = block(queries, latent)
            return {layer: self.decoder_heads[str(layer)](queries) for layer in self.reconstruction_taps}, mask

        def forward(self, features, mask, shapes, global_feature=None):
            latent, embedding = self.encode(features, mask, global_feature)
            decoded, decoded_mask = self.decode(latent, shapes, mask.shape[1])
            return decoded, decoded_mask, embedding

    return TapResampler()


def _load_feature_batch(
    rows: list[dict[str, Any]],
    feature_dir: Path,
    taps: list[int],
    reconstruction_taps: list[int],
    global_kind: str,
):
    import torch
    from safetensors import safe_open

    needed = sorted(set(taps) | set(reconstruction_taps))
    loaded: dict[int, dict[int, torch.Tensor]] = {}
    globals_: dict[int, torch.Tensor] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["feature_shard"])].append(row)
    for shard, shard_rows in grouped.items():
        with safe_open(feature_dir / shard, framework="pt", device="cpu") as handle:
            for row in shard_rows:
                image_id = int(row["id"])
                loaded[image_id] = {
                    layer: handle.get_tensor(f"{image_id}.layer_{layer:02d}_spatial")
                    for layer in needed
                }
                if global_kind.startswith("native_"):
                    layer = int(global_kind.rsplit("_", 1)[-1])
                    globals_[image_id] = handle.get_tensor(
                        f"{image_id}.layer_{layer:02d}_siglip_cls"
                    )
                elif global_kind == "siglip_visual":
                    globals_[image_id] = handle.get_tensor(f"{image_id}.siglip_visual")

    max_tokens = max(int(row["spatial_tokens"]) for row in rows)
    mask = torch.zeros((len(rows), max_tokens), dtype=torch.bool)
    feature_dtype = loaded[int(rows[0]["id"])][needed[0]].dtype
    features = {
        layer: torch.zeros(
            (len(rows), max_tokens, int(rows[0]["spatial_dim"])),
            dtype=feature_dtype,
        )
        for layer in needed
    }
    targets = {layer: features[layer] for layer in reconstruction_taps}
    global_batch = None
    if global_kind != "none":
        global_batch = torch.stack([globals_[int(row["id"])] for row in rows])
    for index, row in enumerate(rows):
        image_id = int(row["id"])
        tokens = int(row["spatial_tokens"])
        mask[index, :tokens] = True
        for layer in needed:
            features[layer][index, :tokens] = loaded[image_id][layer]
    return (
        {layer: features[layer] for layer in taps},
        targets,
        mask,
        [(int(row["target_height"]), int(row["target_width"])) for row in rows],
        global_batch,
    )


class _PinnedCudaBatchPipeline:
    """Two reusable pinned host slots with an independent CUDA transfer stream."""

    def __init__(self, device: str):
        import torch

        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots: list[dict[str, Any]] = [
            {"buffers": {}, "event": None},
            {"buffers": {}, "event": None},
        ]

    @staticmethod
    def _view(buffer, shape: tuple[int, ...]):
        return buffer[tuple(slice(0, size) for size in shape)]

    def _copy_to_slot(self, slot: dict[str, Any], name: str, value):
        import torch

        shape = tuple(value.shape)
        buffer = slot["buffers"].get(name)
        if (
            buffer is None
            or buffer.dtype != value.dtype
            or buffer.ndim != value.ndim
            or any(current < required for current, required in zip(buffer.shape, shape))
        ):
            buffer = torch.empty(shape, dtype=value.dtype, pin_memory=True)
            slot["buffers"][name] = buffer
        view = self._view(buffer, shape)
        view.copy_(value)
        return view

    def stage(self, loaded, slot_index: int):
        import torch

        features, targets, mask, shapes, global_feature = loaded
        slot = self.slots[slot_index]
        if slot["event"] is not None:
            # The slot is reused two iterations later. Ensure its previous H2D
            # copy no longer reads the pinned memory before overwriting it.
            slot["event"].synchronize()
        unique = {**targets, **features}
        host_features = {
            layer: self._copy_to_slot(slot, f"layer_{layer}", value)
            for layer, value in unique.items()
        }
        host_mask = self._copy_to_slot(slot, "mask", mask)
        host_global = (
            self._copy_to_slot(slot, "global", global_feature)
            if global_feature is not None
            else None
        )
        with torch.cuda.stream(self.stream):
            device_features = {
                layer: value.to(self.device, non_blocking=True)
                for layer, value in host_features.items()
            }
            device_mask = host_mask.to(self.device, non_blocking=True)
            device_global = (
                host_global.to(self.device, non_blocking=True)
                if host_global is not None
                else None
            )
            event = torch.cuda.Event()
            event.record(self.stream)
        slot["event"] = event
        return {
            "event": event,
            "features": {layer: device_features[layer] for layer in features},
            "targets": {layer: device_features[layer] for layer in targets},
            "mask": device_mask,
            "shapes": shapes,
            "global": device_global,
            "padding_efficiency": float(mask.sum().item() / mask.numel()),
        }

    @staticmethod
    def wait(batch) -> None:
        import torch

        torch.cuda.current_stream().wait_event(batch["event"])


def _prototype_loss(embeddings, artists_per_batch: int, images_per_artist: int, temperature: float):
    import torch
    import torch.nn.functional as F

    if embeddings.ndim == 3:
        values = embeddings.reshape(
            artists_per_batch, images_per_artist, embeddings.shape[1], embeddings.shape[2]
        )
        values = F.normalize(values, dim=-1)
        support_count = max(1, images_per_artist // 2)
        prototypes = F.normalize(values[:, :support_count].mean(dim=1), dim=-1)
        queries = values[:, support_count:].reshape(-1, values.shape[-2], values.shape[-1])
        labels = torch.arange(artists_per_batch, device=embeddings.device).repeat_interleave(
            images_per_artist - support_count
        )
        logits = torch.einsum("qnd,and->qa", queries, prototypes) / values.shape[-2]
        return F.cross_entropy(logits / temperature, labels)

    values = embeddings.reshape(artists_per_batch, images_per_artist, -1)
    support_count = max(1, images_per_artist // 2)
    prototypes = F.normalize(values[:, :support_count].mean(dim=1), dim=-1)
    queries = values[:, support_count:].reshape(-1, values.shape[-1])
    labels = torch.arange(artists_per_batch, device=embeddings.device).repeat_interleave(
        images_per_artist - support_count
    )
    return F.cross_entropy((queries @ prototypes.T) / temperature, labels)


def _pooled_token_prototype_loss(
    style_tokens, artists_per_batch: int, images_per_artist: int, temperature: float
):
    import torch.nn.functional as F

    pooled = F.normalize(F.normalize(style_tokens, dim=-1).mean(dim=1), dim=-1)
    return _prototype_loss(pooled, artists_per_batch, images_per_artist, temperature)


def _joint_token_descriptor(style_tokens):
    """Use every ordered style slot without a learned projection bottleneck."""
    import torch.nn.functional as F

    if style_tokens.ndim != 3:
        return F.normalize(style_tokens, dim=-1)
    flattened = style_tokens.float().flatten(1)
    return F.normalize(F.layer_norm(flattened, (flattened.shape[-1],)), dim=-1)


def _joint_token_prototype_loss(
    style_tokens, artists_per_batch: int, images_per_artist: int, temperature: float
):
    return _prototype_loss(
        _joint_token_descriptor(style_tokens),
        artists_per_batch,
        images_per_artist,
        temperature,
    )


def _slot_variation_diversity_loss(style_tokens, margin: float = 0.20):
    """Discourage slots from encoding the same image-dependent variation.

    Centering every slot over the batch removes fixed learned-query offsets, so
    the objective cannot be satisfied merely by assigning each slot a constant
    identity vector.
    """
    import torch
    import torch.nn.functional as F

    if style_tokens.ndim != 3 or style_tokens.shape[1] < 2:
        return style_tokens.new_zeros(())
    centered = style_tokens.float() - style_tokens.float().mean(dim=0, keepdim=True)
    variation = F.normalize(centered.transpose(0, 1).flatten(1), dim=-1)
    similarities = variation @ variation.T
    off_diagonal = ~torch.eye(
        similarities.shape[0], dtype=torch.bool, device=similarities.device
    )
    return F.relu(similarities[off_diagonal].abs() - margin).square().mean()


def _evaluation_descriptor(model, representation):
    return model.prototype_descriptor(representation)


def _warmup_cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _training_rows_for_step(
    *,
    step: int,
    seed: int,
    artists: list[str],
    train_by_style: dict[str, list[dict[str, Any]]],
    artists_per_batch: int,
    images_per_artist: int,
    token_bucket_centers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic episode without mutable sampler state.

    Step-addressable episodes let the CPU prefetch queue run ahead without
    changing checkpoint/resume behavior or the batches seen by other variants.
    """
    rng = random.Random((int(seed) << 32) ^ int(step))
    target_tokens = (
        rng.choice(token_bucket_centers) if token_bucket_centers else None
    )
    selected_artists = rng.sample(artists, artists_per_batch)
    rows = []
    for artist in selected_artists:
        candidates = train_by_style[artist]
        if target_tokens is not None:
            candidates = sorted(
                candidates,
                key=lambda row: (
                    abs(int(row["spatial_tokens"]) - target_tokens),
                    str(row["id"]),
                ),
            )[:images_per_artist]
        rows.extend(rng.sample(candidates, images_per_artist))
    return rows


def _token_bucket_centers(rows: list[dict[str, Any]], bucket_count: int) -> list[int]:
    values = sorted(int(row["spatial_tokens"]) for row in rows)
    if not values or bucket_count <= 0:
        return []
    if bucket_count == 1:
        return [values[len(values) // 2]]
    return sorted(
        {
            values[round(index * (len(values) - 1) / (bucket_count - 1))]
            for index in range(bucket_count)
        }
    )


def _reconstruction_loss(decoded, targets, mask, huber_weight: float):
    import torch
    import torch.nn.functional as F

    losses = []
    for layer, prediction in decoded.items():
        pred = prediction[mask]
        target = targets[layer][mask]
        cosine = (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()
        huber = F.smooth_l1_loss(pred.float(), target.float())
        losses.append(cosine + huber_weight * huber)
    return torch.stack(losses).mean()


def _evaluate_fixed_episodes(
    model,
    val_by_style,
    feature_dir,
    variant,
    cfg,
    training,
    device,
    *,
    episodes: int,
    seed: int,
):
    """Measure comparable validation losses on fixed unseen-artist episodes."""
    import torch

    artists = sorted(val_by_style)
    artists_per_batch = int(training["artists_per_batch"])
    images_per_artist = int(training["images_per_artist"])
    bucket_centers = _token_bucket_centers(
        [row for values in val_by_style.values() for row in values],
        int(training.get("token_buckets", 0)),
    )
    totals = defaultdict(float)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for episode in range(episodes):
            rows = _training_rows_for_step(
                step=episode,
                seed=seed,
                artists=artists,
                train_by_style=val_by_style,
                artists_per_batch=artists_per_batch,
                images_per_artist=images_per_artist,
                token_bucket_centers=bucket_centers,
            )
            features, targets, mask, shapes, global_feature = _load_feature_batch(
                rows,
                feature_dir,
                [int(value) for value in variant["taps"]],
                [int(value) for value in cfg["reconstruction_taps"]],
                str(variant.get("global", "none")),
            )
            unique = {**targets, **features}
            device_features = {layer: value.to(device) for layer, value in unique.items()}
            features = {layer: device_features[layer] for layer in features}
            targets = {layer: device_features[layer] for layer in targets}
            mask = mask.to(device)
            if global_feature is not None:
                global_feature = global_feature.to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                decoded, decoded_mask, representation = model(
                    features, mask, shapes, global_feature
                )
                rec = _reconstruction_loss(
                    decoded, targets, decoded_mask, float(training["huber_weight"])
                )
                slot = _prototype_loss(
                    representation,
                    artists_per_batch,
                    images_per_artist,
                    float(training["temperature"]),
                )
                joint = (
                    _prototype_loss(
                        model.prototype_descriptor(representation),
                        artists_per_batch,
                        images_per_artist,
                        float(training["temperature"]),
                    )
                    if representation.ndim == 3
                    else representation.new_zeros(())
                )
                slot_weight = float(
                    training.get("slot_prototype_weight", training.get("prototype_weight", 0.0))
                )
                joint_weight = float(training.get("joint_prototype_weight", 0.0))
                diversity = _slot_variation_diversity_loss(
                    representation, float(training.get("slot_diversity_margin", 0.20))
                )
                total = (
                    rec
                    + slot_weight * slot
                    + joint_weight * joint
                    + float(training.get("slot_diversity_weight", 0.0)) * diversity
                )
            totals["total"] += float(total)
            totals["reconstruction"] += float(rec)
            totals["slot_prototype"] += float(slot)
            totals["joint_prototype"] += float(joint)
            totals["slot_diversity"] += float(diversity)
    if was_training:
        model.train()
    return {key: value / episodes for key, value in totals.items()}


def _evaluate_model(
    model,
    rows,
    feature_dir,
    variant,
    cfg,
    device,
    *,
    batch_size: int,
    prefetch_workers: int,
    prefetch_batches: int,
):
    import torch
    import torch.nn.functional as F

    model.eval()
    embeddings = []
    rec_sum = defaultdict(float)
    rec_batches = 0
    offsets = list(range(0, len(rows), batch_size))

    def load_offset(offset: int):
        batch_rows = rows[offset : offset + batch_size]
        return batch_rows, _load_feature_batch(
            batch_rows,
            feature_dir,
            variant["taps"],
            cfg["reconstruction_taps"],
            variant["global"],
        )

    futures: dict[int, Future] = {}
    next_submit = 0
    started = time.perf_counter()
    with torch.inference_mode():
        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            for batch_index, _ in enumerate(offsets):
                while next_submit < len(offsets) and next_submit < batch_index + prefetch_batches:
                    futures[next_submit] = executor.submit(load_offset, offsets[next_submit])
                    next_submit += 1
                batch_rows, loaded = futures.pop(batch_index).result()
                features, targets, mask, shapes, global_feature = loaded
                features = {key: value.to(device) for key, value in features.items()}
                targets = {key: value.to(device) for key, value in targets.items()}
                mask = mask.to(device)
                if global_feature is not None:
                    global_feature = global_feature.to(device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.startswith("cuda"),
                ):
                    decoded, decoded_mask, representation = model(
                        features, mask, shapes, global_feature
                    )
                embeddings.append(_evaluation_descriptor(model, representation).cpu())
                for layer, prediction in decoded.items():
                    similarity = F.cosine_similarity(
                        prediction[decoded_mask].float(),
                        targets[layer][decoded_mask].float(),
                        dim=-1,
                    ).mean()
                    rec_sum[int(layer)] += float(similarity)
                rec_batches += 1
                if rec_batches % 25 == 0 or rec_batches == len(offsets):
                    elapsed = time.perf_counter() - started
                    print(
                        f"evaluating variant={variant['name']} "
                        f"batches={rec_batches}/{len(offsets)} "
                        f"batch_s={elapsed / rec_batches:.3f}",
                        flush=True,
                    )

    values = torch.cat(embeddings)
    style_ids = [str(row.get("style_id", row["artist"])) for row in rows]
    artists = sorted(set(style_ids))
    metrics = []
    for references in [int(value) for value in cfg["reference_counts"]]:
        prototypes = []
        queries = []
        labels = []
        for artist_index, artist in enumerate(artists):
            indices = [i for i, value in enumerate(style_ids) if value == artist]
            indices.sort(key=lambda i: int(rows[i]["experiment_sample_rank"]))
            prototypes.append(F.normalize(values[indices[:references]].mean(dim=0), dim=0))
            # Always use ranks 8 and 9 as queries so k-shot results are directly comparable.
            for index in indices[-2:]:
                queries.append(values[index])
                labels.append(artist_index)
        prototype_tensor = torch.stack(prototypes)
        query_tensor = torch.stack(queries)
        label_tensor = torch.tensor(labels)
        scores = query_tensor @ prototype_tensor.T
        metrics.append(
            {
                "references": references,
                "queries": len(labels),
                "top1": float((scores.argmax(dim=1) == label_tensor).float().mean()),
                "mrr": float(
                    (
                        1.0
                        / (
                            1
                            + (scores > scores.gather(1, label_tensor[:, None])).sum(dim=1).float()
                        )
                    ).mean()
                ),
            }
        )
    return {
        "prototype": metrics,
        "mean_top1": sum(row["top1"] for row in metrics) / len(metrics),
        "reconstruction_cosine": {
            str(layer): value / rec_batches for layer, value in sorted(rec_sum.items())
        },
    }


def train_tap_resampler_variants(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch

    cfg = config["tap_resampler_experiment"]
    training = cfg["training"]
    root = _experiment_dir(destination, cfg)
    manifest_dir = _feature_manifest_dir(destination, root, cfg["features"])
    feature_dir = _feature_storage_dir(destination, root, cfg["features"])
    rows = read_records(manifest_dir / "manifest.parquet")
    train_by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    val_rows = []
    for row in rows:
        if row["experiment_split"] == "meta_train":
            train_by_style[str(row.get("style_id", row["artist"]))].append(row)
        elif row["experiment_split"] == "meta_val":
            val_rows.append(row)
    for values in train_by_style.values():
        values.sort(key=lambda row: int(row["experiment_sample_rank"]))
    val_rows.sort(
        key=lambda row: (
            str(row.get("style_id", row["artist"])),
            int(row["experiment_sample_rank"]),
        )
    )
    val_by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in val_rows:
        val_by_style[str(row.get("style_id", row["artist"]))].append(row)

    device = str(training.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured tap-resampler experiment")
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    spatial_dim = int(rows[0]["spatial_dim"])
    variants = cfg["variants"]
    results = []
    experiment_root = root / "runs"
    experiment_root.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        name = str(variant["name"])
        run_dir = experiment_root / name
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not bool(training.get("force", False)):
            import json

            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            print(f"skipping completed variant {name}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        global_kind = str(variant.get("global", "none"))
        global_dim = (
            int(rows[0]["siglip_visual_dim"])
            if global_kind == "siglip_visual"
            else spatial_dim
        )
        torch.manual_seed(seed)
        model = build_tap_resampler_model(
            taps=[int(value) for value in variant["taps"]],
            reconstruction_taps=[int(value) for value in cfg["reconstruction_taps"]],
            spatial_dim=spatial_dim,
            global_kind=global_kind,
            global_dim=global_dim,
            model_dim=int(training["model_dim"]),
            latent_tokens=int(training["latent_tokens"]),
            heads=int(training["heads"]),
            resampler_layers=int(training["resampler_layers"]),
            decoder_layers=int(training["decoder_layers"]),
            style_dim=int(training["style_dim"]),
            spatial_fusion=str(training.get("spatial_fusion", "weighted_sum")),
            direct_style_tokens=bool(training.get("direct_style_tokens", False)),
            prototype_pool=str(training.get("prototype_pool", "joint_flatten")),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            fused=bool(training.get("fused_adamw", device.startswith("cuda"))),
        )
        amp_dtype = torch.bfloat16
        artists = sorted(train_by_style)
        steps = int(training["steps"])
        artists_per_batch = int(training["artists_per_batch"])
        images_per_artist = int(training["images_per_artist"])
        checkpoint_path = run_dir / "training_state.pt"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        validation_history_path = run_dir / "validation_history.json"
        validation_history = (
            json.loads(validation_history_path.read_text(encoding="utf-8"))
            if validation_history_path.exists()
            else []
        )
        best_validation_path = run_dir / "best_validation.json"
        best_validation = (
            json.loads(best_validation_path.read_text(encoding="utf-8"))
            if best_validation_path.exists()
            else None
        )
        start_step = 0
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state["step"])
            print(f"resuming variant {name} from step {start_step}", flush=True)
        model.train()
        running = defaultdict(float)
        input_taps = [int(value) for value in variant["taps"]]
        reconstruction_taps = [int(value) for value in cfg["reconstruction_taps"]]
        prefetch_workers = int(training.get("prefetch_workers", 4))
        prefetch_batches = max(prefetch_workers, int(training.get("prefetch_batches", 8)))
        bucket_centers = _token_bucket_centers(
            [row for values in train_by_style.values() for row in values],
            int(training.get("token_buckets", 0)),
        )
        wandb_run = None
        wandb_cfg = dict(training.get("wandb", {}))
        if bool(wandb_cfg.get("enabled", False)):
            import wandb

            wandb_run = wandb.init(
                project=str(wandb_cfg.get("project", "anima-style-adapter")),
                entity=wandb_cfg.get("entity"),
                name=str(wandb_cfg.get("name", name)),
                id=str(wandb_cfg.get("id", f"per-reference-{name}")),
                resume="allow",
                config={
                    "variant": dict(variant),
                    "training": dict(training),
                    "artist_count": len(train_by_style),
                    "token_bucket_centers": bucket_centers,
                },
            )

        def load_step(step: int):
            batch_rows = _training_rows_for_step(
                step=step,
                seed=seed,
                artists=artists,
                train_by_style=train_by_style,
                artists_per_batch=artists_per_batch,
                images_per_artist=images_per_artist,
                token_bucket_centers=bucket_centers,
            )
            return _load_feature_batch(
                batch_rows,
                feature_dir,
                input_taps,
                reconstruction_taps,
                global_kind,
            )

        futures: dict[int, Future] = {}
        next_submit = start_step
        log_started = time.perf_counter()
        transfer = _PinnedCudaBatchPipeline(device)

        def submit_window(executor, anchor: int) -> None:
            nonlocal next_submit
            while next_submit < steps and next_submit < anchor + prefetch_batches:
                futures[next_submit] = executor.submit(load_step, next_submit)
                next_submit += 1

        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            if start_step < steps:
                submit_window(executor, start_step)
                wait_started = time.perf_counter()
                current = transfer.stage(futures.pop(start_step).result(), 0)
                running["data_wait"] += time.perf_counter() - wait_started
            for step in range(start_step, steps):
                transfer.wait(current)
                features = current["features"]
                targets = current["targets"]
                mask = current["mask"]
                shapes = current["shapes"]
                global_feature = current["global"]
                current_lr = _warmup_cosine_learning_rate(
                    step,
                    total_steps=steps,
                    warmup_steps=int(training.get("warmup_steps", 0)),
                    peak_lr=float(training["learning_rate"]),
                    min_lr_ratio=float(training.get("min_lr_ratio", 1.0)),
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = current_lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")
                ):
                    decoded, decoded_mask, representation = model(
                        features, mask, shapes, global_feature
                    )
                    rec_loss = _reconstruction_loss(
                        decoded, targets, decoded_mask, float(training["huber_weight"])
                    )
                    proto_loss = _prototype_loss(
                        representation,
                        artists_per_batch,
                        images_per_artist,
                        float(training["temperature"]),
                    )
                    joint_proto_loss = (
                        _prototype_loss(
                            model.prototype_descriptor(representation),
                            artists_per_batch,
                            images_per_artist,
                            float(training["temperature"]),
                        )
                        if representation.ndim == 3
                        else representation.new_zeros(())
                    )
                    diversity_loss = _slot_variation_diversity_loss(
                        representation, float(training.get("slot_diversity_margin", 0.20))
                    )
                    ramp = min(
                        1.0,
                        (step + 1)
                        / max(1, int(steps * float(training["prototype_ramp"]))),
                    )
                    loss = rec_loss + ramp * (
                        float(
                            training.get(
                                "slot_prototype_weight", training.get("prototype_weight", 0.0)
                            )
                        )
                        * proto_loss
                        + float(training.get("joint_prototype_weight", 0.0))
                        * joint_proto_loss
                        + float(training.get("slot_diversity_weight", 0.0))
                        * diversity_loss
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["max_grad_norm"])
                )
                optimizer.step()
                # Keep scalar accumulators on GPU and synchronize only at the
                # logging interval, not once for every training step.
                running["loss"] += loss.detach()
                running["reconstruction"] += rec_loss.detach()
                running["prototype"] += proto_loss.detach()
                running["joint_prototype"] += joint_proto_loss.detach()
                running["slot_diversity"] += diversity_loss.detach()
                running["grad_norm"] += grad_norm.detach()
                running["padding_efficiency"] += float(current["padding_efficiency"])
                if (step + 1) % int(training["log_every"]) == 0:
                    interval = int(training["log_every"])
                    elapsed = time.perf_counter() - log_started
                    log_values = {
                        "train/loss": float(running["loss"] / interval),
                        "train/reconstruction": float(running["reconstruction"] / interval),
                        "train/slot_prototype": float(running["prototype"] / interval),
                        "train/joint_prototype": float(running["joint_prototype"] / interval),
                        "train/slot_diversity": float(running["slot_diversity"] / interval),
                        "train/grad_norm": float(running["grad_norm"] / interval),
                        "train/prototype_ramp": ramp,
                        "perf/step_s": elapsed / interval,
                        "perf/data_wait_s": running["data_wait"] / interval,
                        "perf/padding_efficiency": running["padding_efficiency"] / interval,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    print(
                        f"variant={name} step={step + 1}/{steps} "
                        f"loss={log_values['train/loss']:.4f} "
                        f"rec={log_values['train/reconstruction']:.4f} "
                        f"proto={log_values['train/slot_prototype']:.4f} "
                        f"joint_proto={log_values['train/joint_prototype']:.4f} "
                        f"diversity={log_values['train/slot_diversity']:.4f} "
                        f"padding={log_values['perf/padding_efficiency']:.3f} "
                        f"step_s={log_values['perf/step_s']:.3f} "
                        f"data_wait_s={log_values['perf/data_wait_s']:.3f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(log_values, step=step + 1)
                    running.clear()
                    log_started = time.perf_counter()
                if (step + 1) % int(training["checkpoint_every"]) == 0:
                    temporary = checkpoint_path.with_suffix(".tmp")
                    torch.save(
                        {
                            "step": step + 1,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        temporary,
                    )
                    temporary.replace(checkpoint_path)
                completed_step = step + 1
                validation_every = int(training.get("validation_every", 0))
                full_validation_every = int(training.get("full_validation_every", 0))
                if validation_every and completed_step % validation_every == 0:
                    fixed = _evaluate_fixed_episodes(
                        model,
                        val_by_style,
                        feature_dir,
                        variant,
                        cfg,
                        training,
                        device,
                        episodes=int(training.get("validation_episodes", 16)),
                        seed=seed ^ 0x5A17,
                    )
                    row = {"step": completed_step, "kind": "fixed_episodes", **fixed}
                    validation_history.append(row)
                    write_json(validation_history_path, validation_history)
                    print(
                        f"validation step={completed_step} "
                        f"loss={fixed['total']:.4f} rec={fixed['reconstruction']:.4f} "
                        f"proto={fixed['slot_prototype']:.4f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"validation_loss/{key}": value for key, value in fixed.items()},
                            step=completed_step,
                        )
                if full_validation_every and completed_step % full_validation_every == 0:
                    retrieval = _evaluate_model(
                        model,
                        val_rows,
                        feature_dir,
                        variant,
                        cfg,
                        device,
                        batch_size=int(training["evaluation_batch_size"]),
                        prefetch_workers=int(training.get("prefetch_workers", 4)),
                        prefetch_batches=int(training.get("prefetch_batches", 8)),
                    )
                    model.train()
                    row = {"step": completed_step, "kind": "full_retrieval", **retrieval}
                    validation_history.append(row)
                    write_json(validation_history_path, validation_history)
                    step_checkpoint = checkpoint_dir / f"step-{completed_step:05d}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "variant": variant,
                            "training": training,
                            "step": completed_step,
                            "validation": retrieval,
                        },
                        step_checkpoint,
                    )
                    if (
                        best_validation is None
                        or retrieval["mean_top1"] > best_validation["mean_top1"]
                    ):
                        best_validation = {
                            "step": completed_step,
                            "mean_top1": retrieval["mean_top1"],
                            "checkpoint": str(step_checkpoint),
                        }
                        write_json(best_validation_path, best_validation)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "validation_curve/mean_top1": retrieval["mean_top1"],
                                **{
                                    f"validation_curve/top1_{item['references']}ref": item["top1"]
                                    for item in retrieval["prototype"]
                                },
                                **{
                                    f"validation_curve/reconstruction_l{layer}": value
                                    for layer, value in retrieval["reconstruction_cosine"].items()
                                },
                            },
                            step=completed_step,
                        )
                next_batch = None
                if step + 1 < steps:
                    submit_window(executor, step + 1)
                    wait_started = time.perf_counter()
                    next_batch = transfer.stage(
                        futures.pop(step + 1).result(),
                        (step - start_step + 1) % 2,
                    )
                    running["data_wait"] += time.perf_counter() - wait_started
                current = next_batch

        selected_step = steps
        if best_validation is not None:
            best_checkpoint = torch.load(
                best_validation["checkpoint"], map_location="cpu", weights_only=False
            )
            model.load_state_dict(best_checkpoint["model"])
            selected_step = int(best_validation["step"])
            model.to(device)
        evaluation = _evaluate_model(
            model,
            val_rows,
            feature_dir,
            variant,
            cfg,
            device,
            batch_size=int(training["evaluation_batch_size"]),
            prefetch_workers=int(training.get("prefetch_workers", 4)),
            prefetch_batches=int(training.get("prefetch_batches", 8)),
        )
        result = {
            "name": name,
            "taps": variant["taps"],
            "global": global_kind,
            "selected_step": selected_step,
            **evaluation,
        }
        if wandb_run is not None:
            validation_log = {
                "validation/mean_top1": result["mean_top1"],
                **{
                    f"validation/top1_{row['references']}ref": row["top1"]
                    for row in result["prototype"]
                },
                **{
                    f"validation/mrr_{row['references']}ref": row["mrr"]
                    for row in result["prototype"]
                },
                **{
                    f"validation/reconstruction_l{layer}": value
                    for layer, value in result["reconstruction_cosine"].items()
                },
            }
            wandb_run.log(validation_log, step=steps)
        write_json(metrics_path, result)
        torch.save(
            {
                "model": model.state_dict(),
                "variant": variant,
                "training": training,
                "selected_step": selected_step,
            },
            run_dir / "checkpoint.pt",
        )
        checkpoint_path.unlink(missing_ok=True)
        results.append(result)
        print(f"completed variant {name}: mean_top1={result['mean_top1']:.4f}", flush=True)
        if wandb_run is not None:
            wandb_run.finish()

    results.sort(key=lambda row: row["mean_top1"], reverse=True)
    summary = {"variants": results}
    write_json(root / "evaluation.json", summary)
    return summary


def evaluate_selected_tap_variant(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Evaluate the validation-selected checkpoint once on held-out artists."""
    import torch

    cfg = config["tap_resampler_experiment"]
    training = cfg["training"]
    root = _experiment_dir(destination, cfg)
    validation_path = root / "evaluation.json"
    if not validation_path.exists():
        raise FileNotFoundError("Run tap-train before the final meta-test")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    selected_name = str(cfg.get("selected_variant", validation["variants"][0]["name"]))
    variants = {str(item["name"]): item for item in cfg["variants"]}
    if selected_name not in variants:
        raise ValueError(f"Unknown selected_variant: {selected_name}")
    variant = variants[selected_name]
    validation_row = next(
        (item for item in validation["variants"] if item["name"] == selected_name), None
    )
    if validation_row is None:
        raise RuntimeError(f"Selected variant has no validation result: {selected_name}")

    manifest_dir = _feature_manifest_dir(destination, root, cfg["features"])
    feature_dir = _feature_storage_dir(destination, root, cfg["features"])
    rows = read_records(manifest_dir / "manifest.parquet")
    test_rows = [row for row in rows if row["experiment_split"] == "meta_test"]
    test_rows.sort(
        key=lambda row: (
            str(row.get("style_id", row["artist"])),
            int(row["experiment_sample_rank"]),
        )
    )
    if not test_rows:
        raise RuntimeError("No meta_test rows found")

    spatial_dim = int(rows[0]["spatial_dim"])
    global_kind = str(variant.get("global", "none"))
    global_dim = (
        int(rows[0]["siglip_visual_dim"])
        if global_kind == "siglip_visual"
        else spatial_dim
    )
    device = str(training.get("device", "cuda"))
    model = build_tap_resampler_model(
        taps=[int(value) for value in variant["taps"]],
        reconstruction_taps=[int(value) for value in cfg["reconstruction_taps"]],
        spatial_dim=spatial_dim,
        global_kind=global_kind,
        global_dim=global_dim,
        model_dim=int(training["model_dim"]),
        latent_tokens=int(training["latent_tokens"]),
        heads=int(training["heads"]),
        resampler_layers=int(training["resampler_layers"]),
        decoder_layers=int(training["decoder_layers"]),
        style_dim=int(training["style_dim"]),
        spatial_fusion=str(training.get("spatial_fusion", "weighted_sum")),
        direct_style_tokens=bool(training.get("direct_style_tokens", False)),
        prototype_pool=str(training.get("prototype_pool", "joint_flatten")),
    ).to(device)
    checkpoint_path = root / "runs" / selected_name / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = _evaluate_model(
        model,
        test_rows,
        feature_dir,
        variant,
        cfg,
        device,
        batch_size=int(training["evaluation_batch_size"]),
        prefetch_workers=int(training.get("prefetch_workers", 4)),
        prefetch_batches=int(training.get("prefetch_batches", 8)),
    )
    result = {
        "selection_rule": "highest validation mean_top1, checked for reconstruction/cost Pareto",
        "selected_variant": selected_name,
        "validation": validation_row,
        "meta_test": test_metrics,
        "meta_test_artists": len(
            {str(row.get("style_id", row["artist"])) for row in test_rows}
        ),
        "meta_test_images": len(test_rows),
    }
    write_json(root / "final_test.json", result)
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", selected_name)),
            id=str(wandb_cfg.get("id", f"per-reference-{selected_name}")),
            resume="allow",
        )
        run.log(
            {
                "test/mean_top1": test_metrics["mean_top1"],
                **{
                    f"test/top1_{row['references']}ref": row["top1"]
                    for row in test_metrics["prototype"]
                },
                **{
                    f"test/mrr_{row['references']}ref": row["mrr"]
                    for row in test_metrics["prototype"]
                },
                **{
                    f"test/reconstruction_l{layer}": value
                    for layer, value in test_metrics["reconstruction_cosine"].items()
                },
            },
            step=int(training["steps"]) + 1,
        )
        run.finish()
    return result


def run_tap_resampler_experiment(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    extract_tap_features(config, destination)
    return train_tap_resampler_variants(config, destination)
