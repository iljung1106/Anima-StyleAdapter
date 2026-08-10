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
        manifest_path = feature_dir / "manifest.parquet"
        write_records(manifest_path, output_rows)
        summary = {
            "artists": int(cfg["artist_count"]),
            "images": len(output_rows),
            "layers": sorted(int(layer) for layer in feature_cfg["layers"]),
            "source_directory": str(source_directory),
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
    source_directory = feature_cfg.get("source_directory")
    return destination / str(source_directory) if source_directory else root / "features"


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
            elif spatial_fusion == "weighted_sum":
                self.spatial_projection = None
                self.shared_spatial_projection = nn.Linear(spatial_dim, model_dim)
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

        def encode(self, features, mask, global_feature=None):
            normalized = [self.tap_norms[str(layer)](features[layer]) for layer in self.taps]
            if self.spatial_fusion == "concat_mlp":
                context = self.spatial_projection(torch.cat(normalized, dim=-1))
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
                latent
                if self.direct_style_tokens
                else F.normalize(self.style_head(latent.mean(dim=1)), dim=-1)
            )
            return latent, representation

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
                    layer: handle.get_tensor(f"{image_id}.layer_{layer:02d}_spatial").float()
                    for layer in needed
                }
                if global_kind.startswith("native_"):
                    layer = int(global_kind.rsplit("_", 1)[-1])
                    globals_[image_id] = handle.get_tensor(
                        f"{image_id}.layer_{layer:02d}_siglip_cls"
                    ).float()
                elif global_kind == "siglip_visual":
                    globals_[image_id] = handle.get_tensor(f"{image_id}.siglip_visual").float()

    max_tokens = max(int(row["spatial_tokens"]) for row in rows)
    mask = torch.zeros((len(rows), max_tokens), dtype=torch.bool)
    features = {
        layer: torch.zeros((len(rows), max_tokens, int(rows[0]["spatial_dim"])))
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


def _evaluation_descriptor(representation):
    import torch.nn.functional as F

    if representation.ndim == 2:
        return representation
    normalized_slots = F.normalize(representation, dim=-1)
    return F.normalize(normalized_slots.flatten(1), dim=-1)


def _training_rows_for_step(
    *,
    step: int,
    seed: int,
    artists: list[str],
    train_by_style: dict[str, list[dict[str, Any]]],
    artists_per_batch: int,
    images_per_artist: int,
) -> list[dict[str, Any]]:
    """Build a deterministic episode without mutable sampler state.

    Step-addressable episodes let the CPU prefetch queue run ahead without
    changing checkpoint/resume behavior or the batches seen by other variants.
    """
    rng = random.Random((int(seed) << 32) ^ int(step))
    selected_artists = rng.sample(artists, artists_per_batch)
    rows = []
    for artist in selected_artists:
        rows.extend(rng.sample(train_by_style[artist], images_per_artist))
    return rows


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
                decoded, decoded_mask, representation = model(
                    features, mask, shapes, global_feature
                )
                embeddings.append(_evaluation_descriptor(representation).cpu())
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
    manifest_dir = root / "features"
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
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        amp_dtype = torch.bfloat16
        artists = sorted(train_by_style)
        steps = int(training["steps"])
        artists_per_batch = int(training["artists_per_batch"])
        images_per_artist = int(training["images_per_artist"])
        checkpoint_path = run_dir / "training_state.pt"
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

        def load_step(step: int):
            batch_rows = _training_rows_for_step(
                step=step,
                seed=seed,
                artists=artists,
                train_by_style=train_by_style,
                artists_per_batch=artists_per_batch,
                images_per_artist=images_per_artist,
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
        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            for step in range(start_step, steps):
                while next_submit < steps and next_submit < step + prefetch_batches:
                    futures[next_submit] = executor.submit(load_step, next_submit)
                    next_submit += 1
                wait_started = time.perf_counter()
                features, targets, mask, shapes, global_feature = futures.pop(step).result()
                running["data_wait"] += time.perf_counter() - wait_started
                features = {
                    key: value.to(device, non_blocking=True) for key, value in features.items()
                }
                targets = {
                    key: value.to(device, non_blocking=True) for key, value in targets.items()
                }
                mask = mask.to(device, non_blocking=True)
                if global_feature is not None:
                    global_feature = global_feature.to(device, non_blocking=True)
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
                    pooled_proto_loss = (
                        _pooled_token_prototype_loss(
                            representation,
                            artists_per_batch,
                            images_per_artist,
                            float(training["temperature"]),
                        )
                        if representation.ndim == 3
                        else representation.new_zeros(())
                    )
                    ramp = min(
                        1.0,
                        (step + 1)
                        / max(1, int(steps * float(training["prototype_ramp"]))),
                    )
                    loss = rec_loss + ramp * (
                        float(training["prototype_weight"]) * proto_loss
                        + float(training.get("pooled_prototype_weight", 0.0))
                        * pooled_proto_loss
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["max_grad_norm"])
                )
                optimizer.step()
                running["loss"] += float(loss.detach())
                running["reconstruction"] += float(rec_loss.detach())
                running["prototype"] += float(proto_loss.detach())
                running["pooled_prototype"] += float(pooled_proto_loss.detach())
                if (step + 1) % int(training["log_every"]) == 0:
                    interval = int(training["log_every"])
                    elapsed = time.perf_counter() - log_started
                    print(
                        f"variant={name} step={step + 1}/{steps} "
                        f"loss={running['loss'] / interval:.4f} "
                        f"rec={running['reconstruction'] / interval:.4f} "
                        f"proto={running['prototype'] / interval:.4f} "
                        f"pooled_proto={running['pooled_prototype'] / interval:.4f} "
                        f"step_s={elapsed / interval:.3f} "
                        f"data_wait_s={running['data_wait'] / interval:.3f}",
                        flush=True,
                    )
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
        result = {"name": name, "taps": variant["taps"], "global": global_kind, **evaluation}
        write_json(metrics_path, result)
        torch.save(
            {"model": model.state_dict(), "variant": variant, "training": training},
            run_dir / "checkpoint.pt",
        )
        checkpoint_path.unlink(missing_ok=True)
        results.append(result)
        print(f"completed variant {name}: mean_top1={result['mean_top1']:.4f}", flush=True)

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

    manifest_dir = root / "features"
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
    return result


def run_tap_resampler_experiment(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    extract_tap_features(config, destination)
    return train_tap_resampler_variants(config, destination)
