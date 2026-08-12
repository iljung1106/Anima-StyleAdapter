from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from safetensors import safe_open

from .io import read_records, write_json, write_records


class ArtistCenteredEffectHead(nn.Module):
    """Small query-conditioned residual head for reference-specific effects.

    The large connector remains a frozen estimator of the population/common
    artist effect.  This head sees an explicit mean/std descriptor of all
    Resampler slots and is trained only on the remaining native-teacher error,
    preventing reconstruction gradients from erasing reference identity.
    """

    def __init__(
        self, style_dim: int = 1024, hidden_dim: int = 2048,
        latent_dim: int = 512, blocks: int = 28,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.descriptor = nn.Sequential(
            nn.Linear(style_dim * 2, latent_dim, bias=False),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )
        self.query = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.modulation = nn.Linear(latent_dim, latent_dim * 2, bias=False)
        self.block_embedding = nn.Parameter(torch.zeros(blocks, latent_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.output[-1].weight)

    def artist_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],)).to(tokens.dtype)
        descriptor = torch.cat(
            (normalized.mean(1), normalized.std(1, correction=0)), dim=-1
        )
        return self.descriptor(descriptor)

    def forward(
        self, tokens: torch.Tensor, queries: torch.Tensor, block: int
    ) -> torch.Tensor:
        # queries: [B,H,Q,D] -> one query-conditioned residual per Q position.
        batch, _, query_count, _ = queries.shape
        query = queries.transpose(1, 2).reshape(batch, query_count, self.hidden_dim)
        latent = self.artist_latent(tokens)
        scale, shift = self.modulation(latent).chunk(2, dim=-1)
        values = self.query(F.layer_norm(query.float(), (self.hidden_dim,)).to(query.dtype))
        values = values * (1 + 0.1 * torch.tanh(scale[:, None]))
        values = values + shift[:, None] + self.block_embedding[block][None, None]
        return self.output(values)


def _root(config: dict[str, Any], destination: Path) -> Path:
    name = config["synthetic_teacher"].get("output_directory", "synthetic_teacher")
    return destination / str(name)


def _feature_descriptors(handle: Any, image_ids: list[int], device: str) -> torch.Tensor:
    """Reduce one same-shape feature shard as GPU batches, returning compact CPU rows."""
    parts = []
    for layer in (18, 24):
        value = torch.stack([
            handle.get_tensor(f"{image_id}.layer_{layer:02d}_spatial") for image_id in image_ids
        ]).to(device=device, dtype=torch.float32)
        parts.extend((value.mean(1), value.std(1, correction=0)))
    cls = torch.stack([
        handle.get_tensor(f"{image_id}.layer_24_siglip_cls").flatten() for image_id in image_ids
    ]).to(device=device, dtype=torch.float32)
    parts.append(cls)
    normalized = [F.layer_norm(value, value.shape[1:]) for value in parts]
    return torch.cat(normalized, dim=-1).cpu()


def classify_artist_effects(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Exclude only distributional extremes with weak/inconsistent effects."""
    finite = [row for row in rows if all(math.isfinite(float(row[key])) for key in (
        "effect_rms", "direction_consistency", "seed_consistency", "content_consistency"
    ))]
    if len(finite) != len(rows):
        invalid = {str(row["artist"]) for row in rows if row not in finite}
    else:
        invalid = set()
    if not finite:
        return {str(row["artist"]): "excluded_nonfinite" for row in rows}
    effects = torch.tensor([float(row["effect_rms"]) for row in finite])
    consistency = torch.tensor([float(row["direction_consistency"]) for row in finite])
    effect_q01, effect_q99 = torch.quantile(effects, torch.tensor([0.01, 0.99])).tolist()
    consistency_q01, consistency_q10, consistency_q25 = torch.quantile(
        consistency, torch.tensor([0.01, 0.10, 0.25])
    ).tolist()
    labels: dict[str, str] = {}
    for row in rows:
        artist = str(row["artist"])
        if artist in invalid:
            labels[artist] = "excluded_nonfinite"
        elif float(row["direction_consistency"]) < consistency_q01:
            labels[artist] = "excluded_unstable"
        elif (
            float(row["effect_rms"]) < effect_q01
            and float(row["direction_consistency"]) < consistency_q10
        ):
            labels[artist] = "excluded_weak"
        elif (
            float(row["effect_rms"]) > effect_q99
            and float(row["direction_consistency"]) < consistency_q25
        ):
            labels[artist] = "excluded_overchanging"
        elif (
            float(row["effect_rms"]) >= float(effects.median())
            and float(row["direction_consistency"]) >= float(consistency.median())
        ):
            labels[artist] = "strong"
        else:
            labels[artist] = "moderate"
    return labels


def assign_bootstrap_splits(
    artists: list[str], *, seed: int, validation_artists: int = 25, meta_test_artists: int = 25
) -> dict[str, str]:
    if len(artists) <= validation_artists + meta_test_artists:
        raise ValueError("Not enough retained artists for validation and meta-test splits")
    ordered = sorted(artists)
    random.Random(seed).shuffle(ordered)
    meta = set(ordered[:meta_test_artists])
    validation = set(ordered[meta_test_artists : meta_test_artists + validation_artists])
    return {
        artist: "meta_test" if artist in meta else "validation" if artist in validation else "train"
        for artist in ordered
    }


def _bootstrap_eligible(row: dict[str, Any]) -> bool:
    return row.get("kind") == "artist" and row.get("artist_split") != "excluded"


def validate_synthetic_teacher_corpus(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["synthetic_teacher"]
    bootstrap = cfg.get("bootstrap", {})
    # Large RunPod hosts may expose more than 100 CPU threads.  The artist
    # effect pass consists of many small reductions where that many OpenMP
    # workers are dramatically slower than a compact pool.
    validation_threads = int(bootstrap.get("cpu_threads", 8))
    torch.set_num_threads(max(1, validation_threads))
    torch.set_num_interop_threads(max(1, min(4, validation_threads)))
    root = _root(config, destination)
    manifest = read_records(root / "manifest.parquet")
    feature_root = root / "style_features"
    feature_rows = read_records(feature_root / "manifest.parquet")
    kv_rows = read_records(root / "anima_kv_teacher" / "manifest.parquet")
    text_rows = read_records(root / "text" / "manifest.parquet")
    plan = read_records(root / "plan.parquet")

    expected_artist = int(cfg.get("artist_count", 500)) * int(cfg.get("contents_per_artist", 8)) * int(cfg.get("seeds_per_content", 2))
    expected_controls = int(cfg.get("contents_per_artist", 8)) * int(cfg.get("seeds_per_content", 2))
    errors: list[str] = []
    ids = [int(row["id"]) for row in manifest]
    if len(ids) != len(set(ids)):
        errors.append("duplicate image IDs")
    if len(manifest) != expected_artist + expected_controls:
        errors.append(f"expected {expected_artist + expected_controls} manifest rows, found {len(manifest)}")
    if len(feature_rows) != len(manifest):
        errors.append(f"feature rows {len(feature_rows)} != image rows {len(manifest)}")
    if {int(row["id"]) for row in feature_rows} != set(ids):
        errors.append("feature/image ID sets differ")
    if len(plan) != len(manifest):
        errors.append(f"plan rows {len(plan)} != manifest rows {len(manifest)}")
    if len(kv_rows) != int(cfg.get("artist_count", 500)) * int(cfg.get("contents_per_artist", 8)):
        errors.append("K/V teacher condition count mismatch")
    if len(text_rows) != int(cfg.get("contents_per_artist", 8)) * (int(cfg.get("artist_count", 500)) + 1):
        errors.append("post-LLM condition count mismatch")
    artists = [row for row in manifest if row["kind"] == "artist"]
    controls = [row for row in manifest if row["kind"] == "content_control"]
    counts = Counter(str(row["artist"]) for row in artists)
    if any(value != expected_controls for value in counts.values()) or len(counts) != int(cfg.get("artist_count", 500)):
        errors.append("artist cross-product is incomplete")
    female_contents = {
        int(row["content_index"]) for row in plan
        if "1girl" in {part.strip() for part in str(row["content_prompt"]).split(",")}
    }
    if len(female_contents) != int(cfg.get("female_contents", 7)):
        errors.append(f"expected {cfg.get('female_contents', 7)} 1girl contents, found {len(female_contents)}")

    # C-RADIO extraction already decoded every WebP successfully and the exact
    # ID-set check above proves full coverage. Reopening 8k NFS files would be
    # a duplicate 10-20 minute pass, so stat every path and fully decode a
    # deterministic sample here.
    decode_stride = max(1, len(manifest) // int(bootstrap.get("image_decode_samples", 128)))
    def inspect_image(item: tuple[int, dict[str, Any]]) -> str | None:
        index, row = item
        try:
            if index % decode_stride == 0:
                path = Path(row["local_path"])
                if not path.is_file() or path.stat().st_size == 0:
                    return f"missing or empty image: {row['id']}"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (int(row["width"]), int(row["height"])):
                        return f"image size mismatch: {row['id']}"
        except Exception as error:
            return f"corrupt image {row['id']}: {error}"
        return None

    image_workers = int(bootstrap.get("image_validation_workers", 32))
    with ThreadPoolExecutor(max_workers=image_workers) as executor:
        for error in executor.map(inspect_image, enumerate(manifest), chunksize=32):
            if error:
                errors.append(error)
    print(
        f"C-RADIO manifest proves {len(manifest)} full decodes; additionally validated "
        f"{math.ceil(len(manifest) / decode_stride)} image samples with {image_workers} workers",
        flush=True,
    )

    if errors:
        summary = {"valid": False, "errors": errors, "images": len(manifest)}
        write_json(root / "integrity_summary.json", summary)
        raise RuntimeError("Synthetic corpus integrity failed: " + "; ".join(errors[:8]))

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_shard[str(row["feature_shard"])].append(row)
    descriptors: dict[int, torch.Tensor] = {}
    reduction_device = str(bootstrap.get("feature_reduction_device", "cuda"))
    reduction_batch = int(bootstrap.get("feature_reduction_batch_size", 64))
    for shard_index, (shard, shard_rows) in enumerate(sorted(by_shard.items())):
        with safe_open(feature_root / shard, framework="pt", device="cpu") as handle:
            for offset in range(0, len(shard_rows), reduction_batch):
                part = shard_rows[offset : offset + reduction_batch]
                image_ids = [int(row["id"]) for row in part]
                values = _feature_descriptors(handle, image_ids, reduction_device)
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite C-RADIO descriptor in {shard}")
                descriptors.update(zip(image_ids, values, strict=True))
        print(
            f"validated feature shard {shard_index + 1}/{len(by_shard)} "
            f"({len(descriptors)}/{len(feature_rows)} images)", flush=True,
        )

    control_descriptor = {int(row["id"]): descriptors[int(row["id"])] for row in controls}
    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artists:
        by_artist[str(row["artist"])].append(row)
    effect_rows = []
    for artist, rows in sorted(by_artist.items()):
        rows.sort(key=lambda row: (int(row["content_index"]), int(row["seed_index"])))
        deltas = torch.stack([
            descriptors[int(row["id"])] - control_descriptor[int(row["control_id"])] for row in rows
        ])
        mean = deltas.mean(0)
        normalized = F.normalize(deltas, dim=-1)
        direction_consistency = float((normalized @ F.normalize(mean, dim=0)).mean())
        seed_cosines = []
        content_means = []
        for content in range(int(cfg.get("contents_per_artist", 8))):
            pair = deltas[[int(row["content_index"]) == content for row in rows]]
            seed_cosines.append(float(F.cosine_similarity(pair[0], pair[1], dim=0)))
            content_means.append(pair.mean(0))
        content_means = F.normalize(torch.stack(content_means), dim=-1)
        content_consistency = float((content_means @ F.normalize(mean, dim=0)).mean())
        effect_rows.append({
            "artist": artist,
            "effect_rms": float(deltas.square().mean(dim=-1).sqrt().median()),
            "direction_consistency": direction_consistency,
            "seed_consistency": sum(seed_cosines) / len(seed_cosines),
            "content_consistency": content_consistency,
        })
    labels = classify_artist_effects(effect_rows)
    split_seed = int(bootstrap.get("split_seed", int(cfg.get("seed", 20260812)) ^ 0xB007))
    retained = [row["artist"] for row in effect_rows if not labels[row["artist"]].startswith("excluded")]
    splits = assign_bootstrap_splits(
        retained, seed=split_seed,
        validation_artists=int(bootstrap.get("validation_artists", 25)),
        meta_test_artists=int(bootstrap.get("meta_test_artists", 25)),
    )
    heldout = [int(value) for value in bootstrap.get("heldout_contents", [6, 7])]
    if len(set(heldout)) != 2 or any(value < 0 or value >= int(cfg.get("contents_per_artist", 8)) for value in heldout):
        raise ValueError("Exactly two distinct content indices must be held out")
    content_split = {index: "train" for index in range(int(cfg.get("contents_per_artist", 8)))}
    content_split[heldout[0]] = "validation"
    content_split[heldout[1]] = "test"
    score_by_artist = {row["artist"]: row for row in effect_rows}
    validated = []
    for row in manifest:
        if row["kind"] == "content_control":
            validated.append({**row, "artist_quality": "control", "artist_split": "control", "content_split": content_split[int(row["content_index"])]})
            continue
        artist = str(row["artist"])
        validated.append({
            **row, **{key: value for key, value in score_by_artist[artist].items() if key != "artist"},
            "artist_quality": labels[artist],
            "artist_split": splits.get(artist, "excluded"),
            "content_split": content_split[int(row["content_index"])],
            "bootstrap_eligible": artist in splits,
        })
    write_records(root / "artist_effects.parquet", [
        {**row, "quality": labels[row["artist"]], "split": splits.get(row["artist"], "excluded")}
        for row in effect_rows
    ])
    write_records(root / "validated_manifest.parquet", validated)
    quality_counts = Counter(labels.values())
    split_counts = Counter(splits.values())
    summary = {
        "valid": True, "images": len(manifest), "artist_images": len(artists), "controls": len(controls),
        "artists": len(counts), "retained_artists": len(retained), "excluded_artists": len(counts) - len(retained),
        "quality_counts": dict(quality_counts), "artist_split_counts": dict(split_counts),
        "content_splits": content_split, "heldout_contents": heldout,
    }
    write_json(root / "integrity_summary.json", summary)
    return summary


def build_anima_query_probe_bank(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Capture normalized native cross-attention queries on content-only trajectories."""
    from safetensors.torch import load_file, save_file

    from .style_transfer import _resolve_anima_model

    cfg = config["synthetic_teacher"]
    probe_cfg = cfg.get("query_probes", {})
    root = _root(config, destination)
    output = root / "query_probe_bank"
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "queries.safetensors"
    manifest_path = output / "manifest.parquet"
    if tensor_path.exists() and manifest_path.exists():
        rows = read_records(manifest_path)
        return {"probes": len(rows), "reused": len(rows), "storage_bytes": tensor_path.stat().st_size}
    device = str(cfg.get("device", "cuda"))
    rows = [row for row in read_records(root / "manifest.parquet") if row["kind"] == "content_control"]
    controls: dict[int, dict[str, Any]] = {}
    for row in rows:
        controls.setdefault(int(row["content_index"]), row)
    contents = int(cfg.get("contents_per_artist", 8))
    if len(controls) != contents:
        raise RuntimeError(f"Expected {contents} content controls, found {len(controls)}")
    text_rows = read_records(root / "text" / "manifest.parquet")
    text_by_id = {int(row["condition_id"]): row for row in text_rows}
    text_cache: dict[str, torch.Tensor] = {}

    def condition(content: int) -> torch.Tensor:
        row = text_by_id[content]
        name = str(row["cache_shard"])
        if name not in text_cache:
            text_cache[name] = load_file(root / "text" / name, device="cpu")["conditioning"]
        return text_cache[name][int(row["row_index"])]

    latent_cache: dict[str, torch.Tensor] = {}

    def latent(row: dict[str, Any]) -> torch.Tensor:
        name = str(row["latent_shard"])
        if name not in latent_cache:
            latent_cache[name] = load_file(root / "latents" / name, device="cpu")["latents"]
        return latent_cache[name][int(row["latent_row"])]

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for block_index, block in enumerate(anima.blocks):
        def hook(_module, _inputs, result, *, index=block_index):
            captured[index] = result.detach()
        handles.append(block.cross_attn.q_norm.register_forward_hook(hook))
    timesteps = [float(value) for value in probe_cfg.get("timesteps", [0.15, 0.35, 0.65, 0.85])]
    trajectories = int(probe_cfg.get("trajectories_per_content", 8))
    queries_per = int(probe_cfg.get("queries_per_block", 32))
    seed = int(probe_cfg.get("seed", int(cfg.get("seed", 20260812)) ^ 0x0BEE))
    tensors: dict[str, torch.Tensor] = {}
    records = []
    for content_index in range(contents):
        base = latent(controls[content_index]).to(device=device, dtype=torch.bfloat16)
        target = base.unsqueeze(0).expand(trajectories, -1, -1, -1)
        generator = torch.Generator(device=device).manual_seed(seed + content_index)
        noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
        context = condition(content_index).unsqueeze(0).expand(trajectories, -1, -1).to(device=device, dtype=torch.bfloat16)
        padding = torch.zeros(trajectories, 1, target.shape[-2], target.shape[-1], device=device, dtype=target.dtype)
        for timestep_index, timestep_value in enumerate(timesteps):
            sigma = torch.tensor(timestep_value, device=device, dtype=target.dtype)
            noisy = (1 - sigma) * target + sigma * noise
            captured.clear()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                anima(
                    noisy.unsqueeze(2),
                    torch.full((trajectories,), timestep_value, device=device, dtype=target.dtype),
                    context=context, padding_mask=padding, target_input_ids=None,
                )
            if len(captured) != 28:
                raise RuntimeError(f"Captured {len(captured)} of 28 Anima query tensors")
            for block_index in range(28):
                values = captured[block_index].reshape(-1, *captured[block_index].shape[-2:])
                sample_generator = torch.Generator(device="cpu").manual_seed(
                    seed + content_index * 10_000 + timestep_index * 100 + block_index
                )
                indices = torch.randperm(values.shape[0], generator=sample_generator)[:queries_per].to(device)
                selected = values.index_select(0, indices).to(device="cpu", dtype=torch.float16).contiguous()
                key = f"c{content_index:02d}.t{timestep_index:02d}.b{block_index:02d}"
                tensors[key] = selected
                records.append({
                    "key": key, "content_index": content_index, "timestep_index": timestep_index,
                    "timestep": timestep_value, "block": block_index, "queries": selected.shape[0],
                    "heads": selected.shape[1], "head_dim": selected.shape[2],
                    "rms": float(selected.float().square().mean().sqrt()),
                })
        print(f"query probes captured content {content_index + 1}/{contents}", flush=True)
    for handle in handles:
        handle.remove()
    save_file(tensors, tensor_path)
    write_records(manifest_path, records)
    summary = {
        "records": len(records), "contents": contents, "timesteps": timesteps,
        "blocks": 28, "queries_per_block": queries_per,
        "trajectories_per_content": trajectories, "storage_bytes": tensor_path.stat().st_size,
    }
    write_json(output / "summary.json", summary)
    return summary


def cache_synthetic_resampler_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Materialize the frozen 128x1024 Resampler output for Phase A."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .style_transfer import load_per_reference_resampler

    cfg = config["synthetic_teacher"]
    cache_cfg = cfg.get("style_token_cache", {})
    root = _root(config, destination)
    output = root / str(cache_cfg.get("output_directory", "resampler_tokens"))
    output.mkdir(parents=True, exist_ok=True)
    final_manifest = output / "manifest.parquet"
    summary_path = output / "summary.json"
    if final_manifest.exists() and summary_path.exists():
        rows = read_records(final_manifest)
        summary = {
            "images": len(rows), "shards": len({row["token_shard"] for row in rows}),
            "storage_bytes": sum(path.stat().st_size for path in output.glob("part-*.safetensors")),
            "reused": len(rows),
        }
        return summary

    validated = [
        row for row in read_records(root / "validated_manifest.parquet")
        if _bootstrap_eligible(row)
    ]
    feature_rows = {
        int(row["id"]): row for row in read_records(root / "style_features" / "manifest.parquet")
    }
    rows = [{**row, **feature_rows[int(row["id"])]} for row in validated]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["feature_shard"])].append(row)

    device = str(cfg.get("device", "cuda"))
    resampler_cfg = config["style_transfer"]["resampler"]
    resampler = load_per_reference_resampler(destination, resampler_cfg, device, trainable=False)
    batch_size = int(cache_cfg.get("batch_size", 16))
    taps = [18, 24]
    records: list[dict[str, Any]] = []
    for shard_index, (feature_shard, shard_rows) in enumerate(sorted(grouped.items())):
        token_name = f"part-{shard_index:05d}.safetensors"
        token_path = output / token_name
        row_path = output / f"part-{shard_index:05d}.parquet"
        if token_path.exists() and row_path.exists():
            records.extend(read_records(row_path))
            print(f"reused Resampler token shard {shard_index + 1}/{len(grouped)}", flush=True)
            continue
        token_parts = []
        with safe_open(root / "style_features" / feature_shard, framework="pt", device="cpu") as handle:
            for offset in range(0, len(shard_rows), batch_size):
                part = shard_rows[offset : offset + batch_size]
                image_ids = [int(row["id"]) for row in part]
                features = {
                    layer: torch.stack([
                        handle.get_tensor(f"{image_id}.layer_{layer:02d}_spatial")
                        for image_id in image_ids
                    ]).to(device, non_blocking=True)
                    for layer in taps
                }
                spatial_tokens = torch.tensor(
                    [int(row["spatial_tokens"]) for row in part], device=device
                )
                mask = torch.arange(features[18].shape[1], device=device)[None] < spatial_tokens[:, None]
                global_feature = torch.stack([
                    handle.get_tensor(f"{image_id}.layer_24_siglip_cls") for image_id in image_ids
                ]).to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, tokens = resampler.encode(features, mask, global_feature)
                token_parts.append(tokens.to(device="cpu", dtype=torch.bfloat16).contiguous())
        tokens = torch.cat(token_parts)
        expected = (len(shard_rows), 128, 1024)
        if tuple(tokens.shape) != expected or not torch.isfinite(tokens).all():
            raise RuntimeError(f"Invalid Resampler token shard {feature_shard}: {tuple(tokens.shape)}")
        temporary = token_path.with_suffix(".safetensors.tmp")
        save_file({"tokens": tokens}, temporary)
        temporary.replace(token_path)
        shard_records = [
            {
                "id": int(row["id"]), "artist": str(row["artist"]),
                "artist_split": str(row["artist_split"]),
                "content_split": str(row["content_split"]),
                "content_index": int(row["content_index"]),
                "seed_index": int(row["seed_index"]),
                "token_shard": token_name, "token_row": index,
                "slots": int(tokens.shape[1]), "style_dim": int(tokens.shape[2]),
            }
            for index, row in enumerate(shard_rows)
        ]
        write_records(row_path, shard_records)
        records.extend(shard_records)
        print(
            f"cached Resampler token shard {shard_index + 1}/{len(grouped)} "
            f"({len(records)}/{len(rows)} images)", flush=True,
        )
    if {int(row["id"]) for row in records} != {int(row["id"]) for row in rows}:
        raise RuntimeError("Resampler token cache ID set does not match eligible synthetic references")
    write_records(final_manifest, sorted(records, key=lambda row: int(row["id"])))
    summary = {
        "images": len(records), "shards": len(grouped), "slots": 128, "style_dim": 1024,
        "dtype": "bfloat16",
        "storage_bytes": sum(path.stat().st_size for path in output.glob("part-*.safetensors")),
    }
    write_json(summary_path, summary)
    return summary


def _native_rms_norm(value: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    normalized = value * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + 1e-6).to(value.dtype)
    return normalized if weight is None else normalized * weight


def _attention_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    # q/k/v are [heads, tokens, head_dim].
    attended = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
    return F.linear(attended.transpose(0, 1).reshape(attended.shape[1], -1), output_weight)


def _batched_attention_output(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output_weight: torch.Tensor
) -> torch.Tensor:
    """Project batched [B,H,Q,D] attention back to Anima hidden space."""
    attended = F.scaled_dot_product_attention(q, k, v)
    return F.linear(
        attended.transpose(1, 2).reshape(attended.shape[0], attended.shape[2], -1),
        output_weight,
    )


def _load_resampler_token_cache(
    root: Path, device: str = "cpu"
) -> tuple[dict[int, dict[str, Any]], dict[str, torch.Tensor]]:
    """Load the small Phase-A cache once, optionally keeping it on the GPU."""
    from safetensors.torch import load_file

    cache_root = root / "resampler_tokens"
    rows = read_records(cache_root / "manifest.parquet")
    row_by_id = {int(row["id"]): row for row in rows}
    tensors = {
        name: load_file(cache_root / name, device="cpu")["tokens"].clone().to(device)
        for name in sorted({str(row["token_shard"]) for row in rows})
    }
    return row_by_id, tensors


def _load_resident_feature_cache(
    rows: list[dict[str, Any]],
    feature_root: Path,
    taps: list[int],
    global_layer: int,
    device: str = "cpu",
    workers: int = 1,
) -> dict[str, Any]:
    """Materialize C-RADIO shards into contiguous system-RAM tensors once.

    The synthetic corpus is fixed-resolution, but the cache retains a token
    count so the same path remains valid if a later corpus contains buckets.
    This replaces thousands of per-step safetensors opens with index_select.
    """
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    id_to_index = {int(row["id"]): index for index, row in enumerate(ordered)}
    max_tokens = max(int(row["spatial_tokens"]) for row in ordered)
    spatial_dim = int(ordered[0]["spatial_dim"])
    features = {
        layer: torch.empty(len(ordered), max_tokens, spatial_dim, dtype=torch.float16)
        for layer in taps
    }
    global_values: torch.Tensor | None = None
    token_counts = torch.empty(len(ordered), dtype=torch.int32)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        grouped[str(row["feature_shard"])].append(row)
    shard_groups = sorted(grouped.items())

    def load_shard(item: tuple[str, list[dict[str, Any]]]) -> int:
        name, shard_rows = item
        with safe_open(feature_root / name, framework="pt", device="cpu") as handle:
            for row in shard_rows:
                image_id = int(row["id"])
                index = id_to_index[image_id]
                count = int(row["spatial_tokens"])
                token_counts[index] = count
                for layer in taps:
                    value = handle.get_tensor(f"{image_id}.layer_{layer:02d}_spatial")
                    features[layer][index, :count].copy_(value)
                value = handle.get_tensor(f"{image_id}.layer_{global_layer:02d}_siglip_cls").flatten()
                if global_values is None:
                    raise RuntimeError("Global cache must be allocated before parallel loading")
                global_values[index].copy_(value)
        return len(shard_rows)

    # Every C-RADIO summary has the backbone width. Allocate this before
    # workers start so they only write disjoint rows into fixed storage.
    global_values = torch.empty(len(ordered), spatial_dim, dtype=torch.float16)
    loaded = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(shard_groups))) as executor:
            for shard_index, count in enumerate(executor.map(load_shard, shard_groups), start=1):
                loaded += count
                print(
                    f"resident C-RADIO cache {shard_index}/{len(grouped)} "
                    f"({loaded}/{len(ordered)} images)",
                    flush=True,
                )
    else:
        for shard_index, item in enumerate(shard_groups, start=1):
            loaded += load_shard(item)
            print(
                f"resident C-RADIO cache {shard_index}/{len(grouped)} "
                f"({loaded}/{len(ordered)} images)",
                flush=True,
            )
    if global_values is None:
        raise RuntimeError("Resident C-RADIO cache is empty")
    gib = (
        sum(value.numel() * value.element_size() for value in features.values())
        + global_values.numel() * global_values.element_size()
    ) / (1024**3)
    if device != "cpu":
        print(f"moving {gib:.2f} GiB C-RADIO cache to {device}", flush=True)
        features = {layer: value.to(device) for layer, value in features.items()}
        global_values = global_values.to(device)
        token_counts = token_counts.to(device)
    print(f"resident C-RADIO cache ready: {gib:.2f} GiB on {device}", flush=True)
    return {
        "id_to_index": id_to_index,
        "features": features,
        "global": global_values,
        "token_counts": token_counts,
        "max_tokens": max_tokens,
    }


def _enable_phase_b_resampler(
    resampler: torch.nn.Module, encoder_layers: int = 1
) -> list[torch.nn.Parameter]:
    """Open the style boundary and a configurable number of final encoder blocks."""
    resampler.requires_grad_(False)
    if encoder_layers == 0:
        resampler.eval()
        return []
    if not 1 <= encoder_layers <= len(resampler.encoder):
        raise ValueError(
            f"resampler_trainable_encoder_layers must be in [1, {len(resampler.encoder)}]"
        )
    modules = [*resampler.encoder[-encoder_layers:], resampler.style_projection]
    parameters: list[torch.nn.Parameter] = []
    for module in modules:
        module.requires_grad_(True)
        parameters.extend(module.parameters())
    resampler.train()
    return parameters


def train_offline_kvo_bootstrap(
    config: dict[str, Any], destination: Path, *, steps_override: int | None = None,
    phase: str = "a",
) -> dict[str, Any]:
    """Distill native artist attention effects without running Anima's transformer."""
    from safetensors.torch import load_file

    from .style_transfer import SharedLowRankStyleAdapter, load_per_reference_resampler
    from .tap_resampler import _load_feature_batch

    cfg = config["synthetic_teacher"]
    if phase not in {"a", "b"}:
        raise ValueError(f"Unknown offline bootstrap phase: {phase}")
    training = cfg.get("offline_bootstrap" if phase == "a" else "offline_phase_b", {})
    root = _root(config, destination)
    output = root / str(training.get("output_directory", "offline_kvo_bootstrap"))
    output.mkdir(parents=True, exist_ok=True)
    validated = [row for row in read_records(root / "validated_manifest.parquet") if _bootstrap_eligible(row)]
    feature_rows = {int(row["id"]): row for row in read_records(root / "style_features" / "manifest.parquet")}
    rows = [{**row, **feature_rows[int(row["id"])]} for row in validated]
    by_split = {name: [row for row in rows if row["artist_split"] == name] for name in ("train", "validation", "meta_test")}
    train_rows = [row for row in by_split["train"] if row["content_split"] == "train"]
    validation_rows = [row for row in by_split["validation"] if row["content_split"] == "validation"]
    meta_rows = [row for row in by_split["meta_test"] if row["content_split"] == "test"]
    if not train_rows or not validation_rows or not meta_rows:
        raise RuntimeError("Offline bootstrap split is empty")

    device = str(cfg.get("device", "cuda"))
    resampler_cfg = config["style_transfer"]["resampler"]
    resampler = load_per_reference_resampler(destination, resampler_cfg, device, trainable=False)
    adapter = SharedLowRankStyleAdapter(**config["style_transfer"]["adapter"]).to(device)
    adapter.aggregator.requires_grad_(False)
    adapter.null_tokens.requires_grad_(False)
    adapter.train()
    centered_head = None
    if bool(training.get("centered_effect_head", False)):
        centered_head = ArtistCenteredEffectHead(
            style_dim=int(resampler_cfg.get("style_dim", 1024)),
            hidden_dim=int(config["style_transfer"]["adapter"]["hidden_dim"]),
            latent_dim=int(training.get("centered_effect_latent_dim", 512)),
        ).to(device)
    if phase == "b":
        initialization = training.get("initial_checkpoint") or training.get(
            "phase_a_checkpoint", "offline_kvo_bootstrap/checkpoints/best.pt"
        )
        initialization_checkpoint = root / str(initialization)
        state = torch.load(
            initialization_checkpoint, map_location="cpu", weights_only=False
        )
        adapter.load_state_dict(state["adapter"])
        if centered_head is not None and state.get("centered_head") is not None:
            centered_head.load_state_dict(state["centered_head"])
        resampler_initialization = training.get("resampler_initial_checkpoint")
        if resampler_initialization:
            resampler_state = torch.load(
                root / str(resampler_initialization),
                map_location="cpu",
                weights_only=False,
            )
            if resampler_state.get("resampler") is None:
                raise RuntimeError(
                    f"Resampler checkpoint has no Resampler state: {resampler_initialization}"
                )
            resampler.load_state_dict(resampler_state["resampler"])
            print(
                f"initialized Phase B Resampler from {root / str(resampler_initialization)}",
                flush=True,
            )
        elif state.get("resampler") is not None:
            resampler.load_state_dict(state["resampler"])
        print(f"initialized Phase B from {initialization_checkpoint}", flush=True)
    basis = load_file(root / "anima_kv_teacher" / "native_cross_attention.safetensors", device="cpu")
    basis = {key: value.to(device=device, dtype=torch.bfloat16) for key, value in basis.items()}
    resident_device = device if bool(training.get("gpu_resident_small_caches", True)) else "cpu"
    probe_values = {
        key: value.to(resident_device)
        for key, value in load_file(
            root / "query_probe_bank" / "queries.safetensors", device="cpu"
        ).items()
    }
    probe_rows = read_records(root / "query_probe_bank" / "manifest.parquet")
    probe_keys: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in probe_rows:
        probe_keys[(int(row["content_index"]), int(row["block"]))].append(str(row["key"]))
    text_rows = {int(row["condition_id"]): row for row in read_records(root / "text" / "manifest.parquet")}
    artist_content_condition: dict[tuple[str, int], int] = {}
    for row in rows:
        artist_content_condition.setdefault(
            (str(row["artist"]), int(row["content_index"])),
            int(row["artist_condition_id"]),
        )
    text_cache: dict[str, torch.Tensor] = {}
    for name in sorted({str(row["cache_shard"]) for row in text_rows.values()}):
        text_cache[name] = load_file(root / "text" / name, device="cpu")[
            "conditioning"
        ].to(resident_device)

    def text(condition_id: int) -> torch.Tensor:
        row = text_rows[condition_id]
        name = str(row["cache_shard"])
        # Anima receives all 512 post-LLM positions without a text padding
        # mask.  The trailing zero K/V positions therefore remain in the
        # attention softmax denominator and must not be trimmed here.
        return text_cache[name][int(row["row_index"])].to(device=device, dtype=torch.bfloat16)

    heads = int(config["style_transfer"]["adapter"]["heads"])
    hidden = int(config["style_transfer"]["adapter"]["hidden_dim"])
    head_dim = hidden // heads

    def projected(condition: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        kw = basis[f"block_{block:02d}.k_proj.weight"]
        vw = basis[f"block_{block:02d}.v_proj.weight"]
        k = F.linear(condition, kw).reshape(-1, heads, head_dim).transpose(0, 1)
        v = F.linear(condition, vw).reshape(-1, heads, head_dim).transpose(0, 1)
        norm_weight = basis.get(f"block_{block:02d}.k_norm.weight")
        return _native_rms_norm(k, norm_weight), v

    taps = [18, 24]
    variant_global = "native_24"
    batch_size = int(training.get("batch_size", 2))
    seed = int(training.get("seed", int(cfg.get("seed", 20260812)) ^ 0x0FF1))
    generator = random.Random(seed)
    total_steps = int(steps_override or training.get("steps", 10000))
    token_rows, token_tensors = _load_resampler_token_cache(root, resident_device)
    resident_features = None
    if phase == "b" and bool(training.get("ram_resident_features", True)):
        configured_feature_root = training.get("local_feature_directory")
        feature_root = (
            Path(str(configured_feature_root))
            if configured_feature_root and Path(str(configured_feature_root)).is_dir()
            else root / "style_features"
        )
        feature_cache_device = (
            device if bool(training.get("gpu_resident_features", False)) else "cpu"
        )
        resident_features = _load_resident_feature_cache(
            rows,
            feature_root,
            [18, 24],
            24,
            feature_cache_device,
            int(training.get("feature_cache_load_workers", 8)),
        )
    bridge_parameters = [value for value in adapter.bridge_parameters() if value.requires_grad]
    connector_parameters = [
        value for module in (adapter.connector_trunk, adapter.connector_branches)
        for value in module.parameters() if value.requires_grad
    ]
    if adapter.block_embeddings is not None and adapter.block_embeddings.requires_grad:
        connector_parameters.append(adapter.block_embeddings)
    kvo_parameters = [
        value for value in (adapter.kv_parameters() + adapter.output_parameters())
        if value.requires_grad
    ]
    grouped_ids = {id(value) for value in bridge_parameters + connector_parameters + kvo_parameters}
    remaining_parameters = [
        value for value in adapter.parameters()
        if value.requires_grad and id(value) not in grouped_ids
    ]
    # Aggregator/null tokens are frozen above, so every live adapter parameter
    # must belong to a functional optimizer group.
    if remaining_parameters:
        raise RuntimeError(f"Unassigned adapter parameters: {len(remaining_parameters)}")
    adapter_parameters = bridge_parameters + connector_parameters + kvo_parameters
    if centered_head is not None and bool(training.get("freeze_base_connector", True)):
        adapter.requires_grad_(False)
        bridge_parameters = []
        connector_parameters = []
        kvo_parameters = []
        adapter_parameters = []
    resampler_parameters = (
        _enable_phase_b_resampler(
            resampler, int(training.get("resampler_trainable_encoder_layers", 1))
        )
        if phase == "b" else []
    )
    parameters = adapter_parameters + resampler_parameters
    optimizer_groups: list[dict[str, Any]] = []
    for name, values, rate in (
        ("bridge", bridge_parameters, float(training.get("bridge_learning_rate", 2e-6))),
        ("connector", connector_parameters, float(training.get("connector_learning_rate", 5e-6))),
        ("kvo", kvo_parameters, float(training.get("learning_rate", 2e-5))),
    ):
        if values:
            optimizer_groups.append({"params": values, "lr": rate, "name": name})
    if centered_head is not None:
        optimizer_groups.append({
            "params": list(centered_head.parameters()),
            "lr": float(training.get("centered_effect_learning_rate", 1e-4)),
            "name": "centered_head",
        })
    if resampler_parameters:
        optimizer_groups.append({
            "params": resampler_parameters, "name": "resampler",
            "lr": float(training.get("resampler_learning_rate", 1e-5)),
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.95), weight_decay=float(training.get("weight_decay", 0.01)),
    )
    warmup = int(training.get("warmup_steps", 500))

    def lr_scale(step: int) -> float:
        if step <= warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return float(training.get("minimum_lr_ratio", 0.1)) + (1 - float(training.get("minimum_lr_ratio", 0.1))) * 0.5 * (1 + math.cos(math.pi * progress))

    def cached_tokens(batch_rows: list[dict[str, Any]]) -> torch.Tensor:
        return torch.stack([
            token_tensors[str(token_rows[int(row["id"])]["token_shard"])][
                int(token_rows[int(row["id"])]["token_row"])
            ]
            for row in batch_rows
        ]).to(device=device, dtype=torch.bfloat16)

    def encode(batch_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = cached_tokens(batch_rows)
        if phase == "a":
            return baseline, baseline.new_zeros(())
        if resident_features is not None:
            cache_device = resident_features["features"][18].device
            indices = torch.tensor(
                [resident_features["id_to_index"][int(row["id"])] for row in batch_rows],
                dtype=torch.long,
                device=cache_device,
            )
            features = {
                layer: value.index_select(0, indices).to(device, non_blocking=True)
                for layer, value in resident_features["features"].items()
            }
            counts = resident_features["token_counts"].index_select(0, indices).to(device)
            mask = torch.arange(resident_features["max_tokens"], device=device)[None] < counts[:, None]
            global_feature = resident_features["global"].index_select(0, indices).to(
                device, non_blocking=True
            )
        else:
            configured_feature_root = training.get("local_feature_directory")
            feature_root = (
                Path(str(configured_feature_root))
                if configured_feature_root and Path(str(configured_feature_root)).is_dir()
                else root / "style_features"
            )
            features, _, mask, _, global_feature = _load_feature_batch(
                batch_rows, feature_root, taps, taps, variant_global
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, tokens = resampler.encode(
                {key: value.to(device) for key, value in features.items()}, mask.to(device), global_feature.to(device)
            )
        drift = (1 - F.cosine_similarity(tokens.float(), baseline.float(), dim=-1)).mean()
        return tokens, drift

    def loss_for(
        batch_rows: list[dict[str, Any]], *, train: bool,
        reference_rows: list[dict[str, Any]] | None = None,
        positive_reference_rows: list[dict[str, Any]] | None = None,
        detach_representation: bool = False,
        current_step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        reference_rows = batch_rows if reference_rows is None else reference_rows
        tokens, drift_loss = encode(reference_rows)
        artist_token_loss = tokens.new_zeros(())
        artist_token_accuracy = tokens.new_zeros(())
        if train and positive_reference_rows is not None:
            positive_tokens, positive_drift = encode(positive_reference_rows)
            drift_loss = 0.5 * (drift_loss + positive_drift)
            def token_descriptor(value: torch.Tensor) -> torch.Tensor:
                normalized = F.layer_norm(value.float(), (value.shape[-1],))
                return F.normalize(
                    torch.cat(
                        (normalized.mean(dim=1), normalized.std(dim=1, correction=0)),
                        dim=-1,
                    ),
                    dim=-1,
                )
            anchor_descriptor = token_descriptor(tokens)
            positive_descriptor = token_descriptor(positive_tokens)
            token_logits = anchor_descriptor @ positive_descriptor.T / float(
                training.get("artist_token_temperature", 0.07)
            )
            token_labels = torch.arange(len(reference_rows), device=tokens.device)
            artist_token_loss = 0.5 * (
                F.cross_entropy(token_logits, token_labels)
                + F.cross_entropy(token_logits.T, token_labels)
            )
            artist_token_accuracy = 0.5 * (
                (token_logits.argmax(1) == token_labels).float().mean()
                + (token_logits.argmax(0) == token_labels).float().mean()
            )
        # Keep trainable parameters and optimizer state in FP32, but execute
        # connector/native K/V/O projections and SDPA in BF16 like Anima.
        # The cached native basis and query probes are BF16; autocast supplies
        # a single consistent compute dtype without permanently downcasting
        # the trainable connector weights.
        output_losses = []
        cosines = []
        rms_losses = []
        zero_mses = []
        student_mses = []
        rank_losses = []
        contrastive_losses = []
        contrastive_accuracies = []
        all_pair_direction_losses = []
        all_pair_magnitude_losses = []
        centered_pair_losses = []
        centered_pair_accuracies = []
        reference_teacher_cosines = []
        reference_teacher_relative_distances = []
        block_count = int(training.get("blocks_per_step", 7)) if train else 28
        if train and block_count == adapter.blocks_per_group:
            group = generator.randrange(adapter.blocks // adapter.blocks_per_group)
            blocks = list(range(group * adapter.blocks_per_group, (group + 1) * adapter.blocks_per_group))
        else:
            blocks = generator.sample(range(28), block_count) if block_count < 28 else list(range(28))
        functional_block_count = min(
            len(blocks), int(training.get("functional_blocks_per_step", 1))
        )
        functional_blocks = set(
            generator.sample(blocks, functional_block_count)
            if train else []
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            contexts = adapter.selected_block_context_tokens(tokens, blocks)
        if detach_representation:
            contexts = {index: value.detach() for index, value in contexts.items()}
        artist_contexts = torch.stack([text(int(row["artist_condition_id"])) for row in batch_rows])
        content_contexts = torch.stack([text(int(row["content_condition_id"])) for row in batch_rows])
        reference_artist_contexts = None
        if not train:
            reference_artist_contexts = torch.stack([
                text(artist_content_condition[
                    (str(reference["artist"]), int(target["content_index"]))
                ])
                for target, reference in zip(batch_rows, reference_rows, strict=True)
            ])
        for block in blocks:
            queries = []
            for row in batch_rows:
                available_queries = probe_keys[(int(row["content_index"]), block)]
                query_key = (
                    generator.choice(available_queries)
                    if train
                    else available_queries[(int(row["id"]) + block) % len(available_queries)]
                )
                queries.append(probe_values[query_key])
            q = torch.stack(queries).to(device=device, dtype=torch.bfloat16).transpose(1, 2)
            kw = basis[f"block_{block:02d}.k_proj.weight"]
            vw = basis[f"block_{block:02d}.v_proj.weight"]
            ow = basis[f"block_{block:02d}.output_proj.weight"]
            norm_weight = basis.get(f"block_{block:02d}.k_norm.weight")

            def native_kv(condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                k = F.linear(condition, kw).reshape(condition.shape[0], -1, heads, head_dim).transpose(1, 2)
                v = F.linear(condition, vw).reshape(condition.shape[0], -1, heads, head_dim).transpose(1, 2)
                return _native_rms_norm(k, norm_weight), v

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ka, va = native_kv(artist_contexts)
                kc, vc = native_kv(content_contexts)
                teacher = (
                    _batched_attention_output(q, ka, va, ow)
                    - _batched_attention_output(q, kc, vc, ow)
                )
                if reference_artist_contexts is not None:
                    kr, vr = native_kv(reference_artist_contexts)
                    reference_teacher = (
                        _batched_attention_output(q, kr, vr, ow)
                        - _batched_attention_output(q, kc, vc, ow)
                    )

            def student_output(
                context: torch.Tensor, query: torch.Tensor = q,
                source_tokens: torch.Tensor = tokens,
            ) -> torch.Tensor:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sk = F.linear(context, kw) + adapter.k_up[block](adapter.k_down[block](context))
                    sv = F.linear(context, vw) + adapter.v_up[block](adapter.v_down[block](context))
                    sk = _native_rms_norm(
                        sk.reshape(context.shape[0], -1, heads, head_dim).transpose(1, 2),
                        norm_weight,
                    )
                    sv = sv.reshape(context.shape[0], -1, heads, head_dim).transpose(1, 2)
                    attended = F.scaled_dot_product_attention(query, sk, sv)
                    attended = attended.transpose(1, 2).reshape(context.shape[0], attended.shape[2], hidden)
                    output = F.linear(attended, ow) + adapter.o_up[block](adapter.o_down[block](attended))
                    if centered_head is not None:
                        output = output + centered_head(source_tokens, query, block)
                    return output

            student = student_output(contexts[block])
            use_rank = train and len(batch_rows) > 1 and float(training.get("functional_rank_weight", 0.0)) > 0
            wrong_student = student_output(contexts[block].roll(1, dims=0)) if use_rank else None
            teacher_float, student_float = teacher.float(), student.float()
            if reference_artist_contexts is not None:
                reference_teacher_float = reference_teacher.float()
                teacher_flat = teacher_float.flatten(1)
                reference_teacher_flat = reference_teacher_float.flatten(1)
                reference_teacher_cosines.extend(F.cosine_similarity(
                    teacher_flat, reference_teacher_flat, dim=1
                ).unbind())
                reference_teacher_relative_distances.extend((
                    (teacher_flat - reference_teacher_flat).square().mean(1).sqrt()
                    / teacher_flat.square().mean(1).sqrt().clamp_min(1e-6)
                ).unbind())
            scale = teacher_float.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
            per_sample_output = F.smooth_l1_loss(
                student_float / scale, teacher_float / scale, beta=0.1, reduction="none"
            ).mean(dim=(1, 2))
            output_losses.extend(per_sample_output.unbind())
            block_cosines = F.cosine_similarity(
                student_float.flatten(1), teacher_float.flatten(1), dim=1
            )
            cosines.extend(block_cosines.unbind())
            student_rms = student_float.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
            teacher_rms = scale.flatten()
            rms_losses.extend((torch.log(student_rms) - torch.log(teacher_rms)).square().unbind())
            student_mses.extend((student_float - teacher_float).square().mean(dim=(1, 2)).unbind())
            zero_mses.extend(teacher_float.square().mean(dim=(1, 2)).unbind())
            if use_rank and wrong_student is not None:
                correct_distance = ((student_float - teacher_float) / scale).square().mean(dim=(1, 2))
                wrong_distance = ((wrong_student.float() - teacher_float) / scale).square().mean(dim=(1, 2))
                rank_losses.extend(F.relu(
                    correct_distance - wrong_distance
                    + float(training.get("functional_rank_margin", 0.02))
                ).unbind())
            contrastive_weight = float(training.get("functional_contrastive_weight", 0.0))
            if (
                train and len(batch_rows) > 1 and contrastive_weight > 0
                and block in functional_blocks
            ):
                count = len(batch_rows)
                # Every target/query attends to every candidate reference.
                # Keeping the query fixed across candidates prevents content
                # or probe identity from solving the matching task.
                candidate_contexts = contexts[block][None].expand(
                    count, count, -1, -1
                ).reshape(count * count, contexts[block].shape[1], -1)
                candidate_queries = q[:, None].expand(
                    count, count, -1, -1, -1
                ).reshape(count * count, *q.shape[1:])
                candidate_outputs = student_output(
                    candidate_contexts, candidate_queries,
                    tokens[None].expand(count, count, -1, -1).reshape(
                        count * count, tokens.shape[1], tokens.shape[2]
                    ),
                ).float().reshape(count, count, *teacher_float.shape[1:])
                candidate_directions = F.normalize(candidate_outputs.flatten(2), dim=-1)
                teacher_directions = F.normalize(teacher_float.flatten(1), dim=-1)
                similarities = torch.einsum(
                    "bcd,bd->bc", candidate_directions, teacher_directions
                )
                logits = similarities / float(
                    training.get("functional_contrastive_temperature", 0.1)
                )
                labels = torch.arange(count, device=logits.device)
                contrastive_losses.append(F.cross_entropy(logits, labels))
                contrastive_accuracies.append((logits.argmax(dim=1) == labels).float().mean())
                if float(training.get("functional_all_pairs_weight", 0.0)) > 0:
                    pair_conditions = torch.stack([
                        text(artist_content_condition[
                            (str(reference["artist"]), int(target["content_index"]))
                        ])
                        for target in batch_rows
                        for reference in reference_rows
                    ])
                    pair_content = content_contexts[:, None].expand(
                        count, count, -1, -1
                    ).reshape(count * count, *content_contexts.shape[1:])
                    pair_q = q[:, None].expand(
                        count, count, -1, -1, -1
                    ).reshape(count * count, *q.shape[1:])
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pair_ka, pair_va = native_kv(pair_conditions)
                        pair_kc, pair_vc = native_kv(pair_content)
                        pair_teacher = (
                            _batched_attention_output(pair_q, pair_ka, pair_va, ow)
                            - _batched_attention_output(pair_q, pair_kc, pair_vc, ow)
                        ).float().reshape_as(candidate_outputs)
                    pair_student_flat = candidate_outputs.flatten(2)
                    pair_teacher_flat = pair_teacher.flatten(2)
                    all_pair_direction_losses.append(
                        (1 - F.cosine_similarity(
                            pair_student_flat, pair_teacher_flat, dim=-1
                        )).mean()
                    )
                    pair_student_rms = pair_student_flat.square().mean(-1).sqrt().clamp_min(1e-6)
                    pair_teacher_rms = pair_teacher_flat.square().mean(-1).sqrt().clamp_min(1e-6)
                    all_pair_magnitude_losses.append(F.smooth_l1_loss(
                        torch.log(pair_student_rms),
                        torch.log(pair_teacher_rms),
                        beta=0.5,
                    ))
                    centered_student = pair_student_flat - pair_student_flat.mean(
                        dim=1, keepdim=True
                    )
                    centered_teacher = pair_teacher_flat - pair_teacher_flat.mean(
                        dim=1, keepdim=True
                    )
                    centered_student = F.normalize(centered_student, dim=-1)
                    centered_teacher = F.normalize(centered_teacher, dim=-1)
                    centered_logits = torch.einsum(
                        "bcd,bkd->bck", centered_student, centered_teacher
                    ) / float(training.get("functional_centered_temperature", 0.07))
                    centered_labels = torch.arange(count, device=centered_logits.device)
                    centered_pair_losses.append(F.cross_entropy(
                        centered_logits.reshape(count * count, count),
                        centered_labels.repeat(count),
                    ))
                    centered_pair_accuracies.append(
                        (
                            centered_logits.argmax(dim=-1)
                            == centered_labels[None]
                        ).float().mean()
                    )
        output_loss = torch.stack(output_losses).mean()
        cosine_loss = (1 - torch.stack(cosines)).mean()
        rms_loss = torch.stack(rms_losses).mean()
        rank_loss = torch.stack(rank_losses).mean() if rank_losses else output_loss.new_zeros(())
        contrastive_loss = (
            torch.stack(contrastive_losses).mean()
            if contrastive_losses else output_loss.new_zeros(())
        )
        all_pair_direction_loss = (
            torch.stack(all_pair_direction_losses).mean()
            if all_pair_direction_losses else output_loss.new_zeros(())
        )
        all_pair_magnitude_loss = (
            torch.stack(all_pair_magnitude_losses).mean()
            if all_pair_magnitude_losses else output_loss.new_zeros(())
        )
        centered_pair_loss = (
            torch.stack(centered_pair_losses).mean()
            if centered_pair_losses else output_loss.new_zeros(())
        )
        centered_pair_accuracy = (
            torch.stack(centered_pair_accuracies).mean()
            if centered_pair_accuracies else output_loss.new_zeros(())
        )
        contrastive_scale = 1.0
        if train:
            contrastive_start = int(
                training.get("functional_contrastive_start_step", 0)
            )
            contrastive_ramp = int(
                training.get("functional_contrastive_ramp_steps", 0)
            )
            contrastive_scale = min(
                1.0,
                max(0.0, (current_step - contrastive_start) / max(contrastive_ramp, 1)),
            )
        loss = (
            output_loss
            + float(training.get("direction_weight", 0.2)) * cosine_loss
            + float(training.get("magnitude_weight", 0.02)) * rms_loss
            + float(training.get("representation_drift_weight", 0.0)) * drift_loss
            + float(training.get("functional_rank_weight", 0.0)) * rank_loss
            + contrastive_scale
            * float(training.get("functional_contrastive_weight", 0.0))
            * contrastive_loss
            + contrastive_scale
            * float(training.get("functional_all_pairs_weight", 0.0))
            * (
                all_pair_direction_loss
                + float(training.get("functional_all_pairs_magnitude_weight", 0.02))
                * all_pair_magnitude_loss
            )
            + contrastive_scale
            * float(training.get("artist_token_contrastive_weight", 0.0))
            * artist_token_loss
            + contrastive_scale
            * float(training.get("functional_centered_all_pairs_weight", 0.0))
            * centered_pair_loss
        )
        zero_mse = torch.stack(zero_mses).mean()
        student_mse = torch.stack(student_mses).mean()
        return loss, {
            "loss": float(loss.detach()), "output_loss": float(output_loss.detach()),
            "cosine": float(torch.stack(cosines).mean().detach()),
            "rms_loss": float(rms_loss.detach()),
            "representation_drift": float(drift_loss.detach()),
            "functional_rank_loss": float(rank_loss.detach()),
            "functional_contrastive_loss": float(contrastive_loss.detach()),
            "functional_contrastive_accuracy": float(
                torch.stack(contrastive_accuracies).mean().detach()
                if contrastive_accuracies else 0.0
            ),
            "functional_contrastive_scale": float(contrastive_scale),
            "functional_all_pairs_direction_loss": float(
                all_pair_direction_loss.detach()
            ),
            "functional_all_pairs_magnitude_loss": float(
                all_pair_magnitude_loss.detach()
            ),
            "artist_token_contrastive_loss": float(artist_token_loss.detach()),
            "artist_token_contrastive_accuracy": float(artist_token_accuracy.detach()),
            "functional_centered_all_pairs_loss": float(centered_pair_loss.detach()),
            "functional_centered_all_pairs_accuracy": float(
                centered_pair_accuracy.detach()
            ),
            "reference_teacher_cosine": float(
                torch.stack(reference_teacher_cosines).mean().detach()
                if reference_teacher_cosines else 1.0
            ),
            "reference_teacher_relative_distance": float(
                torch.stack(reference_teacher_relative_distances).mean().detach()
                if reference_teacher_relative_distances else 0.0
            ),
            "zero_improvement": float((1 - student_mse / zero_mse.clamp_min(1e-12)).detach()),
            "teacher_rms": float(torch.stack(zero_mses).mean().sqrt().detach()),
            "student_rms": float(torch.stack(student_mses).mean().sqrt().detach()),
        }

    all_by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        all_by_artist[str(row["artist"])].append(row)

    def distinct_references(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        references = []
        for target in targets:
            candidates = [
                row for row in all_by_artist[str(target["artist"])]
                if int(row["id"]) != int(target["id"]) and row["content_split"] == "train"
            ]
            references.append(candidates[int(target["id"]) % len(candidates)])
        return references

    def wrong_references(correct: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(correct) > 1 and all(
            correct[index]["artist"] != correct[(index + 1) % len(correct)]["artist"]
            for index in range(len(correct))
        ):
            return correct[1:] + correct[:1]
        pool = [row for row in train_rows if row["content_split"] == "train"]
        return [
            next(row for row in pool if row["artist"] != target["artist"])
            for target in correct
        ]

    def evaluate(source: list[dict[str, Any]], batches: int) -> dict[str, float]:
        adapter.eval()
        resampler.eval()
        metrics: list[dict[str, float]] = []
        wrong_metrics: list[dict[str, float]] = []
        with torch.no_grad():
            for index in range(batches):
                start = (index * batch_size) % len(source)
                batch = (source + source)[start : start + batch_size]
                references = distinct_references(batch)
                _, values = loss_for(batch, train=False, reference_rows=references)
                _, wrong = loss_for(batch, train=False, reference_rows=wrong_references(references))
                metrics.append(values)
                wrong_metrics.append(wrong)
        adapter.train()
        if phase == "b":
            resampler.train()
        result = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
        for key in (
            "cosine", "zero_improvement", "output_loss",
            "reference_teacher_cosine", "reference_teacher_relative_distance",
        ):
            result[f"wrong_{key}"] = sum(item[key] for item in wrong_metrics) / len(wrong_metrics)
        result["correct_wrong_cosine_gap"] = result["cosine"] - result["wrong_cosine"]
        result["correct_wrong_improvement_gap"] = result["zero_improvement"] - result["wrong_zero_improvement"]
        return result

    log_every = int(training.get("log_every", 20))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = output / "evaluation.json"
    if evaluation_path.exists():
        history = list(__import__("json").loads(evaluation_path.read_text(encoding="utf-8")).get("history", []))
    else:
        history = []
    best_score = -float("inf")
    best_step = 0
    for record in history:
        value = record["validation"]
        score = (
            float(value["zero_improvement"])
            + 0.1 * float(value["cosine"])
            + 0.05 * float(value.get("correct_wrong_cosine_gap", 0.0))
        )
        if score > best_score:
            best_score, best_step = score, int(record["step"])
    stale_validations = 0
    minimum_steps = int(training.get("minimum_steps", 0))
    early_stop_patience = int(training.get("early_stop_patience", 0))
    by_train_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_train_artist[str(row["artist"])].append(row)
    start_step = 1
    if bool(training.get("resume", True)):
        candidates = sorted(checkpoint_dir.glob("step-*.pt"))
        if candidates:
            resume_state = torch.load(candidates[-1], map_location="cpu", weights_only=False)
            adapter.load_state_dict(resume_state["adapter"])
            if phase == "b" and resume_state.get("resampler") is not None:
                resampler.load_state_dict(resume_state["resampler"])
            optimizer.load_state_dict(resume_state["optimizer"])
            if "random_state" in resume_state:
                generator.setstate(resume_state["random_state"])
            start_step = int(resume_state["step"]) + 1
            print(f"resumed offline-kvo phase={phase} at step={start_step}", flush=True)

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)) and steps_override is None:
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", f"offline-kvo-phase-{phase}")),
            id=str(wandb_cfg.get("id", f"offline-kvo-phase-{phase}-v1")),
            resume="allow",
            config={"phase": phase, **training},
        )

    last_step = start_step - 1
    representation_warmup = int(training.get("representation_warmup_steps", 0))
    for step in range(start_step, total_steps + 1):
        step_started = __import__("time").perf_counter()
        last_step = step
        # Artist-balanced batches make the rolled wrong-reference branch a
        # guaranteed different artist rather than occasionally another image
        # from the same artist.
        batch_artists = generator.sample(list(by_train_artist), batch_size)
        batch = [generator.choice(by_train_artist[artist]) for artist in batch_artists]
        reference_batch = batch
        positive_reference_batch = None
        target_excluded_start = int(training.get("target_excluded_start_step", total_steps + 1))
        target_excluded_end = int(training.get("target_excluded_end_step", target_excluded_start))
        final_exact_probability = float(training.get("exact_self_probability", 1.0))
        if step >= target_excluded_start:
            progress = min(
                1.0,
                (step - target_excluded_start) / max(target_excluded_end - target_excluded_start, 1),
            )
            exact_probability = 1.0 + progress * (final_exact_probability - 1.0)
            reference_batch = [
                target if generator.random() < exact_probability else generator.choice([
                    row for row in by_train_artist[str(target["artist"])]
                    if int(row["id"]) != int(target["id"])
                ])
                for target in batch
            ]
        if phase == "b" and float(training.get("artist_token_contrastive_weight", 0.0)) > 0:
            positive_reference_batch = [
                generator.choice([
                    row for row in by_train_artist[str(reference["artist"])]
                    if int(row["id"]) != int(reference["id"])
                    and int(row["content_index"]) != int(reference["content_index"])
                ])
                for reference in reference_batch
            ]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_for(
            batch,
            train=True,
            reference_rows=reference_batch,
            positive_reference_rows=positive_reference_batch,
            detach_representation=step <= representation_warmup,
            current_step=step,
        )
        loss.backward()
        grad_norms = {
            "bridge": torch.nn.utils.clip_grad_norm_(
                bridge_parameters, float(training.get("bridge_max_grad_norm", 0.01))
            ),
            "connector": torch.nn.utils.clip_grad_norm_(
                connector_parameters, float(training.get("connector_max_grad_norm", 0.1))
            ),
            "kvo": torch.nn.utils.clip_grad_norm_(
                kvo_parameters, float(training.get("max_grad_norm", 1.0))
            ),
        }
        if centered_head is not None:
            grad_norms["centered_head"] = torch.nn.utils.clip_grad_norm_(
                centered_head.parameters(),
                float(training.get("centered_effect_max_grad_norm", 1.0)),
            )
        if resampler_parameters:
            grad_norms["resampler"] = torch.nn.utils.clip_grad_norm_(
                resampler_parameters, float(training.get("resampler_max_grad_norm", 0.1))
            )
        scale = lr_scale(step)
        learning_rates = {
            "bridge": float(training.get("bridge_learning_rate", 2e-6)),
            "connector": float(training.get("connector_learning_rate", 5e-6)),
            "kvo": float(training.get("learning_rate", 2e-5)),
            "resampler": float(training.get("resampler_learning_rate", 1e-5)),
            "centered_head": float(training.get("centered_effect_learning_rate", 1e-4)),
        }
        for group in optimizer.param_groups:
            group_scale = 0.0 if (
                step <= representation_warmup and str(group["name"]) in {"bridge", "connector", "resampler"}
            ) else scale
            group["lr"] = learning_rates[str(group["name"])] * group_scale
        optimizer.step()
        metrics["step_s"] = __import__("time").perf_counter() - step_started
        if step % log_every == 0:
            grad_text = " ".join(f"g_{key}={float(value):.2f}" for key, value in grad_norms.items())
            print(
                f"offline-kvo step={step}/{total_steps} loss={metrics['loss']:.4f} "
                f"cos={metrics['cosine']:.4f} improve={metrics['zero_improvement']:.4f} "
                f"contrast={metrics['functional_contrastive_loss']:.3f}/"
                f"{metrics['functional_contrastive_accuracy']:.3f}/"
                f"{metrics['functional_contrastive_scale']:.2f} "
                f"token={metrics['artist_token_contrastive_loss']:.3f}/"
                f"{metrics['artist_token_contrastive_accuracy']:.3f} "
                f"centered={metrics['functional_centered_all_pairs_loss']:.3f}/"
                f"{metrics['functional_centered_all_pairs_accuracy']:.3f} "
                f"teacher_rms={metrics['teacher_rms']:.6f} student_rms={metrics['student_rms']:.6f} "
                f"step_s={metrics['step_s']:.3f} {grad_text}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    **{f"train/{key}": value for key, value in metrics.items()},
                    **{f"train/{key}_grad_norm": float(value) for key, value in grad_norms.items()},
                    **{f"train/{group['name']}_learning_rate": group["lr"] for group in optimizer.param_groups},
                }, step=step)
        if step % validation_every == 0 or step == total_steps:
            val = evaluate(validation_rows, int(training.get("validation_batches", 8)))
            record = {"step": step, "validation": val}
            history.append(record)
            write_json(output / "evaluation.json", {"history": history, "latest": record})
            score = (
                val["zero_improvement"] + 0.1 * val["cosine"]
                + 0.05 * val["correct_wrong_cosine_gap"]
            )
            if score > best_score:
                best_score, best_step, stale_validations = score, step, 0
                state = {
                    "phase": phase, "step": step, "adapter": adapter.state_dict(),
                    "resampler": resampler.state_dict() if phase == "b" else None,
                    "centered_head": centered_head.state_dict() if centered_head is not None else None,
                    "config": training, "validation": val,
                }
                torch.save(state, checkpoint_dir / "best.tmp")
                (checkpoint_dir / "best.tmp").replace(checkpoint_dir / "best.pt")
            else:
                stale_validations += 1
            print(
                f"offline-kvo phase={phase} validation step={step} "
                f"cos={val['cosine']:.4f} improve={val['zero_improvement']:.4f} "
                f"wrong_cos={val['wrong_cosine']:.4f} gap={val['correct_wrong_cosine_gap']:.4f} "
                f"best={best_step}", flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({f"validation/{key}": value for key, value in val.items()}, step=step)
        if step % checkpoint_every == 0 or step == total_steps:
            state = {
                "phase": phase, "step": step, "adapter": adapter.state_dict(),
                "resampler": resampler.state_dict() if phase == "b" else None,
                "centered_head": centered_head.state_dict() if centered_head is not None else None,
                "optimizer": optimizer.state_dict(), "config": training,
                "random_state": generator.getstate(),
            }
            temporary = checkpoint_dir / f"step-{step:07d}.tmp"
            torch.save(state, temporary)
            temporary.replace(checkpoint_dir / f"step-{step:07d}.pt")
        if (
            early_stop_patience > 0 and step >= minimum_steps
            and stale_validations >= early_stop_patience
        ):
            print(f"offline-kvo phase={phase} early stop at step={step}; best={best_step}", flush=True)
            break
    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    adapter.load_state_dict(best["adapter"])
    if phase == "b" and best.get("resampler") is not None:
        resampler.load_state_dict(best["resampler"])
    if centered_head is not None and best.get("centered_head") is not None:
        centered_head.load_state_dict(best["centered_head"])
    final_validation = evaluate(validation_rows, int(training.get("validation_batches", 8)))
    # Meta-test is deliberately touched once, only after validation selected
    # the checkpoint.  It never participates in early stopping or tuning.
    final_meta = evaluate(meta_rows, int(training.get("meta_test_batches", 8)))
    summary = {
        "phase": phase, "steps": last_step, "best_step": best_step,
        "checkpoint": str((checkpoint_dir / "best.pt").resolve()),
        "validation": final_validation, "meta_test": final_meta,
    }
    write_json(output / "summary.json", summary)
    if wandb_run is not None:
        wandb_run.log({f"meta_test/{key}": value for key, value in final_meta.items()}, step=best_step)
        wandb_run.finish()
    return summary


def smoke_offline_kvo_bootstrap(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    copied = dict(config)
    copied["synthetic_teacher"] = dict(config["synthetic_teacher"])
    copied["synthetic_teacher"]["offline_bootstrap"] = dict(config["synthetic_teacher"]["offline_bootstrap"])
    copied["synthetic_teacher"]["offline_bootstrap"]["output_directory"] = "offline_kvo_smoke"
    copied["synthetic_teacher"]["offline_bootstrap"]["validation_every"] = 2
    copied["synthetic_teacher"]["offline_bootstrap"]["checkpoint_every"] = 2
    copied["synthetic_teacher"]["offline_bootstrap"]["validation_batches"] = 1
    copied["synthetic_teacher"]["offline_bootstrap"]["meta_test_batches"] = 1
    copied["synthetic_teacher"]["offline_bootstrap"]["blocks_per_step"] = 2
    return train_offline_kvo_bootstrap(copied, destination, steps_override=2, phase="a")


def train_offline_kvo_phase_b(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    return train_offline_kvo_bootstrap(config, destination, phase="b")


def smoke_offline_kvo_phase_b(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    copied = dict(config)
    copied["synthetic_teacher"] = dict(config["synthetic_teacher"])
    copied["synthetic_teacher"]["offline_phase_b"] = dict(config["synthetic_teacher"]["offline_phase_b"])
    copied["synthetic_teacher"]["offline_phase_b"]["output_directory"] = "offline_kvo_phase_b_smoke"
    copied["synthetic_teacher"]["offline_phase_b"]["validation_every"] = 2
    copied["synthetic_teacher"]["offline_phase_b"]["checkpoint_every"] = 2
    copied["synthetic_teacher"]["offline_phase_b"]["validation_batches"] = 1
    copied["synthetic_teacher"]["offline_phase_b"]["meta_test_batches"] = 1
    copied["synthetic_teacher"]["offline_phase_b"]["blocks_per_step"] = 2
    copied["synthetic_teacher"]["offline_phase_b"]["resume"] = False
    copied["synthetic_teacher"]["offline_phase_b"].setdefault("wandb", {})["enabled"] = False
    return train_offline_kvo_bootstrap(copied, destination, steps_override=2, phase="b")
