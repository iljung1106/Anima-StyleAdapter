from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from .io import read_records, write_json, write_records


def _root(config: dict[str, Any], destination: Path) -> Path:
    name = config["synthetic_teacher"].get("output_directory", "synthetic_teacher")
    return destination / str(name)


def _feature_descriptor(handle: Any, image_id: int) -> torch.Tensor:
    parts = []
    for layer in (18, 24):
        value = handle.get_tensor(f"{image_id}.layer_{layer:02d}_spatial").float()
        parts.extend((value.mean(0), value.std(0, correction=0)))
    parts.append(handle.get_tensor(f"{image_id}.layer_24_siglip_cls").float().flatten())
    return torch.cat([F.layer_norm(value, value.shape) for value in parts])


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


def validate_synthetic_teacher_corpus(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["synthetic_teacher"]
    bootstrap = cfg.get("bootstrap", {})
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
    missing_images = [row["id"] for row in manifest if not Path(row["local_path"]).is_file()]
    if missing_images:
        errors.append(f"missing images: {len(missing_images)}")

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

    # Decode every image header and verify representative full pixel payloads.
    for index, row in enumerate(manifest):
        try:
            with Image.open(row["local_path"]) as image:
                if index % 64 == 0:
                    image.load()
                if image.size != (int(row["width"]), int(row["height"])):
                    errors.append(f"image size mismatch: {row['id']}")
        except Exception as error:
            errors.append(f"corrupt image {row['id']}: {error}")

    if errors:
        summary = {"valid": False, "errors": errors, "images": len(manifest)}
        write_json(root / "integrity_summary.json", summary)
        raise RuntimeError("Synthetic corpus integrity failed: " + "; ".join(errors[:8]))

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_shard[str(row["feature_shard"])].append(row)
    descriptors: dict[int, torch.Tensor] = {}
    for shard, shard_rows in sorted(by_shard.items()):
        with safe_open(feature_root / shard, framework="pt", device="cpu") as handle:
            for row in shard_rows:
                descriptor = _feature_descriptor(handle, int(row["id"]))
                if not torch.isfinite(descriptor).all():
                    raise FloatingPointError(f"Non-finite C-RADIO descriptor for {row['id']}")
                descriptors[int(row["id"])] = descriptor

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
