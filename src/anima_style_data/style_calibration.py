from __future__ import annotations

import gc
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .anima_cache import (
    _caption_rows,
    _import_sd_scripts,
    _load_llm_adapter,
    _resolve_model_files,
    _verify_sd_scripts_revision,
)
from .io import read_records, write_json
from .style_transfer import _pad_text_conditions, _resolve_anima_model


def _artist_prompt(row: dict[str, Any], artist: str) -> str:
    normalized = " ".join(artist.replace("_", " ").split())
    parts = []
    rating = str(row.get("rating_anima") or "").strip()
    if rating:
        parts.append(rating)
    parts.extend(str(value) for value in row.get("count_tags") or [])
    parts.extend(str(value) for value in row.get("character_tags") or [])
    parts.append(f"@{normalized}")
    parts.extend(str(value) for value in row.get("general_tags") or [])
    return ", ".join(value for value in parts if value)


def _encode_prompts(
    config: dict[str, Any], destination: Path, prompts: list[str], device: str, batch_size: int
) -> torch.Tensor:
    cache_cfg = config["anima_cache"]
    text_cfg = cache_cfg["text"]
    models = _resolve_model_files(cache_cfg["models"], destination)
    sd_root = str(cache_cfg["sd_scripts_path"])
    _verify_sd_scripts_revision(sd_root, str(cache_cfg["sd_scripts_revision"]))
    anima_models, anima_utils, _ = _import_sd_scripts(sd_root)
    dtype = torch.bfloat16
    qwen, qwen_tokenizer = anima_utils.load_qwen3_text_encoder(
        models["qwen3"], dtype=dtype, device=device
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer()
    adapter = _load_llm_adapter(anima_models, models["dit"], device, dtype)
    encoded = []
    max_qwen = int(text_cfg.get("qwen_max_length", 512))
    max_t5 = int(text_cfg.get("t5_max_length", 512))
    with torch.inference_mode():
        for offset in range(0, len(prompts), batch_size):
            values = prompts[offset : offset + batch_size]
            source = qwen_tokenizer(
                values, return_tensors="pt", truncation=True, padding=True,
                max_length=max_qwen,
            )
            target = t5_tokenizer(
                values, return_tensors="pt", truncation=True, padding=True,
                max_length=max_t5,
            )
            source_ids = source["input_ids"].to(device)
            source_mask = source["attention_mask"].to(device)
            target_ids = target["input_ids"].to(device)
            target_mask = target["attention_mask"].to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
                hidden = qwen(input_ids=source_ids, attention_mask=source_mask).last_hidden_state
                hidden[~source_mask.bool()] = 0
                condition = adapter(
                    hidden, target_ids, target_attention_mask=target_mask,
                    source_attention_mask=source_mask,
                )
                condition[~target_mask.bool()] = 0
            lengths = target_mask.sum(1).cpu().tolist()
            encoded.extend(
                tensor[: int(length)].to(device="cpu", dtype=torch.float16).clone()
                for tensor, length in zip(condition, lengths, strict=True)
            )
            print(f"encoded calibration prompts {min(offset + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    del adapter, qwen
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _pad_text_conditions(encoded, int(text_cfg.get("t5_max_length", 512)))


def filter_artist_effects(
    artist_effects: dict[str, float],
    signatures: torch.Tensor,
    artists: list[str],
    *,
    minimum_effect_ratio: float,
    minimum_effect_quantile: float,
    maximum_similarity: float,
) -> tuple[list[int], dict[str, Any]]:
    effects = torch.tensor([artist_effects[name] for name in artists], dtype=torch.float32)
    quantile_floor = float(torch.quantile(effects, minimum_effect_quantile))
    effect_floor = max(float(minimum_effect_ratio), quantile_floor)
    eligible = [index for index, value in enumerate(effects) if float(value) >= effect_floor]
    if not eligible:
        raise RuntimeError("Artist-effect filtering removed every calibration artist")
    centered = signatures.float() - signatures.float().mean(0, keepdim=True)
    centered = F.normalize(centered, dim=1)
    retained: list[int] = []
    rejected_similar: dict[str, str] = {}
    for index in sorted(eligible, key=lambda item: float(effects[item]), reverse=True):
        if retained:
            similarities = centered[index] @ centered[retained].T
            best_value, best_position = similarities.max(0)
            if float(best_value) >= maximum_similarity:
                rejected_similar[artists[index]] = artists[retained[int(best_position)]]
                continue
        retained.append(index)
    return retained, {
        "effect_floor": effect_floor,
        "quantile_floor": quantile_floor,
        "weak_removed": len(artists) - len(eligible),
        "similar_removed": len(rejected_similar),
        "similar_representatives": rejected_similar,
    }


@torch.no_grad()
def calibrate_artist_tag_velocity(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["style_transfer"]["effect_calibration"]
    device = str(cfg.get("device", config["style_transfer"]["training"].get("device", "cuda")))
    seed = int(cfg.get("seed", 20260811 ^ 0xCA11B))
    rng = random.Random(seed)
    caption_rows = _caption_rows(destination)
    counts = Counter(str(row["artist"]) for row in caption_rows)
    candidate_count = min(int(cfg.get("candidate_artists", 128)), len(counts))
    # The production set is intentionally balanced at roughly 30 images per
    # artist, so Counter.most_common would mostly preserve alphabetical shard
    # order. A seeded sample covers the actual 5k-artist population instead.
    candidates = rng.sample(sorted(counts), candidate_count)

    latent_root = destination / str(config["style_transfer"]["loader"]["latent_cache"])
    latent_rows = read_records(latent_root / "manifest.parquet")
    latent_by_id = {int(row["id"]): row for row in latent_rows}
    eligible_rows = [row for row in caption_rows if int(row["id"]) in latent_by_id]
    by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in eligible_rows:
        latent = latent_by_id[int(row["id"])]
        shape = (int(latent["latent_height"]), int(latent["latent_width"]))
        by_shape.setdefault(shape, []).append(row)
    shape, rows = max(by_shape.items(), key=lambda item: len(item[1]))
    rng.shuffle(rows)
    probe_rows = rows[: int(cfg.get("probe_prompts", 4))]
    if len(probe_rows) < int(cfg.get("probe_prompts", 4)):
        raise RuntimeError("Not enough cached latents in the dominant calibration bucket")

    prompts = []
    for row in probe_rows:
        prompts.append(str(row["anima_caption"]))
    for artist in candidates:
        for row in probe_rows:
            prompts.append(_artist_prompt(row, artist))
    conditions = _encode_prompts(
        config, destination, prompts, device, int(cfg.get("text_batch_size", 64))
    )
    prompt_count = len(probe_rows)
    base_conditions = conditions[:prompt_count]
    tagged_conditions = conditions[prompt_count:].reshape(
        len(candidates), prompt_count, conditions.shape[1], conditions.shape[2]
    )

    latents = []
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}
    for row in probe_rows:
        latent_row = latent_by_id[int(row["id"])]
        shard_name = str(latent_row["cache_shard"])
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(latent_root / shard_name, device="cpu")
        latents.append(shard_cache[shard_name]["latents"][int(latent_row["row_index"])])
    latents = torch.stack(latents).to(device=device, dtype=torch.bfloat16)
    del shard_cache, conditions

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    edges = [float(value) for value in cfg.get("timestep_edges", [0, .2, .4, .6, .8, 1])]
    centers = [(left + right) / 2 for left, right in zip(edges[:-1], edges[1:])]
    artist_batch = int(cfg.get("artist_batch_size", 8))
    pool_size = int(cfg.get("signature_pool_size", 8))
    ratio_records: dict[str, list[list[float]]] = {
        artist: [[] for _ in centers] for artist in candidates
    }
    signature_parts: dict[str, list[torch.Tensor]] = {artist: [] for artist in candidates}
    generator = torch.Generator(device=device).manual_seed(seed)

    for prompt_index in range(prompt_count):
        latent = latents[prompt_index : prompt_index + 1]
        noise = torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=generator)
        padding = torch.zeros(1, 1, shape[0], shape[1], device=device, dtype=latent.dtype)
        for bin_index, timestep_value in enumerate(centers):
            timestep = torch.full((1,), timestep_value, device=device, dtype=latent.dtype)
            sigma = timestep[:, None, None, None]
            noisy = (1 - sigma) * latent + sigma * noise
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                base = anima(
                    noisy.unsqueeze(2), timestep,
                    context=base_conditions[prompt_index : prompt_index + 1].to(
                        device=device, dtype=torch.bfloat16
                    ),
                    padding_mask=padding, target_input_ids=None,
                ).squeeze(2).float()
            base_rms = base.square().mean().sqrt().clamp_min(1e-8)
            for offset in range(0, len(candidates), artist_batch):
                names = candidates[offset : offset + artist_batch]
                count = len(names)
                repeated_noisy = noisy.expand(count, -1, -1, -1)
                repeated_timestep = timestep.expand(count)
                context = tagged_conditions[offset : offset + count, prompt_index].to(
                    device=device, dtype=torch.bfloat16
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    tagged = anima(
                        repeated_noisy.unsqueeze(2), repeated_timestep,
                        context=context, padding_mask=padding.expand(count, -1, -1, -1),
                        target_input_ids=None,
                    ).squeeze(2).float()
                delta = tagged - base
                ratios = delta.square().mean(dim=(1, 2, 3)).sqrt() / base_rms
                pooled = F.adaptive_avg_pool2d(delta, (pool_size, pool_size)).flatten(1).cpu()
                for local_index, name in enumerate(names):
                    ratio_records[name][bin_index].append(float(ratios[local_index]))
                    signature_parts[name].append(pooled[local_index])
            print(
                f"artist-tag probe prompt={prompt_index + 1}/{prompt_count} "
                f"timestep={timestep_value:.3f}", flush=True,
            )

    signatures = torch.stack([torch.cat(signature_parts[name]) for name in candidates])
    artist_effects = {
        name: float(torch.tensor([value for values in ratio_records[name] for value in values]).median())
        for name in candidates
    }
    retained, filtering = filter_artist_effects(
        artist_effects, signatures, candidates,
        minimum_effect_ratio=float(cfg.get("minimum_effect_ratio", 0.005)),
        minimum_effect_quantile=float(cfg.get("minimum_effect_quantile", 0.25)),
        maximum_similarity=float(cfg.get("maximum_artist_similarity", 0.95)),
    )
    bins = []
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        values = torch.tensor([
            value for artist_index in retained
            for value in ratio_records[candidates[artist_index]][bin_index]
        ])
        bins.append({
            "left": left, "right": right, "count": int(values.numel()),
            "p25": float(torch.quantile(values, .25)),
            "median": float(torch.quantile(values, .5)),
            "p75": float(torch.quantile(values, .75)),
        })
    result = {
        "version": "anima-artist-tag-velocity-ratio-v1",
        "timestep_edges": edges,
        "bins": bins,
        "candidate_artists": len(candidates),
        "retained_artists": [candidates[index] for index in retained],
        "artist_effect_median": artist_effects,
        "filtering": filtering,
        "probe_ids": [int(row["id"]) for row in probe_rows],
        "latent_shape": list(shape),
        "direction_used_for_training": False,
    }
    output = destination / str(cfg.get("output", "style_effect_calibration.json"))
    write_json(output, result)
    return result
