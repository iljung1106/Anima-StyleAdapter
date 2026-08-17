from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .anima_cache import _caption_rows, _import_sd_scripts, _resolve_model_files
from .cradio import extract_selected_style_features
from .io import read_records, write_json, write_records
from .style_calibration import _encode_prompts
from .style_transfer import _load_sampling_vae, _resolve_anima_model


def normalize_artist_name(artist: str) -> str:
    """Return the literal Anima artist name used after the required @ prefix."""
    return " ".join(str(artist).replace("_", " ").split())


def artist_tag(artist: str) -> str:
    return f"@{normalize_artist_name(artist)}"


def comfy_literal_artist_tag(artist: str) -> str:
    """Escape only Comfy prompt-weighting delimiters for metadata/UI reuse.

    The production text cache feeds `artist_tag` directly to the tokenizer, so
    parentheses are already literal there. This escaped spelling is retained
    for any later ComfyUI reproduction, where unescaped () alter prompt weight.
    """
    value = artist_tag(artist)
    return re.sub(r"([()\[\]{}])", r"\\\1", value)


def synthetic_artist_split_map(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, str]:
    """Return the deterministic artist split, including for legacy manifests.

    Older heterogeneous Parquet manifests lost fields absent from their first
    control row.  The split is reproducible from the selected artist set and
    configured split seed, so existing expensive image/feature caches do not
    need to be regenerated.
    """

    cfg = dict(config["synthetic_teacher"])
    artists = sorted(
        {
            str(row["artist"])
            for row in rows
            if str(row.get("kind", "artist")) == "artist"
        }
    )
    if not artists:
        raise RuntimeError("Synthetic teacher manifest contains no artist rows")
    split_cfg = dict(cfg.get("bootstrap", {}))
    split_order = list(artists)
    random.Random(int(split_cfg.get("split_seed", cfg.get("seed", 20260812)))).shuffle(
        split_order
    )
    validation_count = int(split_cfg.get("validation_artists", 25))
    meta_test_count = int(split_cfg.get("meta_test_artists", 25))
    if validation_count + meta_test_count >= len(split_order):
        raise ValueError("Synthetic teacher split leaves no training artists")
    meta_test = set(split_order[:meta_test_count])
    validation = set(
        split_order[meta_test_count : meta_test_count + validation_count]
    )
    return {
        artist: (
            "meta_test"
            if artist in meta_test
            else "validation"
            if artist in validation
            else "train"
        )
        for artist in artists
    }


def _content_prompt(row: dict[str, Any]) -> str:
    parts: list[str] = []
    rating = str(row.get("rating_anima") or "safe").strip()
    if rating:
        parts.append(rating)
    parts.extend(str(value) for value in row.get("count_tags") or [])
    parts.extend(str(value) for value in row.get("character_tags") or [])
    parts.extend(str(value) for value in row.get("general_tags") or [])
    return ", ".join(dict.fromkeys(value for value in parts if value))


def build_synthetic_teacher_plan(
    config: dict[str, Any], destination: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config["synthetic_teacher"]
    output = destination / str(cfg.get("output_directory", "synthetic_teacher"))
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 20260812))
    artist_count = int(cfg.get("artist_count", 500))
    content_count = int(cfg.get("contents_per_artist", 8))
    female_contents = int(cfg.get("female_contents", 7))
    seeds_per_content = int(cfg.get("seeds_per_content", 2))
    if not 0 <= female_contents <= content_count:
        raise ValueError("female_contents must be between zero and contents_per_artist")
    rows = [row for row in _caption_rows(destination) if row.get("split", "train") == "train"]
    # `style_id` is an internal cache key such as ``human:foo_(bar)``.  Anima's
    # native artist syntax expects the raw Danbooru artist name, so never feed
    # that namespace prefix to the text encoder.
    counts = Counter(str(row["artist"]) for row in rows)
    style_by_artist: dict[str, str] = {}
    for row in rows:
        artist = str(row["artist"])
        style_id = str(row.get("style_id", artist))
        recorded = style_by_artist.setdefault(artist, style_id)
        if recorded != style_id:
            raise RuntimeError(f"Artist {artist!r} maps to multiple style IDs")
    eligible = sorted(name for name, count in counts.items() if count >= 2)
    if len(eligible) < artist_count:
        raise RuntimeError(f"Need {artist_count} train artists, found {len(eligible)}")
    rng = random.Random(seed)
    artists = rng.sample(eligible, artist_count)
    artist_splits = synthetic_artist_split_map(
        config,
        [{"artist": artist, "kind": "artist"} for artist in artists],
    )

    # Reuse real, artist-free Anima captions as content controls. Selecting
    # different source styles prevents one artist's subject distribution from
    # becoming the shared synthetic content template.
    content_pool = [row for row in rows if str(row["artist"]) not in artists]
    rng.shuffle(content_pool)
    content_rows: list[dict[str, Any]] = []
    used_styles: set[str] = set()
    for want_female, wanted in ((True, female_contents), (False, content_count - female_contents)):
        selected = 0
        for row in content_pool:
            style = str(row["artist"])
            prompt = _content_prompt(row)
            tags = set(str(value) for value in (row.get("count_tags") or []))
            has_1girl = "1girl" in tags or "1girl" in set(str(value) for value in (row.get("general_tags") or []))
            if has_1girl != want_female or style in used_styles or not prompt:
                continue
            used_styles.add(style)
            if want_female and "1girl" not in {part.strip() for part in prompt.split(",")}:
                prompt = f"1girl, {prompt}"
            content_rows.append({
                "content_index": len(content_rows), "source_id": int(row["id"]),
                "prompt": prompt, "contains_1girl": has_1girl,
            })
            selected += 1
            if selected == wanted:
                break
        if selected != wanted:
            raise RuntimeError(f"Could not select {wanted} content prompts for female={want_female}")
    if len(content_rows) != content_count:
        raise RuntimeError("Could not select enough distinct content prompts")

    seed_values = [
        int.from_bytes(
            hashlib.blake2b(f"{seed}:generation:{index}".encode(), digest_size=8).digest(),
            "big",
        )
        % (2**63 - 1)
        for index in range(seeds_per_content)
    ]
    plan: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for content in content_rows:
        prompts.append({
            "condition_id": len(prompts), "kind": "content",
            "artist": None, "content_index": content["content_index"],
            "prompt": content["prompt"],
        })
    tagged_condition_ids: dict[tuple[str, int], int] = {}
    # Shared content-only controls establish the exact teacher baseline for
    # every content/seed pair without redundantly generating 500 identical
    # copies. Artist rows point back to the corresponding control ID.
    control_ids: dict[tuple[int, int], int] = {}
    for content in content_rows:
        for seed_index, generation_seed in enumerate(seed_values):
            image_id = 10_000_000_000 + len(plan)
            control_ids[(int(content["content_index"]), seed_index)] = image_id
            plan.append({
                "id": image_id, "synthetic_index": len(plan), "kind": "content_control",
                "artist_index": -1, "artist": "__content_only__", "style_id": "__content_only__",
                "artist_slug": "content-only", "split": "synthetic_teacher",
                "content_index": int(content["content_index"]),
                "content_source_id": int(content["source_id"]),
                "seed_index": seed_index, "generation_seed": generation_seed,
                "content_condition_id": int(content["content_index"]),
                "artist_condition_id": int(content["content_index"]),
                "artist_tag": "", "comfy_literal_artist_tag": "",
                "content_prompt": content["prompt"], "artist_prompt": content["prompt"],
                "control_id": image_id,
            })
    for artist_index, artist in enumerate(artists):
        raw_tag = artist_tag(artist)
        escaped_tag = comfy_literal_artist_tag(artist)
        style_id = style_by_artist[artist]
        artist_split = artist_splits[artist]
        artist_slug = f"{artist_index:04d}-{hashlib.sha1(artist.encode()).hexdigest()[:10]}"
        for content in content_rows:
            condition_id = len(prompts)
            tagged_condition_ids[(artist, int(content["content_index"]))] = condition_id
            prompts.append({
                "condition_id": condition_id, "kind": "artist",
                "artist": artist, "content_index": content["content_index"],
                "style_id": style_id, "artist_split": artist_split,
                "artist_tag": raw_tag, "comfy_literal_artist_tag": escaped_tag,
                "prompt": f"{content['prompt']}, {raw_tag}",
            })
        for content in content_rows:
            for seed_index, generation_seed in enumerate(seed_values):
                item_index = len(plan)
                image_id = 10_000_000_000 + item_index
                plan.append({
                    "id": image_id, "synthetic_index": item_index, "kind": "artist",
                    "artist_index": artist_index, "artist": artist, "style_id": style_id,
                    "artist_split": artist_split,
                    "artist_slug": artist_slug, "split": "synthetic_teacher",
                    "content_index": int(content["content_index"]),
                    "content_source_id": int(content["source_id"]),
                    "seed_index": seed_index, "generation_seed": generation_seed,
                    "content_condition_id": int(content["content_index"]),
                    "artist_condition_id": tagged_condition_ids[(artist, int(content["content_index"]))],
                    "artist_tag": raw_tag, "comfy_literal_artist_tag": escaped_tag,
                    "content_prompt": content["prompt"],
                    "artist_prompt": f"{content['prompt']}, {raw_tag}",
                    "control_id": control_ids[(int(content["content_index"]), seed_index)],
                })
    write_records(output / "plan.parquet", plan)
    write_records(output / "prompts.parquet", prompts)
    write_json(output / "plan_summary.json", {
        "artists": len(artists), "contents": len(content_rows),
        "female_contents": sum(bool(row["contains_1girl"]) for row in content_rows),
        "seeds_per_content": len(seed_values), "artist_images": len(artists) * len(content_rows) * len(seed_values),
        "content_controls": len(content_rows) * len(seed_values), "images": len(plan),
        "prompts": len(prompts), "seed_values": seed_values,
        "artist_split_counts": dict(Counter(artist_splits.values())),
        "literal_parentheses_are_tokenized_directly": True,
        "comfy_prompts_escape_weighting_delimiters": True,
    })
    return plan, prompts


def cache_synthetic_teacher_text(
    config: dict[str, Any], destination: Path, prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    cfg = config["synthetic_teacher"]
    output = destination / str(cfg.get("output_directory", "synthetic_teacher")) / "text"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.parquet"
    if manifest_path.exists():
        existing = read_records(manifest_path)
        if len(existing) == len(prompts):
            return {"conditions": len(existing), "reused": len(existing)}
    device = str(cfg.get("device", "cuda"))
    negative = str(cfg.get(
        "negative_prompt",
        "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    ))
    # Load Qwen only once: the negative condition is simply the final row of
    # the same batched post-LLM encoding pass.
    encoded = _encode_prompts(
        config, destination,
        [str(row["prompt"]) for row in prompts] + [negative], device,
        int(cfg.get("text_batch_size", 64)),
    )
    values, negative_value = encoded[:-1], encoded[-1:]
    shard_rows = int(cfg.get("text_shard_rows", 128))
    records = []
    for shard, offset in enumerate(range(0, len(prompts), shard_rows)):
        part = prompts[offset : offset + shard_rows]
        path = output / f"part-{shard:05d}.safetensors"
        save_file({"conditioning": values[offset : offset + len(part)].contiguous()}, path)
        for row_index, row in enumerate(part):
            records.append({**row, "cache_shard": path.name, "row_index": row_index})
    write_records(manifest_path, records)

    save_file({"conditioning": negative_value.contiguous()}, output / "negative.safetensors")
    total = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    summary = {"conditions": len(records), "storage_bytes": total, "negative_prompt": negative}
    write_json(output / "summary.json", summary)
    return summary


def cache_synthetic_teacher_kv_basis(
    config: dict[str, Any], destination: Path, prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Cache a lossless compact factorization of native artist K/V targets."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    cfg = config["synthetic_teacher"]
    root = destination / str(cfg.get("output_directory", "synthetic_teacher"))
    output = root / "anima_kv_teacher"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.parquet"
    projection_path = output / "native_cross_attention.safetensors"
    artist_prompts = [row for row in prompts if row["kind"] == "artist"]
    if manifest_path.exists() and projection_path.exists():
        existing = read_records(manifest_path)
        if len(existing) == len(artist_prompts):
            return {
                "conditions": len(existing), "reused": len(existing),
                "projection_bytes": projection_path.stat().st_size,
                "representation": "post_llm_pair_plus_frozen_kvo",
            }

    text_root = root / "text"
    text_rows = read_records(text_root / "manifest.parquet")
    by_id = {int(row["condition_id"]): row for row in text_rows}
    # Compute all sequence lengths in one vectorized reduction per shard.
    # Calling a 512x1024 reduction separately for 40k conditions repeatedly
    # starts the CPU thread pool and can take hours on large hosts.
    rows_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in text_rows:
        rows_by_shard[str(row["cache_shard"])].append(row)
    token_lengths: dict[int, int] = {}
    for shard_index, (name, shard_rows) in enumerate(
        sorted(rows_by_shard.items()), start=1
    ):
        values = load_file(text_root / name, device="cpu")["conditioning"]
        lengths = values.abs().amax(dim=-1).ne(0).sum(dim=-1).tolist()
        for row in shard_rows:
            token_lengths[int(row["condition_id"])] = int(
                lengths[int(row["row_index"])]
            )
        print(
            f"indexed teacher text shard {shard_index}/{len(rows_by_shard)}",
            flush=True,
        )

    base_lengths: dict[int, int] = {}
    for row in prompts:
        if row["kind"] == "content":
            condition_id = int(row["condition_id"])
            base_lengths[condition_id] = token_lengths[condition_id]
    records = []
    for row in artist_prompts:
        artist_id = int(row["condition_id"])
        content_id = int(row["content_index"])
        records.append({
            **row,
            "artist_condition_id": artist_id,
            "content_condition_id": content_id,
            "artist_token_length": token_lengths[artist_id],
            "content_token_length": base_lengths[content_id],
        })
    write_records(manifest_path, records)
    del token_lengths

    models = _resolve_model_files(config["anima_cache"]["models"], destination)
    selected: dict[str, Any] = {}
    wanted = ("k_proj.weight", "v_proj.weight", "output_proj.weight", "k_norm.weight", "v_norm.weight")
    with safe_open(models["dit"], framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for block in range(28):
            prefix = f"net.blocks.{block}.cross_attn."
            for suffix in wanted:
                source = prefix + suffix
                if source in keys:
                    selected[f"block_{block:02d}.{suffix}"] = handle.get_tensor(source).contiguous()
    projection_weights = sum(key.endswith("proj.weight") for key in selected)
    if projection_weights != 28 * 3:
        raise RuntimeError(f"Expected 84 native K/V/O projections, found {projection_weights}")
    save_file(selected, projection_path)
    summary = {
        "conditions": len(records), "content_conditions": len(base_lengths), "blocks": 28,
        "projection_tensors": len(selected), "projection_bytes": projection_path.stat().st_size,
        "representation": "post_llm_pair_plus_frozen_kvo",
        "exact_reconstruction": "project artist and content separately, then apply native per-head normalization",
        "full_kv_materialization_avoided": True,
    }
    write_json(output / "summary.json", summary)
    return summary


def _save_webp(path: Path, pixels, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="WEBP", quality=quality, method=4)
    temporary.replace(path)


_DCT_MATRIX_CACHE: dict[tuple[int, str, int | None], Any] = {}


def _orthonormal_dct_matrix(size: int, device):
    """Return a cached float32 orthonormal DCT-II matrix on ``device``."""
    import torch

    key = (int(size), device.type, device.index)
    cached = _DCT_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    positions = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequencies = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    matrix = torch.cos(math.pi * frequencies * positions[None] / size)
    matrix.mul_(math.sqrt(2.0 / size))
    matrix[0].fill_(1.0 / math.sqrt(size))
    _DCT_MATRIX_CACHE[key] = matrix
    return matrix


def _dct2(value):
    matrix_h = _orthonormal_dct_matrix(value.shape[-2], value.device)
    matrix_w = _orthonormal_dct_matrix(value.shape[-1], value.device)
    source = value.float().flatten(0, -3)
    transformed = matrix_h @ source @ matrix_w.T
    return transformed.unflatten(0, value.shape[:-2])


def _idct2(value):
    matrix_h = _orthonormal_dct_matrix(value.shape[-2], value.device)
    matrix_w = _orthonormal_dct_matrix(value.shape[-1], value.device)
    source = value.float().flatten(0, -3)
    transformed = matrix_h.T @ source @ matrix_w
    return transformed.unflatten(0, value.shape[:-2])


def _dct_downscale(value, scale: float):
    height, width = value.shape[-2:]
    target = (round(height * scale), round(width * scale))
    coefficients = _dct2(value)
    return _idct2(coefficients[..., : target[0], : target[1]]).to(value.dtype)


def _dct_expand(value, target: tuple[int, int], timestep: float, seeds: list[int]):
    import torch

    coefficients = _dct2(value)
    expanded_rows = []
    for batch_index, seed in enumerate(seeds):
        generator = torch.Generator(device=value.device).manual_seed(int(seed) + 10_000)
        expanded = float(timestep) * torch.randn(
            (*value.shape[1:-2], *target),
            device=value.device,
            dtype=torch.float32,
            generator=generator,
        )
        expanded[..., : value.shape[-2], : value.shape[-1]] = coefficients[batch_index]
        expanded_rows.append(expanded)
    output = _idct2(torch.stack(expanded_rows))
    ratio = target[0] / value.shape[-2]
    kappa = ratio / (1.0 + (ratio - 1.0) * float(timestep))
    return (output.mul_(kappa).to(value.dtype), float(timestep) * kappa)


def _sample_anima_batch(
    anima,
    noise,
    positive,
    negative,
    sigmas,
    *,
    text_cfg: float,
    speed: dict[str, Any] | None,
    generation_seeds: list[int],
):
    import torch

    full_shape = tuple(noise.shape[-2:])

    def denoise(x, schedule):
        height, width = x.shape[-2:]
        padding = torch.zeros(
            x.shape[0], 1, height, width, device=x.device, dtype=x.dtype
        )
        for first, second in zip(schedule[:-1], schedule[1:], strict=True):
            timestep = first.to(torch.bfloat16).expand(x.shape[0])
            model_input = torch.cat((x, x), dim=0)
            context = torch.cat((negative, positive), dim=0)
            velocity = anima(
                model_input,
                torch.cat((timestep, timestep)),
                context=context,
                padding_mask=torch.cat((padding, padding)),
                target_input_ids=None,
            ).float()
            uncond, cond = velocity.chunk(2)
            guided = uncond + text_cfg * (cond - uncond)
            x = (x.float() + guided * (second - first)).to(torch.bfloat16)
        return x

    if not speed or not bool(speed.get("enabled", False)):
        return denoise(noise, sigmas)
    scales = [float(value) for value in speed.get("scales", [0.5, 1.0])]
    thresholds = [float(value) for value in speed.get("manual_sigmas", [0.85])]
    if scales != [0.5, 1.0] or len(thresholds) != 1:
        raise ValueError("The production SPEED path currently validates scales=[0.5,1.0]")
    transition = next(
        (index for index in range(len(sigmas) - 1) if float(sigmas[index]) <= thresholds[0]),
        len(sigmas) - 1,
    )
    if transition <= 0 or transition >= len(sigmas) - 1:
        raise ValueError("SPEED transition must leave at least one step at each resolution")
    x = _dct_downscale(noise, scales[0])
    x = denoise(x, sigmas[: transition + 1])
    x, aligned = _dct_expand(x, full_shape, float(sigmas[transition]), generation_seeds)
    remaining = sigmas[transition:].clone()
    remaining[0] = aligned
    return denoise(x, remaining)


def _validate_batched_vae(vae, latents) -> dict[str, float]:
    import torch

    sample = latents
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        serial = torch.cat([vae.decode_to_pixels(value[None]).float() for value in sample])
    serial_s = time.monotonic() - started
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        batched = vae.decode_to_pixels(sample).float()
    batch_s = time.monotonic() - started
    difference = (serial - batched).abs()
    if not torch.isfinite(batched).all():
        raise FloatingPointError("Batched VAE decode produced non-finite pixels")
    return {
        "serial_s": serial_s,
        "batch_s": batch_s,
        "speedup": serial_s / max(batch_s, 1e-9),
        "mean_abs_difference": float(difference.mean()),
        "max_abs_difference": float(difference.max()),
    }


def generate_synthetic_teacher_images(
    config: dict[str, Any], destination: Path, plan: list[dict[str, Any]], *, benchmark_only: bool = False
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    cfg = config["synthetic_teacher"]
    root = destination / str(cfg.get("output_directory", "synthetic_teacher"))
    image_root = root / "images"
    latent_root = root / "latents"
    manifest_dir = root / "manifests"
    image_root.mkdir(parents=True, exist_ok=True)
    latent_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = []
    for path in sorted(manifest_dir.glob("part-*.parquet")):
        completed_rows.extend(read_records(path))
    completed = {int(row["id"]) for row in completed_rows}
    work = [row for row in plan if int(row["id"]) not in completed]
    if not work:
        return {"images": len(completed), "newly_generated": 0}

    device = str(cfg.get("device", "cuda"))
    attn_mode = "sageattn" if bool(cfg.get("sage_attention", True)) else "torch"
    anima = _resolve_anima_model(config, destination, device, attn_mode=attn_mode).requires_grad_(False).eval()
    vae = _load_sampling_vae(config, destination).to(device=device, dtype=torch.bfloat16)
    vae.requires_grad_(False).eval()
    text_root = root / "text"
    condition_rows = read_records(text_root / "manifest.parquet")
    condition_index = {int(row["condition_id"]): row for row in condition_rows}
    condition_shards: dict[str, torch.Tensor] = {}
    negative = load_file(text_root / "negative.safetensors", device="cpu")["conditioning"]
    width = int(cfg.get("width", 512))
    height = int(cfg.get("height", 512))
    latent_h, latent_w = height // 8, width // 8
    steps = int(cfg.get("steps", 20))
    text_cfg = float(cfg.get("text_cfg", 4.0))
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    speed_cfg = dict(cfg.get("speed", {}))
    shard_rows = int(cfg.get("latent_shard_rows", 256))
    image_workers = int(cfg.get("image_writer_workers", 8))
    webp_quality = int(cfg.get("webp_quality", 95))
    writer = ThreadPoolExecutor(max_workers=image_workers)
    pending = []
    shard_buffer: list[tuple[dict[str, Any], torch.Tensor]] = []
    shard_index = len(list(manifest_dir.glob("part-*.parquet")))
    output_rows = list(completed_rows)
    started = time.monotonic()
    benchmark_path = root / "benchmark-gpu-dct-v1.json"
    benchmark_done = benchmark_path.exists()

    def condition_cpu(condition_id: int) -> torch.Tensor:
        row = condition_index[condition_id]
        name = str(row["cache_shard"])
        if name not in condition_shards:
            condition_shards[name] = load_file(text_root / name, device="cpu")["conditioning"]
        return condition_shards[name][int(row["row_index"])]

    # The complete 500-artist text bank is about 4 GiB.  Keeping it on an
    # 80-GiB H100 avoids a pageable CPU->GPU copy at every denoising batch.
    condition_bank = None
    if bool(cfg.get("gpu_resident_text", True)) and device.startswith("cuda"):
        condition_ids = sorted(condition_index)
        if condition_ids != list(range(len(condition_ids))):
            raise RuntimeError("Synthetic condition IDs must be dense for GPU residency")
        condition_bank = torch.empty(
            len(condition_ids), *negative.shape[1:],
            device=device, dtype=torch.bfloat16,
        )
        rows_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            rows_by_shard[str(row["cache_shard"])].append(row)
        for shard_name, text_shard_rows in sorted(rows_by_shard.items()):
            source = load_file(
                text_root / shard_name, device=device
            )["conditioning"]
            source_rows = torch.tensor(
                [int(row["row_index"]) for row in text_shard_rows],
                device=device, dtype=torch.long,
            )
            destination_rows = torch.tensor(
                [int(row["condition_id"]) for row in text_shard_rows],
                device=device, dtype=torch.long,
            )
            condition_bank.index_copy_(
                0,
                destination_rows,
                source.index_select(0, source_rows).to(dtype=torch.bfloat16),
            )
        negative = negative.to(device=device, dtype=torch.bfloat16)
        condition_shards.clear()
        print(
            f"resident synthetic text bank {len(condition_ids)} conditions "
            f"({condition_bank.numel() * condition_bank.element_size() / 2**30:.2f} GiB)",
            flush=True,
        )

    def conditions_for(rows: list[dict[str, Any]]) -> torch.Tensor:
        ids = [int(row["artist_condition_id"]) for row in rows]
        if condition_bank is not None:
            return condition_bank.index_select(
                0, torch.tensor(ids, device=device, dtype=torch.long)
            )
        return torch.stack([condition_cpu(value) for value in ids]).to(
            device=device, dtype=torch.bfloat16
        )

    def noise_for(rows: list[dict[str, Any]]) -> torch.Tensor:
        return torch.stack([
            torch.randn(
                16, 1, latent_h, latent_w,
                generator=torch.Generator(device=device).manual_seed(
                    int(row["generation_seed"])
                ),
                device=device,
                dtype=torch.bfloat16,
            )
            for row in rows
        ])

    work.sort(
        key=lambda row: (
            int(row["content_index"]), int(row["artist_index"]), int(row["seed_index"])
        )
    )

    batch_size = int(cfg.get("batch_size", 8))
    autotune_cfg = dict(cfg.get("batch_autotune", {}))
    autotune_path = root / "batch_autotune.json"
    if bool(autotune_cfg.get("enabled", False)) and device.startswith("cuda"):
        candidates = sorted({
            int(value) for value in autotune_cfg.get("candidates", [batch_size])
            if 0 < int(value) <= len(work)
        })
        signature = {
            "version": "synthetic-batch-autotune-gpu-dct-v1",
            "candidates": candidates,
            "width": width,
            "height": height,
            "steps": steps,
            "speed": speed_cfg,
            "attention_backend": attn_mode,
        }
        cached_tune = None
        if autotune_path.exists():
            candidate_cache = json.loads(autotune_path.read_text(encoding="utf-8"))
            if candidate_cache.get("signature") == signature:
                cached_tune = candidate_cache
        if cached_tune is None:
            results = []
            maximum_fraction = float(
                autotune_cfg.get("maximum_vram_fraction", 0.90)
            )
            total_memory = torch.cuda.get_device_properties(device).total_memory
            warmups = max(0, int(autotune_cfg.get("warmup_runs", 1)))
            timed_runs = max(1, int(autotune_cfg.get("timed_runs", 1)))
            for candidate in candidates:
                probe_rows = work[:candidate]
                positive_probe = negative_probe = noise_probe = None
                sampled = decoded_probe = warm = None
                try:
                    positive_probe = conditions_for(probe_rows)
                    negative_probe = negative.expand(candidate, -1, -1).to(
                        device=device, dtype=torch.bfloat16
                    )
                    noise_probe = noise_for(probe_rows)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        for _ in range(warmups):
                            warm = _sample_anima_batch(
                                anima, noise_probe, positive_probe, negative_probe,
                                sigmas, text_cfg=text_cfg, speed=speed_cfg,
                                generation_seeds=[
                                    int(row["generation_seed"]) for row in probe_rows
                                ],
                            )
                            vae.decode_to_pixels(warm)
                    warm = None
                    torch.cuda.synchronize()
                    started_tune = time.perf_counter()
                    finite = True
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        for _ in range(timed_runs):
                            sampled = _sample_anima_batch(
                                anima, noise_probe, positive_probe, negative_probe,
                                sigmas, text_cfg=text_cfg, speed=speed_cfg,
                                generation_seeds=[
                                    int(row["generation_seed"]) for row in probe_rows
                                ],
                            )
                            decoded_probe = vae.decode_to_pixels(sampled)
                            finite = finite and bool(torch.isfinite(decoded_probe).all())
                    torch.cuda.synchronize()
                    elapsed_tune = time.perf_counter() - started_tune
                    peak = int(torch.cuda.max_memory_allocated(device))
                    result = {
                        "batch_size": candidate,
                        "images_s": candidate * timed_runs / max(elapsed_tune, 1e-9),
                        "elapsed_s": elapsed_tune,
                        "peak_vram_bytes": peak,
                        "peak_vram_fraction": peak / total_memory,
                        "finite": finite,
                        "eligible": finite and peak / total_memory <= maximum_fraction,
                    }
                    results.append(result)
                    print(f"synthetic batch autotune {result}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    results.append({
                        "batch_size": candidate, "eligible": False, "oom": True
                    })
                    print(f"synthetic batch autotune batch={candidate} OOM", flush=True)
                finally:
                    positive_probe = negative_probe = noise_probe = None
                    sampled = decoded_probe = warm = None
                    gc.collect()
                    torch.cuda.empty_cache()
            eligible = [row for row in results if row.get("eligible")]
            if not eligible:
                raise RuntimeError("Every synthetic batch-size candidate failed")
            selected = max(eligible, key=lambda row: float(row["images_s"]))
            cached_tune = {
                "signature": signature,
                "selected_batch_size": int(selected["batch_size"]),
                "results": results,
            }
            write_json(autotune_path, cached_tune)
        batch_size = int(cached_tune["selected_batch_size"])
        print(f"selected synthetic batch size {batch_size}", flush=True)

    if bool(cfg.get("torch_compile", True)):
        anima = torch.compile(
            anima, mode=str(cfg.get("compile_mode", "reduce-overhead")),
            fullgraph=False, dynamic=False,
        )

    transfer_stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None
    pixel_buffers = [
        torch.empty(
            batch_size, height, width, 3,
            dtype=torch.uint8, device="cpu", pin_memory=device.startswith("cuda"),
        )
        for _ in range(2)
    ]
    latent_buffers = [
        torch.empty(
            batch_size, 16, latent_h, latent_w,
            dtype=torch.float16, device="cpu", pin_memory=device.startswith("cuda"),
        )
        for _ in range(2)
    ]
    transfers: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal shard_buffer, shard_index
        if not shard_buffer:
            return
        path = latent_root / f"part-{shard_index:05d}.safetensors"
        records = []
        save_file({
            "latents": torch.stack([value for _, value in shard_buffer]),
            "ids": torch.tensor([int(row["id"]) for row, _ in shard_buffer], dtype=torch.int64),
        }, path)
        for index, (row, _) in enumerate(shard_buffer):
            records.append({**row, "latent_shard": path.name, "latent_row": index})
        write_records(manifest_dir / f"part-{shard_index:05d}.parquet", records)
        output_rows.extend(records)
        shard_buffer = []
        shard_index += 1

    def consume_transfer(item: dict[str, Any]) -> None:
        item["event"].synchronize()
        count = len(item["rows"])
        pixels_cpu = pixel_buffers[item["buffer"]][:count].numpy()
        latents_cpu = latent_buffers[item["buffer"]][:count]
        for row, latent, image_pixels in zip(
            item["rows"], latents_cpu, pixels_cpu, strict=True
        ):
            relative = (
                Path("images") / str(row["artist_slug"])
                / f"{int(row['content_index'])}-{int(row['seed_index'])}.webp"
            )
            # The writer outlives this reusable pinned buffer, so give it a
            # compact owned array while the much larger D2H copy stays async.
            pending.append(
                writer.submit(
                    _save_webp, root / relative, image_pixels.copy(), webp_quality
                )
            )
            record = {
                **row, "local_path": str((root / relative).resolve()),
                "width": width, "height": height,
                "latent_height": latent_h, "latent_width": latent_w,
                "steps": steps, "text_cfg": text_cfg, "flow_shift": shift,
                "attention_backend": attn_mode,
            }
            shard_buffer.append((record, latent.clone().contiguous()))
            if len(shard_buffer) >= shard_rows:
                flush()
        if len(pending) >= image_workers * 4:
            pending.pop(0).result()

    for batch_number, offset in enumerate(range(0, len(work), batch_size)):
        if len(transfers) >= 2:
            consume_transfer(transfers.pop(0))
        batch = work[offset : offset + batch_size]
        positive = conditions_for(batch)
        negative_batch = negative.expand(len(batch), -1, -1).to(device=device, dtype=torch.bfloat16)
        noise = noise_for(batch)
        x = noise
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = _sample_anima_batch(
                anima, x, positive, negative_batch, sigmas, text_cfg=text_cfg,
                speed=speed_cfg,
                generation_seeds=[int(row["generation_seed"]) for row in batch],
            )
            decoded = vae.decode_to_pixels(x)
        if not benchmark_done:
            # Benchmark on the first real production batch. The baseline uses
            # identical prompts/noise and is saved beside SPEED for visual QA.
            torch.cuda.synchronize()
            benchmark_started = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                baseline_x = _sample_anima_batch(
                    anima, noise, positive, negative_batch, sigmas, text_cfg=text_cfg,
                    speed=None,
                    generation_seeds=[int(row["generation_seed"]) for row in batch],
                )
            torch.cuda.synchronize()
            baseline_s = time.monotonic() - benchmark_started
            torch.cuda.synchronize()
            speed_started = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                speed_x = _sample_anima_batch(
                    anima, noise, positive, negative_batch, sigmas, text_cfg=text_cfg,
                    speed=speed_cfg,
                    generation_seeds=[int(row["generation_seed"]) for row in batch],
                )
            torch.cuda.synchronize()
            speed_s = time.monotonic() - speed_started
            comparison_rows = min(8, len(batch))
            vae_validation = _validate_batched_vae(vae, speed_x[:comparison_rows])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                comparison = vae.decode_to_pixels(torch.cat((
                    baseline_x[:comparison_rows], speed_x[:comparison_rows]
                ))).float()
            if comparison.ndim == 5:
                comparison = comparison[:, :, 0]
            compare_pixels = ((comparison.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
            columns = comparison_rows
            sheet = Image.new("RGB", (width * columns, height * 2), "white")
            for index, value in enumerate(compare_pixels):
                sheet.paste(Image.fromarray(value), ((index % columns) * width, (index // columns) * height))
            comparison_path = root / "benchmark-gpu-dct-baseline-top-speed-bottom.webp"
            sheet.save(comparison_path, format="WEBP", quality=95)
            benchmark = {
                "batch_size": len(batch), "baseline_s": baseline_s, "speed_s": speed_s,
                "speedup": baseline_s / max(speed_s, 1e-9),
                "latent_mean_abs_difference": float((baseline_x.float() - speed_x.float()).abs().mean()),
                "baseline_finite": bool(torch.isfinite(baseline_x).all()),
                "speed_finite": bool(torch.isfinite(speed_x).all()),
                "vae": vae_validation,
                "comparison": str(comparison_path.resolve()),
            }
            write_json(benchmark_path, benchmark)
            print(f"synthetic benchmark {benchmark}", flush=True)
            benchmark_done = True
            if benchmark_only:
                writer.shutdown(wait=True)
                del anima, vae
                gc.collect()
                torch.cuda.empty_cache()
                return {"benchmark_only": True, **benchmark}
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
        pixels_gpu = (
            (decoded.clamp(-1, 1) + 1).mul_(127.5).to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous()
        )
        latents_gpu = x[:, :, 0].to(dtype=torch.float16).contiguous()
        buffer_index = batch_number % 2
        assert transfer_stream is not None
        transfer_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(transfer_stream):
            pixel_buffers[buffer_index][: len(batch)].copy_(
                pixels_gpu, non_blocking=True
            )
            latent_buffers[buffer_index][: len(batch)].copy_(
                latents_gpu, non_blocking=True
            )
            event = torch.cuda.Event()
            event.record(transfer_stream)
        transfers.append({
            "event": event,
            "buffer": buffer_index,
            "rows": batch,
            "gpu_sources": (pixels_gpu, latents_gpu, decoded, x),
        })
        done = offset + len(batch)
        if done % max(batch_size * 10, 100) == 0 or done == len(work):
            elapsed = time.monotonic() - started
            print(
                f"synthetic teacher generated {done}/{len(work)} "
                f"({done / max(elapsed, 1e-6):.3f} images/s, batch={len(batch)})",
                flush=True,
            )
    for transfer in transfers:
        consume_transfer(transfer)
    flush()
    for future in pending:
        future.result()
    writer.shutdown(wait=True)
    output_rows.sort(key=lambda row: int(row["id"]))
    write_records(root / "manifest.parquet", output_rows)
    summary = {
        "images": len(output_rows), "newly_generated": len(work),
        "elapsed_s": time.monotonic() - started,
        "throughput_images_s": len(work) / max(time.monotonic() - started, 1e-6),
        "batch_size": batch_size, "vae_batch_size": batch_size,
        "steps": steps, "attention_backend": attn_mode,
        "torch_compile": bool(cfg.get("torch_compile", True)),
    }
    write_json(root / "generation_summary.json", summary)
    del anima, vae
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def build_synthetic_teacher_cache(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    plan, prompts = build_synthetic_teacher_plan(config, destination)
    text = cache_synthetic_teacher_text(config, destination, prompts)
    kv_teacher = cache_synthetic_teacher_kv_basis(config, destination, prompts)
    generation = generate_synthetic_teacher_images(config, destination, plan)
    cfg = config["synthetic_teacher"]
    root = destination / str(cfg.get("output_directory", "synthetic_teacher"))
    feature_config = copy.deepcopy(config)
    feature_config["style_features"].update({
        "output_directory": str(Path(cfg.get("output_directory", "synthetic_teacher")) / "style_features"),
        "manifest_path": str(root / "manifest.parquet"),
        "model_cache_directory": str(destination / "cradio_model_cache"),
    })
    features = extract_selected_style_features(feature_config, destination)
    result = {"plan": len(plan), "text": text, "kv_teacher": kv_teacher, "generation": generation, "features": features}
    write_json(root / "summary.json", result)
    return result


def benchmark_synthetic_teacher_cache(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Build reusable text inputs, then stop after objective and visual QA artifacts."""
    plan, prompts = build_synthetic_teacher_plan(config, destination)
    text = cache_synthetic_teacher_text(config, destination, prompts)
    benchmark = generate_synthetic_teacher_images(
        config, destination, plan, benchmark_only=True
    )
    return {"plan": len(plan), "text": text, "benchmark": benchmark}


def build_synthetic_teacher_kv_cache(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    _, prompts = build_synthetic_teacher_plan(config, destination)
    text = cache_synthetic_teacher_text(config, destination, prompts)
    teacher = cache_synthetic_teacher_kv_basis(config, destination, prompts)
    return {"text": text, "kv_teacher": teacher}


def build_real_artist_teacher_plan(
    config: dict[str, Any], destination: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a 5k-artist offline teacher plan without generating new images."""
    cfg = config["real_artist_teacher"]
    output = destination / str(cfg.get("output_directory", "real_artist_teacher_5000"))
    output.mkdir(parents=True, exist_ok=True)
    feature_manifest = destination / str(
        cfg.get("feature_manifest", "style_features/manifest.parquet")
    )
    feature_rows = read_records(feature_manifest)
    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_artist[str(row["artist"])].append(row)
    artist_count = int(cfg.get("artist_count", 5000))
    if len(by_artist) < artist_count:
        raise RuntimeError(f"Need {artist_count} real artists, found {len(by_artist)}")
    seed = int(cfg.get("seed", 20260813))
    rng = random.Random(seed)
    artists = sorted(by_artist)
    rng.shuffle(artists)
    artists = artists[:artist_count]
    validation_count = int(cfg.get("validation_artists", 250))
    meta_count = int(cfg.get("meta_test_artists", 250))
    artist_splits = {
        artist: (
            "validation" if index < validation_count else
            "meta_test" if index < validation_count + meta_count else "train"
        )
        for index, artist in enumerate(artists)
    }

    synthetic_root = destination / str(
        cfg.get("content_source_directory", "synthetic_teacher_500x16")
    )
    source_prompts = read_records(synthetic_root / "prompts.parquet")
    contents = sorted(
        (row for row in source_prompts if row["kind"] == "content"),
        key=lambda row: int(row["content_index"]),
    )
    if len(contents) != 8:
        raise RuntimeError(f"Expected eight shared content prompts, found {len(contents)}")
    prompts = [
        {
            "condition_id": int(row["content_index"]), "kind": "content",
            "artist": None, "content_index": int(row["content_index"]),
            "prompt": str(row["prompt"]),
        }
        for row in contents
    ]
    plan = []
    for artist_index, artist in enumerate(artists):
        raw_tag = artist_tag(artist)
        escaped_tag = comfy_literal_artist_tag(artist)
        for content in contents:
            condition_id = len(prompts)
            content_index = int(content["content_index"])
            prompts.append({
                "condition_id": condition_id, "kind": "artist",
                "artist": artist, "artist_index": artist_index,
                "artist_split": artist_splits[artist],
                "content_index": content_index,
                "artist_tag": raw_tag, "comfy_literal_artist_tag": escaped_tag,
                "prompt": f"{content['prompt']}, {raw_tag}",
            })
            plan.append({
                "artist": artist, "artist_index": artist_index,
                "artist_split": artist_splits[artist],
                "content_index": content_index,
                "content_split": (
                    "validation" if content_index == 6 else
                    "test" if content_index == 7 else "train"
                ),
                "content_condition_id": content_index,
                "artist_condition_id": condition_id,
                "artist_tag": raw_tag,
            })

    reference_counts = dict(cfg.get(
        "reference_images_per_split", {"train": 4, "validation": 1, "test": 1}
    ))
    references = []
    for artist in artists:
        rows = sorted(
            by_artist[artist],
            key=lambda row: hashlib.blake2b(
                f"{seed}:{row['id']}".encode(), digest_size=8
            ).digest(),
        )
        # The source split is global and can leave an artist without any
        # validation image. Build an explicit within-artist 80/10/10 split so
        # both artist-disjoint and image-disjoint evaluation are guaranteed.
        heldout = max(1, round(len(rows) * 0.1))
        if len(rows) - 2 * heldout < int(reference_counts.get("train", 1)):
            raise RuntimeError(f"Artist {artist!r} has too few images: {len(rows)}")
        internal = {
            "test": rows[:heldout],
            "validation": rows[heldout : heldout * 2],
            "train": rows[heldout * 2 :],
        }
        for reference_split, count in reference_counts.items():
            candidates = internal[reference_split]
            if len(candidates) < int(count):
                raise RuntimeError(
                    f"Artist {artist!r} has {len(candidates)} {reference_split} references; "
                    f"need {count}"
                )
            references.extend({
                **row,
                "source_split": str(row.get("split")),
                "artist_split": artist_splits[artist],
                "reference_split": reference_split,
            } for row in candidates[:int(count)])
    write_records(output / "plan.parquet", plan)
    write_records(output / "prompts.parquet", prompts)
    write_records(output / "references.parquet", references)
    summary = {
        "artists": len(artists), "conditions": len(prompts),
        "artist_conditions": len(plan), "references": len(references),
        "artist_splits": dict(Counter(artist_splits.values())),
        "reference_splits": dict(Counter(str(row["reference_split"]) for row in references)),
        "heldout_contents": [6, 7], "images_generated": 0,
    }
    write_json(output / "plan_summary.json", summary)
    return plan, prompts


def build_real_artist_teacher_kv_cache(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Encode 5k real artists against shared probes and cache native K/V basis."""
    plan, prompts = build_real_artist_teacher_plan(config, destination)
    copied = copy.deepcopy(config)
    real_cfg = config["real_artist_teacher"]
    copied["synthetic_teacher"].update({
        "output_directory": str(real_cfg.get("output_directory", "real_artist_teacher_5000")),
        "text_batch_size": int(real_cfg.get("text_batch_size", 128)),
        "text_shard_rows": int(real_cfg.get("text_shard_rows", 128)),
    })
    text = cache_synthetic_teacher_text(copied, destination, prompts)
    teacher = cache_synthetic_teacher_kv_basis(copied, destination, prompts)
    result = {"plan": len(plan), "text": text, "kv_teacher": teacher}
    write_json(
        destination / str(real_cfg.get("output_directory", "real_artist_teacher_5000"))
        / "summary.json",
        result,
    )
    return result
