from __future__ import annotations

import copy
import gc
import hashlib
import math
import random
import re
import time
from collections import Counter
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
    counts = Counter(str(row.get("style_id", row["artist"])) for row in rows)
    eligible = sorted(name for name, count in counts.items() if count >= 2)
    if len(eligible) < artist_count:
        raise RuntimeError(f"Need {artist_count} train artists, found {len(eligible)}")
    rng = random.Random(seed)
    artists = rng.sample(eligible, artist_count)

    # Reuse real, artist-free Anima captions as content controls. Selecting
    # different source styles prevents one artist's subject distribution from
    # becoming the shared synthetic content template.
    content_pool = [row for row in rows if str(row.get("style_id", row["artist"])) not in artists]
    rng.shuffle(content_pool)
    content_rows: list[dict[str, Any]] = []
    used_styles: set[str] = set()
    for want_female, wanted in ((True, female_contents), (False, content_count - female_contents)):
        selected = 0
        for row in content_pool:
            style = str(row.get("style_id", row["artist"]))
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
        artist_slug = f"{artist_index:04d}-{hashlib.sha1(artist.encode()).hexdigest()[:10]}"
        for content in content_rows:
            condition_id = len(prompts)
            tagged_condition_ids[(artist, int(content["content_index"]))] = condition_id
            prompts.append({
                "condition_id": condition_id, "kind": "artist",
                "artist": artist, "content_index": content["content_index"],
                "artist_tag": raw_tag, "comfy_literal_artist_tag": escaped_tag,
                "prompt": f"{content['prompt']}, {raw_tag}",
            })
        for content in content_rows:
            for seed_index, generation_seed in enumerate(seed_values):
                item_index = len(plan)
                image_id = 10_000_000_000 + item_index
                plan.append({
                    "id": image_id, "synthetic_index": item_index, "kind": "artist",
                    "artist_index": artist_index, "artist": artist, "style_id": artist,
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


def _save_webp(path: Path, pixels, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="WEBP", quality=quality, method=4)
    temporary.replace(path)


def _dct_downscale(value, scale: float):
    import numpy as np
    import torch
    from scipy.fft import dctn, idctn

    source = value.detach().float().cpu().numpy()
    height, width = source.shape[-2:]
    target = (round(height * scale), round(width * scale))
    output = np.empty((*source.shape[:-2], *target), dtype=np.float32)
    for index in np.ndindex(*source.shape[:-2]):
        coefficients = dctn(source[index], type=2, norm="ortho")
        output[index] = idctn(
            coefficients[: target[0], : target[1]], type=2, norm="ortho"
        ).astype(np.float32)
    return torch.from_numpy(output).to(device=value.device, dtype=value.dtype)


def _dct_expand(value, target: tuple[int, int], timestep: float, seeds: list[int]):
    import numpy as np
    import torch
    from scipy.fft import dctn, idctn

    source = value.detach().float().cpu().numpy()
    output = np.empty((*source.shape[:-2], *target), dtype=np.float32)
    for batch_index in range(source.shape[0]):
        rng = np.random.default_rng(int(seeds[batch_index]) + 10_000)
        for leading in np.ndindex(*source.shape[1:-2]):
            index = (batch_index, *leading)
            coefficients = dctn(source[index], type=2, norm="ortho")
            expanded = (float(timestep) * rng.standard_normal(target)).astype(np.float32)
            expanded[: source.shape[-2], : source.shape[-1]] = coefficients
            output[index] = idctn(expanded, type=2, norm="ortho").astype(np.float32)
    ratio = target[0] / source.shape[-2]
    kappa = ratio / (1.0 + (ratio - 1.0) * float(timestep))
    return (
        torch.from_numpy(output * kappa).to(device=value.device, dtype=value.dtype),
        float(timestep) * kappa,
    )


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
    import numpy as np
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
    if bool(cfg.get("torch_compile", True)):
        anima = torch.compile(
            anima, mode=str(cfg.get("compile_mode", "reduce-overhead")),
            fullgraph=False, dynamic=False,
        )
    vae = _load_sampling_vae(config, destination).to(device=device, dtype=torch.bfloat16)
    vae.requires_grad_(False).eval()
    text_root = root / "text"
    condition_rows = read_records(text_root / "manifest.parquet")
    condition_index = {int(row["condition_id"]): row for row in condition_rows}
    condition_shards: dict[str, torch.Tensor] = {}
    negative = load_file(text_root / "negative.safetensors", device="cpu")["conditioning"]
    batch_size = int(cfg.get("batch_size", 8))
    width = int(cfg.get("width", 512))
    height = int(cfg.get("height", 512))
    latent_h, latent_w = height // 8, width // 8
    steps = int(cfg.get("steps", 20))
    text_cfg = float(cfg.get("text_cfg", 4.0))
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    shard_rows = int(cfg.get("latent_shard_rows", 256))
    image_workers = int(cfg.get("image_writer_workers", 8))
    webp_quality = int(cfg.get("webp_quality", 95))
    writer = ThreadPoolExecutor(max_workers=image_workers)
    pending = []
    shard_buffer: list[tuple[dict[str, Any], torch.Tensor]] = []
    shard_index = len(list(manifest_dir.glob("part-*.parquet")))
    output_rows = list(completed_rows)
    started = time.monotonic()
    benchmark_path = root / "benchmark.json"
    benchmark_done = benchmark_path.exists()

    def condition(condition_id: int) -> torch.Tensor:
        row = condition_index[condition_id]
        name = str(row["cache_shard"])
        if name not in condition_shards:
            condition_shards[name] = load_file(text_root / name, device="cpu")["conditioning"]
        return condition_shards[name][int(row["row_index"])]

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

    # Grouping by artist condition keeps text shards hot while seed/content are
    # still fully crossed in the deterministic plan.
    work.sort(key=lambda row: (int(row["content_index"]), int(row["artist_index"]), int(row["seed_index"])))
    for offset in range(0, len(work), batch_size):
        batch = work[offset : offset + batch_size]
        positive = torch.stack([condition(int(row["artist_condition_id"])) for row in batch]).to(
            device=device, dtype=torch.bfloat16
        )
        negative_batch = negative.expand(len(batch), -1, -1).to(device=device, dtype=torch.bfloat16)
        noise = torch.stack([
            torch.randn(
                16, 1, latent_h, latent_w,
                generator=torch.Generator(device="cpu").manual_seed(int(row["generation_seed"])),
                dtype=torch.float32,
            )
            for row in batch
        ]).to(device=device, dtype=torch.bfloat16)
        x = noise
        speed_cfg = dict(cfg.get("speed", {}))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = _sample_anima_batch(
                anima, x, positive, negative_batch, sigmas, text_cfg=text_cfg,
                speed=speed_cfg,
                generation_seeds=[int(row["generation_seed"]) for row in batch],
            )
            decoded = vae.decode_to_pixels(x).float()
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
            vae_validation = _validate_batched_vae(vae, speed_x)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                comparison = vae.decode_to_pixels(torch.cat((baseline_x, speed_x))).float()
            if comparison.ndim == 5:
                comparison = comparison[:, :, 0]
            compare_pixels = ((comparison.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
            columns = len(batch)
            sheet = Image.new("RGB", (width * columns, height * 2), "white")
            for index, value in enumerate(compare_pixels):
                sheet.paste(Image.fromarray(value), ((index % columns) * width, (index // columns) * height))
            sheet.save(root / "benchmark-baseline-top-speed-bottom.webp", format="WEBP", quality=95)
            benchmark = {
                "batch_size": len(batch), "baseline_s": baseline_s, "speed_s": speed_s,
                "speedup": baseline_s / max(speed_s, 1e-9),
                "latent_mean_abs_difference": float((baseline_x.float() - speed_x.float()).abs().mean()),
                "baseline_finite": bool(torch.isfinite(baseline_x).all()),
                "speed_finite": bool(torch.isfinite(speed_x).all()),
                "vae": vae_validation,
                "comparison": str((root / "benchmark-baseline-top-speed-bottom.webp").resolve()),
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
        pixels = ((decoded.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        latents = x[:, :, 0].to(device="cpu", dtype=torch.float16)
        for row, latent, image_pixels in zip(batch, latents, pixels, strict=True):
            relative = Path("images") / str(row["artist_slug"]) / f"{int(row['content_index'])}-{int(row['seed_index'])}.webp"
            pending.append(writer.submit(_save_webp, root / relative, image_pixels, webp_quality))
            record = {
                **row, "local_path": str((root / relative).resolve()),
                "width": width, "height": height,
                "latent_height": latent_h, "latent_width": latent_w,
                "steps": steps, "text_cfg": text_cfg, "flow_shift": shift,
                "attention_backend": attn_mode,
            }
            shard_buffer.append((record, latent.contiguous()))
            if len(shard_buffer) >= shard_rows:
                flush()
        if len(pending) >= image_workers * 4:
            pending.pop(0).result()
        done = offset + len(batch)
        if done % max(batch_size * 10, 100) == 0 or done == len(work):
            elapsed = time.monotonic() - started
            print(
                f"synthetic teacher generated {done}/{len(work)} "
                f"({done / max(elapsed, 1e-6):.3f} images/s, batch={len(batch)})",
                flush=True,
            )
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
    result = {"plan": len(plan), "text": text, "generation": generation, "features": features}
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
