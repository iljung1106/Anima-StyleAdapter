from __future__ import annotations

import argparse
import copy
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.torch import load_file

from .compare_resampler_representations import (
    _OnDemandLegacyTokenLoader,
    _episode_signature,
    _load_ab_tokenizer,
    _loader_config,
)
from .compare_style_tokenizers import _TokenView
from .config import load_config, output_dir
from .external_style_tokenizer_sheet import _decode_latents, _denoise_batch
from .query_style_tokenizer import _select_sample_episodes
from .single_stage_typed_attention_style_tokenizer import (
    SingleStageTypedAttentionStyleTokenizer,
)
from .style_tokenizer import (
    AnimaStyleTokenizer,
    _reference_tokens,
    insert_style_tokens,
)
from .style_transfer import (
    ProductionStyleLoader,
    _optimize_frozen_anima,
    _pad_text_conditions,
    _resolve_anima_model,
    load_per_reference_resampler,
)


@dataclass
class SampleCase:
    key: str
    split: str
    artist_index: int
    style_id: str
    target_id: int
    reference_ids: tuple[int, ...]
    target_path: Path
    reference_paths: list[Path]
    conditioning: torch.Tensor
    conditioning_length: int
    small_tokens: torch.Tensor
    typed_tokens: torch.Tensor
    noise_seed: int


def _load_typed_tokenizer(
    checkpoint: Path, device: str
) -> tuple[_TokenView, dict[str, Any]]:
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    model_cfg = dict(state["config"]["model"])
    model_cfg.pop("architecture", None)
    # The shared trainer injected this old ON/OFF-ablation field into saved
    # configs. Typed attention always consumes its four summary tokens, so the
    # constructor intentionally has no such argument.
    model_cfg.pop("include_artist_summary", None)
    tokenizer = SingleStageTypedAttentionStyleTokenizer(**model_cfg)
    tokenizer.load_state_dict(state["tokenizer"], strict=True)
    tokenizer.to(device).eval()
    metadata = {
        "path": str(checkpoint),
        "step": int(state["step"]),
        "trainable_parameters": sum(
            parameter.numel() for parameter in tokenizer.parameters()
        ),
        "output_tokens": int(tokenizer.output_tokens),
    }
    del state
    return _TokenView(tokenizer), metadata


def _slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-_")
    return value[:80] or "artist"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _label(image: Image.Image, text: str, *, right: bool = False) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(24)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 20
    height = box[3] - box[1] + 16
    left = image.width - width - 8 if right else 8
    draw.rounded_rectangle(
        (left, 8, left + width, 8 + height), radius=7, fill=(0, 0, 0, 180)
    )
    draw.text((left + 10, 15), text, font=font, fill="white")


def _source_tile(path: Path, label: str, size: int = 160) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
    _label(image, label)
    return image


def _make_case_sheet(
    case: SampleCase,
    base: Image.Image,
    small: list[Image.Image],
    typed: list[Image.Image],
    multipliers: list[float],
) -> Image.Image:
    width, height = base.size
    columns = 1 + len(multipliers)
    source_height = 188
    sheet = Image.new(
        "RGB", (columns * width, 2 * height + source_height), "white"
    )
    rows = (("SMALL / STEP 7750", small), ("TYPED / STEP 4000", typed))
    for row_index, (row_name, images) in enumerate(rows):
        baseline = base.copy()
        _label(baseline, f"{row_name} | NO STYLE")
        sheet.paste(baseline, (0, row_index * height))
        for column, (multiplier, image) in enumerate(
            zip(multipliers, images, strict=True), start=1
        ):
            cell = image.copy()
            _label(cell, f"{row_name} | STYLE {multiplier:g}x")
            sheet.paste(cell, (column * width, row_index * height))

    draw = ImageDraw.Draw(sheet)
    font = _font(22)
    source_y = 2 * height + 18
    title = (
        f"{case.split} | {case.style_id} | target={case.target_id} | "
        f"references={len(case.reference_ids)} | seed={case.noise_seed}"
    )
    draw.text((8, 2 * height + 2), title, font=font, fill="black")
    sources = [(case.target_path, "TARGET")] + [
        (path, f"REF {index + 1}")
        for index, path in enumerate(case.reference_paths)
    ]
    for index, (path, label) in enumerate(sources):
        tile = _source_tile(path, label)
        sheet.paste(tile, (index * 160, source_y))
    return sheet


def _paired_loaders(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    *,
    split: str,
    references: int,
    legacy_resampler: torch.nn.Module,
    device: str,
) -> tuple[_OnDemandLegacyTokenLoader, ProductionStyleLoader]:
    source_cfg = dict(config[str(cfg["source_config_section"])])
    source_cfg["seed"] = int(cfg["seed"])
    loader_overrides = dict(source_cfg.get("loader", {}))
    loader_overrides.update(dict(cfg.get("loader", {})))
    source_cfg["loader"] = loader_overrides
    small_base = ProductionStyleLoader(
        destination,
        _loader_config(
            config,
            source_cfg,
            token_cache=None,
            references=references,
            split=split,
        ),
    )
    typed = ProductionStyleLoader(
        destination,
        _loader_config(
            config,
            source_cfg,
            token_cache=str(cfg["typed_token_cache"]),
            references=references,
            split=split,
        ),
    )
    return (
        _OnDemandLegacyTokenLoader(small_base, legacy_resampler, device),
        typed,
    )


def _tokenize_cases(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    device: str,
) -> tuple[list[SampleCase], dict[str, Any], dict[str, Any], torch.Tensor]:
    small, small_metadata = _load_ab_tokenizer(
        destination / str(cfg["small_checkpoint"]),
        AnimaStyleTokenizer,
        device,
    )
    typed, typed_metadata = _load_typed_tokenizer(
        destination / str(cfg["typed_checkpoint"]),
        device,
    )
    legacy_resampler = load_per_reference_resampler(
        destination,
        dict(config["style_transfer"]["resampler"]),
        device,
        trainable=False,
    )
    selected_indices: dict[str, list[int]] = {}
    artist_targets: dict[tuple[str, int], tuple[int, torch.Tensor, int]] = {}
    null_conditioning: torch.Tensor | None = None
    cases: list[SampleCase] = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for split_index, split in enumerate(cfg["splits"]):
            split = str(split)
            for reference_count in [int(value) for value in cfg["reference_counts"]]:
                small_loader, typed_loader = _paired_loaders(
                    config,
                    destination,
                    cfg,
                    split=split,
                    references=reference_count,
                    legacy_resampler=legacy_resampler,
                    device=device,
                )
                if split not in selected_indices:
                    selected_indices[split] = _select_sample_episodes(
                        typed_loader, int(cfg["artists_per_split"])
                    )
                if null_conditioning is None:
                    raw = load_file(
                        typed_loader.text_root / "null_conditioning.safetensors",
                        device="cpu",
                    )["empty_prompt"]
                    null_raw = raw[0] if raw.ndim == 3 else raw
                    null_conditioning = _pad_text_conditions(
                        [null_raw], typed_loader.text_conditioning_length
                    )[0]
                for artist_index, episode_index in enumerate(selected_indices[split]):
                    small_batch = small_loader.load_step(episode_index)
                    typed_batch = typed_loader.load_step(episode_index)
                    if _episode_signature(small_batch) != _episode_signature(typed_batch):
                        raise RuntimeError(
                            f"Episode mismatch split={split} refs={reference_count} "
                            f"episode={episode_index}"
                        )
                    if not torch.equal(
                        small_batch["conditioning"], typed_batch["conditioning"]
                    ):
                        raise RuntimeError("Text conditioning differs between models")
                    small_references, small_mask = _reference_tokens(
                        small_batch, device, mode="heldout"
                    )
                    typed_references, typed_mask = _reference_tokens(
                        typed_batch, device, mode="heldout"
                    )
                    small_tokens = small(small_references, small_mask)[:1]
                    typed_tokens = typed(typed_references, typed_mask)[:1]
                    episode = typed_batch["episodes"][0]
                    artist_key = (split, artist_index)
                    recorded_target = artist_targets.get(artist_key)
                    current_conditioning = typed_batch["conditioning"][:1]
                    current_length = int(typed_batch["conditioning_lengths"][0])
                    if recorded_target is None:
                        artist_targets[artist_key] = (
                            int(episode.target_id),
                            current_conditioning.clone(),
                            current_length,
                        )
                    elif (
                        recorded_target[0] != int(episode.target_id)
                        or recorded_target[2] != current_length
                        or not torch.equal(recorded_target[1], current_conditioning)
                    ):
                        raise RuntimeError(
                            "Target or caption changed across reference counts for "
                            f"{artist_key}"
                        )
                    style_row = typed_loader.style_by_id[int(episode.target_id)]
                    target_path = Path(str(style_row["local_path"]))
                    reference_paths = [
                        Path(str(typed_loader.style_by_id[int(image_id)]["local_path"]))
                        for image_id in episode.reference_ids
                    ]
                    style_slug = _slug(str(episode.style_id))
                    key = (
                        f"{split}-artist{artist_index + 1:02d}-{style_slug}-"
                        f"refs{reference_count}"
                    )
                    cases.append(
                        SampleCase(
                            key=key,
                            split=split,
                            artist_index=artist_index,
                            style_id=str(episode.style_id),
                            target_id=int(episode.target_id),
                            reference_ids=tuple(int(v) for v in episode.reference_ids),
                            target_path=target_path,
                            reference_paths=reference_paths,
                            conditioning=current_conditioning
                            .to("cpu", dtype=torch.bfloat16)
                            .contiguous(),
                            conditioning_length=current_length,
                            small_tokens=small_tokens.to(
                                "cpu", dtype=torch.bfloat16
                            ).contiguous(),
                            typed_tokens=typed_tokens.to(
                                "cpu", dtype=torch.bfloat16
                            ).contiguous(),
                            noise_seed=(
                                int(cfg["generation"]["seed"])
                                + split_index * 100_003
                                + artist_index * 10_007
                            ),
                        )
                    )
                    print(
                        f"prepared {key} target={episode.target_id} "
                        f"refs={list(episode.reference_ids)}",
                        flush=True,
                    )
                del small_loader, typed_loader
    assert null_conditioning is not None
    del small, typed, legacy_resampler
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return cases, small_metadata, typed_metadata, null_conditioning


def _noise(case: SampleCase, generation: dict[str, Any]) -> torch.Tensor:
    return torch.randn(
        1,
        16,
        1,
        int(generation["height"]) // 8,
        int(generation["width"]) // 8,
        generator=torch.Generator(device="cpu").manual_seed(case.noise_seed),
        dtype=torch.float32,
    ).to(torch.bfloat16)


def _generate_groups(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    cases: list[SampleCase],
    null_conditioning: torch.Tensor,
    device: str,
) -> dict[str, list[Image.Image]]:
    generation = dict(cfg["generation"])
    multipliers = [float(value) for value in cfg["style_multipliers"]]
    batch_size = int(generation.get("batch_size", 4))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=True,
        fuse_attention_projections=True,
    )
    groups: dict[str, torch.Tensor] = {}

    # The target, caption, and noise are held fixed across reference counts.
    # Compute each artist's no-style baseline only once.
    baseline_cases: list[SampleCase] = []
    seen_baselines: set[tuple[str, int]] = set()
    for case in cases:
        identity = (case.split, case.artist_index)
        if identity not in seen_baselines:
            seen_baselines.add(identity)
            baseline_cases.append(case)
    for offset in range(0, len(baseline_cases), batch_size):
        chunk = baseline_cases[offset : offset + batch_size]
        positive = torch.cat([case.conditioning for case in chunk]).to(device)
        negative = null_conditioning[None].expand(len(chunk), -1, -1).to(device)
        noise = torch.cat([_noise(case, generation) for case in chunk]).to(device)
        latents = _denoise_batch(
            anima,
            noise,
            positive,
            negative,
            steps=int(generation["steps"]),
            flow_shift=float(generation.get("flow_shift", 3.0)),
            cfg_scale=float(generation["cfg"]),
        ).to("cpu")
        for case, latent in zip(chunk, latents, strict=True):
            groups[f"base::{case.split}::{case.artist_index}"] = latent[None]
        print(
            f"generated baselines {min(offset + len(chunk), len(baseline_cases))}/"
            f"{len(baseline_cases)}",
            flush=True,
        )

    jobs = [
        (case, model_name, multiplier)
        for case in cases
        for model_name in ("small", "typed")
        for multiplier in multipliers
    ]
    completed = 0
    for multiplier in multipliers:
        multiplier_jobs = [job for job in jobs if job[2] == multiplier]
        for offset in range(0, len(multiplier_jobs), batch_size):
            chunk = multiplier_jobs[offset : offset + batch_size]
            positive = torch.cat(
                [case.conditioning for case, _, _ in chunk]
            ).to(device)
            lengths = torch.tensor(
                [case.conditioning_length for case, _, _ in chunk],
                device=device,
                dtype=torch.long,
            )
            tokens = torch.cat(
                [
                    case.small_tokens
                    if model_name == "small"
                    else case.typed_tokens
                    for case, model_name, _ in chunk
                ]
            ).to(device)
            style_context = insert_style_tokens(positive, lengths, tokens)
            negative = null_conditioning[None].expand(len(chunk), -1, -1).to(device)
            noise = torch.cat(
                [_noise(case, generation) for case, _, _ in chunk]
            ).to(device)
            latents = _denoise_batch(
                anima,
                noise,
                positive,
                negative,
                steps=int(generation["steps"]),
                flow_shift=float(generation.get("flow_shift", 3.0)),
                cfg_scale=float(generation["cfg"]),
                style_context=style_context,
                style_multiplier=multiplier,
            ).to("cpu")
            for (case, model_name, value), latent in zip(
                chunk, latents, strict=True
            ):
                groups[f"style::{case.key}::{model_name}::{value:g}"] = latent[None]
            completed += len(chunk)
            del positive, lengths, tokens, style_context, negative, noise, latents
            if completed % 16 == 0 or completed == len(jobs):
                print(
                    f"generated styled {completed}/{len(jobs)}",
                    flush=True,
                )

    del anima
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _decode_latents(
        config,
        destination,
        groups,
        device,
        int(generation.get("vae_batch_size", 4)),
    )


def generate_comparison_sweep(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["typed_attention_small_visual_sweep"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    raw_root = output / "raw"
    sheet_root = output / "sheets"
    raw_root.mkdir(parents=True, exist_ok=True)
    sheet_root.mkdir(parents=True, exist_ok=True)
    cases, small_metadata, typed_metadata, null_conditioning = _tokenize_cases(
        config, destination, cfg, device
    )
    decoded = _generate_groups(
        config, destination, cfg, cases, null_conditioning, device
    )
    multipliers = [float(value) for value in cfg["style_multipliers"]]
    records = []
    for case in cases:
        case_raw = raw_root / case.key
        case_raw.mkdir(parents=True, exist_ok=True)
        base = decoded[f"base::{case.split}::{case.artist_index}"][0]
        base.save(case_raw / "no-style.png")
        model_images: dict[str, list[Image.Image]] = {}
        for model_name in ("small", "typed"):
            images = [
                decoded[f"style::{case.key}::{model_name}::{multiplier:g}"][0]
                for multiplier in multipliers
            ]
            model_images[model_name] = images
            for multiplier, image in zip(multipliers, images, strict=True):
                image.save(case_raw / f"{model_name}-style-{multiplier:g}x.png")
        sheet_path = sheet_root / f"{case.key}.png"
        _make_case_sheet(
            case,
            base,
            model_images["small"],
            model_images["typed"],
            multipliers,
        ).save(sheet_path, compress_level=4)
        records.append(
            {
                "key": case.key,
                "split": case.split,
                "artist_index": case.artist_index,
                "style_id": case.style_id,
                "target_id": case.target_id,
                "reference_ids": list(case.reference_ids),
                "reference_count": len(case.reference_ids),
                "noise_seed": case.noise_seed,
                "sheet": str(sheet_path),
                "raw_directory": str(case_raw),
            }
        )
    summary = {
        "comparison_contract": {
            "same_artist_target_caption_and_seed_across_reference_counts": True,
            "same_prompt_seed_and_references_between_models": True,
            "different_seed_between_artists": True,
            "artist_tags_in_generation_prompt": False,
            "style_multiplier_formula": (
                "uncond + cfg*(text-uncond) + cfg*multiplier*(style-text)"
            ),
        },
        "small": small_metadata,
        "typed": typed_metadata,
        "generation": dict(cfg["generation"]),
        "style_multipliers": multipliers,
        "cases": records,
        "styled_images": len(cases) * 2 * len(multipliers),
        "baseline_images": len({(case.split, case.artist_index) for case in cases}),
        "sheets": len(records),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = copy.deepcopy(config)
        cfg = config["typed_attention_small_visual_sweep"]
        cfg["output_directory"] = "diagnostics/typed-vs-small-sweep-smoke"
        cfg["splits"] = ["validation"]
        cfg["artists_per_split"] = 1
        cfg["reference_counts"] = [1]
        cfg["style_multipliers"] = [1.0]
    generate_comparison_sweep(config, output_dir(config))


if __name__ == "__main__":
    main()
