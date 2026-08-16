from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import load_config, output_dir
from .cradio import (
    _load_cradio,
    _selected_style_tensors,
    preprocess_cradio_image,
)
from .external_style_tokenizer_sheet import (
    _decode_latents,
    _extract_reference_tokens,
    _fit_with_padding,
    _generate_latents,
    _label,
    _pixel_rms_from_baseline,
    _reference_paths,
)
from .style_tokenizer import AnimaStyleTokenizer
from .style_transfer import (
    _optimize_frozen_anima,
    _resolve_anima_model,
    load_per_reference_resampler,
)


def _load_prompt_cache(
    cfg: dict[str, Any], cache_directory: Path
) -> tuple[torch.Tensor, torch.Tensor, int]:
    path = cache_directory / "text_conditions.pt"
    cached = torch.load(path, map_location="cpu", weights_only=True)
    if cached.get("prompt") != str(cfg["prompt"]):
        raise RuntimeError(f"Prompt cache mismatch: {path}")
    if cached.get("negative_prompt") != str(cfg["negative_prompt"]):
        raise RuntimeError(f"Negative-prompt cache mismatch: {path}")
    return cached["positive"], cached["negative"], int(cached["length"])


def _load_small_tokens(
    checkpoint: Path,
    references: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    tokenizer = AnimaStyleTokenizer(**dict(state["config"]["model"]))
    tokenizer.load_state_dict(state["tokenizer"], strict=True)
    tokenizer.to(device).eval()
    if references.ndim == 3:
        reference_batch = references[:, None]
    elif references.ndim == 4:
        reference_batch = references
    else:
        raise RuntimeError(f"Unexpected reference rank: {references.ndim}")
    reference_batch = reference_batch.to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    mask = torch.ones(
        reference_batch.shape[:2], device=device, dtype=torch.bool
    )
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        tokens = tokenizer(reference_batch, mask)
    result = tokens.to("cpu", dtype=torch.bfloat16).contiguous()
    metadata = {
        "checkpoint": str(checkpoint),
        "step": int(state["step"]),
        "parameters": sum(parameter.numel() for parameter in tokenizer.parameters()),
        "input_shape": list(references.shape),
        "output_shape": list(result.shape),
    }
    del tokenizer, state, reference_batch, mask, tokens
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, metadata


def _augmentation_views(path: Path) -> list[tuple[str, Image.Image]]:
    with Image.open(path) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    width, height = original.size
    crop_width = max(16, int(round(width * 0.875)))
    crop_height = max(16, int(round(height * 0.875)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    crop = original.crop((left, top, left + crop_width, top + crop_height))
    return [
        ("original", original),
        ("horizontal-flip", ImageOps.mirror(original)),
        ("center-crop-87.5pct", crop),
        ("center-crop-87.5pct-horizontal-flip", ImageOps.mirror(crop)),
    ]


def _extract_augmented_reference_tokens(
    config: dict[str, Any],
    destination: Path,
    paths: list[Path],
    output: Path,
    device: str,
) -> torch.Tensor:
    feature_cfg = dict(config["style_features"])
    radio_cfg = {**config["cradio"], **feature_cfg.get("preprocess", {})}
    layers = sorted(
        set(int(value) for value in feature_cfg["spatial_layers"])
        | set(int(value) for value in feature_cfg.get("summary_layers", []))
    )
    spatial_layers = {int(value) for value in feature_cfg["spatial_layers"]}
    summary_layers = {int(value) for value in feature_cfg.get("summary_layers", [])}
    amp_dtype = getattr(torch, str(feature_cfg.get("amp_dtype", "bfloat16")))
    view_root = output / "augmented_reference_views"
    view_root.mkdir(parents=True, exist_ok=True)

    cradio, _ = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    grouped_features: list[list[dict[str, torch.Tensor]]] = []
    with torch.inference_mode():
        for sample_index, path in enumerate(paths, start=1):
            group = []
            sample_root = view_root / f"TestSample{sample_index}"
            sample_root.mkdir(exist_ok=True)
            for view_index, (name, image) in enumerate(_augmentation_views(path)):
                image.save(sample_root / f"{view_index}-{name}.png")
                array, _ = preprocess_cradio_image(image, radio_cfg)
                values = torch.from_numpy(array).unsqueeze(0).to(device)
                with torch.autocast(
                    "cuda", dtype=amp_dtype, enabled=device.startswith("cuda")
                ):
                    intermediate = cradio.forward_intermediates(
                        values,
                        indices=layers,
                        return_prefix_tokens=bool(summary_layers),
                        norm=True,
                        stop_early=True,
                        output_fmt="NLC",
                        intermediates_only=True,
                        aggregation="sparse",
                    )
                selected = _selected_style_tensors(
                    intermediate,
                    layers,
                    spatial_layers,
                    set(),
                    summary_layers,
                    torch.float16,
                )[0]
                group.append(selected)
                print(
                    f"encoded augmented reference {sample_index}/7 view {view_index + 1}/4",
                    flush=True,
                )
            grouped_features.append(group)
    del cradio
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    resampler_cfg = dict(config["style_transfer"]["resampler"])
    resampler = load_per_reference_resampler(
        destination, resampler_cfg, device, trainable=False
    )
    grouped_tokens = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for group in grouped_features:
            tokens = []
            for selected in group:
                layer18 = selected["layer_18_spatial"].to(device).unsqueeze(0)
                layer24 = selected["layer_24_spatial"].to(device).unsqueeze(0)
                mask = torch.ones(
                    1, layer18.shape[1], device=device, dtype=torch.bool
                )
                global_feature = selected["layer_24_siglip_cls"].to(device).unsqueeze(0)
                _, representation = resampler.encode(
                    {18: layer18, 24: layer24}, mask, global_feature
                )
                tokens.append(representation[0].to("cpu", dtype=torch.bfloat16))
            grouped_tokens.append(torch.stack(tokens))
    result = torch.stack(grouped_tokens).contiguous()
    if tuple(result.shape) != (7, 4, 128, 1024):
        raise RuntimeError(f"Invalid augmented token shape: {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise RuntimeError("Augmented reference tokens contain non-finite values")
    torch.save(result, output / "augmented_reference_resampler_tokens.pt")
    del resampler
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _pixel_rms(left: Image.Image, right: Image.Image) -> float:
    left_value = np.asarray(left, dtype=np.float32)
    right_value = np.asarray(right, dtype=np.float32)
    if left_value.shape != right_value.shape:
        raise RuntimeError(f"Image-shape mismatch: {left_value.shape} != {right_value.shape}")
    return float(np.sqrt(np.mean((left_value - right_value) ** 2)) / 255.0)


def _make_labeled_sheet(
    paths: list[Path],
    base: Image.Image,
    rows: list[tuple[str, list[Image.Image]]],
    size: tuple[int, int],
) -> Image.Image:
    reference_cells = []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            cell = _fit_with_padding(image, size)
        _label(cell, [f"TestSample {index}"], align="right")
        reference_cells.append(cell)
    all_rows = [("REFERENCE ORIGINALS", reference_cells), *rows]
    sheet = Image.new("RGB", (size[0] * 8, size[1] * len(all_rows)), "white")
    for row_index, (name, images) in enumerate(all_rows):
        baseline = base.copy()
        _label(baseline, [name, "NO STYLE BASELINE"])
        sheet.paste(baseline, (0, row_index * size[1]))
        for column, image in enumerate(images, start=1):
            sheet.paste(image.convert("RGB"), (column * size[0], row_index * size[1]))
    return sheet


def compare_fixed_references(
    config: dict[str, Any],
    destination: Path,
    *,
    checkpoint: Path,
    compact_step: int,
    output_name: str,
) -> dict[str, Any]:
    device = "cuda"
    cfg = dict(config["external_style_tokenizer_sheet"])
    compact_cfg = dict(config["compact_dual_query_style_tokenizer"])
    fixed_cfg = dict(compact_cfg["fixed_reference_sampling"])
    cfg.update(dict(fixed_cfg.get("generation", {})))
    cfg["style_multipliers"] = [1.0]

    reference_root = Path(str(cfg["reference_directory"]))
    if not reference_root.is_absolute():
        reference_root = destination / reference_root
    paths = _reference_paths(reference_root)
    output = destination / output_name
    output.mkdir(parents=True, exist_ok=True)

    # The selected small tokenizer was trained on the original 128-token
    # per-reference Resampler, not the 84-token Dual-query representation.
    legacy_cache = destination / "external_style_tokenizer_sheet_4k"
    references = _extract_reference_tokens(
        config, destination, paths, legacy_cache, device
    )
    if tuple(references.shape) != (7, 128, 1024):
        raise RuntimeError(f"Unexpected legacy reference shape: {tuple(references.shape)}")
    single_tokens, small_metadata = _load_small_tokens(
        checkpoint, references, device
    )
    augmented_references = _extract_augmented_reference_tokens(
        config, destination, paths, output, device
    )
    augmented_tokens, augmented_metadata = _load_small_tokens(
        checkpoint, augmented_references, device
    )

    prompt_cache = destination / str(fixed_cfg["cache_directory"])
    positive, negative, length = _load_prompt_cache(cfg, prompt_cache)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=True,
        fuse_attention_projections=True,
    )
    latent_groups = _generate_latents(
        anima,
        positive,
        negative,
        length,
        single_tokens,
        augmented_tokens,
        cfg,
        device,
    )
    del anima
    gc.collect()
    torch.cuda.empty_cache()

    decoded = _decode_latents(
        config,
        destination,
        latent_groups,
        device,
        int(cfg.get("vae_batch_size", 4)),
    )
    base = decoded["base"][0]
    single = decoded["small"]
    augmented = decoded["large_1x"]
    raw = output / "generated"
    raw.mkdir(exist_ok=True)
    base.save(raw / "no-style.png")
    for index, image in enumerate(single, start=1):
        image.save(raw / f"small-single-TestSample{index}.png")
    for index, image in enumerate(augmented, start=1):
        image.save(raw / f"small-augmented-4ref-TestSample{index}.png")

    compact_root = (
        destination
        / str(compact_cfg["output_directory"])
        / "external_reference_samples"
        / f"step-{compact_step:07d}"
        / "generated"
    )
    compact_base = _load_rgb(compact_root / "no-style.png")
    compact = [
        _load_rgb(compact_root / f"large_1x-TestSample{index}.png")
        for index in range(1, 8)
    ]
    baseline_rms = _pixel_rms(base, compact_base)
    size = (int(cfg["width"]), int(cfg["height"]))
    single_sheet_path = output / "small-single-reference.png"
    _make_labeled_sheet(
        paths,
        base,
        [("SMALL TOKENIZER / 1 ORIGINAL REF", single)],
        size,
    ).save(single_sheet_path, compress_level=4)
    augmented_sheet_path = output / "small-augmented-4-reference.png"
    _make_labeled_sheet(
        paths,
        base,
        [("SMALL TOKENIZER / 4 AUGMENTED REFS", augmented)],
        size,
    ).save(augmented_sheet_path, compress_level=4)
    comparison_sheet_path = output / (
        f"small-single-vs-aug4-vs-compact-step-{compact_step}.png"
    )
    _make_labeled_sheet(
        paths,
        base,
        [
            ("SMALL TOKENIZER / 1 ORIGINAL REF", single),
            ("SMALL TOKENIZER / 4 AUGMENTED REFS", augmented),
            (f"COMPACT DUAL-QUERY / STEP {compact_step}", compact),
        ],
        size,
    ).save(comparison_sheet_path, compress_level=4)

    single_rms = _pixel_rms_from_baseline(base, single)
    augmented_rms = _pixel_rms_from_baseline(base, augmented)
    compact_rms = _pixel_rms_from_baseline(base, compact)
    single_vs_augmented = [
        _pixel_rms(left, right)
        for left, right in zip(single, augmented, strict=True)
    ]
    single_vs_compact = [
        _pixel_rms(left, right) for left, right in zip(single, compact, strict=True)
    ]
    summary = {
        "sheets": {
            "single_reference": str(single_sheet_path),
            "augmented_4_reference": str(augmented_sheet_path),
            "combined": str(comparison_sheet_path),
        },
        "prompt": str(cfg["prompt"]),
        "negative_prompt": str(cfg["negative_prompt"]),
        "width": size[0],
        "height": size[1],
        "cfg": float(cfg["cfg"]),
        "steps": int(cfg["steps"]),
        "seed": int(cfg["seed"]),
        "small": small_metadata,
        "small_augmented": augmented_metadata,
        "augmentation_views": [
            "original",
            "horizontal-flip",
            "center-crop-87.5pct",
            "center-crop-87.5pct-horizontal-flip",
        ],
        "compact_step": compact_step,
        "baseline_pixel_rms": baseline_rms,
        "small_single_pixel_rms_from_baseline": single_rms,
        "small_single_mean_pixel_rms_from_baseline": float(np.mean(single_rms)),
        "small_augmented_pixel_rms_from_baseline": augmented_rms,
        "small_augmented_mean_pixel_rms_from_baseline": float(np.mean(augmented_rms)),
        "compact_pixel_rms_from_baseline": compact_rms,
        "compact_mean_pixel_rms_from_baseline": float(np.mean(compact_rms)),
        "small_single_vs_augmented_pixel_rms": single_vs_augmented,
        "small_single_vs_augmented_mean_pixel_rms": float(
            np.mean(single_vs_augmented)
        ),
        "small_single_vs_compact_pixel_rms": single_vs_compact,
        "small_single_vs_compact_mean_pixel_rms": float(np.mean(single_vs_compact)),
        "references": [str(path) for path in paths],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=(
            "style_tokenizer_native_context_16_multiref_contrastive_8k_v1/"
            "checkpoints/step-0007750.pt"
        ),
    )
    parser.add_argument("--compact-step", type=int, default=2000)
    parser.add_argument(
        "--output-name",
        default="diagnostics/small-fixed-reference-single-vs-aug4-step2000",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    destination = output_dir(config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = destination / checkpoint
    compare_fixed_references(
        config,
        destination,
        checkpoint=checkpoint,
        compact_step=args.compact_step,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
