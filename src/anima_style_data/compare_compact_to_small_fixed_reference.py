from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import load_config, output_dir
from .external_style_tokenizer_sheet import (
    _decode_latents,
    _extract_reference_tokens,
    _generate_latents,
    _make_sheet,
    _pixel_rms_from_baseline,
    _reference_paths,
)
from .style_tokenizer import AnimaStyleTokenizer
from .style_transfer import (
    _optimize_frozen_anima,
    _resolve_anima_model,
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
    reference_batch = references[:, None].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    mask = torch.ones(7, 1, device=device, dtype=torch.bool)
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


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _pixel_rms(left: Image.Image, right: Image.Image) -> float:
    left_value = np.asarray(left, dtype=np.float32)
    right_value = np.asarray(right, dtype=np.float32)
    if left_value.shape != right_value.shape:
        raise RuntimeError(f"Image-shape mismatch: {left_value.shape} != {right_value.shape}")
    return float(np.sqrt(np.mean((left_value - right_value) ** 2)) / 255.0)


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
    small_tokens, small_metadata = _load_small_tokens(
        checkpoint, references, device
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
        None,
        small_tokens,
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
    small = decoded["large_1x"]
    raw = output / "generated"
    raw.mkdir(exist_ok=True)
    base.save(raw / "no-style.png")
    for index, image in enumerate(small, start=1):
        image.save(raw / f"small-step-{small_metadata['step']}-TestSample{index}.png")

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
    sheet = _make_sheet(paths, base, small, compact, size)
    sheet_path = output / (
        f"small-step-{small_metadata['step']}-vs-compact-step-{compact_step}.png"
    )
    sheet.save(sheet_path, compress_level=4)

    small_rms = _pixel_rms_from_baseline(base, small)
    compact_rms = _pixel_rms_from_baseline(base, compact)
    pair_rms = [_pixel_rms(left, right) for left, right in zip(small, compact, strict=True)]
    summary = {
        "sheet": str(sheet_path),
        "prompt": str(cfg["prompt"]),
        "negative_prompt": str(cfg["negative_prompt"]),
        "width": size[0],
        "height": size[1],
        "cfg": float(cfg["cfg"]),
        "steps": int(cfg["steps"]),
        "seed": int(cfg["seed"]),
        "small": small_metadata,
        "compact_step": compact_step,
        "baseline_pixel_rms": baseline_rms,
        "small_pixel_rms_from_baseline": small_rms,
        "small_mean_pixel_rms_from_baseline": float(np.mean(small_rms)),
        "compact_pixel_rms_from_baseline": compact_rms,
        "compact_mean_pixel_rms_from_baseline": float(np.mean(compact_rms)),
        "small_vs_compact_pixel_rms": pair_rms,
        "small_vs_compact_mean_pixel_rms": float(np.mean(pair_rms)),
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
        default="diagnostics/small-vs-compact-fixed-reference-step2000",
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
