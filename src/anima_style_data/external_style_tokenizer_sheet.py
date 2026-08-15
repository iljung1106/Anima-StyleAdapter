from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.torch import save_file

from .config import load_config, output_dir
from .cradio import (
    _load_cradio,
    _selected_style_tensors,
    preprocess_cradio_image,
)
from .query_style_tokenizer import QueryStyleTokenizerV2
from .style_calibration import _encode_prompts
from .style_tokenizer import AnimaStyleTokenizer, insert_style_tokens
from .style_transfer import (
    _load_sampling_vae,
    _optimize_frozen_anima,
    _resolve_anima_model,
    load_per_reference_resampler,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_paths(root: Path) -> list[Path]:
    paths = []
    for index in range(1, 8):
        folder = root / f"TestSample{index}"
        values = sorted(path for path in folder.iterdir() if path.is_file())
        if len(values) != 1:
            raise RuntimeError(
                f"{folder} must contain exactly one reference image, got {len(values)}"
            )
        paths.append(values[0])
    return paths


def _reference_signature(paths: list[Path], resampler_checkpoint: Path) -> dict[str, Any]:
    return {
        "references": [
            {"name": path.parent.name, "file": path.name, "sha256": _sha256(path)}
            for path in paths
        ],
        "resampler_checkpoint": str(resampler_checkpoint),
        "resampler_sha256": _sha256(resampler_checkpoint),
    }


def _extract_reference_tokens(
    config: dict[str, Any],
    destination: Path,
    paths: list[Path],
    output: Path,
    device: str,
) -> torch.Tensor:
    cache_path = output / "reference_resampler_tokens.pt"
    metadata_path = output / "reference_resampler_tokens.json"
    resampler_cfg = dict(config["style_transfer"]["resampler"])
    checkpoint = destination / str(resampler_cfg["checkpoint"])
    signature = _reference_signature(paths, checkpoint)
    if cache_path.exists() and metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if recorded == signature:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if tuple(cached.shape) == (7, 128, 1024):
                print("reused external C-RADIO/Resampler token cache", flush=True)
                return cached

    feature_cfg = dict(config["style_features"])
    radio_cfg = {**config["cradio"], **feature_cfg.get("preprocess", {})}
    layers = sorted(
        set(int(value) for value in feature_cfg["spatial_layers"])
        | set(int(value) for value in feature_cfg.get("summary_layers", []))
    )
    spatial_layers = {int(value) for value in feature_cfg["spatial_layers"]}
    summary_layers = {int(value) for value in feature_cfg.get("summary_layers", [])}
    amp_dtype = getattr(torch, str(feature_cfg.get("amp_dtype", "bfloat16")))
    cradio, _ = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    features = []
    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                array, info = preprocess_cradio_image(image, radio_cfg)
            values = torch.from_numpy(array).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
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
            features.append(selected)
            print(
                f"encoded external reference {index}/7 "
                f"shape={info.target_width}x{info.target_height}",
                flush=True,
            )
    del cradio
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    resampler = load_per_reference_resampler(
        destination, resampler_cfg, device, trainable=False
    )
    tokens = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for selected in features:
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
    result = torch.stack(tokens).contiguous()
    if tuple(result.shape) != (7, 128, 1024) or not torch.isfinite(result).all():
        raise RuntimeError(f"Invalid external reference token shape: {tuple(result.shape)}")
    torch.save(result, cache_path)
    metadata_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del resampler
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _load_tokenizer_tokens(
    checkpoint: Path,
    architecture: type[torch.nn.Module],
    references: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if int(state["step"]) != 4000:
        raise RuntimeError(f"Expected a 4000-step checkpoint, got {state['step']}")
    tokenizer = architecture(**dict(state["config"]["model"]))
    tokenizer.load_state_dict(state["tokenizer"], strict=True)
    tokenizer.to(device).eval()
    del state
    reference_batch = references[:, None].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    mask = torch.ones(7, 1, device=device, dtype=torch.bool)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        output = tokenizer(reference_batch, mask)
        tokens = output if isinstance(output, torch.Tensor) else output.tokens
    metadata = {
        "checkpoint": str(checkpoint),
        "step": 4000,
        "parameters": sum(parameter.numel() for parameter in tokenizer.parameters()),
        "output_tokens": int(tokens.shape[1]),
    }
    result = tokens.to("cpu", dtype=torch.bfloat16).contiguous()
    del tokenizer, reference_batch, output, tokens
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, metadata


def _encode_text_conditions(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    output: Path,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    cache_path = output / "text_conditions.pt"
    prompt = str(cfg["prompt"])
    negative = str(cfg["negative_prompt"])
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("prompt") == prompt and cached.get("negative_prompt") == negative:
            print("reused external prompt condition cache", flush=True)
            return cached["positive"], cached["negative"], int(cached["length"])
    encoded = _encode_prompts(
        config,
        destination,
        [prompt, negative],
        device,
        batch_size=2,
    ).to("cpu", dtype=torch.float16)
    positive, negative_condition = encoded[0:1], encoded[1:2]
    length = int((positive[0].float().abs().sum(dim=-1) > 0).sum())
    torch.save(
        {
            "prompt": prompt,
            "negative_prompt": negative,
            "positive": positive,
            "negative": negative_condition,
            "length": length,
        },
        cache_path,
    )
    return positive, negative_condition, length


def _denoise_batch(
    anima: torch.nn.Module,
    initial_noise: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    steps: int,
    flow_shift: float,
    cfg_scale: float,
) -> torch.Tensor:
    batch = positive.shape[0]
    x = initial_noise.expand(batch, -1, -1, -1, -1).clone()
    sigmas = torch.linspace(
        1.0, 0.0, steps + 1, device=x.device, dtype=torch.bfloat16
    )
    sigmas = sigmas * flow_shift / (1 + (flow_shift - 1) * sigmas)
    padding_mask = torch.zeros(
        batch, 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    negative = negative.expand(batch, -1, -1)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"
    ):
        for index in range(steps):
            timestep = sigmas[index].expand(batch)
            unconditioned = anima(
                x,
                timestep,
                context=negative,
                padding_mask=padding_mask,
                target_input_ids=None,
            ).float()
            conditioned = anima(
                x,
                timestep,
                context=positive,
                padding_mask=padding_mask,
                target_input_ids=None,
            ).float()
            velocity = unconditioned + cfg_scale * (conditioned - unconditioned)
            x = (
                x.float()
                + velocity * (sigmas[index + 1] - sigmas[index]).float()
            ).to(torch.bfloat16)
    return x


def _generate_latents(
    anima: torch.nn.Module,
    positive: torch.Tensor,
    negative: torch.Tensor,
    length: int,
    small_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    cfg: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = int(cfg["width"])
    height = int(cfg["height"])
    seed = int(cfg["seed"])
    batch_size = int(cfg.get("batch_size", 4))
    initial_noise = torch.randn(
        1,
        16,
        1,
        height // 8,
        width // 8,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device, dtype=torch.bfloat16)
    positive = positive.to(device, dtype=torch.bfloat16)
    negative = negative.to(device, dtype=torch.bfloat16)
    steps = int(cfg["steps"])
    flow_shift = float(cfg.get("flow_shift", 3.0))
    cfg_scale = float(cfg["cfg"])
    base = _denoise_batch(
        anima,
        initial_noise,
        positive,
        negative,
        steps=steps,
        flow_shift=flow_shift,
        cfg_scale=cfg_scale,
    ).to("cpu")

    lengths = torch.full((7,), length, device=device, dtype=torch.long)

    def styled(values: torch.Tensor) -> torch.Tensor:
        contexts = insert_style_tokens(
            positive.expand(7, -1, -1).clone(),
            lengths,
            values.to(device, dtype=torch.bfloat16),
        )
        parts = []
        for offset in range(0, 7, batch_size):
            part = contexts[offset : offset + batch_size]
            parts.append(
                _denoise_batch(
                    anima,
                    initial_noise,
                    part,
                    negative,
                    steps=steps,
                    flow_shift=flow_shift,
                    cfg_scale=cfg_scale,
                ).to("cpu")
            )
        return torch.cat(parts)

    return base, styled(small_tokens), styled(current_tokens)


def _to_image(value: torch.Tensor) -> Image.Image:
    if value.ndim == 4:
        value = value[:, 0]
    pixels = (
        (value.clamp(-1, 1) + 1) * 127.5
    ).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels)


def _decode_latents(
    config: dict[str, Any],
    destination: Path,
    groups: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    batch_size: int,
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image]]:
    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    all_latents = torch.cat(groups)
    decoded = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for offset in range(0, len(all_latents), batch_size):
            values = vae.decode_to_pixels(
                all_latents[offset : offset + batch_size].to(
                    device, dtype=torch.bfloat16
                )
            ).float()
            decoded.extend(_to_image(value) for value in values)
    return decoded[:1], decoded[1:8], decoded[8:15]


def _fit_with_padding(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", size, (245, 245, 245, 255))
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    background.alpha_composite(contained, (left, top))
    return background.convert("RGB")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _label(image: Image.Image, lines: list[str], *, align: str = "left") -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(30)
    padding = 14
    spacing = 6
    boxes = [draw.textbbox((0, 0), value, font=font) for value in lines]
    width = max(box[2] - box[0] for box in boxes) + 2 * padding
    height = sum(box[3] - box[1] for box in boxes) + spacing * (len(lines) - 1) + 2 * padding
    left = image.width - width - 12 if align == "right" else 12
    top = 12
    draw.rounded_rectangle(
        (left, top, left + width, top + height), radius=10, fill=(0, 0, 0, 180)
    )
    y = top + padding
    for line, box in zip(lines, boxes, strict=True):
        draw.text((left + padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += box[3] - box[1] + spacing


def _make_sheet(
    paths: list[Path],
    base: Image.Image,
    small: list[Image.Image],
    current: list[Image.Image],
    size: tuple[int, int],
) -> Image.Image:
    reference_cells = []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            cell = _fit_with_padding(image, size)
        _label(cell, [f"TestSample {index}"], align="right")
        reference_cells.append(cell)
    rows = [reference_cells, small, current]
    row_names = ["REFERENCE ORIGINALS", "SMALL TOKENIZER", "LARGE TOKENIZER"]
    sheet = Image.new("RGB", (size[0] * 8, size[1] * 3), "white")
    for row_index, (name, values) in enumerate(zip(row_names, rows, strict=True)):
        baseline = base.copy()
        _label(baseline, [name, "NO STYLE BASELINE"])
        sheet.paste(baseline, (0, row_index * size[1]))
        for column, image in enumerate(values, start=1):
            if image.size != size:
                raise RuntimeError(
                    f"Generated cell was resized unexpectedly: {image.size} != {size}"
                )
            sheet.paste(image.convert("RGB"), (column * size[0], row_index * size[1]))
    return sheet


def generate_external_style_tokenizer_sheet(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["external_style_tokenizer_sheet"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    reference_root = Path(str(cfg["reference_directory"]))
    if not reference_root.is_absolute():
        reference_root = destination / reference_root
    paths = _reference_paths(reference_root)

    reference_tokens = _extract_reference_tokens(
        config, destination, paths, output, device
    )
    small_tokens, small_metadata = _load_tokenizer_tokens(
        destination / str(cfg["small_checkpoint"]),
        AnimaStyleTokenizer,
        reference_tokens,
        device,
    )
    current_tokens, current_metadata = _load_tokenizer_tokens(
        destination / str(cfg["current_checkpoint"]),
        QueryStyleTokenizerV2,
        reference_tokens,
        device,
    )
    save_file(
        {
            "small": small_tokens,
            "current": current_tokens,
        },
        output / "style_tokens.safetensors",
    )
    positive, negative, length = _encode_text_conditions(
        config, destination, cfg, output, device
    )
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
        small_tokens,
        current_tokens,
        cfg,
        device,
    )
    del anima
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    base_images, small_images, current_images = _decode_latents(
        config,
        destination,
        latent_groups,
        device,
        int(cfg.get("vae_batch_size", 4)),
    )
    raw = output / "generated"
    raw.mkdir(exist_ok=True)
    base_images[0].save(raw / "no-style.png")
    for index, image in enumerate(small_images, start=1):
        image.save(raw / f"small-TestSample{index}.png")
    for index, image in enumerate(current_images, start=1):
        image.save(raw / f"large-TestSample{index}.png")

    size = (int(cfg["width"]), int(cfg["height"]))
    sheet = _make_sheet(
        paths, base_images[0], small_images, current_images, size
    )
    expected = (size[0] * 8, size[1] * 3)
    if sheet.size != expected:
        raise RuntimeError(f"Unexpected final sheet size: {sheet.size} != {expected}")
    sheet_path = output / str(cfg.get("sheet_filename", "comparison-3x8.png"))
    sheet.save(sheet_path, compress_level=4)
    summary = {
        "sheet": str(sheet_path),
        "sheet_width": sheet.width,
        "sheet_height": sheet.height,
        "cell_width": size[0],
        "cell_height": size[1],
        "prompt": str(cfg["prompt"]),
        "negative_prompt": str(cfg["negative_prompt"]),
        "cfg": float(cfg["cfg"]),
        "steps": int(cfg["steps"]),
        "seed": int(cfg["seed"]),
        "small": small_metadata,
        "current": current_metadata,
        "references": [str(path) for path in paths],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = generate_external_style_tokenizer_sheet(config, output_dir(config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
