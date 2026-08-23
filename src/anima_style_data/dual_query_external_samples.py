from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .anima_cache import _decode_anima_image
from .cradio import _load_cradio, _selected_style_tensors, preprocess_cradio_image
from .dual_query_style_tokenizer import _file_sha256, _load_resampler
from .external_style_tokenizer_sheet import (
    _encode_text_conditions,
    _reference_paths,
)


def _paths_and_config(
    config: dict[str, Any], destination: Path
) -> tuple[dict[str, Any], Path, list[Path], Path]:
    tokenizer_cfg = config["dual_query_style_tokenizer"]
    fixed_cfg = dict(tokenizer_cfg["fixed_reference_sampling"])
    # Reuse the established prompt/seed/image-sheet contract while keeping a
    # distinct feature cache for the new 84-token encoder.
    sheet_cfg = dict(config["external_style_tokenizer_sheet"])
    sheet_cfg.update(dict(fixed_cfg.get("generation", {})))
    sheet_cfg["include_small"] = False
    sheet_cfg["style_multipliers"] = [1.0]
    reference_root = Path(str(sheet_cfg["reference_directory"]))
    if not reference_root.is_absolute():
        reference_root = destination / reference_root
    paths = _reference_paths(reference_root)
    output = destination / str(fixed_cfg["cache_directory"])
    output.mkdir(parents=True, exist_ok=True)
    return sheet_cfg, reference_root, paths, output


def _signature(
    paths: list[Path], checkpoint: Path, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "kind": "dual-query-external-reference-v1",
        "references": [
            {"name": path.parent.name, "file": path.name, "sha256": _file_sha256(path)}
            for path in paths
        ],
        "resampler_checkpoint": str(checkpoint),
        "resampler_sha256": _file_sha256(checkpoint),
        "cradio_layers": [18, 24],
        "cradio_preprocess": {
            **dict(config["style_features"].get("preprocess", {})),
            "patch_size": int(config["cradio"].get("patch_size", 16)),
        },
        "vae_preprocess": dict(config["anima_cache"]["latents"]["preprocess"]),
    }


def encode_dual_query_reference_images(
    config: dict[str, Any],
    destination: Path,
    paths: list[Path],
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Encode arbitrary images with the production C-RADIO/VAE/Resampler path."""
    if not paths:
        raise ValueError("At least one reference image is required")

    feature_cfg = dict(config["style_features"])
    radio_cfg = {**config["cradio"], **feature_cfg.get("preprocess", {})}
    cradio, _ = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    semantic_features: list[dict[int, torch.Tensor]] = []
    semantic_shapes: list[tuple[int, int]] = []
    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                array, geometry = preprocess_cradio_image(image, radio_cfg)
            pixels = torch.from_numpy(array).unsqueeze(0).to(device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
            ):
                intermediate = cradio.forward_intermediates(
                    pixels,
                    indices=[18, 24],
                    return_prefix_tokens=False,
                    norm=True,
                    stop_early=True,
                    output_fmt="NLC",
                    intermediates_only=True,
                    aggregation="sparse",
                )
            selected = _selected_style_tensors(
                intermediate,
                [18, 24],
                {18, 24},
                set(),
                set(),
                torch.float16,
            )[0]
            semantic_features.append(
                {
                    18: selected["layer_18_spatial"].cpu(),
                    24: selected["layer_24_spatial"].cpu(),
                }
            )
            semantic_shapes.append(
                (int(geometry.target_height) // 16, int(geometry.target_width) // 16)
            )
            print(f"reference C-RADIO {index}/{len(paths)}", flush=True)
    del cradio
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    from .style_transfer import _load_sampling_vae

    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    latent_values: list[torch.Tensor] = []
    image_sizes: list[tuple[int, int]] = []
    preprocess = dict(config["anima_cache"]["latents"]["preprocess"])
    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            array, geometry, _, _ = _decode_anima_image(path.read_bytes(), preprocess)
            pixels = torch.from_numpy(array).unsqueeze(0).to(
                device, dtype=torch.bfloat16
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
            ):
                latent = vae.encode_pixels_to_latents(pixels)
            if latent.ndim == 5:
                latent = latent.squeeze(2)
            latent_values.append(latent[0].to("cpu", dtype=torch.float16))
            image_sizes.append((geometry.target_height, geometry.target_width))
            print(f"reference Qwen VAE {index}/{len(paths)}", flush=True)
    del vae
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    checkpoint = destination / str(
        config["dual_query_style_tokenizer"]["resampler_checkpoint"]
    )
    semantic_dim = int(semantic_features[0][18].shape[-1])
    vae_channels = int(latent_values[0].shape[0])
    resampler, checkpoint_step = _load_resampler(
        config,
        destination,
        checkpoint,
        semantic_dim,
        vae_channels,
        device,
    )
    encoded_tokens: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for index, (features, latent, semantic_shape, image_size) in enumerate(
            zip(semantic_features, latent_values, semantic_shapes, image_sizes),
            start=1,
        ):
            count = int(features[18].shape[0])
            encoded = resampler.encode(
                {
                    layer: value.unsqueeze(0).to(device)
                    for layer, value in features.items()
                },
                torch.ones(1, count, device=device, dtype=torch.bool),
                torch.tensor([semantic_shape], device=device),
                latent.unsqueeze(0).to(device),
                torch.tensor(
                    [[int(latent.shape[-2]), int(latent.shape[-1])]], device=device
                ),
                torch.tensor([image_size], device=device),
                reconstruct=False,
            )
            encoded_tokens.append(
                torch.cat((encoded.tokens, encoded.artist_summary), dim=1)[0].to(
                    "cpu", dtype=torch.bfloat16
                )
            )
            print(f"reference Resampler {index}/{len(paths)}", flush=True)
    tokens = torch.stack(encoded_tokens).contiguous()
    expected = (len(paths), 84, 1024)
    if tuple(tokens.shape) != expected or not torch.isfinite(tokens).all():
        raise RuntimeError(f"Invalid dual-query tokens {tuple(tokens.shape)}")
    return {
        "tokens": tokens,
        "checkpoint_step": int(checkpoint_step),
        "paths": paths,
        "image_sizes": image_sizes,
    }


def cache_dual_query_external_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    sheet_cfg, _, paths, output = _paths_and_config(config, destination)
    device = str(
        config["dual_query_style_tokenizer"]["fixed_reference_sampling"].get(
            "device", "cuda"
        )
    )
    checkpoint = destination / str(
        config["dual_query_style_tokenizer"]["resampler_checkpoint"]
    )
    signature = _signature(paths, checkpoint, config)
    tokens_path = output / "reference_tokens.pt"
    metadata_path = output / "reference_tokens.json"
    if tokens_path.exists() and metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
        if recorded == signature and tuple(tokens.shape) == (7, 84, 1024):
            positive, negative, length = _encode_text_conditions(
                config, destination, sheet_cfg, output, device
            )
            return {
                "references": 7,
                "tokens": list(tokens.shape),
                "reused": True,
                "text_length": length,
                "cache_directory": str(output),
            }

    encoded = encode_dual_query_reference_images(
        config, destination, paths, device=device
    )
    tokens = encoded["tokens"]
    checkpoint_step = int(encoded["checkpoint_step"])
    torch.save(tokens, tokens_path)
    metadata_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    positive, negative, length = _encode_text_conditions(
        config, destination, sheet_cfg, output, device
    )
    return {
        "references": 7,
        "tokens": list(tokens.shape),
        "checkpoint_step": checkpoint_step,
        "reused": False,
        "text_length": length,
        "positive_shape": list(positive.shape),
        "negative_shape": list(negative.shape),
        "cache_directory": str(output),
    }


def load_dual_query_external_sample(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    sheet_cfg, _, paths, output = _paths_and_config(config, destination)
    tokens_path = output / "reference_tokens.pt"
    text_path = output / "text_conditions.pt"
    metadata_path = output / "reference_tokens.json"
    if not tokens_path.exists() or not text_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Run dual-query-style-external-cache before Style Tokenizer training"
        )
    checkpoint = destination / str(
        config["dual_query_style_tokenizer"]["resampler_checkpoint"]
    )
    recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
    if recorded != _signature(paths, checkpoint, config):
        raise RuntimeError(
            "Fixed external references were encoded by another model or preprocess"
        )
    tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
    text = torch.load(text_path, map_location="cpu", weights_only=True)
    if (
        text.get("prompt") != str(sheet_cfg["prompt"])
        or text.get("negative_prompt") != str(sheet_cfg["negative_prompt"])
    ):
        raise RuntimeError(
            "Fixed-reference prompt cache is stale; run "
            "dual-query-style-external-cache before training"
        )
    if tuple(tokens.shape) != (7, 84, 1024):
        raise RuntimeError(f"Unexpected fixed reference cache {tuple(tokens.shape)}")
    return {
        "cfg": sheet_cfg,
        "paths": paths,
        "reference_tokens": tokens,
        "positive": text["positive"],
        "negative": text["negative"],
        "length": int(text["length"]),
    }
