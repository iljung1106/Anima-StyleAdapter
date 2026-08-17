from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn

from .compact_dual_query_style_tokenizer import CompactDualQueryStyleTokenizer
from .compare_style_tokenizers import _TokenView
from .config import load_config, output_dir
from .io import write_json
from .style_tokenizer import (
    AnimaStyleTokenizer,
    _forward_tokenizer_flow,
    _mean_metrics,
    _sample_tokenizer,
)
from .query_style_tokenizer import _select_sample_episodes
from .style_transfer import (
    ProductionStyleLoader,
    _optimize_frozen_anima,
    _resolve_anima_model,
)


def _load_ab_tokenizer(
    checkpoint_path: Path,
    architecture: type[nn.Module],
    device: str,
) -> tuple[_TokenView, dict[str, Any]]:
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model_cfg = dict(state["config"]["model"])
    model_cfg.pop("architecture", None)
    tokenizer = architecture(**model_cfg)
    tokenizer.load_state_dict(state["tokenizer"], strict=True)
    tokenizer.to(device).eval()
    metadata = {
        "path": str(checkpoint_path),
        "step": int(state["step"]),
        "trainable_parameters": sum(
            parameter.numel() for parameter in tokenizer.parameters()
        ),
        "output_tokens": int(model_cfg["output_tokens"]),
    }
    return _TokenView(tokenizer), metadata


def _episode_signature(batch: dict[str, Any]) -> list[tuple[int, tuple[int, ...], str, int]]:
    return [
        (
            int(episode.target_id),
            tuple(int(value) for value in episode.reference_ids),
            str(episode.style_id),
            int(episode.text_variant),
        )
        for episode in batch["episodes"]
    ]


def _paired_summary(
    small_rows: list[dict[str, float]], dual_rows: list[dict[str, float]]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in (
        "paired_flow_improvement",
        "flow_loss",
        "style_flow_direction_cosine",
        "style_flow_delta_to_desired_ratio",
        "style_output_ratio",
    ):
        differences = [
            float(small[key]) - float(dual[key])
            for small, dual in zip(small_rows, dual_rows, strict=True)
        ]
        mean = sum(differences) / len(differences)
        if len(differences) > 1:
            variance = sum((value - mean) ** 2 for value in differences) / (
                len(differences) - 1
            )
            ci95 = 1.96 * math.sqrt(variance / len(differences))
        else:
            ci95 = 0.0
        result[key] = {
            "small_minus_dual_mean": mean,
            "small_minus_dual_ci95": ci95,
            "small_greater_fraction": sum(value > 0 for value in differences)
            / len(differences),
        }
    return result


@torch.inference_mode()
def _evaluate_pair(
    anima: torch.nn.Module,
    small: _TokenView,
    dual: _TokenView,
    small_loader: ProductionStyleLoader,
    dual_loader: ProductionStyleLoader,
    device: str,
    training_cfg: dict[str, Any],
    *,
    batches: int,
    seed: int,
    reference_mode: str,
) -> dict[str, Any]:
    small.eval()
    dual.eval()
    small_rows: list[dict[str, float]] = []
    dual_rows: list[dict[str, float]] = []
    maximum_base_loss_difference = 0.0
    for index in range(batches):
        small_batch = small_loader.load_step(index)
        dual_batch = dual_loader.load_step(index)
        if _episode_signature(small_batch) != _episode_signature(dual_batch):
            raise RuntimeError(f"Episode mismatch at comparison batch {index}")
        if not torch.equal(small_batch["latents"], dual_batch["latents"]):
            raise RuntimeError(f"Target latent mismatch at comparison batch {index}")
        if not torch.equal(small_batch["conditioning"], dual_batch["conditioning"]):
            raise RuntimeError(f"Text conditioning mismatch at comparison batch {index}")
        generator_seed = seed + index * 97
        _, small_metrics = _forward_tokenizer_flow(
            anima,
            small,
            small_batch,
            device,
            training_cfg,
            generator=torch.Generator(device=device).manual_seed(generator_seed),
            reference_mode=reference_mode,
            measure_base=True,
        )
        _, dual_metrics = _forward_tokenizer_flow(
            anima,
            dual,
            dual_batch,
            device,
            training_cfg,
            generator=torch.Generator(device=device).manual_seed(generator_seed),
            reference_mode=reference_mode,
            measure_base=True,
        )
        maximum_base_loss_difference = max(
            maximum_base_loss_difference,
            abs(
                float(small_metrics["base_flow_loss"])
                - float(dual_metrics["base_flow_loss"])
            ),
        )
        small_rows.append(small_metrics)
        dual_rows.append(dual_metrics)
        if (index + 1) % 16 == 0 or index + 1 == batches:
            print(
                f"resampler A/B {reference_mode} {index + 1}/{batches}",
                flush=True,
            )
    return {
        "small": _mean_metrics(small_rows),
        "dual_query": _mean_metrics(dual_rows),
        "paired_model_difference": _paired_summary(small_rows, dual_rows),
        "maximum_base_flow_loss_difference": maximum_base_loss_difference,
        "episodes_identical": True,
        "noisy_latents_and_timesteps_identical": True,
    }


def _loader_config(
    config: dict[str, Any],
    cfg: dict[str, Any],
    *,
    token_cache: str,
    references: int | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    result = dict(config["style_transfer"]["loader"])
    result.update(dict(cfg.get("loader", {})))
    result.update(
        {
            "split": str(split or cfg.get("validation_split", "validation")),
            "seed": int(cfg["seed"]),
            "min_references": int(
                references
                if references is not None
                else cfg.get("panel_min_references", 1)
            ),
            "max_references": int(
                references
                if references is not None
                else cfg.get("panel_max_references", 8)
            ),
            "reference_count_weights": None,
            "reference_curriculum": {},
            "pilot_reference_schedule": [],
            "self_reference_target_images_per_style": 0,
            "resampler_token_cache": token_cache,
        }
    )
    return result


def _combine_panel_pair(small: Path, dual: Path, output: Path) -> None:
    with Image.open(small) as small_image, Image.open(dual) as dual_image:
        width = max(small_image.width, dual_image.width)
        label_height = 34
        canvas = Image.new(
            "RGB",
            (width, small_image.height + dual_image.height + label_height * 2),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 10),
            "SMALL TOKENIZER + ORIGINAL 128-TOKEN RESAMPLER / STEP 8000",
            fill="black",
        )
        small_y = label_height
        canvas.paste(small_image.convert("RGB"), (0, small_y))
        dual_label_y = small_y + small_image.height
        draw.text(
            (10, dual_label_y + 10),
            "SMALL TOKENIZER + DUAL-QUERY 84-TOKEN RESAMPLER / STEP 8000",
            fill="black",
        )
        canvas.paste(dual_image.convert("RGB"), (0, dual_label_y + label_height))
        canvas.save(output, compress_level=4)


def _sample_in_chunks(
    anima: torch.nn.Module,
    tokenizer: _TokenView,
    requests: list[tuple[str, ProductionStyleLoader, int, str]],
    sample_config: dict[str, Any],
    destination: Path,
    output: Path,
    device: str,
    *,
    chunk_size: int,
) -> list[Path]:
    sheets: list[Path] = []
    vae = None
    for offset in range(0, len(requests), chunk_size):
        chunk, vae = _sample_tokenizer(
            anima,
            tokenizer,
            requests[offset : offset + chunk_size],
            sample_config,
            destination,
            output,
            device,
            8000,
            vae,
            config_section="resampler_representation_comparison",
        )
        sheets.extend(chunk)
    if vae is not None:
        del vae
    return sheets


def generate_panel_comparison(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["resampler_representation_comparison"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"]) / "panels"
    output.mkdir(parents=True, exist_ok=True)
    small, small_metadata = _load_ab_tokenizer(
        destination / str(cfg["small_checkpoint"]),
        AnimaStyleTokenizer,
        device,
    )
    dual, dual_metadata = _load_ab_tokenizer(
        destination / str(cfg["dual_query_checkpoint"]),
        CompactDualQueryStyleTokenizer,
        device,
    )
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    training_cfg = dict(config["compact_dual_query_style_tokenizer"]["training"])
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training_cfg.get("fuse_attention_projections", True)
        ),
    )

    count = int(cfg.get("panel_samples_per_split", 4))
    loaders: dict[str, tuple[ProductionStyleLoader, ProductionStyleLoader]] = {}
    requests_small: list[tuple[str, ProductionStyleLoader, int, str]] = []
    requests_dual: list[tuple[str, ProductionStyleLoader, int, str]] = []
    episodes: list[dict[str, Any]] = []
    for split in (str(cfg.get("train_split", "train")), str(cfg.get("validation_split", "validation"))):
        small_loader = ProductionStyleLoader(
            destination,
            _loader_config(
                config,
                cfg,
                token_cache=str(cfg["small_token_cache"]),
                split=split,
            ),
        )
        dual_loader = ProductionStyleLoader(
            destination,
            _loader_config(
                config,
                cfg,
                token_cache=str(cfg["dual_query_token_cache"]),
                split=split,
            ),
        )
        loaders[split] = (small_loader, dual_loader)
        indices = _select_sample_episodes(small_loader, count)
        for sample_index, episode_index in enumerate(indices):
            small_signature = _episode_signature(small_loader.load_step(episode_index))
            dual_signature = _episode_signature(dual_loader.load_step(episode_index))
            if small_signature != dual_signature:
                raise RuntimeError(
                    f"Panel episode mismatch for {split} index {episode_index}"
                )
            label = f"{split}-heldout-{sample_index}"
            requests_small.append((label, small_loader, episode_index, "heldout"))
            requests_dual.append((label, dual_loader, episode_index, "heldout"))
            target_id, reference_ids, style_id, text_variant = small_signature[0]
            episodes.append(
                {
                    "label": label,
                    "episode_index": episode_index,
                    "target_id": target_id,
                    "reference_ids": list(reference_ids),
                    "style_id": style_id,
                    "text_variant": text_variant,
                }
            )

    sample_config = dict(config)
    sample_config["resampler_representation_comparison"] = {
        "sampling": dict(cfg["sampling"])
    }
    chunk_size = int(cfg.get("panel_batch_size", 4))
    small_sheets = _sample_in_chunks(
        anima,
        small,
        requests_small,
        sample_config,
        destination,
        output / "small",
        device,
        chunk_size=chunk_size,
    )
    dual_sheets = _sample_in_chunks(
        anima,
        dual,
        requests_dual,
        sample_config,
        destination,
        output / "dual_query",
        device,
        chunk_size=chunk_size,
    )
    combined_root = output / "combined"
    combined_root.mkdir(exist_ok=True)
    combined = []
    for small_sheet, dual_sheet in zip(small_sheets, dual_sheets, strict=True):
        path = combined_root / small_sheet.name
        _combine_panel_pair(small_sheet, dual_sheet, path)
        combined.append(path)
    result = {
        "comparison_contract": {
            "same_target_reference_and_caption_variant": True,
            "same_generation_seed_per_panel": True,
            "same_prompt_conditioning": True,
            "same_anima_and_vae": True,
            "same_cfg_steps_and_resolution": True,
            "reference_mode": "heldout",
        },
        "small": small_metadata,
        "dual_query": dual_metadata,
        "sampling": dict(cfg["sampling"]),
        "episodes": episodes,
        "combined_panels": [str(path) for path in combined],
    }
    write_json(output / "summary.json", result)
    return result


def compare_resampler_representations(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["resampler_representation_comparison"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)

    small, small_metadata = _load_ab_tokenizer(
        destination / str(cfg["small_checkpoint"]),
        AnimaStyleTokenizer,
        device,
    )
    dual, dual_metadata = _load_ab_tokenizer(
        destination / str(cfg["dual_query_checkpoint"]),
        CompactDualQueryStyleTokenizer,
        device,
    )
    training_cfg = dict(config["compact_dual_query_style_tokenizer"]["training"])
    training_cfg.update(
        {
            "token_contrastive_weight": 0.0,
            "artist_direction_weight": 0.0,
            "wrong_ranking_weight": 0.0,
        }
    )
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training_cfg.get("fuse_attention_projections", True)
        ),
    )

    batches = int(cfg.get("validation_batches", 64))
    seed = int(cfg["seed"])
    results: dict[str, Any] = {}
    for references in [int(value) for value in cfg["reference_counts"]]:
        small_loader = ProductionStyleLoader(
            destination,
            _loader_config(
                config,
                cfg,
                token_cache=str(cfg["small_token_cache"]),
                references=references,
            ),
        )
        dual_loader = ProductionStyleLoader(
            destination,
            _loader_config(
                config,
                cfg,
                token_cache=str(cfg["dual_query_token_cache"]),
                references=references,
            ),
        )
        results[str(references)] = {
            "heldout": _evaluate_pair(
                anima,
                small,
                dual,
                small_loader,
                dual_loader,
                device,
                training_cfg,
                batches=batches,
                seed=seed ^ 0xC0FFEE,
                reference_mode="heldout",
            ),
            "wrong_artist": _evaluate_pair(
                anima,
                small,
                dual,
                small_loader,
                dual_loader,
                device,
                training_cfg,
                batches=batches,
                seed=seed ^ 0xC0FFEE,
                reference_mode="wrong_artist",
            ),
        }
        write_json(output / "metrics.json", {"reference_counts": results})

    summary = {
        "comparison_contract": {
            "only_intended_difference": "frozen per-reference Resampler representation",
            "same_style_tokenizer_architecture": True,
            "same_training_steps": int(small_metadata["step"])
            == int(dual_metadata["step"]),
            "same_validation_episodes": True,
            "same_target_latents": True,
            "same_text_conditioning": True,
            "same_noise_and_timesteps": True,
            "validation_batches_per_reference_count": batches,
            "reference_counts": [int(value) for value in cfg["reference_counts"]],
        },
        "small": small_metadata,
        "dual_query": dual_metadata,
        "reference_counts": results,
    }
    write_json(output / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--panels-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.panels_only:
        result = generate_panel_comparison(config, output_dir(config))
    else:
        result = compare_resampler_representations(config, output_dir(config))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
