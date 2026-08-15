from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn

from .config import load_config, output_dir
from .io import write_json
from .pure_token_injection import _evaluate_controlled_artist_consistency
from .query_style_tokenizer import (
    QueryStyleTokenizerV2,
    _select_sample_episodes,
)
from .style_tokenizer import (
    AnimaStyleTokenizer,
    _evaluate,
    _sample_tokenizer,
    _tokenizer_loader_config,
)
from .style_transfer import (
    ProductionStyleLoader,
    _optimize_frozen_anima,
    _resolve_anima_model,
)


class _TokenView(nn.Module):
    """Present either tokenizer architecture as a token-only module."""

    def __init__(self, tokenizer: nn.Module) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @property
    def log_output_rms(self) -> torch.Tensor:
        return self.tokenizer.log_output_rms

    def forward(
        self, references: torch.Tensor, reference_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.tokenizer(references, reference_mask)
        return output if isinstance(output, torch.Tensor) else output.tokens


class _ControlledOutputView(nn.Module):
    """Adapt a token-only module to the controlled evaluator's output contract."""

    def __init__(self, tokenizer: nn.Module) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self, references: torch.Tensor, reference_mask: torch.Tensor
    ) -> SimpleNamespace:
        return SimpleNamespace(tokens=self.tokenizer(references, reference_mask))


def _load_checkpoint_tokenizer(
    checkpoint_path: Path,
    architecture: type[nn.Module],
    device: str,
) -> tuple[_TokenView, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = architecture(**dict(state["config"]["model"]))
    tokenizer.load_state_dict(state["tokenizer"], strict=True)
    tokenizer.to(device).eval()
    metadata = {
        "path": str(checkpoint_path),
        "step": int(state["step"]),
        "trainable_parameters": sum(
            parameter.numel() for parameter in tokenizer.parameters()
        ),
        "output_tokens": int(state["config"]["model"]["output_tokens"]),
    }
    del state
    gc.collect()
    return _TokenView(tokenizer), metadata


def _common_validation_loaders(
    config: dict[str, Any], destination: Path, comparison_cfg: dict[str, Any]
) -> tuple[ProductionStyleLoader, ProductionStyleLoader, ProductionStyleLoader]:
    source_section = str(
        comparison_cfg.get("loader_config_section", "pure_token_style_tokenizer_v2")
    )
    source_cfg = dict(config[source_section])
    validation_cfg = _tokenizer_loader_config(
        config,
        source_cfg,
        split=str(source_cfg.get("validation_split", "validation")),
    )
    train_cfg = _tokenizer_loader_config(
        config,
        source_cfg,
        split=str(source_cfg.get("train_split", "train")),
    )
    train_cfg.pop("reference_curriculum", None)
    train_cfg["self_reference_target_images_per_style"] = 0

    consistency_cfg = dict(source_cfg.get("consistency_evaluation", {}))
    references_per_view = int(consistency_cfg.get("references_per_view", 4))
    controlled_cfg = dict(validation_cfg)
    controlled_cfg.update({
        "batch_size": int(consistency_cfg.get("artists", 8)),
        "min_references": references_per_view * 2,
        "max_references": references_per_view * 2,
        "artist_balanced": True,
        "gradient_accumulation_steps": 1,
    })
    return (
        ProductionStyleLoader(destination, train_cfg),
        ProductionStyleLoader(destination, validation_cfg),
        ProductionStyleLoader(destination, controlled_cfg),
    )


def _evaluate_one(
    anima: nn.Module,
    tokenizer: _TokenView,
    validation_loader: ProductionStyleLoader,
    controlled_loader: ProductionStyleLoader,
    device: str,
    training_cfg: dict[str, Any],
    consistency_cfg: dict[str, Any],
    *,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    result = {
        "validation_self": _evaluate(
            anima,
            tokenizer,
            validation_loader,
            device,
            training_cfg,
            batches=batches,
            seed=seed ^ 0xBEEF,
            reference_mode="self",
        ),
        "validation_heldout": _evaluate(
            anima,
            tokenizer,
            validation_loader,
            device,
            training_cfg,
            batches=batches,
            seed=seed ^ 0xC0FFEE,
            reference_mode="heldout",
        ),
        "validation_wrong_artist": _evaluate(
            anima,
            tokenizer,
            validation_loader,
            device,
            training_cfg,
            batches=batches,
            seed=seed ^ 0xC0FFEE,
            reference_mode="wrong_artist",
        ),
    }
    result["correct_vs_wrong_paired_advantage"] = (
        result["validation_heldout"]["paired_flow_improvement"]
        - result["validation_wrong_artist"]["paired_flow_improvement"]
    )
    result["controlled_artist_consistency"] = (
        _evaluate_controlled_artist_consistency(
            anima,
            _ControlledOutputView(tokenizer),
            controlled_loader,
            device,
            consistency_cfg,
            seed=seed ^ 0xA77157,
        )
    )
    return result


def _combine_sample_sheets(
    small_paths: list[Path], current_paths: list[Path], output: Path
) -> list[Path]:
    combined = output / "samples" / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    paths = []
    for small, current in zip(small_paths, current_paths, strict=True):
        with Image.open(small) as small_image, Image.open(current) as current_image:
            width = small_image.width + current_image.width
            height = max(small_image.height, current_image.height) + 36
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 10), "small 16-token / 9.48M", fill="black")
            draw.text(
                (small_image.width + 8, 10),
                "current 32-token / 76.03M",
                fill="black",
            )
            canvas.paste(small_image.convert("RGB"), (0, 36))
            canvas.paste(current_image.convert("RGB"), (small_image.width, 36))
            path = combined / small.name
            canvas.save(path)
            paths.append(path)
    return paths


def compare_style_tokenizers(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["style_tokenizer_checkpoint_comparison"])
    device = str(cfg.get("device", "cuda"))
    current_section = str(
        cfg.get("loader_config_section", "pure_token_style_tokenizer_v2")
    )
    current_cfg = dict(config[current_section])
    training_cfg = dict(current_cfg["training"])
    # The common evaluator measures only the flow behavior. Training-only
    # auxiliaries must not differ between architectures during comparison.
    training_cfg.update({
        "token_contrastive_weight": 0.0,
        "artist_direction_weight": 0.0,
    })
    consistency_cfg = dict(current_cfg["consistency_evaluation"])
    seed = int(cfg.get("seed", 20260822))
    batches = int(cfg.get("validation_batches", 32))
    output = destination / str(cfg.get("output_directory", "style_tokenizer_4k_comparison"))
    output.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, controlled_loader = _common_validation_loaders(
        config, destination, cfg
    )
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training_cfg.get("fuse_attention_projections", True)
        ),
    )
    small, small_metadata = _load_checkpoint_tokenizer(
        destination / str(cfg["small_checkpoint"]),
        AnimaStyleTokenizer,
        device,
    )
    current, current_metadata = _load_checkpoint_tokenizer(
        destination / str(cfg["current_checkpoint"]),
        QueryStyleTokenizerV2,
        device,
    )

    metrics = {
        "comparison": {
            "seed": seed,
            "validation_batches": batches,
            "same_validation_loader": True,
            "same_noisy_latents_and_timesteps": True,
        },
        "small": {
            "checkpoint": small_metadata,
            **_evaluate_one(
                anima,
                small,
                validation_loader,
                controlled_loader,
                device,
                training_cfg,
                consistency_cfg,
                batches=batches,
                seed=seed,
            ),
        },
        "current": {
            "checkpoint": current_metadata,
            **_evaluate_one(
                anima,
                current,
                validation_loader,
                controlled_loader,
                device,
                training_cfg,
                consistency_cfg,
                batches=batches,
                seed=seed,
            ),
        },
    }
    write_json(output / "metrics.json", metrics)

    sample_count = int(cfg.get("sample_count", 4))
    requests = [
        (f"validation-{index}", validation_loader, episode, "heldout")
        for index, episode in enumerate(
            _select_sample_episodes(validation_loader, sample_count)
        )
    ]
    sample_config = dict(config)
    sample_config["style_tokenizer_checkpoint_comparison"] = {
        "sampling": dict(current_cfg["sampling"])
    }
    small_sheets, vae = _sample_tokenizer(
        anima,
        small,
        requests,
        sample_config,
        destination,
        output / "small",
        device,
        4000,
        None,
        config_section="style_tokenizer_checkpoint_comparison",
    )
    current_sheets, _ = _sample_tokenizer(
        anima,
        current,
        requests,
        sample_config,
        destination,
        output / "current",
        device,
        4000,
        vae,
        config_section="style_tokenizer_checkpoint_comparison",
    )
    combined = _combine_sample_sheets(small_sheets, current_sheets, output)
    metrics["samples"] = [str(path) for path in combined]
    write_json(output / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = compare_style_tokenizers(config, output_dir(config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
