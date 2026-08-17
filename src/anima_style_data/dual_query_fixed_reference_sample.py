from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch

from .compact_dual_query_style_tokenizer import CompactDualQueryStyleTokenizer
from .config import load_config, output_dir
from .dual_query_external_samples import load_dual_query_external_sample
from .dual_query_style_tokenizer import DualQuerySetStyleTokenizer
from .external_style_tokenizer_sheet import generate_live_external_style_sample
from .hierarchical_dual_query_style_tokenizer import HierarchicalDualQueryStyleTokenizer
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


def _build_tokenizer(cfg: dict[str, Any], device: str) -> torch.nn.Module:
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "flat_set"))
    model_types = {
        "flat_set": DualQuerySetStyleTokenizer,
        "hierarchical": HierarchicalDualQueryStyleTokenizer,
        "compact": CompactDualQueryStyleTokenizer,
    }
    try:
        model_type = model_types[architecture]
    except KeyError as error:
        raise ValueError(
            f"Unknown Dual-query StyleTokenizer architecture {architecture!r}"
        ) from error
    return model_type(**model_cfg).to(device)


def generate_fixed_reference_from_checkpoint(
    config: dict[str, Any],
    destination: Path,
    *,
    section: str,
    checkpoint: Path | None,
    step: int,
    style_multiplier: float,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[section])
    effective_config = copy.deepcopy(config)
    effective_config["dual_query_style_tokenizer"] = cfg
    device = str(cfg["training"].get("device", "cuda"))
    if checkpoint is None:
        checkpoint = (
            destination
            / str(cfg["output_directory"])
            / "checkpoints"
            / f"step-{step:07d}.pt"
        )
    elif not checkpoint.is_absolute():
        checkpoint = destination / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("step", -1)) != step:
        raise RuntimeError(
            f"Checkpoint step {state.get('step')} does not match requested step {step}"
        )

    tokenizer = _build_tokenizer(cfg, device)
    tokenizer.load_state_dict(state["tokenizer"], strict=True)
    tokenizer.requires_grad_(False).eval()
    prepared = load_dual_query_external_sample(effective_config, destination)
    anima = _resolve_anima_model(effective_config, destination, device)
    anima.requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            cfg["training"].get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    try:
        result = generate_live_external_style_sample(
            prepared,
            effective_config,
            destination,
            anima,
            tokenizer,
            destination / str(cfg["output_directory"]),
            device,
            step,
            style_multiplier=style_multiplier,
        )
    finally:
        del anima, tokenizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    result.update(
        {
            "checkpoint": str(checkpoint),
            "config_section": section,
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--section", default="aligned_compact_dual_query_style_tokenizer"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int, default=10_000)
    parser.add_argument("--style-multiplier", type=float, default=2.0)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    generate_fixed_reference_from_checkpoint(
        config,
        output_dir(config),
        section=args.section,
        checkpoint=checkpoint,
        step=args.step,
        style_multiplier=args.style_multiplier,
    )


if __name__ == "__main__":
    main()
