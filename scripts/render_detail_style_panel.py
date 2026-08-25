"""Render the deterministic train/validation panel from an existing checkpoint.

This is a sample-only entry point: it loads neither teacher banks nor optimizer
state and cannot advance training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from anima_style_data.config import load_config
from anima_style_data.detail_style_training import (
    _build_style_adapter,
    _loader_config,
    _merge_detail_style_config,
    _training_loader,
)
from anima_style_data.detail_style_cross_attention import DetailPreservingTypedSlotReader
from anima_style_data.global_query_style_tokenizer import (
    MultiPromptDualQueryCachedStyleLoader,
)
from anima_style_data.query_style_tokenizer import (
    _sample_query_style_tokenizer,
    _select_sample_episodes,
)
from anima_style_data.same_q_style_adapter import attach_same_q_style_adapter
from anima_style_data.style_transfer import _optimize_frozen_anima, _resolve_anima_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--style-cfg", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.style_cfg <= 0:
        raise ValueError("--style-cfg must be positive")

    config = load_config(args.config)
    cfg = _merge_detail_style_config(config, args.config_key)
    config = copy.deepcopy(config)
    config["detail_preserving_style_cross_attention"] = cfg
    cfg["sampling"]["style_cfg"] = float(args.style_cfg)
    device = str(cfg["training"].get("device", "cuda"))

    train_cfg = _loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    validation_cfg = _loader_config(
        config, cfg, split=str(cfg.get("validation_split", "validation"))
    )
    # Panel rendering touches only eight episodes. Avoid the training-time RAM
    # preload of the complete frozen reference-token cache.
    train_cfg["ram_resident_tokens"] = False
    validation_cfg["ram_resident_tokens"] = False
    sample_train_loader, _ = _training_loader(args.destination, cfg, train_cfg)
    validation_loader = MultiPromptDualQueryCachedStyleLoader(
        args.destination, validation_cfg
    )

    anima = _resolve_anima_model(config, args.destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            cfg["training"].get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(cfg).to(device).eval()
    attach_same_q_style_adapter(anima, adapter)

    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = args.destination / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reader.load_state_dict(state["reader"], strict=True)
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    step = int(state.get("step", 0))
    del state

    sample_seed = int(cfg.get("sampling", {}).get("seed", cfg.get("seed", 0) ^ 0x5A17))
    requests = [
        ("train", sample_train_loader, episode, sample_seed + index * 10_007)
        for index, episode in enumerate(
            _select_sample_episodes(sample_train_loader, 4)
        )
    ] + [
        (
            "validation",
            validation_loader,
            episode,
            sample_seed + (index + 4) * 10_007,
        )
        for index, episode in enumerate(_select_sample_episodes(validation_loader, 4))
    ]
    output = args.output or args.destination / str(cfg["output_directory"])
    records, _ = _sample_query_style_tokenizer(
        anima,
        adapter,
        reader,
        requests,
        config,
        args.destination,
        output,
        device,
        step,
        None,
        config_section="detail_preserving_style_cross_attention",
    )
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "step": step,
        "style_cfg": args.style_cfg,
        "panels": [str(path) for _, path in records],
    }, indent=2))


if __name__ == "__main__":
    main()
