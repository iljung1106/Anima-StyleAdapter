"""Measure where external-reference identity is lost: tokens or Reader memory."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from anima_style_data.config import load_config, output_dir
from anima_style_data.dual_query_style_tokenizer import CachedTeacherReferenceLoader
from anima_style_data.io import read_records
from anima_style_data.kv_activation_generator import (
    _load_reader,
    _materialize_reference_token_bank,
    _resolved_experiment_config,
)


def retrieval_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = F.normalize(left.float().flatten(1), dim=-1)
    right = F.normalize(right.float().flatten(1), dim=-1)
    cosine = left @ right.T
    labels = torch.arange(len(left), device=left.device)
    wrong = cosine.masked_fill(torch.eye(len(left), device=left.device, dtype=torch.bool), -torch.inf)
    return {
        "same_subset_cosine": float(cosine.diagonal().mean()),
        "hardest_wrong_cosine": float(wrong.max(dim=1).values.mean()),
        "correct_minus_hardest_wrong": float((cosine.diagonal() - wrong.max(dim=1).values).mean()),
        "retrieval_accuracy": float((cosine.argmax(dim=1) == labels).float().mean()),
        "positive_pairwise_cosine": float(F.relu(left @ left.T)[~torch.eye(len(left), device=left.device, dtype=torch.bool)].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anima500k-human.yaml")
    parser.add_argument("--experiment", default="kv_reference_expert_combined_rms_1500")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-checkpoint")
    parser.add_argument("--styles", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    destination = output_dir(config)
    cfg = _resolved_experiment_config(config, args.experiment)
    external_cfg = cfg["training"]["multi_domain_distillation"]["external_lora"]
    reference_root = destination / str(external_cfg["reference_cache"])
    style_ids = list(dict.fromkeys(str(row["style_id"]) for row in read_records(reference_root / "manifest.parquet")))
    random.Random(20260910).shuffle(style_ids)
    style_ids = style_ids[: min(args.styles, len(style_ids))]

    loader = CachedTeacherReferenceLoader(
        reference_root,
        split="train",
        style_ids=style_ids,
        batch_size=16,
        references=4,
        seed=20260910,
        token_lru_shards=8,
        strict_style_ids=True,
    )
    bank = _materialize_reference_token_bank(
        loader,
        style_ids,
        references=4,
        seed=20260910,
        chunk_size=16,
        device=args.device,
    )
    checkpoint_path = destination / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.compare_checkpoint:
        comparison_path = destination / args.compare_checkpoint
        comparison = torch.load(comparison_path, map_location="cpu", weights_only=False)
        baseline = comparison.get("ema_reader", comparison["reader"])
        current = checkpoint["reader"]
        differences = torch.cat(
            [
                (current[name].float() - baseline[name].float()).abs().flatten()
                for name in current
            ]
        )
        print(
            "reader_parameter_change",
            {
                "baseline": str(comparison_path),
                "changed_fraction": float((differences > 0).float().mean()),
                "mean_absolute": float(differences.mean()),
                "maximum_absolute": float(differences.max()),
            },
        )
    raw_left = bank[:, :2].mean(dim=1)
    raw_right = bank[:, 2:].mean(dim=1)
    mask = torch.ones(len(style_ids), 2, device=args.device, dtype=torch.bool)

    print(f"styles={len(style_ids)} checkpoint={checkpoint_path}")
    print("raw_tokens", retrieval_metrics(raw_left, raw_right))
    for state_name in ("reader", "ema_reader"):
        if state_name not in checkpoint:
            continue
        reader = _load_reader(config, destination, cfg, args.device)
        reader.load_state_dict(checkpoint[state_name], strict=True)
        reader.requires_grad_(False).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            memory_left = reader(bank[:, :2], mask).tokens
            memory_right = reader(bank[:, 2:], mask).tokens
        print(state_name, retrieval_metrics(memory_left, memory_right))


if __name__ == "__main__":
    main()
