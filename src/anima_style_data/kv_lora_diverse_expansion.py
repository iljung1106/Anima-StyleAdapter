"""Expand the K/V-LoRA dictionary with visually farthest artist anchors."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .artist_lora_teachers import (
    _load_plan,
    _selection_signature,
    select_artist_lora_plans,
    train_artist_lora_teachers,
)
from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json
from .lora_oracle_bootstrap import (
    _materialize_reader_code_bank,
    _oracle_detail_config,
)


def _target_teacher_config(
    config: dict[str, Any],
    *,
    expansion_config_key: str,
    teacher_config_key: str,
) -> dict[str, Any]:
    cfg = dict(config[expansion_config_key])
    target = copy.deepcopy(config[teacher_config_key])
    target.update({
        "output_directory": str(cfg["output_directory"]),
        "reuse_completed_prefix_directory": str(cfg["base_lora_directory"]),
        "artist_count": int(cfg["artist_count"]),
    })
    wandb = dict(target["training"].get("wandb", {}))
    wandb.update(dict(cfg.get("wandb", {})))
    target["training"]["wandb"] = wandb
    return target


@torch.no_grad()
def _prepare_diverse_lora_expansion(
    config: dict[str, Any],
    destination: Path,
    *,
    expansion_config_key: str,
    teacher_config_key: str,
) -> dict[str, Any]:
    cfg = dict(config[expansion_config_key])
    target_cfg = _target_teacher_config(
        config,
        expansion_config_key=expansion_config_key,
        teacher_config_key=teacher_config_key,
    )
    output = destination / str(target_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    signature = _selection_signature(target_cfg)
    if plan_path.exists():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        if payload.get("signature") == signature:
            return {
                **payload["summary"],
                "output": str(output),
                "plan": str(plan_path),
                "reused": True,
            }
        if (output / "weights").exists() and any((output / "weights").iterdir()):
            raise RuntimeError("Diverse LoRA plan changed after weights were written")

    latent_rows = read_records(
        destination / str(target_cfg["latent_cache"]) / "manifest.parquet"
    )
    text_rows = read_records(
        destination / str(target_cfg["text_cache"]) / "manifest.parquet"
    )
    candidate_cfg = copy.deepcopy(target_cfg)
    candidate_cfg["artist_count"] = int(cfg["candidate_artists"])
    candidates, candidate_summary = select_artist_lora_plans(
        latent_rows, text_rows, candidate_cfg
    )
    base_root = destination / str(cfg["base_lora_directory"])
    _, base_plans = _load_plan(base_root)
    base_count = len(base_plans)
    if int(target_cfg["artist_count"]) <= base_count:
        raise ValueError("Expanded artist count must exceed the base dictionary")
    if candidates[:base_count] != base_plans:
        raise RuntimeError("Candidate ordering no longer preserves the base LoRA plan")

    device = str(cfg.get("device", "cuda"))
    reader_state = torch.load(
        destination / str(cfg["reader_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    detail_cfg = _oracle_detail_config(
        config, dict(config["kv_lora_oracle_bootstrap"])
    )
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(
        device=device, dtype=torch.bfloat16
    )
    reader.load_state_dict(reader_state["reader"], strict=True)
    reader.requires_grad_(False).eval()
    del reader_state
    style_ids = [plan.style_id for plan in candidates]
    allowed_ids = {
        image_id for plan in candidates for image_id in plan.train_ids
    }
    references = int(cfg.get("references", 4))
    loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split=str(target_cfg.get("split", "train")),
        style_ids=style_ids,
        batch_size=int(cfg.get("materialization_artist_chunk", 16)),
        references=references,
        seed=int(cfg.get("seed", 20260824)),
        token_lru_shards=int(cfg.get("token_lru_shards", 8)),
        strict_style_ids=True,
        allowed_image_ids=allowed_ids,
    )
    codes, counts = _materialize_reader_code_bank(
        reader,
        loader,
        style_ids,
        reference_images=references,
        seed=int(cfg.get("seed", 20260824)) ^ 0x44495645,
        device=device,
        style_chunk_size=int(cfg.get("materialization_artist_chunk", 16)),
    )
    del reader
    active = torch.nonzero(counts == references).flatten()
    if not len(active):
        raise RuntimeError(f"No {references}-reference Reader view was materialized")
    anchors = codes[:, active].float().mean(dim=1)
    common = anchors[:base_count].mean(dim=0, keepdim=True)
    features = F.normalize((anchors - common).flatten(1), dim=-1).to(torch.bfloat16)
    del codes, anchors, common

    maximum_similarity = torch.full(
        (len(candidates),), -torch.inf, device=device, dtype=torch.float32
    )
    chunk = int(cfg.get("similarity_chunk", 32))
    for start in range(0, base_count, chunk):
        similarity = features @ features[start : start + chunk].t()
        maximum_similarity = torch.maximum(
            maximum_similarity, similarity.float().amax(dim=1)
        )
    selectable = torch.ones(len(candidates), device=device, dtype=torch.bool)
    selectable[:base_count] = False
    before_values = maximum_similarity[selectable].clone()
    selected = list(range(base_count))
    selected_coverages: list[float] = []
    additional = int(target_cfg["artist_count"]) - base_count
    for _ in range(additional):
        score = maximum_similarity.masked_fill(~selectable, torch.inf)
        index = int(score.argmin().item())
        selected.append(index)
        selected_coverages.append(float(maximum_similarity[index]))
        selectable[index] = False
        similarity = (features @ features[index]).float()
        maximum_similarity = torch.maximum(maximum_similarity, similarity)
    after_values = maximum_similarity[selectable]
    selected_plans = [
        replace(candidates[candidate_index], index=output_index)
        for output_index, candidate_index in enumerate(selected)
    ]
    coverage = {
        "candidate_artists": len(candidates),
        "base_artists": base_count,
        "additional_artists": additional,
        "nearest_cosine_before_mean": float(before_values.mean()),
        "nearest_cosine_before_p95": float(torch.quantile(before_values, 0.95)),
        "worst_covered_before": float(before_values.min()),
        "nearest_cosine_after_mean": float(after_values.mean()),
        "nearest_cosine_after_p95": float(torch.quantile(after_values, 0.95)),
        "worst_covered_after": float(after_values.min()),
        "selected_nearest_cosine_mean": float(
            torch.tensor(selected_coverages).mean()
        ),
        "selected_candidate_indices": selected[base_count:],
        "selected_style_ids": [plan.style_id for plan in selected_plans[base_count:]],
    }
    save_file(
        {
            "candidate_indices": torch.tensor(selected, dtype=torch.int64),
            "selected_nearest_cosine": torch.tensor(
                selected_coverages, dtype=torch.float32
            ),
        },
        output / "diverse_selection.safetensors",
    )
    summary = {
        **candidate_summary,
        "artists": len(selected_plans),
        "selection": "four-reference Reader-code cosine k-center",
        "coverage": coverage,
        "base_lora_directory": str(base_root),
    }
    payload = {
        "signature": signature,
        "summary": summary,
        "artists": [asdict(plan) for plan in selected_plans],
    }
    write_json(plan_path, payload)
    write_json(output / "diverse_selection.json", coverage)
    return {
        **summary,
        "output": str(output),
        "plan": str(plan_path),
        "reused": False,
    }


def _train_diverse_lora_expansion(
    config: dict[str, Any],
    destination: Path,
    *,
    expansion_config_key: str,
    teacher_config_key: str,
) -> dict[str, Any]:
    _prepare_diverse_lora_expansion(
        config,
        destination,
        expansion_config_key=expansion_config_key,
        teacher_config_key=teacher_config_key,
    )
    effective = copy.deepcopy(config)
    effective[teacher_config_key] = _target_teacher_config(
        config,
        expansion_config_key=expansion_config_key,
        teacher_config_key=teacher_config_key,
    )
    return train_artist_lora_teachers(
        effective,
        destination,
        config_key=teacher_config_key,
    )


def prepare_diverse_kv_lora_expansion(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _prepare_diverse_lora_expansion(
        config,
        destination,
        expansion_config_key="artist_kv_lora_diverse_expansion",
        teacher_config_key="artist_kv_lora_teachers",
    )


def train_diverse_kv_lora_expansion(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _train_diverse_lora_expansion(
        config,
        destination,
        expansion_config_key="artist_kv_lora_diverse_expansion",
        teacher_config_key="artist_kv_lora_teachers",
    )


def prepare_diverse_artist_lora_expansion(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _prepare_diverse_lora_expansion(
        config,
        destination,
        expansion_config_key="artist_lora_diverse_expansion",
        teacher_config_key="artist_lora_teachers",
    )


def train_diverse_artist_lora_expansion(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _train_diverse_lora_expansion(
        config,
        destination,
        expansion_config_key="artist_lora_diverse_expansion",
        teacher_config_key="artist_lora_teachers",
    )
