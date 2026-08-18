from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .dual_query_style_tokenizer import (
    CachedTeacherReferenceLoader,
    DualQueryCachedStyleLoader,
    DualQuerySetStyleTokenizer,
)
from .hierarchical_dual_query_style_tokenizer import (
    HierarchicalDualQueryStyleTokenizer,
)
from .compact_dual_query_style_tokenizer import CompactDualQueryStyleTokenizer
from .global_query_style_tokenizer import (
    GlobalQueryMemoryStyleTokenizer,
    MultiPromptDualQueryCachedStyleLoader,
    SlotPreservingGlobalQueryStyleTokenizer,
    attention_map_diversity_loss,
    reference_conditioned_diversity_loss,
)
from .typed_multi_descriptor_style_tokenizer import (
    TypedMultiDescriptorCompactStyleTokenizer,
)
from .dual_query_external_samples import load_dual_query_external_sample
from .external_style_tokenizer_sheet import generate_live_external_style_sample
from .io import write_json
from .native_centered_teacher import NativeCenteredTeacherBank
from .pure_token_injection import (
    _aligned_velocity_losses,
    _coefficient_floor,
    _evaluate_controlled_artist_consistency,
    _evaluate,
    _reference_batch,
    _sample_panels,
)
from .query_style_tokenizer import (
    _artist_contrastive_loss,
    _linear_weight,
    _reference_inputs,
    _select_sample_episodes,
    _slot_diversity_loss,
)
from .style_tokenizer import (
    _artist_direction_loss,
    _flow_metrics,
    _split_reference_views,
    _style_token_contrastive_loss,
    insert_style_tokens,
)
from .style_transfer import (
    _learning_rate_multiplier,
    _optimize_frozen_anima,
    _resolve_anima_model,
    _sample_flow_timesteps,
)


def _loader_config(
    config: dict[str, Any], cfg: dict[str, Any], *, split: str
) -> dict[str, Any]:
    result = dict(config["style_transfer"]["loader"])
    result.update(dict(cfg.get("loader", {})))
    result["split"] = split
    result["resampler_token_cache"] = str(cfg["cache"]["output_directory"])
    if split == str(cfg.get("train_split", "train")):
        if bool(cfg["training"].get("pilot_enabled", False)):
            result["reference_curriculum"] = {}
            result["pilot_reference_schedule"] = copy.deepcopy(
                cfg["training"]["reference_schedule"]
            )
        else:
            result["reference_curriculum"] = dict(cfg["training"]["curriculum"])
        result["gradient_accumulation_steps"] = int(
            cfg["training"].get("gradient_accumulation_steps", 1)
        )
    else:
        result["reference_curriculum"] = {}
        result["self_reference_target_images_per_style"] = 0
    return result


def _with_excluded_style_ids(
    loader_cfg: dict[str, Any], style_ids: list[str]
) -> dict[str, Any]:
    """Return a loader-local exclusion list without mutating shared config."""
    result = dict(loader_cfg)
    result["excluded_style_ids"] = sorted(
        set(result.get("excluded_style_ids", [])) | set(style_ids)
    )
    return result


def _dual_domain_human_splits(
    cfg: dict[str, Any], human_cfg: dict[str, Any]
) -> tuple[str, str]:
    return (
        str(human_cfg.get("source_split", cfg.get("train_split", "train"))),
        str(cfg.get("validation_split", "validation")),
    )


def _cache_summary(destination: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = destination / str(cfg["cache"]["output_directory"]) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = (84, 80, 4, 1024)
    actual = (
        int(summary.get("slots", 0)),
        int(summary.get("query_slots", 0)),
        int(summary.get("artist_summary_slots", 0)),
        int(summary.get("style_dim", 0)),
    )
    if actual != expected:
        raise RuntimeError(f"Unexpected dual-query token cache contract {actual}")
    if str(summary.get("resampler_checkpoint")) != str(cfg["resampler_checkpoint"]):
        raise RuntimeError("Style token cache was created by another Resampler checkpoint")
    return summary


def _save_state(
    path: Path,
    *,
    step: int,
    tokenizer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    cache_summary: dict[str, Any],
    trainer_state: dict[str, Any] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": int(step),
            "tokenizer": {
                key: value.detach().cpu() for key, value in tokenizer.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "resampler_cache": cache_summary,
            "trainer_state": dict(trainer_state or {}),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _subset_consistency(
    tokenizer: torch.nn.Module,
    batch: dict[str, Any],
    device: str,
) -> torch.Tensor:
    references, mask = _reference_inputs(batch, device, "heldout")
    positions = torch.arange(mask.shape[1], device=device).unsqueeze(0)
    first_mask = mask & positions.remainder(2).eq(0)
    second_mask = mask & positions.remainder(2).eq(1)
    # A one-reference row has no second view; use the same reference so it
    # contributes zero without a Python-side split or CUDA synchronization.
    second_mask = torch.where(mask.sum(dim=1, keepdim=True) > 1, second_mask, first_mask)
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        first = tokenizer(references, first_mask).tokens
        second = tokenizer(references, second_mask).tokens
    return (
        1.0
        - F.cosine_similarity(first.float(), second.float(), dim=-1)
    ).mean()


def _linear_ramp(
    step: int, *, start_step: int, end_step: int, start: float, end: float
) -> float:
    if end_step <= start_step:
        return float(end)
    progress = min(1.0, max(0.0, (step - start_step) / (end_step - start_step)))
    return float(start) + progress * (float(end) - float(start))


def _scheduled_teacher_every(step: int, training: dict[str, Any]) -> int:
    schedule = training.get("dual_domain_teacher_schedule")
    if not schedule:
        teacher = dict(training.get("dual_domain_teacher", {}))
        return max(1, int(teacher.get("every", 1)))
    for phase in schedule:
        if step <= int(phase["end_step"]):
            return max(1, int(phase["every"]))
    return max(1, int(schedule[-1]["every"]))


def _scheduled_teacher_gradient_scale(
    cadence: int, training: dict[str, Any]
) -> float:
    """Keep expected teacher pressure constant when updates are subsampled."""

    return (
        float(cadence)
        if bool(training.get("dual_domain_teacher_scale_by_cadence", False))
        else 1.0
    )


def _scheduled_teacher_reference_count(
    training_by_domain: dict[str, dict[str, Any]],
    available_by_domain: dict[str, int],
    update_index: int,
) -> int:
    """Select one shared reference count for a fused teacher update."""

    schedules: set[tuple[int, ...]] = set()
    for domain, training in training_by_domain.items():
        available = int(available_by_domain[domain])
        raw = training.get("native_teacher_reference_counts")
        schedule = tuple(int(value) for value in (raw or [available]))
        if not schedule or min(schedule) <= 0 or max(schedule) > available:
            raise ValueError(
                f"Invalid {domain} teacher reference schedule {schedule}; "
                f"available={available}"
            )
        schedules.add(schedule)
    if len(schedules) != 1:
        raise ValueError("Fused teacher domains require one reference-count schedule")
    schedule = next(iter(schedules))
    return schedule[int(update_index) % len(schedule)]


def _pilot_stage(step: int, training: dict[str, Any]) -> dict[str, Any]:
    schedule = list(training.get("reference_schedule", []))
    if not schedule:
        raise ValueError("pilot reference_schedule is required")
    stage = next(
        (item for item in schedule if step <= int(item["end_step"])),
        schedule[-1],
    )
    return dict(stage)


def _pilot_alignment_state(step: int, training: dict[str, Any]) -> dict[str, float]:
    exact_end = int(training.get("exact_self_end_step", 500))
    if step <= exact_end:
        return {
            "normalized_weight": float(
                training.get("exact_normalized_residual_weight", 0.05)
            ),
            "floor_weight": float(training.get("exact_aligned_floor_weight", 0.25)),
            "coefficient_floor": _coefficient_floor(
                step, {**training, "pilot_enabled": True}
            ),
            "bounded_min": float(training.get("exact_bounded_min", 0.08)),
            "bounded_max": float(training.get("exact_bounded_max", 1.25)),
            "bounded_weight": float(training.get("exact_bounded_weight", 0.05)),
        }
    return {
        "normalized_weight": float(
            training.get("heldout_normalized_residual_weight", 0.015)
        ),
        "floor_weight": float(training.get("heldout_aligned_floor_weight", 0.075)),
        "coefficient_floor": _coefficient_floor(
            step, {**training, "pilot_enabled": True}
        ),
        "bounded_min": float(training.get("heldout_bounded_min", 0.05)),
        "bounded_max": float(training.get("heldout_bounded_max", 1.25)),
        "bounded_weight": float(training.get("heldout_bounded_weight", 0.015)),
    }


def _bounded_aligned_effect_loss(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
    orthogonal_maximum: float,
    orthogonal_weight: float,
    scale_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep useful target projection in range without rewarding raw output norm."""

    dimensions = tuple(range(1, prediction.ndim))
    delta = prediction - base_prediction
    desired = target - base_prediction
    desired_power = desired.square().mean(dim=dimensions).clamp_min(
        float(scale_floor) ** 2
    )
    coefficient = (delta * desired.detach()).mean(dim=dimensions) / desired_power
    broadcast = coefficient.reshape(-1, *([1] * (delta.ndim - 1)))
    orthogonal = delta - broadcast * desired.detach()
    orthogonal_ratio = (
        orthogonal.square().mean(dim=dimensions).sqrt() / desired_power.sqrt()
    )
    lower = F.relu(float(minimum) - coefficient).square()
    upper = F.relu(coefficient - float(maximum)).square()
    orthogonal_penalty = F.relu(
        orthogonal_ratio - float(orthogonal_maximum)
    ).square()
    loss = (lower + upper + float(orthogonal_weight) * orthogonal_penalty).mean()
    return loss, {
        "bounded_aligned_coefficient": coefficient.detach().mean(),
        "bounded_below_fraction": (coefficient.detach() < float(minimum)).float().mean(),
        "bounded_above_fraction": (coefficient.detach() > float(maximum)).float().mean(),
        "bounded_orthogonal_ratio": orthogonal_ratio.detach().mean(),
    }


def _aligned_projection_target_loss(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    coefficient_target: float,
    huber_beta: float,
    scale_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match useful residual magnitude without rewarding orthogonal output norm."""

    dimensions = tuple(range(1, prediction.ndim))
    delta = prediction - base_prediction
    desired = target - base_prediction
    desired_power = desired.square().mean(dim=dimensions).clamp_min(
        float(scale_floor) ** 2
    )
    coefficient = (delta * desired.detach()).mean(dim=dimensions) / desired_power
    expected = torch.full_like(coefficient, float(coefficient_target))
    loss = F.smooth_l1_loss(
        coefficient,
        expected,
        beta=float(huber_beta),
    )
    return loss, {
        "projection_target_coefficient": coefficient.detach().mean(),
        "projection_target_absolute_error": (
            coefficient.detach() - float(coefficient_target)
        ).abs().mean(),
    }


def _common_output_loss(
    deltas: torch.Tensor, *, threshold: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if deltas.ndim < 2 or deltas.shape[0] < 2:
        raise ValueError("Common-output probe needs at least two artist effects")
    dimensions = tuple(range(1, deltas.ndim))
    per_artist_rms = deltas.square().mean(dim=dimensions).sqrt()
    common_rms = deltas.mean(dim=0).square().mean().sqrt()
    denominator = per_artist_rms.mean().detach().clamp_min(1e-8)
    ratio = common_rms / denominator
    return F.relu(ratio - float(threshold)).square(), {
        "common_output_ratio": ratio.detach(),
        "controlled_artist_effect_rms": per_artist_rms.detach().mean(),
        "controlled_common_rms": common_rms.detach(),
    }


def _centered_teacher_effects(
    student_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    *,
    common_threshold: float,
    centered_ratio_minimum: float,
    centered_ratio_maximum: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Center an artist batch while preserving its absolute teacher energy."""

    if student_delta.shape != teacher_delta.shape or student_delta.shape[0] < 2:
        raise ValueError(
            "Centered teacher effects require matching multi-artist batches"
        )
    dimensions = tuple(range(1, student_delta.ndim))
    student = student_delta.float()
    teacher = teacher_delta.detach().float()
    common = student.mean(dim=0, keepdim=True)
    student_centered = student - common
    teacher_centered = teacher - teacher.mean(dim=0, keepdim=True)
    student_rms = student_centered.square().mean(dim=dimensions).sqrt().mean()
    teacher_rms = (
        teacher_centered.square().mean(dim=dimensions).sqrt().mean().clamp_min(1e-8)
    )
    centered_ratio = student_rms / teacher_rms.detach()
    centered_energy_loss = (
        F.relu(float(centered_ratio_minimum) - centered_ratio).square()
        + F.relu(centered_ratio - float(centered_ratio_maximum)).square()
    )
    common_rms = common.square().mean().sqrt()
    common_ratio = common_rms / student_rms.detach().clamp_min(1e-8)
    common_loss = F.relu(common_ratio - float(common_threshold)).square()
    return student_centered, teacher_centered, {
        "native_teacher_centered_student_rms": student_rms.detach(),
        "native_teacher_centered_target_rms": teacher_rms.detach(),
        "native_teacher_centered_student_to_target_rms": centered_ratio.detach(),
        "native_teacher_centered_energy_loss": centered_energy_loss,
        "common_output_ratio": common_ratio.detach(),
        "common_output_loss": common_loss,
        "controlled_common_rms": common_rms.detach(),
        "controlled_artist_effect_rms": student_rms.detach(),
    }


def _teacher_projected_effect_loss(
    student_centered: torch.Tensor,
    teacher_centered: torch.Tensor,
    *,
    coefficient_minimum: float,
    coefficient_maximum: float,
    orthogonal_maximum: float,
    orthogonal_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve artist effect magnitude specifically along the teacher direction.

    A total-RMS floor can be satisfied by a large component orthogonal to the
    teacher.  This objective instead bounds the signed projection coefficient
    and separately suppresses excessive orthogonal energy.
    """

    if student_centered.shape != teacher_centered.shape:
        raise ValueError("Projected teacher tensors must have matching shapes")
    dimensions = tuple(range(1, student_centered.ndim))
    student = student_centered.float()
    teacher = teacher_centered.detach().float()
    teacher_power = teacher.square().mean(dim=dimensions).clamp_min(1e-8)
    teacher_rms = teacher_power.sqrt()
    coefficient = (student * teacher).mean(dim=dimensions) / teacher_power
    coefficient_view = coefficient.reshape(
        -1, *([1] * (student_centered.ndim - 1))
    )
    orthogonal = student - coefficient_view * teacher
    orthogonal_ratio = (
        orthogonal.square().mean(dim=dimensions).sqrt() / teacher_rms
    )
    lower = F.relu(float(coefficient_minimum) - coefficient).square().mean()
    upper = F.relu(coefficient - float(coefficient_maximum)).square().mean()
    orthogonal_loss = F.relu(
        orthogonal_ratio - float(orthogonal_maximum)
    ).square().mean()
    total = lower + upper + float(orthogonal_weight) * orthogonal_loss
    aligned_rms = coefficient.clamp_min(0.0) * teacher_rms
    return total, {
        "native_teacher_projected_effect_loss": total.detach(),
        "native_teacher_projection_floor_loss": lower.detach(),
        "native_teacher_projection_ceiling_loss": upper.detach(),
        "native_teacher_projection_coefficient": coefficient.detach().mean(),
        "native_teacher_projection_positive_fraction": (
            coefficient.detach() > 0
        ).float().mean(),
        "native_teacher_aligned_effect_rms": aligned_rms.detach().mean(),
        "native_teacher_projection_orthogonal_ratio": (
            orthogonal_ratio.detach().mean()
        ),
    }


def _artist_teacher_contrastive_loss(
    student_centered: torch.Tensor,
    teacher_centered: torch.Tensor,
    *,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Use every artist as a positive once; other same-probe artists are negatives."""

    if student_centered.shape != teacher_centered.shape:
        raise ValueError("Artist contrastive tensors must have the same shape")
    if student_centered.shape[0] < 2:
        raise ValueError("Artist contrastive loss needs at least two artists")
    student = F.normalize(student_centered.flatten(1), dim=-1)
    teacher = F.normalize(teacher_centered.detach().flatten(1), dim=-1)
    similarities = student @ teacher.transpose(0, 1)
    logits = similarities / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    contrastive = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )
    diagonal = similarities.diagonal()
    negative = similarities.masked_fill(
        torch.eye(similarities.shape[0], device=similarities.device, dtype=torch.bool),
        -torch.inf,
    ).amax(dim=1)
    ranking = F.relu(float(margin) - diagonal + negative).mean()
    return contrastive, ranking, {
        "native_teacher_artist_contrastive_loss": contrastive.detach(),
        "native_teacher_artist_ranking_loss": ranking.detach(),
        "native_teacher_artist_positive_cosine": diagonal.detach().mean(),
        "native_teacher_artist_hard_negative_cosine": negative.detach().mean(),
        "native_teacher_artist_retrieval_top1": (
            similarities.argmax(dim=1) == labels
        ).float().mean().detach(),
    }


def _native_kv_functional_diversity_loss(
    anima: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    block_indices: list[int],
    slot_energy_floor: float,
    reference_energy_floor: float,
    decorrelation_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure diversity after Frozen Anima's actual context K/V projections."""

    if tokens.ndim != 3 or tokens.shape[0] < 2 or tokens.shape[1] < 2:
        raise ValueError("Functional K/V diversity needs [batch, slots, dim]")
    projected = []
    blocks = getattr(anima, "blocks", None)
    if blocks is None:
        raise TypeError("Anima model does not expose transformer blocks")
    for index in block_indices:
        cross_attention = blocks[int(index)].cross_attn
        if hasattr(cross_attention, "kv_proj"):
            value = cross_attention.kv_proj(tokens)
        elif hasattr(cross_attention, "k_proj") and hasattr(cross_attention, "v_proj"):
            value = torch.cat(
                (cross_attention.k_proj(tokens), cross_attention.v_proj(tokens)), dim=-1
            )
        else:
            raise TypeError("Anima cross-attention exposes no native K/V projection")
        projected.append(value.float())
    signature = torch.cat(projected, dim=-1)
    total_rms = signature.square().mean().sqrt().detach().clamp_min(1e-8)
    slot_centered = signature - signature.mean(dim=1, keepdim=True)
    reference_centered = signature - signature.mean(dim=0, keepdim=True)
    slot_ratio = slot_centered.square().mean().sqrt() / total_rms
    reference_ratio = reference_centered.square().mean().sqrt() / total_rms
    slot_floor_loss = F.relu(float(slot_energy_floor) - slot_ratio).square()
    reference_floor_loss = F.relu(
        float(reference_energy_floor) - reference_ratio
    ).square()
    normalized = F.normalize(slot_centered, dim=-1)
    similarities = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool)
    decorrelation = similarities[:, ~identity].square().mean()
    total = slot_floor_loss + reference_floor_loss + float(
        decorrelation_fraction
    ) * decorrelation
    return total, {
        "functional_value_diversity_loss": total.detach(),
        "functional_value_slot_energy_ratio": slot_ratio.detach(),
        "functional_value_reference_energy_ratio": reference_ratio.detach(),
        "functional_value_decorrelation_loss": decorrelation.detach(),
    }


def _same_artist_functional_loss(
    first_deltas: torch.Tensor,
    second_deltas: torch.Tensor,
    valid: torch.Tensor,
    *,
    direction_fraction: float,
    huber_beta: float,
    center_across_artists: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match two disjoint reference views in frozen-Anima velocity space."""

    if first_deltas.shape != second_deltas.shape:
        raise ValueError("Functional reference views must have the same shape")
    if valid.shape != first_deltas.shape[:1]:
        raise ValueError("Functional reference-view validity has the wrong shape")
    weights = valid.to(first_deltas.dtype)
    if center_across_artists:
        enough_artists = weights.sum().ge(2)
        weights = weights * enough_artists.to(weights.dtype)
        broadcast = weights.reshape(-1, *([1] * (first_deltas.ndim - 1)))
        denominator = weights.sum().clamp_min(1.0)
        first_common = (first_deltas * broadcast).sum(dim=0, keepdim=True)
        first_deltas = first_deltas - first_common / denominator
        second_common = (second_deltas.detach() * broadcast).sum(
            dim=0, keepdim=True
        )
        second_deltas = second_deltas - second_common / denominator
    dimensions = tuple(range(1, first_deltas.ndim))
    first_flat = first_deltas.flatten(1)
    second_flat = second_deltas.detach().flatten(1)
    cosine = F.cosine_similarity(first_flat, second_flat, dim=-1)
    direction = 1.0 - cosine
    first_rms = first_deltas.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    second_rms = (
        second_deltas.detach().square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    )
    log_rms_error = (first_rms.log() - second_rms.log()).abs()
    magnitude = F.smooth_l1_loss(
        first_rms.log(),
        second_rms.log(),
        beta=float(huber_beta),
        reduction="none",
    )
    weights = weights.to(direction.dtype)
    denominator = weights.sum().clamp_min(1.0)
    direction_loss = (direction * weights).sum() / denominator
    magnitude_loss = (magnitude * weights).sum() / denominator
    fraction = float(direction_fraction)
    loss = fraction * direction_loss + (1.0 - fraction) * magnitude_loss
    return loss, {
        "functional_same_artist_cosine": (cosine.detach() * weights).sum()
        / denominator,
        "functional_same_artist_direction_loss": direction_loss.detach(),
        "functional_same_artist_magnitude_loss": magnitude_loss.detach(),
        "functional_same_artist_log_rms_error": (
            log_rms_error.detach() * weights
        ).sum()
        / denominator,
        "functional_same_artist_valid_fraction": weights.mean().detach(),
    }


def _centered_artist_effect_loss(
    deltas: torch.Tensor, *, floor: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prevent reference-specific effects from collapsing into their global mean."""

    if deltas.ndim < 2 or deltas.shape[0] < 2:
        raise ValueError("Centered artist-effect probe needs at least two artists")
    dimensions = tuple(range(1, deltas.ndim))
    common = deltas.mean(dim=0, keepdim=True)
    centered = deltas - common
    effect_rms = deltas.square().mean(dim=dimensions).sqrt()
    centered_rms = centered.square().mean(dim=dimensions).sqrt()
    denominator = effect_rms.mean().detach().clamp_min(1e-8)
    ratio = centered_rms.mean() / denominator
    flat = F.normalize(deltas.flatten(1), dim=-1)
    similarities = flat @ flat.transpose(0, 1)
    off_diagonal = ~torch.eye(
        deltas.shape[0], device=deltas.device, dtype=torch.bool
    )
    between_cosine = similarities.masked_select(off_diagonal).mean()
    pairwise_rms = (
        deltas[:, None] - deltas[None, :]
    ).square().mean(dim=tuple(range(2, deltas.ndim + 1))).sqrt()
    pairwise_rms = pairwise_rms.masked_select(off_diagonal).mean()
    return F.relu(float(floor) - ratio).square(), {
        "functional_centered_effect_ratio": ratio.detach(),
        "functional_centered_effect_rms": centered_rms.detach().mean(),
        "functional_between_artist_cosine": between_cosine.detach(),
        "functional_between_artist_pairwise_rms": pairwise_rms.detach(),
    }


def _native_teacher_alignment_loss(
    student_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    *,
    huber_beta: float,
    scale_floor: float,
    direction_weight: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress a globally centered native @artist velocity effect."""

    if student_delta.shape != teacher_delta.shape:
        raise ValueError("Native teacher and student residual shapes differ")
    dimensions = tuple(range(1, student_delta.ndim))
    teacher = teacher_delta.detach().float()
    student = student_delta.float()
    teacher_rms = teacher.square().mean(dim=dimensions).sqrt().clamp_min(
        float(scale_floor)
    )
    student_rms = student.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    scale = teacher_rms.reshape(-1, *([1] * (student.ndim - 1)))
    direct = F.smooth_l1_loss(
        (student - teacher) / scale,
        torch.zeros_like(student),
        beta=float(huber_beta),
    )
    cosine = F.cosine_similarity(student.flatten(1), teacher.flatten(1), dim=-1)
    direction = (1.0 - cosine).mean()
    magnitude = F.smooth_l1_loss(
        student_rms.log(), teacher_rms.log(), beta=float(huber_beta)
    )
    teacher_power = teacher.square().mean(dim=dimensions).clamp_min(
        float(scale_floor) ** 2
    )
    coefficient = (student * teacher).mean(dim=dimensions) / teacher_power
    coefficient_view = coefficient.reshape(-1, *([1] * (student.ndim - 1)))
    orthogonal = student - coefficient_view * teacher
    orthogonal_ratio = (
        orthogonal.square().mean(dim=dimensions).sqrt() / teacher_rms
    )
    total = direct + float(direction_weight) * direction + float(
        magnitude_weight
    ) * magnitude
    return total, {
        "native_teacher_direct_loss": direct.detach(),
        "native_teacher_direction_loss": direction.detach(),
        "native_teacher_magnitude_loss": magnitude.detach(),
        "native_teacher_cosine": cosine.detach().mean(),
        "native_teacher_projection_coefficient": coefficient.detach().mean(),
        "native_teacher_orthogonal_ratio": orthogonal_ratio.detach().mean(),
        "native_teacher_target_rms": teacher_rms.detach().mean(),
        "native_teacher_student_rms": student_rms.detach().mean(),
        "native_teacher_student_to_target_rms": (
            student_rms.detach() / teacher_rms.detach()
        ).mean(),
    }


def _native_artist_teacher_objective(
    student_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    training: dict[str, Any],
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build magnitude-preserving centered teacher and artist discrimination losses."""

    common_start = int(training.get("common_output_start_step", 1_000))
    common_ramp_end = int(training.get("common_output_ramp_end_step", 2_000))
    common_threshold = _linear_ramp(
        step,
        start_step=common_start,
        end_step=int(training.get("common_output_threshold_end_step", 8_000)),
        start=float(training.get("common_output_threshold_start", 0.85)),
        end=float(training.get("common_output_threshold_end", 0.55)),
    )
    centered_minimum = _linear_ramp(
        step,
        start_step=int(training.get("centered_energy_start_step", 1)),
        end_step=int(training.get("centered_energy_ramp_end_step", 1_500)),
        start=float(training.get("centered_energy_ratio_start", 0.30)),
        end=float(training.get("centered_energy_ratio_end", 0.80)),
    )
    centered_maximum = float(training.get("centered_energy_ratio_maximum", 1.40))
    centered_enabled = bool(training.get("center_student_teacher", False))
    if centered_enabled:
        student, teacher, centered_metrics = _centered_teacher_effects(
            student_delta,
            teacher_delta,
            common_threshold=common_threshold,
            centered_ratio_minimum=centered_minimum,
            centered_ratio_maximum=centered_maximum,
        )
    else:
        student = student_delta
        teacher = teacher_delta
        common_loss, common_metrics = _common_output_loss(
            student_delta, threshold=common_threshold
        )
        centered_metrics = {
            "native_teacher_centered_student_rms": student_delta.new_zeros(()),
            "native_teacher_centered_target_rms": student_delta.new_zeros(()),
            "native_teacher_centered_student_to_target_rms": (
                student_delta.new_zeros(())
            ),
            "native_teacher_centered_energy_loss": student_delta.new_zeros(()),
            "common_output_loss": common_loss,
            **common_metrics,
        }
    projection_floor = _linear_ramp(
        step,
        start_step=int(training.get("teacher_projection_floor_start_step", 1)),
        end_step=int(training.get("teacher_projection_floor_end_step", 1_500)),
        start=float(training.get("teacher_projection_floor_start", 0.02)),
        end=float(training.get("teacher_projection_floor_end", 0.15)),
    )
    projected_effect, projected_metrics = _teacher_projected_effect_loss(
        student,
        teacher,
        coefficient_minimum=projection_floor,
        coefficient_maximum=float(
            training.get("teacher_projection_coefficient_maximum", 1.25)
        ),
        orthogonal_maximum=float(
            training.get("teacher_projection_orthogonal_maximum", 0.50)
        ),
        orthogonal_weight=float(
            training.get("teacher_projection_orthogonal_weight", 0.25)
        ),
    )
    common_denominator_mode = str(
        training.get("common_output_denominator", "student_centered_rms")
    )
    if centered_enabled and common_denominator_mode in {
        "teacher_aligned_projection",
        "teacher_centered_rms",
    }:
        common_rms = student_delta.float().mean(dim=0).square().mean().sqrt()
        aligned_rms = projected_metrics[
            "native_teacher_aligned_effect_rms"
        ].clamp_min(1e-8)
        dimensions = tuple(range(1, teacher.ndim))
        teacher_rms = teacher.detach().float().square().mean(
            dim=dimensions
        ).sqrt().mean().clamp_min(1e-8)
        common_to_aligned = common_rms / aligned_rms
        denominator = (
            aligned_rms
            if common_denominator_mode == "teacher_aligned_projection"
            else teacher_rms
        )
        common_ratio = common_rms / denominator
        centered_metrics["common_output_ratio"] = common_ratio.detach()
        centered_metrics["common_output_loss"] = F.relu(
            common_ratio - float(common_threshold)
        ).square()
        centered_metrics["controlled_common_rms"] = common_rms.detach()
        centered_metrics["controlled_artist_effect_rms"] = denominator.detach()
        centered_metrics["native_teacher_common_to_aligned_ratio"] = (
            common_to_aligned.detach()
        )
        centered_metrics["native_teacher_common_denominator_rms"] = (
            denominator.detach()
        )
    alignment, metrics = _native_teacher_alignment_loss(
        student,
        teacher,
        huber_beta=float(training.get("native_teacher_huber_beta", 0.10)),
        scale_floor=float(training.get("native_teacher_scale_floor", 1e-4)),
        direction_weight=float(training.get("native_teacher_direction_weight", 0.10)),
        magnitude_weight=float(training.get("native_teacher_magnitude_weight", 0.05)),
    )
    start = int(training.get("native_teacher_start_step", 1))
    ramp_steps = max(0, int(training.get("native_teacher_ramp_steps", 250)))
    teacher_weight = _linear_ramp(
        step,
        start_step=start,
        end_step=start + ramp_steps,
        start=0.0 if ramp_steps else float(training.get("native_teacher_weight", 0.05)),
        end=float(training.get("native_teacher_weight", 0.05)),
    )
    common_weight = (
        _linear_ramp(
            step,
            start_step=common_start,
            end_step=common_ramp_end,
            start=0.0,
            end=float(training.get("common_output_weight", 0.0)),
        )
        if step >= common_start
        else 0.0
    )
    energy_weight = _linear_ramp(
        step,
        start_step=int(training.get("centered_energy_start_step", 1)),
        end_step=int(training.get("centered_energy_weight_ramp_end_step", 500)),
        start=0.0,
        end=float(training.get("centered_energy_weight", 0.0)),
    )
    projected_start = int(training.get("teacher_projected_effect_start_step", 1))
    projected_ramp_steps = max(
        0, int(training.get("teacher_projected_effect_ramp_steps", 250))
    )
    projected_target_weight = float(
        training.get("teacher_projected_effect_weight", 0.0)
    )
    projected_weight = (
        _linear_ramp(
            step,
            start_step=projected_start,
            end_step=projected_start + projected_ramp_steps,
            start=0.0 if projected_ramp_steps else projected_target_weight,
            end=projected_target_weight,
        )
        if step >= projected_start
        else 0.0
    )
    contrast_start = int(training.get("artist_teacher_contrastive_start_step", 750))
    contrast_weight = (
        _linear_ramp(
            step,
            start_step=contrast_start,
            end_step=int(
                training.get("artist_teacher_contrastive_ramp_end_step", 1_500)
            ),
            start=0.0,
            end=float(training.get("artist_teacher_contrastive_weight", 0.0)),
        )
        if step >= contrast_start
        else 0.0
    )
    ranking_start = int(training.get("artist_teacher_ranking_start_step", 1_000))
    ranking_weight = (
        _linear_ramp(
            step,
            start_step=ranking_start,
            end_step=int(training.get("artist_teacher_ranking_ramp_end_step", 2_000)),
            start=0.0,
            end=float(training.get("artist_teacher_ranking_weight", 0.0)),
        )
        if step >= ranking_start
        else 0.0
    )
    contrastive, ranking, artist_metrics = _artist_teacher_contrastive_loss(
        student,
        teacher,
        temperature=float(training.get("artist_teacher_temperature", 0.07)),
        margin=float(training.get("artist_teacher_ranking_margin", 0.10)),
    )
    common_loss = centered_metrics["common_output_loss"]
    energy_loss = centered_metrics["native_teacher_centered_energy_loss"]
    weighted_alignment = teacher_weight * alignment
    weighted_common = common_weight * common_loss
    weighted_energy = energy_weight * energy_loss
    weighted_projected = projected_weight * projected_effect
    weighted_contrastive = contrast_weight * contrastive
    weighted_ranking = ranking_weight * ranking
    total = (
        weighted_alignment
        + weighted_common
        + weighted_energy
        + weighted_projected
        + weighted_contrastive
        + weighted_ranking
    )
    metrics.update(
        {key: value.detach() for key, value in centered_metrics.items()}
    )
    metrics.update(projected_metrics)
    metrics.update(artist_metrics)
    metrics.update(
        {
            "native_teacher_alignment_loss": alignment.detach(),
            "native_teacher_weight": alignment.new_tensor(teacher_weight),
            "native_teacher_weighted_loss": weighted_alignment.detach(),
            "native_teacher_centered_energy_ratio_minimum": alignment.new_tensor(
                centered_minimum
            ),
            "native_teacher_centered_energy_weight": alignment.new_tensor(
                energy_weight
            ),
            "native_teacher_centered_energy_weighted_loss": weighted_energy.detach(),
            "native_teacher_projection_floor": alignment.new_tensor(
                projection_floor
            ),
            "native_teacher_projected_effect_weight": alignment.new_tensor(
                projected_weight
            ),
            "native_teacher_projected_effect_weighted_loss": (
                weighted_projected.detach()
            ),
            "native_teacher_artist_contrastive_weight": alignment.new_tensor(
                contrast_weight
            ),
            "native_teacher_artist_contrastive_weighted_loss": (
                weighted_contrastive.detach()
            ),
            "native_teacher_artist_ranking_weight": alignment.new_tensor(
                ranking_weight
            ),
            "native_teacher_artist_ranking_weighted_loss": weighted_ranking.detach(),
            "common_output_weight": alignment.new_tensor(common_weight),
            "common_output_threshold": alignment.new_tensor(common_threshold),
            "common_output_weighted_loss": weighted_common.detach(),
        }
    )
    return total, metrics


def _native_centered_teacher_step(
    anima: torch.nn.Module,
    tokenizer: torch.nn.Module,
    bank: NativeCenteredTeacherBank,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    *,
    step: int,
    metric_prefix: str = "native_teacher",
    probe_index_override: int | None = None,
    reference_count_override: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    references, reference_mask = _reference_inputs(batch, device, "heldout")
    rows = min(
        int(training.get("native_teacher_batch_rows", 4)), references.shape[0]
    )
    references = references[:rows]
    reference_mask = reference_mask[:rows]
    style_ids = [str(item.style_id) for item in batch["episodes"][:rows]]
    try:
        artist_indices = torch.tensor(
            [bank.artist_to_index[value] for value in style_ids], dtype=torch.long
        )
    except KeyError as error:
        raise RuntimeError(f"Artist is absent from native teacher bank: {error}") from error
    tensors = bank.tensors
    contents = int(tensors["noisy_inputs"].shape[0])
    timestep_count = int(tensors["noisy_inputs"].shape[1])
    probe_index = (
        int(probe_index_override)
        if probe_index_override is not None
        else max(0, step - int(training.get("native_teacher_start_step", 0)))
    )
    reference_schedule = training.get("native_teacher_reference_counts")
    reference_count = (
        int(reference_count_override)
        if reference_count_override is not None
        else int(
            reference_schedule[probe_index % len(reference_schedule)]
            if reference_schedule
            else references.shape[1]
        )
    )
    if reference_count <= 0 or reference_count > references.shape[1]:
        raise ValueError(
            f"Teacher reference count {reference_count} exceeds available "
            f"count {references.shape[1]}"
        )
    references = references[:, :reference_count]
    reference_mask = reference_mask[:, :reference_count]
    content_index = probe_index % contents
    timestep_index = (probe_index // contents) % timestep_count
    noisy = tensors["noisy_inputs"][content_index, timestep_index].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    base_prediction = tensors["base_predictions"][content_index, timestep_index].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    teacher = tensors["centered_teacher"][
        artist_indices, content_index, timestep_index
    ].to(device=device, dtype=torch.float32, non_blocking=True)
    context = tensors["base_context"][content_index : content_index + 1].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    length = tensors["base_lengths"][content_index : content_index + 1].to(
        device=device, non_blocking=True
    )
    timestep = tensors["timesteps"][timestep_index].to(
        device=device, dtype=torch.bfloat16
    )
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        output = tokenizer(references, reference_mask)
        styled = insert_style_tokens(
            context.expand(rows, -1, -1).clone(),
            length.expand(rows),
            output.tokens,
        )
        padding = torch.zeros(
            rows, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=noisy.dtype,
        )
        prediction = anima(
            noisy.expand(rows, -1, -1, -1).unsqueeze(2),
            timestep.expand(rows),
            context=styled,
            padding_mask=padding,
            target_input_ids=None,
        ).squeeze(2).float()
    student = prediction - base_prediction
    weighted, metrics = _native_artist_teacher_objective(
        student, teacher, training, step=step
    )
    metrics.update(
        {
            "native_teacher_content_index": student.new_tensor(content_index),
            "native_teacher_timestep": timestep.detach().float(),
            "native_teacher_reference_count": student.new_tensor(reference_count),
        }
    )
    if metric_prefix != "native_teacher":
        metrics = {
            key.replace("native_teacher", metric_prefix, 1): value
            for key, value in metrics.items()
        }
    return weighted, metrics


def _dual_domain_centered_teacher_step(
    anima: torch.nn.Module,
    tokenizer: torch.nn.Module,
    bank: NativeCenteredTeacherBank,
    batches: dict[str, dict[str, Any]],
    device: str,
    training_by_domain: dict[str, dict[str, Any]],
    *,
    step: int,
    probe_index_override: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run independent human/synthetic objectives in one larger Anima batch."""
    domains = ("human_teacher", "synthetic_teacher")
    prepared = []
    for domain in domains:
        batch = batches[domain]
        domain_training = training_by_domain[domain]
        references, reference_mask = _reference_inputs(batch, device, "heldout")
        rows = min(
            int(domain_training.get("native_teacher_batch_rows", 4)),
            references.shape[0],
        )
        style_ids = [str(item.style_id) for item in batch["episodes"][:rows]]
        artist_indices = torch.tensor(
            [bank.artist_to_index[value] for value in style_ids], dtype=torch.long
        )
        prepared.append((
            domain,
            domain_training,
            references[:rows],
            reference_mask[:rows],
            artist_indices,
            rows,
        ))
    available_reference_counts = {item[2].shape[1] for item in prepared}
    if len(available_reference_counts) != 1:
        raise ValueError("Fused teacher domains require the same reference count")

    tensors = bank.tensors
    contents = int(tensors["noisy_inputs"].shape[0])
    timestep_count = int(tensors["noisy_inputs"].shape[1])
    starts = {
        int(item[1].get("native_teacher_start_step", 1)) for item in prepared
    }
    if len(starts) != 1:
        raise ValueError("Fused teacher domains require the same start step")
    probe_index = (
        int(probe_index_override)
        if probe_index_override is not None
        else max(0, step - next(iter(starts)))
    )
    reference_count = _scheduled_teacher_reference_count(
        training_by_domain,
        {item[0]: int(item[2].shape[1]) for item in prepared},
        probe_index,
    )
    total_probes = contents * timestep_count
    cycle, position = divmod(probe_index, total_probes)
    probe_order = list(range(total_probes))
    random.Random(0x7EA4_CE11 + cycle * 1_000_003).shuffle(probe_order)
    selected_probe = probe_order[position]
    content_index = selected_probe % contents
    timestep_index = selected_probe // contents
    noisy = tensors["noisy_inputs"][content_index, timestep_index].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    base_prediction = tensors["base_predictions"][content_index, timestep_index].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    all_indices = torch.cat([item[4] for item in prepared])
    teacher = tensors["centered_teacher"][
        all_indices, content_index, timestep_index
    ].to(device=device, dtype=torch.float32, non_blocking=True)
    context = tensors["base_context"][content_index : content_index + 1].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    length = tensors["base_lengths"][content_index : content_index + 1].to(
        device=device, non_blocking=True
    )
    timestep = tensors["timesteps"][timestep_index].to(
        device=device, dtype=torch.bfloat16
    )
    references = torch.cat(
        [item[2][:, :reference_count] for item in prepared]
    )
    reference_mask = torch.cat(
        [item[3][:, :reference_count] for item in prepared]
    )
    total_rows = references.shape[0]
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        output = tokenizer(references, reference_mask)
        styled = insert_style_tokens(
            context.expand(total_rows, -1, -1).clone(),
            length.expand(total_rows),
            output.tokens,
        )
        padding = torch.zeros(
            total_rows, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=noisy.dtype,
        )
        prediction = anima(
            noisy.expand(total_rows, -1, -1, -1).unsqueeze(2),
            timestep.expand(total_rows), context=styled,
            padding_mask=padding, target_input_ids=None,
        ).squeeze(2).float()
    student = prediction - base_prediction
    total = student.new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    offset = 0
    for domain, domain_training, _, _, _, rows in prepared:
        domain_loss, domain_metrics = _native_artist_teacher_objective(
            student[offset : offset + rows],
            teacher[offset : offset + rows],
            domain_training,
            step=step,
        )
        total = total + domain_loss
        domain_metrics.update({
            "native_teacher_content_index": domain_loss.new_tensor(content_index),
            "native_teacher_timestep": timestep.detach().float(),
            "native_teacher_update_index": domain_loss.new_tensor(probe_index),
            "native_teacher_probe_cycle": domain_loss.new_tensor(cycle),
            "native_teacher_probe_position": domain_loss.new_tensor(position),
            "native_teacher_reference_count": domain_loss.new_tensor(
                reference_count
            ),
        })
        metrics.update({
            (
                f"{domain}_{key}"
                if key.startswith(("common_output", "controlled_"))
                else key.replace("native_teacher", domain, 1)
            ): value
            for key, value in domain_metrics.items()
        })
        offset += rows
    metrics["dual_domain_fused_batch_rows"] = total.new_tensor(total_rows)
    return total, metrics


def _domain_teacher_training(
    training: dict[str, Any], domain: dict[str, Any]
) -> dict[str, Any]:
    result = dict(training)
    for key in (
        "start_step",
        "ramp_steps",
        "weight",
        "direction_weight",
        "magnitude_weight",
        "huber_beta",
        "scale_floor",
        "batch_rows",
        "references",
        "reference_counts",
    ):
        if key in domain:
            result[f"native_teacher_{key}"] = domain[key]
    return result


def _evaluate_native_teacher_domain(
    anima: torch.nn.Module,
    tokenizer: torch.nn.Module,
    bank: NativeCenteredTeacherBank,
    loader: Any,
    device: str,
    training: dict[str, Any],
    *,
    metric_prefix: str,
    batches: int,
    step: int,
) -> dict[str, float]:
    reference_counts = [
        int(value)
        for value in training.get(
            "native_teacher_reference_counts",
            [training.get("native_teacher_references", 4)],
        )
    ]
    result: dict[str, float] = {}
    for reference_count in reference_counts:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        with torch.no_grad():
            for index in range(max(1, int(batches))):
                _, metrics = _native_centered_teacher_step(
                    anima,
                    tokenizer,
                    bank,
                    loader.load_step(index),
                    device,
                    training,
                    step=step,
                    metric_prefix=metric_prefix,
                    probe_index_override=index,
                    reference_count_override=reference_count,
                )
                for key, value in metrics.items():
                    totals[key] += float(value)
                    counts[key] += 1
        averaged = {key: value / counts[key] for key, value in totals.items()}
        result.update(
            {
                f"references_{reference_count}/{key}": value
                for key, value in averaged.items()
            }
        )
        if reference_count == reference_counts[-1]:
            result.update(averaged)
    return result


def _artist_flow_ranking_loss(
    prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank the correct reference above a detached wrong artist in flow space."""

    dimensions = tuple(range(1, prediction.ndim))
    base_error = (
        (base_prediction - target).square().mean(dim=dimensions).clamp_min(1e-8)
    )
    correct_error = (prediction - target).square().mean(dim=dimensions)
    wrong_error = (wrong_prediction.detach() - target).square().mean(dim=dimensions)
    correct_improvement = (base_error - correct_error) / base_error.detach()
    wrong_improvement = (base_error - wrong_error) / base_error.detach()
    advantage = correct_improvement - wrong_improvement
    return F.relu(float(margin) - advantage).mean(), {
        "artist_correct_flow_improvement": correct_improvement.detach().mean(),
        "artist_wrong_flow_improvement": wrong_improvement.detach().mean(),
        "artist_flow_improvement_advantage": advantage.detach().mean(),
    }


def _pilot_functional_probe_step(
    anima: torch.nn.Module,
    tokenizer: torch.nn.Module,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    *,
    generator: torch.Generator,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train reference identity on matched prompt/noise/timestep Anima effects.

    Keeping this forward outside `_forward_dual_query_flow` is important: an
    Anima graph for the controlled artist batch does not fit beside the main
    batch graph on an 80 GB H100. The second reference view is a detached
    target, so only one controlled Anima graph is retained at a time.
    """

    references, reference_mask = _reference_inputs(batch, device, "heldout")
    rows = min(
        int(training.get("functional_probe_batch_rows", 4)), references.shape[0]
    )
    references = references[:rows]
    reference_mask = reference_mask[:rows]
    positions = torch.arange(reference_mask.shape[1], device=device).unsqueeze(0)
    first_mask = reference_mask & positions.remainder(2).eq(0)
    second_mask = reference_mask & positions.remainder(2).eq(1)
    valid_views = reference_mask.sum(dim=1).ge(2)
    second_mask = torch.where(valid_views[:, None], second_mask, first_mask)
    latents = batch["latents"][0:1].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    conditioning = batch["conditioning"][0:1].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    conditioning_length = batch["conditioning_lengths"][0:1].to(
        device, non_blocking=True
    )
    noise = torch.randn(
        latents.shape, device=device, dtype=latents.dtype, generator=generator
    )
    timesteps = _sample_flow_timesteps(1, device, training, generator)
    sigma = timesteps[:, None, None, None].to(latents.dtype)
    noisy = (1 - sigma) * latents + sigma * noise
    padding_mask = torch.zeros(
        1, 1, latents.shape[-2], latents.shape[-1],
        device=device, dtype=latents.dtype,
    )
    start = int(training.get("functional_probe_start_step", 501))
    ramp_end = start + int(training.get("functional_probe_ramp_steps", 500))

    def weight(name: str, default: float) -> float:
        return (
            _linear_ramp(
                step,
                start_step=start,
                end_step=ramp_end,
                start=0.0,
                end=float(training.get(name, default)),
            )
            if step >= start
            else 0.0
        )

    same_weight = weight("same_artist_functional_weight", 0.02)
    centered_weight = weight("centered_artist_effect_weight", 0.03)
    common_weight_key = (
        "functional_common_output_weight"
        if "functional_common_output_weight" in training
        else "common_output_weight"
    )
    common_weight = weight(common_weight_key, 0.03)
    threshold = _linear_ramp(
        step,
        start_step=start,
        end_step=int(training.get("common_output_threshold_end_step", 10_000)),
        start=float(training.get("common_output_threshold_start", 0.60)),
        end=float(training.get("common_output_threshold_end", 0.55)),
    )
    centered_floor = _linear_ramp(
        step,
        start_step=start,
        end_step=int(training.get("centered_artist_effect_floor_end_step", 4_000)),
        start=float(training.get("centered_artist_effect_floor_start", 0.35)),
        end=float(training.get("centered_artist_effect_floor_end", 0.55)),
    )

    def autocast_context():
        return torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        )

    with torch.no_grad(), autocast_context():
        base_prediction = anima(
            noisy.unsqueeze(2),
            timesteps.to(latents.dtype),
            context=conditioning,
            padding_mask=padding_mask,
            target_input_ids=None,
        ).squeeze(2).float()
        second_output = tokenizer(references, second_mask)
        second_styled = insert_style_tokens(
            conditioning.expand(rows, -1, -1).clone(),
            conditioning_length.expand(rows),
            second_output.tokens,
        )
        second_prediction = anima(
            noisy.expand(rows, -1, -1, -1).unsqueeze(2),
            timesteps.expand(rows).to(latents.dtype),
            context=second_styled,
            padding_mask=padding_mask.expand(rows, -1, -1, -1),
            target_input_ids=None,
        ).squeeze(2).float()
    active = same_weight + centered_weight + common_weight > 0
    gradient_context = torch.enable_grad() if active else torch.no_grad()
    with gradient_context, autocast_context():
        output = tokenizer(references, first_mask)
        styled = insert_style_tokens(
            conditioning.expand(rows, -1, -1).clone(),
            conditioning_length.expand(rows),
            output.tokens,
        )
        prediction = anima(
            noisy.expand(rows, -1, -1, -1).unsqueeze(2),
            timesteps.expand(rows).to(latents.dtype),
            context=styled,
            padding_mask=padding_mask.expand(rows, -1, -1, -1),
            target_input_ids=None,
        ).squeeze(2).float()
    first_deltas = prediction - base_prediction
    second_deltas = second_prediction - base_prediction
    common_loss, common_metrics = _common_output_loss(
        first_deltas, threshold=threshold
    )
    same_loss, same_metrics = _same_artist_functional_loss(
        first_deltas,
        second_deltas,
        valid_views,
        direction_fraction=float(
            training.get("same_artist_functional_direction_fraction", 0.75)
        ),
        huber_beta=float(training.get("same_artist_functional_huber_beta", 0.10)),
        center_across_artists=bool(
            training.get("same_artist_functional_center_across_artists", False)
        ),
    )
    centered_loss, centered_metrics = _centered_artist_effect_loss(
        first_deltas, floor=centered_floor
    )
    total = (
        same_weight * same_loss
        + centered_weight * centered_loss
        + common_weight * common_loss
    )
    cadence = max(1, int(training.get("functional_probe_every", 2)))
    metrics = {
        **common_metrics,
        **same_metrics,
        **centered_metrics,
        "functional_same_artist_loss": same_loss.detach(),
        "functional_same_artist_weight": same_loss.new_tensor(same_weight),
        "functional_same_artist_weighted_loss": (same_weight * same_loss).detach(),
        "functional_same_artist_per_step_loss": (
            same_weight * same_loss / cadence
        ).detach(),
        "functional_centered_effect_loss": centered_loss.detach(),
        "functional_centered_effect_weight": centered_loss.new_tensor(centered_weight),
        "functional_centered_effect_floor": centered_loss.new_tensor(centered_floor),
        "functional_centered_effect_weighted_loss": (
            centered_weight * centered_loss
        ).detach(),
        "functional_centered_effect_per_step_loss": (
            centered_weight * centered_loss / cadence
        ).detach(),
        "common_output_loss": common_loss.detach(),
        "common_output_weight": common_loss.new_tensor(common_weight),
        "common_output_threshold": common_loss.new_tensor(threshold),
        "common_output_weighted_loss": (common_weight * common_loss).detach(),
        "common_output_per_step_loss": (
            common_weight * common_loss / cadence
        ).detach(),
        "functional_probe_weighted_loss": total.detach(),
        "functional_probe_per_step_loss": (total / cadence).detach(),
    }
    return total, metrics


def _forward_dual_query_flow(
    anima: torch.nn.Module,
    tokenizer: torch.nn.Module,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    *,
    generator: torch.Generator,
    step: int,
    measure_base: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latents = batch["latents"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    conditioning = batch["conditioning"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    conditioning_lengths = batch["conditioning_lengths"].to(
        device, non_blocking=True
    )
    pilot_enabled = bool(training.get("pilot_enabled", False))
    if pilot_enabled:
        stage = _pilot_stage(step, training)
        if bool(stage.get("exact_self", False)):
            references, reference_mask = _reference_inputs(batch, device, "self")
            include_target = torch.ones(
                references.shape[0], dtype=torch.bool, device=device
            )
        else:
            references, reference_mask = _reference_inputs(batch, device, "heldout")
            include_target = torch.tensor(
                [
                    int(item.target_id) in set(item.reference_ids)
                    for item in batch["episodes"]
                ],
                dtype=torch.bool,
                device=device,
            )
        curriculum = {
            "target_only": bool(stage.get("exact_self", False)),
            "target_probability": float(include_target.float().mean()),
            "phase": str(stage["name"]),
        }
    else:
        references, reference_mask, include_target, curriculum = _reference_batch(
            batch, device, "curriculum", step, training, generator
        )
    noise = torch.randn(
        latents.shape, device=device, dtype=latents.dtype, generator=generator
    )
    timesteps = _sample_flow_timesteps(
        latents.shape[0], device, training, generator
    )
    sigma = timesteps[:, None, None, None].to(latents.dtype)
    noisy = (1 - sigma) * latents + sigma * noise
    target = (noise - latents).float()
    padding_mask = torch.zeros(
        latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
        device=device, dtype=latents.dtype,
    )

    def autocast_context():
        return torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        )

    reconstruction_weight = _linear_weight(
        step,
        start=float(training.get("reconstruction_weight", 0.0)),
        end=float(training.get("reconstruction_final_weight", 0.0)),
        end_step=int(training.get("reconstruction_decay_steps", 10_000)),
    )

    if pilot_enabled:
        alignment = _pilot_alignment_state(step, training)
        normalized_weight = alignment["normalized_weight"]
        floor_weight = alignment["floor_weight"]
    else:
        direct_active = step <= int(training.get("direct_auxiliary_end_step", 2000))
        normalized_weight = (
            float(training.get("normalized_residual_weight", 0.0))
            if direct_active else 0.0
        )
        floor_weight = (
            float(training.get("aligned_floor_weight", 0.0))
            if direct_active else 0.0
        )
        alignment = {
            "coefficient_floor": _coefficient_floor(step, training),
            "bounded_min": 0.0,
            "bounded_max": 0.0,
            "bounded_weight": 0.0,
        }
    bounded_every = max(1, int(training.get("bounded_effect_every", 4)))
    bounded_active = (
        pilot_enabled
        and float(alignment["bounded_weight"]) > 0
        and step % bounded_every == 0
    )
    exact_end = int(training.get("exact_self_end_step", 500))
    projection_active = (
        pilot_enabled
        and step <= exact_end
        and float(training.get("exact_projection_target_weight", 0.0)) > 0
    )
    ranking_start = int(training.get("wrong_ranking_start_step", 1000))
    ranking_every = max(1, int(training.get("wrong_ranking_every", 4)))
    ranking_active = (
        pilot_enabled
        and step >= ranking_start
        and step % ranking_every == 0
        and float(training.get("wrong_ranking_weight", 0.0)) > 0
    )
    needs_base = (
        measure_base
        or normalized_weight > 0
        or floor_weight > 0
        or bounded_active
        or projection_active
        or ranking_active
    )
    base_prediction = None
    # Run the no-grad baseline before retaining the styled graph. This avoids
    # holding the trainable forward's activations during a second Anima pass.
    if needs_base:
        with torch.no_grad(), autocast_context():
            base_prediction = anima(
                noisy.unsqueeze(2), timesteps.to(latents.dtype),
                context=conditioning, padding_mask=padding_mask,
                target_input_ids=None,
            ).squeeze(2).float()
    with autocast_context():
        output = tokenizer(
            references,
            reference_mask,
            reconstruct=reconstruction_weight > 0,
        )
        styled = insert_style_tokens(
            conditioning, conditioning_lengths, output.tokens
        )
        prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype),
            context=styled, padding_mask=padding_mask, target_input_ids=None,
        ).squeeze(2).float()
    flow_loss = F.mse_loss(prediction, target)
    total = flow_loss

    reconstruction = flow_loss.new_zeros(())
    reconstruction_huber = flow_loss.new_zeros(())
    reconstruction_cosine = flow_loss.new_zeros(())
    if reconstruction_weight > 0:
        if output.reconstruction is None or output.reconstruction_target is None:
            raise RuntimeError("Configured reconstruction requires a tokenizer decoder")
        reconstruction_huber = F.smooth_l1_loss(
            output.reconstruction.float(),
            output.reconstruction_target.float(),
            beta=float(training.get("reconstruction_huber_beta", 0.1)),
        )
        reconstruction_cosine = (
            1.0
            - F.cosine_similarity(
                output.reconstruction.float(),
                output.reconstruction_target.float(),
                dim=-1,
            )
        ).mean()
        reconstruction = reconstruction_huber + float(
            training.get("reconstruction_cosine_weight", 0.25)
        ) * reconstruction_cosine
        total = total + reconstruction_weight * reconstruction

    normalized = flow_loss.new_zeros(())
    floor = flow_loss.new_zeros(())
    direct_metrics = {
        "aligned_coefficient": flow_loss.new_zeros(()),
        "aligned_floor_violation": flow_loss.new_zeros(()),
        "target_auxiliary_fraction": include_target.float().mean(),
    }
    if normalized_weight > 0 or floor_weight > 0:
        assert base_prediction is not None
        aligned_mask = (
            torch.ones_like(include_target) if pilot_enabled else include_target
        )
        normalized, floor, direct_metrics = _aligned_velocity_losses(
            prediction,
            base_prediction,
            target,
            aligned_mask,
            coefficient_floor=float(alignment["coefficient_floor"]),
            huber_beta=float(training.get("normalized_residual_huber_beta", 0.1)),
            scale_floor=float(training.get("normalized_residual_scale_floor", 1e-4)),
        )
        total = total + normalized_weight * normalized + floor_weight * floor

    diversity_target_weight = float(training.get("slot_diversity_weight", 0.0))
    configured_diversity_ramp = int(training.get("diversity_ramp_steps", 0))
    diversity_progress = (
        min(1.0, float(step) / max(1, configured_diversity_ramp))
        if configured_diversity_ramp > 0 else 1.0
    )
    diversity_weight = diversity_target_weight * diversity_progress
    diversity_source = getattr(output, "diversity_tokens", output.tokens)
    diversity = _slot_diversity_loss(diversity_source)
    total = total + diversity_weight * diversity

    attention_target_weight = float(
        training.get("attention_diversity_weight", 0.0)
    )
    reference_target_weight = float(
        training.get("reference_conditioned_diversity_weight", 0.0)
    )
    attention_weight = attention_target_weight * diversity_progress
    reference_diversity_weight = reference_target_weight * diversity_progress
    attention_diversity = flow_loss.new_zeros(())
    if attention_weight > 0:
        attention_maps = getattr(output, "attention_maps", None)
        if attention_maps is None:
            raise RuntimeError(
                "attention_diversity_weight requires tokenizer attention maps"
            )
        attention_diversity = attention_map_diversity_loss(attention_maps)
        total = total + attention_weight * attention_diversity
    reference_diversity = flow_loss.new_zeros(())
    if reference_diversity_weight > 0:
        reference_tokens = getattr(
            output, "reference_conditioned_tokens", None
        )
        if reference_tokens is None:
            raise RuntimeError(
                "reference_conditioned_diversity_weight requires conditioned tokens"
            )
        reference_diversity = reference_conditioned_diversity_loss(
            reference_tokens
        )
        total = total + reference_diversity_weight * reference_diversity

    functional_value_start = int(
        training.get("functional_value_diversity_start_step", 1)
    )
    functional_value_progress = (
        _linear_ramp(
            step,
            start_step=functional_value_start,
            end_step=functional_value_start
            + int(training.get("functional_value_diversity_ramp_steps", 500)),
            start=0.0,
            end=1.0,
        )
        if step >= functional_value_start
        else 0.0
    )
    functional_value_weight = float(
        training.get("functional_value_diversity_weight", 0.0)
    ) * functional_value_progress
    functional_value_diversity = flow_loss.new_zeros(())
    functional_value_metrics = {
        "functional_value_slot_energy_ratio": flow_loss.new_zeros(()),
        "functional_value_reference_energy_ratio": flow_loss.new_zeros(()),
        "functional_value_decorrelation_loss": flow_loss.new_zeros(()),
    }
    if functional_value_weight > 0:
        block_indices = [
            int(value)
            for value in training.get(
                "functional_value_blocks", [0, 4, 8, 12, 16, 20, 24, 27]
            )
        ]
        functional_value_diversity, functional_value_metrics = (
            _native_kv_functional_diversity_loss(
                anima,
                output.tokens,
                block_indices=block_indices,
                slot_energy_floor=float(
                    training.get("functional_value_slot_energy_floor", 0.20)
                ),
                reference_energy_floor=float(
                    training.get("functional_value_reference_energy_floor", 0.03)
                ),
                decorrelation_fraction=float(
                    training.get("functional_value_decorrelation_fraction", 0.10)
                ),
            )
        )
        total = total + functional_value_weight * functional_value_diversity

    context_rms_weight = float(training.get("context_rms_weight", 0.0))
    token_sample_rms = output.tokens.float().square().mean(dim=(1, 2)).sqrt()
    context_rms_minimum = float(training.get("context_rms_minimum", 0.08))
    context_rms_maximum = float(training.get("context_rms_maximum", 0.25))
    log_rms = token_sample_rms.clamp_min(1e-8).log()
    context_rms_loss = (
        F.relu(math.log(context_rms_minimum) - log_rms).square()
        + F.relu(log_rms - math.log(context_rms_maximum)).square()
    ).mean()
    total = total + context_rms_weight * context_rms_loss

    contrastive = flow_loss.new_zeros(())
    positive = flow_loss.new_zeros(())
    negative = flow_loss.new_zeros(())
    contrastive_weight = 0.0
    contrastive_every = max(1, int(training.get("artist_contrastive_every", 2)))
    contrastive_start = int(training.get("artist_contrastive_start_step", 1))
    if (
        float(training.get("artist_contrastive_weight", 0.0)) > 0
        and step >= contrastive_start
        and step % contrastive_every == 0
        and output.tokens.shape[0] > 1
    ):
        heldout, heldout_mask = _reference_inputs(batch, device, "heldout")
        target_tokens = batch["cached_target_tokens"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )
        with autocast_context():
            target_output = (
                output
                if bool(curriculum.get("target_only", False))
                else tokenizer(
                    target_tokens[:, None],
                    torch.ones(
                        target_tokens.shape[0], 1,
                        device=device, dtype=torch.bool,
                    ),
                )
            )
            heldout_output = tokenizer(heldout, heldout_mask)
        contrastive, contrastive_metrics = _artist_contrastive_loss(
            getattr(target_output, "artist_tokens", target_output.tokens),
            getattr(heldout_output, "artist_tokens", heldout_output.tokens),
            [str(item.style_id) for item in batch["episodes"]],
            float(training.get("artist_contrastive_temperature", 0.1)),
        )
        positive = contrastive_metrics["artist_positive_similarity"]
        negative = contrastive_metrics["artist_negative_similarity"]
        contrastive_weight = _linear_ramp(
            step,
            start_step=contrastive_start,
            end_step=contrastive_start
            + int(training.get("artist_contrastive_ramp_steps", 500)),
            start=0.0,
            end=float(training["artist_contrastive_weight"]),
        )
        total = total + contrastive_weight * contrastive

    token_contrastive = flow_loss.new_zeros(())
    token_contrastive_weight = 0.0
    token_positive = flow_loss.new_zeros(())
    token_negative = flow_loss.new_zeros(())
    token_margin = flow_loss.new_zeros(())
    configured_token_weight = float(training.get("token_contrastive_weight", 0.0))
    token_contrastive_start = int(training.get("token_contrastive_start_step", 1))
    if configured_token_weight > 0 and step >= token_contrastive_start:
        eligible, first_mask, second_mask = _split_reference_views(reference_mask)
        if int(eligible.sum()) >= 2:
            selected = references[eligible]
            with autocast_context():
                first_tokens = tokenizer(selected, first_mask).tokens
                second_tokens = tokenizer(selected, second_mask).tokens
            selected_indices = eligible.nonzero(as_tuple=False).flatten().tolist()
            style_ids = [
                str(batch["episodes"][index].style_id) for index in selected_indices
            ]
            token_contrastive, token_metrics = _style_token_contrastive_loss(
                first_tokens,
                second_tokens,
                style_ids,
                float(training.get("token_contrastive_temperature", 0.10)),
            )
            token_contrastive_weight = _linear_ramp(
                step,
                start_step=token_contrastive_start,
                end_step=token_contrastive_start
                + int(training.get("token_contrastive_ramp_steps", 1000)),
                start=0.0,
                end=configured_token_weight,
            )
            total = total + token_contrastive_weight * token_contrastive
            token_positive = token_metrics["token_positive_similarity"]
            token_negative = token_metrics["token_negative_similarity"]
            token_margin = token_metrics["token_similarity_margin"]

    bounded_loss = flow_loss.new_zeros(())
    bounded_weight = 0.0
    bounded_metrics = {
        "bounded_aligned_coefficient": flow_loss.new_zeros(()),
        "bounded_below_fraction": flow_loss.new_zeros(()),
        "bounded_above_fraction": flow_loss.new_zeros(()),
        "bounded_orthogonal_ratio": flow_loss.new_zeros(()),
    }
    if bounded_active:
        assert base_prediction is not None
        bounded_loss, bounded_metrics = _bounded_aligned_effect_loss(
            prediction,
            base_prediction,
            target,
            minimum=float(alignment["bounded_min"]),
            maximum=float(alignment["bounded_max"]),
            orthogonal_maximum=float(training.get("bounded_orthogonal_maximum", 0.12)),
            orthogonal_weight=float(training.get("bounded_orthogonal_weight", 0.25)),
            scale_floor=float(training.get("normalized_residual_scale_floor", 1e-4)),
        )
        bounded_weight = float(alignment["bounded_weight"])
        total = total + bounded_weight * bounded_loss

    projection_target_loss = flow_loss.new_zeros(())
    projection_target_weight = 0.0
    projection_target_metrics = {
        "projection_target_coefficient": flow_loss.new_zeros(()),
        "projection_target_absolute_error": flow_loss.new_zeros(()),
    }
    if pilot_enabled and step <= exact_end:
        projection_target_weight = float(
            training.get("exact_projection_target_weight", 0.0)
        )
        if projection_target_weight > 0:
            assert base_prediction is not None
            projection_target_loss, projection_target_metrics = (
                _aligned_projection_target_loss(
                    prediction,
                    base_prediction,
                    target,
                    coefficient_target=float(
                        training.get("exact_projection_target", 1.0)
                    ),
                    huber_beta=float(
                        training.get("exact_projection_target_huber_beta", 0.1)
                    ),
                    scale_floor=float(
                        training.get("normalized_residual_scale_floor", 1e-4)
                    ),
                )
            )
            total = total + projection_target_weight * projection_target_loss

    direction_loss = flow_loss.new_zeros(())
    direction_weight = 0.0
    direction_metrics = {
        "artist_correct_direction_cosine": flow_loss.new_zeros(()),
        "artist_wrong_direction_cosine": flow_loss.new_zeros(()),
        "artist_centered_direction_cosine": flow_loss.new_zeros(()),
        "artist_direction_ranking_loss": flow_loss.new_zeros(()),
        "artist_flow_ranking_loss": flow_loss.new_zeros(()),
        "artist_correct_flow_improvement": flow_loss.new_zeros(()),
        "artist_wrong_flow_improvement": flow_loss.new_zeros(()),
        "artist_flow_improvement_advantage": flow_loss.new_zeros(()),
    }
    if ranking_active:
        assert base_prediction is not None
        wrong_references, wrong_mask = _reference_inputs(batch, device, "wrong_artist")
        rows = min(
            int(training.get("wrong_ranking_batch_rows", 2)),
            prediction.shape[0],
        )
        with torch.no_grad(), autocast_context():
            wrong_output = tokenizer(wrong_references[:rows], wrong_mask[:rows])
            wrong_styled = insert_style_tokens(
                conditioning[:rows], conditioning_lengths[:rows], wrong_output.tokens
            )
            wrong_prediction = anima(
                noisy[:rows].unsqueeze(2),
                timesteps[:rows].to(latents.dtype),
                context=wrong_styled,
                padding_mask=padding_mask[:rows],
                target_input_ids=None,
            ).squeeze(2).float()
        direction_loss, direction_metrics = _artist_direction_loss(
            prediction[:rows],
            wrong_prediction,
            base_prediction[:rows],
            target[:rows],
            margin=float(training.get("wrong_ranking_margin", 0.02)),
            centered_weight=float(training.get("wrong_centered_weight", 0.10)),
        )
        flow_ranking, flow_ranking_metrics = _artist_flow_ranking_loss(
            prediction[:rows],
            wrong_prediction,
            base_prediction[:rows],
            target[:rows],
            margin=float(training.get("wrong_flow_ranking_margin", 0.01)),
        )
        direction_metrics["artist_flow_ranking_loss"] = flow_ranking.detach()
        direction_metrics.update(flow_ranking_metrics)
        direction_loss = direction_loss + float(
            training.get("wrong_flow_ranking_weight", 1.0)
        ) * flow_ranking
        direction_weight = _linear_ramp(
            step,
            start_step=ranking_start,
            end_step=ranking_start + int(training.get("wrong_ranking_ramp_steps", 1000)),
            start=0.0,
            end=float(training.get("wrong_ranking_weight", 0.00075)),
        )
        total = total + direction_weight * direction_loss

    metrics = {
        "loss": total.detach(),
        "flow_loss": flow_loss.detach(),
        "reconstruction_loss": reconstruction.detach(),
        "reconstruction_huber_loss": reconstruction_huber.detach(),
        "reconstruction_cosine_loss": reconstruction_cosine.detach(),
        "reconstruction_weight": flow_loss.new_tensor(reconstruction_weight),
        "reconstruction_weighted_loss": (
            reconstruction_weight * reconstruction
        ).detach(),
        "normalized_residual_loss": normalized.detach(),
        "normalized_residual_weight": flow_loss.new_tensor(normalized_weight),
        "normalized_residual_weighted_loss": (
            normalized_weight * normalized
        ).detach(),
        "aligned_floor_loss": floor.detach(),
        "aligned_floor_weight": flow_loss.new_tensor(floor_weight),
        "aligned_floor_weighted_loss": (floor_weight * floor).detach(),
        "aligned_coefficient_floor": flow_loss.new_tensor(
            float(alignment["coefficient_floor"])
        ),
        **{key: value.detach() for key, value in direct_metrics.items()},
        "artist_contrastive_loss": contrastive.detach(),
        "artist_contrastive_weight": flow_loss.new_tensor(contrastive_weight),
        "artist_contrastive_weighted_loss": (
            contrastive_weight * contrastive
        ).detach(),
        "artist_positive_similarity": positive.detach(),
        "artist_negative_similarity": negative.detach(),
        "token_contrastive_loss": token_contrastive.detach(),
        "token_contrastive_weight": flow_loss.new_tensor(token_contrastive_weight),
        "token_contrastive_weighted_loss": (
            token_contrastive_weight * token_contrastive
        ).detach(),
        "token_positive_similarity": token_positive.detach(),
        "token_negative_similarity": token_negative.detach(),
        "token_similarity_margin": token_margin.detach(),
        "slot_diversity_loss": diversity.detach(),
        "slot_diversity_weight": flow_loss.new_tensor(diversity_weight),
        "slot_diversity_weighted_loss": (diversity_weight * diversity).detach(),
        "bounded_effect_loss": bounded_loss.detach(),
        "bounded_effect_weight": flow_loss.new_tensor(bounded_weight),
        "bounded_effect_weighted_loss": (bounded_weight * bounded_loss).detach(),
        **{key: value.detach() for key, value in bounded_metrics.items()},
        "projection_target_loss": projection_target_loss.detach(),
        "projection_target_weight": flow_loss.new_tensor(projection_target_weight),
        "projection_target_weighted_loss": (
            projection_target_weight * projection_target_loss
        ).detach(),
        **{
            key: value.detach()
            for key, value in projection_target_metrics.items()
        },
        "artist_direction_loss": direction_loss.detach(),
        "artist_direction_weight": flow_loss.new_tensor(direction_weight),
        "artist_direction_weighted_loss": (
            direction_weight * direction_loss
        ).detach(),
        **{key: value.detach() for key, value in direction_metrics.items()},
        "style_token_rms": output.tokens.detach().float().square().mean().sqrt(),
        "style_token_sample_rms_std": output.tokens.detach().float().square()
        .mean(dim=(1, 2)).sqrt().std(unbiased=False),
        "style_token_slot_rms_std": output.tokens.detach().float().square()
        .mean(dim=(0, 2)).sqrt().std(unbiased=False),
        "attention_diversity_loss": attention_diversity.detach(),
        "attention_diversity_weight": flow_loss.new_tensor(attention_weight),
        "attention_diversity_weighted_loss": (
            attention_weight * attention_diversity
        ).detach(),
        "reference_conditioned_diversity_loss": reference_diversity.detach(),
        "reference_conditioned_diversity_weight": flow_loss.new_tensor(
            reference_diversity_weight
        ),
        "reference_conditioned_diversity_weighted_loss": (
            reference_diversity_weight * reference_diversity
        ).detach(),
        "functional_value_diversity_loss": functional_value_diversity.detach(),
        "functional_value_diversity_weight": flow_loss.new_tensor(
            functional_value_weight
        ),
        "functional_value_diversity_weighted_loss": (
            functional_value_weight * functional_value_diversity
        ).detach(),
        **{
            key: value.detach()
            for key, value in functional_value_metrics.items()
        },
        "context_rms_loss": context_rms_loss.detach(),
        "context_rms_weight": flow_loss.new_tensor(context_rms_weight),
        "context_rms_weighted_loss": (
            context_rms_weight * context_rms_loss
        ).detach(),
        "references": reference_mask.sum(dim=1).float().mean(),
        "reference_count_1_fraction": (
            reference_mask.sum(dim=1) == 1
        ).float().mean(),
        "reference_count_2_fraction": (
            reference_mask.sum(dim=1) == 2
        ).float().mean(),
        "target_inclusion": include_target.float().mean(),
        "target_probability": flow_loss.new_tensor(
            float(curriculum["target_probability"])
        ),
        "timestep_mean": timesteps.detach().mean(),
    }
    output_gain = getattr(output, "output_gain", None)
    if output_gain is not None:
        metrics["style_output_gain_mean"] = output_gain.detach().float().mean()
        metrics["style_output_gain_std"] = output_gain.detach().float().std(
            unbiased=False
        )
    prompt_modes = [str(value) for value in batch.get("prompt_modes", [])]
    if prompt_modes:
        base_modes = [
            value.removesuffix("_quality") for value in prompt_modes
        ]
        for mode in ("full", "tag_dropout", "short", "empty"):
            metrics[f"prompt_mode_{mode}_fraction"] = flow_loss.new_tensor(
                sum(value == mode for value in base_modes) / len(base_modes)
            )
        metrics["prompt_quality_fraction"] = flow_loss.new_tensor(
            sum(value.endswith("_quality") for value in prompt_modes)
            / len(prompt_modes)
        )
    if base_prediction is not None:
        metrics.update(
            {
                key: value.detach()
                for key, value in _flow_metrics(
                    prediction.detach(), base_prediction, target
                ).items()
            }
        )
    return total, metrics


def _selection_score(row: dict[str, Any]) -> float:
    heldout = float(row["validation_heldout"]["paired_flow_improvement"])
    wrong = float(row["validation_wrong_artist"]["paired_flow_improvement"])
    exact = float(row["validation_self"]["paired_flow_improvement"])
    return heldout + 0.5 * (heldout - wrong) + 0.25 * exact


def _train_variant(
    config: dict[str, Any],
    destination: Path,
    anima: torch.nn.Module,
    *,
    include_artist_summary: bool,
    output_name: str,
    steps_override: int | None = None,
    training_overrides: dict[str, Any] | None = None,
    wandb_suffix: str = "",
    cfg_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(
        cfg_override if cfg_override is not None else config["dual_query_style_tokenizer"]
    )
    cfg["model"]["include_artist_summary"] = bool(include_artist_summary)
    training = dict(cfg["training"])
    if training_overrides:
        for key, value in training_overrides.items():
            training[key] = copy.deepcopy(value)
    cfg["training"] = training
    steps = int(steps_override or training["steps"])
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260816))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if bool(training.get("allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    dual_domain_cfg = dict(training.get("dual_domain_teacher", {}))
    dual_domain_enabled = bool(dual_domain_cfg.get("enabled", False))
    dual_domain_bank = None
    heldout_teacher_styles: list[str] = []
    if dual_domain_enabled:
        dual_domain_bank = NativeCenteredTeacherBank.load(
            config,
            destination,
            config_key=str(
                dual_domain_cfg.get(
                    "bank_config_key", "dual_domain_native_teacher"
                )
            ),
        )
        heldout_teacher_styles = [
            str(value)
            for key in ("validation_style_ids", "test_style_ids")
            for value in dual_domain_bank.summary.get(key, [])
        ]

    use_multi_prompt = bool(cfg.get("loader", {}).get("prompt_modes"))
    train_loader_class = (
        MultiPromptDualQueryCachedStyleLoader
        if use_multi_prompt else DualQueryCachedStyleLoader
    )
    train_loader_cfg = _loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    if heldout_teacher_styles:
        train_loader_cfg = _with_excluded_style_ids(
            train_loader_cfg, heldout_teacher_styles
        )
    train_loader = train_loader_class(destination, train_loader_cfg)
    validation_loader_cfg = _loader_config(
        config, cfg, split=str(cfg.get("validation_split", "validation"))
    )
    if use_multi_prompt:
        validation_loader_cfg["prompt_modes"] = dict(
            cfg.get("loader", {}).get(
                "validation_prompt_modes",
                {"full": 1.0, "tag_dropout": 0.0, "short": 0.0, "empty": 0.0},
            )
        )
        validation_loader_cfg["prompt_modes"]["quality_probability"] = float(
            cfg.get("loader", {}).get("validation_quality_probability", 0.0)
        )
    validation_loader = train_loader_class(destination, validation_loader_cfg)
    pilot_enabled = bool(training.get("pilot_enabled", False))
    functional_probe_enabled = pilot_enabled and bool(
        training.get("functional_probe_enabled", True)
    )
    reference_eval_loaders: dict[int, DualQueryCachedStyleLoader] = {}
    controlled_loader = None
    functional_loader = None
    if pilot_enabled:
        for count in (1, 2, 4, 8):
            eval_loader_cfg = _loader_config(
                config, cfg, split=str(cfg.get("validation_split", "validation"))
            )
            eval_loader_cfg.update(
                {
                    "min_references": count,
                    "max_references": count,
                    "reference_count_weights": None,
                    "pilot_reference_schedule": [],
                }
            )
            if use_multi_prompt:
                eval_loader_cfg["prompt_modes"] = dict(
                    validation_loader_cfg["prompt_modes"]
                )
            reference_eval_loaders[count] = train_loader_class(
                destination, eval_loader_cfg
            )
        controlled_cfg = _loader_config(
            config, cfg, split=str(cfg.get("validation_split", "validation"))
        )
        controlled_cfg.update(
            {
                "batch_size": int(training.get("controlled_evaluation_artists", 4)),
                "min_references": 8,
                "max_references": 8,
                "reference_count_weights": None,
                "pilot_reference_schedule": [],
                "artist_balanced": True,
            }
        )
        if use_multi_prompt:
            controlled_cfg["prompt_modes"] = dict(
                validation_loader_cfg["prompt_modes"]
            )
        controlled_loader = train_loader_class(destination, controlled_cfg)
        if functional_probe_enabled:
            functional_cfg = _loader_config(
                config, cfg, split=str(cfg.get("train_split", "train"))
            )
            functional_cfg.update(
                {
                    "batch_size": int(
                        training.get("functional_probe_batch_rows", 4)
                    ),
                    "min_references": int(
                        training.get("functional_probe_references", 4)
                    ),
                    "max_references": int(
                        training.get("functional_probe_references", 4)
                    ),
                    "reference_count_weights": None,
                    "reference_curriculum": {},
                    "pilot_reference_schedule": [],
                    "artist_balanced": True,
                }
            )
            functional_loader = DualQueryCachedStyleLoader(destination, functional_cfg)
    native_teacher_enabled = bool(training.get("native_teacher_enabled", False))
    native_teacher_bank = None
    native_teacher_loader = None
    native_teacher_validation_loader = None
    if native_teacher_enabled:
        native_teacher_bank = NativeCenteredTeacherBank.load(config, destination)

        def build_native_loader(split: str, style_ids: list[str]):
            if not style_ids:
                return None
            loader_cfg = _loader_config(config, cfg, split=split)
            references = int(training.get("native_teacher_references", 4))
            loader_cfg.update(
                {
                    "batch_size": int(training.get("native_teacher_batch_rows", 4)),
                    "min_references": references,
                    "max_references": references,
                    "artist_balanced": True,
                    "gradient_accumulation_steps": 1,
                    "reference_curriculum": {},
                    "pilot_reference_schedule": [],
                    "allowed_style_ids": style_ids,
                }
            )
            return DualQueryCachedStyleLoader(destination, loader_cfg)

        native_teacher_loader = build_native_loader(
            str(cfg.get("train_split", "train")),
            [str(value) for value in native_teacher_bank.summary["train_style_ids"]],
        )
        native_teacher_validation_loader = build_native_loader(
            str(cfg.get("validation_split", "validation")),
            [
                str(value)
                for value in native_teacher_bank.summary["validation_style_ids"]
            ],
        )
        if native_teacher_loader is None:
            raise RuntimeError("Native centered teacher has no training artists")
    dual_domain_train_loaders: dict[str, Any] = {}
    dual_domain_validation_loaders: dict[str, Any] = {}
    dual_domain_training: dict[str, dict[str, Any]] = {}
    if dual_domain_enabled:
        assert dual_domain_bank is not None
        train_style_ids = [
            str(value) for value in dual_domain_bank.summary["train_style_ids"]
        ]
        validation_style_ids = [
            str(value)
            for value in dual_domain_bank.summary["validation_style_ids"]
        ]
        all_teacher_style_ids = list(dual_domain_bank.artist_to_index)
        human_cfg = dict(dual_domain_cfg["human"])
        synthetic_cfg = dict(dual_domain_cfg["synthetic"])
        human_train_split, human_validation_split = _dual_domain_human_splits(
            cfg, human_cfg
        )

        def build_human_domain(style_ids: list[str], *, split: str):
            loader_cfg = _loader_config(
                config,
                cfg,
                split=split,
            )
            reference_counts = [
                int(value)
                for value in human_cfg.get(
                    "reference_counts", [human_cfg.get("references", 4)]
                )
            ]
            references = max(reference_counts)
            loader_cfg.update(
                {
                    "batch_size": int(human_cfg.get("batch_rows", 4)),
                    "min_references": references,
                    "max_references": references,
                    "artist_balanced": True,
                    "gradient_accumulation_steps": 1,
                    "reference_curriculum": {},
                    "pilot_reference_schedule": [],
                    "allowed_style_ids": style_ids,
                    "excluded_style_ids": [],
                }
            )
            return DualQueryCachedStyleLoader(destination, loader_cfg)

        synthetic_root = destination / str(
            dual_domain_cfg["synthetic_reference_cache"]
        )
        dual_domain_train_loaders["human_teacher"] = build_human_domain(
            train_style_ids,
            split=human_train_split,
        )
        dual_domain_validation_loaders["human_teacher"] = build_human_domain(
            validation_style_ids,
            split=human_validation_split,
        )
        dual_domain_train_loaders["synthetic_teacher"] = (
            CachedTeacherReferenceLoader(
                synthetic_root,
                split="train",
                style_ids=train_style_ids,
                batch_size=int(synthetic_cfg.get("batch_rows", 4)),
                references=max(
                    int(value)
                    for value in synthetic_cfg.get(
                        "reference_counts",
                        [synthetic_cfg.get("references", 4)],
                    )
                ),
                seed=seed ^ 0x51A7_0001,
                token_lru_shards=int(synthetic_cfg.get("token_lru_shards", 8)),
                ram_resident_tokens=bool(
                    synthetic_cfg.get("ram_resident_tokens", False)
                ),
                ram_preload_workers=int(
                    synthetic_cfg.get("ram_preload_workers", 8)
                ),
                strict_style_ids=False,
            )
        )
        dual_domain_validation_loaders["synthetic_teacher"] = (
            CachedTeacherReferenceLoader(
                synthetic_root,
                split="validation",
                # Synthetic references have their own 450/25/25 split inside
                # the 500-artist subset (which was originally sampled from
                # the human training artists).  Let that cache choose its
                # held-out validation artists, then look their effects up in
                # the full 5k bank.  Reusing the human validation IDs here
                # makes the valid intersection empty by construction.
                style_ids=all_teacher_style_ids,
                batch_size=int(synthetic_cfg.get("batch_rows", 4)),
                references=max(
                    int(value)
                    for value in synthetic_cfg.get(
                        "reference_counts",
                        [synthetic_cfg.get("references", 4)],
                    )
                ),
                seed=seed ^ 0x51A7_0002,
                token_lru_shards=int(synthetic_cfg.get("token_lru_shards", 8)),
                ram_resident_tokens=bool(
                    synthetic_cfg.get("ram_resident_tokens", False)
                ),
                ram_preload_workers=int(
                    synthetic_cfg.get("ram_preload_workers", 8)
                ),
                strict_style_ids=False,
            )
        )
        dual_domain_training = {
            "human_teacher": _domain_teacher_training(training, human_cfg),
            "synthetic_teacher": _domain_teacher_training(training, synthetic_cfg),
        }
    cache_summary = _cache_summary(destination, cfg)
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "flat_set"))
    if architecture == "flat_set":
        tokenizer = DualQuerySetStyleTokenizer(**model_cfg).to(device)
    elif architecture == "hierarchical":
        tokenizer = HierarchicalDualQueryStyleTokenizer(**model_cfg).to(device)
    elif architecture == "compact":
        tokenizer = CompactDualQueryStyleTokenizer(**model_cfg).to(device)
    elif architecture == "global_query_memory":
        tokenizer = GlobalQueryMemoryStyleTokenizer(**model_cfg).to(device)
    elif architecture == "slot_preserving_global_query":
        tokenizer = SlotPreservingGlobalQueryStyleTokenizer(**model_cfg).to(device)
    elif architecture == "typed_multi_descriptor_compact":
        tokenizer = TypedMultiDescriptorCompactStyleTokenizer(**model_cfg).to(device)
    else:
        raise ValueError(
            f"Unknown Dual-query StyleTokenizer architecture {architecture!r}"
        )
    output = destination / output_name
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    history_path = output / "history.json"
    history = json.loads(history_path.read_text("utf-8")) if history_path.exists() else []
    start_step = 0
    resume_state = None
    resume_source = ""
    if bool(training.get("resume", True)) and state_path.exists():
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        resume_source = "output"
    elif training.get("initial_checkpoint"):
        initial_path = Path(str(training["initial_checkpoint"]))
        if not initial_path.is_absolute():
            initial_path = destination / initial_path
        resume_state = torch.load(initial_path, map_location="cpu", weights_only=False)
        resume_source = "initial"
        initial_history = training.get("initial_history")
        if initial_history and not history:
            initial_history_path = Path(str(initial_history))
            if not initial_history_path.is_absolute():
                initial_history_path = destination / initial_history_path
            source_history = json.loads(initial_history_path.read_text("utf-8"))
            history = [
                row
                for row in source_history
                if int(row.get("step", 0)) <= int(resume_state["step"])
            ]
        print(
            f"initializing {output_name} from {initial_path} "
            f"at step {int(resume_state['step'])}",
            flush=True,
        )
    if resume_state is not None:
        recorded = bool(resume_state["config"]["model"]["include_artist_summary"])
        if recorded != include_artist_summary:
            raise RuntimeError("Cannot resume another artist-summary ablation branch")
        tokenizer.load_state_dict(resume_state["tokenizer"], strict=True)
        start_step = int(resume_state["step"])

    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        betas=tuple(training.get("betas", [0.9, 0.999])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True) and device.startswith("cuda")),
    )
    if resume_state is not None and not (
        resume_source == "initial"
        and bool(training.get("reset_optimizer_on_initial_checkpoint", False))
    ):
        optimizer.load_state_dict(resume_state["optimizer"])
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])

    parameters = sum(parameter.numel() for parameter in tokenizer.parameters())
    print(
        "dual-query StyleTokenizer "
        f"architecture={architecture} "
        f"summary={include_artist_summary} trainable={parameters / 1e6:.2f}M "
        f"output={tokenizer.output_tokens}x{tokenizer.dim} "
        "injection=native-text-context",
        flush=True,
    )
    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        base_id = str(wandb_cfg.get("id", "dual-query-style-tokenizer-v1"))
        base_name = str(wandb_cfg.get("name", base_id))
        run_revision = str(training.get("wandb_run_revision", ""))
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=base_name + wandb_suffix + run_revision,
            id=base_id + wandb_suffix + run_revision,
            resume="allow" if start_step else "never",
            config={
                "dual_query_style_tokenizer": cfg,
                "parameters": parameters,
                "cache": cache_summary,
            },
        )

    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    base_lr = float(training.get("learning_rate", 1e-4))
    warmup = int(training.get("warmup_steps", 500))
    minimum_ratio = float(training.get("minimum_lr_ratio", 0.1))
    lr_schedule_total_steps = int(training.get("lr_schedule_total_steps", steps))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    validation_batches = int(training.get("validation_batches", 8))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    sample_every = int(training.get("sample_every", 500))
    fixed_sample_every = int(training.get("fixed_sample_every", 1000))
    sample_modes = [str(mode) for mode in training.get("sample_modes", ["heldout"])]
    unsupported_modes = set(sample_modes) - {"self", "heldout", "wrong_artist"}
    if unsupported_modes:
        raise ValueError(f"Unsupported sample modes: {sorted(unsupported_modes)}")
    train_sample_episodes = _select_sample_episodes(train_loader, 4)
    validation_sample_episodes = _select_sample_episodes(validation_loader, 4)
    sample_requests = [
        (f"train-{mode}-{index}", train_loader, episode, mode)
        for mode in sample_modes
        for index, episode in enumerate(train_sample_episodes)
    ] + [
        (f"validation-{mode}-{index}", validation_loader, episode, mode)
        for mode in sample_modes
        for index, episode in enumerate(validation_sample_episodes)
    ]
    fixed_prepared = (
        load_dual_query_external_sample(config, destination)
        if fixed_sample_every > 0
        else None
    )
    prefetched = train_loader.prefetch(
        start_step * accumulation,
        max(0, steps - start_step) * accumulation,
        workers=int(training.get("prefetch_workers", 2)),
        depth=int(training.get("prefetch_batches", 4)),
    )
    functional_every = max(1, int(training.get("functional_probe_every", 2)))
    functional_start = int(training.get("functional_probe_start_step", 501))
    functional_prefetched = None
    if functional_probe_enabled:
        assert functional_loader is not None
        functional_batches = (steps - start_step + functional_every - 1) // functional_every
        functional_prefetched = functional_loader.prefetch(
            start_step // functional_every,
            functional_batches,
            workers=int(training.get("functional_prefetch_workers", 1)),
            depth=int(training.get("functional_prefetch_batches", 2)),
        )
    native_teacher_every = max(1, int(training.get("native_teacher_every", 1)))
    native_teacher_prefetched = None
    if native_teacher_enabled:
        assert native_teacher_loader is not None
        native_batches = (
            steps - start_step + native_teacher_every - 1
        ) // native_teacher_every
        native_teacher_prefetched = native_teacher_loader.prefetch(
            start_step // native_teacher_every,
            native_batches,
            workers=int(training.get("native_teacher_prefetch_workers", 1)),
            depth=int(training.get("native_teacher_prefetch_batches", 2)),
        )
    dual_domain_prefetched: dict[str, Any] = {}
    saved_trainer_state = (
        dict(resume_state.get("trainer_state", {}))
        if resume_state is not None else {}
    )
    teacher_update_index = int(
        saved_trainer_state.get(
            "dual_domain_teacher_update_index",
            sum(
                1
                for previous_step in range(1, start_step + 1)
                if previous_step
                % _scheduled_teacher_every(previous_step, training)
                == 0
            ),
        )
    )
    if dual_domain_enabled:
        remaining = sum(
            1
            for future_step in range(start_step + 1, steps + 1)
            if future_step % _scheduled_teacher_every(future_step, training) == 0
        )
        for domain, loader in dual_domain_train_loaders.items():
            domain_cfg = dict(dual_domain_cfg[domain.removesuffix("_teacher")])
            dual_domain_prefetched[domain] = loader.prefetch(
                teacher_update_index,
                remaining,
                workers=int(domain_cfg.get("prefetch_workers", 1)),
                depth=int(domain_cfg.get("prefetch_batches", 4)),
            )
    completed = start_step
    run_started = time.perf_counter()
    running: dict[str, float] = defaultdict(float)
    running_counts: dict[str, int] = defaultdict(int)
    running_steps = 0
    try:
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            schedule_start = int(training.get("lr_schedule_start_step", 0))
            multiplier = _learning_rate_multiplier(
                step - schedule_start,
                lr_schedule_total_steps - schedule_start,
                warmup,
                minimum_ratio,
            )
            optimizer.param_groups[0]["lr"] = base_lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            micro_rows: list[dict[str, torch.Tensor]] = []
            for micro in range(accumulation):
                batch = next(prefetched)
                generator = torch.Generator(device=device).manual_seed(
                    seed + step * 100_003 + micro
                )
                loss, metrics = _forward_dual_query_flow(
                    anima,
                    tokenizer,
                    batch,
                    device,
                    training,
                    generator=generator,
                    step=step,
                    measure_base=(step == 1 or step % log_every == 0),
                )
                consistency_weight = (
                    float(training.get("subset_consistency_weight", 0.0))
                    if (
                        step >= int(training.get("subset_consistency_start", 2001))
                        and step
                        % max(1, int(training.get("subset_consistency_every", 4)))
                        == 0
                    )
                    else 0.0
                )
                consistency = loss.new_zeros(())
                if consistency_weight > 0:
                    consistency = _subset_consistency(tokenizer, batch, device)
                    loss = loss + consistency_weight * consistency
                metrics["subset_consistency"] = consistency.detach()
                metrics["subset_consistency_weight"] = loss.new_tensor(
                    consistency_weight
                )
                metrics["subset_consistency_weighted_loss"] = (
                    consistency_weight * consistency
                ).detach()
                metrics["loss"] = loss.detach()
                metrics["main_auxiliary_weighted_loss"] = (
                    loss.detach() - metrics["flow_loss"]
                )
                metrics["total_auxiliary_weighted_loss"] = metrics[
                    "main_auxiliary_weighted_loss"
                ]
                (loss / accumulation).backward()
                micro_rows.append(metrics)
            if (
                functional_probe_enabled
                and step >= functional_start
                and step % functional_every == 0
            ):
                assert functional_prefetched is not None
                functional_batch = next(functional_prefetched)
                functional_loss, functional_metrics = _pilot_functional_probe_step(
                    anima,
                    tokenizer,
                    functional_batch,
                    device,
                    training,
                    generator=torch.Generator(device=device).manual_seed(
                        seed ^ 0xC011_0A7 ^ step
                    ),
                    step=step,
                )
                if functional_loss.requires_grad:
                    functional_loss.backward()
                for row in micro_rows:
                    row.update(functional_metrics)
                    row["loss"] = row["loss"] + functional_loss.detach()
                    row["total_auxiliary_weighted_loss"] = (
                        row["loss"] - row["flow_loss"]
                    )
            if native_teacher_enabled and step % native_teacher_every == 0:
                assert native_teacher_prefetched is not None
                assert native_teacher_bank is not None
                teacher_batch = next(native_teacher_prefetched)
                teacher_loss, teacher_metrics = _native_centered_teacher_step(
                    anima,
                    tokenizer,
                    native_teacher_bank,
                    teacher_batch,
                    device,
                    training,
                    step=step,
                )
                if teacher_loss.requires_grad:
                    teacher_loss.backward()
                for row in micro_rows:
                    row.update(teacher_metrics)
                    row["loss"] = row["loss"] + teacher_loss.detach()
                    row["total_auxiliary_weighted_loss"] = (
                        row["loss"] - row["flow_loss"]
                    )
            current_teacher_every = _scheduled_teacher_every(step, training)
            if dual_domain_enabled and step % current_teacher_every == 0:
                assert dual_domain_bank is not None
                teacher_gradient_scale = _scheduled_teacher_gradient_scale(
                    current_teacher_every, training
                )
                domain_batches = {
                    domain: next(dual_domain_prefetched[domain])
                    for domain in ("human_teacher", "synthetic_teacher")
                }
                if bool(dual_domain_cfg.get("fuse_domain_forward", False)):
                    domain_loss, domain_metrics = _dual_domain_centered_teacher_step(
                        anima,
                        tokenizer,
                        dual_domain_bank,
                        domain_batches,
                        device,
                        dual_domain_training,
                        step=step,
                        probe_index_override=teacher_update_index,
                    )
                    scaled_domain_loss = domain_loss * teacher_gradient_scale
                    if scaled_domain_loss.requires_grad:
                        scaled_domain_loss.backward()
                    for row in micro_rows:
                        row.update(domain_metrics)
                        row["loss"] = row["loss"] + scaled_domain_loss.detach()
                        row["total_auxiliary_weighted_loss"] = (
                            row["loss"] - row["flow_loss"]
                        )
                else:
                    for domain in ("human_teacher", "synthetic_teacher"):
                        domain_loss, domain_metrics = _native_centered_teacher_step(
                            anima,
                            tokenizer,
                            dual_domain_bank,
                            domain_batches[domain],
                            device,
                            dual_domain_training[domain],
                            step=step,
                            metric_prefix=domain,
                            probe_index_override=teacher_update_index,
                        )
                        scaled_domain_loss = (
                            domain_loss * teacher_gradient_scale
                        )
                        if scaled_domain_loss.requires_grad:
                            scaled_domain_loss.backward()
                        for row in micro_rows:
                            row.update(domain_metrics)
                            row["loss"] = (
                                row["loss"] + scaled_domain_loss.detach()
                            )
                            row["total_auxiliary_weighted_loss"] = (
                                row["loss"] - row["flow_loss"]
                            )
                teacher_update_index += 1
                for row in micro_rows:
                    row["dual_domain_teacher_every"] = row["flow_loss"].new_tensor(
                        current_teacher_every
                    )
                    row["dual_domain_teacher_gradient_scale"] = row[
                        "flow_loss"
                    ].new_tensor(teacher_gradient_scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                tokenizer.parameters(), max_grad_norm
            )
            optimizer.step()
            for key in micro_rows[0]:
                if all(key in row for row in micro_rows):
                    running[key] += float(
                        torch.stack([row[key] for row in micro_rows]).mean()
                    )
                    running_counts[key] += 1
            running["grad_norm"] += float(grad_norm)
            running_counts["grad_norm"] += 1
            running["step_s"] += time.perf_counter() - step_started
            running_counts["step_s"] += 1
            running_steps += 1
            if step == 1 or step % log_every == 0:
                averaged = {
                    key: value / running_counts[key]
                    for key, value in running.items()
                }
                averaged["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                print(
                    f"dual-query-style step={step}/{steps} "
                    f"summary={include_artist_summary} "
                    f"loss={averaged['loss']:.5f} "
                    f"flow={averaged['flow_loss']:.5f} "
                    f"paired={averaged.get('paired_flow_improvement', 0.0):.5f} "
                    f"same={averaged.get('functional_same_artist_cosine', 0.0):.3f} "
                    f"between={averaged.get('functional_between_artist_cosine', 0.0):.3f} "
                    f"common={averaged.get('common_output_ratio', 0.0):.3f} "
                    f"centered={averaged.get('functional_centered_effect_ratio', 0.0):.3f} "
                    f"func_w={averaged.get('functional_probe_per_step_loss', 0.0):.5f} "
                    f"teacher_cos={averaged.get('native_teacher_cosine', 0.0):.3f} "
                    f"teacher_proj={averaged.get('native_teacher_projection_coefficient', 0.0):.3f} "
                    f"human_cos={averaged.get('human_teacher_cosine', 0.0):.3f} "
                    f"human_proj={averaged.get('human_teacher_projection_coefficient', 0.0):.3f} "
                    f"synth_cos={averaged.get('synthetic_teacher_cosine', 0.0):.3f} "
                    f"synth_proj={averaged.get('synthetic_teacher_projection_coefficient', 0.0):.3f} "
                    f"refs={averaged['references']:.2f} "
                    f"target={averaged['target_inclusion']:.3f} "
                    f"step_s={averaged['step_s']:.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in averaged.items()},
                        step=step,
                    )
                running.clear()
                running_counts.clear()
                running_steps = 0

            if validation_every and (step % validation_every == 0 or step == steps):
                validation_self = _evaluate(
                    anima, tokenizer, validation_loader, device, training,
                    step=step, batches=validation_batches,
                    seed=seed ^ 0xBEEF, mode="self",
                )
                validation_heldout = _evaluate(
                    anima, tokenizer, validation_loader, device, training,
                    step=step, batches=validation_batches,
                    seed=seed ^ 0xC0FFEE, mode="heldout",
                )
                validation_wrong = _evaluate(
                    anima, tokenizer, validation_loader, device, training,
                    step=step, batches=validation_batches,
                    seed=seed ^ 0xC0FFEE, mode="wrong_artist",
                )
                row = {
                    "step": step,
                    "validation_self": validation_self,
                    "validation_heldout": validation_heldout,
                    "validation_wrong_artist": validation_wrong,
                    "correct_vs_wrong_paired_advantage": (
                        validation_heldout["paired_flow_improvement"]
                        - validation_wrong["paired_flow_improvement"]
                    ),
                }
                if (
                    native_teacher_enabled
                    and native_teacher_validation_loader is not None
                ):
                    assert native_teacher_bank is not None
                    teacher_validation_batch = (
                        native_teacher_validation_loader.load_step(
                            step // max(1, validation_every)
                        )
                    )
                    with torch.no_grad():
                        _, teacher_validation = _native_centered_teacher_step(
                            anima,
                            tokenizer,
                            native_teacher_bank,
                            teacher_validation_batch,
                            device,
                            training,
                            step=step,
                        )
                    row["validation_native_teacher"] = {
                        key: float(value)
                        for key, value in teacher_validation.items()
                    }
                if dual_domain_enabled:
                    assert dual_domain_bank is not None
                    teacher_validation_batches = int(
                        dual_domain_cfg.get("validation_batches", 16)
                    )
                    for domain in ("human_teacher", "synthetic_teacher"):
                        row[f"validation_{domain}"] = (
                            _evaluate_native_teacher_domain(
                                anima,
                                tokenizer,
                                dual_domain_bank,
                                dual_domain_validation_loaders[domain],
                                device,
                                dual_domain_training[domain],
                                metric_prefix=domain,
                                batches=teacher_validation_batches,
                                step=step,
                            )
                        )
                extended_every = int(
                    training.get("extended_evaluation_every", 1000)
                )
                if pilot_enabled and step % extended_every == 0:
                    reference_metrics = {
                        str(count): _evaluate(
                            anima,
                            tokenizer,
                            loader,
                            device,
                            training,
                            step=step,
                            batches=int(
                                training.get("reference_evaluation_batches", 4)
                            ),
                            seed=(seed ^ 0x51A7) + count * 1009,
                            mode="heldout",
                        )
                        for count, loader in reference_eval_loaders.items()
                    }
                    assert controlled_loader is not None
                    controlled_metrics = _evaluate_controlled_artist_consistency(
                        anima,
                        tokenizer,
                        controlled_loader,
                        device,
                        dict(training.get("controlled_evaluation", {})),
                        seed=seed ^ 0xC017_2026,
                    )
                    row["reference_count_evaluation"] = reference_metrics
                    row["controlled_artist_consistency"] = controlled_metrics
                row["selection_score"] = _selection_score(row)
                history.append(row)
                write_json(history_path, history)
                print(f"dual-query-style validation step={step} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            **{
                                f"validation_self/{key}": value
                                for key, value in validation_self.items()
                            },
                            **{
                                f"validation_heldout/{key}": value
                                for key, value in validation_heldout.items()
                            },
                            **{
                                f"validation_wrong_artist/{key}": value
                                for key, value in validation_wrong.items()
                            },
                            "validation/correct_vs_wrong_paired_advantage": row[
                                "correct_vs_wrong_paired_advantage"
                            ],
                            "validation/selection_score": row["selection_score"],
                            **{
                                f"validation_native_teacher/{key}": value
                                for key, value in row.get(
                                    "validation_native_teacher", {}
                                ).items()
                            },
                            **{
                                f"validation_human_teacher/{key}": value
                                for key, value in row.get(
                                    "validation_human_teacher", {}
                                ).items()
                            },
                            **{
                                f"validation_synthetic_teacher/{key}": value
                                for key, value in row.get(
                                    "validation_synthetic_teacher", {}
                                ).items()
                            },
                            **(
                                {
                                    f"validation_references_{count}/{key}": value
                                    for count, values in row.get(
                                        "reference_count_evaluation", {}
                                    ).items()
                                    for key, value in values.items()
                                }
                            ),
                            **{
                                f"validation_controlled/{key}": value
                                for key, value in row.get(
                                    "controlled_artist_consistency", {}
                                ).items()
                            },
                        },
                        step=step,
                    )

            if checkpoint_every and (step % checkpoint_every == 0 or step == steps):
                _save_state(
                    checkpoints / f"step-{step:07d}.pt",
                    step=step,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                    cache_summary=cache_summary,
                    trainer_state={
                        "dual_domain_teacher_update_index": teacher_update_index,
                        "dual_domain_teacher_every": _scheduled_teacher_every(
                            step, training
                        ),
                        "dual_domain_teacher_schedule": training.get(
                            "dual_domain_teacher_schedule", []
                        ),
                    },
                )
                _save_state(
                    state_path,
                    step=step,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                    cache_summary=cache_summary,
                    trainer_state={
                        "dual_domain_teacher_update_index": teacher_update_index,
                        "dual_domain_teacher_every": _scheduled_teacher_every(
                            step, training
                        ),
                        "dual_domain_teacher_schedule": training.get(
                            "dual_domain_teacher_schedule", []
                        ),
                    },
                )
            if sample_every and (step % sample_every == 0 or step == steps):
                sheets = _sample_panels(
                    anima, tokenizer, sample_requests, config, destination,
                    output, device, step,
                    config_section="dual_query_style_tokenizer",
                )
                print(f"dual-query-style samples step={step} panels={len(sheets)}", flush=True)
                if wandb_run is not None:
                    import wandb

                    wandb_run.log(
                        {
                            "samples/panel": [
                                wandb.Image(str(path), caption=path.stem) for path in sheets
                            ]
                        },
                        step=step,
                    )
            if fixed_prepared is not None and step % fixed_sample_every == 0:
                fixed = generate_live_external_style_sample(
                    fixed_prepared,
                    config,
                    destination,
                    anima,
                    tokenizer,
                    output,
                    device,
                    step,
                )
                print(
                    f"dual-query-style fixed-reference samples step={step} "
                    f"sheet={fixed['sheet']}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log(
                        {
                            "samples/fixed_reference": wandb.Image(
                                fixed["sheet"], caption=f"fixed references step {step}"
                            ),
                            "samples/fixed_reference_mean_pixel_rms": fixed[
                                "mean_pixel_rms_from_baseline"
                            ],
                        },
                        step=step,
                    )
            completed = step
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    result = {
        "steps": completed,
        "requested_steps": steps,
        "include_artist_summary": include_artist_summary,
        "parameters": parameters,
        "cache": cache_summary,
        "elapsed_s": time.perf_counter() - run_started,
        "best_validation": max(history, key=_selection_score) if history else None,
        "last_validation": history[-1] if history else None,
        "output_directory": str(output.resolve()),
    }
    write_json(output / "summary.json", result)
    return result


def train_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["dual_query_style_tokenizer"]
    selected = cfg.get("selected_include_artist_summary")
    if selected is None:
        ablation_path = destination / str(cfg["ablation"]["output_directory"]) / "summary.json"
        if not ablation_path.exists():
            raise RuntimeError("Run dual-query-style-ablate before the full training")
        selected = bool(json.loads(ablation_path.read_text("utf-8"))["selected_include_artist_summary"])
    device = str(cfg["training"].get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(cfg["training"].get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        config,
        destination,
        anima,
        include_artist_summary=bool(selected),
        output_name=str(cfg["output_directory"]),
    )


def train_dual_query_style_tokenizer_pilot(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Run the summary-ON 10k pilot without mutating historical A/B settings."""

    cfg = config["dual_query_style_tokenizer"]
    pilot = copy.deepcopy(cfg["pilot"])
    training_overrides = copy.deepcopy(pilot["training"])
    training_overrides["pilot_enabled"] = True
    training_overrides["steps"] = int(pilot.get("steps", 10_000))
    device = str(training_overrides.get("device", cfg["training"].get("device", "cuda")))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            training_overrides.get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            training_overrides.get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        config,
        destination,
        anima,
        include_artist_summary=True,
        output_name=str(pilot["output_directory"]),
        steps_override=int(pilot.get("steps", 10_000)),
        training_overrides=training_overrides,
        wandb_suffix="-10k-pilot",
    )


def train_dual_query_exact_self_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train an isolated exact-self model before enabling residual distillation."""

    cfg = config["dual_query_style_tokenizer"]
    teacher = copy.deepcopy(cfg["exact_self_teacher"])
    training_overrides = copy.deepcopy(teacher["training"])
    training_overrides["pilot_enabled"] = True
    training_overrides["functional_probe_enabled"] = False
    training_overrides["steps"] = int(teacher.get("steps", 3_000))
    device = str(
        training_overrides.get("device", cfg["training"].get("device", "cuda"))
    )
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            training_overrides.get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            training_overrides.get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        config,
        destination,
        anima,
        include_artist_summary=True,
        output_name=str(teacher["output_directory"]),
        steps_override=int(teacher.get("steps", 3_000)),
        training_overrides=training_overrides,
        wandb_suffix="-exact-self-teacher",
    )


def train_hierarchical_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train the target-excluded hierarchical 16-token production candidate."""

    cfg = copy.deepcopy(config["hierarchical_dual_query_style_tokenizer"])
    effective_config = copy.deepcopy(config)
    effective_config["dual_query_style_tokenizer"] = cfg
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    anima = _resolve_anima_model(effective_config, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training.get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective_config,
        destination,
        anima,
        include_artist_summary=True,
        output_name=str(cfg["output_directory"]),
        steps_override=int(training.get("steps", 10_000)),
        cfg_override=cfg,
    )


def _train_compact_dual_query_style_tokenizer_section(
    config: dict[str, Any], destination: Path, section: str
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[section])
    effective_config = copy.deepcopy(config)
    effective_config["dual_query_style_tokenizer"] = cfg
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    anima = _resolve_anima_model(effective_config, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training.get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective_config,
        destination,
        anima,
        include_artist_summary=True,
        output_name=str(cfg["output_directory"]),
        steps_override=int(training.get("steps", 8_000)),
        cfg_override=cfg,
    )


def train_compact_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train the flow-dominant compact baseline on Dual-query caches."""

    return _train_compact_dual_query_style_tokenizer_section(
        config, destination, "compact_dual_query_style_tokenizer"
    )


def train_aligned_compact_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train the reference-balanced, functionally aligned compact tokenizer."""

    return _train_compact_dual_query_style_tokenizer_section(
        config, destination, "aligned_compact_dual_query_style_tokenizer"
    )


def _native_teacher_continuation_config(
    config: dict[str, Any], *, smoke: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(config["aligned_compact_dual_query_style_tokenizer"])
    continuation = copy.deepcopy(cfg.pop("native_teacher_continuation"))
    cfg["output_directory"] = str(continuation["output_directory"])
    training = dict(cfg["training"])
    training.update(dict(continuation["training"]))
    if smoke:
        cfg["output_directory"] += "_smoke"
        start = int(continuation.get("start_step", 10_000))
        training.update(
            {
                "steps": start + 2,
                "resume": False,
                "log_every": 1,
                "validation_every": 0,
                "checkpoint_every": 0,
                "sample_every": 0,
                "fixed_sample_every": 0,
                "extended_evaluation_every": 0,
                "native_teacher_ramp_steps": 0,
                "wandb": {"enabled": False},
            }
        )
    cfg["training"] = training
    effective = copy.deepcopy(config)
    effective["dual_query_style_tokenizer"] = cfg
    return effective, cfg


def _run_native_teacher_continuation(
    config: dict[str, Any], destination: Path, *, smoke: bool
) -> dict[str, Any]:
    effective, cfg = _native_teacher_continuation_config(config, smoke=smoke)
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    anima = _resolve_anima_model(effective, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training.get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective,
        destination,
        anima,
        include_artist_summary=True,
        output_name=str(cfg["output_directory"]),
        steps_override=int(training["steps"]),
        cfg_override=cfg,
    )


def train_native_teacher_compact_continuation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Continue the 10k compact model on centered native artist effects."""

    return _run_native_teacher_continuation(config, destination, smoke=False)


def smoke_test_native_teacher_compact_continuation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_native_teacher_continuation(config, destination, smoke=True)


def _run_dual_domain_native_distillation(
    config: dict[str, Any], destination: Path, *, smoke: bool
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["dual_domain_style_distillation"])
    output_name = str(cfg["output_directory"])
    if smoke:
        output_name += "_smoke"
        cfg["training"].update(
            {
                "steps": 2,
                "log_every": 1,
                "validation_every": 0,
                "checkpoint_every": 0,
                "sample_every": 0,
                "fixed_sample_every": 0,
                "extended_evaluation_every": 0,
                "resume": False,
                "wandb": {"enabled": False},
            }
        )
        for domain in ("human", "synthetic"):
            cfg["training"]["dual_domain_teacher"][domain]["ramp_steps"] = 0
    effective = copy.deepcopy(config)
    effective["dual_query_style_tokenizer"] = cfg
    device = str(cfg["training"].get("device", "cuda"))
    anima = _resolve_anima_model(effective, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            cfg["training"].get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective,
        destination,
        anima,
        include_artist_summary=True,
        output_name=output_name,
        cfg_override=cfg,
    )


def train_dual_domain_native_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_dual_domain_native_distillation(config, destination, smoke=False)


def _run_global_query_multimode_style_tokenizer(
    config: dict[str, Any], destination: Path, *, smoke: bool
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["global_query_multimode_style_tokenizer"])
    output_name = str(cfg["output_directory"])
    if smoke:
        output_name += "_smoke"
        cfg["training"].update(
            {
                "steps": 2,
                "log_every": 1,
                "validation_every": 0,
                "checkpoint_every": 1,
                "sample_every": 0,
                "fixed_sample_every": 0,
                "extended_evaluation_every": 0,
                "resume": False,
                "wandb": {"enabled": False},
            }
        )
        cfg["training"]["dual_domain_teacher"]["every"] = 1
        cfg["training"]["common_output_start_step"] = 1
        cfg["training"]["common_output_ramp_end_step"] = 1
        for domain in ("human", "synthetic"):
            cfg["training"]["dual_domain_teacher"][domain]["ramp_steps"] = 0
    effective = copy.deepcopy(config)
    effective["dual_query_style_tokenizer"] = cfg
    device = str(cfg["training"].get("device", "cuda"))
    anima = _resolve_anima_model(effective, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            cfg["training"].get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective,
        destination,
        anima,
        include_artist_summary=True,
        output_name=output_name,
        steps_override=int(cfg["training"].get("steps", 8_000)),
        cfg_override=cfg,
    )


def train_global_query_multimode_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_global_query_multimode_style_tokenizer(
        config, destination, smoke=False
    )


def smoke_test_global_query_multimode_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_global_query_multimode_style_tokenizer(
        config, destination, smoke=True
    )


def _run_slot_preserving_global_query_style_tokenizer(
    config: dict[str, Any], destination: Path, *, smoke: bool
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["slot_preserving_global_query_style_tokenizer"])
    output_name = str(cfg["output_directory"])
    if smoke:
        output_name += "_smoke"
        training = cfg["training"]
        training.update(
            {
                "steps": 2,
                "log_every": 1,
                "validation_every": 0,
                "checkpoint_every": 1,
                "sample_every": 0,
                "fixed_sample_every": 0,
                "extended_evaluation_every": 0,
                "resume": False,
                "wandb": {"enabled": False},
                "dual_domain_teacher_schedule": [
                    {"end_step": 2, "every": 1}
                ],
                "centered_energy_ramp_end_step": 1,
                "centered_energy_weight_ramp_end_step": 1,
                "common_output_start_step": 1,
                "common_output_ramp_end_step": 1,
                "artist_teacher_contrastive_start_step": 1,
                "artist_teacher_contrastive_ramp_end_step": 1,
                "artist_teacher_ranking_start_step": 1,
                "artist_teacher_ranking_ramp_end_step": 1,
            }
        )
        training["dual_domain_teacher"]["every"] = 1
        for domain in ("human", "synthetic"):
            training["dual_domain_teacher"][domain]["ramp_steps"] = 0
    effective = copy.deepcopy(config)
    effective["dual_query_style_tokenizer"] = cfg
    device = str(cfg["training"].get("device", "cuda"))
    anima = _resolve_anima_model(effective, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            cfg["training"].get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective,
        destination,
        anima,
        include_artist_summary=True,
        output_name=output_name,
        steps_override=int(cfg["training"].get("steps", 8_000)),
        cfg_override=cfg,
    )


def train_slot_preserving_global_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_slot_preserving_global_query_style_tokenizer(
        config, destination, smoke=False
    )


def smoke_test_slot_preserving_global_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_slot_preserving_global_query_style_tokenizer(
        config, destination, smoke=True
    )


def _run_typed_multi_descriptor_style_tokenizer(
    config: dict[str, Any], destination: Path, *, smoke: bool
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["typed_multi_descriptor_compact_style_tokenizer"])
    output_name = str(cfg["output_directory"])
    if smoke:
        output_name += "_smoke"
        training = cfg["training"]
        training.update(
            {
                "steps": 2,
                "log_every": 1,
                "validation_every": 0,
                "checkpoint_every": 1,
                "sample_every": 0,
                "fixed_sample_every": 0,
                "extended_evaluation_every": 0,
                "resume": False,
                "wandb": {"enabled": False},
                "dual_domain_teacher_schedule": [
                    {"end_step": 2, "every": 1}
                ],
                "teacher_projected_effect_ramp_steps": 0,
                "teacher_projection_floor_end_step": 1,
                "functional_probe_start_step": 1,
                "functional_probe_ramp_steps": 0,
                "functional_probe_every": 1,
                "functional_value_diversity_start_step": 1,
                "functional_value_diversity_ramp_steps": 0,
                "artist_contrastive_start_step": 1,
                "artist_contrastive_ramp_steps": 0,
            }
        )
        training["dual_domain_teacher"]["every"] = 1
        for domain in ("human", "synthetic"):
            training["dual_domain_teacher"][domain]["ramp_steps"] = 0
    effective = copy.deepcopy(config)
    effective["dual_query_style_tokenizer"] = cfg
    device = str(cfg["training"].get("device", "cuda"))
    anima = _resolve_anima_model(effective, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(
            cfg["training"].get("low_precision_rmsnorm", True)
        ),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective,
        destination,
        anima,
        include_artist_summary=True,
        output_name=output_name,
        steps_override=int(cfg["training"].get("steps", 8_000)),
        cfg_override=cfg,
    )


def train_typed_multi_descriptor_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_typed_multi_descriptor_style_tokenizer(
        config, destination, smoke=False
    )


def smoke_test_typed_multi_descriptor_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_typed_multi_descriptor_style_tokenizer(
        config, destination, smoke=True
    )


def smoke_test_dual_domain_native_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _run_dual_domain_native_distillation(config, destination, smoke=True)


def _smoke_test_compact_dual_query_style_tokenizer_section(
    config: dict[str, Any], destination: Path, section: str
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[section])
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    training = dict(cfg["training"])
    training.update(
        {
            "steps": 2,
            "resume": False,
            "log_every": 1,
            "validation_every": 0,
            "checkpoint_every": 0,
            "sample_every": 0,
            "fixed_sample_every": 0,
            "extended_evaluation_every": 0,
            "wandb": {"enabled": False},
        }
    )
    if section == "aligned_compact_dual_query_style_tokenizer":
        # Exercise every delayed objective with two disjoint two-reference
        # views instead of letting a two-step smoke test cover only flow MSE.
        training.update(
            {
                "reference_schedule": [
                    {
                        "name": "smoke_four_references",
                        "end_step": 2,
                        "exact_self": False,
                        "min_references": 1,
                        "max_references": 4,
                        "reference_count_weights": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
                "token_contrastive_start_step": 1,
                "token_contrastive_ramp_steps": 0,
                "wrong_ranking_start_step": 1,
                "wrong_ranking_ramp_steps": 0,
                "wrong_ranking_every": 1,
                "artist_contrastive_every": 1,
                "subset_consistency_start": 1,
                "subset_consistency_every": 1,
                "functional_probe_start_step": 1,
                "functional_probe_ramp_steps": 0,
                "functional_probe_every": 1,
            }
        )
    cfg["training"] = training
    effective_config = copy.deepcopy(config)
    effective_config["dual_query_style_tokenizer"] = cfg
    device = str(training.get("device", "cuda"))
    anima = _resolve_anima_model(effective_config, destination, device).requires_grad_(
        False
    ).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training.get("fuse_attention_projections", True)
        ),
    )
    return _train_variant(
        effective_config,
        destination,
        anima,
        include_artist_summary=True,
        output_name=str(cfg["output_directory"]),
        steps_override=2,
        cfg_override=cfg,
    )


def smoke_test_compact_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Exercise two real-cache compact-tokenizer optimizer steps."""

    return _smoke_test_compact_dual_query_style_tokenizer_section(
        config, destination, "compact_dual_query_style_tokenizer"
    )


def smoke_test_aligned_compact_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Exercise the aligned compact recipe on real production caches."""

    return _smoke_test_compact_dual_query_style_tokenizer_section(
        config, destination, "aligned_compact_dual_query_style_tokenizer"
    )


def compare_artist_summary_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["dual_query_style_tokenizer"]
    ablation = dict(cfg["ablation"])
    device = str(cfg["training"].get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(cfg["training"].get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            cfg["training"].get("fuse_attention_projections", True)
        ),
    )
    root = str(ablation["output_directory"])
    steps = int(ablation.get("steps", 1000))
    overrides = dict(ablation.get("training_overrides", {}))
    query_only = _train_variant(
        config,
        destination,
        anima,
        include_artist_summary=False,
        output_name=f"{root}/query_only",
        steps_override=steps,
        training_overrides=overrides,
        wandb_suffix="-summary-off",
    )
    with_summary = _train_variant(
        config,
        destination,
        anima,
        include_artist_summary=True,
        output_name=f"{root}/with_summary",
        steps_override=steps,
        training_overrides=overrides,
        wandb_suffix="-summary-on",
    )
    candidates = {"query_only": query_only, "with_summary": with_summary}
    selected_name = max(
        candidates,
        key=lambda name: float(candidates[name]["best_validation"]["selection_score"]),
    )
    summary = {
        "steps_per_variant": steps,
        "query_only": query_only,
        "with_summary": with_summary,
        "selected": selected_name,
        "selected_include_artist_summary": selected_name == "with_summary",
        "selection_rule": (
            "heldout paired improvement + 0.5*(heldout-wrong) + 0.25*self"
        ),
    }
    output = destination / root
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_dual_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    torch.manual_seed(23)
    results = {}
    for include_summary in (False, True):
        model = DualQuerySetStyleTokenizer(
            dim=32,
            query_tokens=8,
            artist_summary_tokens=2,
            include_artist_summary=include_summary,
            output_tokens=4,
            heads=4,
            cross_layers=1,
            cross_slot_layers=1,
            ff_dim=64,
        )
        references = torch.randn(3, 4, 10, 32, requires_grad=True)
        mask = torch.tensor(
            [[True, False, False, False], [True, True, False, False], [True] * 4]
        )
        output = model(references, mask)
        loss = output.tokens.float().square().mean()
        loss.backward()
        results[str(include_summary)] = {
            "shape": list(output.tokens.shape),
            "loss": float(loss.detach()),
            "finite_gradient": bool(torch.isfinite(references.grad).all()),
        }
    summary = {"variants": results, "finite": all(row["finite_gradient"] for row in results.values())}
    write_json(destination / "dual_query_style_tokenizer_smoke.json", summary)
    return summary


def smoke_test_dual_query_style_tokenizer_pilot(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Exercise exact, ranking, and common-output pilot branches on real caches."""

    cfg = copy.deepcopy(config["dual_query_style_tokenizer"])
    pilot = copy.deepcopy(cfg["pilot"])
    training = copy.deepcopy(cfg["training"])
    training.update(copy.deepcopy(pilot["training"]))
    training["pilot_enabled"] = True
    training["steps"] = int(pilot.get("steps", 10_000))
    cfg["training"] = training
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260816))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader = DualQueryCachedStyleLoader(
        destination,
        _loader_config(config, cfg, split=str(cfg.get("train_split", "train"))),
    )
    functional_cfg = _loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    functional_cfg.update(
        {
            "batch_size": int(training.get("functional_probe_batch_rows", 4)),
            "min_references": int(training.get("functional_probe_references", 4)),
            "max_references": int(training.get("functional_probe_references", 4)),
            "reference_count_weights": None,
            "reference_curriculum": {},
            "pilot_reference_schedule": [],
            "artist_balanced": True,
        }
    )
    functional_loader = DualQueryCachedStyleLoader(destination, functional_cfg)
    cache_summary = _cache_summary(destination, cfg)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training.get("fuse_attention_projections", True)
        ),
    )
    tokenizer = DualQuerySetStyleTokenizer(**dict(cfg["model"])).to(device).train()
    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    rows = []
    for step in (1, 1000, 1504):
        tokenizer.zero_grad(set_to_none=True)
        batch = loader.load_step((step - 1) * accumulation)
        loss, metrics = _forward_dual_query_flow(
            anima,
            tokenizer,
            batch,
            device,
            training,
            generator=torch.Generator(device=device).manual_seed(seed + step),
            step=step,
            measure_base=True,
        )
        if step >= int(training.get("subset_consistency_start", 501)):
            consistency = _subset_consistency(tokenizer, batch, device)
            loss = loss + float(training.get("subset_consistency_weight", 0.0)) * consistency
            metrics["subset_consistency"] = consistency.detach()
        loss.backward()
        reported_loss = loss.detach()
        if step % max(1, int(training.get("functional_probe_every", 2))) == 0:
            functional_batch = functional_loader.load_step(step)
            functional_loss, functional_metrics = _pilot_functional_probe_step(
                anima,
                tokenizer,
                functional_batch,
                device,
                training,
                generator=torch.Generator(device=device).manual_seed(
                    seed ^ 0xC011_0A7 ^ step
                ),
                step=step,
            )
            if functional_loss.requires_grad:
                functional_loss.backward()
                reported_loss = reported_loss + functional_loss.detach()
            metrics.update(functional_metrics)
        gradients = [
            parameter.grad.detach().float()
            for parameter in tokenizer.parameters()
            if parameter.grad is not None
        ]
        grad_norm = torch.stack(
            [gradient.square().sum() for gradient in gradients]
        ).sum().sqrt()
        row = {
            "step": step,
            "phase": str(_pilot_stage(step, training)["name"]),
            "loss": float(reported_loss),
            "grad_norm": float(grad_norm),
            "finite": bool(
                torch.isfinite(loss).all()
                and torch.isfinite(grad_norm).all()
                and all(torch.isfinite(gradient).all() for gradient in gradients)
            ),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        rows.append(row)
        print(f"dual-query pilot smoke {row}", flush=True)
    summary = {
        "finite": all(row["finite"] for row in rows),
        "rows": rows,
        "cache": cache_summary,
    }
    write_json(destination / "dual_query_style_tokenizer_pilot_smoke.json", summary)
    return summary
