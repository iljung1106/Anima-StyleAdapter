from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .dual_query_style_tokenizer import (
    DualQueryCachedStyleLoader,
    DualQuerySetStyleTokenizer,
)
from .hierarchical_dual_query_style_tokenizer import (
    HierarchicalDualQueryStyleTokenizer,
)
from .compact_dual_query_style_tokenizer import CompactDualQueryStyleTokenizer
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


def _same_artist_functional_loss(
    first_deltas: torch.Tensor,
    second_deltas: torch.Tensor,
    valid: torch.Tensor,
    *,
    direction_fraction: float,
    huber_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match two disjoint reference views in frozen-Anima velocity space."""

    if first_deltas.shape != second_deltas.shape:
        raise ValueError("Functional reference views must have the same shape")
    if valid.shape != first_deltas.shape[:1]:
        raise ValueError("Functional reference-view validity has the wrong shape")
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
    weights = valid.to(direction.dtype)
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


def _native_centered_teacher_step(
    anima: torch.nn.Module,
    tokenizer: torch.nn.Module,
    bank: NativeCenteredTeacherBank,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    *,
    step: int,
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
    probe_index = max(0, step - int(training.get("native_teacher_start_step", 0)))
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
    alignment, metrics = _native_teacher_alignment_loss(
        student,
        teacher,
        huber_beta=float(training.get("native_teacher_huber_beta", 0.10)),
        scale_floor=float(training.get("native_teacher_scale_floor", 1e-4)),
        direction_weight=float(
            training.get("native_teacher_direction_weight", 0.10)
        ),
        magnitude_weight=float(
            training.get("native_teacher_magnitude_weight", 0.05)
        ),
    )
    start = int(training.get("native_teacher_start_step", 1))
    ramp_steps = max(0, int(training.get("native_teacher_ramp_steps", 250)))
    weight = _linear_ramp(
        step,
        start_step=start,
        end_step=start + ramp_steps,
        start=0.0 if ramp_steps else float(training.get("native_teacher_weight", 0.10)),
        end=float(training.get("native_teacher_weight", 0.10)),
    )
    weighted = float(weight) * alignment
    metrics.update(
        {
            "native_teacher_alignment_loss": alignment.detach(),
            "native_teacher_weight": alignment.new_tensor(weight),
            "native_teacher_weighted_loss": weighted.detach(),
            "native_teacher_content_index": alignment.new_tensor(content_index),
            "native_teacher_timestep": timestep.detach().float(),
        }
    )
    return weighted, metrics


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
    common_weight = weight("common_output_weight", 0.03)
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
            include_target = torch.zeros(
                references.shape[0], dtype=torch.bool, device=device
            )
        curriculum = {
            "target_only": bool(stage.get("exact_self", False)),
            "target_probability": float(bool(stage.get("exact_self", False))),
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

    diversity_weight = float(training.get("slot_diversity_weight", 0.0))
    diversity_source = getattr(output, "diversity_tokens", output.tokens)
    diversity = _slot_diversity_loss(diversity_source)
    total = total + diversity_weight * diversity

    contrastive = flow_loss.new_zeros(())
    positive = flow_loss.new_zeros(())
    negative = flow_loss.new_zeros(())
    contrastive_weight = 0.0
    contrastive_every = max(1, int(training.get("artist_contrastive_every", 2)))
    if (
        float(training.get("artist_contrastive_weight", 0.0)) > 0
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
        contrastive_weight = float(training["artist_contrastive_weight"])
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

    train_loader = DualQueryCachedStyleLoader(
        destination,
        _loader_config(config, cfg, split=str(cfg.get("train_split", "train"))),
    )
    validation_loader = DualQueryCachedStyleLoader(
        destination,
        _loader_config(
            config, cfg, split=str(cfg.get("validation_split", "validation"))
        ),
    )
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
            reference_eval_loaders[count] = DualQueryCachedStyleLoader(
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
        controlled_loader = DualQueryCachedStyleLoader(destination, controlled_cfg)
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
    cache_summary = _cache_summary(destination, cfg)
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "flat_set"))
    if architecture == "flat_set":
        tokenizer = DualQuerySetStyleTokenizer(**model_cfg).to(device)
    elif architecture == "hierarchical":
        tokenizer = HierarchicalDualQueryStyleTokenizer(**model_cfg).to(device)
    elif architecture == "compact":
        tokenizer = CompactDualQueryStyleTokenizer(**model_cfg).to(device)
    else:
        raise ValueError(f"Unknown Dual-query StyleTokenizer architecture {architecture!r}")
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
                steps - schedule_start,
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
                )
                _save_state(
                    state_path,
                    step=step,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                    cache_summary=cache_summary,
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
