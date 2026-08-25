"""Production runner for detail-preserving typed-slot Style Cross-Attention."""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .artist_effect_losses import (
    centered_functional_artist_loss,
    common_output_and_artist_magnitude_loss,
    episodic_artist_prototype_loss,
)
from .detail_style_cross_attention import (
    ArtistOnlySharedBaseKVStyleCrossAttention,
    DetailPreservingTypedSlotReader,
    FreshKVStyleCrossAttention,
    SeparatedCommonArtistKVStyleCrossAttention,
    SharedBaseKVStyleCrossAttention,
)
from .detail_style_teacher_context import NativeArtistContextCache
from .data_mixture import ConstantRatioBatchMixer
from .dual_query_external_samples import load_dual_query_external_sample
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .dual_query_style_training import (
    _artist_flow_ranking_loss,
    _build_native_effect_timestep_weighting,
    _native_effect_weights_for_timesteps,
)
from .global_query_style_tokenizer import MultiPromptDualQueryCachedStyleLoader
from .external_style_tokenizer_sheet import (
    _decode_latents,
    _fit_with_padding,
    _make_sheet,
    _pixel_rms_from_baseline,
)
from .io import read_records, write_json
from .native_centered_teacher import NativeCenteredTeacherBank
from .pure_token_injection import (
    _reference_batch,
    _replace_reference_with_target,
)
from .query_style_tokenizer import (
    _reference_inputs,
    _sample_query_style_tokenizer,
    _select_sample_episodes,
)
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_tokenizer import _flow_metrics, _mean_metrics
from .style_tokenizer import _split_reference_views
from .style_transfer import (
    _optimize_frozen_anima,
    _resolve_anima_model,
    _sample_flow_timesteps,
)


def _build_style_adapter(cfg: dict[str, Any]) -> FreshKVStyleCrossAttention:
    adapter_cfg = dict(cfg["adapter"])
    architecture = str(adapter_cfg.pop("architecture", "fresh_per_block"))
    if architecture == "separated_common_artist_shared_base":
        return SeparatedCommonArtistKVStyleCrossAttention(**adapter_cfg)
    if architecture == "artist_only_shared_base":
        for name in (
            "common_tokens",
            "artist_null_residual",
            "artist_residual_gain",
            "common_residual_gain",
            "common_combined_gradient_scale",
        ):
            adapter_cfg.pop(name, None)
        return ArtistOnlySharedBaseKVStyleCrossAttention(**adapter_cfg)
    if architecture == "shared_base_lora":
        return SharedBaseKVStyleCrossAttention(**adapter_cfg)
    if architecture == "fresh_per_block":
        return FreshKVStyleCrossAttention(**adapter_cfg)
    raise ValueError(f"Unsupported style adapter architecture: {architecture}")


def _lora_teacher_schedule_for_step(
    training: dict[str, Any],
    step: int,
    default_schedule: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the staged mixture curriculum for a continuation step."""

    if step <= int(training.get("single_only_steps", 0)):
        return ("single",)
    curriculum = list(training.get("teacher_kind_curriculum", []))
    previous_end = int(training.get("single_only_steps", 0))
    for stage in curriculum:
        end_step = int(stage["end_step"])
        schedule = tuple(str(value) for value in stage["schedule"])
        if end_step <= previous_end or not schedule:
            raise ValueError(
                "LoRA teacher_kind_curriculum needs increasing end_step values "
                "and non-empty schedules"
            )
        if step <= end_step:
            return schedule
        previous_end = end_step
    if not default_schedule:
        raise ValueError("LoRA functional teacher schedule cannot be empty")
    return default_schedule


def _lora_backward_scale_for_step(
    training: dict[str, Any], step: int
) -> float:
    """Ramp the auxiliary LoRA backward contribution without staging teachers."""

    final = float(training.get("backward_scale", 0.25))
    initial = float(training.get("backward_scale_start", final))
    ramp_steps = int(training.get("backward_scale_ramp_steps", 0))
    if initial < 0 or final < 0:
        raise ValueError("LoRA backward scales must be non-negative")
    if ramp_steps <= 1:
        return final
    progress = min(1.0, max(0.0, (int(step) - 1) / (ramp_steps - 1)))
    return initial + progress * (final - initial)


def _teacher_domain_update(
    weighted_domains: tuple[int, ...], update: int
) -> tuple[int, int]:
    """Map a global teacher update to one domain and its local update index."""

    if not weighted_domains:
        raise ValueError("Teacher domain schedule must not be empty")
    if update < 0:
        raise ValueError("Teacher update must be non-negative")
    cycle, offset = divmod(int(update), len(weighted_domains))
    domain = weighted_domains[offset]
    per_cycle = weighted_domains.count(domain)
    local_update = cycle * per_cycle + weighted_domains[:offset].count(domain)
    return domain, local_update


def _teacher_reference_count_weights(
    training: dict[str, Any], step: int
) -> tuple[float, ...] | None:
    """Linearly introduce low-reference teacher batches without a hard shift."""

    curriculum = dict(training.get("teacher_reference_count_curriculum", {}))
    if not curriculum:
        configured = training.get("teacher_reference_count_weights")
        return (
            None
            if configured is None
            else tuple(float(value) for value in configured)
        )
    start_step = int(curriculum["start_step"])
    end_step = int(curriculum["end_step"])
    if end_step < start_step:
        raise ValueError(
            "teacher_reference_count_curriculum end_step must be >= start_step"
        )
    before = tuple(float(value) for value in curriculum["before_weights"])
    start = tuple(float(value) for value in curriculum["start_weights"])
    end = tuple(float(value) for value in curriculum["end_weights"])
    if not (len(before) == len(start) == len(end)):
        raise ValueError(
            "teacher reference-count curriculum weight vectors must match"
        )
    if step < start_step:
        return before
    progress = min(
        1.0,
        max(0.0, (step - start_step) / max(1, end_step - start_step)),
    )
    return tuple(
        left + progress * (right - left)
        for left, right in zip(start, end, strict=True)
    )


def _teacher_infonce_weight(
    training: dict[str, Any], reference_count: int
) -> float:
    """Give noisy low-reference batches stronger artist discrimination."""

    by_count = training.get("teacher_infonce_weight_by_reference_count")
    if by_count is None:
        return float(training.get("teacher_infonce_weight", 0.25))
    weights = tuple(float(value) for value in by_count)
    if not 1 <= int(reference_count) <= len(weights):
        raise ValueError(
            "teacher_infonce_weight_by_reference_count does not cover the "
            f"sampled count {reference_count}"
        )
    return weights[int(reference_count) - 1]


def _checkpoint_has_artist_null_residual(
    state: dict[str, Any] | None,
) -> bool:
    """Whether the actually loaded checkpoint already owns the null baseline."""

    if state is None:
        return False
    config = dict(state.get("config", {}))
    return bool(dict(config.get("adapter", {})).get("artist_null_residual", False))


def _delayed_learning_rate_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_start_step: int,
    minimum_ratio: float,
) -> float:
    """Warm up, hold peak LR through alignment, then cosine decay."""

    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio must be between 0 and 1")
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    decay_start_step = max(warmup_steps, min(int(decay_start_step), total_steps))
    if warmup_steps and step <= warmup_steps:
        return max(1, int(step)) / warmup_steps
    if step <= decay_start_step:
        return 1.0
    progress = min(
        1.0,
        max(0.0, (int(step) - decay_start_step) / max(1, total_steps - decay_start_step)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _separated_bootstrap_phase(
    step: int,
    training: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> str:
    config = dict(training.get("separated_component_bootstrap", {}))
    if not bool(config.get("enabled", False)):
        return "combined"
    transition = dict(config.get("common_transition", {}))
    if bool(transition.get("enabled", False)):
        return (
            "combined"
            if bool((state or {}).get("separated_common_complete", False))
            else "common_only"
        )
    common_steps = int(config.get("common_steps", 500))
    if common_steps <= 0:
        raise ValueError("separated common bootstrap needs common_steps > 0")
    return "common_only" if int(step) <= common_steps else "combined"


def _configure_separated_bootstrap_trainability(
    reader: torch.nn.Module,
    adapter: FreshKVStyleCrossAttention,
    phase: str,
    *,
    train_common_in_combined: bool = False,
) -> None:
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        return
    adapter.set_bootstrap_phase(phase)
    common_active = phase == "common_only" or (
        phase == "combined" and bool(train_common_in_combined)
    )
    artist_active = phase != "common_only"
    for parameter in adapter.common_parameters():
        parameter.requires_grad_(common_active)
    for parameter in adapter.artist_parameters():
        parameter.requires_grad_(artist_active)
    for parameter in reader.parameters():
        parameter.requires_grad_(artist_active)


def _loader_config(
    config: dict[str, Any], cfg: dict[str, Any], *, split: str
) -> dict[str, Any]:
    result = dict(config["style_transfer"]["loader"])
    result.update(dict(cfg["loader"]))
    result["split"] = split
    result["resampler_token_cache"] = str(cfg["cache"]["output_directory"])
    if split == str(cfg.get("train_split", "train")):
        performance_cfg = dict(
            cfg["training"].get("performance_curriculum", {})
        )
        if bool(performance_cfg.get("enabled", False)):
            stages = list(performance_cfg.get("stages", []))
            if not stages:
                raise ValueError("Performance curriculum requires stages")
            maximum = max(int(stage["max_references"]) for stage in stages)
            result["reference_curriculum"] = {}
            result["pilot_reference_schedule"] = [{
                "name": "performance_prefetch_maximum",
                "end_step": int(cfg["training"]["steps"]),
                "min_references": maximum,
                "max_references": maximum,
                "reference_count_weights": [0.0] * (maximum - 1) + [1.0],
            }]
        else:
            result["reference_curriculum"] = dict(cfg["training"]["curriculum"])
            result["pilot_reference_schedule"] = list(
                cfg["training"]["reference_schedule"]
            )
    else:
        result["reference_curriculum"] = {}
        result["pilot_reference_schedule"] = []
        result["self_reference_target_images_per_style"] = 0
        result["prompt_modes"] = dict(
            result.get("validation_prompt_modes", {"full": 1.0})
        )
        result["quality_probability"] = float(
            result.get("validation_quality_probability", 0.0)
        )
    return result


def _performance_stages(training: dict[str, Any]) -> list[dict[str, Any]]:
    config = dict(training.get("performance_curriculum", {}))
    if not bool(config.get("enabled", False)):
        return []
    stages = [dict(value) for value in config.get("stages", [])]
    if not stages:
        raise ValueError("Performance curriculum is enabled without stages")
    for stage in stages:
        lower = int(stage["min_references"])
        upper = int(stage["max_references"])
        weights = [float(value) for value in stage["reference_count_weights"]]
        if lower < 1 or upper < lower or len(weights) < upper:
            raise ValueError(f"Invalid performance curriculum stage {stage}")
        active = weights[lower - 1 : upper]
        if any(value < 0 for value in active) or sum(active) <= 0:
            raise ValueError(f"Empty reference distribution in stage {stage}")
        probability = float(stage["target_probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid target probability in stage {stage}")
    return stages


def _initial_performance_curriculum_state(
    training: dict[str, Any], resume: dict[str, Any] | None
) -> dict[str, Any]:
    stages = _performance_stages(training)
    restored = dict((resume or {}).get("performance_curriculum", {}))
    if not stages:
        return {
            "enabled": False,
            "stage_index": 0,
            "consecutive_passes": 0,
            "teacher_update": int(restored.get("teacher_update", 0)),
            "native_bootstrap_consecutive": int(
                restored.get("native_bootstrap_consecutive", 0)
            ),
            "native_bootstrap_complete": bool(
                restored.get("native_bootstrap_complete", False)
            ),
            "separated_common_complete": bool(
                restored.get("separated_common_complete", False)
            ),
            "separated_common_consecutive": int(
                restored.get("separated_common_consecutive", 0)
            ),
            "separated_common_transition_step": int(
                restored.get("separated_common_transition_step", 0)
            ),
            "separated_common_metrics": dict(
                restored.get("separated_common_metrics", {})
            ),
        }
    stage_index = min(
        max(0, int(restored.get("stage_index", 0))), len(stages) - 1
    )
    return {
        "enabled": True,
        "stage_index": stage_index,
        "consecutive_passes": int(restored.get("consecutive_passes", 0)),
        "transition_step": int(restored.get("transition_step", 0)),
        "teacher_update": int(restored.get("teacher_update", 0)),
        "last_metrics": dict(restored.get("last_metrics", {})),
    }


def _active_performance_stage(
    training: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any] | None:
    stages = _performance_stages(training)
    if not stages:
        return None
    return stages[int(state["stage_index"])]


def _performance_reference_batch(
    batch: dict[str, Any],
    device: str,
    stage: dict[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the current validated curriculum stage to prefetched max references."""

    target_tokens = batch["cached_target_tokens"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    if bool(stage.get("target_only", False)):
        mask = torch.ones(target_tokens.shape[0], 1, dtype=torch.bool, device=device)
        return target_tokens[:, None], mask, mask[:, 0], {
            "phase": str(stage["name"]),
            "target_only": True,
            "target_probability": 1.0,
        }

    references, source_mask = _reference_inputs(batch, device, "heldout")
    lower = int(stage["min_references"])
    upper = int(stage["max_references"])
    configured_weights = [
        float(value) for value in stage["reference_count_weights"]
    ]
    mask = torch.zeros_like(source_mask)
    for row in range(source_mask.shape[0]):
        available = int(source_mask[row].sum())
        row_upper = min(upper, available)
        row_lower = min(lower, row_upper)
        counts = torch.arange(row_lower, row_upper + 1, device=device)
        weights = torch.tensor(
            configured_weights[row_lower - 1 : row_upper],
            device=device,
            dtype=torch.float32,
        )
        selected = int(
            counts[
                torch.multinomial(weights, 1, generator=generator)
            ].item()
        )
        mask[row, :selected] = True

    target_probability = float(stage["target_probability"])
    include_target = torch.rand(
        target_tokens.shape[0], device=device, generator=generator
    ) < target_probability
    prompt_modes = [str(value) for value in batch.get("prompt_modes", [])]
    if prompt_modes:
        empty = torch.tensor(
            [value == "empty" for value in prompt_modes],
            device=device,
            dtype=torch.bool,
        )
        include_target = include_target | empty
        for row, is_empty in enumerate(empty.tolist()):
            if is_empty:
                mask[row].zero_()
                mask[row, 0] = True
    references, mask = _replace_reference_with_target(
        references, mask, target_tokens, include_target
    )
    return references, mask, include_target, {
        "phase": str(stage["name"]),
        "target_only": False,
        "target_probability": target_probability,
    }


def _update_performance_curriculum(
    training: dict[str, Any],
    state: dict[str, Any],
    validation: dict[str, dict[str, float]],
    teacher_rows: list[dict[str, float]],
    *,
    step: int,
) -> tuple[dict[str, float], bool]:
    """Advance only after consecutive validation windows meet every criterion."""

    stage = _active_performance_stage(training, state)
    if stage is None:
        return {}, False
    block_cosines = [
        value
        for row in teacher_rows
        for key, value in row.items()
        if key.startswith("post_gate_teacher_block_")
        and key.endswith("_cosine")
    ]
    projections = [
        row["native_teacher_projection_coefficient"]
        for row in teacher_rows
        if "native_teacher_projection_coefficient" in row
    ]
    final_cosines = [
        row["native_teacher_cosine"]
        for row in teacher_rows
        if "native_teacher_cosine" in row
    ]
    common_ratios = [
        row["native_teacher_common_output_ratio"]
        for row in teacher_rows
        if "native_teacher_common_output_ratio" in row
    ]
    infonce_accuracies = [
        row["teacher_infonce_accuracy"]
        for row in teacher_rows
        if "teacher_infonce_accuracy" in row
    ]
    artist = validation["artist_effect"]
    teacher_rms = max(
        float(artist.get("functional_artist_centered_teacher_rms", 0.0)), 1e-8
    )
    metrics = {
        "post_gate_cosine": float(np.median(block_cosines))
        if block_cosines else -1.0,
        "final_centered_cosine": float(np.median(final_cosines))
        if final_cosines else -1.0,
        "native_projection": float(np.mean(projections)) if projections else -1.0,
        "correct_wrong_advantage": float(
            validation["heldout"].get("paired_flow_improvement", 0.0)
            - validation["wrong_artist"].get("paired_flow_improvement", 0.0)
        ),
        "common_output_ratio": (
            float(np.median(common_ratios)) if common_ratios else float(
                artist.get("functional_artist_common_output_ratio", float("inf"))
            )
        ),
        "teacher_infonce_accuracy": float(np.mean(infonce_accuracies))
        if infonce_accuracies else 0.0,
        "centered_rms_ratio": float(
            artist.get("functional_artist_centered_student_rms", 0.0)
        ) / teacher_rms,
        "heldout_paired_improvement": float(
            validation["heldout"].get("paired_flow_improvement", -float("inf"))
        ),
    }
    advance = dict(stage.get("advance", {}))
    passed = int(state["stage_index"]) + 1 < len(_performance_stages(training))
    passed = passed and step >= int(advance.get("minimum_step", 0))
    passed = passed and metrics["final_centered_cosine"] >= float(
        advance.get("final_centered_cosine", -float("inf"))
    )
    passed = passed and metrics["native_projection"] >= float(
        advance.get("native_projection", -float("inf"))
    )
    passed = passed and metrics["common_output_ratio"] <= float(
        advance.get("common_output_ratio", float("inf"))
    )
    required = int(advance.get("consecutive_validations", 2))
    state["consecutive_passes"] = (
        int(state["consecutive_passes"]) + 1 if passed else 0
    )
    changed = bool(passed and int(state["consecutive_passes"]) >= required)
    if changed:
        state["stage_index"] = min(
            int(state["stage_index"]) + 1,
            len(_performance_stages(training)) - 1,
        )
        state["consecutive_passes"] = 0
        state["transition_step"] = int(step)
    state["last_metrics"] = metrics
    metrics.update({
        "stage_index": float(state["stage_index"]),
        "consecutive_passes": float(state["consecutive_passes"]),
        "stage_changed": float(changed),
    })
    return metrics, changed


def _audit_student_prompts(loader: MultiPromptDualQueryCachedStyleLoader) -> None:
    violations = []
    for row in loader.text_by_key.values():
        caption = str(row.get("caption", ""))
        artist = " ".join(str(row.get("artist", "")).replace("_", " ").split())
        tags = {value.strip().casefold() for value in caption.split(",")}
        artist_tag = f"@{artist}".casefold()
        explicit_artist_tags = {
            value for value in tags
            if value.startswith("@") and value not in {"@", "@ @"}
        }
        if explicit_artist_tags:
            violations.append(
                (row.get("id"), sorted(explicit_artist_tags), caption[:160])
            )
        if artist and ({artist.casefold(), artist_tag} & tags):
            violations.append((row.get("id"), artist, caption[:160]))
        if len(violations) >= 8:
            break
    if violations:
        raise RuntimeError(f"Artist leakage in student text cache: {violations}")


def _training_loader(
    destination: Path,
    cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[MultiPromptDualQueryCachedStyleLoader, Any]:
    """Build the primary loader and optional constant-ratio external mixer."""

    primary = MultiPromptDualQueryCachedStyleLoader(destination, train_cfg)
    mixture = dict(cfg.get("data_mixture", {}))
    if not bool(mixture.get("enabled", False)):
        return primary, primary
    auxiliary_cfg = dict(train_cfg)
    auxiliary_cfg.update(dict(mixture["loader"]))
    auxiliary = MultiPromptDualQueryCachedStyleLoader(destination, auxiliary_cfg)
    mixed = ConstantRatioBatchMixer(
        primary,
        auxiliary,
        auxiliary_fraction=float(mixture.get("auxiliary_fraction", 0.15)),
        primary_name=str(mixture.get("primary_name", "anima")),
        auxiliary_name=str(mixture.get("auxiliary_name", "megastyle")),
    )
    return primary, mixed


def _reconstruction_loss(
    prediction: torch.Tensor | None, target: torch.Tensor | None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction is None or target is None:
        raise RuntimeError("Reader reconstruction was not requested")
    normalized_prediction = F.normalize(prediction.float(), dim=-1)
    normalized_target = F.normalize(target.float(), dim=-1)
    cosine = 1.0 - (normalized_prediction * normalized_target).sum(dim=-1).mean()
    huber = F.smooth_l1_loss(prediction.float(), target.float(), beta=0.1)
    return cosine + 0.1 * huber, {
        "reconstruction_cosine_loss": cosine.detach(),
        "reconstruction_huber_loss": huber.detach(),
    }


def _reconstruction_weight(step: int, training: dict[str, Any]) -> float:
    start = float(training.get("reconstruction_weight", 0.05))
    end = float(training.get("reconstruction_final_weight", 0.01))
    hold = int(training.get("reconstruction_hold_steps", 2_000))
    total = int(training["steps"])
    if step <= hold:
        return start
    progress = min(1.0, (step - hold) / max(1, total - hold))
    return start + progress * (end - start)


def _ramp(step: int, start: int, end: int, maximum: float) -> float:
    if step < start or maximum <= 0:
        return 0.0
    return float(maximum) * min(1.0, (step - start + 1) / max(1, end - start + 1))


def _scheduled_value(
    step: int, start: int, end: int, initial: float, final: float
) -> float:
    if step <= start:
        return float(initial)
    progress = min(1.0, (step - start) / max(1, end - start))
    return float(initial) + progress * (float(final) - float(initial))


class NativeScaleCommonOutputPenalty:
    """Instantaneous common-effect energy normalized by a frozen native scale."""

    def objective(
        self,
        student: torch.Tensor,
        native_teacher: torch.Tensor,
        *,
        ratio_threshold: float,
        artist_energy_floor: float = 0.0,
        artist_energy_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        native_float = native_teacher.detach().float()
        reduce_dims = tuple(range(1, native_float.ndim))
        native_row_rms = native_float.square().mean(dim=reduce_dims).sqrt()
        native_scale = native_row_rms.median().clamp_min(1e-8)
        return self.objective_from_scale(
            student,
            native_scale,
            ratio_threshold=ratio_threshold,
            artist_energy_floor=artist_energy_floor,
            artist_energy_weight=artist_energy_weight,
            artist_teacher=native_float,
        )

    def objective_from_scale(
        self,
        student: torch.Tensor,
        native_scale: torch.Tensor,
        *,
        ratio_threshold: float,
        artist_energy_floor: float = 0.0,
        artist_energy_weight: float = 1.0,
        artist_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Bound common energy while preserving artist-centered energy."""

        if ratio_threshold < 0 or artist_energy_floor < 0:
            raise ValueError("Common threshold and artist floor must be non-negative")
        if artist_energy_weight < 0:
            raise ValueError("Artist-energy weight must be non-negative")

        student_float = student.float()
        batch_common = student_float.mean(dim=0)
        centered = student_float - batch_common.unsqueeze(0)
        reduce_dims = tuple(range(1, centered.ndim))
        batch_energy = batch_common.square().mean()
        native_scale_batch = native_scale.detach().float().clamp_min(1e-8)
        # Stabilize the backward path while reporting the mathematically exact
        # detached RMS (including an exact zero) in diagnostics.
        rms_epsilon = batch_energy.new_tensor(1e-12)
        batch_rms = (batch_energy + rms_epsilon).sqrt()
        batch_rms_metric = batch_energy.detach().sqrt()
        ratio = batch_rms / native_scale_batch
        centered_energy = centered.square().mean(dim=reduce_dims)
        centered_row_rms = (
            (centered_energy + rms_epsilon).sqrt()
        )
        centered_ratio = centered_row_rms.mean() / native_scale_batch
        common_loss = F.relu(ratio - float(ratio_threshold)).square()
        if artist_teacher is not None:
            teacher = artist_teacher.detach().float()
            teacher = teacher - teacher.mean(dim=0, keepdim=True)
            teacher_power = teacher.square().mean(dim=reduce_dims).clamp_min(1e-8)
            artist_projection = (
                centered * teacher
            ).mean(dim=reduce_dims) / teacher_power
            artist_energy_loss = F.relu(
                float(artist_energy_floor) - artist_projection
            ).square().mean()
            artist_energy_ratio = artist_projection.mean()
            artist_energy_uses_teacher_direction = 1.0
        else:
            artist_energy_loss = F.relu(
                float(artist_energy_floor) - centered_ratio
            ).square()
            artist_energy_ratio = centered_ratio
            artist_energy_uses_teacher_direction = 0.0
        loss = common_loss + float(artist_energy_weight) * artist_energy_loss
        return loss, {
            "native_teacher_common_output_batch_ratio": (
                batch_rms_metric / native_scale_batch
            ),
            "native_teacher_common_output_ratio": (
                batch_rms_metric / native_scale_batch
            ),
            "native_teacher_common_output_batch_rms": batch_rms_metric,
            "native_teacher_common_output_scale": native_scale_batch,
            "native_teacher_common_output_scale_batch": native_scale_batch.detach(),
            "native_teacher_common_output_common_loss": common_loss.detach(),
            "native_teacher_artist_energy_ratio": artist_energy_ratio.detach(),
            "native_teacher_artist_centered_rms_ratio": centered_ratio.detach(),
            "native_teacher_artist_energy_uses_teacher_direction": (
                centered_ratio.new_tensor(artist_energy_uses_teacher_direction)
            ),
            "native_teacher_artist_energy_floor": centered_ratio.new_tensor(
                float(artist_energy_floor)
            ),
            "native_teacher_artist_energy_loss": artist_energy_loss.detach(),
            "native_teacher_artist_energy_weight": centered_ratio.new_tensor(
                float(artist_energy_weight)
            ),
            "native_teacher_common_output_loss": loss.detach(),
        }


def _weighted_artist_effect_objective(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    style_ids: list[str],
    training: dict[str, Any],
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    magnitude_teacher = str(
        training.get("artist_magnitude_teacher", "exact_target")
    )
    if magnitude_teacher not in {
        "exact_target", "disjoint_heldout", "native_centered_bank",
    }:
        raise ValueError(
            "artist_magnitude_teacher must be exact_target, "
            "disjoint_heldout, or native_centered_bank"
        )
    pool_scales = tuple(
        int(value)
        for value in training.get("artist_effect_pool_scales", [2, 4])
    )
    effect, metrics = centered_functional_artist_loss(
        teacher_delta,
        student_delta,
        style_ids,
        temperature=float(training.get("artist_effect_temperature", 0.10)),
        pool_scales=pool_scales,
        repeatability_weight=float(
            training.get("artist_effect_repeatability_weight", 0.25)
        ),
        repeatability_floor=(
            float(training["artist_effect_repeatability_floor"])
            if training.get("artist_effect_repeatability_floor") is not None
            else None
        ),
        contrastive_mode=str(
            training.get("artist_effect_contrastive_mode", "symmetric_nce")
        ),
        positive_floor=float(
            training.get("artist_effect_positive_floor", 0.25)
        ),
        wrong_margin=float(
            training.get("artist_effect_wrong_margin", 0.10)
        ),
    )
    common_threshold = _scheduled_value(
        step,
        int(training.get("common_output_threshold_start_step", 250)),
        int(training.get("common_output_threshold_end_step", 2_000)),
        float(training.get("common_output_threshold_start", 0.90)),
        float(training.get("common_output_threshold_end", 0.65)),
    )
    magnitude_lower = _scheduled_value(
        step,
        int(training.get("artist_magnitude_lower_start_step", 250)),
        int(training.get("artist_magnitude_lower_end_step", 2_000)),
        float(training.get("artist_magnitude_lower_start", 0.35)),
        float(training.get("artist_magnitude_lower_end", 0.70)),
    )
    magnitude_upper = float(training.get("artist_magnitude_upper", 1.25))
    common_loss, magnitude_loss, regularizer_metrics = (
        common_output_and_artist_magnitude_loss(
            teacher_delta,
            student_delta,
            common_threshold=common_threshold,
            magnitude_lower=magnitude_lower,
            magnitude_upper=magnitude_upper,
            magnitude_upper_weight=float(
                training.get("artist_magnitude_upper_weight", 0.25)
            ),
            pool_scales=pool_scales,
        )
    )
    effect_weight = _ramp(
        step,
        int(training.get("artist_effect_start_step", 250)),
        int(training.get("artist_effect_full_step", 1_000)),
        float(training.get("artist_effect_weight", 0.0)),
    )
    common_output_teacher = str(
        training.get("common_output_teacher", "student_batch")
    )
    if common_output_teacher not in {"student_batch", "native_centered"}:
        raise ValueError(
            "common_output_teacher must be student_batch or native_centered"
        )
    common_weight = (
        _ramp(
            step,
            int(training.get("common_output_start_step", 250)),
            int(training.get("common_output_full_step", 1_000)),
            float(training.get("common_output_weight", 0.0)),
        )
        if common_output_teacher == "student_batch"
        else 0.0
    )
    magnitude_weight = (
        _ramp(
            step,
            int(training.get("artist_magnitude_start_step", 250)),
            int(training.get("artist_magnitude_full_step", 1_000)),
            float(training.get("artist_magnitude_weight", 0.0)),
        )
        if magnitude_teacher in {"exact_target", "disjoint_heldout"}
        else 0.0
    )
    weighted_effect = effect_weight * effect
    weighted_common = common_weight * common_loss
    weighted_magnitude = magnitude_weight * magnitude_loss
    total = weighted_effect + weighted_common + weighted_magnitude
    metrics.update(regularizer_metrics)
    metrics.update({
        "functional_artist_weight": effect.new_tensor(effect_weight),
        "functional_artist_weighted_loss": weighted_effect.detach(),
        "functional_artist_common_output_threshold": effect.new_tensor(
            common_threshold
        ),
        "functional_artist_common_output_weight": effect.new_tensor(
            common_weight
        ),
        "functional_artist_common_output_weighted_loss": (
            weighted_common.detach()
        ),
        "functional_artist_common_output_uses_native_teacher": effect.new_tensor(
            float(common_output_teacher == "native_centered")
        ),
        "functional_artist_magnitude_lower": effect.new_tensor(
            magnitude_lower
        ),
        "functional_artist_magnitude_upper": effect.new_tensor(
            magnitude_upper
        ),
        "functional_artist_magnitude_weight": effect.new_tensor(
            magnitude_weight
        ),
        "functional_artist_magnitude_uses_native_teacher": effect.new_tensor(
            float(magnitude_teacher == "native_centered_bank")
        ),
        "functional_artist_magnitude_weighted_loss": (
            weighted_magnitude.detach()
        ),
        "functional_artist_total_weighted_loss": total.detach(),
    })
    return total, metrics


def _mean_scalar_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average homogeneous scalar metric rows without flow-specific fields."""

    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    return {
        key: sum(row[key] for row in rows if key in row)
        / sum(key in row for row in rows)
        for key in keys
    }


@torch.no_grad()
def _effect_stage_metrics(
    student: torch.Tensor,
    teacher: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize an artist-conditioned residual without retaining activations."""

    student = student.detach().float()
    dimensions = tuple(range(1, student.ndim))
    row_rms = student.square().mean(dim=dimensions).sqrt()
    common = student.mean(dim=0, keepdim=True)
    centered = student - common
    centered_rms = centered.square().mean(dim=dimensions).sqrt()
    result = {
        "rms": float(row_rms.mean()),
        "centered_rms": float(centered_rms.mean()),
        "common_rms": float(common.square().mean().sqrt()),
        "common_output_ratio": float(
            common.square().mean().sqrt() / row_rms.mean().clamp_min(1e-8)
        ),
    }
    if teacher is None:
        return result
    teacher = teacher.detach().float()
    if teacher.shape != student.shape:
        raise ValueError("Student and teacher stages must have matching shapes")
    teacher = teacher - teacher.mean(dim=0, keepdim=True)
    teacher_rms = teacher.square().mean(dim=dimensions).sqrt()
    student_flat = centered.flatten(1)
    teacher_flat = teacher.flatten(1)
    teacher_energy = teacher_flat.square().sum(dim=1).clamp_min(1e-8)
    coefficient = (student_flat * teacher_flat).sum(dim=1) / teacher_energy
    projection = coefficient.reshape(-1, *([1] * (student.ndim - 1))) * teacher
    orthogonal_rms = (
        (centered - projection).square().mean(dim=dimensions).sqrt()
    )
    result.update({
        "teacher_rms": float(teacher_rms.mean()),
        "student_to_teacher_rms": float(
            (centered_rms / teacher_rms.clamp_min(1e-8)).mean()
        ),
        "teacher_projection": float(coefficient.mean()),
        "teacher_projection_positive_fraction": float(
            (coefficient > 0).float().mean()
        ),
        "teacher_direction_cosine": float(
            F.cosine_similarity(student_flat, teacher_flat, dim=1, eps=1e-8).mean()
        ),
        "orthogonal_to_teacher_rms": float(
            (orthogonal_rms / teacher_rms.clamp_min(1e-8)).mean()
        ),
    })
    return result


class _StyleAttenuationRecorder:
    """Pair base/style block stages and immediately reduce them to scalars."""

    _PAIRED_STAGES = {
        "pre_o_teacher": "pre_o_style",
        "post_o_teacher": "post_o_style",
        "post_gate_teacher": "post_gate_style",
    }
    _HIDDEN_STAGES = {
        "post_self_hidden", "post_cross_hidden", "post_mlp_hidden"
    }

    def __init__(self) -> None:
        self.mode = "base"
        self.base_hidden: dict[tuple[int, str], torch.Tensor] = {}
        self.pending: dict[tuple[int, str], torch.Tensor] = {}
        self.metrics: dict[int, dict[str, dict[str, float]]] = {}
        self.output_base: dict[str, torch.Tensor] = {}
        self.output_metrics: dict[str, dict[str, float]] = {}

    @staticmethod
    def _effect_to_base_rms(effect: torch.Tensor, base: torch.Tensor) -> float:
        dimensions = tuple(range(1, effect.ndim))
        return float(
            effect.detach().float().square().mean(dim=dimensions).sqrt().mean()
            / base.detach().float().square().mean(dim=dimensions).sqrt().mean().clamp_min(1e-8)
        )

    def __call__(self, block_index: int, stage: str, value: torch.Tensor) -> None:
        key = (int(block_index), str(stage))
        if self.mode == "base":
            if stage in self._HIDDEN_STAGES:
                self.base_hidden[key] = value.detach()
            return
        if stage.endswith("_style"):
            self.pending[key] = value.detach()
            return
        if stage in self._PAIRED_STAGES:
            style_stage = self._PAIRED_STAGES[stage]
            student = self.pending.pop((int(block_index), style_stage))
            self.metrics.setdefault(int(block_index), {})[style_stage.removesuffix("_style")] = (
                _effect_stage_metrics(student, value)
            )
            return
        if stage in self._HIDDEN_STAGES:
            base = self.base_hidden.pop(key)
            effect = value.detach() - base
            metrics = _effect_stage_metrics(effect)
            metrics["effect_to_base_rms"] = self._effect_to_base_rms(effect, base)
            self.metrics.setdefault(int(block_index), {})[stage] = metrics

    def record_output_stage(self, stage: str, value: torch.Tensor) -> None:
        if self.mode == "base":
            self.output_base[str(stage)] = value.detach()
            return
        base = self.output_base.pop(str(stage))
        effect = value.detach() - base
        metrics = _effect_stage_metrics(effect)
        metrics["effect_to_base_rms"] = self._effect_to_base_rms(effect, base)
        self.output_metrics[str(stage)] = metrics

    def finish(self) -> dict[int, dict[str, dict[str, float]]]:
        if self.pending or self.base_hidden or self.output_base:
            raise RuntimeError(
                "Incomplete attenuation capture: "
                f"pending={len(self.pending)} base={len(self.base_hidden)} "
                f"output={len(self.output_base)}"
            )
        return self.metrics


def _minimal_native_teacher_objective(
    student_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    training: dict[str, Any],
    *,
    step: int,
    student_center: torch.Tensor | None = None,
    teacher_center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align only the final artist-centered direction and its RMS band."""

    if student_delta.shape != teacher_delta.shape or student_delta.shape[0] < 2:
        raise ValueError("Minimal teacher objective needs matching artist batches")
    student = student_delta.float()
    teacher = teacher_delta.detach().float()
    resolved_student_center = (
        student.mean(dim=0, keepdim=True)
        if student_center is None else student_center.detach().float()
    )
    resolved_teacher_center = (
        teacher.mean(dim=0, keepdim=True)
        if teacher_center is None else teacher_center.detach().float()
    )
    student = student - resolved_student_center
    teacher = teacher - resolved_teacher_center
    dimensions = tuple(range(1, student.ndim))
    scale_floor = float(training.get("native_teacher_scale_floor", 1e-4))
    teacher_power = teacher.square().mean(dim=dimensions).clamp_min(
        scale_floor**2
    )
    teacher_rms = teacher_power.sqrt()
    strength_weighting = bool(
        training.get("native_strength_weighting", True)
    )
    if strength_weighting:
        positive = teacher_rms.detach()[teacher_rms.detach() > scale_floor]
        reference_strength = (
            positive.median() if positive.numel() else teacher_rms.new_tensor(1.0)
        ).clamp_min(scale_floor)
        row_weights = (teacher_rms.detach() / reference_strength).clamp(
            float(training.get("native_strength_weight_min", 0.25)),
            float(training.get("native_strength_weight_max", 4.0)),
        )
        row_weights = row_weights / row_weights.mean().clamp_min(1e-8)
    else:
        row_weights = torch.ones_like(teacher_rms)

    def weighted_mean(rows: torch.Tensor) -> torch.Tensor:
        if rows.shape != row_weights.shape:
            raise ValueError("Native teacher row objective has an invalid shape")
        return (rows * row_weights).sum() / row_weights.sum().clamp_min(1e-8)

    coefficient = (student * teacher).mean(dim=dimensions) / teacher_power
    floor_start_step = int(training.get("magnitude_floor_start_step", 1))
    floor_end_step = int(training.get("magnitude_floor_end_step", 1_000))
    floor_start = float(training.get("magnitude_floor_start", 0.25))
    floor_end = float(training.get("magnitude_floor_end", 0.70))
    if step <= floor_start_step:
        magnitude_floor = floor_start
    else:
        progress = min(
            1.0,
            (step - floor_start_step)
            / max(1, floor_end_step - floor_start_step),
        )
        magnitude_floor = floor_start + progress * (floor_end - floor_start)
    student_rms = (student.square().mean(dim=dimensions) + 1e-12).sqrt()
    rms_ratio = student_rms / teacher_rms.clamp_min(scale_floor)
    magnitude_lower_loss = weighted_mean(F.relu(magnitude_floor - rms_ratio).square())
    magnitude_upper = float(training.get("magnitude_upper", 1.50))
    if magnitude_upper < magnitude_floor:
        raise ValueError("magnitude_upper must not be below the magnitude floor")
    magnitude_upper_loss = weighted_mean(F.relu(rms_ratio - magnitude_upper).square())
    magnitude_upper_weight = float(
        training.get("magnitude_upper_weight", 0.25)
    )
    magnitude = magnitude_lower_loss + magnitude_upper_weight * magnitude_upper_loss
    cosine = F.cosine_similarity(
        student.flatten(1), teacher.flatten(1), dim=-1, eps=1e-8
    )
    direction = weighted_mean(1.0 - cosine)

    direction_weight = float(
        training.get("artist_direction_weight", training.get("direction_weight", 1.0))
    )
    magnitude_weight = float(training.get("magnitude_weight", 0.10))
    weighted_direction = direction_weight * direction
    weighted_magnitude = magnitude_weight * magnitude
    scale = teacher_rms.reshape(-1, *([1] * (teacher.ndim - 1)))
    absolute_huber = F.smooth_l1_loss(
        student / scale,
        teacher / scale,
        beta=float(training.get("artist_huber_beta", 0.10)),
        reduction="none",
    ).mean(dim=dimensions)
    absolute_huber = weighted_mean(absolute_huber)
    absolute_huber_weight = float(training.get("artist_huber_weight", 0.0))
    weighted_absolute_huber = absolute_huber_weight * absolute_huber
    total = weighted_direction + weighted_magnitude + weighted_absolute_huber
    return total, {
        "native_teacher_final_direction_loss": direction.detach(),
        "native_teacher_final_direction_weight": direction.new_tensor(
            direction_weight
        ),
        "native_teacher_final_direction_weighted_loss": weighted_direction.detach(),
        "native_teacher_magnitude_floor": direction.new_tensor(magnitude_floor),
        "native_teacher_magnitude_lower_loss": magnitude_lower_loss.detach(),
        "native_teacher_magnitude_upper": direction.new_tensor(magnitude_upper),
        "native_teacher_magnitude_upper_loss": magnitude_upper_loss.detach(),
        "native_teacher_magnitude_upper_weight": direction.new_tensor(
            magnitude_upper_weight
        ),
        "native_teacher_magnitude_band_loss": magnitude.detach(),
        "native_teacher_magnitude_weight": direction.new_tensor(magnitude_weight),
        "native_teacher_magnitude_weighted_loss": weighted_magnitude.detach(),
        "native_teacher_absolute_huber_loss": absolute_huber.detach(),
        "native_teacher_absolute_huber_weight": direction.new_tensor(
            absolute_huber_weight
        ),
        "native_teacher_absolute_huber_weighted_loss": (
            weighted_absolute_huber.detach()
        ),
        "native_teacher_artist_rms_ratio": rms_ratio.detach().mean(),
        "native_teacher_projection_coefficient": coefficient.detach().mean(),
        "native_teacher_projection_positive_fraction": (
            coefficient.detach() > 0
        ).float().mean(),
        "native_teacher_cosine": cosine.detach().mean(),
        "native_teacher_centered_student_rms": student_rms.detach().mean(),
        "native_teacher_centered_target_rms": teacher_rms.detach().mean(),
        "native_teacher_centered_student_to_target_rms": (
            student_rms.detach() / teacher_rms.detach()
        ).mean(),
        "native_teacher_strength_weight_mean": row_weights.detach().mean(),
        "native_teacher_strength_weight_max": row_weights.detach().max(),
        "native_teacher_strength_weight_min": row_weights.detach().min(),
        "native_teacher_minimal_loss": total.detach(),
    }


def _common_native_teacher_objective(
    student_common: torch.Tensor,
    teacher_common: torch.Tensor,
    training: dict[str, Any],
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align the final cross-artist common direction and a weak RMS band."""

    if student_common.shape != teacher_common.shape or student_common.shape[0] != 1:
        raise ValueError("Common teacher objective expects matching singleton means")
    student = student_common.float()
    teacher = teacher_common.detach().float()
    dimensions = tuple(range(1, student.ndim))
    scale_floor = float(training.get("native_teacher_scale_floor", 1e-4))
    teacher_power = teacher.square().mean(dim=dimensions).clamp_min(scale_floor**2)
    teacher_rms = teacher_power.sqrt()
    student_rms = (student.square().mean(dim=dimensions) + 1e-12).sqrt()
    rms_ratio = student_rms / teacher_rms
    cosine = F.cosine_similarity(
        student.flatten(1), teacher.flatten(1), dim=-1, eps=1e-8
    ).mean()
    direction = 1.0 - cosine

    floor_start_step = int(training.get("magnitude_floor_start_step", 1))
    floor_end_step = int(training.get("magnitude_floor_end_step", 1_000))
    floor_start = float(training.get("magnitude_floor_start", 0.25))
    floor_end = float(training.get("magnitude_floor_end", 0.70))
    if step <= floor_start_step:
        magnitude_floor = floor_start
    else:
        progress = min(
            1.0,
            (step - floor_start_step) / max(1, floor_end_step - floor_start_step),
        )
        magnitude_floor = floor_start + progress * (floor_end - floor_start)
    magnitude_upper = float(training.get("magnitude_upper", 1.50))
    magnitude_upper_weight = float(training.get("magnitude_upper_weight", 0.25))
    lower = F.relu(magnitude_floor - rms_ratio).square().mean()
    upper = F.relu(rms_ratio - magnitude_upper).square().mean()
    magnitude = lower + magnitude_upper_weight * upper

    direction_weight = float(training.get("common_direction_weight", 1.0))
    magnitude_weight = float(training.get("magnitude_weight", 0.10))
    weighted_direction = direction_weight * direction
    weighted_magnitude = magnitude_weight * magnitude
    coefficient = (student * teacher).mean(dim=dimensions) / teacher_power
    total = weighted_direction + weighted_magnitude
    return total, {
        "native_teacher_common_direction_loss": direction.detach(),
        "native_teacher_common_direction_weight": direction.new_tensor(
            direction_weight
        ),
        "native_teacher_common_direction_weighted_loss": weighted_direction.detach(),
        "native_teacher_common_magnitude_lower_loss": lower.detach(),
        "native_teacher_common_magnitude_upper_loss": upper.detach(),
        "native_teacher_common_magnitude_band_loss": magnitude.detach(),
        "native_teacher_common_magnitude_weight": direction.new_tensor(
            magnitude_weight
        ),
        "native_teacher_common_magnitude_weighted_loss": weighted_magnitude.detach(),
        "native_teacher_common_rms_ratio": rms_ratio.detach().mean(),
        "native_teacher_common_cosine": cosine.detach(),
        "native_teacher_common_projection_coefficient": coefficient.detach().mean(),
        "native_teacher_common_student_rms": student_rms.detach().mean(),
        "native_teacher_common_target_rms": teacher_rms.detach().mean(),
        "native_teacher_common_loss": total.detach(),
    }


def _teacher_direction_ranking_loss(
    student_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank the matching frozen artist direction above a cyclic negative."""

    if student_delta.shape != teacher_delta.shape or student_delta.shape[0] < 2:
        raise ValueError("Teacher direction ranking needs matching artist batches")
    student = student_delta.float()
    teacher = teacher_delta.detach().float()
    student = student - student.mean(dim=0, keepdim=True)
    teacher = teacher - teacher.mean(dim=0, keepdim=True)
    student = F.normalize(student.flatten(1), dim=1, eps=1e-8)
    teacher = F.normalize(teacher.flatten(1), dim=1, eps=1e-8)
    correct = (student * teacher).sum(dim=1)
    wrong = (student * teacher.roll(shifts=1, dims=0)).sum(dim=1)
    advantage = correct - wrong
    ranking = F.relu(float(margin) - advantage).mean()
    return ranking, {
        "teacher_direction_ranking_loss": ranking.detach(),
        "teacher_direction_correct_cosine": correct.detach().mean(),
        "teacher_direction_wrong_cosine": wrong.detach().mean(),
        "teacher_direction_advantage": advantage.detach().mean(),
        "teacher_direction_accuracy": advantage.detach().gt(0).float().mean(),
    }


def _all_artist_teacher_infonce(
    student_centered: torch.Tensor,
    teacher_centered: torch.Tensor,
    teacher_bank: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrast each student row against every controlled frozen artist."""

    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive")
    student = F.normalize(student_centered.float().flatten(1), dim=1, eps=1e-8)
    positive_teacher = F.normalize(
        teacher_centered.detach().float().flatten(1), dim=1, eps=1e-8
    )
    bank = F.normalize(teacher_bank.detach().float().flatten(1), dim=1, eps=1e-8)
    logits = student @ bank.transpose(0, 1) / float(temperature)
    loss = F.cross_entropy(logits, labels)
    positive = (student * positive_teacher).sum(dim=1)
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, labels[:, None], False)
    wrong_logits = logits.masked_fill(~wrong_mask, -torch.inf)
    hardest_wrong = wrong_logits.max(dim=1).values * float(temperature)
    return loss, {
        "teacher_infonce_loss": loss.detach(),
        "teacher_infonce_accuracy": logits.detach().argmax(dim=1).eq(labels).float().mean(),
        "teacher_infonce_positive_cosine": positive.detach().mean(),
        "teacher_infonce_hardest_wrong_cosine": hardest_wrong.detach().mean(),
        "teacher_infonce_cosine_gap": (
            positive.detach() - hardest_wrong.detach()
        ).mean(),
    }


def _soft_common_output_objective(
    student_mean: torch.Tensor,
    native_scale: torch.Tensor,
    *,
    ratio_threshold: float,
    softness: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Continuously suppress a 16-artist common residual without a dead hinge."""

    if ratio_threshold < 0 or softness <= 0:
        raise ValueError("Common ratio threshold must be non-negative and softness positive")
    mean = student_mean.float()
    scale = native_scale.detach().float().clamp_min(1e-8)
    ratio = (mean.square().mean() + 1e-12).sqrt() / scale
    loss = float(softness) * F.softplus(
        (ratio - float(ratio_threshold)) / float(softness)
    )
    return loss, {
        "native_teacher_common_output_ratio": ratio.detach(),
        "native_teacher_common_output_soft_loss": loss.detach(),
        "native_teacher_common_output_ratio_threshold": ratio.new_tensor(
            float(ratio_threshold)
        ),
        "native_teacher_common_output_softness": ratio.new_tensor(float(softness)),
    }


def _backward_adapter_only(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> None:
    """Accumulate one objective only into the explicitly selected leaves."""

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            continue
        detached = gradient.detach()
        if parameter.grad is None:
            parameter.grad = detached
        else:
            parameter.grad.add_(detached)


@contextmanager
def _scaled_parameter_gradients(
    parameters: list[torch.nn.Parameter], scale: float
):
    """Scale gradients created inside one auxiliary backward region only."""

    if scale < 0:
        raise ValueError("gradient scale cannot be negative")
    handles = []
    if scale != 1.0:
        for parameter in parameters:
            if parameter.requires_grad:
                handles.append(
                    parameter.register_hook(
                        lambda gradient, factor=float(scale): gradient * factor
                    )
                )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _gradient_rms(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    """Return a parameter-count-weighted gradient RMS for path health logs."""

    squared = None
    count = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        term = value.square().sum()
        squared = term if squared is None else squared + term
        count += value.numel()
    if squared is None or count == 0:
        reference = parameters[0] if parameters else torch.tensor(0.0)
        return reference.new_zeros((), dtype=torch.float32)
    return (squared / count).sqrt()


def _native_teacher_objective_config(training: dict[str, Any]) -> dict[str, Any]:
    """Resolve the final centered native-teacher objective for a teacher step."""

    objective = dict(training)
    objective.update(dict(training.get("teacher_objective", {})))
    return objective


def _native_bootstrap_status(
    rows: list[dict[str, float]],
    config: dict[str, Any],
    *,
    step: int,
    previous_consecutive: int,
    common_override: dict[str, float] | None = None,
) -> tuple[dict[str, float], int, bool]:
    """Evaluate a native-teacher bootstrap on stable window means."""

    required = {
        "native_teacher_cosine": float(config.get("final_cosine", 0.30)),
        "native_teacher_projection_coefficient": float(
            config.get("final_projection", 0.25)
        ),
        "native_teacher_common_cosine": float(
            config.get("common_cosine", 0.30)
        ),
        "native_teacher_common_projection_coefficient": float(
            config.get("common_projection", 0.25)
        ),
        "teacher_infonce_cosine_gap": float(config.get("infonce_gap", 0.02)),
    }
    means = {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in required
        if any(key in row for row in rows)
    }
    if common_override:
        for key in (
            "native_teacher_common_cosine",
            "native_teacher_common_projection_coefficient",
        ):
            if key in common_override:
                means[key] = float(common_override[key])
    enough_steps = step >= int(config.get("minimum_steps", 500))
    passed = enough_steps and all(
        means.get(key, float("-inf")) >= threshold
        for key, threshold in required.items()
    )
    consecutive = previous_consecutive + 1 if passed else 0
    required_consecutive = int(config.get("consecutive_validations", 3))
    complete = consecutive >= required_consecutive
    metrics = {
        **{f"native_bootstrap_{key}": value for key, value in means.items()},
        "native_bootstrap_minimum_steps_met": float(enough_steps),
        "native_bootstrap_window_passed": float(passed),
        "native_bootstrap_consecutive": float(consecutive),
        "native_bootstrap_required_consecutive": float(required_consecutive),
        "native_bootstrap_complete": float(complete),
    }
    return metrics, consecutive, complete


def _separated_common_transition_status(
    rows: list[dict[str, float]],
    config: dict[str, Any],
    *,
    step: int,
    previous_consecutive: int,
) -> tuple[dict[str, float], int, bool]:
    """Gate the Artist path on stable Common-only final-output alignment."""

    keys = (
        "native_teacher_common_cosine",
        "native_teacher_common_projection_coefficient",
        "native_teacher_common_rms_ratio",
    )
    means = {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
        if any(key in row for row in rows)
    }
    enough_steps = int(step) >= int(config.get("minimum_steps", 500))
    passed = (
        enough_steps
        and means.get("native_teacher_common_cosine", -float("inf"))
        >= float(config.get("cosine", 0.70))
        and means.get(
            "native_teacher_common_projection_coefficient", -float("inf")
        ) >= float(config.get("projection", 0.50))
        and means.get("native_teacher_common_rms_ratio", -float("inf"))
        >= float(config.get("rms_lower", 0.70))
        and means.get("native_teacher_common_rms_ratio", float("inf"))
        <= float(config.get("rms_upper", 1.30))
    )
    consecutive = previous_consecutive + 1 if passed else 0
    required = int(config.get("consecutive_validations", 2))
    complete = consecutive >= required
    metrics = {
        **{f"separated_common_{key}": value for key, value in means.items()},
        "separated_common_minimum_steps_met": float(enough_steps),
        "separated_common_window_passed": float(passed),
        "separated_common_consecutive": float(consecutive),
        "separated_common_required_consecutive": float(required),
        "separated_common_complete": float(complete),
    }
    return metrics, consecutive, complete


def _rho_min(step: int) -> float:
    if step <= 250:
        return 0.0
    return 0.5 * min(1.0, (step - 250) / 750)


def _main_flow_projection_floor_loss(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    step: int,
    training: dict[str, Any],
    stage_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep the style branch from winning main flow by shrinking to zero.

    The lower bound is applied only to the component of the final Anima
    velocity change that points toward the actual flow residual.  Increasing
    unrelated/orthogonal energy therefore cannot satisfy this objective.  A
    weak total-energy ceiling prevents the early exact-self bootstrap from
    destabilizing the frozen denoiser.
    """

    delta = prediction.float() - base_prediction.detach().float()
    desired = target.float() - base_prediction.detach().float()
    dimensions = tuple(range(1, delta.ndim))
    scale_floor = float(training.get("main_flow_projection_scale_floor", 1e-4))
    desired_power = desired.square().mean(dim=dimensions).clamp_min(
        scale_floor**2
    )
    coefficient = (delta * desired).mean(dim=dimensions) / desired_power
    delta_rms = delta.square().mean(dim=dimensions).clamp_min(
        scale_floor**2
    ).sqrt()
    desired_rms = desired_power.sqrt()
    rms_ratio = delta_rms / desired_rms

    start_step = int(training.get("main_flow_projection_floor_start_step", 1))
    end_step = int(training.get("main_flow_projection_floor_end_step", 500))
    floor_start = float(training.get("main_flow_projection_floor_start", 0.05))
    floor_end = float(training.get("main_flow_projection_floor_end", 0.20))
    progress = min(
        1.0,
        max(0.0, (step - start_step) / max(1, end_step - start_step)),
    )
    floor = stage_scale * (
        floor_start + progress * (floor_end - floor_start)
    )
    lower = F.relu(floor - coefficient).square().mean()
    upper = float(training.get("main_flow_projection_rms_upper", 0.75))
    upper_loss = F.relu(rms_ratio - upper).square().mean()
    upper_weight = float(
        training.get("main_flow_projection_upper_weight", 0.10)
    )
    loss = lower + upper_weight * upper_loss
    return loss, {
        "main_flow_projection_coefficient": coefficient.detach().mean(),
        "main_flow_projection_positive_fraction": (
            coefficient.detach() > 0
        ).float().mean(),
        "main_flow_projection_floor": loss.new_tensor(floor),
        "main_flow_projection_lower_loss": lower.detach(),
        "main_flow_projection_rms_ratio": rms_ratio.detach().mean(),
        "main_flow_projection_rms_upper": loss.new_tensor(upper),
        "main_flow_projection_upper_loss": upper_loss.detach(),
        "main_flow_projection_loss": loss.detach(),
    }


def _main_flow_total_magnitude_loss(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match final style-delta RMS to the complete desired-residual RMS.

    This objective deliberately ignores direction.  It prevents the frozen
    denoiser's already-good prediction from making a near-zero style branch
    the easiest solution; the ordinary flow MSE remains responsible for
    rotating the resulting non-trivial delta toward the target.
    """

    delta = prediction.float() - base_prediction.detach().float()
    desired = target.float() - base_prediction.detach().float()
    dimensions = tuple(range(1, delta.ndim))
    scale_floor = float(training.get("main_flow_magnitude_scale_floor", 1e-4))
    delta_rms = delta.square().mean(dim=dimensions).clamp_min(
        scale_floor**2
    ).sqrt()
    desired_rms = desired.square().mean(dim=dimensions).clamp_min(
        scale_floor**2
    ).sqrt()
    ratio = delta_rms / desired_rms
    target_ratio = float(training.get("main_flow_magnitude_target_ratio", 1.0))
    huber_beta = float(training.get("main_flow_magnitude_huber_beta", 0.10))
    if target_ratio <= 0:
        raise ValueError("main_flow_magnitude_target_ratio must be positive")
    if huber_beta <= 0:
        raise ValueError("main_flow_magnitude_huber_beta must be positive")
    target_tensor = torch.full_like(ratio, target_ratio)
    loss = F.smooth_l1_loss(ratio, target_tensor, beta=huber_beta)
    return loss, {
        "main_flow_magnitude_rms_ratio": ratio.detach().mean(),
        "main_flow_magnitude_target_ratio": loss.new_tensor(target_ratio),
        "main_flow_magnitude_absolute_error": (
            ratio.detach() - target_ratio
        ).abs().mean(),
        "main_flow_magnitude_below_target_fraction": (
            ratio.detach() < target_ratio
        ).float().mean(),
        "main_flow_magnitude_loss": loss.detach(),
    }


def _flow_step(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    timestep_weighting: dict[str, torch.Tensor] | None,
    *,
    generator: torch.Generator,
    step: int,
    mode: str,
    train_auxiliaries: bool,
    measure_base: bool,
    capture_auxiliary_probe: bool = False,
    backward_scale: float | None = None,
    performance_stage: dict[str, Any] | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
]:
    latents = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
    context = batch["conditioning"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    if performance_stage is not None and mode == "curriculum":
        references, reference_mask, include_target, curriculum = (
            _performance_reference_batch(
                batch, device, performance_stage, generator
            )
        )
    else:
        references, reference_mask, include_target, curriculum = _reference_batch(
            batch, device, mode, step, training, generator
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
    padding = torch.zeros(
        latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
        device=device, dtype=latents.dtype,
    )
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        output = reader(references, reference_mask, reconstruct=train_auxiliaries)
        adapter.set_style_context(output.tokens)
        adapter.set_timesteps(timesteps)
        try:
            prediction = anima(
                noisy.unsqueeze(2), timesteps.to(latents.dtype), context=context,
                padding_mask=padding, target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()
    dimensions = tuple(range(1, prediction.ndim))
    per_row = (prediction - target).square().mean(dim=dimensions)
    timestep_weights = _native_effect_weights_for_timesteps(
        timesteps, timestep_weighting
    )
    flow_loss = (per_row * timestep_weights).mean()
    total = flow_loss
    metrics: dict[str, torch.Tensor] = {
        "loss": flow_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "flow_timestep_weight": timestep_weights.detach().mean(),
        "style_token_rms": output.tokens.detach().float().square().mean().sqrt(),
        "references": reference_mask.sum(dim=1).float().mean(),
        "target_inclusion": include_target.float().mean(),
        "target_probability": flow_loss.new_tensor(
            float(curriculum["target_probability"])
        ),
        "style_enabled_fraction": flow_loss.new_tensor(1.0),
        "timestep_mean": timesteps.detach().mean(),
        "megastyle_batch": flow_loss.new_tensor(
            float(str(batch.get("data_domain", "anima")) == "megastyle")
        ),
    }
    if train_auxiliaries:
        reconstruction, reconstruction_metrics = _reconstruction_loss(
            output.reconstruction, output.reconstruction_target
        )
        reconstruction_weight = _reconstruction_weight(step, training)
        total = total + reconstruction_weight * reconstruction
        metrics.update(reconstruction_metrics)
        metrics.update({
            "reconstruction_loss": reconstruction.detach(),
            "reconstruction_weight": flow_loss.new_tensor(reconstruction_weight),
            "reconstruction_weighted_loss": (
                reconstruction_weight * reconstruction.detach()
            ),
        })

    artist_domain = str(batch.get("data_domain", "anima")) in {
        str(value) for value in training.get("artist_effect_domains", ["anima"])
    }
    prototype_weight = _ramp(
        step,
        int(training.get("artist_prototype_start_step", 250)),
        int(training.get("artist_prototype_full_step", 1_000)),
        float(training.get("artist_prototype_weight", 0.0)),
    )
    need_prototype = (
        train_auxiliaries
        and artist_domain
        and prototype_weight > 0
        and step % int(training.get("artist_prototype_every", 2)) == 0
        and prediction.shape[0] >= 2
    )
    artist_effect_active_weight = max(
        _ramp(
            step,
            int(training.get("artist_effect_start_step", 250)),
            int(training.get("artist_effect_full_step", 1_000)),
            float(training.get("artist_effect_weight", 0.0)),
        ),
        _ramp(
            step,
            int(training.get("common_output_start_step", 250)),
            int(training.get("common_output_full_step", 1_000)),
            float(training.get("common_output_weight", 0.0)),
        ),
        _ramp(
            step,
            int(training.get("artist_magnitude_start_step", 250)),
            int(training.get("artist_magnitude_full_step", 1_000)),
            float(training.get("artist_magnitude_weight", 0.0)),
        ),
    )
    need_artist_effect = (
        train_auxiliaries
        and artist_domain
        and artist_effect_active_weight > 0
        and step % int(training.get("artist_effect_every", 4)) == 0
        and prediction.shape[0] >= 2
    )
    need_wrong = (
        train_auxiliaries
        and float(training.get("functional_weight", 0.0)) > 0
        and step >= int(training.get("functional_start_step", 250))
        and step % int(training.get("functional_every", 4)) == 0
        and prediction.shape[0] >= 2
    )
    main_magnitude_weight = _ramp(
        step,
        int(training.get("main_flow_magnitude_start_step", 1)),
        int(training.get("main_flow_magnitude_full_step", 1)),
        float(training.get("main_flow_magnitude_weight", 0.0)),
    )
    main_magnitude_every = int(
        training.get("main_flow_magnitude_every", 1)
    )
    need_main_magnitude = (
        train_auxiliaries
        and main_magnitude_weight > 0
        and step % max(1, main_magnitude_every) == 0
    )
    base_prediction = None
    if (
        measure_base
        or need_wrong
        or need_artist_effect
        or capture_auxiliary_probe
        or need_main_magnitude
    ):
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            adapter.clear_style_tokens()
            base_prediction = anima(
                noisy.unsqueeze(2), timesteps.to(latents.dtype), context=context,
                padding_mask=padding, target_input_ids=None,
            ).squeeze(2).float()

    if need_main_magnitude:
        assert base_prediction is not None
        magnitude_loss, magnitude_metrics = (
            _main_flow_total_magnitude_loss(
                prediction,
                base_prediction,
                target,
                training=training,
            )
        )
        weighted_magnitude = main_magnitude_weight * magnitude_loss
        total = total + weighted_magnitude
        metrics.update(magnitude_metrics)
        metrics.update({
            "main_flow_magnitude_weight": flow_loss.new_tensor(
                main_magnitude_weight
            ),
            "main_flow_magnitude_weighted_loss": (
                weighted_magnitude.detach()
            ),
        })

    deferred_artist_effect = None
    if need_prototype or need_artist_effect:
        target_references = batch["cached_target_tokens"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )[:, None]
        target_mask = torch.ones(
            target_references.shape[:2], device=device, dtype=torch.bool
        )
        heldout_references, heldout_mask = _reference_inputs(
            batch, device, "heldout"
        )
        main_is_target = mode == "self" or (
            mode == "curriculum" and bool(curriculum.get("target_only", False))
        )
        main_is_heldout = mode == "heldout" or (
            mode == "curriculum"
            and float(curriculum.get("target_probability", 1.0)) == 0.0
        )
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            target_view_tokens = (
                output.tokens
                if main_is_target
                else reader(target_references, target_mask).tokens
            )
            heldout_view_tokens = (
                output.tokens
                if main_is_heldout
                else reader(heldout_references, heldout_mask).tokens
            )
        style_ids = [str(item.style_id) for item in batch["episodes"]]

        if need_prototype:
            prototype, prototype_metrics = episodic_artist_prototype_loss(
                target_view_tokens,
                heldout_view_tokens,
                style_ids,
                temperature=float(
                    training.get("artist_prototype_temperature", 0.10)
                ),
                slot_type_counts=tuple(
                    int(value)
                    for value in training.get(
                        "artist_prototype_slot_type_counts", [16, 8, 4]
                    )
                ),
            )
            total = total + prototype_weight * prototype
            metrics.update(prototype_metrics)
            metrics.update({
                "artist_prototype_weight": flow_loss.new_tensor(
                    prototype_weight
                ),
                "artist_prototype_weighted_loss": (
                    prototype_weight * prototype.detach()
                ),
            })

        if need_artist_effect:
            assert base_prediction is not None
            artist_effect_teacher_view = str(
                training.get("artist_effect_teacher_view", "exact_target")
            )
            if artist_effect_teacher_view not in {
                "exact_target", "disjoint_heldout",
            }:
                raise ValueError(
                    "artist_effect_teacher_view must be exact_target or "
                    "disjoint_heldout"
                )
            if artist_effect_teacher_view == "disjoint_heldout":
                # DEADiff-style non-reconstructive supervision: both views
                # exclude the target and contain different images by the same
                # artist.  They are evaluated with exactly the same x_t,
                # timestep, text and frozen-Anima Q.  One view is detached;
                # the other receives the functional/InfoNCE gradient after
                # the primary graph has been released.
                eligible, first_mask, second_mask = _split_reference_views(
                    heldout_mask
                )
                eligible_indices = eligible.nonzero(as_tuple=False).flatten()
                if eligible_indices.numel() >= 2:
                    first_references = heldout_references[eligible]
                    second_references = heldout_references[eligible]
                    disjoint_style_ids = [
                        style_ids[int(index)]
                        for index in eligible_indices.detach().cpu().tolist()
                    ]
                    with torch.no_grad(), torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=torch.bfloat16,
                        enabled=torch.device(device).type == "cuda",
                    ):
                        first_tokens = reader(
                            first_references, first_mask
                        ).tokens
                        adapter.set_style_context(first_tokens)
                        adapter.set_timesteps(timesteps[eligible])
                        try:
                            first_prediction = anima(
                                noisy[eligible].unsqueeze(2),
                                timesteps[eligible].to(latents.dtype),
                                context=context[eligible],
                                padding_mask=padding[eligible],
                                target_input_ids=None,
                            ).squeeze(2).float()
                        finally:
                            adapter.clear_style_tokens()
                    deferred_artist_effect = {
                        "teacher_prediction": first_prediction.detach(),
                        "student_references": second_references,
                        "student_mask": second_mask,
                        "style_ids": disjoint_style_ids,
                        "row_indices": eligible_indices,
                        "teacher_view": "disjoint_heldout",
                    }
                else:
                    metrics["functional_artist_eligible_fraction"] = (
                        flow_loss.new_tensor(0.0)
                    )
                # The exact-target branch below is intentionally skipped.
                target_prediction = None
            else:
                target_prediction = None
            # Exact-target style is a detached functional teacher.  If the
            # primary path is not already the heldout view, defer that
            # trainable Anima pass until after the primary graph is backwarded.
            # Two simultaneous batch-four DiT graphs exceed an H100 80GB.
            if artist_effect_teacher_view == "exact_target":
                if main_is_target:
                    target_prediction = prediction.detach()
                else:
                    with torch.no_grad(), torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=torch.bfloat16,
                        enabled=torch.device(device).type == "cuda",
                    ):
                        adapter.set_style_context(target_view_tokens.detach())
                        adapter.set_timesteps(timesteps)
                        try:
                            target_prediction = anima(
                                noisy.unsqueeze(2), timesteps.to(latents.dtype),
                                context=context, padding_mask=padding,
                                target_input_ids=None,
                            ).squeeze(2).float()
                        finally:
                            adapter.clear_style_tokens()
                if main_is_heldout:
                    heldout_prediction = prediction
                    weighted_artist_effect, artist_effect_metrics = (
                        _weighted_artist_effect_objective(
                            target_prediction.detach() - base_prediction,
                            heldout_prediction - base_prediction,
                            style_ids,
                            training,
                            step=step,
                        )
                    )
                    total = total + weighted_artist_effect
                    metrics.update(artist_effect_metrics)
                else:
                    deferred_artist_effect = {
                        "teacher_prediction": target_prediction.detach(),
                        "student_references": heldout_references,
                        "student_mask": heldout_mask,
                        "style_ids": style_ids,
                        "row_indices": torch.arange(
                            prediction.shape[0], device=device
                        ),
                        "teacher_view": "exact_target",
                    }
    if need_wrong:
        wrong_references, wrong_mask = _reference_inputs(batch, device, "wrong_artist")
        # The wrong-reference branch is a comparator, not a target that should be
        # made deliberately worse.  Keeping its complete frozen-Anima graph alive
        # beside the correct graph nearly doubles peak activation memory.  Detach
        # the comparator while retaining gradients through ``prediction`` so the
        # ranking objective can only improve the correct-reference result.
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            wrong_tokens = reader(wrong_references, wrong_mask).tokens
            adapter.set_style_context(wrong_tokens)
            adapter.set_timesteps(timesteps)
            try:
                wrong_prediction = anima(
                    noisy.unsqueeze(2), timesteps.to(latents.dtype), context=context,
                    padding_mask=padding, target_input_ids=None,
                ).squeeze(2).float()
            finally:
                adapter.clear_style_tokens()
        assert base_prediction is not None
        ranking, ranking_metrics = _artist_flow_ranking_loss(
            prediction, wrong_prediction, base_prediction, target,
            margin=float(training.get("functional_margin", 0.01)),
        )
        weight = _ramp(
            step,
            int(training.get("functional_start_step", 250)),
            int(training.get("functional_full_step", 750)),
            float(training.get("functional_weight", 0.10)),
        )
        total = total + weight * ranking
        metrics.update(ranking_metrics)
        metrics.update({
            "functional_ranking_loss": ranking.detach(),
            "functional_weight": flow_loss.new_tensor(weight),
            "functional_weighted_loss": weight * ranking.detach(),
        })
    if measure_base:
        assert base_prediction is not None
        metrics.update(_flow_metrics(prediction.detach(), base_prediction, target))

    reported_loss = total.detach()
    if deferred_artist_effect is not None:
        target_prediction = deferred_artist_effect["teacher_prediction"]
        heldout_references = deferred_artist_effect["student_references"]
        heldout_mask = deferred_artist_effect["student_mask"]
        style_ids = deferred_artist_effect["style_ids"]
        row_indices = deferred_artist_effect["row_indices"]
        if backward_scale is not None:
            (total * float(backward_scale)).backward()
            total = flow_loss.new_zeros(())
        # Recompute the inexpensive reader view after the primary backward.
        # This also ensures no reader graph from the prototype objective is
        # accidentally reused after its saved tensors have been released.
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            heldout_view_tokens = reader(
                heldout_references, heldout_mask
            ).tokens
            adapter.set_style_context(heldout_view_tokens)
            adapter.set_timesteps(timesteps[row_indices])
            try:
                heldout_prediction = anima(
                    noisy[row_indices].unsqueeze(2),
                    timesteps[row_indices].to(latents.dtype),
                    context=context[row_indices], padding_mask=padding[row_indices],
                    target_input_ids=None,
                ).squeeze(2).float()
            finally:
                adapter.clear_style_tokens()
        assert base_prediction is not None
        base_rows = base_prediction[row_indices]
        weighted_artist_effect, artist_effect_metrics = (
            _weighted_artist_effect_objective(
                target_prediction - base_rows,
                heldout_prediction - base_rows,
                style_ids,
                training,
                step=step,
            )
        )
        total = total + weighted_artist_effect
        reported_loss = reported_loss + weighted_artist_effect.detach()
        metrics.update(artist_effect_metrics)
        metrics.update({
            "functional_artist_eligible_fraction": flow_loss.new_tensor(
                len(style_ids) / prediction.shape[0]
            ),
            "functional_artist_disjoint_heldout": flow_loss.new_tensor(
                float(
                    deferred_artist_effect["teacher_view"]
                    == "disjoint_heldout"
                )
            ),
        })
    auxiliary_probe = None
    if capture_auxiliary_probe:
        assert base_prediction is not None
        auxiliary_probe = {
            "noisy": noisy.detach(),
            "timesteps": timesteps.detach(),
            "context": context.detach(),
            "padding": padding.detach(),
            "target": target.detach(),
            "base_prediction": base_prediction.detach(),
            "correct_prediction": prediction.detach(),
        }
    metrics["loss"] = reported_loss
    return total, metrics, auxiliary_probe


def _native_effect_scales_for_timesteps(
    timesteps: torch.Tensor,
    weighting: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Interpolate the frozen native centered-effect median RMS profile."""

    if weighting is None or "median_rms" not in weighting:
        raise ValueError(
            "Main common-output supervision requires native timestep RMS statistics"
        )
    grid = weighting["timesteps"].to(timesteps.device).float()
    values = weighting["median_rms"].to(timesteps.device).float()
    order = grid.argsort()
    grid = grid[order]
    values = values[order]
    query = timesteps.float().clamp(grid[0], grid[-1])
    if len(grid) == 1:
        return values[0].expand_as(query)
    upper = torch.searchsorted(grid, query).clamp(1, len(grid) - 1)
    lower = upper - 1
    left = grid[lower]
    right = grid[upper]
    fraction = (query - left) / (right - left).clamp_min(1e-8)
    return values[lower] + fraction * (values[upper] - values[lower])


def _wrong_flow_ranking_loss(
    correct_prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank a trainable wrong path below a detached correct comparator."""

    dimensions = tuple(range(1, wrong_prediction.ndim))
    base_error = (
        (base_prediction.detach().float() - target.float())
        .square().mean(dim=dimensions).clamp_min(1e-8)
    )
    correct_error = (
        (correct_prediction.detach().float() - target.float())
        .square().mean(dim=dimensions)
    )
    wrong_error = (
        (wrong_prediction.float() - target.float()).square().mean(dim=dimensions)
    )
    correct_improvement = (base_error - correct_error) / base_error
    wrong_improvement = (base_error - wrong_error) / base_error
    advantage = correct_improvement - wrong_improvement
    ranking = F.relu(float(margin) - advantage).mean()
    return ranking, {
        "correct_improvement": correct_improvement.detach().mean(),
        "wrong_improvement": wrong_improvement.detach().mean(),
        "advantage": advantage.detach().mean(),
    }


def _wrong_reference_gradient_step(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    batch: dict[str, Any],
    probe: dict[str, torch.Tensor],
    device: str,
    training: dict[str, Any],
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Give the cyclic wrong-reference comparator a weak, sequential gradient."""

    wrong_references, wrong_mask = _reference_inputs(batch, device, "wrong_artist")
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        wrong_tokens = reader(wrong_references, wrong_mask).tokens
        adapter.set_style_context(wrong_tokens)
        adapter.set_timesteps(probe["timesteps"])
        try:
            wrong_prediction = anima(
                probe["noisy"].unsqueeze(2),
                probe["timesteps"].to(probe["noisy"].dtype),
                context=probe["context"],
                padding_mask=probe["padding"],
                target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()

    ranking, ranking_metrics = _wrong_flow_ranking_loss(
        probe["correct_prediction"],
        wrong_prediction,
        probe["base_prediction"],
        probe["target"],
        margin=float(training.get("functional_margin", 0.01)),
    )
    full_weight = _ramp(
        step,
        int(training.get("functional_start_step", 250)),
        int(training.get("functional_full_step", 750)),
        float(training.get("functional_weight", 0.10)),
    )
    gradient_scale = float(training.get("functional_wrong_gradient_scale", 0.10))
    weighted = full_weight * gradient_scale * ranking
    return weighted, {
        "functional_wrong_gradient_ranking_loss": ranking.detach(),
        "functional_wrong_gradient_scale": ranking.new_tensor(gradient_scale),
        "functional_wrong_gradient_weight": ranking.new_tensor(
            full_weight * gradient_scale
        ),
        "functional_wrong_gradient_weighted_loss": weighted.detach(),
        "functional_wrong_gradient_flow_improvement": (
            ranking_metrics["wrong_improvement"]
        ),
        "functional_wrong_gradient_advantage": ranking_metrics["advantage"],
    }


def _main_common_output_step(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    penalty: NativeScaleCommonOutputPenalty,
    batch: dict[str, Any],
    probe: dict[str, torch.Tensor],
    device: str,
    training: dict[str, Any],
    timestep_weighting: dict[str, torch.Tensor] | None,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize common artist output on a controlled normal-train probe."""

    references, mask = _reference_inputs(batch, device, "heldout")
    rows = min(
        int(training.get("main_common_output_batch_rows", 4)),
        references.shape[0],
    )
    references, mask = references[:rows], mask[:rows]
    timestep = probe["timesteps"][0:1]
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        if (
            isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention)
            and adapter.bootstrap_phase == "common_only"
        ):
            # The common dictionary is reference-free. A one-token sentinel
            # only activates the adapter hook; the artist branch is disabled.
            style = references.new_zeros(rows, 1, adapter.context_dim)
        else:
            style = reader(references, mask).tokens
        adapter.set_style_context(style)
        adapter.set_timesteps(timestep.expand(rows))
        try:
            prediction = anima(
                probe["noisy"][0:1].expand(rows, -1, -1, -1).unsqueeze(2),
                timestep.to(probe["noisy"].dtype).expand(rows),
                context=probe["context"][0:1].expand(rows, -1, -1),
                padding_mask=probe["padding"][0:1].expand(rows, -1, -1, -1),
                target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()
    student = prediction - probe["base_prediction"][0:1].float()
    native_scale = _native_effect_scales_for_timesteps(
        timestep.float(), timestep_weighting
    )[0]
    common_loss, raw_metrics = penalty.objective_from_scale(
        student,
        native_scale,
        ratio_threshold=float(
            training.get("main_common_output_native_ratio_threshold", 0.20)
        ),
        artist_energy_floor=float(
            training.get("main_artist_energy_native_ratio_floor", 0.50)
        ),
        artist_energy_weight=float(
            training.get("main_artist_energy_weight", 1.0)
        ),
    )
    weight = _ramp(
        step,
        int(training.get("main_common_output_start_step", 250)),
        int(training.get("main_common_output_full_step", 1_000)),
        float(training.get("main_common_output_weight", 0.04)),
    )
    timestep_weight = _native_effect_weights_for_timesteps(
        timestep.float(), timestep_weighting
    )[0]
    weighted = weight * timestep_weight * common_loss
    metrics = {}
    for key, value in raw_metrics.items():
        key = key.replace("native_teacher_common_output", "main_common_output")
        key = key.replace("native_teacher_artist_energy", "main_artist_energy")
        metrics[key] = value
    metrics.update({
        "main_common_output_weight": weighted.new_tensor(weight),
        "main_common_output_timestep_weight": timestep_weight.detach(),
        "main_common_output_weighted_loss": weighted.detach(),
        "main_common_output_artist_rows": weighted.new_tensor(rows),
        "main_common_output_timestep": timestep.detach().float()[0],
    })
    return weighted, metrics


def _centered_native_magnitude_band(
    student_delta: torch.Tensor,
    native_scale: torch.Tensor,
    *,
    lower: float,
    upper: float,
    upper_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep artist-specific final velocity energy away from the zero shortcut.

    The batch must contain different artists evaluated with one shared probe.
    Removing the artist mean before measuring RMS prevents an identical common
    residual from satisfying the lower bound. Direction remains the job of the
    flow, short teacher bootstrap, and cross-view contrastive objectives.
    """

    if student_delta.ndim < 2 or student_delta.shape[0] < 2:
        raise ValueError("Centered magnitude needs at least two artist rows")
    if lower < 0 or upper < lower or upper_weight < 0:
        raise ValueError("Invalid centered native magnitude band")
    student = student_delta.float()
    centered = student - student.mean(dim=0, keepdim=True)
    dimensions = tuple(range(1, centered.ndim))
    row_rms = centered.square().mean(dim=dimensions).sqrt()
    scale = native_scale.detach().float().clamp_min(1e-8)
    ratio = row_rms / scale
    lower_loss = F.relu(float(lower) - ratio).square().mean()
    upper_loss = F.relu(ratio - float(upper)).square().mean()
    loss = lower_loss + float(upper_weight) * upper_loss
    return loss, {
        "controlled_artist_centered_rms": row_rms.detach().mean(),
        "controlled_artist_native_scale": scale.detach(),
        "controlled_artist_magnitude_ratio": ratio.detach().mean(),
        "controlled_artist_magnitude_lower": loss.new_tensor(float(lower)),
        "controlled_artist_magnitude_upper": loss.new_tensor(float(upper)),
        "controlled_artist_magnitude_lower_loss": lower_loss.detach(),
        "controlled_artist_magnitude_upper_loss": upper_loss.detach(),
        "controlled_artist_magnitude_loss": loss.detach(),
    }


def _controlled_artist_bootstrap_step(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    batch: dict[str, Any],
    probe: dict[str, torch.Tensor],
    device: str,
    training: dict[str, Any],
    timestep_weighting: dict[str, torch.Tensor] | None,
    *,
    step: int,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Train nonzero, reference-repeatable effects on one controlled probe.

    Through the exact-self phase the target image supplies the style view and
    only the centered magnitude band is active. Afterwards two disjoint
    heldout views from each artist receive the same noisy latent, timestep,
    text context, and therefore the same frozen-Anima Q. This makes every
    cross-artist negative in the functional InfoNCE genuinely comparable.
    """

    magnitude_weight = _ramp(
        step,
        int(training.get("controlled_magnitude_start_step", 1)),
        int(training.get("controlled_magnitude_full_step", 500)),
        float(training.get("controlled_magnitude_weight", 0.0)),
    )
    contrastive_weight = _ramp(
        step,
        int(training.get("controlled_contrastive_start_step", 251)),
        int(training.get("controlled_contrastive_full_step", 500)),
        float(training.get("controlled_contrastive_weight", 0.0)),
    )
    if max(magnitude_weight, contrastive_weight) <= 0:
        return None, {}
    if timestep_weighting is None:
        raise ValueError("Controlled artist magnitude needs frozen native scales")

    exact_end = int(training.get("controlled_exact_self_end_step", 250))
    style_ids = [str(item.style_id) for item in batch["episodes"]]
    first_references = None
    first_mask = None
    if step <= exact_end:
        second_references = batch["cached_target_tokens"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )[:, None]
        second_mask = torch.ones(
            second_references.shape[:2], device=device, dtype=torch.bool
        )
        eligible_indices = torch.arange(
            second_references.shape[0], device=device
        )
    else:
        heldout_references, heldout_mask = _reference_inputs(
            batch, device, "heldout"
        )
        eligible, first_view_mask, second_view_mask = _split_reference_views(
            heldout_mask
        )
        eligible_indices = eligible.nonzero(as_tuple=False).flatten()
        if eligible_indices.numel() < 2:
            reference = probe["timesteps"].new_zeros(())
            return None, {
                "controlled_artist_eligible_fraction": reference,
                "controlled_artist_skipped": reference.new_tensor(1.0),
            }
        first_references = heldout_references[eligible]
        # _split_reference_views already returns masks compacted to eligible
        # rows; indexing them again with the original batch mask is invalid.
        first_mask = first_view_mask
        second_references = heldout_references[eligible]
        second_mask = second_view_mask
        style_ids = [
            style_ids[int(index)]
            for index in eligible_indices.detach().cpu().tolist()
        ]

    rows = int(eligible_indices.numel())
    if rows < 2:
        return None, {}
    noisy = probe["noisy"][0:1]
    base = probe["base_prediction"][0:1]
    context = probe["context"][0:1]
    timestep = probe["timesteps"][0:1]

    first_delta = None
    if contrastive_weight > 0 and first_references is not None:
        with torch.no_grad():
            first_delta = _controlled_teacher_forward(
                anima, reader, adapter,
                first_references, first_mask,
                noisy, base, context, timestep, device,
            ).detach()
    second_delta = _controlled_teacher_forward(
        anima, reader, adapter,
        second_references, second_mask,
        noisy, base, context, timestep, device,
    )

    lower = _scheduled_value(
        step,
        int(training.get("controlled_magnitude_lower_start_step", 1)),
        int(training.get("controlled_magnitude_lower_end_step", 500)),
        float(training.get("controlled_magnitude_lower_start", 0.25)),
        float(training.get("controlled_magnitude_lower_end", 0.50)),
    )
    native_scale = _native_effect_scales_for_timesteps(
        timestep.float(), timestep_weighting
    )[0]
    magnitude, metrics = _centered_native_magnitude_band(
        second_delta,
        native_scale,
        lower=lower,
        upper=float(training.get("controlled_magnitude_upper", 1.25)),
        upper_weight=float(
            training.get("controlled_magnitude_upper_weight", 0.10)
        ),
    )
    total = magnitude_weight * magnitude
    metrics.update({
        "controlled_artist_magnitude_weight": magnitude.new_tensor(
            magnitude_weight
        ),
        "controlled_artist_magnitude_weighted_loss": (
            magnitude_weight * magnitude.detach()
        ),
        "controlled_artist_exact_self": magnitude.new_tensor(
            float(step <= exact_end)
        ),
        "controlled_artist_eligible_fraction": magnitude.new_tensor(
            rows / max(1, len(batch["episodes"]))
        ),
        "controlled_artist_skipped": magnitude.new_tensor(0.0),
    })

    if contrastive_weight > 0 and first_delta is not None:
        contrastive, contrastive_metrics = centered_functional_artist_loss(
            first_delta,
            second_delta,
            style_ids,
            temperature=float(
                training.get("controlled_contrastive_temperature", 0.10)
            ),
            pool_scales=tuple(
                int(value)
                for value in training.get(
                    "controlled_contrastive_pool_scales", [2, 4]
                )
            ),
            repeatability_weight=float(
                training.get("controlled_repeatability_weight", 0.0)
            ),
        )
        total = total + contrastive_weight * contrastive
        metrics.update({
            f"controlled_{key}": value
            for key, value in contrastive_metrics.items()
        })
        metrics.update({
            "controlled_artist_contrastive_weight": contrastive.new_tensor(
                contrastive_weight
            ),
            "controlled_artist_contrastive_weighted_loss": (
                contrastive_weight * contrastive.detach()
            ),
        })
    else:
        metrics.update({
            "controlled_artist_contrastive_weight": magnitude.new_zeros(()),
            "controlled_artist_contrastive_weighted_loss": magnitude.new_zeros(()),
        })
    metrics["controlled_artist_total_weighted_loss"] = total.detach()
    return total, metrics


def _controlled_teacher_forward(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    references: torch.Tensor,
    mask: torch.Tensor,
    noisy: torch.Tensor,
    base: torch.Tensor,
    base_context: torch.Tensor,
    timestep: torch.Tensor,
    device: str,
    *,
    reference_weights: torch.Tensor | None = None,
    tagged: torch.Tensor | None = None,
    block_indices: tuple[int, ...] = (),
) -> torch.Tensor:
    rows = references.shape[0]
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        if (
            isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention)
            and adapter.bootstrap_phase == "common_only"
        ):
            # The common dictionary is reference-free. A one-token sentinel
            # only activates the adapter hook; the artist branch is disabled.
            style = references.new_zeros(rows, 1, adapter.context_dim)
        else:
            style = reader(
                references,
                mask,
                reference_weights=reference_weights,
            ).tokens
        adapter.reset_internal_teacher()
        adapter.set_style_context(style)
        if tagged is not None:
            adapter.set_teacher_context(
                tagged,
                block_indices=block_indices,
                post_gate_distillation=True,
            )
        adapter.set_timesteps(timestep.expand(rows))
        padding = torch.zeros(
            rows, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=noisy.dtype,
        )
        try:
            prediction = anima(
                noisy.expand(rows, -1, -1, -1).unsqueeze(2),
                timestep.expand(rows),
                context=base_context.expand(rows, -1, -1),
                padding_mask=padding, target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()
    return prediction - base


@torch.no_grad()
def _native_final_teacher_forward(
    anima: torch.nn.Module,
    adapter: FreshKVStyleCrossAttention,
    noisy: torch.Tensor,
    base: torch.Tensor,
    tagged_context: torch.Tensor,
    timestep: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Measure the native tagged final effect with the style path bypassed."""

    rows = tagged_context.shape[0]
    adapter.clear_style_tokens()
    padding = torch.zeros(
        rows, 1, noisy.shape[-2], noisy.shape[-1],
        device=device, dtype=noisy.dtype,
    )
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        prediction = anima(
            noisy.expand(rows, -1, -1, -1).unsqueeze(2),
            timestep.expand(rows),
            context=tagged_context,
            padding_mask=padding,
            target_input_ids=None,
        ).squeeze(2).float()
    adapter.clear_style_tokens()
    return prediction - base


def _teacher_step(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    bank: NativeCenteredTeacherBank,
    contexts: NativeArtistContextCache,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    timestep_weighting: dict[str, torch.Tensor] | None,
    native_common_cache: dict[tuple[int, int], torch.Tensor],
    *,
    step: int,
    probe_index: int,
    post_gate_magnitude_enabled: bool = True,
) -> dict[str, torch.Tensor]:
    """Backprop one 16-artist controlled update through bounded microbatches."""

    references, mask = _reference_inputs(batch, device, "heldout")
    rows = min(int(training.get("teacher_batch_rows", 16)), references.shape[0])
    microbatch_rows = min(
        rows, int(training.get("teacher_microbatch_rows", 4))
    )
    if rows < 2 or microbatch_rows < 1:
        raise ValueError("Controlled teacher update needs at least two artists")
    references, mask = references[:rows], mask[:rows]
    style_ids = [str(item.style_id) for item in batch["episodes"][:rows]]
    if len(set(style_ids)) != rows:
        raise ValueError("Controlled teacher update requires distinct artists")
    artist_indices = torch.tensor(
        [bank.artist_to_index[value] for value in style_ids], dtype=torch.long
    )
    tensors = bank.tensors
    content_count = int(tensors["noisy_inputs"].shape[0])
    timestep_count = int(tensors["noisy_inputs"].shape[1])
    content_index = probe_index % content_count
    timestep_index = (probe_index // content_count) % timestep_count
    noisy = tensors["noisy_inputs"][content_index, timestep_index].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    base = tensors["base_predictions"][content_index, timestep_index].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    teacher = tensors["centered_teacher"][
        artist_indices, content_index, timestep_index
    ].to(device=device, dtype=torch.float32, non_blocking=True)
    artist_only_fixed_population = isinstance(
        adapter, ArtistOnlySharedBaseKVStyleCrossAttention
    ) and bool(training.get("fixed_population_artist_only", False))
    if artist_only_fixed_population:
        population_offset = bank.train_population_offset()[
            content_index, timestep_index
        ].to(device=device, dtype=torch.float32, non_blocking=True)
        teacher = teacher - population_offset.unsqueeze(0)
    # The bank is centered over every cached artist, not over this sampled
    # sixteen-row batch. Keep those artist targets intact. Recover the removed
    # global native component once per content/timestep from one tagged native
    # forward: raw_effect - globally_centered_effect = exact global mean.
    teacher_centered = teacher
    teacher_center = teacher.new_zeros((1, *teacher.shape[1:]))
    base_context = tensors["base_context"][content_index : content_index + 1].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    timestep = tensors["timesteps"][timestep_index].to(
        device=device, dtype=torch.bfloat16
    )
    post_gate_cfg = dict(training.get("post_gate_teacher_distillation", {}))
    post_gate_weight = _ramp(
        step,
        int(post_gate_cfg.get("start_step", 1)),
        int(post_gate_cfg.get("full_step", 500)),
        float(post_gate_cfg.get("weight", 0.10)),
    )
    post_gate_active = bool(post_gate_cfg.get("enabled", False)) and (
        post_gate_weight > 0
    )
    block_indices = tuple(
        int(index) for index in post_gate_cfg.get(
            "block_indices", (11, 12, 13, 14, 15, 21, 22, 23)
        )
    )
    tagged = contexts.get(style_ids, content_index).to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    common_key = (content_index, timestep_index)
    teacher_common = native_common_cache.get(common_key)
    if artist_only_fixed_population:
        teacher_common = teacher.new_zeros((1, *teacher.shape[1:]))
    elif teacher_common is None:
        raw_teacher = _native_final_teacher_forward(
            anima, adapter, noisy, base, tagged[:1], timestep, device
        )
        teacher_common = (raw_teacher - teacher[:1]).detach()
        native_common_cache[common_key] = teacher_common
    else:
        teacher_common = teacher_common.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    # First pass obtains the exact sixteen-artist common effect without keeping
    # four frozen-DiT graphs alive. The second pass supplies all gradients.
    detached_students = []
    if post_gate_active:
        adapter.begin_post_gate_center_collection()
    with torch.no_grad():
        for offset in range(0, rows, microbatch_rows):
            stop = min(rows, offset + microbatch_rows)
            detached_students.append(
                _controlled_teacher_forward(
                    anima, reader, adapter,
                    references[offset:stop], mask[offset:stop],
                    noisy, base, base_context, timestep, device,
                    tagged=(tagged[offset:stop] if post_gate_active else None),
                    block_indices=block_indices,
                ).detach()
            )
    if post_gate_active:
        adapter.set_post_gate_centers(
            adapter.finish_post_gate_center_collection()
        )
    student_center = torch.cat(detached_students, dim=0).mean(dim=0, keepdim=True)
    if artist_only_fixed_population:
        # Fixed population residuals already have an externally defined zero.
        # Do not make the Student target depend on the sampled minibatch.
        student_center = torch.zeros_like(student_center)
    objective_cfg = _native_teacher_objective_config(training)
    separated_cfg = dict(training.get("separated_component_bootstrap", {}))
    common_objective_cfg = dict(objective_cfg)
    common_objective_cfg.update(dict(separated_cfg.get("common_objective", {})))
    artist_objective_cfg = dict(objective_cfg)
    artist_objective_cfg.update(dict(separated_cfg.get("artist_objective", {})))
    common_proxy = student_center.detach().float().requires_grad_(True)
    common_loss, common_metrics = _common_native_teacher_objective(
        common_proxy,
        teacher_common,
        common_objective_cfg,
        step=step,
    )
    common_gradient = torch.autograd.grad(common_loss, common_proxy)[0].detach()
    common_phase = bool(
        isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention)
        and adapter.bootstrap_phase == "common_only"
    )
    artist_objective_weight = (
        0.0 if common_phase else float(
            separated_cfg.get("artist_objective_weight", 1.0)
        )
    )
    common_objective_weight = float(
        separated_cfg.get(
            "common_objective_weight" if common_phase else "artist_mean_weight",
            1.0 if common_phase else 0.10,
        )
    )
    infonce_weight = (
        0.0
        if common_phase
        else _teacher_infonce_weight(training, int(references.shape[1]))
    )
    global_weight = float(training.get("teacher_global_weight", 0.10))
    timestep_weight = _native_effect_weights_for_timesteps(
        timestep.float().reshape(1), timestep_weighting
    )[0]
    separate_component_gradients = bool(
        training.get("separated_teacher_gradients", False)
    ) and isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention)
    common_surrogate_weight = common_objective_weight
    direct_common_weighted = common_loss.new_zeros(())
    if separate_component_gradients and not common_phase:
        # Common owns the cross-artist mean; Artist owns only the centered
        # residual. Train the two functions through their actual isolated
        # forward paths so Artist never has to cancel a drifting Common output.
        previous_phase = adapter.bootstrap_phase
        adapter.set_bootstrap_phase("common_only")
        try:
            common_student = _controlled_teacher_forward(
                anima,
                reader,
                adapter,
                references[:1],
                mask[:1],
                noisy,
                base,
                base_context,
                timestep,
                device,
            )
            common_loss, common_metrics = _common_native_teacher_objective(
                common_student,
                teacher_common,
                common_objective_cfg,
                step=step,
            )
            direct_common_weighted = (
                global_weight
                * timestep_weight
                * common_objective_weight
                * common_loss
            )
            direct_common_weighted.backward()
        finally:
            adapter.set_bootstrap_phase(previous_phase)
        common_surrogate_weight = 0.0
    metric_rows: list[dict[str, torch.Tensor]] = []
    detached_total = common_objective_weight * common_loss.detach()
    for offset in range(0, rows, microbatch_rows):
        stop = min(rows, offset + microbatch_rows)
        row_fraction = (stop - offset) / rows
        student = _controlled_teacher_forward(
            anima, reader, adapter,
            references[offset:stop], mask[offset:stop],
            noisy, base, base_context, timestep, device,
            tagged=(tagged[offset:stop] if post_gate_active else None),
            block_indices=block_indices,
        )
        teacher_rows = teacher[offset:stop]
        final_loss, metrics = _minimal_native_teacher_objective(
            student,
            teacher_rows,
            artist_objective_cfg,
            step=step,
            student_center=student_center,
            teacher_center=teacher_center,
        )
        labels = torch.arange(offset, stop, device=device, dtype=torch.long)
        infonce_loss, infonce_metrics = _all_artist_teacher_infonce(
            student.float() - student_center,
            teacher_rows - teacher_center,
            teacher_centered,
            labels,
            temperature=float(training.get("teacher_infonce_temperature", 0.10)),
        )
        # This linear surrogate has the exact dL_common/dStudent for the
        # detached sixteen-row mean measured by the first pass.
        common_surrogate = (
            student.float() * common_gradient
        ).sum() / rows
        main_loss = global_weight * timestep_weight * (
            row_fraction * (
                artist_objective_weight * final_loss
                + infonce_weight * infonce_loss
            )
            + common_surrogate_weight * common_surrogate
        )
        post_gate_loss = None
        if post_gate_active:
            post_gate_loss, post_gate_metrics = adapter.post_gate_teacher_loss(
                direction_weight=float(post_gate_cfg.get("direction_weight", 1.0)),
                magnitude_weight=(
                    float(post_gate_cfg.get("magnitude_weight", 0.0))
                    if post_gate_magnitude_enabled else 0.0
                ),
                huber_weight=float(post_gate_cfg.get("huber_weight", 0.0)),
                magnitude_lower=float(post_gate_cfg.get("magnitude_lower", 0.50)),
                magnitude_upper=float(post_gate_cfg.get("magnitude_upper", 1.25)),
                native_strength_weighting=bool(
                    post_gate_cfg.get("native_strength_weighting", True)
                ),
                strength_weight_min=float(
                    post_gate_cfg.get("native_strength_weight_min", 0.25)
                ),
                strength_weight_max=float(
                    post_gate_cfg.get("native_strength_weight_max", 4.0)
                ),
                common_weight=float(post_gate_cfg.get("common_weight", 0.25)),
                artist_common_leakage_weight=float(
                    post_gate_cfg.get("artist_common_leakage_weight", 0.10)
                ),
            )
            if not post_gate_metrics:
                raise RuntimeError("Post-gate distillation captured no selected blocks")
            weighted_post_gate = (
                global_weight * timestep_weight * row_fraction
                * post_gate_weight * post_gate_loss
            )
            # Reader and adapter are one trainable visual path.  The local
            # native teacher must align both sides of that path, especially
            # when the Reader starts from a fresh initialization.
            main_loss = main_loss + weighted_post_gate
            metrics.update(post_gate_metrics)
            metrics.update({
                "post_gate_teacher_weight": main_loss.new_tensor(post_gate_weight),
                "post_gate_teacher_weighted_loss": weighted_post_gate.detach(),
                "post_gate_teacher_reader_detached": main_loss.new_tensor(0.0),
            })
            detached_total = detached_total + row_fraction * (
                post_gate_weight * post_gate_loss.detach()
            )
        main_loss.backward()
        detached_total = detached_total + row_fraction * (
            artist_objective_weight * final_loss.detach()
            + infonce_weight * infonce_loss.detach()
        )
        metrics.update(infonce_metrics)
        metrics.update({
            "teacher_reference_count": main_loss.new_tensor(
                references.shape[1]
            ),
            "teacher_infonce_weight": main_loss.new_tensor(infonce_weight),
            "teacher_artist_objective_weight": main_loss.new_tensor(
                artist_objective_weight
            ),
            "teacher_common_objective_weight": main_loss.new_tensor(
                common_objective_weight
            ),
            "teacher_common_separate_update": main_loss.new_tensor(
                float(separate_component_gradients and not common_phase)
            ),
            "teacher_common_direct_weighted_loss": (
                direct_common_weighted.detach()
            ),
            "teacher_common_bootstrap_phase": main_loss.new_tensor(
                float(common_phase)
            ),
            "teacher_infonce_weighted_loss": (
                infonce_weight * infonce_loss.detach()
            ),
            "teacher_controlled_artist_rows": main_loss.new_tensor(rows),
            "teacher_microbatch_rows": main_loss.new_tensor(stop - offset),
        })
        metric_rows.append(metrics)

    metrics = {
        key: torch.stack([row[key] for row in metric_rows if key in row]).mean()
        for key in set().union(*(row.keys() for row in metric_rows))
    }
    metrics.update(common_metrics)
    metrics.update({
        "teacher_global_weight": timestep.new_tensor(global_weight),
        "teacher_timestep_weight": timestep_weight.detach(),
        "teacher_content_index": timestep.new_tensor(content_index),
        "teacher_timestep": timestep.detach().float(),
        "teacher_total_loss": (
            global_weight * timestep_weight * detached_total
        ),
        "teacher_student_view_rms": torch.cat(detached_students).square().mean().sqrt(),
        "post_gate_teacher_enabled": timestep.new_tensor(float(post_gate_active)),
    })
    adapter.clear_post_gate_centers()
    return metrics


@torch.no_grad()
def _calibrate_alpha(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    bank: NativeCenteredTeacherBank,
    contexts: NativeArtistContextCache,
    loader: CachedTeacherReferenceLoader,
    device: str,
    batches: int,
    training: dict[str, Any],
    *,
    reset_alpha: bool = True,
    inject_style: bool = False,
    apply_alpha: bool = True,
    recommended_lower_multiplier: float = 1.0,
    recommended_upper_multiplier: float = 1.5,
) -> dict[str, Any]:
    bin_edges = tuple(
        float(value)
        for value in training.get(
            "alpha_calibration_timestep_bin_edges",
            (0.0, 0.325, 0.625, 0.86, 1.000001),
        )
    )
    adapter.begin_alpha_calibration(
        timestep_bin_edges=bin_edges,
        reset_alpha=reset_alpha,
        inject_style=inject_style,
    )
    reader_was_training = reader.training
    reader.eval()
    content_count = int(bank.tensors["noisy_inputs"].shape[0])
    timestep_count = int(bank.tensors["noisy_inputs"].shape[1])
    for index in range(max(1, batches)):
        batch = loader.load_step(index)
        references, mask = _reference_inputs(batch, device, "heldout")
        style_ids = [str(item.style_id) for item in batch["episodes"]]
        calibration_rows = min(
            len(style_ids), int(training.get("teacher_microbatch_rows", 4))
        )
        references = references[:calibration_rows]
        mask = mask[:calibration_rows]
        style_ids = style_ids[:calibration_rows]
        # Traverse all cached timesteps before advancing content.  With the
        # default 32 probes this covers the complete 4-content x 8-timestep
        # teacher grid exactly once instead of repeatedly observing only the
        # diagonal content/timestep pairs.
        timestep_index = index % timestep_count
        content_index = (index // timestep_count) % content_count
        noisy = bank.tensors["noisy_inputs"][content_index, timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        timestep = bank.tensors["timesteps"][timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        context = bank.tensors["base_context"][content_index : content_index + 1].to(
            device=device, dtype=torch.bfloat16
        )
        adapter.set_alpha_calibration_timestep(float(timestep.float().item()))
        tagged = contexts.get(style_ids, content_index).to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            style = reader(references, mask).tokens
            adapter.reset_internal_teacher()
            adapter.set_style_context(style)
            adapter.set_teacher_context(tagged)
            try:
                anima(
                    noisy.expand(len(style_ids), -1, -1, -1).unsqueeze(2),
                    timestep.expand(len(style_ids)),
                    context=context.expand(len(style_ids), -1, -1),
                    padding_mask=torch.zeros(
                        len(style_ids), 1, noisy.shape[-2], noisy.shape[-1],
                        device=device, dtype=noisy.dtype,
                    ),
                    target_input_ids=None,
                )
            finally:
                adapter.clear_style_tokens()
        adapter.internal_teacher_loss(rho_min=0.0)
    result = adapter.finish_alpha_calibration(
        minimum=float(training.get("alpha_calibration_minimum", 1e-6)),
        maximum=float(training.get("alpha_calibration_maximum", 2.0)),
        relative_block_gain=None,
        global_gain=float(getattr(adapter, "global_gain", 1.0)),
        apply_alpha=apply_alpha,
        recommended_lower_multiplier=recommended_lower_multiplier,
        recommended_upper_multiplier=recommended_upper_multiplier,
        fixed_output_quantile=(
            float(training["fixed_output_quantile"])
            if training.get("fixed_output_quantile") is not None
            else None
        ),
    )
    reader.train(reader_was_training)
    return result


def _save_state(
    path: Path,
    *,
    step: int,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    performance_curriculum: dict[str, Any] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
        "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "performance_curriculum": dict(performance_curriculum or {}),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _evaluate(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    loader: MultiPromptDualQueryCachedStyleLoader,
    device: str,
    training: dict[str, Any],
    timestep_weighting: dict[str, torch.Tensor] | None,
    *,
    step: int,
    batches: int,
    mode: str,
    seed: int,
    performance_stage: dict[str, Any] | None = None,
) -> dict[str, float]:
    reader.eval()
    adapter.eval()
    rows = []
    for index in range(batches):
        _, metrics, _ = _flow_step(
            anima, reader, adapter, loader.load_step(index), device, training,
            timestep_weighting,
            generator=torch.Generator(device=device).manual_seed(seed + index * 97),
            step=step, mode=mode, train_auxiliaries=False, measure_base=True,
            performance_stage=performance_stage,
        )
        rows.append({key: float(value) for key, value in metrics.items()})
    reader.train()
    adapter.train()
    return _mean_metrics(rows)


@torch.no_grad()
def _evaluate_artist_effect_consistency(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    loader: MultiPromptDualQueryCachedStyleLoader,
    device: str,
    training: dict[str, Any],
    *,
    step: int,
    batches: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate repeatable artist effects on matched latent-flow probes."""

    reader.eval()
    adapter.eval()
    effect_rows: list[dict[str, float]] = []
    single_reference_effect_rows: list[dict[str, float]] = []
    prototype_rows: list[dict[str, float]] = []
    single_reference_prototype_rows: list[dict[str, float]] = []
    timestep_values = [
        float(value)
        for value in training.get(
            "artist_effect_validation_timesteps", [0.2, 0.45, 0.7, 0.9]
        )
    ]
    for batch_index in range(max(1, batches)):
        batch = loader.load_step(batch_index)
        target_references = batch["cached_target_tokens"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )[:, None]
        target_mask = torch.ones(
            target_references.shape[:2], device=device, dtype=torch.bool
        )
        heldout_references, heldout_mask = _reference_inputs(
            batch, device, "heldout"
        )
        style_ids = [str(item.style_id) for item in batch["episodes"]]
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            target_tokens = reader(target_references, target_mask).tokens
            heldout_tokens = reader(heldout_references, heldout_mask).tokens
            single_reference_tokens = reader(
                heldout_references[:, :1], heldout_mask[:, :1]
            ).tokens
        prototype_loss, prototype_metrics = episodic_artist_prototype_loss(
            target_tokens,
            heldout_tokens,
            style_ids,
            temperature=float(training.get("artist_prototype_temperature", 0.10)),
            slot_type_counts=tuple(
                int(value)
                for value in training.get(
                    "artist_prototype_slot_type_counts", [16, 8, 4]
                )
            ),
        )
        prototype_metrics["artist_prototype_loss"] = prototype_loss.detach()
        prototype_rows.append({
            key: float(value) for key, value in prototype_metrics.items()
        })
        single_prototype_loss, single_prototype_metrics = (
            episodic_artist_prototype_loss(
                target_tokens,
                single_reference_tokens,
                style_ids,
                temperature=float(
                    training.get("artist_prototype_temperature", 0.10)
                ),
                slot_type_counts=tuple(
                    int(value)
                    for value in training.get(
                        "artist_prototype_slot_type_counts", [16, 8, 4]
                    )
                ),
            )
        )
        single_prototype_metrics["artist_prototype_loss"] = (
            single_prototype_loss.detach()
        )
        single_reference_prototype_rows.append({
            key: float(value)
            for key, value in single_prototype_metrics.items()
        })

        artists = len(style_ids)
        for timestep_index, timestep_value in enumerate(timestep_values):
            row_index = (batch_index + timestep_index) % artists
            latent = batch["latents"][row_index : row_index + 1].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            context = batch["conditioning"][row_index : row_index + 1].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            generator = torch.Generator(device=device).manual_seed(
                seed + batch_index * 100_003 + timestep_index * 1_009
            )
            noise = torch.randn(
                latent.shape, device=device, dtype=latent.dtype,
                generator=generator,
            )
            noisy = (1.0 - timestep_value) * latent + timestep_value * noise
            timestep = torch.full(
                (artists,), timestep_value, device=device, dtype=torch.float32
            )
            padding = torch.zeros(
                artists, 1, latent.shape[-2], latent.shape[-1],
                device=device, dtype=latent.dtype,
            )
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ):
                adapter.clear_style_tokens()
                base = anima(
                    noisy.unsqueeze(2), timestep[:1].to(latent.dtype),
                    context=context, padding_mask=padding[:1],
                    target_input_ids=None,
                ).squeeze(2).float()

                predictions = []
                for tokens in (
                    target_tokens,
                    heldout_tokens,
                    single_reference_tokens,
                ):
                    adapter.set_style_context(tokens)
                    adapter.set_timesteps(timestep)
                    try:
                        prediction = anima(
                            noisy.expand(artists, -1, -1, -1).unsqueeze(2),
                            timestep.to(latent.dtype),
                            context=context.expand(artists, -1, -1),
                            padding_mask=padding, target_input_ids=None,
                        ).squeeze(2).float()
                    finally:
                        adapter.clear_style_tokens()
                    predictions.append(prediction)
            _, effect_metrics = _weighted_artist_effect_objective(
                predictions[0] - base,
                predictions[1] - base,
                style_ids,
                training,
                step=step,
            )
            effect_rows.append({
                key: float(value) for key, value in effect_metrics.items()
            })
            _, single_effect_metrics = _weighted_artist_effect_objective(
                predictions[0] - base,
                predictions[2] - base,
                style_ids,
                training,
                step=step,
            )
            single_reference_effect_rows.append({
                key: float(value)
                for key, value in single_effect_metrics.items()
            })

    result = _mean_scalar_rows(effect_rows)
    result.update(_mean_scalar_rows(prototype_rows))
    result.update({
        f"single_reference_{key}": value
        for key, value in _mean_scalar_rows(
            single_reference_effect_rows
        ).items()
    })
    result.update({
        f"single_reference_{key}": value
        for key, value in _mean_scalar_rows(
            single_reference_prototype_rows
        ).items()
    })
    result.update({
        "artists_per_probe": float(loader.batch_size),
        "random_retrieval_top1": 1.0 / max(1, loader.batch_size),
        "timestep_probes": float(len(timestep_values)),
        "batches": float(max(1, batches)),
    })
    reader.train()
    adapter.train()
    return result


@torch.no_grad()
def _generate_fixed_reference_sample(
    prepared: dict[str, Any],
    config: dict[str, Any],
    destination: Path,
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    output: Path,
    device: str,
    step: int,
    *,
    component_mode: str | None = None,
    strengths_override: list[float] | None = None,
    sample_group: str = "external_reference_samples",
    sample_suffix: str | None = None,
) -> dict[str, Any]:
    """Render the fixed TestSample1--7 contract through the style branch."""

    cfg = dict(prepared["cfg"])
    strengths = (
        [float(value) for value in strengths_override]
        if strengths_override is not None
        else [
            float(value)
            for value in config["detail_preserving_style_cross_attention"]
            ["training"].get("fixed_sample_strengths", [1.0, 1.5, 2.0])
        ]
    )
    if not strengths or any(value <= 0 for value in strengths):
        raise ValueError("fixed_sample_strengths must contain positive values")
    references, mask = _fixed_reference_batch(prepared, device)
    reader_was_training = reader.training
    adapter_was_training = adapter.training
    previous_component_mode = None
    if component_mode is not None:
        if not isinstance(
            adapter, SeparatedCommonArtistKVStyleCrossAttention
        ):
            raise ValueError(
                "fixed sample component_mode requires the separated adapter"
            )
        if component_mode not in adapter._VALID_PHASES:
            raise ValueError(f"Unsupported fixed sample component mode {component_mode}")
        previous_component_mode = adapter.bootstrap_phase
        adapter.set_bootstrap_phase(component_mode)
    reader.eval()
    adapter.eval()
    anima.eval()
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        style_tokens = reader(references, mask).tokens

    width = int(cfg["width"])
    height = int(cfg["height"])
    batch_size = max(1, int(cfg.get("batch_size", 4)))
    initial_noise = torch.randn(
        1, 16, 1, height // 8, width // 8,
        generator=torch.Generator(device="cpu").manual_seed(int(cfg["seed"])),
        dtype=torch.float32,
    ).to(device, dtype=torch.bfloat16)
    positive = prepared["positive"].to(device, dtype=torch.bfloat16)
    negative = prepared["negative"].to(device, dtype=torch.bfloat16)
    if positive.ndim == 2:
        positive = positive[None]
    if negative.ndim == 2:
        negative = negative[None]
    sigmas = torch.linspace(
        1.0, 0.0, int(cfg["steps"]) + 1,
        device=device, dtype=torch.bfloat16,
    )
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    text_cfg = float(cfg["cfg"])

    def denoise(
        text_batch: torch.Tensor,
        style: torch.Tensor | None,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        batch = text_batch.shape[0]
        x = initial_noise.expand(batch, -1, -1, -1, -1).clone()
        negative_batch = negative.expand(batch, -1, -1)
        padding_mask = torch.zeros(
            batch, 1, height // 8, width // 8,
            device=device, dtype=torch.bfloat16,
        )
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            for index in range(len(sigmas) - 1):
                timestep = sigmas[index].expand(batch)

                adapter.clear_style_tokens()
                negative_null = anima(
                    x, timestep, context=negative_batch,
                    padding_mask=padding_mask, target_input_ids=None,
                ).float()
                positive_null = anima(
                    x, timestep, context=text_batch,
                    padding_mask=padding_mask, target_input_ids=None,
                ).float()
                if style is None:
                    velocity = negative_null + text_cfg * (
                        positive_null - negative_null
                    )
                else:
                    if isinstance(
                        adapter, SeparatedCommonArtistKVStyleCrossAttention
                    ):
                        adapter.set_style_context(style, strength=strength)
                        guidance_strength = 1.0
                    else:
                        adapter.set_style_context(style)
                        guidance_strength = strength
                    adapter.set_timesteps(timestep)
                    positive_style = anima(
                        x, timestep, context=text_batch,
                        padding_mask=padding_mask, target_input_ids=None,
                    ).float()
                    velocity = _compose_separate_text_style_guidance(
                        negative_null,
                        positive_null,
                        positive_style,
                        text_cfg=text_cfg,
                        style_strength=guidance_strength,
                    )
                x = (
                    x.float()
                    + velocity * (sigmas[index + 1] - sigmas[index]).float()
                ).to(torch.bfloat16)
        return x.to("cpu")

    try:
        base_latents = denoise(positive, None)
        styled_latents_by_strength = {}
        for strength in strengths:
            styled_parts = []
            for offset in range(0, len(style_tokens), batch_size):
                style = style_tokens[offset : offset + batch_size]
                styled_parts.append(
                    denoise(
                        positive.expand(len(style), -1, -1),
                        style,
                        strength=strength,
                    )
                )
            styled_latents_by_strength[strength] = torch.cat(styled_parts)
    finally:
        adapter.clear_style_tokens()
        if previous_component_mode is not None:
            adapter.set_bootstrap_phase(previous_component_mode)
        if reader_was_training:
            reader.train()
        if adapter_was_training:
            adapter.train()

    latent_groups = {"base": base_latents}
    latent_groups.update({
        f"styled_{strength:g}x": values
        for strength, values in styled_latents_by_strength.items()
    })
    decoded = _decode_latents(
        config,
        destination,
        latent_groups,
        device,
        int(cfg.get("vae_batch_size", 4)),
    )
    base = decoded["base"][0]
    sample_name = f"step-{step:07d}"
    if component_mode is not None:
        sample_name += f"-{component_mode}"
    if sample_suffix:
        sample_name += f"-{sample_suffix}"
    sample_dir = output / sample_group / sample_name
    generated_dir = sample_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    base.save(generated_dir / "no-style.png")
    size = (width, height)
    sheets = {}
    metrics = {}
    for strength in strengths:
        label = f"{strength:g}x"
        styled = decoded[f"styled_{label}"]
        for index, image in enumerate(styled, start=1):
            image.save(
                generated_dir
                / f"style-cross-attention-{label}-TestSample{index}.png"
            )
        sheet = _make_sheet(
            prepared["paths"], base, None, styled, size,
            current_label=(
                f"STYLE CROSS-ATTENTION {label}"
                if component_mode is None
                else f"STYLE {component_mode.upper()} {label}"
            ),
        )
        sheet_path = sample_dir / f"detail-style-fixed-reference-{label}.png"
        sheet.save(sheet_path, compress_level=4)
        rms = _pixel_rms_from_baseline(base, styled)
        pairwise_rms = []
        for left in range(len(styled)):
            left_pixels = np.asarray(styled[left], dtype=np.float32) / 255.0
            for right in range(left + 1, len(styled)):
                right_pixels = (
                    np.asarray(styled[right], dtype=np.float32) / 255.0
                )
                pairwise_rms.append(float(np.sqrt(np.mean(
                    np.square(left_pixels - right_pixels)
                ))))
        mean_pairwise_rms = float(np.mean(pairwise_rms))
        mean_baseline_rms = float(np.mean(rms))
        sheets[label] = str(sheet_path)
        metrics[label] = {
            "mean_pixel_rms_from_baseline": mean_baseline_rms,
            "pixel_rms_from_baseline": rms,
            "mean_pairwise_output_pixel_rms": mean_pairwise_rms,
            "pairwise_output_to_baseline_ratio": (
                mean_pairwise_rms / max(mean_baseline_rms, 1e-8)
            ),
        }
    primary_label = "1x" if "1x" in sheets else f"{strengths[0]:g}x"
    summary = {
        "step": int(step),
        "component_mode": component_mode or "combined",
        "sheet": sheets[primary_label],
        "sheets": sheets,
        "strengths": strengths,
        "artists": int(references.shape[0]),
        "references_per_artist": [
            int(value) for value in mask.sum(dim=1).detach().cpu().tolist()
        ],
        "prompt": str(cfg["prompt"]),
        "negative_prompt": str(cfg["negative_prompt"]),
        "text_cfg": text_cfg,
        "steps": int(cfg["steps"]),
        "seed": int(cfg["seed"]),
        "metrics": metrics,
        "mean_pixel_rms_from_baseline": metrics[primary_label][
            "mean_pixel_rms_from_baseline"
        ],
    }
    write_json(sample_dir / "summary.json", summary)
    del references, style_tokens, base_latents, styled_latents_by_strength, decoded
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def _fixed_reference_batch(
    prepared: dict[str, Any], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one- or multi-reference cached tokens for fixed sampling."""

    values = prepared["reference_tokens"]
    if values.ndim == 3:
        values = values[:, None]
    elif values.ndim != 4:
        raise ValueError(
            "reference_tokens must have shape [artist,tokens,dim] or "
            "[artist,references,tokens,dim]"
        )
    references = values.to(device, dtype=torch.bfloat16, non_blocking=True)
    configured_mask = prepared.get("reference_mask")
    if configured_mask is None:
        mask = torch.ones(references.shape[:2], dtype=torch.bool)
    else:
        mask = configured_mask.to(dtype=torch.bool)
        if tuple(mask.shape) != tuple(references.shape[:2]):
            raise ValueError(
                f"reference_mask shape {tuple(mask.shape)} does not match "
                f"reference batch {tuple(references.shape[:2])}"
            )
    return references, mask.to(device, non_blocking=True)


def _fixed_sample_complete(
    summary_path: Path, strengths: list[float]
) -> bool:
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if [float(value) for value in summary.get("strengths", [])] != strengths:
        return False
    sheets = summary.get("sheets", {})
    expected = {f"{value:g}x" for value in strengths}
    return set(sheets) == expected and all(
        Path(path).exists() for path in sheets.values()
    )


def _compose_separate_text_style_guidance(
    negative_null: torch.Tensor,
    positive_null: torch.Tensor,
    positive_style: torch.Tensor,
    *,
    text_cfg: float,
    style_strength: float,
) -> torch.Tensor:
    """Combine text and style guidance without applying text CFG to style.

    The style branch is defined relative to the same positive-text trajectory,
    so ``style_strength=1`` has the same meaning as val/functional/panel.
    """

    return (
        negative_null
        + float(text_cfg) * (positive_null - negative_null)
        + float(style_strength) * (positive_style - positive_null)
    )


def _merge_detail_style_config(
    config: dict[str, Any], config_key: str
) -> dict[str, Any]:
    """Resolve a detail-style experiment without duplicating the large base config."""

    value = copy.deepcopy(config[config_key])
    base_key = value.pop("extends", None)
    if base_key is None:
        return value
    base = _merge_detail_style_config(config, str(base_key))

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        for key, replacement in right.items():
            if isinstance(left.get(key), dict) and isinstance(replacement, dict):
                left[key] = merge(dict(left[key]), replacement)
            else:
                left[key] = replacement
        return left

    return merge(base, value)


def train_detail_style_cross_attention(
    config: dict[str, Any], destination: Path, *, steps_override: int | None = None,
    config_key: str = "detail_preserving_style_cross_attention",
) -> dict[str, Any]:
    cfg = _merge_detail_style_config(config, config_key)
    # Existing loaders and samplers intentionally share one production config
    # contract. Install the resolved experiment as that contract for this run.
    config = copy.deepcopy(config)
    config["detail_preserving_style_cross_attention"] = cfg
    training = dict(cfg["training"])
    lora_teacher_cfg = dict(cfg.get("lora_functional_teacher", {}))
    lora_teacher_enabled = bool(lora_teacher_cfg.get("enabled", False))
    exact_self_flow_only = bool(training.get("exact_self_flow_only", False))
    native_bootstrap_cfg = dict(training.get("native_bootstrap", {}))
    native_bootstrap_only = bool(native_bootstrap_cfg.get("enabled", False))
    if exact_self_flow_only and native_bootstrap_only:
        raise ValueError("exact_self_flow_only and native_bootstrap are exclusive")
    steps = int(steps_override or training["steps"])
    training["steps"] = steps
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260819))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
    torch.set_float32_matmul_precision("high")

    accumulation = int(training.get("gradient_accumulation_steps", 4))
    train_cfg = _loader_config(config, cfg, split=str(cfg.get("train_split", "train")))
    train_cfg["gradient_accumulation_steps"] = accumulation
    validation_cfg = _loader_config(
        config, cfg, split=str(cfg.get("validation_split", "validation"))
    )
    sample_train_loader, train_loader = _training_loader(
        destination, cfg, train_cfg
    )
    validation_loader = MultiPromptDualQueryCachedStyleLoader(destination, validation_cfg)
    for loader in getattr(train_loader, "loaders", (train_loader,)):
        _audit_student_prompts(loader)
    _audit_student_prompts(validation_loader)

    teacher_loaders: list[tuple[str, CachedTeacherReferenceLoader]] = []
    weighted_teacher_domains: list[int] = []
    bank = None
    contexts = None
    if exact_self_flow_only:
        timestep_weighting = None
        print(
            "detail-style exact-self flow-only mode: no teacher bank, "
            "no native timestep weighting, no auxiliary losses",
            flush=True,
        )
    else:
        bank_key = str(
            cfg["teacher"].get("bank_config_key", "dual_domain_native_teacher")
        )
        bank = NativeCenteredTeacherBank.load(
            config, destination, config_key=bank_key
        )
        context_root = destination / str(cfg["teacher"]["context_cache"])
        contexts = NativeArtistContextCache(
            context_root,
            capacity=int(cfg["teacher"].get("context_lru_shards", 8)),
        )
        domain_cfgs = list(cfg["teacher"].get("reference_domains", []))
        if not domain_cfgs:
            domain_cfgs = [{
                "name": str(
                    cfg["teacher"].get(
                        "reference_domain_name", "synthetic_anima_artist_tag"
                    )
                ),
                "weight": 1,
                "reference_caches": list(cfg["teacher"]["reference_caches"]),
            }]
        for domain_index, domain_cfg in enumerate(domain_cfgs):
            name = str(domain_cfg["name"])
            weight = int(domain_cfg.get("weight", 1))
            roots = [
                destination / str(value)
                for value in domain_cfg["reference_caches"]
            ]
            if weight <= 0 or not roots:
                raise ValueError(
                    f"Teacher reference domain {name!r} needs positive weight "
                    "and at least one cache"
                )
            loader = CachedTeacherReferenceLoader(
                roots,
                split="train",
                style_ids=list(bank.summary["train_style_ids"]),
                batch_size=int(training.get("teacher_batch_rows", 16)),
                references=int(training.get("teacher_references", 4)),
                reference_count_weights=training.get(
                    "teacher_reference_count_weights"
                ),
                seed=(seed ^ 0x7EA4CE11) + domain_index * 1_000_003,
                token_lru_shards=int(
                    cfg["teacher"].get("reference_lru_shards", 8)
                ),
                strict_style_ids=False,
            )
            teacher_loaders.append((name, loader))
            weighted_teacher_domains.extend([domain_index] * weight)
        print(
            "detail-style teacher domains "
            + ", ".join(
                f"{name}={len(loader.styles)} styles roots="
                f"{[str(root) for root in loader.token_roots]}"
                for name, loader in teacher_loaders
            )
            + f" schedule={tuple(weighted_teacher_domains)}",
            flush=True,
        )
        timestep_weighting = _build_native_effect_timestep_weighting(bank, training)
    teacher_domain_schedule = tuple(weighted_teacher_domains)
    lora_teacher_bank = None
    lora_teacher_loaders: dict[str, CachedTeacherReferenceLoader] = {}
    lora_mixture_loader: CachedTeacherReferenceLoader | None = None
    lora_teacher_schedule: tuple[str, ...] = ()
    lora_teacher_domains: tuple[str, ...] = ()
    lora_teacher_update = 0
    lora_teacher_root: Path | None = None
    lora_teacher_load_kinds: set[str] | None = None
    if lora_teacher_enabled:
        # Kept as an optional runtime dependency: the LoRA module already uses
        # this trainer's frozen-Anima forward helpers, so importing it at module
        # import time would create a circular dependency.
        from .lora_functional_distillation import (
            FunctionalLoRATeacherBank,
            _load_lora_plan,
        )

        lora_teacher_root = destination / str(lora_teacher_cfg["teacher_cache"])
        lora_plans = _load_lora_plan(
            destination / str(lora_teacher_cfg["lora_directory"])
        )
        lora_style_ids = [plan.style_id for plan in lora_plans]
        lora_training = dict(lora_teacher_cfg.get("training", {}))
        lora_references = max(
            int(value)
            for value in lora_training.get("reference_counts", [4])
        )
        for domain, cache_value in {
            "human": lora_teacher_cfg["human_reference_cache"],
            "synthetic": lora_teacher_cfg["synthetic_reference_cache"],
        }.items():
            cache_roots = (
                [destination / str(value) for value in cache_value]
                if isinstance(cache_value, list)
                else destination / str(cache_value)
            )
            lora_teacher_loaders[domain] = CachedTeacherReferenceLoader(
                cache_roots,
                split="train",
                style_ids=lora_style_ids,
                batch_size=int(lora_training.get("batch_rows", 4)),
                references=lora_references,
                seed=seed ^ (0x48554D41 if domain == "human" else 0x53594E54),
                token_lru_shards=int(lora_training.get("token_lru_shards", 8)),
                ram_resident_tokens=bool(
                    lora_training.get("ram_resident_tokens", True)
                ),
                strict_style_ids=True,
            )
        lora_teacher_schedule = tuple(
            str(value)
            for value in lora_training.get(
                "teacher_schedule", ["single", "single", "pair", "triple"]
            )
        )
        lora_teacher_domains = tuple(
            str(value)
            for value in lora_training.get(
                "reference_domain_schedule", ["human", "synthetic"]
            )
        )
        if not lora_teacher_schedule:
            raise ValueError("LoRA functional teacher schedule cannot be empty")
        if not lora_teacher_domains or any(
            value not in lora_teacher_loaders for value in lora_teacher_domains
        ):
            raise ValueError("Invalid LoRA functional reference-domain schedule")
        lora_every = max(1, int(lora_training.get("every", 2)))
        lora_start = int(lora_training.get("start_step", 1))
        lora_end = int(lora_training.get("end_step", steps))
        lora_teacher_load_kinds = set(lora_teacher_schedule)
        for stage in lora_training.get("teacher_kind_curriculum", []):
            lora_teacher_load_kinds.update(
                str(value) for value in stage.get("schedule", [])
            )
        if lora_start <= min(
            lora_end, int(lora_training.get("single_only_steps", 500))
        ):
            lora_teacher_load_kinds.add("single")
        mixture_kinds = lora_teacher_load_kinds - {"single"}
        mixture_reference_cache = lora_teacher_cfg.get(
            "mixture_reference_cache"
        )
        if mixture_kinds and not mixture_reference_cache:
            raise ValueError(
                "Non-single LoRA teachers require a materialized "
                "mixture_reference_cache"
            )
        if mixture_reference_cache:
            mixture_rows = read_records(lora_teacher_root / "mixtures.parquet")
            mixture_style_ids = [
                str(row["mixture_style_id"])
                for row in mixture_rows
                if str(row["kind"]) in mixture_kinds
                and bool(row.get("enabled", True))
            ]
            lora_mixture_loader = CachedTeacherReferenceLoader(
                destination / str(mixture_reference_cache),
                split="train",
                style_ids=mixture_style_ids,
                batch_size=int(lora_training.get("batch_rows", 4)),
                references=lora_references,
                seed=seed ^ 0x4D495854,
                token_lru_shards=int(lora_training.get("token_lru_shards", 8)),
                ram_resident_tokens=bool(
                    lora_training.get("ram_resident_tokens", True)
                ),
                strict_style_ids=True,
            )
        if lora_start <= 1:
            lora_teacher_bank = FunctionalLoRATeacherBank(
                lora_teacher_root, load_kinds=lora_teacher_load_kinds
            )
        print(
            "detail-style added LoRA functional teacher "
            f"styles={len(lora_style_ids)} schedule={lora_teacher_schedule} "
            f"domains={lora_teacher_domains} every={lora_every} "
            f"materialized_mixtures={len(lora_mixture_loader.styles) if lora_mixture_loader else 0} "
            f"load_at_step={max(1, lora_start)}",
            flush=True,
        )
    main_common_output_penalty = None
    if (
        not exact_self_flow_only
        and float(training.get("main_common_output_weight", 0.0)) > 0
    ):
        main_common_output_penalty = NativeScaleCommonOutputPenalty()

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(training.get("fuse_attention_projections", True)),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device)
    adapter = _build_style_adapter(cfg).to(device)
    if lora_teacher_enabled and not isinstance(
        adapter,
        (
            SeparatedCommonArtistKVStyleCrossAttention,
            ArtistOnlySharedBaseKVStyleCrossAttention,
        ),
    ):
        raise TypeError(
            "LoRA functional augmentation requires a component-aware adapter"
        )
    attach_same_q_style_adapter(anima, adapter)

    output = destination / str(cfg["output_directory"])
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    resume = None
    loaded_checkpoint_state = None
    if bool(training.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        loaded_checkpoint_state = resume
        reader.load_state_dict(resume["reader"], strict=True)
        adapter.load_state_dict(resume["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        start_step = int(resume["step"])
    else:
        resume_checkpoint = cfg.get("resume_checkpoint")
        if resume_checkpoint:
            resume = torch.load(
                destination / str(resume_checkpoint),
                map_location="cpu",
                weights_only=False,
            )
            loaded_checkpoint_state = resume
            reader.load_state_dict(resume["reader"], strict=True)
            adapter.load_state_dict(resume["adapter"], strict=True)
            adapter.restore_timestep_strength_state()
            start_step = int(resume["step"])
            print(
                "resumed detail-style model, optimizer and RNG from "
                f"{resume_checkpoint} step={start_step}",
                flush=True,
            )
        initial_checkpoint = cfg.get("initial_checkpoint")
        if initial_checkpoint and resume_checkpoint is None:
            initial_state = torch.load(
                destination / str(initial_checkpoint),
                map_location="cpu",
                weights_only=False,
            )
            loaded_checkpoint_state = initial_state
            reader.load_state_dict(initial_state["reader"], strict=True)
            adapter.load_state_dict(initial_state["adapter"], strict=True)
            adapter.restore_timestep_strength_state()
            print(
                f"loaded detail-style Reader and adapter bootstrap from "
                f"{initial_checkpoint} step={int(initial_state.get('step', 0))}",
                flush=True,
            )
        reader_checkpoint = cfg.get("reader_checkpoint")
        if reader_checkpoint and not initial_checkpoint and resume_checkpoint is None:
            reader_state = torch.load(
                destination / str(reader_checkpoint),
                map_location="cpu",
                weights_only=False,
            )
            reader.load_state_dict(reader_state["reader"], strict=True)
            print(
                f"loaded pretrained detail-style Reader from {reader_checkpoint} "
                f"step={int(reader_state.get('step', 0))}",
                flush=True,
            )
        if initial_checkpoint:
            pass
        elif exact_self_flow_only or bool(
            training.get("use_fixed_initial_alpha", False)
        ):
            initial_alpha = float(training.get("fixed_initial_alpha", 0.10))
            if initial_alpha <= 0:
                raise ValueError("fixed_initial_alpha must be positive")
            with torch.no_grad():
                adapter.alpha.fill_(initial_alpha)
                adapter.alpha_by_timestep.fill_(initial_alpha)
                adapter.timestep_strength_enabled.fill_(False)
                adapter.fixed_output_strength_enabled.fill_(False)
            adapter.restore_timestep_strength_state()
            write_json(output / "style_strength.json", {
                "mode": (
                    "fixed_teacher_free" if exact_self_flow_only
                    else "fixed_auxiliary_bootstrap"
                ),
                "alpha": initial_alpha,
                "global_gain": float(getattr(adapter, "global_gain", 1.0)),
                "timestep_strength": False,
                "fixed_output_strength": False,
            })
        else:
            assert bank is not None and contexts is not None and teacher_loaders
            if isinstance(
                adapter, SeparatedCommonArtistKVStyleCrossAttention
            ):
                # Alpha calibration is an artist-path scale calibration.  The
                # tiny common K/V bank learns against its final native target
                # during phase A rather than contaminating these statistics.
                adapter.set_bootstrap_phase("artist_only")
            calibration = _calibrate_alpha(
                # Calibrate against the same synthetic-reference domain used by
                # native artist-tag distillation. Human references never receive
                # an artist-tag residual target.
                anima, reader, adapter, bank, contexts, teacher_loaders[0][1], device,
                int(training.get("alpha_calibration_batches", 4)),
                training,
            )
            if isinstance(
                adapter, SeparatedCommonArtistKVStyleCrossAttention
            ):
                adapter.set_bootstrap_phase("combined")
            write_json(output / "alpha_calibration.json", calibration)
            if isinstance(adapter, SharedBaseKVStyleCrossAttention):
                write_json(output / "style_strength.json", {
                    "mode": "block_timestep_native_effect_calibrated",
                    "global_gain": adapter.global_gain,
                    "base_alpha": calibration["base_alpha"],
                    "alpha": adapter.alpha.detach().float().cpu().tolist(),
                    "alpha_by_timestep": calibration["alpha_by_timestep"],
                    "native_lower_by_timestep": calibration[
                        "native_lower_by_timestep"
                    ],
                    "native_upper_by_timestep": calibration[
                        "native_upper_by_timestep"
                    ],
                    "native_fixed_output_by_timestep": calibration[
                        "native_fixed_output_by_timestep"
                    ],
                    "fixed_output_quantile": calibration["fixed_output_quantile"],
                    "relative_block_gain": "disabled",
                    "medoid_blocks": list(adapter.medoid_blocks),
                    "block_to_base": adapter.block_to_base.detach().cpu().tolist(),
                })

    if (
        isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention)
        and adapter.artist_null_residual
        and bool(training.get("reset_artist_null_context_on_upgrade", False))
    ):
        if not _checkpoint_has_artist_null_residual(loaded_checkpoint_state):
            with torch.no_grad():
                adapter.null_style_context.zero_()
            print(
                "detail-style initialized the new artist-null baseline at zero",
                flush=True,
            )

    if lora_teacher_enabled:
        lora_training = dict(lora_teacher_cfg.get("training", {}))
        lora_every = max(1, int(lora_training.get("every", 2)))
        lora_start = int(lora_training.get("start_step", 1))
        lora_teacher_update = sum(
            1
            for prior_step in range(lora_start, start_step + 1)
            if prior_step % lora_every == 0
        )

    reader_parameters = [value for value in reader.parameters() if value.requires_grad]
    kv_parameters = adapter.kv_parameters()
    null_parameter_ids = {id(value) for value in adapter.null_parameters()}
    gain_parameter_ids = {id(value) for value in adapter.gain_parameters()}
    adapter_core_parameters = [
        value for value in kv_parameters
        if id(value) not in null_parameter_ids | gain_parameter_ids
    ]
    optimizer_groups: list[dict[str, Any]] = [
        {
            "params": reader_parameters,
            "lr": float(training.get("learning_rate", 1e-4)),
            "name": "reader",
        }
    ]
    if isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        optimizer_groups.extend([
            {
                "params": adapter.common_parameters(),
                "lr": float(training.get("common_kv_learning_rate", 1e-4)),
                "name": "common_kv",
                "weight_decay": 0.0,
            },
            {
                "params": adapter.shared_parameters(),
                "lr": float(training.get("shared_kv_learning_rate", 5e-5)),
                "name": "artist_shared_kv",
            },
            {
                "params": adapter.delta_parameters(),
                "lr": float(training.get("block_delta_learning_rate", 1e-4)),
                "name": "artist_block_delta",
            },
            {
                "params": adapter.mixing_parameters(),
                "lr": float(training.get("base_mix_learning_rate", 2e-5)),
                "name": "artist_base_mix",
                "weight_decay": 0.0,
            },
        ])
        if adapter.null_parameters():
            optimizer_groups.append({
                "params": adapter.null_parameters(),
                "lr": float(
                    training.get("artist_null_learning_rate", 1e-4)
                ),
                "name": "artist_null_context",
                "weight_decay": 0.0,
            })
    elif isinstance(adapter, SharedBaseKVStyleCrossAttention):
        optimizer_groups.extend([
            {
                "params": adapter.shared_parameters(),
                "lr": float(training.get("shared_kv_learning_rate", 5e-5)),
                "name": "shared_kv",
            },
            {
                "params": adapter.delta_parameters(),
                "lr": float(training.get("block_delta_learning_rate", 1e-4)),
                "name": "block_delta",
            },
            {
                "params": adapter.mixing_parameters(),
                "lr": float(training.get("base_mix_learning_rate", 2e-5)),
                "name": "base_mix",
                "weight_decay": 0.0,
            },
            {
                "params": adapter.null_parameters(),
                "lr": float(training.get("null_context_learning_rate", 1e-4)),
                "name": "null_context",
                "weight_decay": 0.0,
            },
        ])
        if adapter.gain_parameters():
            optimizer_groups.append({
                "params": adapter.gain_parameters(),
                "lr": float(training.get("component_gain_learning_rate", 1e-2)),
                "name": "component_gains",
                "weight_decay": 0.0,
            })
    else:
        optimizer_groups.append({
            "params": kv_parameters,
            "lr": float(training.get("kv_learning_rate", 5e-5)),
            "name": "style_kv",
        })
    base_group_lrs = {
        str(group["name"]): float(group["lr"])
        for group in optimizer_groups
    }
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True) and device.startswith("cuda")),
    )
    if resume is not None:
        if bool(training.get("resume_optimizer", True)):
            optimizer.load_state_dict(resume["optimizer"])
        else:
            print(
                "detail-style initialized a fresh optimizer for the resumed "
                "model state",
                flush=True,
            )
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])

    reader_count = sum(value.numel() for value in reader_parameters)
    kv_count = sum(value.numel() for value in kv_parameters)
    print(
        f"detail-style model reader={reader_count/1e6:.2f}M "
        f"style_adapter={kv_count/1e6:.2f}M "
        f"architecture={cfg['adapter'].get('architecture', 'fresh_per_block')} "
        f"alpha_fixed={adapter.alpha.tolist()}",
        flush=True,
    )
    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "detail-style-cross-attention-v1")),
            id=str(wandb_cfg.get("id", "detail-style-cross-attention-v1")),
            resume="allow" if start_step else "never",
            config={
                config_key: cfg,
                "reader_parameters": reader_count,
                "style_adapter_parameters": kv_count,
            },
        )

    warmup = int(training.get("warmup_steps", 500))
    decay_start = int(training.get("lr_decay_start_step", steps))
    minimum_ratio = float(training.get("minimum_lr_ratio", 0.1))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    state_every = int(training.get("state_every", checkpoint_every))
    sample_every = int(training.get("sample_every", 500))
    fixed_sample_every = int(training.get("fixed_sample_every", 1000))
    fixed_sample_strengths = [
        float(value)
        for value in training.get("fixed_sample_strengths", [1.0, 1.5, 2.0])
    ]
    teacher_every_after = int(training.get("teacher_every_after_bootstrap", 2))
    teacher_bootstrap_end = int(training.get("teacher_every_step_until", 500))
    prefetched = None if native_bootstrap_only else train_loader.prefetch(
        start_step * accumulation,
        (steps - start_step) * accumulation,
        workers=int(training.get("prefetch_workers", 2)),
        depth=int(training.get("prefetch_batches", 6)),
    )
    sample_seed = int(cfg.get("sampling", {}).get("seed", seed ^ 0x5A17))
    sample_requests = [
        ("train", sample_train_loader, episode, sample_seed + index * 10_007)
        for index, episode in enumerate(_select_sample_episodes(sample_train_loader, 4))
    ] + [
        ("validation", validation_loader, episode, sample_seed + (index + 4) * 10_007)
        for index, episode in enumerate(_select_sample_episodes(validation_loader, 4))
    ]
    exact_self_sampling = str(
        cfg.get("sampling", {}).get("reference_mode", "heldout")
    ) == "self"
    fixed_sample_requests = (
        [
            (
                "fixed_validation",
                validation_loader,
                episode,
                sample_seed + (index + 100) * 10_007,
            )
            for index, episode in enumerate(
                _select_sample_episodes(validation_loader, 7)
            )
        ]
        if exact_self_sampling and fixed_sample_every > 0
        else []
    )
    fixed_prepared = (
        load_dual_query_external_sample(config, destination)
        if fixed_sample_every > 0 and not exact_self_sampling
        else None
    )
    history_path = output / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    performance_curriculum = _initial_performance_curriculum_state(
        training, resume
    )
    teacher_curriculum_rows: list[dict[str, float]] = []
    native_common_cache: dict[tuple[int, int], torch.Tensor] = {}
    teacher_update = (
        int(performance_curriculum.get("teacher_update", 0))
        if bool(performance_curriculum.get("enabled", False))
        else (
            start_step if start_step <= teacher_bootstrap_end else (
                teacher_bootstrap_end
                + (start_step - teacher_bootstrap_end) // teacher_every_after
            )
        )
    )
    completed = start_step
    bootstrap_complete = False
    started = time.perf_counter()
    vae = None
    initial_phase = _separated_bootstrap_phase(
        max(1, start_step), training, performance_curriculum
    )
    train_common_in_combined = bool(
        training.get("train_common_in_combined", False)
    )
    _configure_separated_bootstrap_trainability(
        reader,
        adapter,
        initial_phase,
        train_common_in_combined=train_common_in_combined,
    )
    try:
        resumed_fixed_summary = (
            output / "external_reference_samples"
            / f"step-{start_step:07d}" / "summary.json"
        )
        if (
            fixed_prepared is not None
            and start_step > 0
            and start_step % fixed_sample_every == 0
            and not _fixed_sample_complete(
                resumed_fixed_summary, fixed_sample_strengths
            )
        ):
            fixed = _generate_fixed_reference_sample(
                fixed_prepared, config, destination, anima, reader, adapter,
                output, device, start_step,
            )
            print(
                f"detail-style fixed-reference samples step={start_step} "
                f"sheet={fixed['sheet']}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb_run.log({
                    "val/functional/fixed_reference": [
                        wandb.Image(path, caption=f"fixed {label} step {start_step}")
                        for label, path in fixed["sheets"].items()
                    ],
                    **{
                        f"val/functional/fixed_reference_pixel_rms/{label}":
                        values["mean_pixel_rms_from_baseline"]
                        for label, values in fixed["metrics"].items()
                    },
                    **{
                        f"val/functional/fixed_reference_pairwise_rms/{label}":
                        values["mean_pairwise_output_pixel_rms"]
                        for label, values in fixed["metrics"].items()
                    },
                    **{
                        f"val/functional/fixed_reference_diversity_ratio/{label}":
                        values["pairwise_output_to_baseline_ratio"]
                        for label, values in fixed["metrics"].items()
                    },
                }, step=start_step)
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            separated_phase = _separated_bootstrap_phase(
                step, training, performance_curriculum
            )
            _configure_separated_bootstrap_trainability(
                reader,
                adapter,
                separated_phase,
                train_common_in_combined=train_common_in_combined,
            )
            performance_stage = (
                None
                if exact_self_flow_only
                else _active_performance_stage(training, performance_curriculum)
            )
            lr_scale = _delayed_learning_rate_multiplier(
                step, steps, warmup, decay_start, minimum_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = base_group_lrs[str(group["name"])] * lr_scale
                if isinstance(
                    adapter, SeparatedCommonArtistKVStyleCrossAttention
                ):
                    name = str(group["name"])
                    active = (
                        name == "common_kv"
                        if separated_phase == "common_only"
                        else (
                            True
                            if name != "common_kv"
                            else train_common_in_combined
                        )
                    )
                    if not active:
                        group["lr"] = 0.0
            optimizer.zero_grad(set_to_none=True)
            metric_rows: list[dict[str, torch.Tensor]] = []
            main_common_due = bool(
                not native_bootstrap_only
                and main_common_output_penalty is not None
                and step >= int(training.get("main_common_output_start_step", 250))
                and step % int(training.get("main_common_output_every", 4)) == 0
            )
            wrong_gradient_due = bool(
                not native_bootstrap_only
                and float(training.get("functional_wrong_gradient_scale", 0.0)) > 0
                and step >= int(training.get("functional_start_step", 250))
                and step % int(training.get("functional_wrong_gradient_every", 1)) == 0
            )
            controlled_artist_due = bool(
                not exact_self_flow_only
                and not native_bootstrap_only
                and step % max(
                    1, int(training.get("controlled_artist_every", 4))
                ) == 0
                and (
                    _ramp(
                        step,
                        int(training.get("controlled_magnitude_start_step", 1)),
                        int(training.get("controlled_magnitude_full_step", 500)),
                        float(training.get("controlled_magnitude_weight", 0.0)),
                    ) > 0
                    or _ramp(
                        step,
                        int(training.get("controlled_contrastive_start_step", 251)),
                        int(training.get("controlled_contrastive_full_step", 500)),
                        float(training.get("controlled_contrastive_weight", 0.0)),
                    ) > 0
                )
            )
            auxiliary_probe = None
            auxiliary_batch = None
            for micro in range(0 if native_bootstrap_only else accumulation):
                assert prefetched is not None
                batch = next(prefetched)
                generator = torch.Generator(device=device).manual_seed(
                    seed + step * 100_003 + micro
                )
                loss, metrics, captured_probe = _flow_step(
                    anima, reader, adapter, batch, device, training,
                    timestep_weighting,
                    generator=generator, step=step,
                    mode=("self" if exact_self_flow_only else "curriculum"),
                    train_auxiliaries=not exact_self_flow_only,
                    measure_base=(step % log_every == 0 and micro == accumulation - 1),
                    capture_auxiliary_probe=(
                        (
                            main_common_due
                            or wrong_gradient_due
                            or controlled_artist_due
                        )
                        and micro == accumulation - 1
                    ),
                    backward_scale=1.0 / accumulation,
                    performance_stage=performance_stage,
                )
                (loss / accumulation).backward()
                metric_rows.append(metrics)
                if captured_probe is not None:
                    auxiliary_probe = captured_probe
                    auxiliary_batch = batch
            if wrong_gradient_due:
                assert auxiliary_probe is not None and auxiliary_batch is not None
                wrong_loss, wrong_metrics = _wrong_reference_gradient_step(
                    anima, reader, adapter, auxiliary_batch, auxiliary_probe,
                    device, training, step=step,
                )
                wrong_loss.backward()
                metric_rows[-1].update(wrong_metrics)
            if main_common_due:
                assert main_common_output_penalty is not None
                assert auxiliary_probe is not None and auxiliary_batch is not None
                main_common_loss, main_common_metrics = _main_common_output_step(
                    anima, reader, adapter, main_common_output_penalty,
                    auxiliary_batch, auxiliary_probe, device, training,
                    timestep_weighting, step=step,
                )
                main_common_loss.backward()
                metric_rows[-1].update(main_common_metrics)
            if controlled_artist_due:
                assert auxiliary_probe is not None and auxiliary_batch is not None
                controlled_loss, controlled_metrics = (
                    _controlled_artist_bootstrap_step(
                        anima, reader, adapter,
                        auxiliary_batch, auxiliary_probe,
                        device, training, timestep_weighting,
                        step=step,
                    )
                )
                if controlled_loss is not None:
                    controlled_loss.backward()
                    metric_rows[-1]["loss"] = (
                        metric_rows[-1]["loss"] + controlled_loss.detach()
                    )
                metric_rows[-1].update(controlled_metrics)
            stage_teacher_every = (
                int(performance_stage.get("teacher_every", 1))
                if performance_stage is not None
                else (
                    1 if step <= teacher_bootstrap_end else teacher_every_after
                )
            )
            teacher_due = (
                not exact_self_flow_only
                and (native_bootstrap_only or step <= int(training.get("teacher_end_step", steps)))
                and step % max(1, stage_teacher_every) == 0
            )
            if teacher_due:
                assert bank is not None and contexts is not None
                teacher_domain, domain_update = _teacher_domain_update(
                    teacher_domain_schedule, teacher_update
                )
                teacher_domain_name, teacher_loader = teacher_loaders[
                    teacher_domain
                ]
                teacher_reader_gradient_scale = float(
                    training.get("teacher_reader_gradient_scale", 1.0)
                )
                with _scaled_parameter_gradients(
                    reader_parameters, teacher_reader_gradient_scale
                ):
                    teacher_metrics = _teacher_step(
                        anima, reader, adapter, bank, contexts,
                        teacher_loader.load_step(
                            domain_update,
                            reference_count_weights=_teacher_reference_count_weights(
                                training, step
                            ),
                        ),
                        device, training,
                        timestep_weighting, native_common_cache,
                        step=step, probe_index=teacher_update,
                        post_gate_magnitude_enabled=bool(
                            performance_stage.get(
                                "post_gate_magnitude_enabled", True
                            )
                        ) if performance_stage is not None else True,
                    )
                teacher_metrics["teacher_reader_gradient_scale"] = torch.tensor(
                    teacher_reader_gradient_scale, device=device
                )
                teacher_metrics["teacher_reference_domain_index"] = (
                    torch.tensor(float(teacher_domain), device=device)
                )
                for index, (name, _) in enumerate(teacher_loaders):
                    metric_name = "".join(
                        char if char.isalnum() else "_" for char in name
                    )
                    teacher_metrics[
                        f"teacher_reference_domain_{metric_name}"
                    ] = torch.tensor(
                        float(index == teacher_domain), device=device
                    )
                if native_bootstrap_only:
                    metric_rows.append({
                        "loss": teacher_metrics["teacher_total_loss"],
                        "flow_loss": teacher_metrics["teacher_total_loss"].new_zeros(()),
                        **teacher_metrics,
                    })
                else:
                    metric_rows[-1].update(teacher_metrics)
                teacher_curriculum_rows.append({
                    key: float(value.detach())
                    for key, value in teacher_metrics.items()
                    if (
                        key == "native_teacher_projection_coefficient"
                        or key == "native_teacher_cosine"
                        or key == "native_teacher_common_cosine"
                        or key == "native_teacher_common_projection_coefficient"
                        or key == "native_teacher_common_rms_ratio"
                        or key == "teacher_infonce_accuracy"
                        or key == "teacher_infonce_cosine_gap"
                        or (
                            key.startswith("post_gate_teacher_block_")
                            and key.endswith("_cosine")
                        )
                    )
                })
                teacher_update += 1
                performance_curriculum["teacher_update"] = teacher_update
            lora_training = dict(lora_teacher_cfg.get("training", {}))
            lora_every = max(1, int(lora_training.get("every", 2)))
            lora_start = int(lora_training.get("start_step", 1))
            lora_end = int(lora_training.get("end_step", steps))
            lora_due = bool(
                lora_teacher_enabled
                and lora_start <= step <= lora_end
                and step % lora_every == 0
            )
            if lora_due:
                from .lora_functional_distillation import (
                    FunctionalLoRATeacherBank,
                    _lora_teacher_step,
                )

                if lora_teacher_bank is None:
                    assert lora_teacher_root is not None
                    lora_teacher_bank = FunctionalLoRATeacherBank(
                        lora_teacher_root,
                        load_kinds=lora_teacher_load_kinds,
                    )
                assert isinstance(
                    adapter,
                    (
                        SeparatedCommonArtistKVStyleCrossAttention,
                        ArtistOnlySharedBaseKVStyleCrossAttention,
                    ),
                )
                active_schedule = _lora_teacher_schedule_for_step(
                    lora_training, step, lora_teacher_schedule
                )
                kind = active_schedule[lora_teacher_update % len(active_schedule)]
                if not lora_teacher_bank.by_kind.get(kind):
                    raise ValueError(
                        f"Unavailable LoRA functional teacher kind: {kind}"
                    )
                materialized_mixture = kind != "single"
                if materialized_mixture:
                    if lora_mixture_loader is None:
                        raise RuntimeError(
                            "Materialized LoRA mixture loader is unavailable"
                        )
                    domain = "materialized_mixture"
                    lora_loader = lora_mixture_loader
                else:
                    domain = lora_teacher_domains[
                        lora_teacher_update % len(lora_teacher_domains)
                    ]
                    lora_loader = lora_teacher_loaders[domain]
                lora_metrics = _lora_teacher_step(
                    anima,
                    reader,
                    adapter,
                    lora_teacher_bank,
                    lora_loader,
                    kind=kind,
                    update=lora_teacher_update,
                    step=step,
                    device=device,
                    training={**lora_training, "seed": seed},
                    materialized_mixture=materialized_mixture,
                    backward_scale=_lora_backward_scale_for_step(
                        lora_training, step
                    ),
                )
                metric_rows[-1].update({
                    f"lora_teacher_{key}": value
                    for key, value in lora_metrics.items()
                })
                metric_rows[-1]["lora_teacher_domain_synthetic"] = (
                    torch.tensor(float(domain == "synthetic"), device=device)
                )
                metric_rows[-1]["lora_teacher_domain_materialized_mixture"] = (
                    torch.tensor(
                        float(domain == "materialized_mixture"), device=device
                    )
                )
                metric_rows[-1]["lora_teacher_kind_index"] = torch.tensor(
                    float(lora_teacher_schedule.index(kind))
                    if kind in lora_teacher_schedule else -1.0,
                    device=device,
                )
                lora_teacher_update += 1
            if not metric_rows:
                raise RuntimeError("Training step produced no objective")
            if step == 1 or step % log_every == 0:
                metric_rows[-1].update({
                    "reader_gradient_rms": _gradient_rms(reader_parameters),
                    "common_kv_gradient_rms": _gradient_rms(
                        adapter.common_parameters()
                        if isinstance(
                            adapter, SeparatedCommonArtistKVStyleCrossAttention
                        ) else []
                    ),
                    "artist_kv_gradient_rms": _gradient_rms(
                        adapter.kv_parameters()
                        if isinstance(
                            adapter, ArtistOnlySharedBaseKVStyleCrossAttention
                        )
                        else adapter.artist_parameters()
                        if isinstance(
                            adapter, SeparatedCommonArtistKVStyleCrossAttention
                        )
                        else []
                    ),
                    "null_context_gradient_rms": _gradient_rms(
                        adapter.null_parameters()
                    ),
                    "shared_kv_gradient_rms": _gradient_rms(
                        adapter.shared_parameters()
                        if isinstance(adapter, SharedBaseKVStyleCrossAttention)
                        else adapter.kv_parameters()
                    ),
                    "component_gain_gradient_rms": _gradient_rms(
                        adapter.gain_parameters()
                    ),
                })
            reader_max_grad_norm = float(
                training.get("reader_max_grad_norm", 1.0)
            )
            adapter_max_grad_norm = float(
                training.get("adapter_max_grad_norm", 1.0)
            )
            null_max_grad_norm = float(
                training.get("null_context_max_grad_norm", 0.1)
            )
            gain_max_grad_norm = float(
                training.get("component_gain_max_grad_norm", 1.0)
            )
            reader_grad_norm = torch.nn.utils.clip_grad_norm_(
                reader_parameters,
                reader_max_grad_norm,
            )
            adapter_grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter_core_parameters,
                adapter_max_grad_norm,
            )
            null_grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.null_parameters(),
                null_max_grad_norm,
            )
            gain_grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.gain_parameters(),
                gain_max_grad_norm,
            )
            grad_norm = torch.stack([
                value.to(device=device, dtype=torch.float32)
                for value in (
                    reader_grad_norm,
                    adapter_grad_norm,
                    null_grad_norm,
                    gain_grad_norm,
                )
            ]).square().sum().sqrt()
            metric_rows[-1].update({
                "reader_grad_norm": reader_grad_norm.detach(),
                "adapter_grad_norm": adapter_grad_norm.detach(),
                "null_context_grad_norm": null_grad_norm.detach(),
                "component_gain_grad_norm": gain_grad_norm.detach(),
                "reader_grad_clipped": (
                    reader_grad_norm > reader_max_grad_norm
                ).float().detach(),
                "adapter_grad_clipped": (
                    adapter_grad_norm > adapter_max_grad_norm
                ).float().detach(),
                "null_context_grad_clipped": (
                    null_grad_norm > null_max_grad_norm
                ).float().detach(),
                "component_gain_grad_clipped": (
                    gain_grad_norm > gain_max_grad_norm
                ).float().detach(),
            })
            if not bool(torch.isfinite(grad_norm)):
                nonfinite = []
                for prefix, module in (("reader", reader), ("adapter", adapter)):
                    for name, parameter in module.named_parameters():
                        if parameter.grad is not None and not bool(
                            torch.isfinite(parameter.grad).all()
                        ):
                            nonfinite.append(f"{prefix}.{name}")
                            if len(nonfinite) == 16:
                                break
                    if len(nonfinite) == 16:
                        break
                raise FloatingPointError(
                    "Non-finite gradients before optimizer step: "
                    + ", ".join(nonfinite)
                )
            optimizer.step()
            if step == 1 or step % log_every == 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                keys = set().union(*(row.keys() for row in metric_rows))
                averaged = {
                    key: float(torch.stack([row[key] for row in metric_rows if key in row]).mean())
                    for key in keys
                }
                averaged.update({
                    "grad_norm": float(grad_norm),
                    "step_s": time.perf_counter() - step_started,
                    "images_per_s": (
                        int(training.get("teacher_batch_rows", 16))
                        if native_bootstrap_only
                        else train_loader.batch_size * accumulation
                    ) /
                    max(time.perf_counter() - step_started, 1e-6),
                })
                averaged.update({
                    f"{group['name']}_lr": float(group["lr"])
                    for group in optimizer.param_groups
                })
                averaged.update(adapter.runtime_stats())
                teacher_label = (
                    f"{averaged['teacher_total_loss']:.5f}"
                    if "teacher_total_loss" in averaged else "disabled"
                )
                print(
                    f"detail-style step={step}/{steps} loss={averaged['loss']:.5f} "
                    f"flow={averaged['flow_loss']:.5f} grad={averaged['grad_norm']:.4f} "
                    f"teacher={teacher_label} "
                    f"reader_grad={averaged.get('reader_gradient_rms', 0.0):.3e} "
                    f"common_grad={averaged.get('common_kv_gradient_rms', 0.0):.3e} "
                    f"artist_grad={averaged.get('artist_kv_gradient_rms', 0.0):.3e} "
                    f"null_grad={averaged.get('null_context_gradient_rms', 0.0):.3e} "
                    f"step_s={averaged['step_s']:.3f}", flush=True,
                )
                if wandb_run is not None:
                    namespaces = {}
                    for key, value in averaged.items():
                        if key.startswith("reconstruction"):
                            namespace = "train/reconstruction"
                        elif "teacher" in key:
                            namespace = "train/teacher"
                        elif key in {"step_s", "images_per_s"}:
                            namespace = "system/perf"
                        elif (
                            key.startswith(("style_", "grad_"))
                            or key.endswith(("_lr", "_gradient_rms", "_grad_norm"))
                        ):
                            namespace = "model/activation"
                        elif (
                            "artist" in key
                            or "functional" in key
                            or "common_output" in key
                        ):
                            namespace = "train/artist"
                        else:
                            namespace = "train/flow"
                        namespaces[f"{namespace}/{key}"] = value
                    wandb_run.log(namespaces, step=step)

            if step % validation_every == 0 or step == steps:
                if exact_self_flow_only:
                    validation = {
                        "exact_self": _evaluate(
                            anima, reader, adapter, validation_loader, device,
                            training, None, step=step,
                            batches=int(training.get("validation_batches", 8)),
                            mode="self", seed=seed ^ 0xEAC751E,
                        )
                    }
                else:
                    validation = {
                        "self": _evaluate(
                            anima, reader, adapter, validation_loader, device, training,
                            timestep_weighting, step=step,
                            batches=int(training.get("validation_batches", 8)),
                            mode="curriculum", seed=seed ^ 0xBEEF,
                            performance_stage=performance_stage,
                        ),
                        "exact_self_probe": _evaluate(
                            anima, reader, adapter, validation_loader, device, training,
                            timestep_weighting, step=step,
                            batches=int(training.get("validation_batches", 8)),
                            mode="self", seed=seed ^ 0xEAC751E,
                        ),
                        "heldout": _evaluate(
                            anima, reader, adapter, validation_loader, device, training,
                            timestep_weighting, step=step,
                            batches=int(training.get("validation_batches", 8)),
                            mode="heldout", seed=seed ^ 0xC0FFEE,
                        ),
                        "wrong_artist": _evaluate(
                            anima, reader, adapter, validation_loader, device, training,
                            timestep_weighting, step=step,
                            batches=int(training.get("validation_batches", 8)),
                            mode="wrong_artist", seed=seed ^ 0xBAD417,
                        ),
                    }
                    validation["artist_effect"] = _evaluate_artist_effect_consistency(
                        anima, reader, adapter, validation_loader, device, training,
                        step=step,
                        batches=int(
                            training.get("artist_effect_validation_batches", 2)
                        ),
                        seed=seed ^ 0xA47157,
                    )
                if lora_teacher_enabled and lora_teacher_bank is not None:
                    from .lora_functional_distillation import _lora_teacher_step

                    validation_kinds = tuple(
                        str(value)
                        for value in lora_training.get(
                            "validation_kinds",
                            ["single", "pair", "triple", "amplified", "signed"],
                        )
                        if lora_teacher_bank.by_kind.get(str(value))
                    )
                    validation_metrics: dict[str, float] = {}
                    validation_training = {
                        **lora_training,
                        "seed": seed ^ 0x4C4F5241,
                        "batch_rows": int(
                            lora_training.get("validation_batch_rows", 6)
                        ),
                        "reference_counts": [int(
                            lora_training.get("validation_reference_count", 2)
                        )],
                        "reference_count_weights": [1.0],
                    }
                    reported_metrics = {
                        "loss",
                        "pair_cosine",
                        "pair_student_to_teacher_rms",
                        "functional_infonce_accuracy",
                        "functional_infonce_cosine_gap",
                        "student_population_mean_to_teacher_residual_rms",
                        "coefficient_sum_mean",
                        "reference_count",
                    }
                    with torch.no_grad():
                        for kind_index, kind in enumerate(validation_kinds):
                            materialized = kind != "single"
                            validation_loader_for_kind = (
                                lora_mixture_loader
                                if materialized
                                else lora_teacher_loaders["human"]
                            )
                            if validation_loader_for_kind is None:
                                raise RuntimeError(
                                    f"Validation loader unavailable for {kind}"
                                )
                            kind_metrics = _lora_teacher_step(
                                anima,
                                reader,
                                adapter,
                                lora_teacher_bank,
                                validation_loader_for_kind,
                                kind=kind,
                                update=step * 101 + kind_index,
                                step=step,
                                device=device,
                                training=validation_training,
                                materialized_mixture=materialized,
                                backward_scale=0.0,
                                backward=False,
                            )
                            validation_metrics.update({
                                f"{kind}/{key}": float(value)
                                for key, value in kind_metrics.items()
                                if key in reported_metrics
                            })
                    validation["lora_teacher_kinds"] = validation_metrics
                curriculum_metrics, curriculum_changed = (
                    _update_performance_curriculum(
                        training,
                        performance_curriculum,
                        validation,
                        teacher_curriculum_rows,
                        step=step,
                    )
                )
                if native_bootstrap_only:
                    separated_cfg = dict(
                        training.get("separated_component_bootstrap", {})
                    )
                    common_transition_cfg = dict(
                        separated_cfg.get("common_transition", {})
                    )
                    if (
                        bool(common_transition_cfg.get("enabled", False))
                        and separated_phase == "common_only"
                    ):
                        common_metrics, common_consecutive, common_complete = (
                            _separated_common_transition_status(
                                teacher_curriculum_rows,
                                common_transition_cfg,
                                step=step,
                                previous_consecutive=int(
                                    performance_curriculum.get(
                                        "separated_common_consecutive", 0
                                    )
                                ),
                            )
                        )
                        performance_curriculum[
                            "separated_common_consecutive"
                        ] = common_consecutive
                        performance_curriculum[
                            "separated_common_complete"
                        ] = common_complete
                        curriculum_metrics.update(common_metrics)
                        if common_complete:
                            frozen_common_metrics = {
                                key.removeprefix("separated_common_"): value
                                for key, value in common_metrics.items()
                                if key.startswith(
                                    "separated_common_native_teacher_common_"
                                )
                            }
                            performance_curriculum[
                                "separated_common_metrics"
                            ] = frozen_common_metrics
                            performance_curriculum[
                                "separated_common_transition_step"
                            ] = int(step)
                            print(
                                "detail-style separated Common criteria "
                                f"satisfied at step={step}; Artist opens next step",
                                flush=True,
                            )
                    bootstrap_metrics, consecutive, bootstrap_complete = (
                        _native_bootstrap_status(
                            teacher_curriculum_rows,
                            native_bootstrap_cfg,
                            step=step,
                            previous_consecutive=int(
                                performance_curriculum.get(
                                    "native_bootstrap_consecutive", 0
                                )
                            ),
                            common_override=dict(
                                performance_curriculum.get(
                                    "separated_common_metrics", {}
                                )
                            ),
                        )
                    )
                    performance_curriculum[
                        "native_bootstrap_consecutive"
                    ] = consecutive
                    performance_curriculum[
                        "native_bootstrap_complete"
                    ] = bootstrap_complete
                    curriculum_metrics.update(bootstrap_metrics)
                teacher_curriculum_rows.clear()
                row = {
                    "step": step,
                    **validation,
                    "performance_curriculum": curriculum_metrics,
                }
                history.append(row)
                write_json(history_path, history)
                print(f"detail-style validation step={step} {row}", flush=True)
                if curriculum_changed:
                    next_stage = _active_performance_stage(
                        training, performance_curriculum
                    )
                    print(
                        "detail-style performance curriculum advanced "
                        f"step={step} stage={next_stage['name']}",
                        flush=True,
                    )
                if wandb_run is not None:
                    wandb_run.log({
                        f"val/functional/{mode}/{key}": value
                        for mode, values in validation.items()
                        for key, value in values.items()
                    }, step=step)
                    wandb_run.log({
                        f"val/curriculum/{key}": value
                        for key, value in curriculum_metrics.items()
                    }, step=step)
            if state_every > 0 and (step % state_every == 0 or step == steps):
                _save_state(
                    state_path, step=step, reader=reader, adapter=adapter,
                    optimizer=optimizer, cfg=cfg,
                    performance_curriculum=performance_curriculum,
                )
            if step % checkpoint_every == 0 or step == steps:
                _save_state(
                    checkpoint_dir / f"step-{step:07d}.pt", step=step,
                    reader=reader, adapter=adapter, optimizer=optimizer, cfg=cfg,
                    performance_curriculum=performance_curriculum,
                )
            panel_due = sample_every > 0 and (
                step % sample_every == 0 or step == steps
            )
            exact_fixed_due = bool(
                fixed_sample_requests
                and fixed_sample_every > 0
                and step % fixed_sample_every == 0
            )
            if panel_due or exact_fixed_due:
                requested_samples = (
                    (sample_requests if panel_due else [])
                    + (fixed_sample_requests if exact_fixed_due else [])
                )
                sample_records, vae = _sample_query_style_tokenizer(
                    anima, adapter, reader, requested_samples, config, destination,
                    output, device, step, vae,
                    config_section="detail_preserving_style_cross_attention",
                )
                if wandb_run is not None:
                    import wandb
                    panel_count = len(sample_requests) if panel_due else 0
                    payload = {}
                    if panel_due:
                        payload["val/functional/panel"] = [
                            wandb.Image(str(path), caption=label)
                            for label, path in sample_records[:panel_count]
                        ]
                    if exact_fixed_due:
                        payload["val/functional/fixed_reference"] = [
                            wandb.Image(str(path), caption=label)
                            for label, path in sample_records[panel_count:]
                        ]
                    wandb_run.log(payload, step=step)
            if fixed_prepared is not None and step % fixed_sample_every == 0:
                fixed = _generate_fixed_reference_sample(
                    fixed_prepared, config, destination, anima, reader, adapter,
                    output, device, step,
                )
                print(
                    f"detail-style fixed-reference samples step={step} "
                    f"sheet={fixed['sheet']}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "val/functional/fixed_reference": [
                            wandb.Image(path, caption=f"fixed {label} step {step}")
                            for label, path in fixed["sheets"].items()
                        ],
                        **{
                            f"val/functional/fixed_reference_pixel_rms/{label}":
                            values["mean_pixel_rms_from_baseline"]
                            for label, values in fixed["metrics"].items()
                        },
                        **{
                            f"val/functional/fixed_reference_pairwise_rms/{label}":
                            values["mean_pairwise_output_pixel_rms"]
                            for label, values in fixed["metrics"].items()
                        },
                        **{
                            f"val/functional/fixed_reference_diversity_ratio/{label}":
                            values["pairwise_output_to_baseline_ratio"]
                            for label, values in fixed["metrics"].items()
                        },
                    }, step=step)
            completed = step
            if bootstrap_complete:
                print(
                    f"detail-style native bootstrap criteria satisfied at step={step}",
                    flush=True,
                )
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        adapter.clear_style_tokens()
        del vae
        gc.collect()

    result = {
        "steps": completed,
        "requested_steps": steps,
        "reader_parameters": reader_count,
        "style_adapter_parameters": kv_count,
        "elapsed_s": time.perf_counter() - started,
        "final_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", result)
    return result


def train_detail_style_v34_lora_joint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_detail_style_cross_attention(
        config,
        destination,
        config_key="detail_preserving_style_cross_attention_v34_lora_joint",
    )


def train_artist_only_fixed_population_bootstrap(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_detail_style_cross_attention(
        config,
        destination,
        config_key="detail_style_artist_only_fixed_population_bootstrap",
    )


def train_artist_only_mixture_continuation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_detail_style_cross_attention(
        config,
        destination,
        config_key="detail_style_artist_only_mixture_continuation",
    )


def train_artist_only_mixture_one_stage(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_detail_style_cross_attention(
        config,
        destination,
        config_key="detail_style_artist_only_fixed_population_mixture_one_stage",
    )


@torch.no_grad()
def profile_detail_style_block_timestep_strength(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Measure clean-path native and adapter effects for every block/timestep bin."""

    cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    training = dict(cfg["training"])
    profile = dict(cfg.get("strength_profile", {}))
    device = str(profile.get("device", training.get("device", "cuda")))
    seed = int(profile.get("seed", cfg.get("seed", 20260819)))
    torch.manual_seed(seed)

    bank_key = str(cfg["teacher"].get("bank_config_key", "dual_domain_native_teacher"))
    bank = NativeCenteredTeacherBank.load(
        config, destination, config_key=bank_key
    )
    contexts = NativeArtistContextCache(
        destination / str(cfg["teacher"]["context_cache"]),
        capacity=int(cfg["teacher"].get("context_lru_shards", 8)),
    )
    synthetic_roots = [
        destination / str(value) for value in cfg["teacher"]["reference_caches"]
    ]
    loader = CachedTeacherReferenceLoader(
        synthetic_roots,
        split="train",
        style_ids=list(bank.summary["train_style_ids"]),
        batch_size=int(profile.get("batch_size", 4)),
        references=int(profile.get("references", 4)),
        seed=seed ^ 0xB10C7A9,
        token_lru_shards=int(cfg["teacher"].get("reference_lru_shards", 8)),
        strict_style_ids=False,
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(training.get("fuse_attention_projections", True)),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(cfg).to(device).eval()
    attach_same_q_style_adapter(anima, adapter)

    training_output = destination / str(cfg["output_directory"])
    checkpoint_value = str(
        profile.get("checkpoint", "checkpoints/step-0001000.pt")
    )
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = training_output / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reader.load_state_dict(state["reader"], strict=True)
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    checkpoint_step = int(state.get("step", 0))
    del state

    profile_training = dict(training)
    profile_training["alpha_calibration_timestep_bin_edges"] = profile.get(
        "timestep_bin_edges",
        training.get(
            "alpha_calibration_timestep_bin_edges",
            [0.0, 0.325, 0.625, 0.86, 1.000001],
        ),
    )
    result = _calibrate_alpha(
        anima,
        reader,
        adapter,
        bank,
        contexts,
        loader,
        device,
        int(profile.get("probe_batches", 128)),
        profile_training,
        reset_alpha=False,
        inject_style=False,
        apply_alpha=False,
        recommended_lower_multiplier=float(
            profile.get("recommended_lower_multiplier", 1.5)
        ),
        recommended_upper_multiplier=float(
            profile.get("recommended_upper_multiplier", 1.5)
        ),
    )
    result.update({
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "probe_batches": int(profile.get("probe_batches", 128)),
        "batch_size": loader.batch_size,
        "references": int(profile.get("references", 4)),
        "clean_frozen_anima_path": True,
    })
    output = destination / str(
        profile.get(
            "output_directory",
            "diagnostics/detail_style_block_timestep_strength_v1",
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "profile.json", result)
    summary = {
        "output": str(output / "profile.json"),
        "checkpoint_step": checkpoint_step,
        "probe_batches": result["probe_batches"],
        "batch_size": result["batch_size"],
        "timestep_bins": len(result["block_timestep_profiles"]),
        "blocks": adapter.blocks,
    }
    write_json(output / "summary.json", summary)
    return summary


@torch.no_grad()
def diagnose_detail_style_attenuation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Locate style-effect attenuation from attention space to final velocity."""

    cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    training = dict(cfg["training"])
    diagnostic = dict(cfg.get("attenuation_diagnostic", {}))
    device = str(diagnostic.get("device", training.get("device", "cuda")))
    seed = int(diagnostic.get("seed", cfg.get("seed", 20260819)))
    torch.manual_seed(seed)

    bank_key = str(cfg["teacher"].get("bank_config_key", "dual_domain_native_teacher"))
    bank = NativeCenteredTeacherBank.load(config, destination, config_key=bank_key)
    contexts = NativeArtistContextCache(
        destination / str(cfg["teacher"]["context_cache"]),
        capacity=int(cfg["teacher"].get("context_lru_shards", 8)),
    )
    loader = CachedTeacherReferenceLoader(
        [destination / str(value) for value in cfg["teacher"]["reference_caches"]],
        split="train",
        style_ids=list(bank.summary["train_style_ids"]),
        batch_size=int(diagnostic.get("batch_size", 4)),
        references=int(diagnostic.get("references", 4)),
        seed=seed ^ 0xA77E0A7E,
        token_lru_shards=int(cfg["teacher"].get("reference_lru_shards", 8)),
        strict_style_ids=False,
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(training.get("fuse_attention_projections", True)),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(cfg).to(device).eval()
    attach_same_q_style_adapter(anima, adapter)

    training_output = destination / str(cfg["output_directory"])
    checkpoint_value = str(
        diagnostic.get("checkpoint", "checkpoints/step-0000500.pt")
    )
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = training_output / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reader.load_state_dict(state["reader"], strict=True)
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    checkpoint_step = int(state.get("step", 0))
    del state

    tensors = bank.tensors
    content_count = int(tensors["noisy_inputs"].shape[0])
    timestep_count = int(tensors["noisy_inputs"].shape[1])
    probe_batches = int(diagnostic.get("probe_batches", content_count * timestep_count))
    edges = tuple(
        float(value)
        for value in diagnostic.get(
            "timestep_bin_edges",
            training.get(
                "alpha_calibration_timestep_bin_edges",
                [0.0, 0.325, 0.625, 0.86, 1.000001],
            ),
        )
    )
    block_stage_rows: dict[int, dict[str, list[dict[str, float]]]] = {
        block: {} for block in range(adapter.blocks)
    }
    bin_stage_rows: dict[int, dict[str, list[dict[str, float]]]] = {
        index: {} for index in range(len(edges) - 1)
    }
    final_rows: list[dict[str, float]] = []
    final_bin_rows: dict[int, list[dict[str, float]]] = {
        index: [] for index in range(len(edges) - 1)
    }
    output_stage_rows: dict[str, list[dict[str, float]]] = {}
    bin_output_stage_rows: dict[int, dict[str, list[dict[str, float]]]] = {
        index: {} for index in range(len(edges) - 1)
    }
    active_recorder: list[_StyleAttenuationRecorder | None] = [None]

    def record_output_stage(stage: str, value: torch.Tensor) -> None:
        if active_recorder[0] is not None:
            active_recorder[0].record_output_stage(stage, value)

    final_handles = [
        anima.final_layer.register_forward_pre_hook(
            lambda _module, args: record_output_stage("final_input", args[0])
        ),
        anima.final_layer.layer_norm.register_forward_hook(
            lambda _module, _args, output: record_output_stage(
                "final_layer_norm", output
            )
        ),
        anima.final_layer.linear.register_forward_pre_hook(
            lambda _module, args: record_output_stage("final_adaln", args[0])
        ),
        anima.final_layer.linear.register_forward_hook(
            lambda _module, _args, output: record_output_stage(
                "final_linear", output
            )
        ),
    ]

    for probe_index in range(probe_batches):
        batch = loader.load_step(probe_index)
        references, mask = _reference_inputs(batch, device, "heldout")
        rows = references.shape[0]
        style_ids = [str(item.style_id) for item in batch["episodes"]]
        artist_indices = torch.tensor(
            [bank.artist_to_index[value] for value in style_ids], dtype=torch.long
        )
        timestep_index = probe_index % timestep_count
        content_index = (probe_index // timestep_count) % content_count
        timestep = tensors["timesteps"][timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        timestep_value = float(timestep.float())
        bin_index = max(
            0,
            min(
                len(edges) - 2,
                next(
                    (
                        index
                        for index in range(len(edges) - 1)
                        if edges[index] <= timestep_value < edges[index + 1]
                    ),
                    len(edges) - 2,
                ),
            ),
        )
        noisy = tensors["noisy_inputs"][content_index, timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        context = tensors["base_context"][content_index : content_index + 1].to(
            device=device, dtype=torch.bfloat16
        )
        tagged = contexts.get(style_ids, content_index).to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        padding = torch.zeros(
            rows, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=noisy.dtype,
        )
        recorder = _StyleAttenuationRecorder()
        active_recorder[0] = recorder
        adapter.set_diagnostic_recorder(recorder)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            adapter.clear_style_tokens()
            recorder.mode = "base"
            base_prediction = anima(
                noisy.expand(rows, -1, -1, -1).unsqueeze(2),
                timestep.expand(rows),
                context=context.expand(rows, -1, -1),
                padding_mask=padding,
                target_input_ids=None,
            ).squeeze(2).float()

            style = reader(references, mask).tokens
            recorder.mode = "style"
            adapter.reset_internal_teacher()
            adapter.set_style_context(style)
            adapter.set_teacher_context(tagged)
            adapter.set_timesteps(timestep.expand(rows))
            try:
                style_prediction = anima(
                    noisy.expand(rows, -1, -1, -1).unsqueeze(2),
                    timestep.expand(rows),
                    context=context.expand(rows, -1, -1),
                    padding_mask=padding,
                    target_input_ids=None,
                ).squeeze(2).float()
            finally:
                adapter.clear_style_tokens()
                adapter.set_diagnostic_recorder(None)
                active_recorder[0] = None
        # ``record_gated_internal_teacher`` retains its differentiable terms
        # until the objective is consumed.  The diagnostic uses only the
        # reduced recorder statistics, so consume and release them per probe.
        adapter.internal_teacher_loss(rho_min=0.0)

        captured = recorder.finish()
        for block_index, stages in captured.items():
            for stage, metrics in stages.items():
                block_stage_rows[block_index].setdefault(stage, []).append(metrics)
                bin_stage_rows[bin_index].setdefault(stage, []).append(metrics)
        for stage, metrics in recorder.output_metrics.items():
            output_stage_rows.setdefault(stage, []).append(metrics)
            bin_output_stage_rows[bin_index].setdefault(stage, []).append(metrics)

        final_teacher = tensors["centered_teacher"][
            artist_indices, content_index, timestep_index
        ].to(device=device, dtype=torch.float32, non_blocking=True)
        final_metrics = _effect_stage_metrics(
            style_prediction - base_prediction, final_teacher
        )
        dimensions = tuple(range(1, base_prediction.ndim))
        final_metrics.update({
            "output_to_base_rms": float(
                (style_prediction - base_prediction).square()
                .mean(dim=dimensions).sqrt().mean()
                / base_prediction.square().mean(dim=dimensions).sqrt().mean().clamp_min(1e-8)
            ),
            "timestep": timestep_value,
        })
        final_rows.append(final_metrics)
        final_bin_rows[bin_index].append(final_metrics)
        print(
            f"attenuation probe={probe_index + 1}/{probe_batches} "
            f"content={content_index} timestep={timestep_value:.4f} "
            f"final_ratio={final_metrics['output_to_base_rms']:.5f} "
            f"projection={final_metrics['teacher_projection']:.5f}",
            flush=True,
        )

    for handle in final_handles:
        handle.remove()

    block_summary: list[dict[str, Any]] = []
    for block_index, stages in block_stage_rows.items():
        reduced = {
            stage: _mean_scalar_rows(rows) for stage, rows in stages.items()
        }
        attenuation = {}
        if {"pre_o", "post_o"} <= reduced.keys():
            attenuation["o_rms_retention"] = (
                reduced["post_o"]["centered_rms"]
                / max(reduced["pre_o"]["centered_rms"], 1e-8)
            )
            attenuation["o_projection_change"] = (
                reduced["post_o"]["teacher_projection"]
                - reduced["pre_o"]["teacher_projection"]
            )
        if {"post_o", "post_gate"} <= reduced.keys():
            attenuation["gate_rms_retention"] = (
                reduced["post_gate"]["centered_rms"]
                / max(reduced["post_o"]["centered_rms"], 1e-8)
            )
            attenuation["gate_projection_change"] = (
                reduced["post_gate"]["teacher_projection"]
                - reduced["post_o"]["teacher_projection"]
            )
        if {"post_cross_hidden", "post_mlp_hidden"} <= reduced.keys():
            attenuation["mlp_cumulative_rms_retention"] = (
                reduced["post_mlp_hidden"]["centered_rms"]
                / max(reduced["post_cross_hidden"]["centered_rms"], 1e-8)
            )
        if {"post_self_hidden", "post_cross_hidden"} <= reduced.keys():
            attenuation["cross_relative_effect_retention"] = (
                reduced["post_cross_hidden"]["effect_to_base_rms"]
                / max(reduced["post_self_hidden"]["effect_to_base_rms"], 1e-8)
            )
        if {"post_cross_hidden", "post_mlp_hidden"} <= reduced.keys():
            attenuation["mlp_relative_effect_retention"] = (
                reduced["post_mlp_hidden"]["effect_to_base_rms"]
                / max(reduced["post_cross_hidden"]["effect_to_base_rms"], 1e-8)
            )
        block_summary.append({
            "block": block_index,
            "stages": reduced,
            "attenuation": attenuation,
        })
    for block_index in range(1, len(block_summary)):
        current = block_summary[block_index]
        previous = block_summary[block_index - 1]
        current["attenuation"]["self_relative_effect_retention"] = (
            current["stages"]["post_self_hidden"]["effect_to_base_rms"]
            / max(
                previous["stages"]["post_mlp_hidden"]["effect_to_base_rms"],
                1e-8,
            )
        )

    bin_summary = []
    for bin_index, stages in bin_stage_rows.items():
        bin_summary.append({
            "bin": bin_index,
            "range": [edges[bin_index], edges[bin_index + 1]],
            "stages": {
                stage: _mean_scalar_rows(rows) for stage, rows in stages.items()
            },
            "final_layer_stages": {
                stage: _mean_scalar_rows(rows)
                for stage, rows in bin_output_stage_rows[bin_index].items()
            },
            "final_velocity": _mean_scalar_rows(final_bin_rows[bin_index]),
        })
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "probe_batches": probe_batches,
        "batch_size": loader.batch_size,
        "references": int(diagnostic.get("references", 4)),
        "timestep_bin_edges": list(edges),
        "final_layer_stages": {
            stage: _mean_scalar_rows(rows)
            for stage, rows in output_stage_rows.items()
        },
        "final_velocity": _mean_scalar_rows(final_rows),
        "blocks": block_summary,
        "timestep_bins": bin_summary,
    }
    output = destination / str(
        diagnostic.get(
            "output_directory",
            "diagnostics/detail_style_attenuation_v10_step500",
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "attenuation.json", result)
    summary = {
        "output": str(output / "attenuation.json"),
        "checkpoint_step": checkpoint_step,
        "probe_batches": probe_batches,
        "final_velocity": result["final_velocity"],
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_detail_style_cross_attention(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    cfg = smoke["detail_preserving_style_cross_attention"]
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke16"
    cfg.setdefault("data_mixture", {})["enabled"] = False
    cfg["loader"]["batch_size"] = 2
    cfg["training"].update({
        "steps": 2,
        "gradient_accumulation_steps": 1,
        "validation_every": 2,
        "validation_batches": 1,
        "checkpoint_every": 2,
        "sample_every": 0,
        "resume": False,
        "alpha_calibration_batches": 1,
        "teacher_batch_rows": 16,
        "teacher_microbatch_rows": 4,
        "functional_start_step": 1,
        "functional_every": 1,
        "artist_effect_start_step": 1,
        "artist_effect_full_step": 1,
        "artist_effect_every": 1,
        "common_output_start_step": 1,
        "common_output_full_step": 1,
        "main_common_output_start_step": 1,
        "main_common_output_full_step": 1,
        "main_common_output_every": 1,
        "main_common_output_batch_rows": 2,
        "artist_magnitude_start_step": 1,
        "artist_magnitude_full_step": 1,
        "artist_effect_validation_batches": 1,
        "artist_effect_validation_timesteps": [0.45],
        "artist_prototype_start_step": 1,
        "artist_prototype_full_step": 1,
        "artist_prototype_every": 1,
    })
    cfg["training"].setdefault("wandb", {})["enabled"] = False
    return train_detail_style_cross_attention(smoke, destination, steps_override=2)


def smoke_test_detail_style_v34_lora_joint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    cfg = _merge_detail_style_config(
        smoke, "detail_preserving_style_cross_attention_v34_lora_joint"
    )
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    cfg["initial_checkpoint"] = None
    cfg.setdefault("data_mixture", {})["enabled"] = False
    cfg["loader"]["batch_size"] = 2
    cfg["training"].update({
        "steps": 2,
        "gradient_accumulation_steps": 1,
        "validation_every": 2,
        "validation_batches": 1,
        "checkpoint_every": 2,
        "state_every": 2,
        "sample_every": 0,
        "fixed_sample_every": 0,
        "resume": False,
        "alpha_calibration_batches": 1,
        "teacher_batch_rows": 4,
        "teacher_microbatch_rows": 2,
        "artist_effect_validation_batches": 1,
        "artist_effect_validation_timesteps": [0.45],
    })
    cfg["training"].setdefault("wandb", {})["enabled"] = False
    cfg["lora_functional_teacher"]["training"].update({
        "every": 1,
        "batch_rows": 2,
        "single_only_steps": 2,
        "reference_counts": [1],
        "reference_count_weights": [1.0],
    })
    smoke["detail_preserving_style_cross_attention"] = cfg
    return train_detail_style_cross_attention(smoke, destination, steps_override=2)


def smoke_test_artist_only_fixed_population_bootstrap(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    cfg = _merge_detail_style_config(
        smoke, "detail_style_artist_only_fixed_population_bootstrap"
    )
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    cfg.setdefault("data_mixture", {})["enabled"] = False
    cfg["loader"]["batch_size"] = 2
    cfg["training"].update({
        "steps": 2,
        "validation_every": 2,
        "validation_batches": 1,
        "checkpoint_every": 2,
        "state_every": 2,
        "sample_every": 0,
        "fixed_sample_every": 0,
        "resume": False,
        "alpha_calibration_batches": 1,
        "teacher_batch_rows": 4,
        "teacher_microbatch_rows": 2,
    })
    cfg["training"].setdefault("wandb", {})["enabled"] = False
    cfg["lora_functional_teacher"]["training"].update({
        "start_step": 1,
        "every": 1,
        "batch_rows": 2,
        "reference_counts": [1],
        "reference_count_weights": [1.0],
    })
    smoke["detail_preserving_style_cross_attention"] = cfg
    return train_detail_style_cross_attention(smoke, destination, steps_override=2)


def backfill_detail_style_fixed_samples(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render missing fixed TestSample panels from periodic checkpoints."""

    cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    every = int(training.get("fixed_sample_every", 1000))
    strengths = [
        float(value)
        for value in training.get(
            "fixed_sample_backfill_strengths",
            training.get("fixed_sample_strengths", [1.0, 1.5, 2.0]),
        )
    ]
    if every <= 0:
        raise ValueError("fixed_sample_every must be positive for backfill")
    output = destination / str(cfg["output_directory"])
    checkpoint_dir = output / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
    requested_steps = {
        int(value) for value in training.get("fixed_sample_backfill_steps", [])
    }
    if requested_steps:
        checkpoints = [
            path for path in checkpoints
            if int(path.stem.removeprefix("step-")) in requested_steps
        ]
    else:
        checkpoints = [
            path for path in checkpoints
            if int(path.stem.removeprefix("step-")) % every == 0
        ]
    if not checkpoints:
        raise FileNotFoundError(f"No {every}-step checkpoints in {checkpoint_dir}")
    configured_modes = training.get("fixed_sample_backfill_component_modes")
    component_modes: list[str | None] = (
        [None]
        if configured_modes is None
        else [str(value) for value in configured_modes]
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(training.get("fuse_attention_projections", True)),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(cfg).to(device).eval()
    attach_same_q_style_adapter(anima, adapter)
    prepared = load_dual_query_external_sample(config, destination)

    generated = []
    reused = []
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.removeprefix("step-"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reader.load_state_dict(state["reader"], strict=True)
        adapter.load_state_dict(state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        del state
        for component_mode in component_modes:
            sample_name = f"step-{step:07d}"
            if component_mode is not None:
                sample_name += f"-{component_mode}"
            summary_path = (
                output / "external_reference_samples"
                / sample_name / "summary.json"
            )
            if _fixed_sample_complete(summary_path, strengths):
                reused.append({"step": step, "component_mode": component_mode})
                continue
            result = _generate_fixed_reference_sample(
                prepared, config, destination, anima, reader, adapter,
                output, device, step, component_mode=component_mode,
                strengths_override=strengths,
            )
            generated.append({
                "step": step,
                "component_mode": component_mode or "combined",
                "sheet": result["sheet"],
            })
            print(
                f"detail-style fixed-reference backfill step={step} "
                f"component={component_mode or 'combined'} "
                f"sheet={result['sheet']}",
                flush=True,
            )
    summary = {
        "checkpoint_count": len(checkpoints),
        "generated": generated,
        "reused_steps": reused,
    }
    write_json(output / "external_reference_samples" / "backfill_summary.json", summary)
    return summary


def _reference_montage(
    paths: list[Path], destination: Path, size: tuple[int, int]
) -> Path:
    columns = 1 if len(paths) == 1 else 2
    rows = math.ceil(len(paths) / columns)
    cell = (size[0] // columns, size[1] // rows)
    canvas = Image.new("RGB", size, "white")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            preview = _fit_with_padding(source, cell)
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        canvas.paste(preview, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, compress_level=4)
    return destination


def evaluate_detail_style_reference_counts(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Compare one and four held-out references under identical generation."""

    cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    evaluation = dict(cfg.get("controlled_reference_evaluation", {}))
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    checkpoint_dir = output / "checkpoints"
    checkpoint_steps = [
        int(value) for value in evaluation.get("checkpoint_steps", [500])
    ]
    reference_counts = [
        int(value) for value in evaluation.get("reference_counts", [1, 4])
    ]
    artist_count = int(evaluation.get("artists", 8))
    selection_seed = int(evaluation.get("selection_seed", 20260822))
    strengths = [
        float(value) for value in evaluation.get("strengths", [1.0, 2.0])
    ]
    if reference_counts != [1, 4]:
        raise ValueError("controlled reference evaluation requires counts [1,4]")
    if artist_count <= 1:
        raise ValueError("controlled reference evaluation needs at least two artists")

    loaders: dict[int, MultiPromptDualQueryCachedStyleLoader] = {}
    for count in reference_counts:
        loader_cfg = _loader_config(
            config, cfg, split=str(cfg.get("validation_split", "validation"))
        )
        loader_cfg.update({
            "seed": selection_seed,
            "batch_size": 1,
            "min_references": count,
            "max_references": count,
            "reference_count_weights": None,
            "reference_curriculum": {},
            "pilot_reference_schedule": [],
            "quality_probability": 0.0,
            "prompt_modes": {
                "full": 1.0,
                "tag_dropout": 0.0,
                "short": 0.0,
                "empty": 0.0,
            },
        })
        loaders[count] = MultiPromptDualQueryCachedStyleLoader(
            destination, loader_cfg
        )
    episode_indices = _select_sample_episodes(loaders[1], artist_count)

    fixed = load_dual_query_external_sample(config, destination)
    width = int(fixed["cfg"]["width"])
    height = int(fixed["cfg"]["height"])
    prepared_by_count: dict[int, dict[str, Any]] = {}
    artist_records: dict[int, list[dict[str, Any]]] = {}
    expected_styles: list[str] | None = None
    preview_root = output / "controlled_reference_counts" / "reference_previews"
    for count, loader in loaders.items():
        token_batches = []
        preview_paths = []
        records = []
        styles = []
        for artist_index, episode_index in enumerate(episode_indices, start=1):
            batch = loader.load_step(episode_index)
            episode = batch["episodes"][0]
            styles.append(str(episode.style_id))
            reference_ids = [int(value) for value in episode.reference_ids]
            reference_paths = [
                Path(str(loader.style_by_id[image_id]["local_path"]))
                for image_id in reference_ids
            ]
            preview = _reference_montage(
                reference_paths,
                preview_root / f"{count}ref-artist-{artist_index:02d}.png",
                (width, height),
            )
            token_batches.append(batch["cached_reference_tokens"])
            preview_paths.append(preview)
            records.append({
                "artist_index": artist_index,
                "style_id": str(episode.style_id),
                "target_id_used_only_for_selection": int(episode.target_id),
                "reference_ids": reference_ids,
                "reference_paths": [str(value) for value in reference_paths],
                "preview": str(preview),
            })
        if expected_styles is None:
            expected_styles = styles
        elif styles != expected_styles:
            raise RuntimeError(
                "1-reference and 4-reference loaders selected different artists"
            )
        prepared_by_count[count] = {
            **fixed,
            "reference_tokens": torch.stack(token_batches),
            "reference_mask": torch.ones(
                artist_count, count, dtype=torch.bool
            ),
            "paths": preview_paths,
        }
        artist_records[count] = records

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(training.get("fuse_attention_projections", True)),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(cfg).to(device).eval()
    attach_same_q_style_adapter(anima, adapter)

    results = []
    for step in checkpoint_steps:
        checkpoint = checkpoint_dir / f"step-{step:07d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reader.load_state_dict(state["reader"], strict=True)
        adapter.load_state_dict(state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        del state
        for count in reference_counts:
            result = _generate_fixed_reference_sample(
                prepared_by_count[count], config, destination,
                anima, reader, adapter, output, device, step,
                strengths_override=strengths,
                sample_group="controlled_reference_counts",
                sample_suffix=f"{count}ref",
            )
            results.append({
                "step": step,
                "reference_count": count,
                "sheets": result["sheets"],
                "metrics": result["metrics"],
            })
            print(
                f"detail-style controlled reference evaluation step={step} "
                f"references={count} sheet={result['sheet']}",
                flush=True,
            )

    summary = {
        "contract": {
            "split": str(cfg.get("validation_split", "validation")),
            "artists": artist_count,
            "same_artists_across_reference_counts": True,
            "same_prompt_across_artists_and_counts": True,
            "same_seed_across_artists_and_counts": True,
            "reference_counts": reference_counts,
            "strengths": strengths,
            "artist_id_in_prompt": False,
        },
        "prompt": str(fixed["cfg"]["prompt"]),
        "negative_prompt": str(fixed["cfg"]["negative_prompt"]),
        "seed": int(fixed["cfg"]["seed"]),
        "checkpoint_steps": checkpoint_steps,
        "artists_by_reference_count": {
            str(key): value for key, value in artist_records.items()
        },
        "results": results,
    }
    write_json(output / "controlled_reference_counts" / "summary.json", summary)
    return summary
