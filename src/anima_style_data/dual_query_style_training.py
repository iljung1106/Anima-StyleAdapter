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
from .dual_query_external_samples import load_dual_query_external_sample
from .external_style_tokenizer_sheet import generate_live_external_style_sample
from .io import write_json
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
    _reference_inputs,
    _select_sample_episodes,
    _slot_diversity_loss,
)
from .style_tokenizer import _artist_direction_loss, _flow_metrics, insert_style_tokens
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
    tokenizer: DualQuerySetStyleTokenizer,
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
    tokenizer: DualQuerySetStyleTokenizer,
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
            "coefficient_floor": _linear_ramp(
                step,
                start_step=1,
                end_step=exact_end,
                start=float(training.get("exact_aligned_floor_start", 0.02)),
                end=float(training.get("exact_aligned_floor_end", 0.15)),
            ),
            "bounded_min": float(training.get("exact_bounded_min", 0.08)),
            "bounded_max": float(training.get("exact_bounded_max", 0.25)),
            "bounded_weight": float(training.get("exact_bounded_weight", 0.05)),
        }
    steps = int(training.get("steps", 10_000))
    return {
        "normalized_weight": float(
            training.get("heldout_normalized_residual_weight", 0.015)
        ),
        "floor_weight": float(training.get("heldout_aligned_floor_weight", 0.075)),
        "coefficient_floor": _linear_ramp(
            step,
            start_step=exact_end + 1,
            end_step=steps,
            start=float(training.get("heldout_aligned_floor_start", 0.03)),
            end=float(training.get("heldout_aligned_floor_end", 0.06)),
        ),
        "bounded_min": float(training.get("heldout_bounded_min", 0.05)),
        "bounded_max": float(training.get("heldout_bounded_max", 0.20)),
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


def _forward_dual_query_flow(
    anima: torch.nn.Module,
    tokenizer: DualQuerySetStyleTokenizer,
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
    needs_base = pilot_enabled or measure_base or normalized_weight > 0 or floor_weight > 0
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
        output = tokenizer(references, reference_mask)
        styled = insert_style_tokens(
            conditioning, conditioning_lengths, output.tokens
        )
        prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype),
            context=styled, padding_mask=padding_mask, target_input_ids=None,
        ).squeeze(2).float()
    flow_loss = F.mse_loss(prediction, target)
    total = flow_loss

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
    diversity = _slot_diversity_loss(output.tokens)
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
            target_output.tokens,
            heldout_output.tokens,
            [str(item.style_id) for item in batch["episodes"]],
            float(training.get("artist_contrastive_temperature", 0.1)),
        )
        positive = contrastive_metrics["artist_positive_similarity"]
        negative = contrastive_metrics["artist_negative_similarity"]
        contrastive_weight = float(training["artist_contrastive_weight"])
        total = total + contrastive_weight * contrastive

    bounded_loss = flow_loss.new_zeros(())
    bounded_weight = 0.0
    bounded_metrics = {
        "bounded_aligned_coefficient": flow_loss.new_zeros(()),
        "bounded_below_fraction": flow_loss.new_zeros(()),
        "bounded_above_fraction": flow_loss.new_zeros(()),
        "bounded_orthogonal_ratio": flow_loss.new_zeros(()),
    }
    bounded_every = max(1, int(training.get("bounded_effect_every", 4)))
    if pilot_enabled and step % bounded_every == 0:
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

    direction_loss = flow_loss.new_zeros(())
    direction_weight = 0.0
    direction_metrics = {
        "artist_correct_direction_cosine": flow_loss.new_zeros(()),
        "artist_wrong_direction_cosine": flow_loss.new_zeros(()),
        "artist_centered_direction_cosine": flow_loss.new_zeros(()),
        "artist_direction_ranking_loss": flow_loss.new_zeros(()),
    }
    ranking_start = int(training.get("wrong_ranking_start_step", 1000))
    ranking_every = max(1, int(training.get("wrong_ranking_every", 4)))
    if pilot_enabled and step >= ranking_start and step % ranking_every == 0:
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
        direction_weight = _linear_ramp(
            step,
            start_step=ranking_start,
            end_step=ranking_start + int(training.get("wrong_ranking_ramp_steps", 1000)),
            start=0.0,
            end=float(training.get("wrong_ranking_weight", 0.00075)),
        )
        total = total + direction_weight * direction_loss

    common_loss = flow_loss.new_zeros(())
    common_weight = 0.0
    common_metrics = {
        "common_output_ratio": flow_loss.new_zeros(()),
        "controlled_artist_effect_rms": flow_loss.new_zeros(()),
        "controlled_common_rms": flow_loss.new_zeros(()),
    }
    common_every = max(1, int(training.get("common_output_every", 8)))
    if pilot_enabled and step % common_every == 0:
        assert base_prediction is not None
        rows = min(
            int(training.get("common_output_batch_rows", 4)),
            prediction.shape[0],
        )
        common_conditioning = conditioning[0:1].expand(rows, -1, -1).clone()
        common_lengths = conditioning_lengths[0:1].expand(rows)
        common_styled = insert_style_tokens(
            common_conditioning, common_lengths, output.tokens[:rows]
        )
        common_start = int(training.get("common_output_start_step", 1500))
        common_weight = (
            _linear_ramp(
                step,
                start_step=common_start,
                end_step=common_start + int(training.get("common_output_ramp_steps", 1000)),
                start=0.0,
                end=float(training.get("common_output_weight", 0.002)),
            )
            if step >= common_start
            else 0.0
        )
        gradient_context = (
            torch.enable_grad() if common_weight > 0 else torch.no_grad()
        )
        with gradient_context, autocast_context():
            common_prediction = anima(
                noisy[0:1].expand(rows, -1, -1, -1).unsqueeze(2),
                timesteps[0:1].expand(rows).to(latents.dtype),
                context=common_styled,
                padding_mask=padding_mask[0:1].expand(rows, -1, -1, -1),
                target_input_ids=None,
            ).squeeze(2).float()
        common_delta = common_prediction - base_prediction[0:1]
        common_loss, common_metrics = _common_output_loss(
            common_delta,
            threshold=float(training.get("common_output_threshold", 0.70)),
        )
        total = total + common_weight * common_loss

    metrics = {
        "loss": total.detach(),
        "flow_loss": flow_loss.detach(),
        "normalized_residual_loss": normalized.detach(),
        "normalized_residual_weight": flow_loss.new_tensor(normalized_weight),
        "aligned_floor_loss": floor.detach(),
        "aligned_floor_weight": flow_loss.new_tensor(floor_weight),
        "aligned_coefficient_floor": flow_loss.new_tensor(
            float(alignment["coefficient_floor"])
        ),
        **{key: value.detach() for key, value in direct_metrics.items()},
        "artist_contrastive_loss": contrastive.detach(),
        "artist_contrastive_weight": flow_loss.new_tensor(contrastive_weight),
        "artist_positive_similarity": positive.detach(),
        "artist_negative_similarity": negative.detach(),
        "slot_diversity_loss": diversity.detach(),
        "bounded_effect_loss": bounded_loss.detach(),
        "bounded_effect_weight": flow_loss.new_tensor(bounded_weight),
        **{key: value.detach() for key, value in bounded_metrics.items()},
        "artist_direction_loss": direction_loss.detach(),
        "artist_direction_weight": flow_loss.new_tensor(direction_weight),
        **{key: value.detach() for key, value in direction_metrics.items()},
        "common_output_loss": common_loss.detach(),
        "common_output_weight": flow_loss.new_tensor(common_weight),
        **{key: value.detach() for key, value in common_metrics.items()},
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
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["dual_query_style_tokenizer"])
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
    reference_eval_loaders: dict[int, DualQueryCachedStyleLoader] = {}
    controlled_loader = None
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
                "artist_balanced": True,
            }
        )
        controlled_loader = DualQueryCachedStyleLoader(destination, controlled_cfg)
    cache_summary = _cache_summary(destination, cfg)
    tokenizer = DualQuerySetStyleTokenizer(**dict(cfg["model"])).to(device)
    output = destination / output_name
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    history_path = output / "history.json"
    history = json.loads(history_path.read_text("utf-8")) if history_path.exists() else []
    start_step = 0
    resume_state = None
    if bool(training.get("resume", True)) and state_path.exists():
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
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
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])

    parameters = sum(parameter.numel() for parameter in tokenizer.parameters())
    print(
        "dual-query StyleTokenizer "
        f"summary={include_artist_summary} trainable={parameters / 1e6:.2f}M "
        "output=32x1024 injection=native-text-context",
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
    sample_requests = [
        (f"train-{index}", train_loader, episode, "heldout")
        for index, episode in enumerate(_select_sample_episodes(train_loader, 4))
    ] + [
        (f"validation-{index}", validation_loader, episode, "heldout")
        for index, episode in enumerate(_select_sample_episodes(validation_loader, 4))
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
    completed = start_step
    run_started = time.perf_counter()
    running: dict[str, float] = defaultdict(float)
    running_counts: dict[str, int] = defaultdict(int)
    running_steps = 0
    try:
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            multiplier = _learning_rate_multiplier(
                step, steps, warmup, minimum_ratio
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
                    if step >= int(training.get("subset_consistency_start", 2001))
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
                (loss / accumulation).backward()
                micro_rows.append(metrics)
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
