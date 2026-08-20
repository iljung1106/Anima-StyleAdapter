"""Gradient-conflict diagnostics for detail Style Cross-Attention.

The diagnostic restores a frozen checkpoint, runs one controlled native-teacher
batch with identical noisy latent, timestep, content and Student Q, and measures
sampled parameter-space gradient cosines for each active objective.  A separate
exact-self flow row is included and explicitly labelled cross-batch.
"""

from __future__ import annotations

import copy
import gc
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .detail_style_cross_attention import (
    DetailPreservingTypedSlotReader,
    FreshKVStyleCrossAttention,
    SharedBaseKVStyleCrossAttention,
)
from .detail_style_teacher_context import NativeArtistContextCache
from .detail_style_training import (
    _all_artist_teacher_infonce,
    _active_performance_stage,
    _build_native_effect_timestep_weighting,
    _build_style_adapter,
    _flow_step,
    _initial_performance_curriculum_state,
    _loader_config,
    _native_effect_weights_for_timesteps,
    _native_teacher_objective_config,
    _reconstruction_loss,
    _scheduled_value,
    _soft_common_output_objective,
    _training_loader,
)
from .io import write_json
from .native_centered_teacher import NativeCenteredTeacherBank
from .query_style_tokenizer import _reference_inputs
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader


def _parameter_groups(
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "reader": [value for value in reader.parameters() if value.requires_grad]
    }
    if isinstance(adapter, SharedBaseKVStyleCrossAttention):
        groups.update({
            "shared_kv": list(adapter.shared_parameters()),
            "block_delta": list(adapter.delta_parameters()),
            "base_mix": list(adapter.mixing_parameters()),
        })
    else:
        groups["style_kv"] = list(adapter.kv_parameters())
    seen: set[int] = set()
    for name, values in groups.items():
        duplicate = [value for value in values if id(value) in seen]
        if duplicate:
            raise RuntimeError(f"Parameter groups overlap at {name}")
        seen.update(id(value) for value in values)
    return groups


def _gradient_sample_plan(
    groups: dict[str, list[torch.nn.Parameter]],
    *,
    samples_per_group: int,
    seed: int,
) -> tuple[
    list[torch.nn.Parameter],
    dict[str, dict[str, Any]],
]:
    """Uniformly sample coordinates without constructing a full permutation."""

    parameters: list[torch.nn.Parameter] = []
    plans: dict[str, dict[str, Any]] = {}
    for group_index, (name, values) in enumerate(groups.items()):
        first = len(parameters)
        parameters.extend(values)
        total = sum(value.numel() for value in values)
        sample_count = min(int(samples_per_group), total)
        positions = sorted(
            random.Random(seed + group_index * 1_000_003).sample(
                range(total), sample_count
            )
        )
        entries = []
        cursor = 0
        offset = 0
        for local_parameter_index, value in enumerate(values):
            stop = offset + value.numel()
            begin_cursor = cursor
            while cursor < sample_count and positions[cursor] < stop:
                cursor += 1
            if cursor > begin_cursor:
                local_indices = torch.tensor(
                    [
                        position - offset
                        for position in positions[begin_cursor:cursor]
                    ],
                    device=value.device,
                    dtype=torch.long,
                )
                entries.append({
                    "parameter_index": first + local_parameter_index,
                    "indices": local_indices,
                    "output_start": begin_cursor,
                    "output_stop": cursor,
                })
            offset = stop
        if cursor != sample_count:
            raise RuntimeError(f"Failed to map gradient samples for {name}")
        plans[name] = {
            "total_parameters": total,
            "sample_count": sample_count,
            "entries": entries,
        }
    return parameters, plans


def _measure_gradient_sketches(
    losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    plans: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, float]],
]:
    sketches: dict[str, dict[str, torch.Tensor]] = {}
    statistics: dict[str, dict[str, float]] = {}
    items = list(losses.items())
    for loss_index, (loss_name, loss) in enumerate(items):
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"Non-finite diagnostic loss {loss_name}: {loss}")
        if not loss.requires_grad:
            raise RuntimeError(f"Diagnostic loss has no gradient: {loss_name}")
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=loss_index + 1 < len(items),
            allow_unused=True,
        )
        sketches[loss_name] = {}
        statistics[loss_name] = {}
        for group_name, plan in plans.items():
            vector = torch.zeros(int(plan["sample_count"]), dtype=torch.float32)
            exact_norm_squared = 0.0
            for entry in plan["entries"]:
                gradient = gradients[int(entry["parameter_index"])]
                if gradient is None:
                    continue
                detached = gradient.detach().float()
                exact_norm_squared += float(detached.square().sum())
                selected = detached.flatten().index_select(
                    0, entry["indices"]
                ).cpu()
                vector[
                    int(entry["output_start"]):int(entry["output_stop"])
                ] = selected
            sketches[loss_name][group_name] = vector
            statistics[loss_name][f"{group_name}_exact_norm"] = math.sqrt(
                exact_norm_squared
            )
            statistics[loss_name][f"{group_name}_sample_norm"] = float(
                vector.norm()
            )
            statistics[loss_name][f"{group_name}_sample_nonzero_fraction"] = (
                float(vector.ne(0).float().mean()) if vector.numel() else 0.0
            )
        del gradients
    return sketches, statistics


def _append_gradient_sketch(
    name: str,
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    plans: dict[str, dict[str, Any]],
    sketches: dict[str, dict[str, torch.Tensor]],
    statistics: dict[str, dict[str, float]],
) -> None:
    measured, measured_statistics = _measure_gradient_sketches(
        {name: loss}, parameters, plans
    )
    sketches.update(measured)
    statistics.update(measured_statistics)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(left.norm() * right.norm())
    if denominator <= 0:
        return None
    return float(torch.dot(left, right) / denominator)


def _cosine_matrices(
    sketches: dict[str, dict[str, torch.Tensor]],
    plans: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    names = list(sketches)
    matrices: dict[str, dict[str, dict[str, float | None]]] = {}
    group_names = list(plans)
    for group_name in group_names:
        matrices[group_name] = {
            left: {
                right: _cosine(
                    sketches[left][group_name], sketches[right][group_name]
                )
                for right in names
            }
            for left in names
        }
    weighted = {}
    for loss_name in names:
        rows = []
        for group_name, plan in plans.items():
            sample_count = max(1, int(plan["sample_count"]))
            weight = math.sqrt(int(plan["total_parameters"]) / sample_count)
            rows.append(sketches[loss_name][group_name] * weight)
        weighted[loss_name] = torch.cat(rows)
    matrices["all_trainable"] = {
        left: {
            right: _cosine(weighted[left], weighted[right])
            for right in names
        }
        for left in names
    }
    return matrices


def _most_conflicting_pairs(
    matrices: dict[str, dict[str, dict[str, float | None]]],
    *,
    maximum: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for group_name, matrix in matrices.items():
        names = list(matrix)
        pairs = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                cosine = matrix[left][right]
                if cosine is not None:
                    pairs.append({"left": left, "right": right, "cosine": cosine})
        result[group_name] = sorted(pairs, key=lambda row: row["cosine"])[
            :maximum
        ]
    return result


def _controlled_losses(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    bank: NativeCenteredTeacherBank,
    contexts: NativeArtistContextCache,
    batch: dict[str, Any],
    device: str,
    training: dict[str, Any],
    diagnostic: dict[str, Any],
    timestep_weighting: dict[str, torch.Tensor] | None,
    *,
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    references, mask = _reference_inputs(batch, device, "heldout")
    rows = min(int(diagnostic.get("batch_size", 4)), references.shape[0])
    references, mask = references[:rows], mask[:rows]
    style_ids = [str(item.style_id) for item in batch["episodes"][:rows]]
    if len(set(style_ids)) != rows:
        raise RuntimeError("Controlled gradient batch requires distinct artists")
    artist_indices = torch.tensor(
        [bank.artist_to_index[value] for value in style_ids], dtype=torch.long
    )
    content_index = int(diagnostic.get("content_index", 0))
    timestep_index = int(diagnostic.get("timestep_index", 3))
    tensors = bank.tensors
    noisy = tensors["noisy_inputs"][content_index, timestep_index].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    base = tensors["base_predictions"][content_index, timestep_index].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    teacher = tensors["centered_teacher"][
        artist_indices, content_index, timestep_index
    ].to(device=device, dtype=torch.float32, non_blocking=True)
    base_context = tensors["base_context"][content_index:content_index + 1].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    timestep = tensors["timesteps"][timestep_index].to(
        device=device, dtype=torch.bfloat16
    )
    post_gate_cfg = dict(training["post_gate_teacher_distillation"])
    block_indices = tuple(int(value) for value in post_gate_cfg["block_indices"])
    tagged = contexts.get(style_ids, content_index).to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )

    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        output = reader(references, mask, reconstruct=True)
        adapter.reset_internal_teacher()
        adapter.set_style_context(output.tokens)
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
                padding_mask=padding,
                target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()
    student = prediction - base
    dimensions = tuple(range(1, teacher.ndim))
    student_center = student.mean(dim=0, keepdim=True)
    teacher_center = teacher.mean(dim=0, keepdim=True)
    centered_student = student - student_center
    centered_teacher = teacher - teacher_center
    objective = _native_teacher_objective_config(training)
    scale_floor = float(objective.get("native_teacher_scale_floor", 1e-4))
    teacher_power = centered_teacher.square().mean(dim=dimensions).clamp_min(
        scale_floor**2
    )
    teacher_rms = teacher_power.sqrt()
    scale_view = teacher_rms.reshape(-1, *([1] * (teacher.ndim - 1)))
    beta = float(objective.get("native_teacher_huber_beta", 0.1))
    full_residual = F.smooth_l1_loss(
        (centered_student - centered_teacher) / scale_view,
        torch.zeros_like(centered_student),
        beta=beta,
    )

    coefficient = (
        centered_student * centered_teacher
    ).mean(dim=dimensions) / teacher_power
    magnitude_floor = _scheduled_value(
        step,
        int(objective.get("magnitude_floor_start_step", 1)),
        int(objective.get("magnitude_floor_end_step", 1000)),
        float(objective.get("magnitude_floor_start", 0.30)),
        float(objective.get("magnitude_floor_end", 0.70)),
    )
    magnitude_upper = float(objective.get("magnitude_upper", 1.50))
    magnitude_band = (
        F.relu(magnitude_floor - coefficient).square().mean()
        + float(objective.get("magnitude_upper_weight", 0.25))
        * F.relu(coefficient - magnitude_upper).square().mean()
    )
    absolute_direction = (
        1.0
        - F.cosine_similarity(
            centered_student.flatten(1), centered_teacher.flatten(1),
            dim=1, eps=1e-8,
        )
    ).mean()
    labels = torch.arange(rows, device=device, dtype=torch.long)
    infonce, infonce_metrics = _all_artist_teacher_infonce(
        centered_student,
        centered_teacher,
        centered_teacher,
        labels,
        temperature=float(training.get("teacher_infonce_temperature", 0.10)),
    )

    native_scale = teacher_rms.median().clamp_min(1e-8)
    common_loss, common_metrics = _soft_common_output_objective(
        student_center,
        native_scale,
        ratio_threshold=float(
            training.get("teacher_common_output_ratio_threshold", 0.60)
        ),
        softness=float(training.get("teacher_common_output_softness", 0.05)),
    )
    common_ratio = common_metrics["native_teacher_common_output_ratio"]

    low_frequency_losses = {}
    scales = [
        int(value)
        for value in objective.get("low_frequency_residual_pool_scales", [2, 4])
    ]
    for pool_scale in scales:
        pooled_student = F.avg_pool2d(centered_student, pool_scale, pool_scale)
        pooled_teacher = F.avg_pool2d(centered_teacher, pool_scale, pool_scale)
        pooled_dimensions = tuple(range(1, pooled_teacher.ndim))
        pooled_scale = pooled_teacher.square().mean(
            dim=pooled_dimensions
        ).sqrt().clamp_min(1e-4)
        pooled_scale = pooled_scale.reshape(
            -1, *([1] * (pooled_teacher.ndim - 1))
        )
        low_frequency_losses[pool_scale] = F.smooth_l1_loss(
            (pooled_student - pooled_teacher) / pooled_scale,
            torch.zeros_like(pooled_student),
            beta=beta,
        )

    reconstruction, reconstruction_metrics = _reconstruction_loss(
        output.reconstruction, output.reconstruction_target
    )
    global_weight = float(training.get("teacher_global_weight", 0.10))
    timestep_weight = _native_effect_weights_for_timesteps(
        timestep.float().reshape(1), timestep_weighting
    )[0]
    outer = global_weight * timestep_weight
    losses: dict[str, torch.Tensor] = {
        "final_full_residual": outer
        * float(objective.get("residual_weight", 0.025)) * full_residual,
        "final_absolute_direction": outer
        * float(objective.get("direction_weight", 0.10)) * absolute_direction,
        "final_magnitude_band": outer
        * float(objective.get("magnitude_weight", 0.10)) * magnitude_band,
        "all_artist_infonce": outer
        * float(training.get("teacher_infonce_weight", 0.10)) * infonce,
        "common_suppression": outer
        * float(training.get("teacher_common_output_weight", 0.10))
        * common_loss,
        "candidate_reconstruction": float(training.get("reconstruction_weight", 0.01))
        * reconstruction,
    }
    low_frequency_weight = float(objective.get("low_frequency_residual_weight", 0.10))
    for pool_scale, value in low_frequency_losses.items():
        losses[f"final_low_frequency_{pool_scale}x"] = (
            outer * low_frequency_weight / max(1, len(low_frequency_losses))
            * value
        )

    post_gate_losses = []
    terms = list(adapter._post_gate_distillation_terms)
    if len(terms) != len(block_indices):
        raise RuntimeError(
            f"Expected {len(block_indices)} post-gate terms, captured {len(terms)}"
        )
    for block_index, values in terms:
        weights = values["valid"].to(values["cosine"].dtype)
        direction = (
            (1.0 - values["cosine"]) * weights
        ).sum() / weights.sum().clamp_min(1.0)
        weighted = (
            outer * float(post_gate_cfg.get("weight", 0.10))
            * float(post_gate_cfg.get("direction_weight", 1.0))
            / len(terms)
            * direction
        )
        losses[f"postgate_block_{block_index}_direction"] = weighted
        post_gate_losses.append(weighted)

    active_names = [
        "final_full_residual",
        "final_absolute_direction",
        "final_magnitude_band",
        "all_artist_infonce",
        "common_suppression",
        *[f"final_low_frequency_{value}x" for value in scales],
        *[f"postgate_block_{value}_direction" for value in block_indices],
    ]
    losses["active_teacher_total"] = torch.stack(
        [losses[name] for name in active_names]
    ).sum()
    metrics = {
        "style_ids": style_ids,
        "content_index": content_index,
        "timestep_index": timestep_index,
        "timestep": float(timestep),
        "student_rms": float(student.detach().square().mean().sqrt()),
        "teacher_rms": float(teacher.detach().square().mean().sqrt()),
        "native_projection": float(coefficient.detach().mean()),
        "native_cosine": float(
            F.cosine_similarity(
                centered_student.detach().flatten(1),
                centered_teacher.flatten(1), dim=1, eps=1e-8
            ).mean()
        ),
        "absolute_centered_cosine": float(1.0 - absolute_direction.detach()),
        "infonce_accuracy": float(infonce_metrics["teacher_infonce_accuracy"]),
        "infonce_cosine_gap": float(infonce_metrics["teacher_infonce_cosine_gap"]),
        "common_ratio": float(common_ratio.detach()),
        "artist_projection": float(coefficient.detach().mean()),
        "teacher_global_weight": global_weight,
        "teacher_timestep_weight": float(timestep_weight.detach()),
        "reconstruction_cosine_loss": float(
            reconstruction_metrics["reconstruction_cosine_loss"]
        ),
    }
    return losses, metrics


def run_detail_style_gradient_diagnostics(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    training = dict(cfg["training"])
    diagnostic = dict(cfg.get("gradient_diagnostics", {}))
    device = str(diagnostic.get("device", training.get("device", "cuda")))
    seed = int(diagnostic.get("seed", cfg.get("seed", 20260819)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))

    output = destination / str(cfg["output_directory"])
    checkpoint_step = int(diagnostic.get("checkpoint_step", 1500))
    checkpoint = output / "checkpoints" / f"step-{checkpoint_step:07d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    bank_key = str(cfg["teacher"].get("bank_config_key", "dual_domain_native_teacher"))
    bank = NativeCenteredTeacherBank.load(config, destination, config_key=bank_key)
    contexts = NativeArtistContextCache(
        destination / str(cfg["teacher"]["context_cache"]),
        capacity=int(cfg["teacher"].get("context_lru_shards", 8)),
    )
    teacher_loader = CachedTeacherReferenceLoader(
        [destination / str(value) for value in cfg["teacher"]["reference_caches"]],
        split="train",
        style_ids=list(bank.summary["train_style_ids"]),
        batch_size=int(diagnostic.get("batch_size", 4)),
        references=int(training.get("teacher_references", 4)),
        seed=seed ^ 0x7EA4CE11,
        token_lru_shards=int(cfg["teacher"].get("reference_lru_shards", 8)),
        strict_style_ids=False,
    )
    timestep_weighting = _build_native_effect_timestep_weighting(bank, training)

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(training.get("fuse_attention_projections", True)),
    )
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(cfg).to(device).eval()
    attach_same_q_style_adapter(anima, adapter)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reader.load_state_dict(state["reader"], strict=True)
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    del state

    groups = _parameter_groups(reader, adapter)
    parameters, plans = _gradient_sample_plan(
        groups,
        samples_per_group=int(diagnostic.get("gradient_samples_per_group", 131072)),
        seed=seed,
    )
    teacher_batch_step = int(diagnostic.get("teacher_batch_step", checkpoint_step))
    controlled_losses, controlled_metrics = _controlled_losses(
        anima,
        reader,
        adapter,
        bank,
        contexts,
        teacher_loader.load_step(teacher_batch_step),
        device,
        training,
        diagnostic,
        timestep_weighting,
        step=checkpoint_step,
    )
    loss_values = {
        name: float(value.detach()) for name, value in controlled_losses.items()
    }
    sketches, gradient_statistics = _measure_gradient_sketches(
        controlled_losses, parameters, plans
    )
    del controlled_losses
    adapter.reset_internal_teacher()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Main flow uses the same frozen checkpoint but a fixed exact-self human
    # batch. Its cosine with controlled losses is intentionally labelled
    # cross-batch; it diagnoses optimizer-level conflict, not per-example fit.
    train_cfg = _loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    primary_loader, _ = _training_loader(destination, cfg, train_cfg)
    stage_state = _initial_performance_curriculum_state(training, None)
    stage = _active_performance_stage(training, stage_state)
    generator = torch.Generator(device=device).manual_seed(seed ^ 0xF10A1500)
    main_flow, main_metrics, _ = _flow_step(
        anima,
        reader,
        adapter,
        primary_loader.load_step(checkpoint_step),
        device,
        training,
        timestep_weighting,
        generator=generator,
        step=checkpoint_step,
        mode="curriculum",
        train_auxiliaries=False,
        measure_base=True,
        performance_stage=stage,
    )
    loss_values["cross_batch_exact_self_flow"] = float(main_flow.detach())
    _append_gradient_sketch(
        "cross_batch_exact_self_flow",
        main_flow,
        parameters,
        plans,
        sketches,
        gradient_statistics,
    )
    main_summary = {
        key: float(value.detach())
        for key, value in main_metrics.items()
        if value.numel() == 1
    }

    matrices = _cosine_matrices(sketches, plans)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "controlled_batch": controlled_metrics,
        "cross_batch_exact_self": main_summary,
        "loss_values": loss_values,
        "parameter_groups": {
            name: {
                "parameters": int(plan["total_parameters"]),
                "gradient_samples": int(plan["sample_count"]),
                "sample_fraction": (
                    int(plan["sample_count"]) / max(1, int(plan["total_parameters"]))
                ),
            }
            for name, plan in plans.items()
        },
        "gradient_statistics": gradient_statistics,
        "gradient_cosine": matrices,
        "most_conflicting_pairs": _most_conflicting_pairs(matrices),
        "notes": {
            "sampling": "uniform coordinate sample per parameter group",
            "all_trainable": "dimension-weighted estimate from group samples",
            "cross_batch_exact_self_flow": (
                "fixed human exact-self batch; all other losses share one "
                "controlled native-teacher graph"
            ),
            "performance_stage": (
                stage["name"] if stage is not None else "legacy"
            ),
        },
    }
    output_path = output / str(
        diagnostic.get(
            "output_file", f"gradient_diagnostics_step_{checkpoint_step}.json"
        )
    )
    write_json(output_path, result)
    result["output_path"] = str(output_path)
    return result
