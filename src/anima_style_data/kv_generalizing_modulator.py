"""Train a reference-conditioned native K/V modulator with artist holdout.

Unlike the fixed-anchor capacity experiment, this runner sees stochastic
human reference views and is evaluated on artists whose LoRA functions never
appear in optimization.  It supervises only the represented native K/V
activation.  Two/three-artist convex mixtures are an auxiliary linearity
regularizer, never a replacement for real single-artist examples.
"""

from __future__ import annotations

import gc
import copy
import random
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import write_json
from .kv_activation_modulation import (
    NativeKVFactorModulator,
    apply_kv_factors,
    canonicalize_lora_factor_bank,
    kv_activation_objective,
    load_kv_lora_factor_bank,
)
from .lora_oracle_bootstrap import (
    _materialize_reader_code_bank,
    _oracle_detail_config,
)


def _save_state(
    path: Path,
    *,
    step: int,
    model: NativeKVFactorModulator,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    train_indices: list[int],
    validation_indices: list[int],
    best_heldout_centered_cosine: float,
    best_step: int,
    best_validation: dict[str, float] | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "architecture": {
            "style_dim": int(model.style_norm.normalized_shape[0]),
            "blocks": int(model.blocks),
            "rank": int(model.rank),
            "context_dim": int(model.context_dim),
            "output_dim": int(model.output_dim),
        },
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "best_heldout_centered_cosine": float(best_heldout_centered_cosine),
        "best_step": int(best_step),
        "best_validation": best_validation,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


def _view_probabilities(
    reference_counts: torch.Tensor,
    weights: dict[str, Any],
) -> torch.Tensor:
    values = torch.tensor([
        float(weights.get(str(int(count)), 1.0)) for count in reference_counts
    ])
    if float(values.sum()) <= 0:
        raise ValueError("reference_count_weights sum to zero")
    return values / values.sum()


def build_mixed_activation_batch(
    sampled_contexts: torch.Tensor,
    predicted_down: torch.Tensor,
    predicted_up: torch.Tensor,
    group_down: torch.Tensor,
    group_up: torch.Tensor,
    mixture_weights: torch.Tensor,
    output_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply student factors and the exact convex teacher function mixture."""

    batch, group_size = group_down.shape[:2]
    context_count, tokens, context_dim = sampled_contexts.shape
    context_bc = sampled_contexts[None].expand(
        batch, -1, -1, -1
    ).reshape(batch * context_count, tokens, context_dim)
    student = apply_kv_factors(
        context_bc,
        predicted_down[:, None].expand(
            -1, context_count, -1, -1, -1
        ).reshape(batch * context_count, *predicted_down.shape[1:]),
        predicted_up.index_select(2, output_indices)[:, None].expand(
            -1, context_count, -1, -1, -1
        ).reshape(
            batch * context_count,
            2,
            len(output_indices),
            predicted_up.shape[-1],
        ),
    )
    context_bcg = context_bc[:, None].expand(
        -1, group_size, -1, -1
    ).reshape(batch * context_count * group_size, tokens, context_dim)
    selected_group_up = group_up.index_select(3, output_indices)
    teacher = apply_kv_factors(
        context_bcg,
        group_down[:, None].expand(
            -1, context_count, -1, -1, -1, -1
        ).reshape(
            batch * context_count * group_size,
            *group_down.shape[2:],
        ),
        selected_group_up[:, None].expand(
            -1, context_count, -1, -1, -1, -1
        ).reshape(
            batch * context_count * group_size,
            *selected_group_up.shape[2:],
        ),
    ).reshape(
        batch, context_count, group_size,
        2, tokens, len(output_indices),
    )
    target = (
        teacher * mixture_weights[:, None, :, None, None, None]
    ).sum(dim=2).reshape_as(student)
    return student, target


@torch.no_grad()
def _validate(
    model: NativeKVFactorModulator,
    code_bank: torch.Tensor,
    reference_counts: torch.Tensor,
    artist_indices: torch.Tensor,
    contexts: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    *,
    views: int,
    tokens: int,
    output_channels: int,
    direction_weight: float,
    magnitude_weight: float,
) -> dict[str, float]:
    model.eval()
    rows: dict[str, list[float]] = defaultdict(list)
    view_indices = torch.linspace(
        0, code_bank.shape[1] - 1, min(views, code_bank.shape[1]),
        device=code_bank.device,
    ).round().long().unique()
    token_indices = torch.linspace(
        0, contexts.shape[1] - 1, min(tokens, contexts.shape[1]),
        device=contexts.device,
    ).round().long().unique()
    output_indices = torch.linspace(
        0, teacher_up.shape[-2] - 1,
        min(output_channels, teacher_up.shape[-2]),
        device=teacher_up.device,
    ).round().long().unique()
    sampled_contexts = contexts[:, token_indices]
    for view in view_indices.tolist():
        codes = code_bank[artist_indices, view]
        for block in range(model.blocks):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_down, predicted_up = model(codes, block)
            for context in sampled_contexts:
                expanded = context.expand(codes.shape[0], -1, -1)
                student = apply_kv_factors(
                    expanded,
                    predicted_down,
                    predicted_up[:, :, output_indices],
                )
                teacher = apply_kv_factors(
                    expanded,
                    teacher_down[artist_indices, block],
                    teacher_up[artist_indices, block][:, :, output_indices],
                )
                _, raw_metrics = kv_activation_objective(
                    student,
                    teacher,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
                student_centered = student.float() - student.float().mean(
                    dim=0, keepdim=True
                )
                teacher_centered = teacher.float() - teacher.float().mean(
                    dim=0, keepdim=True
                )
                _, centered_metrics = kv_activation_objective(
                    student_centered,
                    teacher_centered,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
                for key, value in raw_metrics.items():
                    rows[f"raw_{key}"].append(float(value))
                for key, value in centered_metrics.items():
                    rows[f"centered_{key}"].append(float(value))
                student_common = student.float().mean(dim=0)
                teacher_common = teacher.float().mean(dim=0)
                rows["student_common_to_centered_ratio"].append(float(
                    student_common.square().mean().sqrt()
                    / student_centered.square().mean().sqrt().clamp_min(1e-8)
                ))
                rows["teacher_common_to_centered_ratio"].append(float(
                    teacher_common.square().mean().sqrt()
                    / teacher_centered.square().mean().sqrt().clamp_min(1e-8)
                ))
                rows["student_raw_rms"].append(float(
                    student.float().square().mean().sqrt()
                ))
                rows["teacher_raw_rms"].append(float(
                    teacher.float().square().mean().sqrt()
                ))
                rows["student_centered_rms"].append(float(
                    student_centered.square().mean().sqrt()
                ))
                rows["teacher_centered_rms"].append(float(
                    teacher_centered.square().mean().sqrt()
                ))
    model.train()
    result = {key: sum(values) / len(values) for key, values in rows.items()}
    result["mean_references"] = float(reference_counts[view_indices].float().mean())
    return result


def train_generalizing_kv_activation_modulator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    cfg = dict(config["kv_activation_generalizing_modulator"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 8000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
    )
    teacher_down, teacher_up = canonicalize_lora_factor_bank(
        teacher_down.to(device),
        teacher_up.to(device),
        chunk_size=int(training.get("canonicalization_chunk_size", 64)),
    )
    teacher_down = teacher_down.to(dtype=torch.bfloat16)
    teacher_up = teacher_up.to(dtype=torch.bfloat16)

    reader_state = torch.load(
        destination / str(cfg["reader_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
    detail_cfg = _oracle_detail_config(config, oracle_cfg)
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(
        device=device, dtype=torch.bfloat16
    )
    reader.load_state_dict(reader_state["reader"], strict=True)
    reader.requires_grad_(False).eval()
    del reader_state
    gc.collect()

    reference_images = int(training.get("materialized_reference_images", 8))
    loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=artist_ids,
        batch_size=int(training.get("materialization_artist_chunk", 16)),
        references=reference_images,
        seed=seed ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    code_bank, reference_counts = _materialize_reader_code_bank(
        reader,
        loader,
        artist_ids,
        reference_images=reference_images,
        seed=seed ^ 0x11111111,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    del reader
    torch.cuda.empty_cache()

    validation_count = int(training.get("validation_artists", 32))
    validation_list = [
        int(value) for value in torch.linspace(
            0, len(artist_ids) - 1, validation_count
        ).round().long().unique()
    ]
    validation_set = set(validation_list)
    train_list = [
        index for index in range(len(artist_ids)) if index not in validation_set
    ]
    train_indices = torch.tensor(train_list, device=device)
    validation_indices = torch.tensor(validation_list, device=device)

    context_bank = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    heldout_context_count = int(training.get("heldout_contexts", 32))
    train_contexts = context_bank[:-heldout_context_count].to(
        device=device, dtype=torch.bfloat16
    )
    heldout_contexts = context_bank[-heldout_context_count:]
    validation_context_count = int(training.get("validation_contexts", 3))
    validation_context_indices = torch.linspace(
        0, heldout_context_count - 1, validation_context_count
    ).round().long().unique()
    validation_contexts = heldout_contexts[validation_context_indices].to(
        device=device, dtype=torch.bfloat16
    )

    model = NativeKVFactorModulator(
        style_dim=int(code_bank.shape[-1]),
        blocks=int(teacher_down.shape[1]),
        rank=int(teacher_down.shape[-2]),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        **dict(cfg["model"]),
    ).to(device)
    model.set_factor_scales(
        teacher_down[train_indices].float().square().mean(dim=-1).sqrt().mean(dim=0),
        teacher_up[train_indices].float().transpose(-1, -2).square()
        .mean(dim=-1).sqrt().mean(dim=0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )

    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    validation_history_path = output / "validation_history.json"
    start_step = 0
    best_heldout_centered_cosine = float("-inf")
    best_step = 0
    best_validation: dict[str, float] | None = None
    validation_history: list[dict[str, Any]] = []
    if validation_history_path.exists():
        validation_history = json.loads(
            validation_history_path.read_text(encoding="utf-8")
        )
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_heldout_centered_cosine = float(
            state.get("best_heldout_centered_cosine", float("-inf"))
        )
        best_step = int(state.get("best_step", 0))
        best_validation = state.get("best_validation")
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch_artists = int(training.get("batch_artists", 16))
    mixture_probability = float(training.get("mixture_probability", 0.25))
    mixture_sizes = tuple(int(value) for value in training.get("mixture_sizes", [2, 3]))
    contexts_per_step = int(training.get("contexts_per_step", 2))
    tokens_per_step = int(training.get("tokens_per_step", 64))
    channels_per_step = int(training.get("output_channels_per_step", 256))
    direction_weight = float(training.get("direction_weight", 1.0))
    magnitude_weight = float(training.get("magnitude_weight", 0.2))
    raw_function_weight = float(training.get("raw_function_weight", 0.25))
    centered_function_weight = float(
        training.get("centered_function_weight", 1.0)
    )
    base_lr = float(training.get("learning_rate", 3e-4))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 2.0))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    view_probabilities = _view_probabilities(
        reference_counts, dict(training.get("reference_count_weights", {}))
    )

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-generalizing-modulator")),
            id=str(wandb_cfg.get("id", "kv-generalizing-modulator")),
            resume="allow",
            config={"kv_activation_generalizing_modulator": cfg},
        )

    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        model.train()
        for step in range(start_step + 1, steps + 1):
            learning_rate = base_lr * min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = learning_rate
            generator = torch.Generator().manual_seed(seed + step * 1_000_003)
            is_mixture = float(torch.rand((), generator=generator)) < mixture_probability
            group_size = (
                mixture_sizes[int(torch.randint(len(mixture_sizes), (1,), generator=generator))]
                if is_mixture else 1
            )
            needed = batch_artists * group_size
            if needed > len(train_list):
                raise ValueError("batch_artists * mixture_size exceeds train artists")
            local = torch.randperm(len(train_list), generator=generator)[:needed]
            artist_groups = train_indices[local.to(device)].reshape(
                batch_artists, group_size
            )
            view_indices = torch.multinomial(
                view_probabilities,
                needed,
                replacement=True,
                generator=generator,
            ).to(device).reshape(batch_artists, group_size)
            if group_size == 1:
                mixture_weights = torch.ones(batch_artists, 1, device=device)
            else:
                mixture_weights = -torch.rand(
                    batch_artists, group_size, generator=generator
                ).clamp_min(1e-6).log().to(device)
                mixture_weights /= mixture_weights.sum(dim=-1, keepdim=True)
            visual_groups = code_bank[artist_groups, view_indices]
            visual = torch.einsum(
                "bg,bgsd->bsd", mixture_weights.to(visual_groups), visual_groups
            )

            block = (step - 1) % model.blocks
            context_indices = torch.randperm(
                train_contexts.shape[0], generator=generator
            )[:contexts_per_step].to(device)
            token_indices = torch.randperm(
                train_contexts.shape[1], generator=generator
            )[:tokens_per_step].to(device)
            output_indices = torch.randperm(
                teacher_up.shape[-2], generator=generator
            )[:channels_per_step].to(device)
            sampled_contexts = train_contexts[context_indices][:, token_indices]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_down, predicted_up = model(visual, block)
                group_down = teacher_down[artist_groups, block]
                group_up = teacher_up[artist_groups, block]
                student, target = build_mixed_activation_batch(
                    sampled_contexts,
                    predicted_down,
                    predicted_up,
                    group_down,
                    group_up,
                    mixture_weights,
                    output_indices,
                )
            raw_loss, raw_metrics = kv_activation_objective(
                student,
                target,
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
            )
            student_by_artist = student.reshape(
                batch_artists, contexts_per_step, *student.shape[1:]
            ).float()
            target_by_artist = target.reshape(
                batch_artists, contexts_per_step, *target.shape[1:]
            ).float()
            student_centered = student_by_artist - student_by_artist.mean(
                dim=0, keepdim=True
            )
            target_centered = target_by_artist - target_by_artist.mean(
                dim=0, keepdim=True
            )
            centered_loss, centered_metrics = kv_activation_objective(
                student_centered.flatten(0, 1),
                target_centered.flatten(0, 1),
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
            )
            loss = (
                raw_function_weight * raw_loss
                + centered_function_weight * centered_loss
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            optimizer.step()
            row_metrics = {
                "loss": loss.detach(),
                "raw_loss": raw_loss.detach(),
                "centered_loss": centered_loss.detach(),
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **{
                    f"centered_{key}": value
                    for key, value in centered_metrics.items()
                },
                "student_common_to_centered_ratio": (
                    student_by_artist.mean(dim=0).square().mean().sqrt()
                    / student_centered.square().mean().sqrt().clamp_min(1e-8)
                ).detach(),
                "target_common_to_centered_ratio": (
                    target_by_artist.mean(dim=0).square().mean().sqrt()
                    / target_centered.square().mean().sqrt().clamp_min(1e-8)
                ).detach(),
                "student_raw_rms": student_by_artist.square().mean().sqrt().detach(),
                "target_raw_rms": target_by_artist.square().mean().sqrt().detach(),
                "student_centered_rms": student_centered.square().mean().sqrt().detach(),
                "target_centered_rms": target_centered.square().mean().sqrt().detach(),
                "grad_norm": grad_norm.detach(),
                "learning_rate": torch.tensor(learning_rate),
                "mixture": torch.tensor(float(is_mixture)),
                "mixture_size": torch.tensor(float(group_size)),
                "mean_references": reference_counts[view_indices].float().mean(),
                "block": torch.tensor(float(block)),
            }
            for key, value in row_metrics.items():
                running[key].append(float(value))
            if step % log_every == 0:
                row = {
                    key: sum(values) / len(values) for key, values in running.items()
                }
                print(f"K/V generalizing step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/generalizing_kv/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                common_kwargs = {
                    "code_bank": code_bank,
                    "reference_counts": reference_counts,
                    "contexts": validation_contexts,
                    "teacher_down": teacher_down,
                    "teacher_up": teacher_up,
                    "views": int(training.get("validation_views", 2)),
                    "tokens": int(training.get("validation_tokens", 64)),
                    "output_channels": int(training.get("validation_output_channels", 256)),
                    "direction_weight": direction_weight,
                    "magnitude_weight": magnitude_weight,
                }
                validation = _validate(
                    model, artist_indices=validation_indices, **common_kwargs
                )
                train_validation_indices = train_indices[torch.linspace(
                    0, len(train_list) - 1,
                    min(len(validation_list), len(train_list)),
                    device=device,
                ).round().long().unique()]
                train_validation = _validate(
                    model, artist_indices=train_validation_indices, **common_kwargs
                )
                print(
                    f"K/V generalizing validation step={step} "
                    f"heldout={validation} train={train_validation}",
                    flush=True,
                )
                validation_history.append({
                    "step": step,
                    "heldout": validation,
                    "train": train_validation,
                })
                write_json(validation_history_path, validation_history)
                heldout_centered_cosine = float(
                    validation["centered_cosine"]
                )
                if heldout_centered_cosine > best_heldout_centered_cosine:
                    best_heldout_centered_cosine = heldout_centered_cosine
                    best_step = step
                    best_validation = dict(validation)
                    _save_state(
                        output / "best.pt",
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        train_indices=train_list,
                        validation_indices=validation_list,
                        best_heldout_centered_cosine=best_heldout_centered_cosine,
                        best_step=best_step,
                        best_validation=best_validation,
                    )
                if wandb_run is not None:
                    wandb_run.log({
                        **{f"val/heldout_kv/{key}": value for key, value in validation.items()},
                        **{f"val/train_kv/{key}": value for key, value in train_validation.items()},
                    }, step=step)
            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_state(
                        path,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        train_indices=train_list,
                        validation_indices=validation_list,
                        best_heldout_centered_cosine=best_heldout_centered_cosine,
                        best_step=best_step,
                        best_validation=best_validation,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    acceptance_cfg = dict(training.get("acceptance", {}))
    accepted = False
    acceptance_checks: dict[str, bool] = {}
    if best_validation is not None:
        centered_cosine_minimum = float(
            acceptance_cfg.get("centered_cosine_minimum", 0.20)
        )
        centered_rms_minimum = float(
            acceptance_cfg.get("centered_rms_ratio_minimum", 0.60)
        )
        centered_rms_maximum = float(
            acceptance_cfg.get("centered_rms_ratio_maximum", 1.50)
        )
        common_ratio_multiplier = float(
            acceptance_cfg.get("common_ratio_maximum_teacher_multiplier", 1.50)
        )
        centered_rms_ratio = float(
            best_validation["centered_student_to_teacher_rms"]
        )
        acceptance_checks = {
            "centered_cosine": float(best_validation["centered_cosine"])
            >= centered_cosine_minimum,
            "centered_rms_ratio": centered_rms_minimum
            <= centered_rms_ratio <= centered_rms_maximum,
            "common_to_centered_ratio": float(
                best_validation["student_common_to_centered_ratio"]
            ) <= common_ratio_multiplier * max(
                float(best_validation["teacher_common_to_centered_ratio"]),
                1e-8,
            ),
        }
        accepted = all(acceptance_checks.values())
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "train_artists": len(train_list),
        "validation_artists": len(validation_list),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "reader_frozen": True,
        "best_heldout_centered_cosine": best_heldout_centered_cosine,
        "best_step": best_step,
        "best_validation": best_validation,
        "acceptance_checks": acceptance_checks,
        "accepted_for_visual_evaluation": accepted,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(state_path),
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_generalizing_kv_activation_modulator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    import copy

    effective = copy.deepcopy(config)
    cfg = effective["kv_activation_generalizing_modulator"]
    cfg["output_directory"] += "_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["wandb"]["enabled"] = False
    return train_generalizing_kv_activation_modulator(
        effective, destination, steps_override=2
    )


@torch.no_grad()
def sample_generalizing_kv_activation_modulator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render unseen artists from fresh 1/4-reference visual codes."""

    from .kv_activation_sampling import sample_kv_activation_modulator

    sample_cfg = dict(config["kv_activation_generalizing_sample"])
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = dict(checkpoint["config"])
    device = str(sample_cfg.get("device", "cuda"))
    artist_ids, _, _ = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
    )
    validation_indices = [int(value) for value in checkpoint["validation_indices"]]
    artist_count = min(
        int(sample_cfg.get("artists", 7)), len(validation_indices)
    )
    selected_positions = torch.linspace(
        0, len(validation_indices) - 1, artist_count
    ).round().long().unique()
    selected_indices = [
        validation_indices[int(position)] for position in selected_positions
    ]
    selected_ids = [artist_ids[index] for index in selected_indices]

    reader_state = torch.load(
        destination / str(cfg["reader_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
    detail_cfg = _oracle_detail_config(config, oracle_cfg)
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(
        device=device, dtype=torch.bfloat16
    )
    reader.load_state_dict(reader_state["reader"], strict=True)
    reader.requires_grad_(False).eval()
    del reader_state

    loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=selected_ids,
        batch_size=artist_count,
        references=max(int(value) for value in sample_cfg.get("reference_counts", [1, 4])),
        seed=int(sample_cfg.get("seed", 20260824)),
        token_lru_shards=int(sample_cfg.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for reference_count in sample_cfg.get("reference_counts", [1, 4]):
        loaded = loader.load_styles(
            selected_ids,
            references_per_style=int(reference_count),
            seed=int(sample_cfg.get("seed", 20260824)) + int(reference_count) * 1_000_003,
        )
        tokens = loaded["tokens"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        mask = torch.ones(tokens.shape[:2], device=device, dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            codes = reader(tokens, mask).tokens.cpu()
        style_codes = torch.zeros(
            len(artist_ids), *codes.shape[1:], dtype=codes.dtype
        )
        style_codes[selected_indices] = codes
        compatibility = {
            "model": checkpoint["model"],
            "style_codes": style_codes,
            "config": {
                "model": cfg["model"],
                "lora_directory": cfg["lora_directory"],
                "blocks": int(cfg.get("blocks", 28)),
            },
        }
        compatibility_path = output / f"reference-{int(reference_count)}-codes.pt"
        torch.save(compatibility, compatibility_path)
        effective = copy.deepcopy(config)
        effective_sample = dict(effective["kv_activation_modulator_sample"])
        effective_sample.update({
            "checkpoint": str(compatibility_path.relative_to(destination)),
            "output_directory": str(
                (output / f"reference-{int(reference_count)}").relative_to(destination)
            ),
            "device": device,
            "artist_indices": selected_indices,
            "predicted_strengths": [
                float(value) for value in sample_cfg.get(
                    "predicted_strengths", [0.5, 1.0, 1.5]
                )
            ],
            "batch_size": int(sample_cfg.get("batch_size", 4)),
            "panel_tile_width": int(sample_cfg.get("panel_tile_width", 416)),
        })
        effective["kv_activation_modulator_sample"] = effective_sample
        rendered = sample_kv_activation_modulator(effective, destination)
        rendered["reference_ids"] = [list(rows) for rows in loaded["ids"]]
        summaries[f"{int(reference_count)}ref"] = rendered
    del reader, checkpoint
    torch.cuda.empty_cache()
    summary = {
        "checkpoint": str(checkpoint_path),
        "artists": selected_ids,
        "artist_indices": selected_indices,
        "results": summaries,
    }
    write_json(output / "summary.json", summary)
    return summary
