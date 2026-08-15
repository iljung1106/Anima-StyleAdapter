from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .io import write_json
from .query_style_tokenizer import (
    QueryStyleTokenizerV2,
    _artist_contrastive_loss,
    _linear_weight,
    _query_loader_config,
    _reference_inputs,
    _select_sample_episodes,
    _slot_diversity_loss,
)
from .style_tokenizer import (
    _assert_resampler_cache_identity,
    _flow_metrics,
    _mean_metrics,
    _sample_tokenizer,
    insert_style_tokens,
)
from .style_transfer import (
    ProductionStyleLoader,
    _learning_rate_multiplier,
    _optimize_frozen_anima,
    _replace_reference_with_target,
    _resolve_anima_model,
    _sample_flow_timesteps,
    _self_reference_curriculum_state,
)


class _SamplingTokenView(nn.Module):
    """Expose only final tokens to the established token-injection sampler."""

    def __init__(self, tokenizer: QueryStyleTokenizerV2) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self, references: torch.Tensor, reference_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.tokenizer(references, reference_mask).tokens


def _aligned_velocity_losses(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    target_included: torch.Tensor,
    *,
    coefficient_floor: float,
    huber_beta: float,
    scale_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Directly supervise a nonzero velocity change in the target direction.

    A raw norm floor can be satisfied by an orthogonal image perturbation.  The
    projection coefficient below is one only when the injected-token delta
    equals the frozen Anima residual and zero for an orthogonal or null delta.
    Losses are averaged over the complete batch after masking, so their total
    contribution naturally decays with the curriculum's target inclusion rate.
    """

    if target_included.shape != (prediction.shape[0],):
        raise ValueError("target_included must contain one value per batch row")
    dimensions = tuple(range(1, prediction.ndim))
    delta = prediction - base_prediction
    desired = target - base_prediction
    desired_power = desired.square().mean(dim=dimensions).clamp_min(
        float(scale_floor) ** 2
    )
    desired_rms = desired_power.sqrt()
    broadcast = desired_rms.reshape(-1, *([1] * (desired.ndim - 1)))
    normalized_per_sample = F.smooth_l1_loss(
        delta / broadcast,
        desired.detach() / broadcast,
        beta=float(huber_beta),
        reduction="none",
    ).mean(dim=dimensions)
    coefficient = (delta * desired.detach()).mean(dim=dimensions) / desired_power
    floor_per_sample = F.relu(float(coefficient_floor) - coefficient).square()
    mask = target_included.to(dtype=normalized_per_sample.dtype)
    normalized_loss = (normalized_per_sample * mask).mean()
    floor_loss = (floor_per_sample * mask).mean()
    included = mask.sum().clamp_min(1.0)
    return normalized_loss, floor_loss, {
        "aligned_coefficient": (coefficient.detach() * mask).sum() / included,
        "aligned_floor_violation": (
            (coefficient.detach() < float(coefficient_floor)).to(mask) * mask
        ).sum() / included,
        "target_auxiliary_fraction": mask.mean(),
    }


def _coefficient_floor(step: int, training_cfg: dict[str, Any]) -> float:
    start = float(training_cfg.get("aligned_floor_start", 0.05))
    end = float(training_cfg.get("aligned_floor_end", 0.25))
    ramp_steps = max(1, int(training_cfg.get("aligned_floor_ramp_steps", 500)))
    progress = min(1.0, max(0.0, step / ramp_steps))
    return start + progress * (end - start)


def _reference_batch(
    batch: dict[str, Any],
    device: str,
    mode: str,
    step: int,
    training_cfg: dict[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    target_tokens = batch["cached_target_tokens"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    curriculum = _self_reference_curriculum_state(
        step, dict(training_cfg.get("curriculum", {}))
    )
    if mode == "self":
        return (
            target_tokens[:, None],
            torch.ones(
                target_tokens.shape[0], 1, dtype=torch.bool, device=device
            ),
            torch.ones(target_tokens.shape[0], dtype=torch.bool, device=device),
            curriculum,
        )
    if mode in {"heldout", "wrong_artist"}:
        references, mask = _reference_inputs(batch, device, mode)
        return (
            references,
            mask,
            torch.zeros(target_tokens.shape[0], dtype=torch.bool, device=device),
            curriculum,
        )
    if mode != "curriculum":
        raise ValueError(f"Unknown pure-token reference mode: {mode}")
    references, mask = _reference_inputs(batch, device, "heldout")
    include_target = torch.rand(
        target_tokens.shape[0], device=device, generator=generator
    ) < float(curriculum["target_probability"])
    references, mask = _replace_reference_with_target(
        references, mask, target_tokens, include_target
    )
    return references, mask, include_target, curriculum


def _forward_pure_token_flow(
    anima: nn.Module,
    tokenizer: QueryStyleTokenizerV2,
    batch: dict[str, Any],
    device: str,
    training_cfg: dict[str, Any],
    *,
    generator: torch.Generator,
    step: int,
    mode: str,
    train_auxiliaries: bool,
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
    references, reference_mask, include_target, curriculum = _reference_batch(
        batch, device, mode, step, training_cfg, generator
    )
    heldout, heldout_mask = _reference_inputs(batch, device, "heldout")

    noise = torch.randn(
        latents.shape, device=device, dtype=latents.dtype, generator=generator
    )
    timesteps = _sample_flow_timesteps(
        latents.shape[0], device, training_cfg, generator
    )
    sigma = timesteps[:, None, None, None].to(latents.dtype)
    noisy = (1 - sigma) * latents + sigma * noise
    target = (noise - latents).float()
    padding_mask = torch.zeros(
        latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
        device=device, dtype=latents.dtype,
    )
    reconstruction_weight = (
        _linear_weight(
            step,
            start=float(training_cfg.get("reconstruction_weight", 0.02)),
            end=float(training_cfg.get("reconstruction_final_weight", 0.005)),
            end_step=int(training_cfg.get("reconstruction_decay_steps", 8000)),
        )
        if train_auxiliaries else 0.0
    )
    def autocast_context():
        return torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        )

    with torch.no_grad(), autocast_context():
        base_prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype),
            context=conditioning, padding_mask=padding_mask,
            target_input_ids=None,
        ).squeeze(2).float()
    with autocast_context():
        output = tokenizer(
            references, reference_mask, reconstruct=reconstruction_weight > 0
        )
        styled_conditioning = insert_style_tokens(
            conditioning, conditioning_lengths, output.tokens
        )
        prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype),
            context=styled_conditioning, padding_mask=padding_mask,
            target_input_ids=None,
        ).squeeze(2).float()

    flow_loss = F.mse_loss(prediction, target)
    total_loss = flow_loss
    normalized_loss = flow_loss.new_zeros(())
    aligned_floor_loss = flow_loss.new_zeros(())
    coefficient_floor = _coefficient_floor(step, training_cfg)
    normalized_weight = (
        float(training_cfg.get("normalized_residual_weight", 0.0))
        if train_auxiliaries else 0.0
    )
    floor_weight = (
        float(training_cfg.get("aligned_floor_weight", 0.0))
        if train_auxiliaries else 0.0
    )
    # Validation reports alignment for every row even when its reference does
    # not contain the target. Training auxiliaries remain strictly masked by
    # the curriculum's actual target-inclusion decisions.
    aligned_mask = (
        include_target
        if train_auxiliaries
        else torch.ones_like(include_target)
    )
    normalized_loss, aligned_floor_loss, direct_metrics = (
        _aligned_velocity_losses(
            prediction,
            base_prediction,
            target,
            aligned_mask,
            coefficient_floor=coefficient_floor,
            huber_beta=float(
                training_cfg.get("normalized_residual_huber_beta", 0.1)
            ),
            scale_floor=float(
                training_cfg.get("normalized_residual_scale_floor", 1e-4)
            ),
        )
    )
    if normalized_weight > 0 or floor_weight > 0:
        total_loss = (
            total_loss
            + normalized_weight * normalized_loss
            + floor_weight * aligned_floor_loss
        )

    reconstruction_loss = flow_loss.new_zeros(())
    if reconstruction_weight > 0:
        assert output.reconstruction is not None
        assert output.reconstruction_target is not None
        reconstruction_loss = F.smooth_l1_loss(
            output.reconstruction.float(),
            output.reconstruction_target.float(),
            beta=float(training_cfg.get("reconstruction_huber_beta", 0.1)),
        )
        total_loss = total_loss + reconstruction_weight * reconstruction_loss

    diversity_weight = (
        float(training_cfg.get("slot_diversity_weight", 0.0))
        if train_auxiliaries else 0.0
    )
    diversity_loss = _slot_diversity_loss(output.tokens)
    total_loss = total_loss + diversity_weight * diversity_loss

    contrastive_loss = flow_loss.new_zeros(())
    positive_similarity = flow_loss.new_zeros(())
    negative_similarity = flow_loss.new_zeros(())
    contrastive_weight = 0.0
    contrastive_every = max(
        1, int(training_cfg.get("artist_contrastive_every", 2))
    )
    if (
        train_auxiliaries
        and float(training_cfg.get("artist_contrastive_weight", 0.0)) > 0
        and step % contrastive_every == 0
        and output.tokens.shape[0] > 1
    ):
        with autocast_context():
            target_output = (
                output
                if mode == "self" or (
                    mode == "curriculum" and bool(curriculum["target_only"])
                )
                else tokenizer(
                    batch["cached_target_tokens"].to(
                        device, dtype=torch.bfloat16, non_blocking=True
                    )[:, None],
                    torch.ones(
                        output.tokens.shape[0], 1,
                        dtype=torch.bool, device=device,
                    ),
                )
            )
            heldout_output = (
                output
                if mode == "heldout"
                else tokenizer(heldout, heldout_mask)
            )
        style_ids = [str(item.style_id) for item in batch["episodes"]]
        contrastive_loss, contrastive_metrics = _artist_contrastive_loss(
            target_output.tokens,
            heldout_output.tokens,
            style_ids,
            float(training_cfg.get("artist_contrastive_temperature", 0.1)),
        )
        positive_similarity = contrastive_metrics["artist_positive_similarity"]
        negative_similarity = contrastive_metrics["artist_negative_similarity"]
        contrastive_weight = float(training_cfg["artist_contrastive_weight"])
        total_loss = total_loss + contrastive_weight * contrastive_loss

    metrics = {
        "loss": total_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "normalized_residual_loss": normalized_loss.detach(),
        "normalized_residual_weight": flow_loss.new_tensor(normalized_weight),
        "aligned_floor_loss": aligned_floor_loss.detach(),
        "aligned_floor_weight": flow_loss.new_tensor(floor_weight),
        "aligned_coefficient_floor": flow_loss.new_tensor(coefficient_floor),
        **{key: value.detach() for key, value in direct_metrics.items()},
        "reconstruction_loss": reconstruction_loss.detach(),
        "reconstruction_weight": flow_loss.new_tensor(reconstruction_weight),
        "artist_contrastive_loss": contrastive_loss.detach(),
        "artist_contrastive_weight": flow_loss.new_tensor(contrastive_weight),
        "artist_positive_similarity": positive_similarity.detach(),
        "artist_negative_similarity": negative_similarity.detach(),
        "slot_diversity_loss": diversity_loss.detach(),
        "style_token_rms": output.tokens.detach().float().square().mean().sqrt(),
        "references": reference_mask.sum(dim=1).float().mean(),
        "target_inclusion": include_target.float().mean(),
        "target_probability": flow_loss.new_tensor(
            float(curriculum["target_probability"])
        ),
        "timestep_mean": timesteps.detach().mean(),
        **{
            key: value.detach()
            for key, value in _flow_metrics(
                prediction.detach(), base_prediction, target
            ).items()
        },
    }
    return total_loss, metrics


@torch.no_grad()
def _evaluate(
    anima: nn.Module,
    tokenizer: QueryStyleTokenizerV2,
    loader: ProductionStyleLoader,
    device: str,
    training_cfg: dict[str, Any],
    *,
    step: int,
    batches: int,
    seed: int,
    mode: str,
) -> dict[str, float]:
    tokenizer.eval()
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for index in range(batches):
        _, metrics = _forward_pure_token_flow(
            anima,
            tokenizer,
            loader.load_step(index),
            device,
            training_cfg,
            generator=torch.Generator(device=device).manual_seed(seed + index * 97),
            step=step,
            mode=mode,
            train_auxiliaries=False,
        )
        rows.append({key: float(value) for key, value in metrics.items()})
    result = _mean_metrics(rows)
    result["elapsed_s"] = time.perf_counter() - started
    tokenizer.train()
    return result


def _save_training_state(
    path: Path,
    *,
    step: int,
    tokenizer: QueryStyleTokenizerV2,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    cache_summary: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "tokenizer": {
                key: value.detach().cpu()
                for key, value in tokenizer.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "resampler_cache": cache_summary,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def _sample_panels(
    anima: nn.Module,
    tokenizer: QueryStyleTokenizerV2,
    requests: list[tuple[str, ProductionStyleLoader, int, str]],
    config: dict[str, Any],
    destination: Path,
    output: Path,
    device: str,
    step: int,
    *,
    config_section: str,
) -> list[Path]:
    chunk_size = max(
        1, int(config[config_section].get("sampling", {}).get("batch_size", 4))
    )
    view = _SamplingTokenView(tokenizer)
    sheets: list[Path] = []
    vae = None
    for offset in range(0, len(requests), chunk_size):
        chunk_sheets, vae = _sample_tokenizer(
            anima,
            view,
            requests[offset : offset + chunk_size],
            config,
            destination,
            output,
            device,
            step,
            vae,
            config_section=config_section,
        )
        sheets.extend(chunk_sheets)
    if vae is not None:
        del vae
    del view
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    tokenizer.train()
    return sheets


def train_pure_token_style_tokenizer(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_section: str = "pure_token_style_tokenizer_v2",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[config_section])
    training_cfg = dict(cfg["training"])
    steps = int(steps_override or training_cfg["steps"])
    device = str(training_cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260822))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(training_cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    accumulation = max(
        1, int(training_cfg.get("gradient_accumulation_steps", 1))
    )
    train_loader_cfg = _query_loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    train_loader_cfg["gradient_accumulation_steps"] = accumulation
    validation_loader_cfg = _query_loader_config(
        config, cfg, split=str(cfg.get("validation_split", "validation"))
    )
    train_loader = ProductionStyleLoader(destination, train_loader_cfg)
    validation_loader = ProductionStyleLoader(destination, validation_loader_cfg)
    resampler_checkpoint = str(config["style_transfer"]["resampler"]["checkpoint"])
    cache_summary = _assert_resampler_cache_identity(
        destination, train_loader_cfg, resampler_checkpoint
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training_cfg.get("fuse_attention_projections", True)
        ),
    )
    tokenizer = QueryStyleTokenizerV2(**dict(cfg["model"])).to(device)
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    resume_state = None
    if bool(training_cfg.get("resume", True)) and state_path.exists():
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(resume_state["tokenizer"], strict=True)
        start_step = int(resume_state["step"])

    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
        eps=float(training_cfg.get("adam_eps", 1e-8)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        fused=bool(
            training_cfg.get("fused_adamw", True) and device.startswith("cuda")
        ),
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])

    trainable = sum(parameter.numel() for parameter in tokenizer.parameters())
    print(
        "pure-token StyleTokenizer "
        f"trainable={trainable / 1e6:.2f}M output=32x1024 "
        "anima=native-single-QKVO",
        flush=True,
    )
    wandb_run = None
    wandb_cfg = dict(training_cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "pure-token-style-tokenizer-v2")),
            id=str(wandb_cfg.get("id", "pure-token-style-tokenizer-v2")),
            resume="allow" if start_step else "never",
            config={
                config_section: cfg,
                "trainable_parameters": trainable,
                "injection": "native_text_context_null_positions",
            },
        )

    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    warmup = int(training_cfg.get("warmup_steps", 500))
    minimum_ratio = float(training_cfg.get("minimum_lr_ratio", 0.1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    validation_every = int(training_cfg.get("validation_every", 250))
    validation_batches = int(training_cfg.get("validation_batches", 8))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 500))
    sample_every = int(training_cfg.get("sample_every", 500))
    prefetched = train_loader.prefetch(
        start_step * accumulation,
        max(0, steps - start_step) * accumulation,
        workers=int(training_cfg.get("prefetch_workers", 2)),
        depth=int(training_cfg.get("prefetch_batches", 4)),
    )
    history_path = output / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.exists() else []
    )
    sample_requests = [
        (
            f"train-{index}", train_loader, episode_index, "heldout"
        )
        for index, episode_index in enumerate(
            _select_sample_episodes(train_loader, 4)
        )
    ] + [
        (
            f"validation-{index}", validation_loader, episode_index, "heldout"
        )
        for index, episode_index in enumerate(
            _select_sample_episodes(validation_loader, 4)
        )
    ]
    completed_step = start_step
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            multiplier = _learning_rate_multiplier(
                step, steps, warmup, minimum_ratio
            )
            optimizer.param_groups[0]["lr"] = base_lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            micro_metrics: list[dict[str, torch.Tensor]] = []
            for micro in range(accumulation):
                batch = next(prefetched)
                loss, metrics = _forward_pure_token_flow(
                    anima,
                    tokenizer,
                    batch,
                    device,
                    training_cfg,
                    generator=torch.Generator(device=device).manual_seed(
                        seed + step * 100_003 + micro
                    ),
                    step=step,
                    mode="curriculum",
                    train_auxiliaries=True,
                )
                (loss / accumulation).backward()
                micro_metrics.append(metrics)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                tokenizer.parameters(), max_grad_norm
            )
            optimizer.step()
            if step == 1 or step % log_every == 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                averaged = {
                    key: float(
                        torch.stack([row[key] for row in micro_metrics]).mean()
                    )
                    for key in micro_metrics[0]
                    if all(key in row for row in micro_metrics)
                }
                elapsed = time.perf_counter() - step_started
                averaged.update({
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "step_s": elapsed,
                    "images_per_s": (
                        train_loader.batch_size * accumulation / max(elapsed, 1e-6)
                    ),
                })
                print(
                    f"pure-token step={step}/{steps} "
                    f"loss={averaged['loss']:.5f} "
                    f"flow={averaged['flow_loss']:.5f} "
                    f"aligned={averaged['aligned_coefficient']:.4f} "
                    f"delta/desired={averaged['style_flow_delta_to_desired_ratio']:.4f} "
                    f"grad={averaged['grad_norm']:.4f} "
                    f"step_s={averaged['step_s']:.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in averaged.items()},
                        step=step,
                    )

            if step % validation_every == 0 or step == steps:
                validation_self = _evaluate(
                    anima, tokenizer, validation_loader, device, training_cfg,
                    step=step, batches=validation_batches,
                    seed=seed ^ 0xBEEF, mode="self",
                )
                validation_heldout = _evaluate(
                    anima, tokenizer, validation_loader, device, training_cfg,
                    step=step, batches=validation_batches,
                    seed=seed ^ 0xC0FFEE, mode="heldout",
                )
                validation_wrong = _evaluate(
                    anima, tokenizer, validation_loader, device, training_cfg,
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
                history.append(row)
                write_json(history_path, history)
                print(f"pure-token validation step={step} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({
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
                    }, step=step)

            if step % checkpoint_every == 0 or step == steps:
                checkpoint = checkpoint_dir / f"step-{step:07d}.pt"
                _save_training_state(
                    checkpoint,
                    step=step,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                    cache_summary=cache_summary,
                )
                _save_training_state(
                    state_path,
                    step=step,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                    cache_summary=cache_summary,
                )
            if sample_every > 0 and (step % sample_every == 0 or step == steps):
                sheets = _sample_panels(
                    anima,
                    tokenizer,
                    sample_requests,
                    config,
                    destination,
                    output,
                    device,
                    step,
                    config_section=config_section,
                )
                print(
                    f"pure-token samples step={step} panels={len(sheets)}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log({
                        "samples/panel": [
                            wandb.Image(str(path), caption=path.stem)
                            for path in sheets
                        ]
                    }, step=step)
            completed_step = step
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    result = {
        "steps": completed_step,
        "requested_steps": steps,
        "start_step": start_step,
        "trainable_parameters": trainable,
        "architecture": "32x1024 native text-context token injection",
        "resampler_cache": cache_summary,
        "elapsed_s": time.perf_counter() - started,
        "final_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", result)
    return result


def smoke_test_pure_token_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    cfg = smoke["pure_token_style_tokenizer_v2"]
    cfg["output_directory"] = "pure_token_style_tokenizer_v2_smoke"
    cfg["loader"]["batch_size"] = 1
    cfg["training"].update({
        "steps": 2,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 1,
        "validation_every": 2,
        "validation_batches": 1,
        "checkpoint_every": 2,
        "sample_every": 0,
        "resume": False,
        "wandb": {"enabled": False},
    })
    return train_pure_token_style_tokenizer(
        smoke, destination, steps_override=2
    )
