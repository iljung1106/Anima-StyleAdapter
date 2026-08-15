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
from .style_tokenizer import _flow_metrics, insert_style_tokens
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
    first = tokenizer(references, first_mask).tokens
    second = tokenizer(references, second_mask).tokens
    return (
        1.0
        - F.cosine_similarity(first.float(), second.float(), dim=-1)
    ).mean()


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

    direct_active = step <= int(training.get("direct_auxiliary_end_step", 2000))
    normalized_weight = (
        float(training.get("normalized_residual_weight", 0.0))
        if direct_active else 0.0
    )
    floor_weight = (
        float(training.get("aligned_floor_weight", 0.0))
        if direct_active else 0.0
    )
    needs_base = measure_base or normalized_weight > 0 or floor_weight > 0
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
        normalized, floor, direct_metrics = _aligned_velocity_losses(
            prediction,
            base_prediction,
            target,
            include_target,
            coefficient_floor=_coefficient_floor(step, training),
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

    metrics = {
        "loss": total.detach(),
        "flow_loss": flow_loss.detach(),
        "normalized_residual_loss": normalized.detach(),
        "normalized_residual_weight": flow_loss.new_tensor(normalized_weight),
        "aligned_floor_loss": floor.detach(),
        "aligned_floor_weight": flow_loss.new_tensor(floor_weight),
        **{key: value.detach() for key, value in direct_metrics.items()},
        "artist_contrastive_loss": contrastive.detach(),
        "artist_contrastive_weight": flow_loss.new_tensor(contrastive_weight),
        "artist_positive_similarity": positive.detach(),
        "artist_negative_similarity": negative.detach(),
        "slot_diversity_loss": diversity.detach(),
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
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=base_name + wandb_suffix,
            id=base_id + wandb_suffix,
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
