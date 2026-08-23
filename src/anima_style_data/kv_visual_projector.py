from __future__ import annotations

import copy
import gc
import random
import time
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
    kv_factor_objective,
    load_kv_lora_factor_bank,
)
from .kv_activation_sampling import (
    _activation_bank_metrics,
    _predict_factor_bank,
    _style_code_metrics,
)
from .lora_oracle_bootstrap import (
    OracleVisualProjector,
    _materialize_reader_code_bank,
    _oracle_code_alignment_objective,
    _oracle_detail_config,
)


def _save_projector_state(
    path: Path,
    *,
    step: int,
    projector: OracleVisualProjector,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    train_indices: list[int],
    validation_indices: list[int],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "projector": {
            key: value.detach().cpu()
            for key, value in projector.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _validate_projector(
    projector: OracleVisualProjector,
    modulator: NativeKVFactorModulator,
    code_banks: torch.Tensor,
    anchors: torch.Tensor,
    validation_indices: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    contexts: torch.Tensor,
    *,
    views_per_domain: int,
) -> dict[str, float]:
    projector.eval()
    rows: dict[str, list[float]] = defaultdict(list)
    view_count = int(code_banks.shape[2])
    view_indices = torch.linspace(
        0, view_count - 1, min(views_per_domain, view_count),
        device=code_banks.device,
    ).round().long().unique()
    for domain in range(int(code_banks.shape[0])):
        for view in view_indices.tolist():
            visual = code_banks[domain, validation_indices, view]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                projected = projector(visual)
            metrics = _style_code_metrics(projected, anchors[validation_indices])
            down, up = _predict_factor_bank(modulator, projected)
            metrics.update(_activation_bank_metrics(
                contexts,
                teacher_down[validation_indices],
                teacher_up[validation_indices],
                down,
                up,
            ))
            for key, value in metrics.items():
                rows[key].append(float(value))
    projector.train()
    return {key: sum(values) / len(values) for key, values in rows.items()}


def train_kv_activation_visual_projector(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["kv_activation_visual_projector"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 1500))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    state = torch.load(
        destination / str(cfg["modulator_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    mod_cfg = dict(state["config"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(mod_cfg["lora_directory"]),
        blocks=int(mod_cfg.get("blocks", 28)),
    )
    teacher_down = teacher_down.to(device)
    teacher_up = teacher_up.to(device)
    teacher_down, teacher_up = canonicalize_lora_factor_bank(
        teacher_down,
        teacher_up,
        chunk_size=int(training.get("canonicalization_chunk_size", 64)),
    )
    teacher_down = teacher_down.to(dtype=torch.bfloat16)
    teacher_up = teacher_up.to(dtype=torch.bfloat16)
    anchors = state["style_codes"].to(device=device, dtype=torch.bfloat16)
    modulator = NativeKVFactorModulator(
        style_dim=int(anchors.shape[-1]),
        blocks=int(teacher_down.shape[1]),
        rank=int(teacher_down.shape[-2]),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        **dict(mod_cfg["model"]),
    )
    modulator.load_state_dict(state["model"], strict=True)
    modulator.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()

    oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
    reader_state = torch.load(
        destination / str(mod_cfg["style_code_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    detail_cfg = _oracle_detail_config(config, oracle_cfg)
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(
        device=device, dtype=torch.bfloat16
    )
    reader.load_state_dict(reader_state["reader"], strict=True)
    reader.requires_grad_(False).eval()
    del state, reader_state
    gc.collect()

    validation_count = int(training.get("validation_artists", 16))
    validation_indices_list = [
        int(value) for value in torch.linspace(
            0, len(artist_ids) - 1, validation_count
        ).round().long().unique()
    ]
    validation_set = set(validation_indices_list)
    train_indices_list = [
        index for index in range(len(artist_ids)) if index not in validation_set
    ]
    validation_indices = torch.tensor(
        validation_indices_list, device=device, dtype=torch.long
    )
    train_indices = torch.tensor(train_indices_list, device=device, dtype=torch.long)

    reference_images = int(training.get("materialized_reference_images", 8))
    loader_kwargs = {
        "split": "train",
        "style_ids": artist_ids,
        "batch_size": int(training.get("batch_artists", 16)),
        "references": reference_images,
        "token_lru_shards": int(training.get("token_lru_shards", 8)),
        "strict_style_ids": True,
    }
    human_loader = CachedTeacherReferenceLoader(
        destination / str(oracle_cfg["human_reference_cache"]),
        seed=seed ^ 0x48554D41,
        **loader_kwargs,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(oracle_cfg["synthetic_reference_cache"]),
        seed=seed ^ 0x53594E54,
        **loader_kwargs,
    )
    human_codes, reference_counts = _materialize_reader_code_bank(
        reader,
        human_loader,
        artist_ids,
        reference_images=reference_images,
        seed=seed ^ 0x11111111,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    synthetic_codes, synthetic_counts = _materialize_reader_code_bank(
        reader,
        synthetic_loader,
        artist_ids,
        reference_images=reference_images,
        seed=seed ^ 0x22222222,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    if not torch.equal(reference_counts, synthetic_counts):
        raise RuntimeError("Human and Synthetic reference views disagree")
    code_banks = torch.stack((human_codes, synthetic_codes))
    del human_codes, synthetic_codes, reader
    torch.cuda.empty_cache()

    context_bank = load_file(
        destination / str(mod_cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    heldout_contexts = int(mod_cfg["training"].get("heldout_contexts", 32))
    train_contexts = context_bank[:-heldout_contexts].to(
        device=device, dtype=torch.bfloat16
    )
    validation_context_bank = context_bank[-heldout_contexts:]
    validation_context_count = int(training.get("validation_contexts", 2))
    validation_context_indices = torch.linspace(
        0, heldout_contexts - 1, validation_context_count
    ).round().long().unique()
    validation_contexts = validation_context_bank[
        validation_context_indices
    ].to(device=device, dtype=torch.bfloat16)

    model_cfg = dict(cfg.get("model", {}))
    projector = OracleVisualProjector(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        projector.load_state_dict(resume["projector"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])
        random.setstate(resume["python_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])

    code_weight = float(training.get("code_weight", 1.0))
    factor_weight = float(training.get("factor_weight", 0.25))
    activation_weight = float(training.get("activation_weight", 0.25))
    batch_artists = int(training.get("batch_artists", 16))
    base_lr = float(training.get("learning_rate", 2e-4))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    loss_weights = dict(training.get("code_loss", {}))

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-activation-visual-projector")),
            id=str(wandb_cfg.get("id", "kv-activation-visual-projector")),
            resume="allow",
            config={"kv_activation_visual_projector": cfg},
        )
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        projector.train()
        for step in range(start_step + 1, steps + 1):
            learning_rate = base_lr * min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = learning_rate
            generator = torch.Generator().manual_seed(seed + step * 1_000_003)
            local = torch.randperm(
                len(train_indices_list), generator=generator
            )[:batch_artists].to(device)
            artists = train_indices[local]
            views = torch.randint(
                int(code_banks.shape[2]),
                (batch_artists,),
                generator=generator,
            ).to(device)
            domain = step % 2
            visual = code_banks[domain, artists, views]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                projected = projector(visual)
                block = (step - 1) % modulator.blocks
                predicted_down, predicted_up = modulator(projected, block)
            code_loss, code_metrics = _oracle_code_alignment_objective(
                projected, anchors[artists], visual, loss_weights
            )
            down_loss, down_metrics = kv_factor_objective(
                predicted_down,
                teacher_down[artists, block],
                direction_weight=float(training.get("factor_direction_weight", 1.0)),
                magnitude_weight=float(training.get("factor_magnitude_weight", 0.1)),
            )
            up_loss, up_metrics = kv_factor_objective(
                predicted_up,
                teacher_up[artists, block],
                direction_weight=float(training.get("factor_direction_weight", 1.0)),
                magnitude_weight=float(training.get("factor_magnitude_weight", 0.1)),
            )
            context_index = int(torch.randint(
                len(train_contexts), (1,), generator=generator
            ))
            context = train_contexts[context_index].expand(batch_artists, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_activation = apply_kv_factors(
                    context, predicted_down, predicted_up
                )
                teacher_activation = apply_kv_factors(
                    context,
                    teacher_down[artists, block],
                    teacher_up[artists, block],
                )
            activation_loss, activation_metrics = kv_activation_objective(
                student_activation,
                teacher_activation,
                direction_weight=float(training.get("activation_direction_weight", 0.5)),
                magnitude_weight=float(training.get("activation_magnitude_weight", 0.1)),
            )
            loss = (
                code_weight * code_loss
                + factor_weight * (down_loss + up_loss)
                + activation_weight * activation_loss
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                projector.parameters(), max_grad_norm, foreach=True
            )
            optimizer.step()
            metrics = {
                "loss": loss.detach(),
                "code_loss": code_loss.detach(),
                "factor_down_cosine": down_metrics["cosine"],
                "factor_up_cosine": up_metrics["cosine"],
                "activation_cosine": activation_metrics["cosine"],
                "activation_rms_ratio": activation_metrics["student_to_teacher_rms"],
                "grad_norm": grad_norm.detach(),
                "references": reference_counts[views].float().mean(),
                "domain_is_synthetic": torch.tensor(float(domain), device=device),
                **{f"code_{key}": value for key, value in code_metrics.items()},
            }
            for key, value in metrics.items():
                running[key].append(float(value))
            if step % log_every == 0:
                row = {
                    key: sum(values) / len(values) for key, values in running.items()
                }
                row["learning_rate"] = learning_rate
                print(f"K/V visual projector step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/projector/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                validation = _validate_projector(
                    projector,
                    modulator,
                    code_banks,
                    anchors,
                    validation_indices,
                    teacher_down,
                    teacher_up,
                    validation_contexts,
                    views_per_domain=int(training.get("validation_views_per_domain", 2)),
                )
                print(f"K/V visual projector validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/projector/{key}": value for key, value in validation.items()},
                        step=step,
                    )
            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_projector_state(
                        path,
                        step=step,
                        projector=projector,
                        optimizer=optimizer,
                        cfg=cfg,
                        train_indices=train_indices_list,
                        validation_indices=validation_indices_list,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "train_artists": len(train_indices_list),
        "validation_artists": len(validation_indices_list),
        "projector_parameters": sum(
            parameter.numel() for parameter in projector.parameters()
        ),
        "modulator_frozen": True,
        "reader_frozen": True,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(state_path),
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_kv_activation_visual_projector(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_activation_visual_projector"]
    cfg["output_directory"] = "kv_activation_visual_projector_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["wandb"]["enabled"] = False
    return train_kv_activation_visual_projector(
        effective, destination, steps_override=2
    )
