from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .detail_style_cross_attention import (
    DetailPreservingTypedSlotReader,
    SeparatedCommonArtistKVStyleCrossAttention,
)
from .detail_style_training import (
    _build_style_adapter,
    _generate_fixed_reference_sample,
)
from .dual_query_external_samples import load_dual_query_external_sample
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import write_json
from .lora_functional_distillation import (
    FunctionalLoRATeacherBank,
    decompose_teacher_effects,
)
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


def _artist_centered_oracle_objective(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project full LoRA effects into the student artist branch's subspace."""

    student = student.float()
    teacher = teacher.detach().float()
    if student.shape != teacher.shape or student.shape[0] < 2:
        raise ValueError("Oracle supervision needs matching multi-artist batches")
    student_common, student_centered = decompose_teacher_effects(student)
    _, teacher_centered = decompose_teacher_effects(teacher)
    reduce_dims = tuple(range(1, student.ndim))
    row_shape = (-1,) + (1,) * (student.ndim - 1)
    teacher_rms = teacher_centered.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-4)
    student_rms = (student_centered.square().mean(dim=reduce_dims) + 1e-12).sqrt()
    scale = teacher_rms.reshape(row_shape)
    huber = F.smooth_l1_loss(
        student_centered / scale,
        teacher_centered / scale,
        beta=0.10,
    )
    cosine = F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=1
    ).mean()
    magnitude = F.smooth_l1_loss(
        (student_rms / teacher_rms).clamp_min(1e-4).log().clamp(-4, 4),
        torch.zeros_like(student_rms),
        beta=0.10,
    )
    teacher_scale = teacher_rms.mean().clamp_min(1e-4)
    zero_mean = (student_common / teacher_scale).square().mean()

    temperature = float(weights.get("infonce_temperature", 0.10))
    if temperature <= 0:
        raise ValueError("infonce_temperature must be positive")
    student_unit = F.normalize(student_centered.flatten(1), dim=1)
    teacher_unit = F.normalize(teacher_centered.flatten(1), dim=1)
    logits = student_unit @ teacher_unit.t() / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    infonce = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    positive = logits.diagonal() * temperature
    wrong = logits.masked_fill(
        torch.eye(student.shape[0], device=student.device, dtype=torch.bool),
        torch.finfo(logits.dtype).min,
    ).max(dim=1).values * temperature
    total = (
        float(weights.get("huber", 1.0)) * huber
        + float(weights.get("direction", 1.0)) * (1 - cosine)
        + float(weights.get("magnitude", 0.25)) * magnitude
        + float(weights.get("infonce", 0.50)) * infonce
        + float(weights.get("zero_mean", 0.25)) * zero_mean
    )
    student_total_rms = student.square().mean().sqrt().clamp_min(1e-8)
    return total, {
        "loss": total.detach(),
        "centered_huber": huber.detach(),
        "centered_cosine": cosine.detach(),
        "centered_magnitude_loss": magnitude.detach(),
        "centered_student_to_teacher_rms": (student_rms / teacher_rms).mean().detach(),
        "functional_infonce_loss": infonce.detach(),
        "functional_infonce_accuracy": (
            logits.argmax(dim=1) == labels
        ).float().mean().detach(),
        "functional_infonce_positive_cosine": positive.mean().detach(),
        "functional_infonce_hardest_wrong_cosine": wrong.mean().detach(),
        "functional_infonce_cosine_gap": (positive - wrong).mean().detach(),
        "artist_common_zero_loss": zero_mean.detach(),
        "artist_common_output_ratio": (
            student_common.square().mean().sqrt() / student_total_rms
        ).detach(),
    }


def _controlled_style_context_forward(
    anima: torch.nn.Module,
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    style_context: torch.Tensor,
    noisy: torch.Tensor,
    base: torch.Tensor,
    base_context: torch.Tensor,
    timestep: torch.Tensor,
    device: str,
) -> torch.Tensor:
    rows = style_context.shape[0]
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        adapter.set_style_context(style_context)
        adapter.set_timesteps(timestep.expand(rows))
        padding = torch.zeros(
            rows,
            1,
            noisy.shape[-2],
            noisy.shape[-1],
            device=device,
            dtype=noisy.dtype,
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
    return prediction - base


@torch.no_grad()
def _initialize_oracle_codes(
    reader: DetailPreservingTypedSlotReader,
    human_loader: CachedTeacherReferenceLoader,
    synthetic_loader: CachedTeacherReferenceLoader,
    style_ids: list[str],
    *,
    references: int,
    batch_size: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Anchor oracle codes to the existing visual Reader's token manifold."""

    values = []
    reader.eval()
    for offset in range(0, len(style_ids), batch_size):
        part = style_ids[offset : offset + batch_size]
        domain_values = []
        for domain_index, loader in enumerate((human_loader, synthetic_loader)):
            loaded = loader.load_styles(
                part,
                references_per_style=references,
                seed=seed + offset * 1009 + domain_index * 1_000_003,
            )
            tokens = loaded["tokens"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            mask = torch.ones(
                tokens.shape[:2], device=device, dtype=torch.bool
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                domain_values.append(reader(tokens, mask).tokens.float())
        values.append(torch.stack(domain_values).mean(dim=0).cpu())
    return torch.cat(values, dim=0)


def _save_oracle_state(
    path: Path,
    *,
    step: int,
    reader: DetailPreservingTypedSlotReader,
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    oracle_codes: torch.nn.Parameter,
    oracle_anchor: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
        "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "oracle_codes": oracle_codes.detach().cpu(),
        "oracle_anchor": oracle_anchor.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


def train_lora_oracle_bootstrap(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["lora_oracle_bootstrap"])
    training = dict(cfg["training"])
    if steps_override is not None:
        training["steps"] = int(steps_override)
    steps = int(training.get("steps", 1500))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260823))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=True)
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    adapter = _build_style_adapter(detail_cfg).to(device)
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("LoRA oracle bootstrap requires the separated adapter")
    attach_same_q_style_adapter(anima, adapter)
    initial = torch.load(
        destination / str(cfg["initial_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    reader.load_state_dict(initial["reader"], strict=True)
    adapter.load_state_dict(initial["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("artist_only")
    reader.requires_grad_(False).eval()
    for parameter in adapter.common_parameters():
        parameter.requires_grad_(False)

    bank = FunctionalLoRATeacherBank(
        destination / str(cfg["teacher_cache"])
    )
    single_ids = list(bank.by_kind["single"])
    style_ids = [str(bank.mixtures[index]["style_ids"][0]) for index in single_ids]
    references = int(training.get("initialization_references", 4))
    human_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=style_ids,
        batch_size=int(training.get("batch_rows", 8)),
        references=references,
        seed=seed ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["synthetic_reference_cache"]),
        split="train",
        style_ids=style_ids,
        batch_size=int(training.get("batch_rows", 8)),
        references=references,
        seed=seed ^ 0x53594E54,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    resume_state = (
        torch.load(state_path, map_location="cpu", weights_only=False)
        if bool(training.get("resume", True)) and state_path.exists()
        else None
    )
    if resume_state is None:
        oracle_anchor = _initialize_oracle_codes(
            reader,
            human_loader,
            synthetic_loader,
            style_ids,
            references=references,
            batch_size=int(training.get("initialization_batch_size", 8)),
            seed=seed,
            device=device,
        ).to(device)
        oracle_codes = torch.nn.Parameter(oracle_anchor.clone())
        start_step = 0
    else:
        reader.load_state_dict(resume_state["reader"], strict=True)
        adapter.load_state_dict(resume_state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        adapter.set_bootstrap_phase("artist_only")
        oracle_anchor = resume_state["oracle_anchor"].to(device)
        oracle_codes = torch.nn.Parameter(resume_state["oracle_codes"].to(device))
        start_step = int(resume_state["step"])

    groups = [
        {
            "params": [oracle_codes],
            "lr": float(training.get("oracle_learning_rate", 5e-4)),
            "name": "oracle_codes",
            "weight_decay": 0.0,
        },
        {
            "params": adapter.shared_parameters(),
            "lr": float(training.get("shared_learning_rate", 5e-5)),
            "name": "shared_kv",
        },
        {
            "params": adapter.delta_parameters(),
            "lr": float(training.get("delta_learning_rate", 1e-4)),
            "name": "block_delta",
        },
        {
            "params": adapter.mixing_parameters(),
            "lr": float(training.get("mix_learning_rate", 2e-5)),
            "name": "base_mix",
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    base_lrs = {str(group["name"]): float(group["lr"]) for group in groups}

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "lora-oracle-bootstrap-v1")),
            id=str(wandb_cfg.get("id", "lora-oracle-bootstrap-v1")),
            resume="allow" if start_step else "never",
            config={"lora_oracle_bootstrap": cfg},
        )
    fixed = load_dual_query_external_sample(config, destination)
    batch_rows = int(training.get("batch_rows", 8))
    warmup = int(training.get("warmup_steps", 100))
    log_every = int(training.get("log_every", 10))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    sample_every = int(training.get("sample_every", 500))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    anchor_weight = float(training.get("oracle_anchor_weight", 0.01))
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            lr_scale = min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = base_lrs[str(group["name"])] * lr_scale
            optimizer.zero_grad(set_to_none=True)
            rng = random.Random(seed + step * 1_000_003)
            positions = rng.sample(range(len(single_ids)), batch_rows)
            teacher_indices = [single_ids[index] for index in positions]
            content_index = step % int(bank.base["noisy_inputs"].shape[0])
            timestep_index = (step // int(bank.base["noisy_inputs"].shape[0])) % int(
                bank.base["noisy_inputs"].shape[1]
            )
            noisy = bank.base["noisy_inputs"][content_index, timestep_index].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            base = bank.base["base_predictions"][content_index, timestep_index].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            context = bank.base["base_context"][content_index : content_index + 1].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            timestep = bank.base["timesteps"][timestep_index].to(
                device=device, dtype=torch.bfloat16
            )
            teacher = bank.effects[
                teacher_indices, content_index, timestep_index
            ].to(device=device, dtype=torch.float32, non_blocking=True)
            codes = oracle_codes[positions]
            student = _controlled_style_context_forward(
                anima, adapter, codes, noisy, base, context, timestep, device
            )
            loss, metrics = _artist_centered_oracle_objective(
                student, teacher, dict(training.get("loss_weights", {}))
            )
            anchor_loss = F.smooth_l1_loss(
                codes.float(), oracle_anchor[positions].float(), beta=0.10
            )
            total = loss + anchor_weight * anchor_loss
            total.backward()
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_grad_norm, foreach=True
            )
            optimizer.step()
            metrics.update({
                "oracle_anchor_loss": anchor_loss.detach(),
                "oracle_anchor_weighted_loss": (anchor_weight * anchor_loss).detach(),
                "oracle_code_rms": oracle_codes.detach().square().mean().sqrt(),
                "timestep": timestep.detach().float(),
            })
            for key, value in metrics.items():
                running[key].append(float(value))
            running["grad_norm"].append(float(grad_norm))
            if step % log_every == 0:
                row = {
                    key: sum(values) / len(values)
                    for key, values in running.items()
                    if values
                }
                row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                print(f"LoRA oracle step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()
            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_oracle_state(
                        path,
                        step=step,
                        reader=reader,
                        adapter=adapter,
                        oracle_codes=oracle_codes,
                        oracle_anchor=oracle_anchor,
                        optimizer=optimizer,
                        cfg=cfg,
                    )
            if sample_every > 0 and step % sample_every == 0:
                sample = _generate_fixed_reference_sample(
                    fixed,
                    config,
                    destination,
                    anima,
                    reader,
                    adapter,
                    output,
                    device,
                    step,
                    component_mode="artist_only",
                    strengths_override=[1.0],
                    sample_group="fixed_reference_samples",
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log({
                        "val/functional/fixed_reference": wandb.Image(
                            sample["sheet"], caption=f"step {step}"
                        )
                    }, step=step)
    finally:
        adapter.clear_style_tokens()
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(style_ids),
        "component_mode": "artist_only",
        "reader_frozen": True,
        "common_frozen_and_bypassed": True,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_oracle_bootstrap(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_oracle_bootstrap"]
    cfg["output_directory"] = "lora_oracle_bootstrap_smoke"
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    return train_lora_oracle_bootstrap(effective, destination, steps_override=3)
