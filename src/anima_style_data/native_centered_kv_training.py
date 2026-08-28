"""Train the reference K/V generator on fixed-population native artist effects."""

from __future__ import annotations

import copy
import gc
import hashlib
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import write_json
from .kv_activation_generator import (
    _build_direct_delta_generator,
    _clip_outlier_grad_norm,
    _ema_checkpoint_state,
    _excess_common_direction_loss,
    _final_effect_constraints,
    _final_effect_retrieval_loss,
    _load_reader,
    _materialize_reference_token_bank,
    _open_direct_delta_reader,
    _prediction_population_metrics,
    _resolved_experiment_config,
    _save_training_state,
    _select_reference_tokens,
    _update_parameter_ema,
    sample_direct_reference_kv_delta_320,
)
from .native_centered_teacher import NativeCenteredTeacherBank


def fixed_train_population_target(
    centered_teacher: torch.Tensor,
    train_population_offset: torch.Tensor,
    artist_indices: list[int] | torch.Tensor,
    content_index: int,
    timestep_index: int,
) -> torch.Tensor:
    """Return artist effects centered by one frozen train-population mean.

    ``centered_teacher`` was originally centered by the full corpus.  Removing
    the cached mean of its train rows algebraically changes the origin to the
    train population.  No current minibatch statistic participates.
    """

    indices = torch.as_tensor(artist_indices, device="cpu", dtype=torch.long)
    selected = centered_teacher.index_select(0, indices)[
        :, int(content_index), int(timestep_index)
    ].float()
    offset = train_population_offset[
        int(content_index), int(timestep_index)
    ].float()
    return selected - offset


def fixed_validation_case(
    batch_index: int,
    *,
    content_count: int,
    timestep_count: int,
) -> tuple[int, int]:
    """Return a stable, spread-out cache probe for a validation artist batch."""
    return batch_index % content_count, (2 * batch_index + 1) % timestep_count


def _reference_bank_signature(
    style_ids: list[str], *, references: int, seed: int, source: str
) -> dict[str, Any]:
    digest = hashlib.sha256("\n".join(style_ids).encode("utf-8")).hexdigest()
    return {
        "styles": len(style_ids),
        "style_ids_sha256": digest,
        "references": int(references),
        "seed": int(seed),
        "source": str(source),
    }


def _cached_reference_bank(
    loader: CachedTeacherReferenceLoader,
    style_ids: list[str],
    *,
    references: int,
    seed: int,
    chunk_size: int,
    device: str,
    cache_root: Path | None,
    split: str,
    source: str,
) -> torch.Tensor:
    signature = _reference_bank_signature(
        style_ids, references=references, seed=seed, source=source
    )
    tensor_path = cache_root / f"{split}.safetensors" if cache_root else None
    summary_path = cache_root / f"{split}.json" if cache_root else None
    if tensor_path is not None and summary_path is not None:
        if tensor_path.exists() and summary_path.exists():
            import json

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("signature") == signature:
                values = load_file(tensor_path, device="cpu")["tokens"]
                print(
                    f"reused contiguous native reference bank {split}: "
                    f"{tuple(values.shape)}",
                    flush=True,
                )
                return values.to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
    values = _materialize_reference_token_bank(
        loader,
        style_ids,
        references=references,
        seed=seed,
        chunk_size=chunk_size,
        device=device,
    )
    if tensor_path is not None and summary_path is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cpu_values = values.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        temporary = tensor_path.with_name(f".{tensor_path.name}.tmp")
        save_file({"tokens": cpu_values}, temporary)
        temporary.replace(tensor_path)
        write_json(
            summary_path,
            {"signature": signature, "tensor_shape": list(cpu_values.shape)},
        )
        del cpu_values
        print(f"cached contiguous native reference bank {split}: {tensor_path}", flush=True)
    return values


def _scheduled_lr(relative_step: int, config: dict[str, Any]) -> float:
    peak = float(config["peak_lr"])
    warmup = max(1, int(config.get("warmup_steps", 1)))
    if relative_step <= warmup:
        return peak * relative_step / warmup
    decay_start = int(config.get("decay_start_step", warmup))
    decay_end = max(decay_start + 1, int(config.get("decay_end_step", decay_start + 1)))
    final = float(config.get("final_lr", peak))
    progress = min(1.0, max(0.0, (relative_step - decay_start) / (decay_end - decay_start)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final + (peak - final) * cosine


def _model_dimensions(model_cfg: dict[str, Any], state: dict[str, torch.Tensor]) -> tuple[int, int]:
    architecture = str(model_cfg.get("architecture", "direct_cross_attention"))
    if architecture != "direct_cross_attention":
        raise ValueError("Native centered training currently requires direct_cross_attention")
    context_dim = int(state["context_query.0.weight"].shape[1])
    if int(model_cfg.get("output_experts", 0)):
        output_dim = int(state["output_expert_up"].shape[-1])
    elif int(model_cfg.get("output_rank", 0)):
        output_dim = int(state["output_head.0.up"].shape[-1])
    else:
        output_dim = int(state["output_head.0.weight"].shape[0] // 2)
    return context_dim, output_dim


def train_native_centered_reference_kv(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_key: str = "kv_reference_expert_kv_native_centered_750",
) -> dict[str, Any]:
    """Regress held-out-generalizing visual references to cached final velocity."""

    from .kv_activation_sampling import NativeKVActivationInjector
    from .style_transfer import _optimize_frozen_anima, _resolve_anima_model

    cfg = _resolved_experiment_config(config, config_key)
    training = dict(cfg["training"])
    native_cfg = dict(training["native_centered"])
    steps = int(steps_override or training["steps"])
    initial_step = int(cfg.get("initial_step", 0))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260907))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"

    teacher_bank = NativeCenteredTeacherBank.load(
        config,
        destination,
        config_key=str(native_cfg.get("teacher_config_key", "dual_domain_native_teacher")),
    )
    train_population_offset = teacher_bank.train_population_offset()
    train_style_ids = [str(value) for value in teacher_bank.summary["train_style_ids"]]
    validation_style_ids = [
        str(value) for value in teacher_bank.summary["validation_style_ids"]
    ]
    train_teacher_indices = [
        teacher_bank.artist_to_index[value] for value in train_style_ids
    ]
    validation_teacher_indices = [
        teacher_bank.artist_to_index[value] for value in validation_style_ids
    ]
    if len(train_style_ids) < 2 or len(validation_style_ids) < 2:
        raise RuntimeError("Native centered training requires disjoint train/validation artists")

    reference_images = int(native_cfg.get("reference_images", 4))
    chunk = int(native_cfg.get("materialization_style_chunk", 16))
    token_lru = int(native_cfg.get("token_lru_shards", 8))
    reference_root = destination / str(cfg["single_reference_cache"])
    train_loader = CachedTeacherReferenceLoader(
        reference_root,
        split="train",
        style_ids=train_style_ids,
        batch_size=chunk,
        references=reference_images,
        seed=seed ^ 0x54524149,
        token_lru_shards=token_lru,
        strict_style_ids=True,
    )
    validation_loader = CachedTeacherReferenceLoader(
        reference_root,
        split="validation",
        style_ids=validation_style_ids,
        batch_size=chunk,
        references=reference_images,
        seed=seed ^ 0x56414C49,
        token_lru_shards=token_lru,
        strict_style_ids=True,
    )
    cache_root_value = native_cfg.get("reference_bank_cache")
    cache_root = (
        destination / str(cache_root_value) if cache_root_value else None
    )
    train_reference_bank = _cached_reference_bank(
        train_loader,
        train_style_ids,
        references=reference_images,
        seed=seed ^ 0x54524149,
        chunk_size=chunk,
        device=device,
        cache_root=cache_root,
        split="train",
        source=str(cfg["single_reference_cache"]),
    )
    validation_reference_bank = _cached_reference_bank(
        validation_loader,
        validation_style_ids,
        references=reference_images,
        seed=seed ^ 0x56414C49,
        chunk_size=chunk,
        device=device,
        cache_root=cache_root,
        split="validation",
        source=str(cfg["single_reference_cache"]),
    )

    initial = torch.load(
        destination / str(cfg["initial_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    use_initial_ema = bool(cfg.get("initial_use_ema", True))
    initial_model_state = (
        initial["ema_model"] if use_initial_ema and "ema_model" in initial else initial["model"]
    )
    initial_reader_state = (
        initial["ema_reader"] if use_initial_ema and "ema_reader" in initial else initial["reader"]
    )
    model_cfg = dict(cfg["model"])
    context_dim, output_dim = _model_dimensions(model_cfg, initial_model_state)
    reader = _load_reader(config, destination, cfg, device)
    reader_parameters = _open_direct_delta_reader(reader)
    model = _build_direct_delta_generator(
        model_cfg,
        style_dim=int(reader.dim),
        context_dim=context_dim,
        output_dim=output_dim,
        blocks=int(cfg.get("blocks", 28)),
    ).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(initial_model_state, strict=True)
    reader.load_state_dict(initial_reader_state, strict=True)

    generator_lr_cfg = dict(training["generator_lr_schedule"])
    reader_lr_cfg = dict(training["reader_lr_schedule"])
    optimizer = torch.optim.AdamW(
        [
            {"name": "generator", "params": list(model.parameters()), "lr": float(generator_lr_cfg["peak_lr"])},
            {"name": "reader", "params": reader_parameters, "lr": float(reader_lr_cfg["peak_lr"])},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    start_step = initial_step
    resumed = False
    saved_state: dict[str, Any] | None = None
    if bool(training.get("resume", True)) and state_path.exists():
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved_state["model"], strict=True)
        reader.load_state_dict(saved_state["reader"], strict=True)
        optimizer.load_state_dict(saved_state["optimizer"])
        start_step = int(saved_state["step"])
        random.setstate(saved_state["python_rng"])
        torch.set_rng_state(saved_state["torch_rng"])
        torch.cuda.set_rng_state_all(saved_state["cuda_rng"])
        resumed = True
    if start_step < initial_step:
        raise ValueError("Native centered checkpoint precedes initial_step")

    ema_cfg = dict(training.get("ema", {}))
    ema_enabled = bool(ema_cfg.get("enabled", True))
    ema_decay = float(ema_cfg.get("decay", 0.995))
    saved_model_ema = saved_state.get("ema_model") if saved_state is not None else None
    saved_reader_ema = saved_state.get("ema_reader") if saved_state is not None else None
    ema_model = {
        name: (
            saved_model_ema[name].to(device=device, dtype=torch.float32)
            if saved_model_ema is not None
            else parameter.detach().float().clone()
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    } if ema_enabled else None
    ema_reader = {
        name: (
            saved_reader_ema[name].to(device=device, dtype=torch.float32)
            if saved_reader_ema is not None
            else parameter.detach().float().clone()
        )
        for name, parameter in reader.named_parameters()
        if parameter.requires_grad
    } if ema_enabled else None

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    injector = NativeKVActivationInjector(anima, model)
    if hasattr(model, "set_routing_recording"):
        model.set_routing_recording(False)

    batch = int(training.get("batch_size", 4))
    reference_counts = [int(value) for value in native_cfg.get("reference_counts", [1, 2, 4])]
    reference_weights = [float(value) for value in native_cfg.get("reference_count_weights", [0.3, 0.3, 0.4])]
    loss_cfg = dict(native_cfg.get("loss", {}))
    generator_clip = float(training.get("max_grad_norm", 18000.0))
    reader_clip = float(training.get("reader_max_grad_norm", 12.0))
    adaptive = dict(training.get("adaptive_clip", {}))
    generator_adaptive = dict(adaptive.get("generator", {}))
    reader_adaptive = dict(adaptive.get("reader", {}))
    generator_history: list[float] = []
    reader_history: list[float] = []
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 50))
    checkpoint_every = int(training.get("checkpoint_every", 125))

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-native-centered")),
            id=str(wandb_cfg.get("id", "kv-reference-native-centered")),
            resume="allow" if resumed else "never",
            config={config_key: cfg},
        )

    def target_batch(
        teacher_indices: list[int], content: int, timestep: int
    ) -> torch.Tensor:
        return fixed_train_population_target(
            teacher_bank.tensors["centered_teacher"],
            train_population_offset,
            teacher_indices,
            content,
            timestep,
        ).to(device=device, dtype=torch.float32, non_blocking=True)

    def forward_effect(
        style: torch.Tensor, content: int, timestep: int
    ) -> torch.Tensor:
        count = len(style)
        context = teacher_bank.tensors["base_context"][content : content + 1].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        ).expand(count, -1, -1)
        noisy = teacher_bank.tensors["noisy_inputs"][content, timestep].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )[None].expand(count, -1, -1, -1)
        base = teacher_bank.tensors["base_predictions"][content, timestep].to(
            device=device, dtype=torch.float32, non_blocking=True
        )[None].expand(count, -1, -1, -1)
        times = teacher_bank.tensors["timesteps"][timestep].to(
            device=device, dtype=torch.bfloat16
        ).expand(count)
        padding = torch.zeros(
            count, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=torch.bfloat16,
        )
        injector.set_style(style)
        prediction = anima(
            noisy.unsqueeze(2),
            times,
            context=context,
            padding_mask=padding,
            target_input_ids=None,
        ).squeeze(2).float()
        injector.disable()
        return prediction - base

    def objective(
        student: torch.Tensor, teacher: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dimensions = tuple(range(1, teacher.ndim))
        scale = teacher.square().mean(dim=dimensions, keepdim=True).sqrt().clamp_min(
            float(loss_cfg.get("rms_floor", 1e-4))
        )
        huber = F.smooth_l1_loss(
            student / scale,
            teacher / scale,
            beta=float(loss_cfg.get("huber_beta", 0.1)),
        )
        cosine = F.cosine_similarity(
            student.flatten(1), teacher.flatten(1), dim=-1
        ).mean()
        retrieval, retrieval_metrics = _final_effect_retrieval_loss(
            student,
            teacher,
            temperature=float(loss_cfg.get("retrieval_temperature", 0.07)),
        )
        constraints, constraint_metrics = _final_effect_constraints(
            student,
            teacher,
            common_cap=float(loss_cfg.get("common_cap", 0.40)),
            rms_lower=float(loss_cfg.get("rms_lower", 0.80)),
            rms_upper=float(loss_cfg.get("rms_upper", 1.40)),
            common_cap_weight=float(loss_cfg.get("common_cap_weight", 0.05)),
            rms_band_weight=float(loss_cfg.get("rms_band_weight", 1.0)),
            rms_floor=float(loss_cfg.get("rms_floor", 1e-4)),
        )
        relative_common = _excess_common_direction_loss(student, teacher)
        loss = (
            float(loss_cfg.get("huber_weight", 1.0)) * huber
            + float(loss_cfg.get("direction_weight", 1.0)) * (1.0 - cosine)
            + float(loss_cfg.get("retrieval_weight", 0.5)) * retrieval
            + float(loss_cfg.get("constraint_weight", 0.25)) * constraints
            + float(loss_cfg.get("relative_common_weight", 0.10)) * relative_common
        )
        return loss, {
            "normalized_huber": huber.detach(),
            "cosine": cosine.detach(),
            "relative_common_loss": relative_common.detach(),
            **retrieval_metrics,
            **constraint_metrics,
            **_prediction_population_metrics(student),
        }

    @torch.no_grad()
    def validate(step: int) -> dict[str, float]:
        model.eval()
        reader.eval()
        rows: dict[str, list[float]] = defaultdict(list)
        validation_count = min(
            int(native_cfg.get("validation_artists", 16)),
            len(validation_style_ids),
        )
        selected = list(range(validation_count))
        validation_batch = int(native_cfg.get("validation_batch_size", batch))
        for offset in range(0, validation_count, validation_batch):
            local_indices = selected[offset : offset + validation_batch]
            teacher_indices = [validation_teacher_indices[index] for index in local_indices]
            references, mask = _select_reference_tokens(
                validation_reference_bank,
                local_indices,
                reference_counts=[reference_images] * len(local_indices),
                reference_start=0,
                reference_stop=reference_images,
                # Compare identical reference evidence at every validation.
                rng=random.Random(seed ^ 0x56414C ^ offset),
            )
            content, timestep = fixed_validation_case(
                offset // validation_batch,
                content_count=int(teacher_bank.tensors["noisy_inputs"].shape[0]),
                timestep_count=int(teacher_bank.tensors["noisy_inputs"].shape[1]),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = reader(references, mask).tokens
                student = forward_effect(style, content, timestep)
            teacher = target_batch(teacher_indices, content, timestep)
            loss, metrics = objective(student, teacher)
            metrics["loss"] = loss.detach()
            for key, value in metrics.items():
                rows[key].append(float(value))
        model.train()
        reader.train()
        return {key: sum(values) / len(values) for key, values in rows.items()}

    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            relative_step = step - initial_step
            rng = random.Random(seed + relative_step * 1_000_003)
            local_indices = rng.sample(range(len(train_style_ids)), batch)
            teacher_indices = [train_teacher_indices[index] for index in local_indices]
            counts = rng.choices(reference_counts, weights=reference_weights, k=batch)
            references, mask = _select_reference_tokens(
                train_reference_bank,
                local_indices,
                reference_counts=counts,
                reference_start=0,
                reference_stop=reference_images,
                rng=rng,
            )
            combinations = int(teacher_bank.tensors["noisy_inputs"].shape[0]) * int(
                teacher_bank.tensors["noisy_inputs"].shape[1]
            )
            position = (relative_step - 1) % combinations
            content = position % int(teacher_bank.tensors["noisy_inputs"].shape[0])
            timestep = position // int(teacher_bank.tensors["noisy_inputs"].shape[0])
            teacher = target_batch(teacher_indices, content, timestep)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "set_routing_recording"):
                model.set_routing_recording(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = reader(references, mask).tokens
                student = forward_effect(style, content, timestep)
            if hasattr(model, "set_routing_recording"):
                model.set_routing_recording(False)
            loss, metrics = objective(student, teacher)
            if hasattr(model, "routing_auxiliary"):
                routing_balance, _, routing_metrics = model.routing_auxiliary()
            else:
                routing_balance = loss.new_zeros(())
                routing_metrics = {}
            routing_weight = float(native_cfg.get("expert_balance_weight", 0.001))
            total = loss + routing_weight * routing_balance
            total.backward()
            generator_grad, generator_clip_used = _clip_outlier_grad_norm(
                model.parameters(),
                generator_history,
                fallback=generator_clip,
                config=generator_adaptive,
            )
            reader_grad, reader_clip_used = _clip_outlier_grad_norm(
                reader_parameters,
                reader_history,
                fallback=reader_clip,
                config=reader_adaptive,
            )
            generator_history.append(float(generator_grad))
            reader_history.append(float(reader_grad))
            optimizer.param_groups[0]["lr"] = _scheduled_lr(relative_step, generator_lr_cfg)
            optimizer.param_groups[1]["lr"] = _scheduled_lr(relative_step, reader_lr_cfg)
            if hasattr(model, "apply_routing_population_update"):
                routing_metrics.update(model.apply_routing_population_update())
            optimizer.step()
            if ema_model is not None and ema_reader is not None:
                _update_parameter_ema(ema_model, model, ema_decay)
                _update_parameter_ema(ema_reader, reader, ema_decay)

            running["loss"].append(float(total.detach()))
            running["generator_grad_norm_unclipped"].append(float(generator_grad))
            running["reader_grad_norm_unclipped"].append(float(reader_grad))
            running["generator_grad_clip_threshold"].append(generator_clip_used)
            running["reader_grad_clip_threshold"].append(reader_clip_used)
            running["reference_count"].append(sum(counts) / len(counts))
            running["content_index"].append(float(content))
            running["timestep_index"].append(float(timestep))
            for key, value in metrics.items():
                running[key].append(float(value))
            for key, value in routing_metrics.items():
                running[f"routing/{key}"].append(float(value))
            running["routing/expert_balance_weighted"].append(
                float((routing_weight * routing_balance).detach())
            )

            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["generator_lr"] = optimizer.param_groups[0]["lr"]
                row["reader_lr"] = optimizer.param_groups[1]["lr"]
                print(f"Native centered K/V step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()

            if validation_every > 0 and step % validation_every == 0:
                validation = validate(step)
                print(f"Native centered K/V validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val_native/{key}": value for key, value in validation.items()},
                        step=step,
                    )

            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path,
                        step=step,
                        model=model,
                        reader=reader,
                        optimizer=optimizer,
                        cfg=cfg,
                        ema_model=(
                            _ema_checkpoint_state(model, ema_model)
                            if ema_model is not None else None
                        ),
                        ema_reader=(
                            _ema_checkpoint_state(reader, ema_reader)
                            if ema_reader is not None else None
                        ),
                    )
    finally:
        injector.close()
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps": steps,
        "start_step": start_step,
        "training_artists": len(train_style_ids),
        "validation_artists": len(validation_style_ids),
        "teacher_origin": "fixed_train_population_mean",
        "batch_mean_subtraction": False,
        "primary_objective": "native_centered_final_velocity",
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def _train_scheduled_native_centered_reference_kv(
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str,
    sample_key: str,
) -> dict[str, Any]:
    cfg = _resolved_experiment_config(config, config_key)
    training = dict(cfg["training"])
    steps = int(training["steps"])
    targets = sorted(
        int(value) for value in training.get("sample_steps", [])
        if int(cfg.get("initial_step", 0)) < int(value) <= steps
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    completed = max(
        (int(path.stem.removeprefix("step-")) for path in checkpoints.glob("step-*.pt")),
        default=int(cfg.get("initial_step", 0)),
    )
    remaining = [value for value in targets if value > completed]
    if completed < steps and (not remaining or remaining[-1] != steps):
        remaining.append(steps)
    summaries = []
    samples = []
    for target in remaining:
        summaries.append(
            train_native_centered_reference_kv(
                config, destination, steps_override=target, config_key=config_key
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
        sample_config = copy.deepcopy(config)
        sample_config[sample_key]["checkpoint"] = (
            f"{cfg['output_directory']}/checkpoints/step-{target:07d}.pt"
        )
        sample_config[sample_key]["output_directory"] = (
            f"{training['sample_output_root']}-step{target}"
        )
        samples.append(
            sample_direct_reference_kv_delta_320(
                sample_config, destination, sample_config_key=sample_key
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "steps": steps,
        "initial_step": completed,
        "segment_targets": remaining,
        "segments": summaries,
        "samples": samples,
    }


def train_scheduled_native_centered_reference_kv(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _train_scheduled_native_centered_reference_kv(
        config,
        destination,
        config_key="kv_reference_expert_kv_native_centered_750",
        sample_key="kv_reference_expert_kv_native_centered_750_sample",
    )


def train_scheduled_native_centered_reference_kv_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _train_scheduled_native_centered_reference_kv(
        config,
        destination,
        config_key="kv_reference_expert_kv_native_centered_5k",
        sample_key="kv_reference_expert_kv_native_centered_5k_sample",
    )


def sample_native_centered_reference_kv(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kv_native_centered_750_sample",
    )


def sample_native_centered_reference_kv_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kv_native_centered_5k_sample",
    )
