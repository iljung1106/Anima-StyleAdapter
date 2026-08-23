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
from safetensors.torch import load_file, save_file

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .detail_style_training import (
    _build_style_adapter,
    _generate_fixed_reference_sample,
)
from .dual_query_external_samples import load_dual_query_external_sample
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json
from .lora_oracle_bootstrap import (
    _FixedOracleCodeReader,
    _initialize_oracle_codes,
    _oracle_adapter_initial_state,
    _oracle_detail_config,
    _save_oracle_state,
)
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_calibration import _encode_prompts
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


def cache_lora_image_flow_targets(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Consolidate LoRA-generated x0 latents and their eight text contexts."""

    cfg = copy.deepcopy(config["lora_image_flow_oracle"])
    source = destination / str(cfg["source_directory"])
    output = destination / str(cfg["target_cache"])
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "targets.safetensors"
    summary_path = output / "summary.json"
    if tensor_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cached = load_file(tensor_path, device="cpu")
        expected = (
            int(summary["artists"]),
            int(summary["contents"]),
            16,
            int(summary["latent_height"]),
            int(summary["latent_width"]),
        )
        if tuple(cached["latents"].shape) != expected:
            raise ValueError(
                "Existing LoRA image-flow target cache has the wrong shape"
            )
        return summary

    rows = sorted(
        read_records(source / "manifest.parquet"),
        key=lambda row: (int(row["artist_index"]), int(row["content_index"])),
    )
    artist_indices = sorted({int(row["artist_index"]) for row in rows})
    content_indices = sorted({int(row["content_index"]) for row in rows})
    if artist_indices != list(range(len(artist_indices))):
        raise ValueError("LoRA target artist indices must be contiguous")
    if content_indices != list(range(len(content_indices))):
        raise ValueError("LoRA target content indices must be contiguous")
    expected_rows = len(artist_indices) * len(content_indices)
    if len(rows) != expected_rows:
        raise ValueError("LoRA target manifest is not a complete artist/content grid")

    prompts: list[str] = []
    for content_index in content_indices:
        values = {
            str(row["content_prompt"])
            for row in rows
            if int(row["content_index"]) == content_index
        }
        if len(values) != 1:
            raise ValueError("Every content index must have exactly one shared prompt")
        prompts.append(values.pop())
    contexts = _encode_prompts(
        config,
        destination,
        prompts,
        str(cfg.get("cache_device", "cuda")),
        int(cfg.get("text_batch_size", len(prompts))),
    ).contiguous()

    first = load_file(source / "latents" / str(rows[0]["latent_shard"]), device="cpu")
    latent_shape = tuple(first["latents"].shape[1:])
    latents = torch.empty(
        len(artist_indices), len(content_indices), *latent_shape, dtype=torch.float16
    )
    style_ids = [""] * len(artist_indices)
    loaded_shards: dict[str, torch.Tensor] = {}
    for row in rows:
        shard_name = str(row["latent_shard"])
        if shard_name not in loaded_shards:
            loaded_shards[shard_name] = load_file(
                source / "latents" / shard_name, device="cpu"
            )["latents"]
        artist_index = int(row["artist_index"])
        content_index = int(row["content_index"])
        latents[artist_index, content_index].copy_(
            loaded_shards[shard_name][int(row["latent_row"])]
        )
        style_id = str(row["style_id"])
        if style_ids[artist_index] and style_ids[artist_index] != style_id:
            raise ValueError("One artist index maps to multiple style IDs")
        style_ids[artist_index] = style_id

    save_file(
        {"latents": latents.contiguous(), "contexts": contexts.contiguous()},
        tensor_path,
    )
    summary = {
        "artists": len(artist_indices),
        "contents": len(content_indices),
        "images": len(rows),
        "latent_height": int(latent_shape[-2]),
        "latent_width": int(latent_shape[-1]),
        "style_ids": style_ids,
        "prompts": prompts,
        "source_directory": str(source),
        "tensor_path": str(tensor_path),
    }
    write_json(summary_path, summary)
    return summary


def _flow_forward(
    anima: torch.nn.Module,
    adapter: torch.nn.Module,
    codes: torch.Tensor | None,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    device: str,
) -> torch.Tensor:
    rows = noisy.shape[0]
    padding = torch.zeros(
        rows, 1, noisy.shape[-2], noisy.shape[-1], device=device, dtype=noisy.dtype
    )
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        if codes is None:
            adapter.clear_style_tokens()
        else:
            adapter.set_style_context(codes)
            adapter.set_timesteps(timestep)
        try:
            return (
                anima(
                    noisy.unsqueeze(2),
                    timestep,
                    context=context,
                    padding_mask=padding,
                    target_input_ids=None,
                )
                .squeeze(2)
                .float()
            )
        finally:
            adapter.clear_style_tokens()


def _centered_image_flow_objective(
    styled_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    weights: dict[str, float],
    *,
    artist_weight_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prioritize artist-specific x0 effects over the shared content shortcut."""

    styled = styled_prediction.float()
    base = base_prediction.detach().float()
    target = target_velocity.detach().float()
    if (
        styled.shape != base.shape
        or styled.shape != target.shape
        or styled.shape[0] < 2
    ):
        raise ValueError("Centered image-flow supervision needs matching artist rows")
    student = styled - base
    desired = target - base
    student_common = student.mean(dim=0, keepdim=True)
    desired_common = desired.mean(dim=0, keepdim=True)
    student_centered = student - student_common
    desired_centered = desired - desired_common

    reduce_dims = tuple(range(1, desired.ndim))
    row_shape = (-1,) + (1,) * (desired.ndim - 1)
    desired_rms = desired_centered.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-4)
    student_rms = (student_centered.square().mean(dim=reduce_dims) + 1e-12).sqrt()
    scale = desired_rms.reshape(row_shape)
    centered_mse = F.mse_loss(student_centered / scale, desired_centered / scale)
    centered_cosine = F.cosine_similarity(
        student_centered.flatten(1), desired_centered.flatten(1), dim=1
    ).mean()
    centered_magnitude = F.smooth_l1_loss(
        (student_rms / desired_rms).clamp_min(1e-4).log().clamp(-4, 4),
        torch.zeros_like(student_rms),
        beta=0.10,
    )

    common_scale = desired_common.square().mean().sqrt().clamp_min(1e-4)
    common_huber = F.smooth_l1_loss(
        student_common / common_scale,
        desired_common / common_scale,
        beta=0.10,
    )
    flow_mse = F.mse_loss(styled, target)

    factors = [int(value) for value in weights.get("descriptor_factors", [4, 8, 16])]
    student_parts: list[torch.Tensor] = []
    desired_parts: list[torch.Tensor] = []
    for factor in factors:
        if factor <= 0 or min(student.shape[-2:]) < factor:
            continue
        student_parts.append(
            F.normalize(
                F.avg_pool2d(student_centered, factor, factor).flatten(1), dim=1
            )
        )
        desired_parts.append(
            F.normalize(
                F.avg_pool2d(desired_centered, factor, factor).flatten(1), dim=1
            )
        )
    student_parts.append(F.normalize(student_centered.mean(dim=(-2, -1)), dim=1))
    desired_parts.append(F.normalize(desired_centered.mean(dim=(-2, -1)), dim=1))
    student_descriptor = F.normalize(torch.cat(student_parts, dim=1), dim=1)
    desired_descriptor = F.normalize(torch.cat(desired_parts, dim=1), dim=1)
    temperature = float(weights.get("infonce_temperature", 0.10))
    logits = student_descriptor @ desired_descriptor.t() / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    infonce = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    positive = logits.diagonal() * temperature
    wrong = (
        logits.masked_fill(
            torch.eye(student.shape[0], device=student.device, dtype=torch.bool),
            torch.finfo(logits.dtype).min,
        )
        .max(dim=1)
        .values
        * temperature
    )

    artist_objective = (
        float(weights.get("centered_mse", 1.0)) * centered_mse
        + float(weights.get("centered_direction", 1.0)) * (1 - centered_cosine)
        + float(weights.get("centered_magnitude", 0.10)) * centered_magnitude
        + float(weights.get("infonce", 0.50)) * infonce
    )
    total = (
        float(weights.get("flow_mse", 0.10)) * flow_mse
        + float(weights.get("common_huber", 0.10)) * common_huber
        + float(artist_weight_multiplier) * artist_objective
    )
    student_total_rms = student.square().mean().sqrt().clamp_min(1e-8)
    return total, {
        "loss": total.detach(),
        "flow_loss": flow_mse.detach(),
        "common_huber": common_huber.detach(),
        "centered_mse": centered_mse.detach(),
        "centered_cosine": centered_cosine.detach(),
        "centered_magnitude_loss": centered_magnitude.detach(),
        "centered_student_to_desired_rms": (student_rms / desired_rms).mean().detach(),
        "infonce_loss": infonce.detach(),
        "infonce_accuracy": (logits.argmax(dim=1) == labels).float().mean().detach(),
        "infonce_positive_cosine": positive.mean().detach(),
        "infonce_hardest_wrong_cosine": wrong.mean().detach(),
        "infonce_cosine_gap": (positive - wrong).mean().detach(),
        "common_output_ratio": (
            student_common.square().mean().sqrt() / student_total_rms
        ).detach(),
    }


def _sample_weighted_timestep(
    rng: random.Random,
    edges: list[float],
    sampling_weights: list[float],
) -> tuple[float, int]:
    if len(edges) < 2 or len(sampling_weights) != len(edges) - 1:
        raise ValueError("Timestep edges and sampling weights do not match")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("Timestep edges must be strictly increasing")
    if any(weight <= 0 for weight in sampling_weights):
        raise ValueError("Timestep sampling weights must be positive")
    bin_index = rng.choices(range(len(sampling_weights)), weights=sampling_weights)[0]
    return rng.uniform(edges[bin_index], edges[bin_index + 1]), bin_index


@torch.no_grad()
def _validate_image_flow_oracle(
    anima: torch.nn.Module,
    adapter: torch.nn.Module,
    oracle_codes: torch.Tensor,
    latents: torch.Tensor,
    contexts: torch.Tensor,
    *,
    seed: int,
    rows: int,
    timesteps: list[float],
    device: str,
) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    artist_indices = torch.arange(rows, device=device)
    content_index = 0
    clean = latents[artist_indices, content_index].to(torch.bfloat16)
    context = contexts[content_index : content_index + 1].expand(rows, -1, -1)
    noise = torch.randn(
        clean[0:1].shape, device=device, dtype=torch.bfloat16, generator=generator
    ).expand_as(clean)
    values: dict[str, list[float]] = defaultdict(list)
    for timestep_value in timesteps:
        timestep = torch.full(
            (rows,), timestep_value, device=device, dtype=torch.bfloat16
        )
        sigma = timestep[:, None, None, None]
        noisy = (1 - sigma) * clean + sigma * noise
        target = noise.float() - clean.float()
        base = _flow_forward(anima, adapter, None, noisy, timestep, context, device)
        styled = _flow_forward(
            anima,
            adapter,
            oracle_codes[artist_indices],
            noisy,
            timestep,
            context,
            device,
        )
        base_loss = (base - target).square().flatten(1).mean(dim=1)
        styled_loss = (styled - target).square().flatten(1).mean(dim=1)
        desired = target - base
        student = styled - base
        cosine = F.cosine_similarity(student.flatten(1), desired.flatten(1), dim=1)
        student_centered = student - student.mean(dim=0, keepdim=True)
        desired_centered = desired - desired.mean(dim=0, keepdim=True)
        centered_cosine = F.cosine_similarity(
            student_centered.flatten(1), desired_centered.flatten(1), dim=1
        )
        timestep_key = f"timestep_{timestep_value:.2f}"
        values["base_flow_loss"].append(float(base_loss.mean()))
        values["flow_loss"].append(float(styled_loss.mean()))
        values["paired_flow_improvement"].append(
            float((base_loss - styled_loss).mean())
        )
        values["residual_cosine"].append(float(cosine.mean()))
        values["student_to_desired_rms"].append(
            float(
                student.square().mean().sqrt()
                / desired.square().mean().sqrt().clamp_min(1e-8)
            )
        )
        values[f"{timestep_key}/paired_flow_improvement"].append(
            float((base_loss - styled_loss).mean())
        )
        values[f"{timestep_key}/centered_cosine"].append(float(centered_cosine.mean()))
        values[f"{timestep_key}/centered_student_to_desired_rms"].append(
            float(
                student_centered.square().mean().sqrt()
                / desired_centered.square().mean().sqrt().clamp_min(1e-8)
            )
        )
        values[f"{timestep_key}/common_output_ratio"].append(
            float(
                student.mean(dim=0).square().mean().sqrt()
                / student.square().mean().sqrt().clamp_min(1e-8)
            )
        )
    return {key: sum(items) / len(items) for key, items in values.items()}


def train_lora_image_flow_oracle(
    config: dict[str, Any], destination: Path, *, steps_override: int | None = None
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["lora_image_flow_oracle"])
    training = dict(cfg["training"])
    if steps_override is not None:
        training["steps"] = int(steps_override)
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    resume_state = (
        torch.load(state_path, map_location="cpu", weights_only=False)
        if bool(training.get("resume", True)) and state_path.exists()
        else None
    )
    target_summary = cache_lora_image_flow_targets(config, destination)
    targets = load_file(destination / str(cfg["target_cache"]) / "targets.safetensors")
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260823))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    anima = (
        _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    )
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    detail_cfg = _oracle_detail_config(config, cfg)
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    adapter = _build_style_adapter(detail_cfg).to(device)
    attach_same_q_style_adapter(anima, adapter)
    initial = torch.load(
        destination / str(cfg["initial_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    reader.load_state_dict(initial["reader"], strict=True)
    adapter.load_state_dict(
        _oracle_adapter_initial_state(
            adapter.state_dict(),
            initial["adapter"],
            str(cfg.get("adapter_initialization", "fresh_kv_checkpoint_strength")),
        ),
        strict=True,
    )
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("artist_only")
    reader.requires_grad_(False).eval()
    for parameter in adapter.common_parameters():
        parameter.requires_grad_(False)

    style_ids = list(target_summary["style_ids"])
    initial_oracle_state = None
    if resume_state is None and cfg.get("initial_oracle_checkpoint"):
        initial_oracle_state = torch.load(
            destination / str(cfg["initial_oracle_checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
    if resume_state is None and initial_oracle_state is None:
        references = int(training.get("initialization_references", 4))
        human_loader = CachedTeacherReferenceLoader(
            destination / str(cfg["human_reference_cache"]),
            split="train",
            style_ids=style_ids,
            batch_size=int(training.get("batch_size", 16)),
            references=references,
            seed=seed ^ 0x48554D41,
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            strict_style_ids=True,
        )
        synthetic_loader = CachedTeacherReferenceLoader(
            destination / str(cfg["synthetic_reference_cache"]),
            split="train",
            style_ids=style_ids,
            batch_size=int(training.get("batch_size", 16)),
            references=references,
            seed=seed ^ 0x53594E54,
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            strict_style_ids=True,
        )
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
    elif resume_state is None:
        assert initial_oracle_state is not None
        reader.load_state_dict(initial_oracle_state["reader"], strict=True)
        adapter.load_state_dict(initial_oracle_state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        adapter.set_bootstrap_phase("artist_only")
        oracle_anchor = initial_oracle_state["oracle_anchor"].to(device)
        oracle_codes = torch.nn.Parameter(
            initial_oracle_state["oracle_codes"].to(device)
        )
        start_step = 0
    else:
        reader.load_state_dict(resume_state["reader"], strict=True)
        adapter.load_state_dict(resume_state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        adapter.set_bootstrap_phase("artist_only")
        oracle_anchor = resume_state["oracle_anchor"].to(device)
        oracle_codes = torch.nn.Parameter(resume_state["oracle_codes"].to(device))
        start_step = int(resume_state["step"])
    latents = targets["latents"].to(device=device, dtype=torch.bfloat16)
    contexts = targets["contexts"].to(device=device, dtype=torch.bfloat16)

    groups = [
        {
            "params": [oracle_codes],
            "lr": float(training["oracle_learning_rate"]),
            "name": "oracle_codes",
            "weight_decay": 0.0,
        },
        {
            "params": adapter.shared_parameters(),
            "lr": float(training["shared_learning_rate"]),
            "name": "shared_kv",
        },
        {
            "params": adapter.delta_parameters(),
            "lr": float(training.get("delta_learning_rate", 0.0)),
            "name": "block_delta",
        },
        {
            "params": adapter.mixing_parameters(),
            "lr": float(training.get("mix_learning_rate", 0.0)),
            "name": "base_mix",
            "weight_decay": 0.0,
        },
    ]
    if adapter.null_parameters():
        groups.append(
            {
                "params": adapter.null_parameters(),
                "lr": float(training["null_learning_rate"]),
                "name": "artist_null",
                "weight_decay": 0.0,
            }
        )
    groups = [group for group in groups if group["params"] and float(group["lr"]) > 0]
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    base_lrs = {
        str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
    }

    steps = int(training.get("steps", 2000))
    batch_size = int(training.get("batch_size", 16))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    sample_every = int(training.get("sample_every", 500))
    anchor_weight = float(training.get("oracle_anchor_weight", 0.001))
    timestep_edges = [
        float(value)
        for value in training.get("timestep_bin_edges", [0.0, 0.2, 0.45, 0.75, 1.0])
    ]
    timestep_sampling_weights = [
        float(value)
        for value in training.get(
            "timestep_sampling_weights", [1.0] * (len(timestep_edges) - 1)
        )
    ]
    timestep_artist_multipliers = [
        float(value)
        for value in training.get(
            "timestep_artist_loss_multipliers",
            [1.0] * (len(timestep_edges) - 1),
        )
    ]
    if len(timestep_artist_multipliers) != len(timestep_edges) - 1:
        raise ValueError("Timestep artist-loss multipliers do not match the bins")
    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg["name"]),
            id=str(wandb_cfg["id"]),
            resume="allow",
            config={"lora_image_flow_oracle": cfg},
        )

    fixed = dict(load_dual_query_external_sample(config, destination))
    fixed_rows = len(fixed["paths"])
    reference_root = destination / str(cfg["source_directory"]) / "images"
    fixed["paths"] = [
        reference_root / f"artist-{index:03d}" / "content-00.webp"
        for index in range(fixed_rows)
    ]
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            lr_scale = min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = base_lrs[str(group["name"])] * lr_scale
            optimizer.zero_grad(set_to_none=True)
            rng = random.Random(seed + step * 1_000_003)
            step_generator = torch.Generator(device=device).manual_seed(
                seed ^ 0x464C4F57 ^ (step * 0x9E3779B1)
            )
            artists = torch.tensor(
                rng.sample(range(latents.shape[0]), batch_size),
                device=device,
                dtype=torch.long,
            )
            content_index = (step - 1) % latents.shape[1]
            clean = latents[artists, content_index]
            noise = torch.randn(
                clean[0:1].shape,
                device=device,
                dtype=torch.bfloat16,
                generator=step_generator,
            ).expand_as(clean)
            timestep_scalar, timestep_bin = _sample_weighted_timestep(
                rng, timestep_edges, timestep_sampling_weights
            )
            timestep_value = torch.tensor(timestep_scalar, device=device)
            timestep = timestep_value.to(torch.bfloat16).expand(batch_size)
            sigma = timestep[:, None, None, None]
            noisy = (1 - sigma) * clean + sigma * noise
            target = noise.float() - clean.float()
            context = contexts[content_index : content_index + 1].expand(
                batch_size, -1, -1
            )
            with torch.no_grad():
                base_prediction = _flow_forward(
                    anima,
                    adapter,
                    None,
                    noisy,
                    timestep,
                    context,
                    device,
                )
            prediction = _flow_forward(
                anima,
                adapter,
                oracle_codes[artists],
                noisy,
                timestep,
                context,
                device,
            )
            objective, metrics = _centered_image_flow_objective(
                prediction,
                base_prediction,
                target,
                dict(training.get("loss_weights", {})),
                artist_weight_multiplier=timestep_artist_multipliers[timestep_bin],
            )
            anchor_loss = F.smooth_l1_loss(
                oracle_codes[artists].float(), oracle_anchor[artists].float(), beta=0.10
            )
            loss = objective + anchor_weight * anchor_loss
            loss.backward()
            group_norms = {
                str(group["name"]): torch.nn.utils.clip_grad_norm_(
                    group["params"], max_grad_norm, foreach=True
                )
                for group in optimizer.param_groups
            }
            optimizer.step()
            for key, value in metrics.items():
                running[key].append(float(value))
            running["anchor_loss"].append(float(anchor_loss.detach()))
            running["timestep"].append(timestep_scalar)
            running["timestep_artist_loss_multiplier"].append(
                timestep_artist_multipliers[timestep_bin]
            )
            for index in range(len(timestep_sampling_weights)):
                running[f"timestep/bin_{index}_fraction"].append(
                    float(index == timestep_bin)
                )
            for key in (
                "centered_cosine",
                "centered_student_to_desired_rms",
                "infonce_accuracy",
                "common_output_ratio",
            ):
                running[f"timestep/bin_{timestep_bin}_{key}"].append(
                    float(metrics[key])
                )
            for name, value in group_norms.items():
                running[f"grad_norm_{name}"].append(float(value))
            if step % log_every == 0:
                row = {key: sum(items) / len(items) for key, items in running.items()}
                print(f"LoRA image-flow oracle step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in row.items()}, step=step
                    )
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                metrics = _validate_image_flow_oracle(
                    anima,
                    adapter,
                    oracle_codes,
                    latents,
                    contexts,
                    seed=seed ^ 0x56414C,
                    rows=min(16, latents.shape[0]),
                    timesteps=[0.10, 0.35, 0.65, 0.90],
                    device=device,
                )
                print(f"LoRA image-flow validation step={step} {metrics}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/{key}": value for key, value in metrics.items()},
                        step=step,
                    )
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
                direct_reader = (
                    _FixedOracleCodeReader(oracle_codes[:fixed_rows].detach())
                    .to(device)
                    .eval()
                )
                sample = _generate_fixed_reference_sample(
                    fixed,
                    config,
                    destination,
                    anima,
                    direct_reader,
                    adapter,
                    output,
                    device,
                    step,
                    component_mode="artist_only",
                    strengths_override=[1.0],
                    sample_group="oracle_code_samples",
                    sample_suffix="image-flow-direct",
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log(
                        {
                            "val/oracle/image_flow_fixed_reference": wandb.Image(
                                sample["sheet"]
                            )
                        },
                        step=step,
                    )
    finally:
        adapter.clear_style_tokens()
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": int(latents.shape[0]),
        "contents": int(latents.shape[1]),
        "images": int(latents.shape[0] * latents.shape[1]),
        "flow_loss_only": True,
        "reader_frozen": True,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_image_flow_oracle(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_image_flow_oracle"]
    cfg["output_directory"] = "lora_image_flow_oracle_smoke"
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    cfg["training"]["batch_size"] = 4
    return train_lora_image_flow_oracle(effective, destination, steps_override=2)
