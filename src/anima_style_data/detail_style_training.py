"""Production runner for detail-preserving typed-slot Style Cross-Attention."""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .detail_style_cross_attention import (
    DetailPreservingTypedSlotReader,
    FreshKVStyleCrossAttention,
)
from .detail_style_teacher_context import NativeArtistContextCache
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .dual_query_style_training import (
    _artist_flow_ranking_loss,
    _build_native_effect_timestep_weighting,
    _native_artist_teacher_objective,
    _native_effect_weights_for_timesteps,
)
from .global_query_style_tokenizer import MultiPromptDualQueryCachedStyleLoader
from .io import write_json
from .native_centered_teacher import NativeCenteredTeacherBank
from .pure_token_injection import _reference_batch
from .query_style_tokenizer import (
    _reference_inputs,
    _sample_query_style_tokenizer,
    _select_sample_episodes,
)
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_tokenizer import _flow_metrics, _mean_metrics
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
    result.update(dict(cfg["loader"]))
    result["split"] = split
    result["resampler_token_cache"] = str(cfg["cache"]["output_directory"])
    if split == str(cfg.get("train_split", "train")):
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


def _rho_min(step: int) -> float:
    if step <= 250:
        return 0.0
    return 0.5 * min(1.0, (step - 250) / 750)


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latents = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
    context = batch["conditioning"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
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
        dropout = float(training.get("style_dropout", 0.05))
        enabled = torch.ones(latents.shape[0], dtype=torch.bool, device=device)
        if reader.training and dropout > 0:
            enabled = torch.rand(
                latents.shape[0], device=device, generator=generator
            ) >= dropout
        adapter.set_style_context(output.tokens, enabled=enabled)
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
        "style_enabled_fraction": enabled.float().mean(),
        "timestep_mean": timesteps.detach().mean(),
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

    need_wrong = (
        train_auxiliaries
        and step >= int(training.get("functional_start_step", 500))
        and step % int(training.get("functional_every", 4)) == 0
        and prediction.shape[0] >= 2
    )
    base_prediction = None
    if measure_base or need_wrong:
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
            step, 500, 1_500, float(training.get("functional_weight", 0.10))
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
    metrics["loss"] = total.detach()
    return total, metrics


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
    *,
    step: int,
    probe_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    references, mask = _reference_inputs(batch, device, "heldout")
    rows = min(int(training.get("teacher_batch_rows", 4)), references.shape[0])
    references, mask = references[:rows], mask[:rows]
    style_ids = [str(item.style_id) for item in batch["episodes"][:rows]]
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
    base_context = tensors["base_context"][content_index : content_index + 1].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    tagged = contexts.get(style_ids, content_index).to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    timestep = tensors["timesteps"][timestep_index].to(
        device=device, dtype=torch.bfloat16
    )

    # Two disjoint reference views share x_t, text, timestep and teacher.  One
    # fused forward supplies both the functional consistency signal and dense
    # same-Q supervision without a second Anima graph.
    positions = torch.arange(mask.shape[1], device=device)[None]
    first_mask = mask & positions.remainder(2).eq(0)
    second_mask = mask & positions.remainder(2).eq(1)
    has_two = mask.sum(dim=1).ge(2)
    second_mask = torch.where(has_two[:, None], second_mask, first_mask)
    combined_references = torch.cat((references, references))
    combined_mask = torch.cat((first_mask, second_mask))
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        style = reader(combined_references, combined_mask).tokens
        adapter.reset_internal_teacher()
        adapter.set_style_context(style)
        adapter.set_teacher_context(torch.cat((tagged, tagged)))
        padding = torch.zeros(
            rows * 2, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=noisy.dtype,
        )
        try:
            prediction = anima(
                noisy.expand(rows * 2, -1, -1, -1).unsqueeze(2),
                timestep.expand(rows * 2),
                context=base_context.expand(rows * 2, -1, -1),
                padding_mask=padding, target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()
    first, second = (prediction - base).chunk(2)
    student = 0.5 * (first + second)
    objective_cfg = dict(training)
    objective_cfg.update(dict(training.get("teacher_objective", {})))
    final_loss, metrics = _native_artist_teacher_objective(
        student, teacher, objective_cfg, step=step
    )
    internal, internal_metrics = adapter.internal_teacher_loss(
        rho_min=_rho_min(step), rho_max=1.5
    )
    teacher_scale = teacher.float().square().mean(
        dim=tuple(range(1, teacher.ndim))
    ).sqrt().clamp_min(1e-4)
    consistency = (
        (first - second).float().square().mean(
            dim=tuple(range(1, first.ndim))
        ).sqrt() / teacher_scale
    ).mean()
    consistency_weight = _ramp(
        step, 500, 1_500, float(training.get("same_artist_weight", 0.05))
    )
    timestep_weight = _native_effect_weights_for_timesteps(
        timestep.float().reshape(1), timestep_weighting
    )[0]
    internal_weight = float(training.get("internal_teacher_weight", 0.25))
    total = timestep_weight * (
        final_loss + internal_weight * internal
        + consistency_weight * consistency
    )
    metrics.update(internal_metrics)
    metrics.update({
        "same_artist_consistency_loss": consistency.detach(),
        "same_artist_consistency_weight": consistency.new_tensor(
            consistency_weight
        ),
        "teacher_timestep_weight": timestep_weight.detach(),
        "teacher_content_index": consistency.new_tensor(content_index),
        "teacher_timestep": timestep.detach().float(),
        "teacher_total_loss": total.detach(),
        "teacher_student_view_rms": student.detach().square().mean().sqrt(),
    })
    return total, metrics


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
) -> dict[str, Any]:
    adapter.begin_alpha_calibration()
    reader.eval()
    for index in range(max(1, batches)):
        batch = loader.load_step(index)
        references, mask = _reference_inputs(batch, device, "heldout")
        style_ids = [str(item.style_id) for item in batch["episodes"]]
        content_index = index % int(bank.tensors["noisy_inputs"].shape[0])
        timestep_index = index % int(bank.tensors["noisy_inputs"].shape[1])
        noisy = bank.tensors["noisy_inputs"][content_index, timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        timestep = bank.tensors["timesteps"][timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        context = bank.tensors["base_context"][content_index : content_index + 1].to(
            device=device, dtype=torch.bfloat16
        )
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
    result = adapter.finish_alpha_calibration()
    reader.train()
    return result


def _save_state(
    path: Path,
    *,
    step: int,
    reader: DetailPreservingTypedSlotReader,
    adapter: FreshKVStyleCrossAttention,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
        "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg,
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
) -> dict[str, float]:
    reader.eval()
    adapter.eval()
    rows = []
    for index in range(batches):
        _, metrics = _flow_step(
            anima, reader, adapter, loader.load_step(index), device, training,
            timestep_weighting,
            generator=torch.Generator(device=device).manual_seed(seed + index * 97),
            step=step, mode=mode, train_auxiliaries=False, measure_base=True,
        )
        rows.append({key: float(value) for key, value in metrics.items()})
    reader.train()
    adapter.train()
    return _mean_metrics(rows)


def train_detail_style_cross_attention(
    config: dict[str, Any], destination: Path, *, steps_override: int | None = None
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    training = dict(cfg["training"])
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
    train_loader = MultiPromptDualQueryCachedStyleLoader(destination, train_cfg)
    validation_loader = MultiPromptDualQueryCachedStyleLoader(destination, validation_cfg)
    _audit_student_prompts(train_loader)
    _audit_student_prompts(validation_loader)

    bank_key = str(cfg["teacher"].get("bank_config_key", "dual_domain_native_teacher"))
    bank = NativeCenteredTeacherBank.load(config, destination, config_key=bank_key)
    context_root = destination / str(cfg["teacher"]["context_cache"])
    contexts = NativeArtistContextCache(
        context_root, capacity=int(cfg["teacher"].get("context_lru_shards", 8))
    )
    synthetic_roots = [destination / str(value) for value in cfg["teacher"]["reference_caches"]]
    teacher_loader = CachedTeacherReferenceLoader(
        synthetic_roots,
        split="train",
        style_ids=list(bank.summary["train_style_ids"]),
        batch_size=int(training.get("teacher_batch_rows", 4)),
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
    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device)
    adapter = FreshKVStyleCrossAttention(**dict(cfg["adapter"])).to(device)
    attach_same_q_style_adapter(anima, adapter)

    output = destination / str(cfg["output_directory"])
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    resume = None
    if bool(training.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(resume["reader"], strict=True)
        adapter.load_state_dict(resume["adapter"], strict=True)
        start_step = int(resume["step"])
    else:
        calibration = _calibrate_alpha(
            anima, reader, adapter, bank, contexts, teacher_loader, device,
            int(training.get("alpha_calibration_batches", 4)),
        )
        write_json(output / "alpha_calibration.json", calibration)

    reader_parameters = [value for value in reader.parameters() if value.requires_grad]
    kv_parameters = adapter.kv_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": reader_parameters, "lr": float(training.get("learning_rate", 1e-4)), "name": "reader"},
            {"params": kv_parameters, "lr": float(training.get("kv_learning_rate", 5e-5)), "name": "style_kv"},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True) and device.startswith("cuda")),
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])

    reader_count = sum(value.numel() for value in reader_parameters)
    kv_count = sum(value.numel() for value in kv_parameters)
    print(
        f"detail-style model reader={reader_count/1e6:.2f}M "
        f"fresh_kv={kv_count/1e6:.2f}M alpha_fixed={adapter.alpha.tolist()}",
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
                "detail_preserving_style_cross_attention": cfg,
                "reader_parameters": reader_count,
                "fresh_kv_parameters": kv_count,
            },
        )

    base_lr = float(training.get("learning_rate", 1e-4))
    kv_lr = float(training.get("kv_learning_rate", 5e-5))
    warmup = int(training.get("warmup_steps", 500))
    minimum_ratio = float(training.get("minimum_lr_ratio", 0.1))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    state_every = int(training.get("state_every", checkpoint_every))
    sample_every = int(training.get("sample_every", 500))
    teacher_every_after = int(training.get("teacher_every_after_bootstrap", 2))
    teacher_bootstrap_end = int(training.get("teacher_every_step_until", 500))
    prefetched = train_loader.prefetch(
        start_step * accumulation,
        (steps - start_step) * accumulation,
        workers=int(training.get("prefetch_workers", 2)),
        depth=int(training.get("prefetch_batches", 6)),
    )
    sample_seed = int(cfg.get("sampling", {}).get("seed", seed ^ 0x5A17))
    sample_requests = [
        ("train", train_loader, episode, sample_seed + index * 10_007)
        for index, episode in enumerate(_select_sample_episodes(train_loader, 4))
    ] + [
        ("validation", validation_loader, episode, sample_seed + (index + 4) * 10_007)
        for index, episode in enumerate(_select_sample_episodes(validation_loader, 4))
    ]
    history_path = output / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    teacher_update = start_step if start_step <= teacher_bootstrap_end else (
        teacher_bootstrap_end + (start_step - teacher_bootstrap_end) // teacher_every_after
    )
    completed = start_step
    started = time.perf_counter()
    vae = None
    try:
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            lr_scale = _learning_rate_multiplier(step, steps, warmup, minimum_ratio)
            optimizer.param_groups[0]["lr"] = base_lr * lr_scale
            optimizer.param_groups[1]["lr"] = kv_lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            metric_rows: list[dict[str, torch.Tensor]] = []
            for micro in range(accumulation):
                batch = next(prefetched)
                generator = torch.Generator(device=device).manual_seed(
                    seed + step * 100_003 + micro
                )
                loss, metrics = _flow_step(
                    anima, reader, adapter, batch, device, training,
                    timestep_weighting,
                    generator=generator, step=step, mode="curriculum",
                    train_auxiliaries=True,
                    measure_base=(step % log_every == 0 and micro == accumulation - 1),
                )
                (loss / accumulation).backward()
                metric_rows.append(metrics)
            teacher_due = step <= teacher_bootstrap_end or step % teacher_every_after == 0
            if teacher_due:
                teacher_loss, teacher_metrics = _teacher_step(
                    anima, reader, adapter, bank, contexts,
                    teacher_loader.load_step(teacher_update), device, training,
                    timestep_weighting, step=step, probe_index=teacher_update,
                )
                teacher_loss.backward()
                metric_rows[-1].update(teacher_metrics)
                teacher_update += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                reader_parameters + kv_parameters,
                float(training.get("max_grad_norm", 1.0)),
            )
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
                    "reader_lr": optimizer.param_groups[0]["lr"],
                    "kv_lr": optimizer.param_groups[1]["lr"],
                    "step_s": time.perf_counter() - step_started,
                    "images_per_s": train_loader.batch_size * accumulation /
                    max(time.perf_counter() - step_started, 1e-6),
                })
                averaged.update(adapter.runtime_stats())
                print(
                    f"detail-style step={step}/{steps} loss={averaged['loss']:.5f} "
                    f"flow={averaged['flow_loss']:.5f} grad={averaged['grad_norm']:.4f} "
                    f"teacher={averaged.get('teacher_total_loss', float('nan')):.5f} "
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
                        elif key.startswith(("style_", "grad_", "reader_lr", "kv_lr")):
                            namespace = "model/activation"
                        elif "artist" in key or "functional" in key:
                            namespace = "train/artist"
                        else:
                            namespace = "train/flow"
                        namespaces[f"{namespace}/{key}"] = value
                    wandb_run.log(namespaces, step=step)

            if step % validation_every == 0 or step == steps:
                validation = {
                    mode: _evaluate(
                        anima, reader, adapter, validation_loader, device, training,
                        timestep_weighting, step=step,
                        batches=int(training.get("validation_batches", 8)),
                        mode=mode, seed=seed ^ (0xBEEF if mode == "self" else 0xC0FFEE),
                    )
                    for mode in ("self", "heldout", "wrong_artist")
                }
                row = {"step": step, **validation}
                history.append(row)
                write_json(history_path, history)
                print(f"detail-style validation step={step} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({
                        f"val/functional/{mode}/{key}": value
                        for mode, values in validation.items()
                        for key, value in values.items()
                    }, step=step)
            if state_every > 0 and (step % state_every == 0 or step == steps):
                _save_state(
                    state_path, step=step, reader=reader, adapter=adapter,
                    optimizer=optimizer, cfg=cfg,
                )
            if step % checkpoint_every == 0 or step == steps:
                _save_state(
                    checkpoint_dir / f"step-{step:07d}.pt", step=step,
                    reader=reader, adapter=adapter, optimizer=optimizer, cfg=cfg,
                )
            if sample_every > 0 and (step % sample_every == 0 or step == steps):
                sample_records, vae = _sample_query_style_tokenizer(
                    anima, adapter, reader, sample_requests, config, destination,
                    output, device, step, vae,
                    config_section="detail_preserving_style_cross_attention",
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "val/functional/panel": [
                            wandb.Image(str(path), caption=label)
                            for label, path in sample_records
                        ]
                    }, step=step)
            completed = step
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
        "fresh_kv_parameters": kv_count,
        "elapsed_s": time.perf_counter() - started,
        "final_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", result)
    return result


def smoke_test_detail_style_cross_attention(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    cfg = smoke["detail_preserving_style_cross_attention"]
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
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
        "teacher_batch_rows": 2,
        "functional_start_step": 1,
        "functional_every": 1,
    })
    cfg["training"].setdefault("wandb", {})["enabled"] = False
    return train_detail_style_cross_attention(smoke, destination, steps_override=2)
