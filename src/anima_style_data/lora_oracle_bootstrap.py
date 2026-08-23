from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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
from .native_centered_teacher import NativeCenteredTeacherBank
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


class _FixedOracleCodeReader(torch.nn.Module):
    """Expose learned oracle codes through the fixed-sample Reader contract."""

    def __init__(self, tokens: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("tokens", tokens)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> SimpleNamespace:
        del reference_mask, reconstruct
        if references.shape[0] != self.tokens.shape[0]:
            raise ValueError(
                f"Expected {self.tokens.shape[0]} oracle rows, got {references.shape[0]}"
            )
        return SimpleNamespace(tokens=self.tokens)


class OracleVisualProjector(torch.nn.Module):
    """Low-capacity residual map from content-bearing Reader codes to oracle codes."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        slots: int = 28,
        heads: int = 16,
        ff_dim: int = 2048,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.slots = int(slots)
        self.slot_identity = torch.nn.Parameter(torch.empty(slots, dim))
        self.attention_norm = torch.nn.LayerNorm(dim)
        self.attention = torch.nn.MultiheadAttention(
            dim, heads, batch_first=True, bias=False
        )
        self.ff_norm = torch.nn.LayerNorm(dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, ff_dim, bias=False),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(ff_dim, dim, bias=False),
        )
        self.output_norm = torch.nn.LayerNorm(dim)
        self.output_down = torch.nn.Linear(dim, bottleneck_dim, bias=False)
        self.output_up = torch.nn.Linear(bottleneck_dim, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.dim**-0.5
        torch.nn.init.normal_(self.slot_identity, std=std)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
        # A tiny nonzero residual exposes every layer to gradients immediately
        # while preserving the pretrained Reader path at initialization.
        torch.nn.init.normal_(self.output_up.weight, std=1e-4)

    def forward(self, visual_codes: torch.Tensor) -> torch.Tensor:
        if visual_codes.ndim != 3 or visual_codes.shape[1:] != (
            self.slots,
            self.dim,
        ):
            raise ValueError(
                f"Expected [batch,{self.slots},{self.dim}] visual codes"
            )
        hidden = visual_codes + self.slot_identity.to(visual_codes.dtype)[None]
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = hidden + attended
        hidden = hidden + self.ff(self.ff_norm(hidden))
        delta = self.output_up(
            torch.nn.functional.silu(self.output_down(self.output_norm(hidden)))
        )
        return visual_codes + delta


class _ProjectedReader(torch.nn.Module):
    """Compose the frozen Reader and learned visual-to-oracle projector."""

    def __init__(
        self,
        reader: DetailPreservingTypedSlotReader,
        projector: OracleVisualProjector,
    ) -> None:
        super().__init__()
        self.reader = reader
        self.projector = projector

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> SimpleNamespace:
        output = self.reader(
            references, reference_mask, reconstruct=reconstruct
        )
        return SimpleNamespace(tokens=self.projector(output.tokens))


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
            resume="allow",
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
                running[key].append(float(value.detach()))
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


def sample_lora_oracle_checkpoint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render learned oracle codes directly, bypassing the frozen visual Reader."""

    cfg = copy.deepcopy(config["lora_oracle_bootstrap"])
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    checkpoint = output / "training_state.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=True)
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    adapter = _build_style_adapter(detail_cfg).to(device)
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("LoRA oracle sampling requires the separated adapter")
    # The block-conditioned K/V modules are materialized when the adapter is
    # bound to Anima's 28 attention blocks, so binding must precede loading.
    attach_same_q_style_adapter(anima, adapter)
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("artist_only")
    adapter.requires_grad_(False).eval()

    prepared = dict(load_dual_query_external_sample(config, destination))
    rows = len(prepared["paths"])
    codes = state["oracle_codes"][:rows].to(device=device, dtype=torch.bfloat16)
    reader = _FixedOracleCodeReader(codes).to(device).eval()

    reference_cache = destination / str(cfg["synthetic_reference_cache"])
    reference_root = reference_cache / "images"
    if not reference_root.is_dir() and (reference_cache.parent / "images").is_dir():
        reference_root = reference_cache.parent / "images"
    paths = [
        reference_root / f"artist-{index:03d}" / "content-00.webp"
        for index in range(rows)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing oracle reference previews: {missing[:3]}")
    prepared["paths"] = paths

    result = _generate_fixed_reference_sample(
        prepared,
        config,
        destination,
        anima,
        reader,
        adapter,
        output,
        device,
        int(state["step"]),
        component_mode="artist_only",
        strengths_override=[1.0],
        sample_group="oracle_code_samples",
        sample_suffix="direct",
    )
    result.update({
        "checkpoint": str(checkpoint),
        "oracle_codes_direct": True,
        "reader_bypassed": True,
    })
    write_json(
        output / "oracle_code_samples" / f"step-{int(state['step']):07d}-summary.json",
        result,
    )
    return result


def _interpolate_oracle_visual(
    oracle: torch.Tensor, visual: torch.Tensor, fraction: float
) -> torch.Tensor:
    if oracle.shape != visual.shape:
        raise ValueError("Oracle and visual contexts must have identical shapes")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Visual interpolation fraction must be in [0, 1]")
    return torch.lerp(oracle, visual, float(fraction))


def _oracle_code_alignment_objective(
    projected: torch.Tensor,
    oracle: torch.Tensor,
    visual: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    projected = projected.float()
    oracle = oracle.detach().float()
    visual = visual.detach().float()
    if projected.shape != oracle.shape or projected.shape != visual.shape:
        raise ValueError("Projected, oracle, and visual codes must match")
    reduce_dims = tuple(range(1, projected.ndim))
    row_shape = (-1,) + (1,) * (projected.ndim - 1)
    oracle_rms = oracle.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-4)
    scale = oracle_rms.reshape(row_shape)
    huber = F.smooth_l1_loss(
        projected / scale, oracle / scale, beta=0.10
    )
    raw_cosine = F.cosine_similarity(
        projected.flatten(1), oracle.flatten(1), dim=1
    ).mean()
    projected_centered = projected - projected.mean(dim=0, keepdim=True)
    oracle_centered = oracle - oracle.mean(dim=0, keepdim=True)
    centered_cosine = F.cosine_similarity(
        projected_centered.flatten(1), oracle_centered.flatten(1), dim=1
    ).mean()
    projected_centered_rms = projected_centered.square().mean(
        dim=reduce_dims
    ).sqrt().clamp_min(1e-5)
    oracle_centered_rms = oracle_centered.square().mean(
        dim=reduce_dims
    ).sqrt().clamp_min(1e-5)
    magnitude = F.smooth_l1_loss(
        (projected_centered_rms / oracle_centered_rms).log().clamp(-4, 4),
        torch.zeros_like(projected_centered_rms),
        beta=0.10,
    )
    temperature = float(weights.get("infonce_temperature", 0.10))
    projected_unit = F.normalize(projected_centered.flatten(1), dim=1)
    oracle_unit = F.normalize(oracle_centered.flatten(1), dim=1)
    logits = projected_unit @ oracle_unit.t() / temperature
    labels = torch.arange(projected.shape[0], device=projected.device)
    infonce = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    identity = ((projected - visual) / scale).square().mean()
    total = (
        float(weights.get("huber", 1.0)) * huber
        + float(weights.get("raw_direction", 0.25)) * (1 - raw_cosine)
        + float(weights.get("centered_direction", 1.0)) * (1 - centered_cosine)
        + float(weights.get("centered_magnitude", 0.10)) * magnitude
        + float(weights.get("infonce", 0.50)) * infonce
        + float(weights.get("identity", 0.01)) * identity
    )
    return total, {
        "loss": total.detach(),
        "code_huber": huber.detach(),
        "raw_cosine": raw_cosine.detach(),
        "centered_cosine": centered_cosine.detach(),
        "centered_magnitude_loss": magnitude.detach(),
        "infonce_loss": infonce.detach(),
        "infonce_accuracy": (logits.argmax(dim=1) == labels).float().mean().detach(),
        "identity_loss": identity.detach(),
        "projected_to_oracle_rms": (projected - oracle).square().mean().sqrt().detach(),
        "visual_to_oracle_rms": (visual - oracle).square().mean().sqrt().detach(),
        "projected_centered_to_oracle_rms": (
            projected_centered_rms / oracle_centered_rms
        ).mean().detach(),
    }


def _save_visual_bridge_state(
    path: Path,
    *,
    step: int,
    reader: DetailPreservingTypedSlotReader,
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    oracle_codes: torch.Tensor,
    projector: OracleVisualProjector,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
        "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "oracle_codes": oracle_codes.detach().cpu(),
        "projector": {
            key: value.detach().cpu()
            for key, value in projector.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "config": config,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


def _piecewise_linear_value(
    step: int, points: list[list[float]] | list[tuple[float, float]]
) -> float:
    """Interpolate a scalar curriculum from monotonically increasing steps."""

    if not points:
        raise ValueError("piecewise schedule needs at least one point")
    normalized = [(int(point[0]), float(point[1])) for point in points]
    if any(right[0] <= left[0] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("piecewise schedule steps must be strictly increasing")
    if step <= normalized[0][0]:
        return normalized[0][1]
    for (left_step, left_value), (right_step, right_value) in zip(
        normalized, normalized[1:]
    ):
        if step <= right_step:
            fraction = (step - left_step) / max(1, right_step - left_step)
            return left_value + fraction * (right_value - left_value)
    return normalized[-1][1]


def train_lora_oracle_visual_bridge(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Expand the oracle-trained connector onto frozen visual Reader outputs."""

    root_cfg = copy.deepcopy(config["lora_oracle_bootstrap"])
    cfg = dict(root_cfg["visual_bridge"])
    steps = int(steps_override if steps_override is not None else cfg["steps"])
    device = str(root_cfg["training"].get("device", "cuda"))
    seed = int(root_cfg.get("seed", 20260823)) ^ 0x42524944
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
        raise TypeError("Oracle visual bridge requires the separated adapter")
    attach_same_q_style_adapter(anima, adapter)

    source = torch.load(
        destination / str(cfg["source_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    reader.load_state_dict(source["reader"], strict=True)
    adapter.load_state_dict(source["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("artist_only")
    reader.requires_grad_(False).eval()
    for parameter in adapter.common_parameters():
        parameter.requires_grad_(False)
    oracle_codes = source["oracle_codes"].to(device=device, dtype=torch.bfloat16)
    projector_cfg = dict(root_cfg["visual_projector"])
    projector = OracleVisualProjector(
        dim=int(projector_cfg.get("dim", 1024)),
        slots=int(projector_cfg.get("slots", 28)),
        heads=int(projector_cfg.get("heads", 16)),
        ff_dim=int(projector_cfg.get("ff_dim", 2048)),
        bottleneck_dim=int(projector_cfg.get("bottleneck_dim", 256)),
    ).to(device)
    projector_state = torch.load(
        destination / str(cfg["projector_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    projector.load_state_dict(projector_state["projector"], strict=True)
    projector.requires_grad_(False).eval()
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    state_path = output / "training_state.pt"
    initial_state = None
    if (
        not state_path.exists()
        and cfg.get("initial_bridge_checkpoint")
    ):
        initial_state = torch.load(
            destination / str(cfg["initial_bridge_checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        reader.load_state_dict(initial_state["reader"], strict=True)
        adapter.load_state_dict(initial_state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        adapter.set_bootstrap_phase("artist_only")
        if "projector" in initial_state:
            projector.load_state_dict(initial_state["projector"], strict=True)
    projected_reader = _ProjectedReader(reader, projector).eval()

    bank = FunctionalLoRATeacherBank(destination / str(root_cfg["teacher_cache"]))
    single_ids = list(bank.by_kind["single"])
    style_ids = [str(bank.mixtures[index]["style_ids"][0]) for index in single_ids]
    loader_kwargs = {
        "split": "train",
        "style_ids": style_ids,
        "batch_size": len(style_ids),
        "references": int(cfg.get("materialized_reference_images", 8)),
        "token_lru_shards": int(cfg.get(
            "token_lru_shards",
            root_cfg["training"].get("token_lru_shards", 8),
        )),
        "strict_style_ids": True,
    }
    human_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["human_reference_cache"]),
        seed=seed ^ 0x48554D41,
        **loader_kwargs,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["synthetic_reference_cache"]),
        seed=seed ^ 0x53594E54,
        **loader_kwargs,
    )
    reference_images = int(cfg.get("materialized_reference_images", 8))
    human_codes, reference_counts = _materialize_reader_code_bank(
        reader,
        human_loader,
        style_ids,
        reference_images=reference_images,
        seed=seed ^ 0x11111111,
        device=device,
    )
    synthetic_codes, synthetic_reference_counts = _materialize_reader_code_bank(
        reader,
        synthetic_loader,
        style_ids,
        reference_images=reference_images,
        seed=seed ^ 0x22222222,
        device=device,
    )
    if not torch.equal(reference_counts, synthetic_reference_counts):
        raise RuntimeError("Human and synthetic Reader banks disagree on views")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        projected_parts = []
        for values in (human_codes, synthetic_codes):
            flat = values.flatten(0, 1)
            chunks = [
                projector(flat[offset : offset + 64])
                for offset in range(0, len(flat), 64)
            ]
            projected_parts.append(
                torch.cat(chunks).reshape_as(values).contiguous()
            )
    visual_code_banks = torch.stack(projected_parts, dim=0)
    del human_codes, synthetic_codes, projected_parts

    groups = [
        {"params": adapter.shared_parameters(), "lr": float(cfg["shared_learning_rate"]), "name": "shared_kv"},
        {"params": adapter.delta_parameters(), "lr": float(cfg["delta_learning_rate"]), "name": "block_delta"},
        {"params": adapter.mixing_parameters(), "lr": float(cfg["mix_learning_rate"]), "name": "base_mix", "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=bool(cfg.get("fused_adamw", True)),
    )
    base_lrs = {str(group["name"]): float(group["lr"]) for group in groups}
    checkpoints.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if bool(cfg.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(resume["reader"], strict=True)
        adapter.load_state_dict(resume["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        adapter.set_bootstrap_phase("artist_only")
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])
    elif initial_state is not None:
        optimizer.load_state_dict(initial_state["optimizer"])
        start_step = int(initial_state["step"])

    wandb_run = None
    wandb_cfg = dict(cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "lora-oracle-visual-bridge-v1")),
            id=str(wandb_cfg.get("id", "lora-oracle-visual-bridge-v1")),
            resume="allow",
            config={"lora_oracle_visual_bridge": cfg},
        )

    fixed = load_dual_query_external_sample(config, destination)
    batch_rows = int(cfg["batch_rows"])
    warmup = int(cfg.get("warmup_steps", 100))
    ramp_steps = int(cfg.get("visual_fraction_ramp_steps", 750))
    fraction_points = cfg.get("visual_fraction_points")

    def visual_fraction(step: int) -> float:
        if fraction_points:
            return _piecewise_linear_value(step, fraction_points)
        return min(1.0, step / max(1, ramp_steps))

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
            position_index = torch.tensor(
                positions, device=device, dtype=torch.long
            )
            view_index = torch.tensor(
                [rng.randrange(visual_code_banks.shape[2]) for _ in positions],
                device=device,
                dtype=torch.long,
            )
            domain_index = step % 2
            visual_codes = visual_code_banks[
                domain_index, position_index, view_index
            ]
            fraction = visual_fraction(step)
            contexts = _interpolate_oracle_visual(
                oracle_codes[position_index], visual_codes, fraction
            )

            content_index = step % int(bank.base["noisy_inputs"].shape[0])
            timestep_index = (step // int(bank.base["noisy_inputs"].shape[0])) % int(bank.base["noisy_inputs"].shape[1])
            noisy = bank.base["noisy_inputs"][content_index, timestep_index].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            base = bank.base["base_predictions"][content_index, timestep_index].to(device=device, dtype=torch.float32, non_blocking=True)
            context = bank.base["base_context"][content_index : content_index + 1].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            timestep = bank.base["timesteps"][timestep_index].to(device=device, dtype=torch.bfloat16)
            teacher_indices = [single_ids[index] for index in positions]
            teacher = bank.effects[teacher_indices, content_index, timestep_index].to(device=device, dtype=torch.float32, non_blocking=True)
            student = _controlled_style_context_forward(
                anima, adapter, contexts, noisy, base, context, timestep, device
            )
            loss, metrics = _artist_centered_oracle_objective(
                student, teacher, dict(cfg.get("loss_weights", {}))
            )
            loss.backward()
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg.get("max_grad_norm", 1.0)), foreach=True
            )
            optimizer.step()
            visual_delta = (
                visual_codes.float() - oracle_codes[position_index].float()
            )
            metrics.update({
                "visual_fraction": torch.tensor(fraction, device=device),
                "references": reference_counts[view_index].mean(),
                "domain_is_human": torch.tensor(
                    float(domain_index == 1), device=device
                ),
                "visual_to_oracle_rms": visual_delta.square().mean().sqrt(),
                "visual_to_oracle_cosine": F.cosine_similarity(
                    visual_codes.float().flatten(1),
                    oracle_codes[positions].float().flatten(1),
                    dim=1,
                ).mean(),
            })
            for key, value in metrics.items():
                running[key].append(float(value.detach()))
            running["grad_norm"].append(float(grad_norm))

            if step % int(cfg.get("log_every", 10)) == 0:
                row = {key: sum(values) / len(values) for key, values in running.items() if values}
                row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                print(f"LoRA visual bridge step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()
            if step % int(cfg.get("checkpoint_every", 250)) == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_visual_bridge_state(
                        path,
                        step=step,
                        reader=reader,
                        adapter=adapter,
                        oracle_codes=oracle_codes,
                        projector=projector,
                        optimizer=optimizer,
                        config=cfg,
                    )
            if int(cfg.get("sample_every", 500)) > 0 and step % int(cfg["sample_every"]) == 0:
                sample = _generate_fixed_reference_sample(
                    fixed, config, destination, anima, projected_reader, adapter, output,
                    device, step, component_mode="artist_only",
                    strengths_override=[1.0], sample_group="fixed_reference_samples",
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "val/functional/fixed_reference": wandb.Image(
                            sample["sheet"], caption=f"step {step}, visual={fraction:.2f}"
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
        "reader_frozen": True,
        "oracle_codes_frozen": True,
        "visual_projector_frozen": True,
        "common_frozen_and_bypassed": True,
        "final_visual_fraction": visual_fraction(steps),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_oracle_visual_bridge(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_oracle_bootstrap"]["visual_bridge"]
    cfg["output_directory"] = "lora_oracle_visual_bridge_smoke"
    cfg["resume"] = False
    cfg.pop("initial_bridge_checkpoint", None)
    cfg.pop("visual_fraction_points", None)
    cfg["checkpoint_every"] = 1
    cfg["sample_every"] = 0
    cfg["wandb"]["enabled"] = False
    return train_lora_oracle_visual_bridge(effective, destination, steps_override=3)


def _save_visual_projector_state(
    path: Path,
    *,
    step: int,
    projector: OracleVisualProjector,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "projector": {
            key: value.detach().cpu()
            for key, value in projector.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "config": config,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _materialize_reader_code_bank(
    reader: DetailPreservingTypedSlotReader,
    loader: CachedTeacherReferenceLoader,
    style_ids: list[str],
    *,
    reference_images: int,
    seed: int,
    device: str,
    style_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reusable 1/2/4-reference Reader outputs once on the GPU."""

    if reference_images < 4 or reference_images % 4:
        raise ValueError("materialized_reference_images must be a multiple of four")
    groups: list[tuple[int, int]] = []
    groups.extend((index, 1) for index in range(reference_images))
    groups.extend((index, 2) for index in range(0, reference_images, 2))
    groups.extend((index, 4) for index in range(0, reference_images, 4))
    reference_counts = torch.tensor(
        [count for _, count in groups], device=device, dtype=torch.float32
    )
    chunk_size = len(style_ids) if style_chunk_size is None else int(style_chunk_size)
    if chunk_size <= 0:
        raise ValueError("style_chunk_size must be positive")
    materialized: torch.Tensor | None = None
    reader.eval()
    device_type = torch.device(device).type
    chunk_offsets = list(range(0, len(style_ids), chunk_size))
    materialize_started = time.perf_counter()
    for chunk_index, offset in enumerate(chunk_offsets):
        chunk_ids = style_ids[offset : offset + chunk_size]
        loaded = loader.load_styles(
            chunk_ids,
            references_per_style=reference_images,
            seed=seed + offset * 1_000_003,
        )
        tokens = loaded["tokens"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        codes = []
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            for start, count in groups:
                part = tokens[:, start : start + count]
                mask = torch.ones(part.shape[:2], device=device, dtype=torch.bool)
                codes.append(reader(part, mask).tokens)
        chunk_codes = torch.stack(codes, dim=1)
        if materialized is None:
            materialized = torch.empty(
                len(style_ids),
                len(groups),
                *chunk_codes.shape[2:],
                device=device,
                dtype=chunk_codes.dtype,
            )
        materialized[offset : offset + len(chunk_ids)].copy_(chunk_codes)
        del tokens, codes, chunk_codes
        if len(chunk_offsets) > 1 and (
            (chunk_index + 1) % 5 == 0 or chunk_index + 1 == len(chunk_offsets)
        ):
            elapsed = time.perf_counter() - materialize_started
            print(
                "materialized Reader codes "
                f"{min(offset + chunk_size, len(style_ids))}/{len(style_ids)} "
                f"styles ({elapsed:.1f}s)",
                flush=True,
            )
    if materialized is None:
        raise ValueError("Cannot materialize an empty style list")
    return materialized, reference_counts


def train_lora_oracle_visual_projector(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Learn a generic content-suppressing residual map into oracle space."""

    root_cfg = copy.deepcopy(config["lora_oracle_bootstrap"])
    cfg = dict(root_cfg["visual_projector"])
    steps = int(steps_override if steps_override is not None else cfg["steps"])
    device = str(root_cfg["training"].get("device", "cuda"))
    seed = int(root_cfg.get("seed", 20260823)) ^ 0x50524F4A
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    source = torch.load(
        destination / str(cfg["source_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    reader.load_state_dict(source["reader"], strict=True)
    reader.requires_grad_(False).eval()
    oracle_codes = source["oracle_codes"].to(device=device, dtype=torch.bfloat16)
    projector = OracleVisualProjector(
        dim=int(cfg.get("dim", 1024)),
        slots=int(cfg.get("slots", 28)),
        heads=int(cfg.get("heads", 16)),
        ff_dim=int(cfg.get("ff_dim", 2048)),
        bottleneck_dim=int(cfg.get("bottleneck_dim", 256)),
    ).to(device)

    bank = FunctionalLoRATeacherBank(destination / str(root_cfg["teacher_cache"]))
    single_ids = list(bank.by_kind["single"])
    style_ids = [str(bank.mixtures[index]["style_ids"][0]) for index in single_ids]
    batch_rows = int(cfg["batch_rows"])
    reference_images = int(cfg.get("materialized_reference_images", 8))
    loader_kwargs = {
        "split": "train",
        "style_ids": style_ids,
        "batch_size": batch_rows,
        "references": reference_images,
        "token_lru_shards": int(cfg.get(
            "token_lru_shards",
            root_cfg["training"].get("token_lru_shards", 8),
        )),
        "strict_style_ids": True,
    }
    human_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["human_reference_cache"]),
        seed=seed ^ 0x48554D41,
        **loader_kwargs,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["synthetic_reference_cache"]),
        seed=seed ^ 0x53594E54,
        **loader_kwargs,
    )
    human_codes, reference_counts = _materialize_reader_code_bank(
        reader,
        human_loader,
        style_ids,
        reference_images=reference_images,
        seed=seed ^ 0x11111111,
        device=device,
    )
    synthetic_codes, synthetic_reference_counts = _materialize_reader_code_bank(
        reader,
        synthetic_loader,
        style_ids,
        reference_images=reference_images,
        seed=seed ^ 0x22222222,
        device=device,
    )
    if not torch.equal(reference_counts, synthetic_reference_counts):
        raise RuntimeError("Human and synthetic Reader banks disagree on views")
    visual_code_banks = torch.stack((human_codes, synthetic_codes), dim=0)
    del human_codes, synthetic_codes

    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=bool(cfg.get("fused_adamw", True)),
    )
    base_lr = float(cfg["learning_rate"])
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(cfg.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        projector.load_state_dict(resume["projector"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])

    wandb_run = None
    wandb_cfg = dict(cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "lora-oracle-visual-projector-v1")),
            id=str(wandb_cfg.get("id", "lora-oracle-visual-projector-v1")),
            resume="allow",
            config={"lora_oracle_visual_projector": cfg},
        )

    warmup = int(cfg.get("warmup_steps", 100))
    log_every = int(cfg.get("log_every", 10))
    checkpoint_every = int(cfg.get("checkpoint_every", 250))
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            lr_scale = min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = base_lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            rng = random.Random(seed + step * 1_000_003)
            positions = rng.sample(range(len(single_ids)), batch_rows)
            position_index = torch.tensor(positions, device=device, dtype=torch.long)
            view_index = torch.tensor(
                [rng.randrange(visual_code_banks.shape[2]) for _ in positions],
                device=device,
                dtype=torch.long,
            )
            domain_index = step % 2
            visual_codes = visual_code_banks[
                domain_index, position_index, view_index
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                projected = projector(visual_codes)
            loss, metrics = _oracle_code_alignment_objective(
                projected,
                oracle_codes[position_index],
                visual_codes,
                dict(cfg.get("loss_weights", {})),
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                projector.parameters(),
                float(cfg.get("max_grad_norm", 1.0)),
                foreach=True,
            )
            optimizer.step()
            metrics.update({
                "references": reference_counts[view_index].mean(),
                "domain_is_human": torch.tensor(
                    float(domain_index == 1), device=device
                ),
                "projector_delta_rms": (
                    projected.float() - visual_codes.float()
                ).square().mean().sqrt(),
            })
            for key, value in metrics.items():
                running[key].append(float(value.detach()))
            running["grad_norm"].append(float(grad_norm))
            if step % log_every == 0:
                row = {
                    key: sum(values) / len(values)
                    for key, values in running.items()
                    if values
                }
                row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                print(f"LoRA visual projector step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()
            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_visual_projector_state(
                        path,
                        step=step,
                        projector=projector,
                        optimizer=optimizer,
                        config=cfg,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(style_ids),
        "reader_frozen": True,
        "oracle_codes_frozen": True,
        "projector_parameters": sum(
            parameter.numel() for parameter in projector.parameters()
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_oracle_visual_projector(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_oracle_bootstrap"]["visual_projector"]
    cfg["output_directory"] = "lora_oracle_visual_projector_smoke"
    cfg["resume"] = False
    cfg["checkpoint_every"] = 1
    cfg["wandb"]["enabled"] = False
    return train_lora_oracle_visual_projector(
        effective, destination, steps_override=3
    )


def train_lora_oracle_functional_projector(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Align visual codes through the frozen, oracle-trained connector.

    Code-space regression is only a weak regularizer. The primary target is
    the final centered LoRA effect after a frozen Anima forward, so the
    connector's already verified oracle mapping is never relearned or erased.
    """

    root_cfg = copy.deepcopy(config["lora_oracle_bootstrap"])
    cfg = dict(root_cfg["functional_projector"])
    steps = int(steps_override if steps_override is not None else cfg["steps"])
    device = str(root_cfg["training"].get("device", "cuda"))
    seed = int(root_cfg.get("seed", 20260823)) ^ 0x46554E43
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(int(cfg.get("cpu_threads", 16)))

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=True)
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    adapter = _build_style_adapter(detail_cfg).to(device)
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("Functional projector requires the separated adapter")
    attach_same_q_style_adapter(anima, adapter)

    source = torch.load(
        destination / str(cfg["source_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    reader.load_state_dict(source["reader"], strict=True)
    adapter.load_state_dict(source["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("artist_only")
    reader.requires_grad_(False).eval()
    adapter.requires_grad_(False).eval()
    oracle_codes = source["oracle_codes"].to(device=device, dtype=torch.bfloat16)

    projector_cfg = dict(root_cfg["visual_projector"])
    projector = OracleVisualProjector(
        dim=int(projector_cfg.get("dim", 1024)),
        slots=int(projector_cfg.get("slots", 28)),
        heads=int(projector_cfg.get("heads", 16)),
        ff_dim=int(projector_cfg.get("ff_dim", 2048)),
        bottleneck_dim=int(projector_cfg.get("bottleneck_dim", 256)),
    ).to(device)
    projector_source = torch.load(
        destination / str(cfg["projector_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    projector.load_state_dict(projector_source["projector"], strict=True)
    projected_reader = _ProjectedReader(reader, projector)

    bank = FunctionalLoRATeacherBank(destination / str(root_cfg["teacher_cache"]))
    single_ids = list(bank.by_kind["single"])
    style_ids = [str(bank.mixtures[index]["style_ids"][0]) for index in single_ids]
    reference_images = int(cfg.get("materialized_reference_images", 8))
    loader_kwargs = {
        "split": "train",
        "style_ids": style_ids,
        "batch_size": len(style_ids),
        "references": reference_images,
        "token_lru_shards": int(cfg.get(
            "token_lru_shards",
            root_cfg["training"].get("token_lru_shards", 8),
        )),
        "strict_style_ids": True,
    }
    human_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["human_reference_cache"]),
        seed=seed ^ 0x48554D41,
        **loader_kwargs,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["synthetic_reference_cache"]),
        seed=seed ^ 0x53594E54,
        **loader_kwargs,
    )
    human_codes, reference_counts = _materialize_reader_code_bank(
        reader,
        human_loader,
        style_ids,
        reference_images=reference_images,
        seed=seed ^ 0x11111111,
        device=device,
    )
    synthetic_codes, synthetic_reference_counts = _materialize_reader_code_bank(
        reader,
        synthetic_loader,
        style_ids,
        reference_images=reference_images,
        seed=seed ^ 0x22222222,
        device=device,
    )
    if not torch.equal(reference_counts, synthetic_reference_counts):
        raise RuntimeError("Human and synthetic Reader banks disagree on views")
    visual_code_banks = torch.stack((human_codes, synthetic_codes), dim=0)
    del human_codes, synthetic_codes

    native_steps = int(cfg.get("native_pretrain_steps", 0))
    native_bank = None
    native_visual_codes = None
    native_teacher_indices: list[int] = []
    if native_steps > 0:
        native_bank = NativeCenteredTeacherBank.load(
            config, destination, config_key="dual_domain_native_teacher"
        )
        signature = native_bank.summary["signature"]
        native_all_ids = [str(value) for value in signature["style_ids"]]
        native_splits = [
            str(value) for value in signature.get(
                "splits", ["train"] * len(native_all_ids)
            )
        ]
        native_teacher_indices = [
            index
            for index, split in enumerate(native_splits)
            if split == "train"
        ]
        native_artist_limit = int(cfg.get("native_artist_limit", 0))
        if native_artist_limit > 0:
            native_teacher_indices = native_teacher_indices[:native_artist_limit]
        native_style_ids = [native_all_ids[index] for index in native_teacher_indices]
        native_loader = CachedTeacherReferenceLoader(
            destination / str(root_cfg["human_reference_cache"]),
            split="train",
            style_ids=native_style_ids,
            batch_size=int(cfg.get("native_materialize_chunk_size", 64)),
            references=reference_images,
            seed=seed ^ 0x4E415449,
            token_lru_shards=int(cfg.get("token_lru_shards", 512)),
            ram_resident_tokens=True,
            ram_preload_workers=int(cfg.get("native_preload_workers", 16)),
            strict_style_ids=True,
        )
        native_visual_codes, native_reference_counts = _materialize_reader_code_bank(
            reader,
            native_loader,
            native_style_ids,
            reference_images=reference_images,
            seed=seed ^ 0x33333333,
            device=device,
            style_chunk_size=int(cfg.get("native_materialize_chunk_size", 128)),
        )
        if not torch.equal(reference_counts, native_reference_counts):
            raise RuntimeError("Native and LoRA Reader banks disagree on views")

    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=bool(cfg.get("fused_adamw", True)),
    )
    base_lr = float(cfg["learning_rate"])
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(cfg.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        projector.load_state_dict(resume["projector"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])

    wandb_run = None
    wandb_cfg = dict(cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "lora-oracle-functional-projector-v1")),
            id=str(wandb_cfg.get("id", "lora-oracle-functional-projector-v1")),
            resume="allow",
            config={"lora_oracle_functional_projector": cfg},
        )

    fixed = load_dual_query_external_sample(config, destination)
    batch_rows = int(cfg["batch_rows"])
    warmup = int(cfg.get("warmup_steps", 50))
    code_weight = float(cfg.get("code_alignment_weight", 0.10))
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            optimizer.param_groups[0]["lr"] = base_lr * min(
                1.0, step / max(1, warmup)
            )
            optimizer.zero_grad(set_to_none=True)
            rng = random.Random(seed + step * 1_000_003)
            is_native = step <= native_steps
            available_rows = (
                len(native_teacher_indices) if is_native else len(single_ids)
            )
            positions = rng.sample(range(available_rows), batch_rows)
            position_index = torch.tensor(positions, device=device, dtype=torch.long)
            view_index = torch.tensor(
                [rng.randrange(visual_code_banks.shape[2]) for _ in positions],
                device=device,
                dtype=torch.long,
            )
            if is_native:
                if native_visual_codes is None or native_bank is None:
                    raise RuntimeError("Native pretraining bank was not initialized")
                domain_index = -1
                visual_codes = native_visual_codes[position_index, view_index]
            else:
                domain_index = step % 2
                visual_codes = visual_code_banks[
                    domain_index, position_index, view_index
                ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                projected = projector(visual_codes)

            functional_bank = native_bank.tensors if is_native else bank.base
            content_index = step % int(functional_bank["noisy_inputs"].shape[0])
            timestep_index = (
                step // int(functional_bank["noisy_inputs"].shape[0])
            ) % int(functional_bank["noisy_inputs"].shape[1])
            noisy = functional_bank["noisy_inputs"][content_index, timestep_index].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            base = functional_bank["base_predictions"][content_index, timestep_index].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            context = functional_bank["base_context"][content_index : content_index + 1].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            timestep = functional_bank["timesteps"][timestep_index].to(
                device=device, dtype=torch.bfloat16
            )
            if is_native:
                teacher_rows = [native_teacher_indices[index] for index in positions]
                teacher = native_bank.tensors["centered_teacher"][
                    teacher_rows, content_index, timestep_index
                ].to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                teacher_rows = [single_ids[index] for index in positions]
                teacher = bank.effects[
                    teacher_rows, content_index, timestep_index
                ].to(device=device, dtype=torch.float32, non_blocking=True)
            student = _controlled_style_context_forward(
                anima, adapter, projected, noisy, base, context, timestep, device
            )
            functional_loss, functional_metrics = _artist_centered_oracle_objective(
                student, teacher, dict(cfg.get("functional_loss_weights", {}))
            )
            if is_native:
                visual_scale = visual_codes.float().square().mean(
                    dim=(1, 2), keepdim=True
                ).sqrt().clamp_min(1e-4)
                identity_loss = (
                    (projected.float() - visual_codes.float()) / visual_scale
                ).square().mean()
                regularizer = float(cfg.get("native_identity_weight", 0.001)) * identity_loss
                code_metrics = {"identity_loss": identity_loss.detach()}
            else:
                code_loss, code_metrics = _oracle_code_alignment_objective(
                    projected,
                    oracle_codes[position_index],
                    visual_codes,
                    dict(cfg.get("code_loss_weights", {})),
                )
                regularizer = code_weight * code_loss
            loss = functional_loss + regularizer
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                projector.parameters(),
                float(cfg.get("max_grad_norm", 1.0)),
                foreach=True,
            )
            optimizer.step()

            metrics = {
                "loss": loss.detach(),
                "functional_loss": functional_loss.detach(),
                "code_regularizer_weighted_loss": regularizer.detach(),
                **{
                    key: value
                    for key, value in functional_metrics.items()
                    if key != "loss"
                },
                **{f"code/{key}": value for key, value in code_metrics.items()},
                "references": reference_counts[view_index].mean(),
                "domain_is_human": torch.tensor(
                    float(domain_index != 0), device=device
                ),
                "teacher_is_native": torch.tensor(float(is_native), device=device),
            }
            for key, value in metrics.items():
                running[key].append(float(value.detach()))
            running["grad_norm"].append(float(grad_norm))

            if step % int(cfg.get("log_every", 10)) == 0:
                row = {
                    key: sum(values) / len(values)
                    for key, values in running.items()
                    if values
                }
                row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                print(f"LoRA functional projector step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()
            if step % int(cfg.get("checkpoint_every", 250)) == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_visual_projector_state(
                        path,
                        step=step,
                        projector=projector,
                        optimizer=optimizer,
                        config=cfg,
                    )
            if int(cfg.get("sample_every", 250)) > 0 and step % int(cfg["sample_every"]) == 0:
                sample = _generate_fixed_reference_sample(
                    fixed,
                    config,
                    destination,
                    anima,
                    projected_reader,
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
                            sample["sheet"], caption=f"step {step}, functional projector"
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
        "reader_frozen": True,
        "adapter_frozen": True,
        "anima_frozen": True,
        "projector_trainable": True,
        "code_alignment_weight": code_weight,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_oracle_functional_projector(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_oracle_bootstrap"]["functional_projector"]
    cfg["output_directory"] = "lora_oracle_functional_projector_smoke"
    cfg["resume"] = False
    cfg["checkpoint_every"] = 1
    cfg["sample_every"] = 0
    cfg["native_artist_limit"] = 64
    cfg["native_materialize_chunk_size"] = 32
    cfg["wandb"]["enabled"] = False
    return train_lora_oracle_functional_projector(
        effective, destination, steps_override=3
    )


def _compose_lora_mixture_contexts(
    projector: OracleVisualProjector,
    visual_banks: torch.Tensor,
    oracle_codes: torch.Tensor,
    mixture_rows: list[dict[str, Any]],
    *,
    domain_index: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose visual and oracle contexts with the cached LoRA mixture weights."""

    visual_parts: list[torch.Tensor] = []
    oracle_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    reference_counts: list[float] = []
    for row in mixture_rows:
        components = [int(value) for value in row["components"]]
        weights = torch.tensor(
            [float(value) for value in row["weights"]],
            device=visual_banks.device,
            dtype=torch.float32,
        )
        view_indices = [rng.randrange(visual_banks.shape[2]) for _ in components]
        component_index = torch.tensor(
            components, device=visual_banks.device, dtype=torch.long
        )
        view_index = torch.tensor(
            view_indices, device=visual_banks.device, dtype=torch.long
        )
        raw = visual_banks[domain_index, component_index, view_index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            projected = projector(raw)
        shape = (-1,) + (1,) * (projected.ndim - 1)
        blend = weights.reshape(shape)
        visual_parts.append((projected.float() * blend).sum(dim=0))
        oracle_parts.append(
            (oracle_codes[component_index].float() * blend).sum(dim=0)
        )
        raw_parts.append((raw.float() * blend).sum(dim=0))
        reference_counts.append(float(len(components)))
    return (
        torch.stack(visual_parts).to(dtype=torch.bfloat16),
        torch.stack(oracle_parts).to(dtype=torch.bfloat16),
        torch.stack(raw_parts).to(dtype=torch.bfloat16),
        torch.tensor(reference_counts, device=visual_banks.device),
    )


def _cross_view_artist_objective(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep broad visual artists separable without imposing native tag directions."""

    if left.shape != right.shape or left.shape[0] < 2:
        raise ValueError("Cross-view artist loss needs matching multi-artist batches")
    common = torch.cat((left.float(), right.float()), dim=0).mean(
        dim=0, keepdim=True
    )
    left_unit = F.normalize((left.float() - common).flatten(1), dim=1)
    right_unit = F.normalize((right.float() - common).flatten(1), dim=1)
    logits = left_unit @ right_unit.t() / temperature
    labels = torch.arange(left.shape[0], device=left.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    positive = logits.diagonal() * temperature
    wrong = logits.masked_fill(
        torch.eye(left.shape[0], device=left.device, dtype=torch.bool),
        torch.finfo(logits.dtype).min,
    ).max(dim=1).values * temperature
    return loss, {
        "loss": loss.detach(),
        "accuracy": (logits.argmax(dim=1) == labels).float().mean().detach(),
        "positive_cosine": positive.mean().detach(),
        "hardest_wrong_cosine": wrong.mean().detach(),
        "cosine_gap": (positive - wrong).mean().detach(),
    }


def _cross_view_functional_objective(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    temperature: float,
    target_rms: torch.Tensor,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Separate artists after the frozen Anima path, not only in token space.

    The ordinary visual cross-view loss can be satisfied by projector
    directions that the style connector ignores.  Here both views are measured
    after the same controlled Anima forward.  A weak LoRA-derived magnitude
    anchor prevents normalized InfoNCE from accepting an infinitesimal effect.
    """

    contrastive, metrics = _cross_view_artist_objective(
        left, right, temperature=temperature
    )
    values = torch.cat((left.float(), right.float()), dim=0)
    centered = values - values.mean(dim=0, keepdim=True)
    reduce_dims = tuple(range(1, centered.ndim))
    effect_rms = (centered.square().mean(dim=reduce_dims) + 1e-12).sqrt()
    scale = target_rms.detach().float().clamp_min(1e-4)
    magnitude = F.smooth_l1_loss(
        (effect_rms / scale).clamp_min(1e-4).log().clamp(-4, 4),
        torch.zeros_like(effect_rms),
        beta=0.10,
    )
    total = contrastive + float(magnitude_weight) * magnitude
    return total, {
        **metrics,
        "magnitude_loss": magnitude.detach(),
        "effect_to_lora_teacher_rms": (effect_rms / scale).mean().detach(),
    }


def _functional_effect_fingerprints(effects: torch.Tensor) -> torch.Tensor:
    """Build compact cosine fingerprints from cached functional effects."""

    if effects.ndim < 3 or effects.shape[0] < 2:
        raise ValueError("Functional fingerprints need at least two effects")
    values = effects.float()
    if values.ndim >= 5:
        prefix = values.shape[:-3]
        channels, height, width = values.shape[-3:]
        values = F.adaptive_avg_pool2d(
            values.reshape(-1, channels, height, width), (4, 4)
        ).reshape(*prefix, channels, 4, 4)
    values = values.flatten(1)
    values = values - values.mean(dim=0, keepdim=True)
    return F.normalize(values, dim=1)


def _sample_diverse_functional_batch(
    candidates: list[int],
    similarity: torch.Tensor,
    batch_rows: int,
    *,
    rng: random.Random,
    pool_size: int,
) -> tuple[list[int], float]:
    """Greedily spread a random candidate pool in cached effect space."""

    if similarity.shape != (len(candidates), len(candidates)):
        raise ValueError("Similarity matrix does not match candidate list")
    if not 1 <= batch_rows <= len(candidates):
        raise ValueError("batch_rows must fit the candidate list")
    pool_count = min(len(candidates), max(batch_rows, int(pool_size)))
    pool = rng.sample(range(len(candidates)), pool_count)
    selected = [pool.pop(rng.randrange(len(pool)))]
    while len(selected) < batch_rows:
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        best_position = min(
            pool,
            key=lambda position: float(
                similarity[position].index_select(0, selected_tensor).max()
            ),
        )
        selected.append(best_position)
        pool.remove(best_position)
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    selected_similarity = similarity.index_select(
        0, selected_tensor
    ).index_select(1, selected_tensor)
    if batch_rows > 1:
        mask = ~torch.eye(batch_rows, dtype=torch.bool)
        mean_similarity = float(selected_similarity[mask].mean())
    else:
        mean_similarity = 1.0
    return [candidates[position] for position in selected], mean_similarity


def train_lora_oracle_joint_manifold(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Jointly align visual styles while replaying the verified oracle mapping."""

    root_cfg = copy.deepcopy(config["lora_oracle_bootstrap"])
    cfg = dict(root_cfg["joint_manifold"])
    steps = int(steps_override if steps_override is not None else cfg["steps"])
    device = str(root_cfg["training"].get("device", "cuda"))
    seed = int(root_cfg.get("seed", 20260823)) ^ 0x4A4F494E
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(int(cfg.get("cpu_threads", 16)))

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=True)
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    adapter = _build_style_adapter(detail_cfg).to(device)
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("Joint oracle manifold training requires the separated adapter")
    attach_same_q_style_adapter(anima, adapter)

    source = torch.load(
        destination / str(cfg["source_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    reader.load_state_dict(source["reader"], strict=True)
    adapter.load_state_dict(source["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("artist_only")
    reader.requires_grad_(False).eval()
    for parameter in adapter.common_parameters():
        parameter.requires_grad_(False)
    oracle_codes = source["oracle_codes"].to(device=device, dtype=torch.bfloat16)

    projector_cfg = dict(root_cfg["visual_projector"])
    projector = OracleVisualProjector(
        dim=int(projector_cfg.get("dim", 1024)),
        slots=int(projector_cfg.get("slots", 28)),
        heads=int(projector_cfg.get("heads", 16)),
        ff_dim=int(projector_cfg.get("ff_dim", 2048)),
        bottleneck_dim=int(projector_cfg.get("bottleneck_dim", 256)),
    ).to(device)
    projector_source = torch.load(
        destination / str(cfg["projector_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    projector.load_state_dict(projector_source["projector"], strict=True)
    projected_reader = _ProjectedReader(reader, projector)

    bank = FunctionalLoRATeacherBank(destination / str(root_cfg["teacher_cache"]))
    single_ids = list(bank.by_kind["single"])
    style_ids = [str(bank.mixtures[index]["style_ids"][0]) for index in single_ids]
    reference_images = int(cfg.get("materialized_reference_images", 8))
    loader_kwargs = {
        "split": "train",
        "style_ids": style_ids,
        "batch_size": len(style_ids),
        "references": reference_images,
        "token_lru_shards": int(cfg.get("token_lru_shards", 512)),
        "strict_style_ids": True,
    }
    human_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["human_reference_cache"]),
        seed=seed ^ 0x48554D41,
        **loader_kwargs,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["synthetic_reference_cache"]),
        seed=seed ^ 0x53594E54,
        **loader_kwargs,
    )
    human_codes, reference_counts = _materialize_reader_code_bank(
        reader, human_loader, style_ids,
        reference_images=reference_images, seed=seed ^ 0x11111111, device=device,
    )
    synthetic_codes, synthetic_reference_counts = _materialize_reader_code_bank(
        reader, synthetic_loader, style_ids,
        reference_images=reference_images, seed=seed ^ 0x22222222, device=device,
    )
    if not torch.equal(reference_counts, synthetic_reference_counts):
        raise RuntimeError("Human and synthetic Reader banks disagree on views")
    visual_banks = torch.stack((human_codes, synthetic_codes), dim=0)
    del human_codes, synthetic_codes

    native_cfg = dict(config["dual_domain_native_teacher"])
    native_root = destination / str(native_cfg["output_directory"])
    native_summary = json.loads(
        (native_root / "summary.json").read_text(encoding="utf-8")
    )
    signature = native_summary["signature"]
    native_all_ids = [str(value) for value in signature["style_ids"]]
    native_splits = [
        str(value)
        for value in signature.get("splits", ["train"] * len(native_all_ids))
    ]
    native_style_ids = [
        style_id
        for style_id, split in zip(native_all_ids, native_splits, strict=True)
        if split == "train"
    ]
    native_limit = int(cfg.get("native_artist_limit", 0))
    if native_limit > 0:
        native_style_ids = native_style_ids[:native_limit]
    native_loader = CachedTeacherReferenceLoader(
        destination / str(root_cfg["human_reference_cache"]),
        split="train",
        style_ids=native_style_ids,
        batch_size=int(cfg.get("native_materialize_chunk_size", 128)),
        references=reference_images,
        seed=seed ^ 0x4E415449,
        token_lru_shards=int(cfg.get("token_lru_shards", 512)),
        ram_resident_tokens=True,
        ram_preload_workers=int(cfg.get("native_preload_workers", 16)),
        strict_style_ids=True,
    )
    native_codes, native_reference_counts = _materialize_reader_code_bank(
        reader, native_loader, native_style_ids,
        reference_images=reference_images, seed=seed ^ 0x33333333,
        device=device,
        style_chunk_size=int(cfg.get("native_materialize_chunk_size", 128)),
    )
    if not torch.equal(reference_counts, native_reference_counts):
        raise RuntimeError("Native and LoRA Reader banks disagree on views")

    groups = [
        {"params": projector.parameters(), "lr": float(cfg["projector_learning_rate"]), "name": "projector"},
        {"params": adapter.shared_parameters(), "lr": float(cfg["shared_learning_rate"]), "name": "shared_kv"},
        {"params": adapter.delta_parameters(), "lr": float(cfg["delta_learning_rate"]), "name": "block_delta"},
        {"params": adapter.mixing_parameters(), "lr": float(cfg["mix_learning_rate"]), "name": "base_mix", "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=bool(cfg.get("fused_adamw", True)),
    )
    base_lrs = {str(group["name"]): float(group["lr"]) for group in groups}
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(cfg.get("resume", True)) and state_path.exists():
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(resume["reader"], strict=True)
        adapter.load_state_dict(resume["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        adapter.set_bootstrap_phase("artist_only")
        projector.load_state_dict(resume["projector"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])

    wandb_run = None
    wandb_cfg = dict(cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "lora-oracle-joint-manifold-v1")),
            id=str(wandb_cfg.get("id", "lora-oracle-joint-manifold-v1")),
            resume="allow",
            config={"lora_oracle_joint_manifold": cfg},
        )

    fixed = load_dual_query_external_sample(config, destination)
    batch_rows = int(cfg.get("batch_rows", 8))
    native_batch_rows = int(cfg.get("native_batch_rows", 8))
    warmup = int(cfg.get("warmup_steps", 50))
    replay_weight = float(cfg.get("oracle_replay_weight", 0.5))
    code_weight = float(cfg.get("code_alignment_weight", 0.05))
    native_weight = float(cfg.get("native_cross_view_weight", 0.10))
    native_functional_weight = float(
        cfg.get("native_functional_cross_view_weight", 0.0)
    )
    native_functional_every = max(
        1, int(cfg.get("native_functional_cross_view_every", 1))
    )
    native_functional_rows = int(
        cfg.get("native_functional_cross_view_rows", native_batch_rows)
    )
    if not 2 <= native_functional_rows <= native_batch_rows:
        raise ValueError(
            "native_functional_cross_view_rows must be in [2, native_batch_rows]"
        )
    native_functional_magnitude_weight = float(
        cfg.get("native_functional_magnitude_weight", 0.10)
    )
    temperature = float(cfg.get("native_temperature", 0.10))
    loss_weights = dict(cfg.get("functional_loss_weights", {}))
    code_weights = dict(cfg.get("code_loss_weights", {}))
    categories = ("single", "pair", "triple")
    diverse_pool_size = int(cfg.get("functional_diverse_sampling_pool", 0))
    functional_similarities: dict[str, torch.Tensor] = {}
    if diverse_pool_size > 0:
        for category in categories:
            category_indices = bank.by_kind[category]
            fingerprints = _functional_effect_fingerprints(
                bank.effects[category_indices]
            )
            functional_similarities[category] = fingerprints @ fingerprints.t()
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            lr_scale = min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = base_lrs[str(group["name"])] * lr_scale
            optimizer.zero_grad(set_to_none=True)
            rng = random.Random(seed + step * 1_000_003)
            category = categories[(step - 1) % len(categories)]
            candidates = bank.by_kind[category]
            if diverse_pool_size > 0:
                mixture_indices, teacher_batch_similarity = (
                    _sample_diverse_functional_batch(
                        candidates,
                        functional_similarities[category],
                        batch_rows,
                        rng=rng,
                        pool_size=diverse_pool_size,
                    )
                )
            else:
                mixture_indices = rng.sample(candidates, batch_rows)
                teacher_batch_similarity = float("nan")
            mixture_rows = [bank.mixtures[index] for index in mixture_indices]
            domain_index = step % 2
            visual_context, oracle_context, raw_context, component_counts = (
                _compose_lora_mixture_contexts(
                    projector, visual_banks, oracle_codes, mixture_rows,
                    domain_index=domain_index, rng=rng,
                )
            )
            content_index = step % int(bank.base["noisy_inputs"].shape[0])
            timestep_index = (
                step // int(bank.base["noisy_inputs"].shape[0])
            ) % int(bank.base["noisy_inputs"].shape[1])
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
                mixture_indices, content_index, timestep_index
            ].to(device=device, dtype=torch.float32, non_blocking=True)

            visual_student = _controlled_style_context_forward(
                anima, adapter, visual_context, noisy, base, context, timestep, device
            )
            visual_loss, visual_metrics = _artist_centered_oracle_objective(
                visual_student, teacher, loss_weights
            )
            code_loss, code_metrics = _oracle_code_alignment_objective(
                visual_context, oracle_context, raw_context, code_weights
            )
            visual_total = visual_loss + code_weight * code_loss
            visual_total.backward()
            del visual_student, visual_total

            replay_student = _controlled_style_context_forward(
                anima, adapter, oracle_context, noisy, base, context, timestep, device
            )
            replay_loss, replay_metrics = _artist_centered_oracle_objective(
                replay_student, teacher, loss_weights
            )
            (replay_weight * replay_loss).backward()
            del replay_student

            native_positions = rng.sample(range(len(native_style_ids)), native_batch_rows)
            native_index = torch.tensor(
                native_positions, device=device, dtype=torch.long
            )
            left_view = torch.tensor(
                [rng.randrange(native_codes.shape[1]) for _ in native_positions],
                device=device, dtype=torch.long,
            )
            right_values = []
            for value in left_view.tolist():
                other = rng.randrange(native_codes.shape[1] - 1)
                right_values.append(other + int(other >= value))
            right_view = torch.tensor(right_values, device=device, dtype=torch.long)
            native_raw = torch.cat((
                native_codes[native_index, left_view],
                native_codes[native_index, right_view],
            ))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                native_projected = projector(native_raw)
            native_left, native_right = native_projected.chunk(2)
            native_loss, native_metrics = _cross_view_artist_objective(
                native_left, native_right, temperature=temperature
            )
            (native_weight * native_loss).backward()

            native_functional_metrics: dict[str, torch.Tensor] = {}
            if (
                native_functional_weight > 0
                and step % native_functional_every == 0
            ):
                functional_context = torch.cat((
                    native_left[:native_functional_rows],
                    native_right[:native_functional_rows],
                ))
                native_functional_student = _controlled_style_context_forward(
                    anima,
                    adapter,
                    functional_context,
                    noisy,
                    base,
                    context,
                    timestep,
                    device,
                )
                native_functional_left, native_functional_right = (
                    native_functional_student.chunk(2)
                )
                teacher_centered = teacher - teacher.mean(dim=0, keepdim=True)
                teacher_target_rms = (
                    teacher_centered.square().mean().sqrt().clamp_min(1e-4)
                )
                native_functional_loss, native_functional_metrics = (
                    _cross_view_functional_objective(
                        native_functional_left,
                        native_functional_right,
                        temperature=temperature,
                        target_rms=teacher_target_rms,
                        magnitude_weight=native_functional_magnitude_weight,
                    )
                )
                (native_functional_weight * native_functional_loss).backward()
                del native_functional_student

            parameters = [
                parameter for group in optimizer.param_groups for parameter in group["params"]
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg.get("max_grad_norm", 1.0)), foreach=True
            )
            optimizer.step()
            metrics: dict[str, torch.Tensor] = {
                "loss": (
                    visual_loss.detach()
                    + code_weight * code_loss.detach()
                    + replay_weight * replay_loss.detach()
                    + native_weight * native_loss.detach()
                ),
                "category": torch.tensor(float(categories.index(category)), device=device),
                "domain_is_human": torch.tensor(float(domain_index == 0), device=device),
                "mixture_components": component_counts.mean(),
                "teacher_batch_effect_cosine": torch.tensor(
                    teacher_batch_similarity, device=device
                ),
                **{f"visual/{key}": value for key, value in visual_metrics.items()},
                **{f"replay/{key}": value for key, value in replay_metrics.items()},
                **{f"code/{key}": value for key, value in code_metrics.items()},
                **{f"native/{key}": value for key, value in native_metrics.items()},
                **{
                    f"native_functional/{key}": value
                    for key, value in native_functional_metrics.items()
                },
            }
            for key, value in metrics.items():
                running[key].append(float(value.detach()))
            running["grad_norm"].append(float(grad_norm))
            if step % int(cfg.get("log_every", 10)) == 0:
                row = {
                    key: sum(values) / len(values)
                    for key, values in running.items() if values
                }
                for group in optimizer.param_groups:
                    row[f"lr/{group['name']}"] = float(group["lr"])
                print(f"LoRA joint manifold step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in row.items()}, step=step
                    )
                running.clear()
            if step % int(cfg.get("checkpoint_every", 250)) == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_visual_bridge_state(
                        path, step=step, reader=reader, adapter=adapter,
                        oracle_codes=oracle_codes, projector=projector,
                        optimizer=optimizer, config=cfg,
                    )
            if int(cfg.get("sample_every", 250)) > 0 and step % int(cfg["sample_every"]) == 0:
                sample = _generate_fixed_reference_sample(
                    fixed, config, destination, anima, projected_reader, adapter,
                    output, device, step, component_mode="artist_only",
                    strengths_override=[1.0], sample_group="fixed_reference_samples",
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "val/functional/fixed_reference": wandb.Image(
                            sample["sheet"], caption=f"step {step}, joint manifold"
                        )
                    }, step=step)
    finally:
        adapter.clear_style_tokens()
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps": steps,
        "start_step": start_step,
        "lora_artists": len(style_ids),
        "native_visual_artists": len(native_style_ids),
        "reader_frozen": True,
        "common_frozen_and_bypassed": True,
        "anima_frozen": True,
        "projector_and_artist_connector_trainable": True,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_oracle_joint_manifold(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_oracle_bootstrap"]["joint_manifold"]
    cfg["output_directory"] = "lora_oracle_joint_manifold_smoke"
    cfg["resume"] = False
    cfg["checkpoint_every"] = 1
    cfg["sample_every"] = 0
    cfg["native_artist_limit"] = 64
    cfg["native_materialize_chunk_size"] = 32
    cfg["wandb"]["enabled"] = False
    return train_lora_oracle_joint_manifold(
        effective, destination, steps_override=3
    )
