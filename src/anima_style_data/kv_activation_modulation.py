"""Direct native text K/V modulation distilled from K/V-only LoRA teachers.

The earlier oracle experiments asked a short, separate style-attention branch to
reproduce a teacher which changes the native 512-token text K/V projections.
Those are different function classes.  This module first runs a cheaper and
stricter capacity gate: frozen visual Reader codes must predict the actual
per-block low-rank K/V activation deltas produced by the teacher LoRAs.

No Anima forward is needed while fitting the modulator.  Teacher deltas are
materialized online from the cached text contexts and the small rank-16 LoRA
factors, so the experiment neither duplicates a multi-terabyte activation bank
nor optimizes a noisy final-flow proxy.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .io import write_json
from .lora_functional_distillation import _load_lora_plan, _weight_paths


_LORA_KEY = re.compile(
    r"lora_unet_blocks_(\d+)_cross_attn_([kv])_proj\."
    r"(lora_down\.weight|lora_up\.weight|alpha)"
)


def canonicalize_lora_factors(
    down: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered, balanced factors with the exact same weight delta.

    Independent LoRA training leaves rank permutation, sign, and reciprocal
    scale arbitrary.  Directly supervising those coordinates would force a
    visual model to predict training accidents rather than the represented
    function.  Thin QR decompositions reduce the expensive SVD to the small
    rank-by-rank core and produce singular-value ordered canonical factors.
    """

    q_up, r_up = torch.linalg.qr(up.float(), mode="reduced")
    q_down, r_down = torch.linalg.qr(down.float().t(), mode="reduced")
    left, singular, right_h = torch.linalg.svd(
        r_up @ r_down.t(), full_matrices=False
    )
    root = singular.clamp_min(0).sqrt()
    canonical_up = (q_up @ left) * root[None]
    canonical_down = root[:, None] * (right_h @ q_down.t())
    pivot = canonical_down.gather(
        -1, canonical_down.abs().argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
    canonical_down = canonical_down * sign[:, None]
    canonical_up = canonical_up * sign[None]
    return canonical_down, canonical_up


def canonicalize_lora_factor_bank(
    down: torch.Tensor,
    up: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched canonicalization for the complete artist/block/KV bank."""

    leading = down.shape[:-2]
    if up.shape[:-2] != leading or down.shape[-2] != up.shape[-1]:
        raise ValueError("LoRA factor-bank shapes disagree")
    flat_down = down.reshape(-1, *down.shape[-2:])
    flat_up = up.reshape(-1, *up.shape[-2:])
    canonical_down: list[torch.Tensor] = []
    canonical_up: list[torch.Tensor] = []
    for start in range(0, flat_down.shape[0], chunk_size):
        stop = min(flat_down.shape[0], start + chunk_size)
        down_chunk = flat_down[start:stop].float()
        up_chunk = flat_up[start:stop].float()
        q_up, r_up = torch.linalg.qr(up_chunk, mode="reduced")
        q_down, r_down = torch.linalg.qr(
            down_chunk.transpose(-1, -2), mode="reduced"
        )
        left, singular, right_h = torch.linalg.svd(
            r_up @ r_down.transpose(-1, -2), full_matrices=False
        )
        root = singular.clamp_min(0).sqrt()
        canonical_up_chunk = (q_up @ left) * root.unsqueeze(-2)
        canonical_down_chunk = root.unsqueeze(-1) * (
            right_h @ q_down.transpose(-1, -2)
        )
        pivot = canonical_down_chunk.gather(
            -1,
            canonical_down_chunk.abs().argmax(dim=-1, keepdim=True),
        ).squeeze(-1)
        sign = torch.where(
            pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot)
        )
        canonical_down.append(canonical_down_chunk * sign.unsqueeze(-1))
        canonical_up.append(canonical_up_chunk * sign.unsqueeze(-2))
    return (
        torch.cat(canonical_down).reshape(*leading, *down.shape[-2:]),
        torch.cat(canonical_up).reshape(*leading, *up.shape[-2:]),
    )


def load_kv_lora_factor_bank(
    root: Path,
    *,
    blocks: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Load K/V-only teacher factors as ``[artist, block, kind, ...]``.

    ``down`` has shape ``[A, B, 2, R, C]`` and ``up`` has shape
    ``[A, B, 2, O, R]``.  The LoRA alpha/rank multiplier is folded into the up
    factor, making online activation targets a pair of plain einsums.
    """

    plans = _load_lora_plan(root)
    paths = _weight_paths(root, plans)
    artist_ids = [plan.style_id for plan in plans]
    all_down: list[torch.Tensor] = []
    all_up: list[torch.Tensor] = []
    expected: tuple[int, int, int] | None = None
    for path in paths:
        tensors = load_file(path, device="cpu")
        parsed: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for key, value in tensors.items():
            match = _LORA_KEY.fullmatch(key)
            if match is None:
                raise RuntimeError(f"Unexpected tensor in K/V-only LoRA: {key}")
            block = int(match.group(1))
            kind = 0 if match.group(2) == "k" else 1
            parsed.setdefault((block, kind), {})[match.group(3)] = value
        if set(parsed) != {
            (block, kind) for block in range(blocks) for kind in range(2)
        }:
            raise RuntimeError(f"Incomplete K/V-only LoRA tensor set: {path}")
        artist_down: list[torch.Tensor] = []
        artist_up: list[torch.Tensor] = []
        for block in range(blocks):
            block_down: list[torch.Tensor] = []
            block_up: list[torch.Tensor] = []
            for kind in range(2):
                values = parsed[(block, kind)]
                down = values["lora_down.weight"].float()
                up = values["lora_up.weight"].float()
                rank = int(down.shape[0])
                scale = float(values["alpha"].float()) / rank
                if expected is None:
                    expected = (rank, int(down.shape[1]), int(up.shape[0]))
                if (rank, int(down.shape[1]), int(up.shape[0])) != expected:
                    raise RuntimeError("K/V-only LoRA factor dimensions disagree")
                block_down.append(down)
                block_up.append(up * scale)
            artist_down.append(torch.stack(block_down))
            artist_up.append(torch.stack(block_up))
        all_down.append(torch.stack(artist_down))
        all_up.append(torch.stack(artist_up))
    return artist_ids, torch.stack(all_down), torch.stack(all_up)


def apply_kv_factors(
    context: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Apply per-row K/V low-rank factors to native text context.

    Args:
        context: ``[batch, tokens, context_dim]``.
        down: ``[batch, 2, rank, context_dim]``.
        up: ``[batch, 2, output_dim, rank]``.
    Returns:
        Raw pre-normalization K/V deltas, ``[batch, 2, tokens, output_dim]``.
    """

    hidden = torch.einsum("bnc,btrc->btnr", context, down)
    return torch.einsum("btnr,btor->btno", hidden, up)


class NativeKVFactorModulator(nn.Module):
    """Generate block-specific low-rank K/V factors from frozen style codes.

    The 28 Reader tokens remain a memory set.  For the requested Anima block,
    32 factor queries (K/V x rank-16) read the entire set and emit one down and
    one up vector each.  This preserves the exact algebra of a K/V-only LoRA
    without assuming a shared LoRA weight basis across artists.
    """

    def __init__(
        self,
        *,
        style_dim: int,
        blocks: int,
        rank: int,
        context_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        heads: int = 8,
        layers: int = 2,
        ff_dim: int = 2048,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.blocks = int(blocks)
        self.rank = int(rank)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.style_norm = nn.LayerNorm(style_dim)
        self.style_projection = nn.Linear(style_dim, hidden_dim, bias=False)
        self.block_embedding = nn.Embedding(blocks, hidden_dim)
        self.kind_embedding = nn.Embedding(2, hidden_dim)
        self.rank_embedding = nn.Embedding(rank, hidden_dim)
        self.factor_query = nn.Parameter(torch.empty(2, rank, hidden_dim))
        nn.init.normal_(self.factor_query, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.factor_mixer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.down_head = nn.Linear(hidden_dim, context_dim, bias=False)
        self.up_head = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.down_head.weight)
        nn.init.xavier_uniform_(self.up_head.weight)
        self.down_log_scale = nn.Linear(hidden_dim, 1)
        self.up_log_scale = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.down_log_scale.weight)
        nn.init.zeros_(self.down_log_scale.bias)
        nn.init.zeros_(self.up_log_scale.weight)
        nn.init.zeros_(self.up_log_scale.bias)
        self.register_buffer(
            "down_output_scale", torch.ones(blocks, 2, rank), persistent=True
        )
        self.register_buffer(
            "up_output_scale", torch.ones(blocks, 2, rank), persistent=True
        )

    def set_factor_scales(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> None:
        if down.shape != self.down_output_scale.shape:
            raise ValueError("Down factor-scale shape does not match the model")
        if up.shape != self.up_output_scale.shape:
            raise ValueError("Up factor-scale shape does not match the model")
        with torch.no_grad():
            self.down_output_scale.copy_(down.to(self.down_output_scale))
            self.up_output_scale.copy_(up.to(self.up_output_scale))

    def forward(
        self,
        style_codes: torch.Tensor,
        block_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(style_codes.shape[0])
        memory = self.style_projection(self.style_norm(style_codes))
        if not torch.is_tensor(block_index):
            block_index = torch.full(
                (batch,), int(block_index), device=style_codes.device,
                dtype=torch.long,
            )
        elif block_index.ndim == 0:
            block_index = block_index.expand(batch)
        block = self.block_embedding(block_index)[:, None, None, :]
        kind_ids = torch.arange(2, device=style_codes.device)
        rank_ids = torch.arange(self.rank, device=style_codes.device)
        queries = (
            self.factor_query[None]
            + block
            + self.kind_embedding(kind_ids)[None, :, None, :]
            + self.rank_embedding(rank_ids)[None, None, :, :]
        ).reshape(batch, 2 * self.rank, -1)
        attended, _ = self.cross_attention(
            queries, memory, memory, need_weights=False
        )
        factors = self.output_norm(
            self.factor_mixer(queries + attended)
        ).reshape(batch, 2, self.rank, -1)
        down_direction = F.normalize(
            self.down_head(factors).float(), dim=-1
        ).to(factors.dtype) * math.sqrt(self.context_dim)
        up_direction = F.normalize(
            self.up_head(factors).float(), dim=-1
        ).to(factors.dtype) * math.sqrt(self.output_dim)
        down_scale = self.down_output_scale[block_index].to(factors.dtype)
        up_scale = self.up_output_scale[block_index].to(factors.dtype)
        # The canonical bank supplies the absolute scale.  A bounded learned
        # correction captures real per-artist singular-value variation without
        # reopening the reciprocal A/B scale shortcut.
        down_scale = down_scale * torch.exp(
            0.5 * torch.tanh(self.down_log_scale(factors).squeeze(-1))
        )
        up_scale = up_scale * torch.exp(
            0.5 * torch.tanh(self.up_log_scale(factors).squeeze(-1))
        )
        down = down_direction * down_scale.unsqueeze(-1)
        up = (up_direction * up_scale.unsqueeze(-1)).transpose(-1, -2)
        return down, up


def kv_activation_objective(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    direction_weight: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale-independent regression plus explicit absolute magnitude matching."""

    student_f = student.float()
    teacher_f = teacher.float()
    dimensions = tuple(range(2, teacher_f.ndim))
    teacher_rms = (
        teacher_f.square().mean(dim=dimensions, keepdim=True) + 1e-12
    ).sqrt()
    student_rms = (
        student_f.square().mean(dim=dimensions, keepdim=True) + 1e-12
    ).sqrt()
    scale = teacher_rms.detach().clamp_min(1e-6)
    regression = F.smooth_l1_loss(
        student_f / scale,
        teacher_f / scale,
        beta=0.10,
    )
    cosine = F.cosine_similarity(
        student_f.flatten(2), teacher_f.flatten(2), dim=-1
    )
    direction = (1.0 - cosine).mean()
    log_ratio = (student_rms.clamp_min(1e-8).log() - scale.log()).abs().mean()
    loss = regression + float(direction_weight) * direction + float(
        magnitude_weight
    ) * log_ratio
    relative_error = (
        (student_f - teacher_f).square().mean(dim=dimensions).sqrt()
        / teacher_rms.flatten(2).squeeze(-1).clamp_min(1e-6)
    )
    return loss, {
        "loss": loss.detach(),
        "normalized_huber": regression.detach(),
        "direction_loss": direction.detach(),
        "cosine": cosine.mean().detach(),
        "magnitude_log_error": log_ratio.detach(),
        "student_to_teacher_rms": (
            student_rms / scale
        ).mean().detach(),
        "relative_rms_error": relative_error.mean().detach(),
        "k_cosine": cosine[:, 0].mean().detach(),
        "v_cosine": cosine[:, 1].mean().detach(),
    }


def kv_factor_objective(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    direction_weight: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress canonical K/V LoRA factors without bilinear scale ambiguity."""

    student_f = student.float()
    teacher_f = teacher.float()
    teacher_rms = (
        teacher_f.square().mean(dim=(-2, -1), keepdim=True) + 1e-12
    ).sqrt()
    student_rms = (
        student_f.square().mean(dim=(-2, -1), keepdim=True) + 1e-12
    ).sqrt()
    scale = teacher_rms.detach().clamp_min(1e-7)
    regression = F.smooth_l1_loss(
        student_f / scale,
        teacher_f / scale,
        beta=0.10,
    )
    cosine = F.cosine_similarity(
        student_f.flatten(2), teacher_f.flatten(2), dim=-1
    )
    direction = (1.0 - cosine).mean()
    magnitude = (
        student_rms.clamp_min(1e-8).log() - scale.log()
    ).abs().mean()
    loss = regression + float(direction_weight) * direction + float(
        magnitude_weight
    ) * magnitude
    return loss, {
        "loss": loss.detach(),
        "normalized_huber": regression.detach(),
        "direction_loss": direction.detach(),
        "cosine": cosine.mean().detach(),
        "magnitude_log_error": magnitude.detach(),
        "student_to_teacher_rms": (student_rms / scale).mean().detach(),
    }


def _save_state(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    style_codes: torch.Tensor,
    cfg: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "style_codes": style_codes.detach().cpu(),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _validate(
    model: NativeKVFactorModulator,
    style_codes: torch.Tensor,
    contexts: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    *,
    artists: int,
    contents: int,
    direction_weight: float,
    magnitude_weight: float,
) -> dict[str, float]:
    model.eval()
    rows: dict[str, list[float]] = defaultdict(list)
    artist_indices = torch.linspace(
        0, style_codes.shape[0] - 1, artists, device=style_codes.device
    ).round().long().unique()
    content_indices = torch.linspace(
        0, contexts.shape[0] - 1, contents, device=contexts.device
    ).round().long().unique()
    for block in range(model.blocks):
        for content_index in content_indices.tolist():
            codes = style_codes[artist_indices]
            context = contexts[content_index].expand(codes.shape[0], -1, -1)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=style_codes.device.type == "cuda",
            ):
                predicted_down, predicted_up = model(codes, block)
                student = apply_kv_factors(context, predicted_down, predicted_up)
                teacher = apply_kv_factors(
                    context,
                    teacher_down[artist_indices, block],
                    teacher_up[artist_indices, block],
                )
            _, metrics = kv_activation_objective(
                student,
                teacher,
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
            )
            for key, value in metrics.items():
                rows[key].append(float(value))
    model.train()
    return {key: sum(values) / len(values) for key, values in rows.items()}


def train_kv_activation_modulator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    cfg = dict(config["kv_activation_modulator"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 3000))
    seed = int(cfg.get("seed", 20260824))
    device = str(training.get("device", "cuda"))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    lora_root = destination / str(cfg["lora_directory"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        lora_root, blocks=int(cfg.get("blocks", 28))
    )
    source = torch.load(
        destination / str(cfg["style_code_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    code_key = str(cfg.get("style_code_key", "oracle_anchor"))
    style_codes = source[code_key].float()
    if style_codes.shape[0] != len(artist_ids):
        raise RuntimeError("Style-code and K/V-LoRA artist counts disagree")
    base = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )
    contexts = base["base_context"]
    model_cfg = dict(cfg["model"])
    rank = int(teacher_down.shape[-2])
    context_dim = int(teacher_down.shape[-1])
    output_dim = int(teacher_up.shape[-2])
    model = NativeKVFactorModulator(
        style_dim=int(style_codes.shape[-1]),
        blocks=int(teacher_down.shape[1]),
        rank=rank,
        context_dim=context_dim,
        output_dim=output_dim,
        **model_cfg,
    ).to(device)
    style_codes = style_codes.to(device=device, dtype=torch.bfloat16)
    contexts = contexts.to(device=device, dtype=torch.bfloat16)
    teacher_down = teacher_down.to(device=device, dtype=torch.float32)
    teacher_up = teacher_up.to(device=device, dtype=torch.float32)
    teacher_down, teacher_up = canonicalize_lora_factor_bank(
        teacher_down,
        teacher_up,
        chunk_size=int(training.get("canonicalization_chunk_size", 64)),
    )
    teacher_down = teacher_down.to(dtype=torch.bfloat16)
    teacher_up = teacher_up.to(dtype=torch.bfloat16)
    model.set_factor_scales(
        teacher_down.float().square().mean(dim=-1).sqrt().mean(dim=0),
        teacher_up.float().transpose(-1, -2).square().mean(dim=-1).sqrt().mean(dim=0),
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)) and device.startswith("cuda"),
    )
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    batch_artists = int(training.get("batch_artists", 16))
    warmup = int(training.get("warmup_steps", 100))
    direction_weight = float(training.get("direction_weight", 0.5))
    magnitude_weight = float(training.get("magnitude_weight", 0.1))
    factor_direction_weight = float(
        training.get("factor_direction_weight", 0.5)
    )
    factor_magnitude_weight = float(
        training.get("factor_magnitude_weight", 0.1)
    )
    down_factor_weight = float(training.get("down_factor_weight", 1.0))
    up_factor_weight = float(training.get("up_factor_weight", 1.0))
    activation_weight = float(training.get("activation_weight", 0.1))
    activation_start = int(training.get("activation_start_step", 500))
    activation_ramp = int(training.get("activation_ramp_steps", 500))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    base_lr = float(training.get("learning_rate", 2e-4))
    validation_contexts = int(training.get("validation_contexts", 4))
    validation_artists = int(training.get("validation_artists", 8))
    train_context_count = int(
        contexts.shape[0] - int(training.get("heldout_contexts", 32))
    )
    if train_context_count <= 0:
        raise ValueError("heldout_contexts leaves no training text contexts")
    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-activation-modulator")),
            id=str(wandb_cfg.get("id", "kv-activation-modulator")),
            resume="allow",
            config={"kv_activation_modulator": cfg},
        )
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        model.train()
        for step in range(start_step + 1, steps + 1):
            learning_rate = base_lr * min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            generator = torch.Generator().manual_seed(seed + step * 1_000_003)
            artists = torch.randperm(
                len(artist_ids), generator=generator
            )[:batch_artists].to(device)
            content_index = int(torch.randint(
                train_context_count, (1,), generator=generator
            ))
            block = (step - 1) % model.blocks
            context = contexts[content_index].expand(batch_artists, -1, -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                predicted_down, predicted_up = model(style_codes[artists], block)
            target_down = teacher_down[artists, block]
            target_up = teacher_up[artists, block]
            down_loss, down_metrics = kv_factor_objective(
                predicted_down,
                target_down,
                direction_weight=factor_direction_weight,
                magnitude_weight=factor_magnitude_weight,
            )
            up_loss, up_metrics = kv_factor_objective(
                predicted_up,
                target_up,
                direction_weight=factor_direction_weight,
                magnitude_weight=factor_magnitude_weight,
            )
            active_activation_weight = activation_weight * max(
                0.0,
                min(1.0, (step - activation_start) / max(1, activation_ramp)),
            )
            if active_activation_weight > 0:
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=device.startswith("cuda"),
                ):
                    student = apply_kv_factors(
                        context, predicted_down, predicted_up
                    )
                    teacher = apply_kv_factors(context, target_down, target_up)
                activation_loss, activation_metrics = kv_activation_objective(
                    student,
                    teacher,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
            else:
                activation_loss = down_loss.new_zeros(())
                activation_metrics = {
                    key: down_loss.new_zeros(())
                    for key in (
                        "loss", "normalized_huber", "direction_loss", "cosine",
                        "magnitude_log_error", "student_to_teacher_rms",
                        "relative_rms_error", "k_cosine", "v_cosine",
                    )
                }
            loss = (
                down_factor_weight * down_loss
                + up_factor_weight * up_loss
                + active_activation_weight * activation_loss
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            optimizer.step()
            metrics = {
                "loss": loss.detach(),
                **{
                    f"factor_down_{key}": value
                    for key, value in down_metrics.items()
                },
                **{
                    f"factor_up_{key}": value
                    for key, value in up_metrics.items()
                },
                **{
                    f"activation_{key}": value
                    for key, value in activation_metrics.items()
                },
            }
            metrics.update({
                "activation_weight": torch.tensor(active_activation_weight),
                "grad_norm": grad_norm.detach(),
                "learning_rate": torch.tensor(learning_rate),
                "block": torch.tensor(float(block)),
            })
            for key, value in metrics.items():
                running[key].append(float(value))
            if step % log_every == 0:
                row = {
                    key: sum(values) / len(values)
                    for key, values in running.items()
                }
                print(f"K/V activation step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/activation/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                validation = _validate(
                    model,
                    style_codes,
                    contexts[train_context_count:],
                    teacher_down,
                    teacher_up,
                    artists=validation_artists,
                    contents=validation_contexts,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
                print(f"K/V activation validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/activation/{key}": value for key, value in validation.items()},
                        step=step,
                    )
            if step % checkpoint_every == 0 or step == steps:
                for path in (
                    state_path,
                    checkpoints / f"step-{step:07d}.pt",
                ):
                    _save_state(
                        path,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        style_codes=style_codes,
                        cfg=cfg,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "blocks": model.blocks,
        "rank": model.rank,
        "contexts": int(contexts.shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(state_path),
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_kv_activation_modulator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = dict(config)
    cfg = dict(config["kv_activation_modulator"])
    training = dict(cfg["training"])
    cfg["output_directory"] = "kv_activation_modulator_smoke"
    training["resume"] = False
    training["validation_every"] = 0
    training["checkpoint_every"] = 1
    training["wandb"] = {"enabled": False}
    cfg["training"] = training
    effective["kv_activation_modulator"] = cfg
    return train_kv_activation_modulator(effective, destination, steps_override=2)
