"""Detail-preserving typed reader and separate Style Cross-Attention for Anima.

The frozen Dual-query Resampler supplies 64 spatial, 16 global, and four
artist-summary tokens per reference.  This module reads each reference into 28
canonical slots, lets 28 output queries softly align across the complete
reference-by-slot memory set, and injects the result through fresh block-local
K/V projections. Native Anima Q, O, output dropout, and gate_cross remain frozen.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .same_q_style_adapter.adapter import (
    _match_native_attention_dtypes,
    _run_attention,
)


@dataclass
class DetailStyleOutput:
    tokens: torch.Tensor
    per_reference_tokens: torch.Tensor
    reconstruction: torch.Tensor | None
    reconstruction_target: torch.Tensor | None
    pooled_reconstruction: torch.Tensor | None = None
    pooled_reconstruction_target: torch.Tensor | None = None


def _leave_one_out_artist_center(values: torch.Tensor) -> torch.Tensor:
    """Remove the other artists' mean without shrinking a batch-four target."""

    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Artist centering requires at least two batch rows")
    rows = values.shape[0]
    return (values * rows - values.sum(dim=0, keepdim=True)) / (rows - 1)


def _sincos_2d(side: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D sin/cos width must be divisible by four")
    coordinates = torch.linspace(-1.0, 1.0, side, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(dim // 4, dtype=torch.float32)
        / max(1, dim // 4 - 1)
    )
    values = torch.cat(
        (
            torch.sin(x.flatten()[:, None] * frequencies),
            torch.cos(x.flatten()[:, None] * frequencies),
            torch.sin(y.flatten()[:, None] * frequencies),
            torch.cos(y.flatten()[:, None] * frequencies),
        ),
        dim=-1,
    )
    return values


class _BiasFreeAttention(nn.Module):
    """Small explicit attention used where trainable per-slot biases are needed."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, queries = query.shape[:2]
        keys = key.shape[1]
        q = self.q(query).reshape(batch, queries, self.heads, self.head_dim)
        k = self.k(key).reshape(batch, keys, self.heads, self.head_dim)
        v = self.v(value).reshape(batch, keys, self.heads, self.head_dim)
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            if attention_bias.ndim == 2:
                logits = logits + attention_bias[None, None]
            elif attention_bias.ndim == 3:
                logits = logits + attention_bias[:, None]
            else:
                raise ValueError("attention_bias must be [Q,K] or [B,Q,K]")
        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask[:, None, None], torch.finfo(logits.dtype).min
            )
        probabilities = logits.softmax(dim=-1)
        attended = torch.einsum("bhqk,bkhd->bqhd", probabilities, v)
        attended = attended.reshape(batch, queries, self.dim)
        return self.o(attended), probabilities


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(values).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class _CanonicalReaderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = _BiasFreeAttention(dim, heads)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = _BiasFreeAttention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, ff_dim)

    def forward(
        self,
        content: torch.Tensor,
        slot_identity: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        type_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.cross_norm(content) + slot_identity
        update, attention = self.cross_attention(
            query,
            self.memory_norm(memory_key),
            self.memory_norm(memory_value),
            attention_bias=type_bias,
        )
        content = content + update
        normalized = self.self_norm(content)
        update, _ = self.self_attention(
            normalized + slot_identity,
            normalized + slot_identity,
            normalized,
        )
        content = content + update
        return content + self.ff(self.ff_norm(content)), attention


class _CrossSlotBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = _BiasFreeAttention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, ff_dim)

    def forward(
        self, content: torch.Tensor, slot_identity: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm(content)
        update, _ = self.attention(
            normalized + slot_identity,
            normalized + slot_identity,
            normalized,
        )
        content = content + update
        return content + self.ff(self.ff_norm(content))


class DetailPreservingTypedSlotReader(nn.Module):
    """Read 84-token references and softly align them into 28 style slots."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        spatial_tokens: int = 64,
        global_tokens: int = 16,
        summary_tokens: int = 4,
        output_tokens: int = 28,
        heads: int = 16,
        reader_layers: int = 2,
        reader_ff_dim: int = 3072,
        mixer_ff_dim: int = 3072,
        mixer_layers: int = 2,
        position_gain: float = 0.1,
        type_preference_bias: float = 1.0,
        same_slot_attention_bias: float = 1.0,
        reference_identity_gain: float = 0.25,
        slot_type_counts: tuple[int, int, int] = (16, 8, 4),
        strict_v1: bool = True,
    ) -> None:
        super().__init__()
        if strict_v1 and (
            spatial_tokens, global_tokens, summary_tokens, output_tokens
        ) != (64, 16, 4, 28):
            raise ValueError("v1 token contract is fixed at 64/16/4 -> 28")
        side = int(round(math.sqrt(spatial_tokens)))
        if side * side != spatial_tokens:
            raise ValueError("spatial_tokens must form a square grid")
        if sum(slot_type_counts) != output_tokens:
            raise ValueError("slot_type_counts must sum to output_tokens")
        if mixer_layers <= 0:
            raise ValueError("mixer_layers must be positive")
        if same_slot_attention_bias <= 0 or reference_identity_gain <= 0:
            raise ValueError(
                "same_slot_attention_bias and reference_identity_gain must be positive"
            )
        self.dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.summary_tokens = int(summary_tokens)
        self.cached_tokens = spatial_tokens + global_tokens + summary_tokens
        self.output_tokens = int(output_tokens)
        self.position_gain = float(position_gain)
        self.same_slot_attention_bias = float(same_slot_attention_bias)
        self.reference_identity_gain = float(reference_identity_gain)

        self.input_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.input_projections = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        )
        self.type_embeddings = nn.Parameter(torch.empty(3, dim))
        self.slot_identity = nn.Parameter(torch.empty(output_tokens, dim))
        self.type_preference = nn.Parameter(torch.zeros(output_tokens, 3))
        preferred = torch.tensor(
            [0] * slot_type_counts[0]
            + [1] * slot_type_counts[1]
            + [2] * slot_type_counts[2]
        )
        with torch.no_grad():
            self.type_preference.scatter_(
                1, preferred[:, None], float(type_preference_bias)
            )
        self.register_buffer(
            "output_slot_type_ids",
            preferred,
            persistent=True,
        )
        self.register_buffer(
            "spatial_position",
            _sincos_2d(side, dim),
            persistent=True,
        )
        self.register_buffer(
            "memory_type_ids",
            torch.tensor(
                [0] * spatial_tokens
                + [1] * global_tokens
                + [2] * summary_tokens,
                dtype=torch.long,
            ),
            persistent=True,
        )
        self.reader = nn.ModuleList(
            _CanonicalReaderBlock(dim, heads, reader_ff_dim)
            for _ in range(reader_layers)
        )
        self.set_query = nn.Parameter(torch.empty(output_tokens, dim))
        self.set_norm = nn.LayerNorm(dim)
        self.reference_identity_norm = nn.LayerNorm(dim)
        self.reference_identity_projection = nn.Linear(dim, dim, bias=False)
        self.pool_type_embeddings = nn.Parameter(torch.empty(3, dim))
        self.pool_type_preference = nn.Parameter(torch.zeros(output_tokens, 3))
        with torch.no_grad():
            self.pool_type_preference.scatter_(
                1, preferred[:, None], float(type_preference_bias)
            )
        self.set_attention = _BiasFreeAttention(dim, heads)
        self.set_ff_norm = nn.LayerNorm(dim)
        self.set_ff = _SwiGLU(dim, reader_ff_dim)
        self.mixers = nn.ModuleList(
            _CrossSlotBlock(dim, heads, mixer_ff_dim)
            for _ in range(mixer_layers)
        )

        self.reconstruction_queries = nn.Parameter(
            torch.empty(self.cached_tokens, dim)
        )
        self.reconstruction_norm = nn.LayerNorm(dim)
        self.reconstruction_attention = _BiasFreeAttention(dim, heads)
        self.reconstruction_output = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        std = self.dim**-0.5
        nn.init.normal_(self.type_embeddings, std=std)
        nn.init.normal_(self.slot_identity, std=std)
        nn.init.normal_(self.set_query, std=std)
        nn.init.normal_(self.pool_type_embeddings, std=std)
        nn.init.normal_(self.reconstruction_queries, std=std)

    def _typed_memory(
        self, references: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_end = self.spatial_tokens
        global_end = spatial_end + self.global_tokens
        boundaries = (0, spatial_end, global_end, self.cached_tokens)
        values = []
        targets = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            normalized = self.input_norms[index](references[:, start:end])
            targets.append(normalized)
            projected = self.input_projections[index](normalized)
            projected = projected + self.type_embeddings[index]
            values.append(projected)
        memory_value = torch.cat(values, dim=1)
        memory_key = memory_value.clone()
        memory_key[:, : self.spatial_tokens] = memory_key[:, : self.spatial_tokens] + (
            self.position_gain * self.spatial_position.to(memory_key.dtype)
        )
        return memory_key, memory_value, torch.cat(targets, dim=1)

    def _read_valid(
        self, references: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory_key, memory_value, reconstruction_target = self._typed_memory(
            references
        )
        content = references.new_zeros(
            references.shape[0], self.output_tokens, self.dim
        )
        identity = self.slot_identity.to(references.dtype).unsqueeze(0)
        type_bias = self.type_preference[:, self.memory_type_ids].to(
            dtype=references.dtype
        )
        maps = []
        for block in self.reader:
            content, attention = block(
                content, identity, memory_key, memory_value, type_bias
            )
            maps.append(attention)
        return content, reconstruction_target, torch.stack(maps, dim=1)

    def _pool_references(
        self, per_reference: torch.Tensor, reference_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, references, slots, dim = per_reference.shape
        normalized = self.set_norm(per_reference)
        reference_identity = self.reference_identity_projection(
            self.reference_identity_norm(normalized.mean(dim=2))
        )
        reference_identity = (
            self.reference_identity_gain
            * reference_identity
            * reference_mask[:, :, None].to(reference_identity.dtype)
        )
        slot_identity = self.slot_identity.to(normalized.dtype)
        slot_type = self.pool_type_embeddings[
            self.output_slot_type_ids
        ].to(normalized.dtype)
        memory_key = (
            normalized
            + slot_identity[None, None]
            + slot_type[None, None]
            + reference_identity[:, :, None]
        ).reshape(batch, references * slots, dim)
        # Metadata changes routing, not the visual payload.  Keeping values as
        # normalized per-reference content avoids leaking arbitrary identity
        # embeddings into the final Anima context.
        memory_value = normalized.reshape(batch, references * slots, dim)
        memory_slot_ids = torch.arange(slots, device=normalized.device).repeat(
            references
        )
        memory_type_ids = self.output_slot_type_ids[memory_slot_ids]
        type_bias = self.pool_type_preference[:, memory_type_ids].to(
            normalized.dtype
        )
        same_slot = (
            torch.arange(slots, device=normalized.device)[:, None]
            == memory_slot_ids[None]
        ).to(normalized.dtype)
        attention_bias = type_bias + self.same_slot_attention_bias * same_slot
        query = (
            self.set_query.to(normalized.dtype) + slot_identity
        )[None].expand(batch, -1, -1)
        mask = ~reference_mask[:, :, None].expand(-1, -1, slots).reshape(
            batch, references * slots
        )
        pooled, _ = self.set_attention(
            query,
            memory_key,
            memory_value,
            attention_bias=attention_bias,
            key_padding_mask=mask,
        )
        pooled = pooled + self.set_ff(self.set_ff_norm(pooled))
        identity = self.slot_identity.to(pooled.dtype).unsqueeze(0)
        for mixer in self.mixers:
            pooled = mixer(pooled, identity)
        return pooled

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> DetailStyleOutput:
        if references.ndim != 4:
            raise ValueError("references must be [batch,references,84,1024]")
        if references.shape[2:] != (self.cached_tokens, self.dim):
            raise ValueError(
                f"Expected reference tail {(self.cached_tokens, self.dim)}, "
                f"got {tuple(references.shape[2:])}"
            )
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference_mask does not match references")
        if not reference_mask.is_cuda and not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample requires at least one reference")

        valid = references[reference_mask]
        encoded, targets, _ = self._read_valid(valid)
        per_reference = encoded.new_zeros(
            *references.shape[:2], self.output_tokens, self.dim
        )
        per_reference[reference_mask] = encoded
        tokens = self._pool_references(per_reference, reference_mask)

        reconstruction = None
        reconstruction_target = None
        pooled_reconstruction = None
        pooled_reconstruction_target = None
        if reconstruct:
            def decode(values: torch.Tensor) -> torch.Tensor:
                queries = self.reconstruction_queries.to(values.dtype)[None].expand(
                    values.shape[0], -1, -1
                )
                normalized = self.reconstruction_norm(values)
                decoded, _ = self.reconstruction_attention(
                    queries, normalized, normalized
                )
                return self.reconstruction_output(decoded)

            reconstruction = decode(encoded)
            reconstruction_target = targets.detach()
            full_targets = targets.new_zeros(
                *references.shape[:2], self.cached_tokens, self.dim
            )
            full_targets[reference_mask] = targets
            counts = reference_mask.sum(dim=1).clamp_min(1).to(targets.dtype)
            pooled_reconstruction_target = (
                full_targets.sum(dim=1) / counts[:, None, None]
            ).detach()
            pooled_reconstruction = decode(tokens)
        return DetailStyleOutput(
            tokens=tokens,
            per_reference_tokens=per_reference,
            reconstruction=reconstruction,
            reconstruction_target=reconstruction_target,
            pooled_reconstruction=pooled_reconstruction,
            pooled_reconstruction_target=pooled_reconstruction_target,
        )


class FreshKVStyleCrossAttention(nn.Module):
    """Fresh per-block K/V with native Anima Q/O and separate softmax."""

    def __init__(
        self,
        *,
        context_dim: int = 1024,
        blocks: int = 28,
        initial_alpha: float = 0.1,
        null_tokens: int = 28,
        null_init_std: float = 0.02,
        common_gain: float = 1.0,
        artist_gain: float = 1.0,
        gain_maximum: float = 100.0,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.blocks = int(blocks)
        self.null_tokens = int(null_tokens)
        if self.null_tokens <= 0 or null_init_std <= 0:
            raise ValueError("null_tokens and null_init_std must be positive")
        if common_gain <= 0 or artist_gain <= 0 or gain_maximum <= 1:
            raise ValueError("Style gains must be positive and gain_maximum > 1")
        self.style_k = nn.ModuleList()
        self.style_v = nn.ModuleList()
        # The adapter represents style as a residual from a learned null
        # context.  This makes the artist-varying path identifiable instead of
        # letting one large reference-independent effect dominate every row.
        self.null_style_context = nn.Parameter(
            torch.empty(1, self.null_tokens, self.context_dim)
        )
        nn.init.normal_(self.null_style_context, std=float(null_init_std))
        self.gain_maximum = float(gain_maximum)
        self.log_common_gain = nn.Parameter(
            torch.tensor(math.log(float(common_gain)))
        )
        self.log_artist_gain = nn.Parameter(
            torch.tensor(math.log(float(artist_gain)))
        )
        self.register_buffer(
            "alpha", torch.full((blocks,), float(initial_alpha)), persistent=True
        )
        default_edges = torch.tensor(
            (0.0, 0.325, 0.625, 0.86, 1.000001), dtype=torch.float32
        )
        self.register_buffer(
            "strength_timestep_centers",
            0.5 * (default_edges[:-1] + default_edges[1:]),
            persistent=True,
        )
        self.register_buffer(
            "alpha_by_timestep",
            torch.full((4, blocks), float(initial_alpha)),
            persistent=True,
        )
        self.register_buffer(
            "native_lower_by_timestep",
            torch.zeros(4, blocks, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "native_upper_by_timestep",
            torch.full((4, blocks), float("inf"), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "timestep_strength_enabled", torch.tensor(False), persistent=True
        )
        self.register_buffer(
            "native_fixed_output_by_timestep",
            torch.zeros(4, blocks, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fixed_output_strength_enabled", torch.tensor(False), persistent=True
        )
        self._timestep_strength_active = False
        self._fixed_output_strength_active = False
        self._style_context: torch.Tensor | None = None
        self._style_enabled: torch.Tensor | None = None
        self._style_strength = 1.0
        self._timesteps: torch.Tensor | None = None
        self._timestep_interpolation: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._teacher_context: torch.Tensor | None = None
        self._teacher_block_indices: frozenset[int] | None = None
        self._post_gate_distillation_enabled = False
        self._pending_internal: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._block_gate_context: dict[
            int, tuple[torch.Tensor, tuple[int, int, int]]
        ] = {}
        self._internal_terms: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._post_gate_distillation_terms: list[
            tuple[int, dict[str, torch.Tensor]]
        ] = []
        self._post_gate_center_collection = False
        self._post_gate_center_samples: dict[
            int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {}
        self._post_gate_centers: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._calibration = False
        self._calibration_bin_edges: tuple[float, ...] = (0.0, 1.000001)
        self._calibration_bin_index: int | None = None
        self._calibration_inject_style = False
        self._calibration_measured_alpha = torch.empty(0)
        self._calibration_teacher: list[list[list[torch.Tensor]]] = []
        self._calibration_student: list[list[list[torch.Tensor]]] = []
        self._calibration_alpha: list[list[list[torch.Tensor]]] = []
        self._calibration_raw_attention: list[list[list[torch.Tensor]]] = []
        self._calibration_cosine: list[list[list[torch.Tensor]]] = []
        self._calibration_aligned_rms: list[list[list[torch.Tensor]]] = []
        self._calibration_orthogonal_rms: list[list[list[torch.Tensor]]] = []
        self._runtime_ratios: list[torch.Tensor | None] = [None] * blocks
        self._runtime_alphas: list[torch.Tensor | None] = [None] * blocks
        self._runtime_fixed_targets: list[torch.Tensor | None] = [None] * blocks
        self._runtime_fixed_actuals: list[torch.Tensor | None] = [None] * blocks
        self._runtime_fixed_scales: list[torch.Tensor | None] = [None] * blocks
        self._diagnostic_recorder: (
            Callable[[int, str, torch.Tensor], None] | None
        ) = None
        self._initialized = False

    def initialize_from_anima(self, anima: nn.Module) -> None:
        if self._initialized:
            return
        if len(anima.blocks) != self.blocks:
            raise ValueError(f"Expected {self.blocks} Anima blocks, got {len(anima.blocks)}")
        for block in anima.blocks:
            cross = block.cross_attn
            output_dim = int(cross.n_heads) * int(cross.head_dim)
            if self.context_dim <= 0 or output_dim <= 0:
                raise ValueError("Invalid Anima cross-attention dimensions")
            key = nn.Linear(
                self.context_dim,
                output_dim,
                bias=False,
                device=cross.output_proj.weight.device,
                dtype=cross.output_proj.weight.dtype,
            )
            value = nn.Linear(
                self.context_dim,
                output_dim,
                bias=False,
                device=cross.output_proj.weight.device,
                dtype=cross.output_proj.weight.dtype,
            )
            nn.init.xavier_uniform_(key.weight)
            nn.init.xavier_uniform_(value.weight)
            self.style_k.append(key)
            self.style_v.append(value)
        self._initialized = True

    def kv_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.style_k.parameters())
            + list(self.style_v.parameters())
            + [self.null_style_context]
            + self.gain_parameters()
        )

    def null_parameters(self) -> list[nn.Parameter]:
        return [self.null_style_context]

    def gain_parameters(self) -> list[nn.Parameter]:
        return [self.log_common_gain, self.log_artist_gain]

    def _component_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.log_common_gain.exp().clamp_max(self.gain_maximum),
            self.log_artist_gain.exp().clamp_max(self.gain_maximum),
        )

    def set_style_context(
        self,
        tokens: torch.Tensor,
        *,
        enabled: torch.Tensor | None = None,
        strength: float = 1.0,
    ) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != self.context_dim:
            raise ValueError("style context must be [batch,tokens,context_dim]")
        if enabled is not None and enabled.shape != (tokens.shape[0],):
            raise ValueError("enabled must contain one value per batch row")
        self._style_context = tokens
        self._style_enabled = enabled
        self._style_strength = float(strength)

    set_style_tokens = set_style_context

    def set_teacher_context(
        self,
        context: torch.Tensor | None,
        *,
        block_indices: tuple[int, ...] | list[int] | None = None,
        post_gate_distillation: bool = False,
    ) -> None:
        if context is not None and (
            context.ndim != 3 or context.shape[-1] != self.context_dim
        ):
            raise ValueError("teacher context must be [batch,tokens,context_dim]")
        selected = None
        if block_indices is not None:
            selected = frozenset(int(index) for index in block_indices)
            if not selected or min(selected) < 0 or max(selected) >= self.blocks:
                raise ValueError("teacher block indices must select valid Anima blocks")
        self._teacher_context = context
        self._teacher_block_indices = selected
        self._post_gate_distillation_enabled = bool(post_gate_distillation)

    def _teacher_enabled_for_block(self, block_index: int) -> bool:
        return self._teacher_context is not None and (
            self._teacher_block_indices is None
            or int(block_index) in self._teacher_block_indices
        )

    def set_diagnostic_recorder(
        self,
        recorder: Callable[[int, str, torch.Tensor], None] | None,
    ) -> None:
        """Attach a transient stage recorder without changing model state."""

        self._diagnostic_recorder = recorder

    def record_diagnostic_stage(
        self, block_index: int, stage: str, value: torch.Tensor
    ) -> None:
        if self._diagnostic_recorder is not None:
            self._diagnostic_recorder(int(block_index), str(stage), value)

    def set_timesteps(self, timesteps: torch.Tensor | float) -> None:
        value = torch.as_tensor(timesteps, dtype=torch.float32, device=self.alpha.device)
        self._timesteps = value.reshape(-1)
        self._timestep_interpolation = None

    @torch.no_grad()
    def configure_timestep_strength(
        self,
        *,
        timestep_bin_edges: tuple[float, ...] | list[float],
        alpha_by_timestep: torch.Tensor,
        native_lower_by_timestep: torch.Tensor,
        native_upper_by_timestep: torch.Tensor,
    ) -> None:
        edges = torch.as_tensor(
            timestep_bin_edges, dtype=torch.float32, device=self.alpha.device
        )
        expected = (edges.numel() - 1, self.blocks)
        if edges.numel() < 2 or not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError("timestep_bin_edges must be strictly increasing")
        values = (
            alpha_by_timestep,
            native_lower_by_timestep,
            native_upper_by_timestep,
        )
        if any(tuple(value.shape) != expected for value in values):
            raise ValueError(f"Timestep strength tensors must have shape {expected}")
        if bool((alpha_by_timestep <= 0).any()):
            raise ValueError("alpha_by_timestep must be positive")
        if bool((native_lower_by_timestep < 0).any()) or bool(
            (native_upper_by_timestep <= native_lower_by_timestep).any()
        ):
            raise ValueError("Native strength bounds must be nonnegative and ordered")
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.strength_timestep_centers = centers
        self.alpha_by_timestep = alpha_by_timestep.detach().float().to(self.alpha.device)
        self.native_lower_by_timestep = (
            native_lower_by_timestep.detach().float().to(self.alpha.device)
        )
        self.native_upper_by_timestep = (
            native_upper_by_timestep.detach().float().to(self.alpha.device)
        )
        self.alpha.copy_(self.alpha_by_timestep.median(dim=0).values)
        self.timestep_strength_enabled.fill_(True)
        self._timestep_strength_active = True

    @torch.no_grad()
    def configure_fixed_output_strength(
        self,
        *,
        timestep_bin_edges: tuple[float, ...] | list[float],
        native_fixed_output_by_timestep: torch.Tensor,
    ) -> None:
        """Fix post-native-gate style RMS to a measured native target.

        The target is defined on ``gate_cross(t) * O(style_attention)``.  It
        therefore controls the residual that Anima actually receives, rather
        than a pre-O attention norm or an indirect alpha coefficient.
        """

        edges = torch.as_tensor(
            timestep_bin_edges, dtype=torch.float32, device=self.alpha.device
        )
        expected = (edges.numel() - 1, self.blocks)
        if edges.numel() < 2 or not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError("timestep_bin_edges must be strictly increasing")
        if tuple(native_fixed_output_by_timestep.shape) != expected:
            raise ValueError(f"Fixed output tensor must have shape {expected}")
        if not bool(torch.isfinite(native_fixed_output_by_timestep).all()) or bool(
            (native_fixed_output_by_timestep <= 0).any()
        ):
            raise ValueError("Fixed output targets must be finite and positive")
        self.strength_timestep_centers = 0.5 * (edges[:-1] + edges[1:])
        self.native_fixed_output_by_timestep = (
            native_fixed_output_by_timestep.detach().float().to(self.alpha.device)
        )
        self.fixed_output_strength_enabled.fill_(True)
        self._fixed_output_strength_active = True

    def restore_timestep_strength_state(self) -> None:
        """Refresh the host-side fast path once after loading a checkpoint."""
        self._timestep_strength_active = bool(
            self.timestep_strength_enabled.detach().cpu().item()
        )
        self._fixed_output_strength_active = bool(
            self.fixed_output_strength_enabled.detach().cpu().item()
        )

    def _interpolate_timestep_values(
        self, values: torch.Tensor, block_index: int, batch: int
    ) -> torch.Tensor:
        if self._timesteps is None:
            raise RuntimeError(
                "set_timesteps() is required before a timestep-calibrated style forward"
            )
        timesteps = self._timesteps.to(device=values.device, dtype=torch.float32)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch)
        elif timesteps.numel() != batch:
            raise ValueError(
                f"Expected one timestep or {batch} per-row timesteps, got "
                f"{timesteps.numel()}"
            )
        centers = self.strength_timestep_centers.to(values.device)
        if centers.numel() == 1:
            return values[0, block_index].expand(batch)
        if self._timestep_interpolation is None:
            right = torch.searchsorted(centers, timesteps).clamp(
                1, centers.numel() - 1
            )
            left = right - 1
            left_center = centers[left]
            right_center = centers[right]
            weight = (
                (timesteps - left_center) / (right_center - left_center)
            ).clamp(0, 1)
            self._timestep_interpolation = (left, right, weight)
        left, right, weight = self._timestep_interpolation
        column = values[:, block_index]
        return column[left] + weight * (column[right] - column[left])

    def _effective_alpha(
        self, block_index: int, batch: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._timestep_strength_active:
            base = self._interpolate_timestep_values(
                self.alpha_by_timestep, block_index, batch
            )
        else:
            base = self.alpha[block_index].float().expand(batch)
        gain = float(getattr(self, "global_gain", 1.0)) * float(self._style_strength)
        return (base * gain).to(device=device, dtype=dtype).reshape(batch, 1, 1)

    def _native_strength_bounds(
        self, block_index: int, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._timestep_strength_active:
            zero = self.alpha.new_zeros(batch, dtype=torch.float32)
            return zero, zero.new_full((batch,), float("inf"))
        return (
            self._interpolate_timestep_values(
                self.native_lower_by_timestep, block_index, batch
            ),
            self._interpolate_timestep_values(
                self.native_upper_by_timestep, block_index, batch
            ),
        )

    def _fixed_output_target(self, block_index: int, batch: int) -> torch.Tensor:
        if not self._fixed_output_strength_active:
            raise RuntimeError("Fixed output strength is not configured")
        base = self._interpolate_timestep_values(
            self.native_fixed_output_by_timestep, block_index, batch
        )
        gain = float(getattr(self, "global_gain", 1.0)) * float(
            self._style_strength
        )
        return base * gain

    def set_block_gate_context(
        self,
        block_index: int,
        gate_cross: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> None:
        self._block_gate_context[int(block_index)] = (gate_cross, spatial_shape)

    def clear_style_tokens(self) -> None:
        self._style_context = None
        self._style_enabled = None
        self._style_strength = 1.0
        self._teacher_context = None
        self._teacher_block_indices = None
        self._post_gate_distillation_enabled = False
        self._timesteps = None
        self._timestep_interpolation = None
        self._pending_internal.clear()
        self._block_gate_context.clear()

    def reset_internal_teacher(self) -> None:
        self._pending_internal.clear()
        self._internal_terms.clear()
        self._post_gate_distillation_terms.clear()

    def begin_post_gate_center_collection(self) -> None:
        self._post_gate_center_collection = True
        self._post_gate_center_samples.clear()
        self._post_gate_centers.clear()

    def finish_post_gate_center_collection(
        self,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if not self._post_gate_center_collection:
            raise RuntimeError("Post-gate center collection is not active")
        centers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for block_index, rows in self._post_gate_center_samples.items():
            student_artist = torch.cat([row[0] for row in rows], dim=0)
            teacher = torch.cat([row[2] for row in rows], dim=0)
            centers[block_index] = (
                student_artist.mean(dim=0, keepdim=True),
                teacher.mean(dim=0, keepdim=True),
            )
        self._post_gate_center_collection = False
        self._post_gate_center_samples.clear()
        return centers

    def set_post_gate_centers(
        self, centers: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        self._post_gate_centers = {
            int(index): (student.detach(), teacher.detach())
            for index, (student, teacher) in centers.items()
        }

    def clear_post_gate_centers(self) -> None:
        self._post_gate_centers.clear()

    def begin_alpha_calibration(
        self,
        *,
        timestep_bin_edges: tuple[float, ...] = (0.0, 1.000001),
        reset_alpha: bool = True,
        inject_style: bool = False,
    ) -> None:
        edges = tuple(float(value) for value in timestep_bin_edges)
        if len(edges) < 2 or any(
            right <= left
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ):
            raise ValueError("timestep_bin_edges must be strictly increasing")
        self._calibration = True
        self._calibration_bin_edges = edges
        self._calibration_bin_index = None
        self._calibration_inject_style = bool(inject_style)
        bins = len(edges) - 1
        self._calibration_teacher = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        self._calibration_student = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        self._calibration_alpha = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        self._calibration_raw_attention = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        self._calibration_cosine = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        self._calibration_aligned_rms = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        self._calibration_orthogonal_rms = [
            [[] for _ in range(bins)] for _ in range(self.blocks)
        ]
        # Measure hypothetical style effects on the clean frozen-Anima path.
        # Injecting alpha=1 into early blocks would corrupt the queries used to
        # measure every later block.  ``inject_style`` exists only for explicit
        # diagnostics of that old behaviour.
        if reset_alpha:
            self.timestep_strength_enabled.fill_(False)
            self._timestep_strength_active = False
            self.fixed_output_strength_enabled.fill_(False)
            self._fixed_output_strength_active = False
            self.alpha.fill_(1.0)
        self._calibration_measured_alpha = self.alpha.detach().float().clone()

    def set_alpha_calibration_timestep(self, timestep: float) -> None:
        if not self._calibration:
            raise RuntimeError("Alpha calibration was not started")
        index = bisect_right(self._calibration_bin_edges, float(timestep)) - 1
        self._calibration_bin_index = min(
            max(index, 0), len(self._calibration_bin_edges) - 2
        )
        self.set_timesteps(float(timestep))

    @torch.no_grad()
    def finish_alpha_calibration(
        self,
        *,
        minimum: float = 1e-6,
        maximum: float = 2.0,
        relative_block_gain: torch.Tensor | None = None,
        global_gain: float = 1.0,
        apply_alpha: bool = True,
        recommended_lower_multiplier: float = 1.5,
        recommended_upper_multiplier: float = 1.5,
        fixed_output_quantile: float | None = None,
    ) -> dict[str, Any]:
        if not self._calibration:
            raise RuntimeError("Alpha calibration was not started")
        if minimum <= 0 or maximum < minimum or global_gain <= 0:
            raise ValueError("Calibration scales must be positive and ordered")
        if fixed_output_quantile is not None and not 0.0 < fixed_output_quantile < 1.0:
            raise ValueError("fixed_output_quantile must be between zero and one")

        bins = len(self._calibration_bin_edges) - 1
        teacher_by_bin = torch.full(
            (bins, self.blocks), float("nan"), device=self.alpha.device
        )
        student_by_bin = torch.full_like(teacher_by_bin, float("nan"))
        raw_residual_by_bin = torch.full_like(teacher_by_bin, float("nan"))
        raw_attention_by_bin = torch.full_like(teacher_by_bin, float("nan"))
        native_p95_by_bin = torch.full_like(teacher_by_bin, float("nan"))
        native_fixed_by_bin = torch.full_like(teacher_by_bin, float("nan"))
        per_bin_ratio = torch.full_like(teacher_by_bin, float("nan"))
        cell_values: list[list[dict[str, torch.Tensor] | None]] = [
            [None for _ in range(self.blocks)] for _ in range(bins)
        ]
        base_alpha = torch.empty_like(self.alpha.float())
        for block in range(self.blocks):
            ratios = []
            for bin_index in range(bins):
                teacher_values = self._calibration_teacher[block][bin_index]
                student_values = self._calibration_student[block][bin_index]
                alpha_values = self._calibration_alpha[block][bin_index]
                attention_values = self._calibration_raw_attention[block][bin_index]
                cosine_values = self._calibration_cosine[block][bin_index]
                aligned_values = self._calibration_aligned_rms[block][bin_index]
                orthogonal_values = self._calibration_orthogonal_rms[block][bin_index]
                if (
                    not teacher_values
                    or not student_values
                    or not alpha_values
                    or not attention_values
                ):
                    continue
                teacher_samples = torch.cat(teacher_values).float()
                student_samples = torch.cat(student_values).float()
                alpha_samples = torch.cat(alpha_values).float().abs().clamp_min(1e-8)
                # Student RMS is measured after removing the batch-common
                # component.  Derive the corresponding alpha=1 artist-specific
                # residual from that same centered quantity; measuring a second
                # uncentered tensor would mix common output into the denominator.
                raw_residual_samples = student_samples / alpha_samples
                attention_samples = torch.cat(attention_values).float()
                cosine_samples = torch.cat(cosine_values).float()
                aligned_samples = torch.cat(aligned_values).float()
                orthogonal_samples = torch.cat(orthogonal_values).float()
                teacher = teacher_samples.median()
                teacher_p95 = teacher_samples.quantile(0.95)
                teacher_fixed = teacher_samples.quantile(
                    0.75 if fixed_output_quantile is None else fixed_output_quantile
                )
                student = student_samples.median()
                raw_residual = raw_residual_samples.median()
                attention = attention_samples.median()
                ratio = teacher / raw_residual.clamp_min(1e-8)
                teacher_by_bin[bin_index, block] = teacher
                student_by_bin[bin_index, block] = student
                raw_residual_by_bin[bin_index, block] = raw_residual
                raw_attention_by_bin[bin_index, block] = attention
                native_p95_by_bin[bin_index, block] = teacher_p95
                native_fixed_by_bin[bin_index, block] = teacher_fixed
                per_bin_ratio[bin_index, block] = ratio
                ratios.append(ratio)
                cell_values[bin_index][block] = {
                    "native": teacher_samples,
                    "effective": student_samples,
                    "raw_residual": raw_residual_samples,
                    "raw_attention": attention_samples,
                    "cosine": cosine_samples,
                    "aligned_rms": aligned_samples,
                    "orthogonal_rms": orthogonal_samples,
                }
            if not ratios:
                raise RuntimeError(
                    f"Alpha calibration did not observe Anima block {block}"
                )
            base_alpha[block] = torch.stack(ratios).median()

        if relative_block_gain is not None:
            raise ValueError(
                "relative_block_gain is disabled; block differences come from "
                "the measured block-by-timestep profile"
            )
        relative = torch.ones_like(base_alpha)
        alpha_by_timestep = per_bin_ratio.clone()
        lower_source = teacher_by_bin.clone()
        upper_source = native_p95_by_bin.clone()
        fixed_source = native_fixed_by_bin.clone()
        for block in range(self.blocks):
            observed = torch.isfinite(per_bin_ratio[:, block])
            alpha_by_timestep[~observed, block] = base_alpha[block]
            native_fallback = teacher_by_bin[observed, block].median()
            p95_fallback = native_p95_by_bin[observed, block].median()
            fixed_fallback = native_fixed_by_bin[observed, block].median()
            lower_source[~observed, block] = native_fallback
            upper_source[~observed, block] = p95_fallback
            fixed_source[~observed, block] = fixed_fallback
        alpha_by_timestep = alpha_by_timestep.clamp(
            min=float(minimum), max=float(maximum)
        )
        alpha = alpha_by_timestep.median(dim=0).values
        native_lower_by_timestep = (
            lower_source * float(recommended_lower_multiplier)
        )
        native_upper_by_timestep = (
            upper_source * float(recommended_upper_multiplier)
        )
        if apply_alpha:
            self.configure_timestep_strength(
                timestep_bin_edges=self._calibration_bin_edges,
                alpha_by_timestep=alpha_by_timestep,
                native_lower_by_timestep=native_lower_by_timestep,
                native_upper_by_timestep=native_upper_by_timestep,
            )
            if fixed_output_quantile is not None:
                self.configure_fixed_output_strength(
                    timestep_bin_edges=self._calibration_bin_edges,
                    native_fixed_output_by_timestep=fixed_source.clamp_min(1e-8),
                )
        self._calibration = False
        self._calibration_bin_index = None

        def serializable(values: torch.Tensor) -> list[list[float | None]]:
            return [
                [None if not math.isfinite(float(value)) else float(value) for value in row]
                for row in values.detach().cpu()
            ]

        profiles = []
        for bin_index, (left, right) in enumerate(
            zip(
                self._calibration_bin_edges[:-1],
                self._calibration_bin_edges[1:],
                strict=True,
            )
        ):
            blocks = []
            for block, values in enumerate(cell_values[bin_index]):
                if values is None:
                    continue

                def distribution(name: str) -> dict[str, float]:
                    samples = values[name]
                    return {
                        "mean": float(samples.mean()),
                        "median": float(samples.median()),
                        "p25": float(samples.quantile(0.25)),
                        "p75": float(samples.quantile(0.75)),
                        "p95": float(samples.quantile(0.95)),
                        "maximum": float(samples.max()),
                    }

                native = distribution("native")
                blocks.append({
                    "block": block,
                    "samples": int(values["native"].numel()),
                    "native_residual_rms": native,
                    "effective_style_residual_rms": distribution("effective"),
                    "raw_style_residual_rms": distribution("raw_residual"),
                    "raw_style_attention_rms": distribution("raw_attention"),
                    "student_teacher_cosine": distribution("cosine"),
                    "teacher_aligned_style_rms": distribution("aligned_rms"),
                    "orthogonal_style_rms": distribution("orthogonal_rms"),
                    "recommended_lower": (
                        float(recommended_lower_multiplier) * native["median"]
                    ),
                    "recommended_upper": (
                        float(recommended_upper_multiplier) * native["p95"]
                    ),
                })
            profiles.append({
                "bin": bin_index,
                "timestep_min": left,
                "timestep_max": right,
                "blocks": blocks,
            })

        result = {
            "timestep_bin_edges": list(self._calibration_bin_edges),
            "native_artist_residual_rms_by_timestep_bin": serializable(
                teacher_by_bin
            ),
            "effective_style_residual_rms_by_timestep_bin": serializable(
                student_by_bin
            ),
            "raw_style_residual_rms_by_timestep_bin": serializable(
                raw_residual_by_bin
            ),
            "raw_style_attention_rms_by_timestep_bin": serializable(
                raw_attention_by_bin
            ),
            "teacher_to_centered_raw_ratio_by_timestep_bin": serializable(
                per_bin_ratio
            ),
            "base_alpha": base_alpha.detach().cpu().tolist(),
            "relative_block_gain_normalized": relative.detach().cpu().tolist(),
            "global_gain": float(global_gain),
            "minimum_alpha": float(minimum),
            "maximum_alpha": float(maximum),
            "alpha": alpha.detach().cpu().tolist(),
            "alpha_by_timestep": serializable(alpha_by_timestep),
            "native_lower_by_timestep": serializable(native_lower_by_timestep),
            "native_upper_by_timestep": serializable(native_upper_by_timestep),
            "native_fixed_output_by_timestep": serializable(fixed_source),
            "fixed_output_quantile": (
                None if fixed_output_quantile is None else float(fixed_output_quantile)
            ),
            "fixed_output_applied": bool(
                apply_alpha and fixed_output_quantile is not None
            ),
            "measured_alpha": self._calibration_measured_alpha.cpu().tolist(),
            "alpha_applied": bool(apply_alpha),
            "recommended_lower_multiplier": float(recommended_lower_multiplier),
            "recommended_upper_multiplier": float(recommended_upper_multiplier),
            "block_timestep_profiles": profiles,
        }
        self._calibration_teacher = []
        self._calibration_student = []
        self._calibration_alpha = []
        self._calibration_raw_attention = []
        self._calibration_cosine = []
        self._calibration_aligned_rms = []
        self._calibration_orthogonal_rms = []
        return result

    def _style_kv(
        self, index: int, cross_attention: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._style_context is None:
            raise RuntimeError("No active style context")
        key = self.style_k[index](self._style_context)
        value = self.style_v[index](self._style_context)
        key = rearrange(
            key,
            "b s (h d) -> b s h d",
            h=cross_attention.n_heads,
            d=cross_attention.head_dim,
        )
        value = rearrange(
            value,
            "b s (h d) -> b s h d",
            h=cross_attention.n_heads,
            d=cross_attention.head_dim,
        )
        return cross_attention.k_norm(key), cross_attention.v_norm(value)

    def _null_style_kv(
        self, index: int, cross_attention: nn.Module, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.style_k[index](self.null_style_context)
        value = self.style_v[index](self.null_style_context)
        key = rearrange(
            key, "b s (h d) -> b s h d",
            h=cross_attention.n_heads, d=cross_attention.head_dim,
        )
        value = rearrange(
            value, "b s (h d) -> b s h d",
            h=cross_attention.n_heads, d=cross_attention.head_dim,
        )
        return (
            cross_attention.k_norm(key).expand(batch, -1, -1, -1),
            cross_attention.v_norm(value).expand(batch, -1, -1, -1),
        )

    @staticmethod
    def _native_output_weight(cross_attention: nn.Module, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, cross_attention.output_proj.weight, None)

    @staticmethod
    def _native_context_kv(
        cross_attention: nn.Module, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a second native context without recomputing the shared Q."""

        if hasattr(cross_attention, "kv_proj"):
            key, value = cross_attention.kv_proj(context).unflatten(
                -1,
                (2, cross_attention.n_heads, cross_attention.head_dim),
            ).unbind(dim=-3)
        else:
            key = cross_attention.k_proj(context).unflatten(
                -1, (cross_attention.n_heads, cross_attention.head_dim)
            )
            value = cross_attention.v_proj(context).unflatten(
                -1, (cross_attention.n_heads, cross_attention.head_dim)
            )
        return cross_attention.k_norm(key), cross_attention.v_norm(value)

    def merged_cross_attention(
        self,
        block_index: int,
        normalized_x: torch.Tensor,
        text_context: torch.Tensor,
        cross_attention: nn.Module,
        attn_params: Any,
    ) -> torch.Tensor:
        if self._style_context is None:
            return cross_attention(normalized_x, attn_params, text_context)
        query, text_key, text_value = cross_attention.compute_qkv(
            normalized_x, text_context
        )
        style_key, style_value = self._style_kv(block_index, cross_attention)
        null_key, null_value = self._null_style_kv(
            block_index, cross_attention, style_key.shape[0]
        )
        query, text_key, text_value, style_key, style_value = (
            _match_native_attention_dtypes(
                query, text_key, text_value, style_key, style_value, attn_params
            )
        )
        null_key = null_key.to(style_key.dtype)
        null_value = null_value.to(style_value.dtype)
        text_attended = _run_attention(
            cross_attention, query, text_key, text_value, attn_params
        )
        style_attended = _run_attention(
            cross_attention, query, style_key, style_value, attn_params
        )
        null_attended = _run_attention(
            cross_attention, query, null_key, null_value, attn_params
        )
        # Preserve the exact forward decomposition while preventing the common
        # path's gradient from cancelling when common_gain == artist_gain.
        # Centered final-artist losses train the reference path; the final
        # common loss trains the learned null path directly.
        artist_attended = style_attended - null_attended.detach()
        common_gain, artist_gain = self._component_gains()
        artist_component = artist_gain.to(
            device=artist_attended.device, dtype=artist_attended.dtype
        ) * artist_attended
        common_component = common_gain.to(
            device=artist_attended.device, dtype=artist_attended.dtype
        ) * null_attended
        effective_attended = artist_component + common_component
        if self._style_enabled is not None:
            effective_attended = effective_attended * self._style_enabled.to(
                device=effective_attended.device, dtype=effective_attended.dtype
            )[:, None, None]
        fixed_output = self._fixed_output_strength_active and not self._calibration
        if fixed_output:
            gate_context = self._block_gate_context.pop(block_index, None)
            if gate_context is None:
                raise RuntimeError(
                    "set_block_gate_context() is required before fixed-strength "
                    "style attention"
                )
            gate_cross, spatial_shape = gate_context
            frames, height, width = spatial_shape
            raw_style_delta = self._native_output_weight(
                cross_attention, effective_attended
            )
            gate = gate_cross.expand(-1, frames, height, width, -1).reshape(
                raw_style_delta.shape[0], -1, raw_style_delta.shape[-1]
            )
            dimensions = tuple(range(1, raw_style_delta.ndim))
            gated_raw = raw_style_delta.float() * gate.float()
            raw_energy = gated_raw.square().mean(dim=dimensions)
            # A dropped style row is exactly zero. sqrt(x) has an infinite
            # derivative at x=0, which can produce 0*inf NaNs even when a
            # later torch.where selects the zero branch. Keep the denominator
            # differentiable and explicitly mask inactive rows.
            safe_raw_rms = (raw_energy + 1e-8).sqrt()
            target = self._fixed_output_target(
                block_index, raw_style_delta.shape[0]
            ).to(raw_energy.device)
            active = raw_energy > 0
            scale = torch.where(
                active,
                target / safe_raw_rms,
                torch.zeros_like(safe_raw_rms),
            )
            effective_style_delta = raw_style_delta * scale.to(
                raw_style_delta.dtype
            ).reshape(-1, 1, 1)
            text_output = cross_attention.output_proj(text_attended)
            merged_output = text_output + effective_style_delta
            gated_text_rms = (
                (text_output.detach().float() * gate.float())
                .square().mean().sqrt().clamp_min(1e-8)
            )
            actual_rms = (
                (effective_style_delta.detach().float() * gate.float())
                .square().mean(dim=dimensions).sqrt()
            )
            self._runtime_ratios[block_index] = actual_rms.mean() / gated_text_rms
            self._runtime_alphas[block_index] = scale.detach().float().mean()
            self._runtime_fixed_targets[block_index] = target.detach().float().mean()
            self._runtime_fixed_actuals[block_index] = actual_rms.detach().float().mean()
            self._runtime_fixed_scales[block_index] = scale.detach().float().mean()

            if self._teacher_enabled_for_block(block_index):
                assert self._teacher_context is not None
                teacher_key, teacher_value = self._native_context_kv(
                    cross_attention, self._teacher_context
                )
                teacher_key = teacher_key.to(query.dtype)
                teacher_value = teacher_value.to(text_value.dtype)
                teacher_attended = _run_attention(
                    cross_attention, query, teacher_key, teacher_value, attn_params
                )
                teacher_delta = self._native_output_weight(
                    cross_attention, teacher_attended - text_attended
                )
                self._pending_internal[block_index] = (
                    effective_style_delta,
                    teacher_delta,
                    effective_attended,
                    scale.detach().float().reshape(-1, 1, 1),
                    self._native_output_weight(
                        cross_attention, common_component
                    ) * scale.to(common_component.dtype).reshape(-1, 1, 1),
                    self._native_output_weight(
                        cross_attention, artist_component
                    ) * scale.to(artist_component.dtype).reshape(-1, 1, 1),
                )
            return cross_attention.output_dropout(merged_output)

        alpha = self._effective_alpha(
            block_index,
            effective_attended.shape[0],
            device=effective_attended.device,
            dtype=effective_attended.dtype,
        )
        effective_style_attended = alpha * effective_attended
        self._runtime_alphas[block_index] = alpha.detach().float().mean()
        self.record_diagnostic_stage(
            block_index, "pre_o_style", effective_style_attended
        )
        merged = text_attended + (
            effective_style_attended
            if not self._calibration or self._calibration_inject_style
            else torch.zeros_like(effective_style_attended)
        )
        text_rms = text_attended.detach().float().square().mean().sqrt().clamp_min(1e-8)
        self._runtime_ratios[block_index] = (
            (alpha * effective_attended.detach()).float().square().mean().sqrt()
            / text_rms
        )

        if self._teacher_enabled_for_block(block_index):
            assert self._teacher_context is not None
            teacher_key, teacher_value = self._native_context_kv(
                cross_attention, self._teacher_context
            )
            teacher_key = teacher_key.to(query.dtype)
            teacher_value = teacher_value.to(text_value.dtype)
            teacher_attended = _run_attention(
                cross_attention, query, teacher_key, teacher_value, attn_params
            )
            teacher_attention_delta = teacher_attended - text_attended
            teacher_delta = self._native_output_weight(
                cross_attention, teacher_attention_delta
            )
            student_delta = self._native_output_weight(
                cross_attention, effective_style_attended
            )
            student_common_delta = self._native_output_weight(
                cross_attention, alpha * common_component
            )
            student_artist_delta = self._native_output_weight(
                cross_attention, alpha * artist_component
            )
            self.record_diagnostic_stage(
                block_index, "pre_o_teacher", teacher_attention_delta
            )
            self.record_diagnostic_stage(
                block_index, "post_o_style", student_delta
            )
            self.record_diagnostic_stage(
                block_index, "post_o_teacher", teacher_delta
            )
            self._pending_internal[block_index] = (
                student_delta, teacher_delta, effective_attended,
                alpha.detach().float(), student_common_delta, student_artist_delta
            )
        return cross_attention.output_dropout(cross_attention.output_proj(merged))

    def record_gated_internal_teacher(
        self,
        block_index: int,
        gate_cross: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> None:
        pair = self._pending_internal.pop(block_index, None)
        if pair is None:
            return
        (
            student,
            teacher,
            raw_style_attention,
            effective_alpha,
            student_common,
            student_artist,
        ) = pair
        frames, height, width = spatial_shape
        gate = gate_cross.expand(-1, frames, height, width, -1).reshape(
            student.shape[0], -1, student.shape[-1]
        )
        student = student * gate
        teacher = teacher * gate
        student_common = student_common * gate
        student_artist = student_artist * gate
        self.record_diagnostic_stage(block_index, "post_gate_style", student)
        self.record_diagnostic_stage(block_index, "post_gate_teacher", teacher)
        if self._post_gate_distillation_enabled:
            self._record_post_gate_distillation(
                block_index, student_artist, student_common, teacher
            )
            return
        if student.shape[0] > 1:
            student = student - student.mean(dim=0, keepdim=True)
            teacher = teacher - teacher.mean(dim=0, keepdim=True)
        dimensions = tuple(range(1, teacher.ndim))
        teacher_rms = teacher.float().square().mean(dim=dimensions).sqrt()
        student_rms = student.float().square().mean(dim=dimensions).sqrt()
        if self._calibration:
            if self._calibration_bin_index is None:
                raise RuntimeError("Calibration timestep was not set before forward")
            bin_index = self._calibration_bin_index
            raw_attention_rms = raw_style_attention.detach().float().square().mean(
                dim=dimensions
            ).sqrt()
            student_flat = student.detach().float().flatten(1)
            teacher_flat = teacher.detach().float().flatten(1)
            teacher_energy = teacher_flat.square().sum(dim=1).clamp_min(1e-8)
            projection_coefficient = (
                (student_flat * teacher_flat).sum(dim=1) / teacher_energy
            )
            cosine = F.cosine_similarity(
                student_flat, teacher_flat, dim=1, eps=1e-8
            )
            aligned_rms = projection_coefficient.clamp_min(0) * teacher_rms
            projected = teacher * projection_coefficient.reshape(-1, 1, 1)
            orthogonal_rms = (
                (student.detach().float() - projected).square()
                .mean(dim=dimensions).sqrt()
            )
            self._calibration_teacher[block_index][bin_index].append(
                teacher_rms.detach()
            )
            self._calibration_student[block_index][bin_index].append(
                student_rms.detach()
            )
            self._calibration_alpha[block_index][bin_index].append(
                effective_alpha.reshape(-1).detach()
            )
            self._calibration_raw_attention[block_index][bin_index].append(
                raw_attention_rms.detach()
            )
            self._calibration_cosine[block_index][bin_index].append(
                cosine.detach()
            )
            self._calibration_aligned_rms[block_index][bin_index].append(
                aligned_rms.detach()
            )
            self._calibration_orthogonal_rms[block_index][bin_index].append(
                orthogonal_rms.detach()
            )
        scale = teacher_rms.clamp_min(1e-4)
        broadcast = scale.reshape(-1, 1, 1)
        normalized_huber = F.smooth_l1_loss(
            student.float() / broadcast,
            teacher.detach().float() / broadcast,
            beta=0.1,
            reduction="none",
        )
        normalized_huber = normalized_huber.mean(dim=dimensions)
        flat_student = student.float().flatten(1)
        flat_teacher = teacher.detach().float().flatten(1)
        cosine = F.cosine_similarity(flat_student, flat_teacher, dim=1, eps=1e-8)
        coefficient = (flat_student * flat_teacher).sum(dim=1) / flat_teacher.square().sum(
            dim=1
        ).clamp_min(1e-8)
        projection = coefficient[:, None] * flat_teacher
        orthogonal_ratio_squared = (
            (flat_student - projection).square().mean(dim=1)
            / scale.square().clamp_min(1e-8)
        )
        # The RMS form is logging-only. Differentiating sqrt(x)^2 at x=0
        # creates a 0*inf NaN for alpha-disabled blocks; optimize energy
        # directly and take the root only after detaching.
        orthogonal_ratio = orthogonal_ratio_squared.detach().sqrt()
        aligned_rms = coefficient.clamp_min(0) * teacher_rms
        lower_target, upper_target = self._native_strength_bounds(
            block_index, teacher.shape[0]
        )
        valid = teacher_rms >= torch.quantile(teacher_rms.detach(), 0.10)
        self._internal_terms.append(
            (
                block_index,
                {
                    "huber": normalized_huber,
                    "cosine": cosine,
                    "coefficient": coefficient,
                    "orthogonal_ratio": orthogonal_ratio,
                    "orthogonal_ratio_squared": orthogonal_ratio_squared,
                    "teacher_rms": teacher_rms,
                    "student_rms": student_rms,
                    "aligned_rms": aligned_rms,
                    "native_lower_target": lower_target,
                    "native_upper_target": upper_target,
                    "valid": valid,
                },
            )
        )

    def _record_post_gate_distillation(
        self,
        block_index: int,
        student_artist: torch.Tensor,
        student_common: torch.Tensor,
        teacher: torch.Tensor,
    ) -> None:
        """Record separate native common and artist post-gate targets."""

        raw_teacher = teacher.detach().float()
        raw_student_artist = student_artist.float()
        raw_student_common = student_common.float()
        if self._post_gate_center_collection:
            self._post_gate_center_samples.setdefault(
                int(block_index), []
            ).append((
                raw_student_artist.detach(),
                raw_student_common.detach(),
                raw_teacher.detach(),
            ))
            return
        external = self._post_gate_centers.get(int(block_index))
        if external is None:
            metric_student_artist_mean = raw_student_artist.mean(
                dim=0, keepdim=True
            )
            teacher_common = raw_teacher.mean(dim=0, keepdim=True)
        else:
            metric_student_artist_mean, teacher_common = external
            metric_student_artist_mean = metric_student_artist_mean.to(
                raw_student_artist
            )
            teacher_common = teacher_common.to(raw_teacher)
        teacher_artist = raw_teacher - teacher_common
        # The artist branch is already defined as ref-null. Match it directly
        # to a zero-mean native target so null cannot disappear through another
        # student-centering invariance.
        centered_student_artist = raw_student_artist
        live_student_artist_mean = raw_student_artist.mean(dim=0, keepdim=True)
        dimensions = tuple(range(1, teacher_artist.ndim))
        teacher_row_rms = teacher_artist.square().mean(dim=dimensions).sqrt()
        student_row_rms = centered_student_artist.square().mean(dim=dimensions).sqrt()
        native_scale = teacher_artist.square().mean().sqrt().clamp_min(1e-4)
        normalized_delta = (
            centered_student_artist - teacher_artist
        ) / native_scale
        huber = F.smooth_l1_loss(
            normalized_delta,
            torch.zeros_like(normalized_delta),
            beta=0.10,
            reduction="none",
        ).mean(dim=dimensions)
        cosine = F.cosine_similarity(
            centered_student_artist.flatten(1),
            teacher_artist.flatten(1), dim=1, eps=1e-8
        )
        teacher_energy = teacher_artist.flatten(1).square().sum(dim=1).clamp_min(1e-8)
        projection_coefficient = (
            centered_student_artist.flatten(1) * teacher_artist.flatten(1)
        ).sum(dim=1) / teacher_energy
        valid = teacher_row_rms >= native_scale.detach() * 0.10
        common_scale = teacher_common.square().mean().sqrt().clamp_min(1e-4)
        combined_native_scale = (
            native_scale.square() + common_scale.square()
        ).sqrt()
        common_huber = F.smooth_l1_loss(
            (raw_student_common - teacher_common) / common_scale,
            torch.zeros_like(raw_student_common),
            beta=0.10,
        )
        common_cosine = F.cosine_similarity(
            raw_student_common.flatten(1),
            teacher_common.expand_as(raw_student_common).flatten(1),
            dim=1, eps=1e-8,
        ).mean()
        common_projection = (
            raw_student_common.flatten(1)
            * teacher_common.expand_as(raw_student_common).flatten(1)
        ).sum(dim=1) / teacher_common.flatten(1).square().sum(dim=1).clamp_min(1e-8)
        common_projection = common_projection.mean()
        artist_common_leakage = (
            metric_student_artist_mean.square().mean().sqrt()
            / native_scale.detach().clamp_min(1e-8)
        )
        artist_common_leakage_loss = (
            live_student_artist_mean.square().mean()
            / native_scale.detach().square().clamp_min(1e-8)
        )
        self._post_gate_distillation_terms.append(
            (
                int(block_index),
                {
                    "huber": huber,
                    "cosine": cosine,
                    "projection_coefficient": projection_coefficient,
                    "teacher_rms": teacher_row_rms,
                    "student_rms": student_row_rms,
                    "native_scale": native_scale.expand_as(teacher_row_rms),
                    "valid": valid,
                    "common_huber": common_huber,
                    "common_cosine": common_cosine,
                    "common_projection_coefficient": common_projection,
                    "common_teacher_rms": common_scale,
                    "common_student_rms": raw_student_common.square().mean().sqrt(),
                    "artist_common_leakage": artist_common_leakage,
                    "artist_common_leakage_loss": artist_common_leakage_loss,
                    "combined_native_scale": combined_native_scale,
                },
            )
        )

    def post_gate_teacher_loss(
        self,
        *,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.0,
        huber_weight: float = 0.0,
        magnitude_lower: float = 0.50,
        magnitude_upper: float = 1.25,
        cosine_weight: float | None = None,
        native_strength_weighting: bool = True,
        strength_weight_min: float = 0.25,
        strength_weight_max: float = 4.0,
        common_weight: float = 0.25,
        artist_common_leakage_weight: float = 0.10,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Align post-gate direction first, then its teacher-axis magnitude.

        Direction is measured on normalized artist-centered rows, so the
        calibrated alpha affects the real forward strength without suppressing
        the direction-learning signal.  Magnitude is a separate projection
        band and can stay disabled during the bootstrap stage.
        """

        if cosine_weight is not None:
            # Compatibility for diagnostics created before the direction and
            # magnitude objectives were separated.
            direction_weight = float(cosine_weight)
        if min(
            direction_weight, magnitude_weight, huber_weight,
            common_weight, artist_common_leakage_weight,
        ) < 0:
            raise ValueError("Post-gate loss weights must be non-negative")
        if magnitude_lower < 0 or magnitude_upper < magnitude_lower:
            raise ValueError("Invalid post-gate magnitude band")
        if not self._post_gate_distillation_terms:
            reference = self.alpha.float().sum() * 0.0
            return reference, {}
        block_losses: list[torch.Tensor] = []
        block_strengths: list[torch.Tensor] = []
        block_cosines: list[torch.Tensor] = []
        block_coefficients: list[torch.Tensor] = []
        block_common_cosines: list[torch.Tensor] = []
        block_common_coefficients: list[torch.Tensor] = []
        block_artist_common_leakages: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] = {}
        for block_index, values in self._post_gate_distillation_terms:
            valid = values["valid"]
            weights = valid.to(values["huber"].dtype)
            if native_strength_weighting:
                relative = (
                    values["teacher_rms"].detach()
                    / values["native_scale"].detach().clamp_min(1e-8)
                ).clamp(float(strength_weight_min), float(strength_weight_max))
                weights = weights * relative
            denominator = weights.sum().clamp_min(1.0)
            huber = (values["huber"] * weights).sum() / denominator
            direction = ((1.0 - values["cosine"]) * weights).sum() / denominator
            coefficient = values["projection_coefficient"]
            magnitude_lower_loss = (
                F.relu(float(magnitude_lower) - coefficient).square() * weights
            ).sum() / denominator
            magnitude_upper_loss = (
                F.relu(coefficient - float(magnitude_upper)).square() * weights
            ).sum() / denominator
            magnitude = magnitude_lower_loss + 0.25 * magnitude_upper_loss
            common = (
                (1.0 - values["common_cosine"])
                + values["common_huber"]
                + F.relu(0.50 - values["common_projection_coefficient"]).square()
            )
            artist_common_leakage = values["artist_common_leakage_loss"]
            loss = (
                float(direction_weight) * direction
                + float(magnitude_weight) * magnitude
                + float(huber_weight) * huber
                + float(common_weight) * common
                + float(artist_common_leakage_weight) * artist_common_leakage
            )
            block_losses.append(loss)
            block_strengths.append(values["combined_native_scale"].detach())
            block_cosines.append(
                (values["cosine"].detach() * weights).sum() / denominator
            )
            block_coefficients.append(
                (coefficient.detach() * weights).sum() / denominator
            )
            block_common_cosines.append(values["common_cosine"].detach())
            block_common_coefficients.append(
                values["common_projection_coefficient"].detach()
            )
            block_artist_common_leakages.append(
                values["artist_common_leakage"].detach()
            )
            prefix = f"post_gate_teacher_block_{block_index}"
            metrics.update({
                f"{prefix}_loss": loss.detach(),
                f"{prefix}_huber": huber.detach(),
                f"{prefix}_direction_loss": direction.detach(),
                f"{prefix}_magnitude_loss": magnitude.detach(),
                f"{prefix}_magnitude_lower_loss": magnitude_lower_loss.detach(),
                f"{prefix}_magnitude_upper_loss": magnitude_upper_loss.detach(),
                f"{prefix}_projection_coefficient": (
                    coefficient.detach() * weights
                ).sum() / denominator,
                f"{prefix}_cosine": (
                    values["cosine"].detach() * weights
                ).sum() / denominator,
                f"{prefix}_teacher_rms": (
                    values["teacher_rms"].detach() * weights
                ).sum() / denominator,
                f"{prefix}_student_rms": (
                    values["student_rms"].detach() * weights
                ).sum() / denominator,
                f"{prefix}_native_scale": values["native_scale"].detach().mean(),
                f"{prefix}_valid_fraction": valid.detach().float().mean(),
                f"{prefix}_common_loss": common.detach(),
                f"{prefix}_common_cosine": values["common_cosine"].detach(),
                f"{prefix}_common_projection_coefficient": values[
                    "common_projection_coefficient"
                ].detach(),
                f"{prefix}_common_teacher_rms": values[
                    "common_teacher_rms"
                ].detach(),
                f"{prefix}_common_student_rms": values[
                    "common_student_rms"
                ].detach(),
                f"{prefix}_artist_common_leakage": values[
                    "artist_common_leakage"
                ].detach(),
            })
        self._post_gate_distillation_terms.clear()
        if not block_losses:
            reference = self.alpha.float().sum() * 0.0
            return reference, metrics
        stacked_losses = torch.stack(block_losses)
        if native_strength_weighting:
            strengths = torch.stack(block_strengths)
            reference = strengths.median().clamp_min(1e-8)
            block_weights = (strengths / reference).clamp(
                float(strength_weight_min), float(strength_weight_max)
            )
            block_weights = block_weights / block_weights.mean().clamp_min(1e-8)
            total = (stacked_losses * block_weights).sum() / block_weights.sum()
        else:
            block_weights = torch.ones_like(stacked_losses)
            total = stacked_losses.mean()
        metrics.update({
            "post_gate_teacher_loss": total.detach(),
            "post_gate_teacher_blocks": total.new_tensor(len(block_losses)),
            "post_gate_teacher_direction_weight": total.new_tensor(
                float(direction_weight)
            ),
            "post_gate_teacher_magnitude_weight": total.new_tensor(
                float(magnitude_weight)
            ),
            "post_gate_teacher_huber_weight": total.new_tensor(
                float(huber_weight)
            ),
            "post_gate_teacher_magnitude_lower": total.new_tensor(
                float(magnitude_lower)
            ),
            "post_gate_teacher_magnitude_upper": total.new_tensor(
                float(magnitude_upper)
            ),
            "post_gate_teacher_native_strength_weighting": total.new_tensor(
                float(native_strength_weighting)
            ),
            "post_gate_teacher_block_weight_max": block_weights.detach().max(),
            "post_gate_teacher_block_weight_min": block_weights.detach().min(),
            "post_gate_teacher_cosine": (
                torch.stack(block_cosines) * block_weights
            ).sum() / block_weights.sum(),
            "post_gate_teacher_projection_coefficient": (
                torch.stack(block_coefficients) * block_weights
            ).sum() / block_weights.sum(),
            "post_gate_teacher_common_cosine": (
                torch.stack(block_common_cosines) * block_weights
            ).sum() / block_weights.sum(),
            "post_gate_teacher_common_projection_coefficient": (
                torch.stack(block_common_coefficients) * block_weights
            ).sum() / block_weights.sum(),
            "post_gate_teacher_artist_common_leakage": (
                torch.stack(block_artist_common_leakages) * block_weights
            ).sum() / block_weights.sum(),
            "post_gate_teacher_common_weight": total.new_tensor(float(common_weight)),
            "post_gate_teacher_artist_common_leakage_weight": total.new_tensor(
                float(artist_common_leakage_weight)
            ),
        })
        return total, metrics

    def internal_teacher_loss(
        self,
        *,
        rho_min: float = 0.0,
        rho_max: float = 1.5,
        aligned_floor_weight: float = 1.0,
        total_upper_weight: float = 0.05,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self._internal_terms:
            reference = self.alpha.float().sum() * 0.0
            return reference, {}
        losses = []
        collected: dict[str, list[torch.Tensor]] = {}
        for _, metrics in self._internal_terms:
            valid = metrics["valid"]

            def selected_mean(value: torch.Tensor) -> torch.Tensor:
                return value[valid].mean()

            huber = selected_mean(metrics["huber"])
            coefficient = metrics["coefficient"]
            floor = selected_mean(
                F.relu(coefficient.new_tensor(float(rho_min)) - coefficient).square()
            )
            upper = selected_mean(F.relu(coefficient - float(rho_max)).square())
            lower_target = metrics["native_lower_target"].clamp_min(1e-8)
            aligned_floor = selected_mean(
                (
                    F.relu(lower_target - metrics["aligned_rms"])
                    / lower_target
                ).square()
            )
            upper_target = metrics["native_upper_target"]
            finite_upper = torch.isfinite(upper_target)
            total_upper = selected_mean(
                torch.where(
                    finite_upper,
                    (
                        F.relu(metrics["student_rms"] - upper_target)
                        / upper_target.clamp_min(1e-8)
                    ).square(),
                    torch.zeros_like(upper_target),
                )
            )
            direction = selected_mean(1.0 - metrics["cosine"])
            orthogonal = selected_mean(metrics["orthogonal_ratio_squared"])
            losses.append(
                0.25 * huber + 0.10 * direction + 0.05 * (floor + upper)
                + 0.02 * orthogonal
                + float(aligned_floor_weight) * aligned_floor
                + float(total_upper_weight) * total_upper
            )
            local = {
                key: selected_mean(value.detach().float())
                for key, value in metrics.items()
                if key not in {"valid", "orthogonal_ratio_squared"}
            }
            local["valid_fraction"] = valid.float().mean()
            local.update({
                "floor": floor.detach(),
                "upper": upper.detach(),
                "aligned_floor": aligned_floor.detach(),
                "total_upper": total_upper.detach(),
            })
            for key, value in local.items():
                collected.setdefault(key, []).append(value)
        loss = torch.stack(losses).mean()
        result = {
            f"internal_teacher_{key}": torch.stack(values).mean()
            for key, values in collected.items()
        }
        result["internal_teacher_loss"] = loss.detach()
        self._internal_terms.clear()
        return loss, result

    def runtime_stats(self) -> dict[str, float]:
        active = [value for value in self._runtime_ratios if value is not None]
        values = (
            torch.stack(active).float().cpu()
            if active
            else torch.zeros(1, dtype=torch.float32)
        )
        active_alpha = [value for value in self._runtime_alphas if value is not None]
        runtime_alpha = (
            torch.stack(active_alpha).float().cpu()
            if active_alpha
            else self.alpha.detach().float().cpu()
        )
        fixed_targets = [
            value for value in self._runtime_fixed_targets if value is not None
        ]
        fixed_actuals = [
            value for value in self._runtime_fixed_actuals if value is not None
        ]
        fixed_scales = [
            value for value in self._runtime_fixed_scales if value is not None
        ]
        target_values = torch.stack(fixed_targets).float().cpu() if fixed_targets else torch.zeros(1)
        actual_values = torch.stack(fixed_actuals).float().cpu() if fixed_actuals else torch.zeros(1)
        scale_values = torch.stack(fixed_scales).float().cpu() if fixed_scales else torch.zeros(1)
        common_gain, artist_gain = self._component_gains()
        return {
            "style_block_residual_ratio_mean": float(values.mean()),
            "style_block_residual_ratio_max": float(values.max()),
            "style_alpha_mean": float(self.alpha.mean()),
            "style_alpha_max": float(self.alpha.max()),
            "style_alpha_active_blocks": float(self.alpha.ne(0).sum()),
            "style_alpha_runtime_mean": float(runtime_alpha.mean()),
            "style_alpha_runtime_max": float(runtime_alpha.max()),
            "style_global_gain": float(getattr(self, "global_gain", 1.0)),
            "style_common_gain": float(common_gain.detach()),
            "style_artist_gain": float(artist_gain.detach()),
            "style_fixed_output_enabled": float(self._fixed_output_strength_active),
            "style_fixed_target_mean": float(target_values.mean()),
            "style_fixed_actual_mean": float(actual_values.mean()),
            "style_fixed_scale_mean": float(scale_values.mean()),
            "style_fixed_scale_max": float(scale_values.max()),
        }


class SharedBaseKVStyleCrossAttention(FreshKVStyleCrossAttention):
    """Four fresh Xavier K/V bases plus a low-rank block-specific residual.

    The shared full-rank projections are evaluated once per active style
    context.  Each Anima block softly selects the four bases and adds its own
    rank-limited delta before the native K/V normalization.  Native Q/O and
    the separate text/style softmax remain implemented by the parent class.
    """

    def __init__(
        self,
        *,
        context_dim: int = 1024,
        blocks: int = 28,
        shared_bases: int = 4,
        medoid_blocks: list[int] | tuple[int, ...] = (3, 12, 18, 26),
        block_to_base: list[int] | tuple[int, ...] = (
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2,
            3, 3, 3, 3, 3, 3, 3, 3,
        ),
        delta_rank: int = 64,
        delta_scale: float = 1.0,
        delta_k_init_scale: float = 0.01,
        delta_v_init_scale: float = 0.03,
        mix_logit_scale: float = 4.0,
        global_gain: float = 1.0,
        null_tokens: int = 28,
        null_init_std: float = 0.02,
        common_gain: float = 1.0,
        artist_gain: float = 1.0,
        gain_maximum: float = 100.0,
        relative_block_gain: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if shared_bases <= 0 or delta_rank <= 0:
            raise ValueError("shared_bases and delta_rank must be positive")
        if delta_k_init_scale <= 0 or delta_v_init_scale <= 0:
            raise ValueError("Block-delta initialization scales must be positive")
        if len(medoid_blocks) != shared_bases:
            raise ValueError("medoid_blocks must contain one block per shared base")
        if len(block_to_base) != blocks:
            raise ValueError("block_to_base must contain one assignment per block")
        if min(block_to_base) < 0 or max(block_to_base) >= shared_bases:
            raise ValueError("block_to_base contains an invalid base index")
        gains = [1.0] * blocks if relative_block_gain is None else list(relative_block_gain)
        if len(gains) != blocks or any(value <= 0 for value in gains):
            raise ValueError("relative_block_gain must be positive for every block")
        if global_gain <= 0:
            raise ValueError("global_gain must be positive")

        super().__init__(
            context_dim=context_dim,
            blocks=blocks,
            initial_alpha=1.0,
            null_tokens=null_tokens,
            null_init_std=null_init_std,
            common_gain=common_gain,
            artist_gain=artist_gain,
            gain_maximum=gain_maximum,
        )
        self.shared_bases = int(shared_bases)
        self.delta_rank = int(delta_rank)
        self.delta_scale = float(delta_scale)
        self.delta_k_init_scale = float(delta_k_init_scale)
        self.delta_v_init_scale = float(delta_v_init_scale)
        self.global_gain = float(global_gain)
        self.medoid_blocks = tuple(int(value) for value in medoid_blocks)
        self.register_buffer(
            "block_to_base",
            torch.tensor(block_to_base, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "relative_block_gain",
            torch.tensor(gains, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "relative_block_gain_normalized",
            self.relative_block_gain / self.relative_block_gain.mean(),
            persistent=True,
        )
        # This is only the pre-calibration value.  Production initialization
        # replaces it with native-effect-calibrated absolute scales while
        # retaining this vector solely as a mean-one relative profile.
        self.alpha.copy_(self.relative_block_gain_normalized * self.global_gain)
        logits = torch.zeros(blocks, shared_bases, dtype=torch.float32)
        logits[torch.arange(blocks), self.block_to_base.cpu()] = float(mix_logit_scale)
        self.base_mix_logits = nn.Parameter(logits)
        self.base_k = nn.ModuleList()
        self.base_v = nn.ModuleList()
        self.delta_k_down = nn.ModuleList()
        self.delta_k_up = nn.ModuleList()
        self.delta_v_down = nn.ModuleList()
        self.delta_v_up = nn.ModuleList()
        self._base_projection_cache: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
        self._null_base_projection_cache: (
            tuple[list[torch.Tensor], list[torch.Tensor]] | None
        ) = None

    def initialize_from_anima(self, anima: nn.Module) -> None:
        if self._initialized:
            return
        if len(anima.blocks) != self.blocks:
            raise ValueError(f"Expected {self.blocks} Anima blocks, got {len(anima.blocks)}")
        if any(index < 0 or index >= self.blocks for index in self.medoid_blocks):
            raise ValueError("medoid_blocks contains an invalid Anima block")

        output_dim: int | None = None
        for block_index in self.medoid_blocks:
            cross = anima.blocks[block_index].cross_attn
            current_output_dim = int(cross.n_heads) * int(cross.head_dim)
            output_dim = current_output_dim if output_dim is None else output_dim
            if current_output_dim != output_dim:
                raise ValueError("All shared-base blocks must use the same K/V width")
            native = cross.output_proj.weight
            key = nn.Linear(
                self.context_dim, output_dim, bias=False,
                device=native.device, dtype=native.dtype,
            )
            value = nn.Linear(
                self.context_dim, output_dim, bias=False,
                device=native.device, dtype=native.dtype,
            )
            with torch.no_grad():
                nn.init.xavier_uniform_(key.weight)
                nn.init.xavier_uniform_(value.weight)
            self.base_k.append(key)
            self.base_v.append(value)

        assert output_dim is not None
        native = anima.blocks[0].cross_attn.output_proj.weight
        for _ in range(self.blocks):
            modules = []
            for input_dim, result_dim, init_scale in (
                (self.context_dim, self.delta_rank, 1.0),
                (self.delta_rank, output_dim, self.delta_k_init_scale),
                (self.context_dim, self.delta_rank, 1.0),
                (self.delta_rank, output_dim, self.delta_v_init_scale),
            ):
                module = nn.Linear(
                    input_dim, result_dim, bias=False,
                    device=native.device, dtype=native.dtype,
                )
                nn.init.xavier_uniform_(module.weight)
                if init_scale != 1.0:
                    with torch.no_grad():
                        module.weight.mul_(float(init_scale))
                modules.append(module)
            k_down, k_up, v_down, v_up = modules
            self.delta_k_down.append(k_down)
            self.delta_k_up.append(k_up)
            self.delta_v_down.append(v_down)
            self.delta_v_up.append(v_up)
        self._initialized = True

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.base_k.parameters()) + list(self.base_v.parameters())

    def delta_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.delta_k_down.parameters())
            + list(self.delta_k_up.parameters())
            + list(self.delta_v_down.parameters())
            + list(self.delta_v_up.parameters())
        )

    def mixing_parameters(self) -> list[nn.Parameter]:
        return [self.base_mix_logits]

    def kv_parameters(self) -> list[nn.Parameter]:
        return (
            self.shared_parameters()
            + self.delta_parameters()
            + self.mixing_parameters()
            + self.null_parameters()
            + self.gain_parameters()
        )

    def set_style_context(
        self,
        tokens: torch.Tensor,
        *,
        enabled: torch.Tensor | None = None,
        strength: float = 1.0,
    ) -> None:
        self._base_projection_cache = None
        self._null_base_projection_cache = None
        super().set_style_context(tokens, enabled=enabled, strength=strength)

    set_style_tokens = set_style_context

    def clear_style_tokens(self) -> None:
        self._base_projection_cache = None
        self._null_base_projection_cache = None
        super().clear_style_tokens()

    def _base_projections(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._style_context is None:
            raise RuntimeError("No active style context")
        if self._base_projection_cache is None:
            self._base_projection_cache = (
                [module(self._style_context) for module in self.base_k],
                [module(self._style_context) for module in self.base_v],
            )
        return self._base_projection_cache

    def _null_base_projections(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._null_base_projection_cache is None:
            self._null_base_projection_cache = (
                [module(self.null_style_context) for module in self.base_k],
                [module(self.null_style_context) for module in self.base_v],
            )
        return self._null_base_projection_cache

    def _style_kv(
        self, index: int, cross_attention: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._style_context is None:
            raise RuntimeError("No active style context")
        base_keys, base_values = self._base_projections()
        weights = self.base_mix_logits[index].float().softmax(dim=0)
        key = sum(
            weight.to(value.dtype) * value
            for weight, value in zip(weights, base_keys, strict=True)
        )
        value = sum(
            weight.to(item.dtype) * item
            for weight, item in zip(weights, base_values, strict=True)
        )
        key = key + self.delta_scale * self.delta_k_up[index](
            self.delta_k_down[index](self._style_context)
        )
        value = value + self.delta_scale * self.delta_v_up[index](
            self.delta_v_down[index](self._style_context)
        )
        key = rearrange(
            key, "b s (h d) -> b s h d",
            h=cross_attention.n_heads, d=cross_attention.head_dim,
        )
        value = rearrange(
            value, "b s (h d) -> b s h d",
            h=cross_attention.n_heads, d=cross_attention.head_dim,
        )
        return cross_attention.k_norm(key), cross_attention.v_norm(value)

    def _null_style_kv(
        self, index: int, cross_attention: nn.Module, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_keys, base_values = self._null_base_projections()
        weights = self.base_mix_logits[index].float().softmax(dim=0)
        key = sum(
            weight.to(value.dtype) * value
            for weight, value in zip(weights, base_keys, strict=True)
        )
        value = sum(
            weight.to(item.dtype) * item
            for weight, item in zip(weights, base_values, strict=True)
        )
        key = key + self.delta_scale * self.delta_k_up[index](
            self.delta_k_down[index](self.null_style_context)
        )
        value = value + self.delta_scale * self.delta_v_up[index](
            self.delta_v_down[index](self.null_style_context)
        )
        key = rearrange(
            key, "b s (h d) -> b s h d",
            h=cross_attention.n_heads, d=cross_attention.head_dim,
        )
        value = rearrange(
            value, "b s (h d) -> b s h d",
            h=cross_attention.n_heads, d=cross_attention.head_dim,
        )
        return (
            cross_attention.k_norm(key).expand(batch, -1, -1, -1),
            cross_attention.v_norm(value).expand(batch, -1, -1, -1),
        )

    def runtime_stats(self) -> dict[str, float]:
        result = super().runtime_stats()
        probabilities = self.base_mix_logits.detach().float().softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        result.update({
            "style_base_mix_entropy": float(entropy.mean()),
            "style_base_mix_assigned_probability": float(
                probabilities[
                    torch.arange(self.blocks, device=probabilities.device),
                    self.block_to_base,
                ].mean()
            ),
        })
        return result
