"""Detail-preserving typed reader and separate Style Cross-Attention for Anima.

The frozen Dual-query Resampler supplies 64 spatial, 16 global, and four
artist-summary tokens per reference.  This module reads each reference into 28
canonical, reference-conditioned slots, pools matching slots across the
unordered reference set, and injects the result through fresh block-local K/V
projections.  Native Anima Q, O, output dropout, and gate_cross remain frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    """Convert each 84-token reference into a fixed 28-token style set."""

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
        position_gain: float = 0.1,
        type_preference_bias: float = 1.0,
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
        self.dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.summary_tokens = int(summary_tokens)
        self.cached_tokens = spatial_tokens + global_tokens + summary_tokens
        self.output_tokens = int(output_tokens)
        self.position_gain = float(position_gain)

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
        self.set_attention = _BiasFreeAttention(dim, heads)
        self.set_ff_norm = nn.LayerNorm(dim)
        self.set_ff = _SwiGLU(dim, reader_ff_dim)
        self.mixer = _CrossSlotBlock(dim, heads, mixer_ff_dim)

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
        values = per_reference.permute(0, 2, 1, 3).reshape(batch * slots, references, dim)
        query = self.set_query.to(values.dtype)[None].expand(batch, -1, -1)
        query = query.reshape(batch * slots, 1, dim)
        mask = reference_mask[:, None].expand(-1, slots, -1).reshape(
            batch * slots, references
        )
        pooled, _ = self.set_attention(
            query,
            self.set_norm(values),
            self.set_norm(values),
            key_padding_mask=~mask,
        )
        pooled = pooled.reshape(batch, slots, dim)
        pooled = pooled + self.set_ff(self.set_ff_norm(pooled))
        identity = self.slot_identity.to(pooled.dtype).unsqueeze(0)
        return self.mixer(pooled, identity)

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
        if reconstruct:
            queries = self.reconstruction_queries.to(encoded.dtype)[None].expand(
                encoded.shape[0], -1, -1
            )
            reconstruction, _ = self.reconstruction_attention(
                queries,
                self.reconstruction_norm(encoded),
                self.reconstruction_norm(encoded),
            )
            reconstruction = self.reconstruction_output(reconstruction)
            reconstruction_target = targets.detach()
        return DetailStyleOutput(
            tokens=tokens,
            per_reference_tokens=per_reference,
            reconstruction=reconstruction,
            reconstruction_target=reconstruction_target,
        )


class FreshKVStyleCrossAttention(nn.Module):
    """Fresh per-block K/V with native Anima Q/O and separate softmax."""

    def __init__(
        self,
        *,
        context_dim: int = 1024,
        blocks: int = 28,
        initial_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.blocks = int(blocks)
        self.style_k = nn.ModuleList()
        self.style_v = nn.ModuleList()
        self.register_buffer(
            "alpha", torch.full((blocks,), float(initial_alpha)), persistent=True
        )
        self._style_context: torch.Tensor | None = None
        self._style_enabled: torch.Tensor | None = None
        self._style_strength = 1.0
        self._teacher_context: torch.Tensor | None = None
        self._pending_internal: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._internal_terms: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._calibration = False
        self._calibration_teacher: list[list[torch.Tensor]] = [
            [] for _ in range(blocks)
        ]
        self._calibration_student: list[list[torch.Tensor]] = [
            [] for _ in range(blocks)
        ]
        self._runtime_ratios: list[torch.Tensor | None] = [None] * blocks
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
        return list(self.style_k.parameters()) + list(self.style_v.parameters())

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

    def set_teacher_context(self, context: torch.Tensor | None) -> None:
        self._teacher_context = context

    def clear_style_tokens(self) -> None:
        self._style_context = None
        self._style_enabled = None
        self._style_strength = 1.0
        self._teacher_context = None
        self._pending_internal.clear()

    def reset_internal_teacher(self) -> None:
        self._pending_internal.clear()
        self._internal_terms.clear()

    def begin_alpha_calibration(self) -> None:
        self._calibration = True
        # Calibration measures the raw alpha=1 branch.  Otherwise the result
        # would be divided by the arbitrary constructor value a second time.
        self.alpha.fill_(1.0)
        self._calibration_teacher = [[] for _ in range(self.blocks)]
        self._calibration_student = [[] for _ in range(self.blocks)]

    @torch.no_grad()
    def finish_alpha_calibration(
        self,
        *,
        minimum: float = 0.02,
        maximum: float = 2.0,
        weak_block_fraction: float = 0.1,
    ) -> dict[str, Any]:
        teacher = torch.stack(
            [torch.stack(values).median() for values in self._calibration_teacher]
        )
        student = torch.stack(
            [torch.stack(values).median() for values in self._calibration_student]
        )
        alpha = (teacher / student.clamp_min(1e-8)).clamp(minimum, maximum)
        alpha[teacher < teacher.median() * float(weak_block_fraction)] = 0
        self.alpha.copy_(alpha.to(self.alpha))
        self._calibration = False
        return {
            "teacher_rms": teacher.cpu().tolist(),
            "student_rms": student.cpu().tolist(),
            "alpha": alpha.cpu().tolist(),
        }

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
        query, text_key, text_value, style_key, style_value = (
            _match_native_attention_dtypes(
                query, text_key, text_value, style_key, style_value, attn_params
            )
        )
        text_attended = _run_attention(
            cross_attention, query, text_key, text_value, attn_params
        )
        style_attended = _run_attention(
            cross_attention, query, style_key, style_value, attn_params
        )
        if self._style_enabled is not None:
            style_attended = style_attended * self._style_enabled.to(
                device=style_attended.device, dtype=style_attended.dtype
            )[:, None, None, None]
        alpha = self.alpha[block_index].to(style_attended.dtype) * float(
            self._style_strength
        )
        merged = text_attended + alpha * style_attended
        text_rms = text_attended.detach().float().square().mean().sqrt().clamp_min(1e-8)
        self._runtime_ratios[block_index] = (
            (alpha * style_attended.detach()).float().square().mean().sqrt() / text_rms
        )

        if self._teacher_context is not None:
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
            student_delta = self._native_output_weight(
                cross_attention, alpha * style_attended
            )
            self._pending_internal[block_index] = (student_delta, teacher_delta)
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
        student, teacher = pair
        frames, height, width = spatial_shape
        gate = gate_cross.expand(-1, frames, height, width, -1).reshape(
            student.shape[0], -1, student.shape[-1]
        )
        student = student * gate
        teacher = teacher * gate
        if student.shape[0] > 1:
            student = student - student.mean(dim=0, keepdim=True)
            teacher = teacher - teacher.mean(dim=0, keepdim=True)
        dimensions = tuple(range(1, teacher.ndim))
        teacher_rms = teacher.float().square().mean(dim=dimensions).sqrt()
        student_rms = student.float().square().mean(dim=dimensions).sqrt()
        if self._calibration:
            self._calibration_teacher[block_index].append(teacher_rms.median().detach())
            self._calibration_student[block_index].append(student_rms.median().detach())
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
        orthogonal = (flat_student - projection).square().mean(dim=1).sqrt() / scale
        valid = teacher_rms >= torch.quantile(teacher_rms.detach(), 0.10)
        self._internal_terms.append(
            (
                block_index,
                {
                    "huber": normalized_huber,
                    "cosine": cosine,
                    "coefficient": coefficient,
                    "orthogonal_ratio": orthogonal,
                    "teacher_rms": teacher_rms,
                    "student_rms": student_rms,
                    "valid": valid,
                },
            )
        )

    def internal_teacher_loss(
        self,
        *,
        rho_min: float,
        rho_max: float = 1.5,
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
            direction = selected_mean(1.0 - metrics["cosine"])
            orthogonal = selected_mean(metrics["orthogonal_ratio"].square())
            losses.append(
                0.25 * huber + 0.10 * direction + 0.05 * (floor + upper)
                + 0.02 * orthogonal
            )
            local = {
                key: selected_mean(value.detach().float())
                for key, value in metrics.items()
                if key != "valid"
            }
            local["valid_fraction"] = valid.float().mean()
            local.update({"floor": floor.detach(), "upper": upper.detach()})
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
        return {
            "style_block_residual_ratio_mean": float(values.mean()),
            "style_block_residual_ratio_max": float(values.max()),
            "style_alpha_mean": float(self.alpha.mean()),
            "style_alpha_max": float(self.alpha.max()),
            "style_alpha_active_blocks": float(self.alpha.ne(0).sum()),
        }
