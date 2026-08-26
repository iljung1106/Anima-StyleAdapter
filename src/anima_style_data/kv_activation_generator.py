"""Reference-conditioned native text K/V activation generation.

The student predicts the raw pre-normalization K/V residual used by a K/V-only
LoRA, rather than generating LoRA factors or adding a separate style-attention
branch.  Offline LoRA factors and materialized merged-LoRA images are training
data only; inference receives visual references and the current text context.
"""

from __future__ import annotations

import copy
import gc
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json, write_records
from .kv_activation_modulation import apply_kv_factors, load_kv_lora_factor_bank
from .lora_functional_distillation import build_mixture_specs
from .lora_oracle_bootstrap import (
    _materialize_reader_code_bank,
    _oracle_detail_config,
)


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, hidden * 2, bias=False)
        self.output = nn.Linear(hidden, dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, content = self.input(values).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


class ReferenceConditionedKVActivationGenerator(nn.Module):
    """Generate block-local raw delta-K/delta-V activations.

    The 512 text tokens query visual style memory.  Every Anima block owns its
    context projection and final K/V head; only the visual memory projection
    and the small feed-forward trunk are shared.
    """

    def __init__(
        self,
        *,
        style_dim: int = 1024,
        context_dim: int = 1024,
        output_dim: int = 2048,
        blocks: int = 28,
        hidden_dim: int = 256,
        heads: int = 8,
        ff_dim: int = 1024,
        output_init_scale: float = 0.02,
        normalize_style: bool = True,
        normalize_attended: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if output_init_scale <= 0:
            raise ValueError("output_init_scale must be positive")
        self.style_dim = int(style_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.blocks = int(blocks)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.style_norm = (
            nn.LayerNorm(style_dim) if bool(normalize_style) else nn.Identity()
        )
        self.context_norm = nn.LayerNorm(context_dim)
        self.style_key = nn.Linear(style_dim, hidden_dim, bias=False)
        self.style_value = nn.Linear(style_dim, hidden_dim, bias=False)
        self.block_embedding = nn.Embedding(blocks, hidden_dim)
        self.context_query = nn.ModuleList(
            nn.Linear(context_dim, hidden_dim, bias=False) for _ in range(blocks)
        )
        self.output_norm = (
            nn.LayerNorm(hidden_dim) if bool(normalize_attended) else nn.Identity()
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = _SwiGLU(hidden_dim, ff_dim)
        self.output_head = nn.ModuleList(
            nn.Linear(hidden_dim, output_dim * 2, bias=False)
            for _ in range(blocks)
        )
        self.log_gain = nn.Parameter(torch.zeros(blocks, 2))
        self.reset_parameters(float(output_init_scale))

    def reset_parameters(self, output_init_scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.block_embedding.weight, std=self.hidden_dim**-0.5)
        for head in self.output_head:
            with torch.no_grad():
                head.weight.mul_(output_init_scale)

    def forward(
        self,
        style_memory: torch.Tensor,
        text_context: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if style_memory.ndim != 3 or style_memory.shape[-1] != self.style_dim:
            raise ValueError("style_memory must be [batch,slots,style_dim]")
        if text_context.ndim != 3 or text_context.shape[-1] != self.context_dim:
            raise ValueError("text_context must be [batch,tokens,context_dim]")
        if style_memory.shape[0] != text_context.shape[0]:
            raise ValueError("style and text batches disagree")
        block = int(block_index)
        if not 0 <= block < self.blocks:
            raise ValueError("block_index is outside the model")
        style = self.style_norm(style_memory)
        block_code = self.block_embedding.weight[block].to(style.dtype)
        key = self.style_key(style) + block_code[None, None]
        value = self.style_value(style)
        query = self.context_query[block](self.context_norm(text_context))
        batch, text_tokens, _ = query.shape
        style_tokens = int(style.shape[1])
        query = query.reshape(
            batch, text_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch, style_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch, style_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(
            batch, text_tokens, self.hidden_dim
        )
        hidden = self.output_norm(attended)
        hidden = hidden + self.ff(self.ff_norm(hidden))
        output = self.output_head[block](hidden).reshape(
            batch, text_tokens, 2, self.output_dim
        ).transpose(1, 2)
        gain = self.log_gain[block].float().clamp(-4.0, 4.0).exp().to(output.dtype)
        return output * gain[None, :, None, None]


class _OperatorCrossBlock(nn.Module):
    """Let operator queries read the complete typed reference memory."""

    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("operator hidden dimension must divide heads")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, ff_dim)

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        batch, query_tokens, dim = queries.shape
        memory_tokens = int(memory.shape[1])
        query = self.query(self.query_norm(queries)).reshape(
            batch, query_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        normalized_memory = self.memory_norm(memory)
        key = self.key(normalized_memory).reshape(
            batch, memory_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized_memory).reshape(
            batch, memory_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, query_tokens, dim)
        queries = queries + self.output(attended)
        return queries + self.ff(self.ff_norm(queries))


class ReferenceConditionedLowRankKVOperator(nn.Module):
    """Generate a fresh low-rank text K/V operator from each reference set.

    The reference does not emit extra style-attention tokens.  Instead it
    generates normalized input/output directions and singular strengths for
    a block-local operator, then applies that operator to all native text
    tokens.  This preserves the exact algebraic form of a K/V-only LoRA while
    never exposing teacher factors, artist IDs, or mixture coefficients.
    """

    def __init__(
        self,
        *,
        style_dim: int = 1024,
        context_dim: int = 1024,
        output_dim: int = 2048,
        blocks: int = 28,
        hidden_dim: int = 256,
        heads: int = 8,
        ff_dim: int = 1024,
        operator_layers: int = 2,
        operator_rank: int = 32,
        initial_sigma: float = 0.01,
        minimum_sigma: float = 1e-6,
        maximum_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if operator_layers <= 0 or operator_rank <= 0:
            raise ValueError("operator layers/rank must be positive")
        if not 0 < minimum_sigma <= initial_sigma <= maximum_sigma:
            raise ValueError("operator sigma bounds/initializer are invalid")
        self.style_dim = int(style_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.blocks = int(blocks)
        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.minimum_sigma = float(minimum_sigma)
        self.maximum_sigma = float(maximum_sigma)
        self.style_norm = nn.LayerNorm(style_dim)
        self.style_input = nn.Linear(style_dim, hidden_dim, bias=False)
        # [block, K/V, down/up, rank, hidden].  The identities are explicit;
        # no mean pooling or shared rank token is used.
        self.operator_queries = nn.Parameter(
            torch.empty(blocks, 2, 2, operator_rank, hidden_dim)
        )
        self.reader = nn.ModuleList(
            _OperatorCrossBlock(hidden_dim, heads, ff_dim)
            for _ in range(operator_layers)
        )
        # Dense output directions are block-local.  Sharing only the reader
        # trunk avoids constraining unseen styles to a stored global LoRA basis.
        self.down_output = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, context_dim, bias=False) for _ in range(2)
            )
            for _ in range(blocks)
        )
        self.up_output = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, output_dim, bias=False) for _ in range(2)
            )
            for _ in range(blocks)
        )
        self.log_sigma = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, 1, bias=True) for _ in range(2)
            )
            for _ in range(blocks)
        )
        self.reset_parameters(float(initial_sigma))

    def reset_parameters(self, initial_sigma: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.operator_queries, std=self.hidden_dim**-0.5)
        for block_heads in self.log_sigma:
            for head in block_heads:
                nn.init.normal_(head.weight, std=0.01)
                nn.init.constant_(head.bias, float(torch.tensor(initial_sigma).log()))

    def _operator(
        self, style_memory: torch.Tensor, block: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(style_memory.shape[0])
        memory = self.style_input(self.style_norm(style_memory))
        queries = self.operator_queries[block].reshape(
            4 * self.operator_rank, self.hidden_dim
        )[None].expand(batch, -1, -1)
        for layer in self.reader:
            queries = layer(queries, memory)
        queries = queries.reshape(
            batch, 2, 2, self.operator_rank, self.hidden_dim
        )
        down_values = []
        up_values = []
        sigma_values = []
        for kind in range(2):
            down_hidden = queries[:, kind, 0]
            up_hidden = queries[:, kind, 1]
            down = self.down_output[block][kind](down_hidden)
            up = self.up_output[block][kind](up_hidden)
            # Unit directions plus an explicit singular strength remove the
            # arbitrary up/down scale gauge that otherwise destabilizes a
            # dynamically generated factorization.
            down_values.append(
                F.normalize(down.float(), dim=-1).to(down.dtype)
            )
            up_values.append(F.normalize(up.float(), dim=-1).to(up.dtype))
            log_sigma = self.log_sigma[block][kind](
                0.5 * (down_hidden + up_hidden)
            ).squeeze(-1)
            sigma_values.append(
                log_sigma.float().exp().clamp(
                    self.minimum_sigma, self.maximum_sigma
                ).to(down.dtype)
            )
        return (
            torch.stack(down_values, dim=1),
            torch.stack(up_values, dim=1),
            torch.stack(sigma_values, dim=1),
        )

    def forward(
        self,
        style_memory: torch.Tensor,
        text_context: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if style_memory.ndim != 3 or style_memory.shape[-1] != self.style_dim:
            raise ValueError("style_memory must be [batch,slots,style_dim]")
        if text_context.ndim != 3 or text_context.shape[-1] != self.context_dim:
            raise ValueError("text_context must be [batch,tokens,context_dim]")
        if style_memory.shape[0] != text_context.shape[0]:
            raise ValueError("style and text batches disagree")
        block = int(block_index)
        if not 0 <= block < self.blocks:
            raise ValueError("block_index is outside the model")
        down, up, sigma = self._operator(style_memory, block)
        hidden = torch.einsum("bnc,bkrc->bknr", text_context, down)
        hidden = hidden * sigma[:, :, None]
        return torch.einsum("bknr,bkro->bkno", hidden, up)


class _NativeAttentionProbe(nn.Module):
    """Frozen native K/V normalization and O for functional probe queries."""

    def __init__(self, cross: nn.Module) -> None:
        super().__init__()
        self.n_heads = int(cross.n_heads)
        self.head_dim = int(cross.head_dim)
        self.k_proj = copy.deepcopy(cross.k_proj)
        self.v_proj = copy.deepcopy(cross.v_proj)
        self.q_norm = copy.deepcopy(cross.q_norm)
        self.k_norm = copy.deepcopy(cross.k_norm)
        self.v_norm = copy.deepcopy(cross.v_norm)
        self.output_proj = copy.deepcopy(cross.output_proj)
        self.requires_grad_(False)

    def project_context(
        self, context: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens = context.shape[:2]
        key = self.k_proj(context) + delta[:, 0]
        value = self.v_proj(context) + delta[:, 1]
        key = key.reshape(batch, tokens, self.n_heads, self.head_dim)
        value = value.reshape(batch, tokens, self.n_heads, self.head_dim)
        return self.k_norm(key), self.v_norm(value)

    def attend(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        queries_normalized: bool = False,
    ) -> torch.Tensor:
        if not queries_normalized:
            queries = self.q_norm(queries)
        attended = F.scaled_dot_product_attention(
            queries.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        attended = attended.transpose(1, 2).reshape(
            attended.shape[0], attended.shape[2], -1
        )
        return self.output_proj(attended)


def _normalized_activation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    direction_weight: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_f = teacher.float()
    student_f = student.float()
    dimensions = tuple(range(2, teacher_f.ndim))
    teacher_rms = teacher_f.square().mean(dim=dimensions).sqrt().clamp_min(1e-5)
    student_rms = student_f.square().mean(dim=dimensions).sqrt()
    scale = teacher_rms[..., None, None]
    huber = F.smooth_l1_loss(student_f / scale, teacher_f / scale, beta=0.1)
    cosine = F.cosine_similarity(
        student_f.flatten(2), teacher_f.flatten(2), dim=-1
    )
    direction = (1.0 - cosine).mean()
    magnitude = (
        student_rms.clamp_min(1e-8).log() - teacher_rms.log()
    ).abs().mean()
    loss = huber + float(direction_weight) * direction + float(
        magnitude_weight
    ) * magnitude
    return loss, {
        "loss": loss.detach(),
        "normalized_huber": huber.detach(),
        "cosine": cosine.mean().detach(),
        "direction_loss": direction.detach(),
        "magnitude_log_error": magnitude.detach(),
        "student_to_teacher_rms": (student_rms / teacher_rms).mean().detach(),
    }


def _mixture_target(
    context: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    components: torch.Tensor,
    weights: torch.Tensor,
    block: int,
) -> torch.Tensor:
    result = context.new_zeros(
        context.shape[0], 2, context.shape[1], teacher_up.shape[-2]
    )
    for slot in range(components.shape[1]):
        indices = components[:, slot].clamp_min(0)
        active = (components[:, slot] >= 0).to(context.dtype)
        value = apply_kv_factors(
            context,
            teacher_down[indices, block].to(context.dtype),
            teacher_up[indices, block].to(context.dtype),
        )
        result = result + value * (
            weights[:, slot].to(context.dtype) * active
        )[:, None, None, None]
    return result


def _mean_teacher_operator(
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
) -> torch.Tensor:
    """Compose and average LoRA functions, never their gauge-dependent factors."""

    artists, blocks, kinds, rank, context_dim = teacher_down.shape
    if teacher_up.shape[:3] != (artists, blocks, kinds):
        raise ValueError("Teacher factor-bank leading dimensions disagree")
    if teacher_up.shape[-1] != rank:
        raise ValueError("Teacher factor-bank rank dimensions disagree")
    output_dim = int(teacher_up.shape[-2])
    values: list[torch.Tensor] = []
    for block in range(blocks):
        block_values: list[torch.Tensor] = []
        for kind in range(kinds):
            up = teacher_up[:, block, kind].float().permute(1, 0, 2).reshape(
                output_dim, artists * rank
            )
            down = teacher_down[:, block, kind].float().reshape(
                artists * rank, context_dim
            )
            block_values.append((up @ down).div_(artists).to(torch.bfloat16))
        values.append(torch.stack(block_values))
    return torch.stack(values)


def _apply_dense_kv_operator(
    context: torch.Tensor,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Apply one block's [K/V, output, context] operator."""

    return torch.einsum(
        "bnc,koc->bkno", context, operator.to(context.dtype)
    )


def _centered_residual_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    direction_weight: float,
    magnitude_weight: float,
    relation_weight: float,
    common_weight: float,
    magnitude_floor: float,
    magnitude_ceiling: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit artist-centered effects without rewarding a second common output."""

    student_f = student.float()
    teacher_f = teacher.detach().float()
    reduce_dims = tuple(range(2, teacher_f.ndim))
    teacher_rms = teacher_f.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-5)
    student_rms = student_f.square().mean(dim=reduce_dims).sqrt()
    scale = teacher_rms[..., None, None]
    reconstruction = F.smooth_l1_loss(
        student_f / scale, teacher_f / scale, beta=0.1
    )
    cosine_rows = F.cosine_similarity(
        student_f.flatten(2), teacher_f.flatten(2), dim=-1
    )
    direction = (1.0 - cosine_rows).mean()
    ratio = student_rms / teacher_rms
    magnitude = (
        F.softplus((float(magnitude_floor) - ratio) / 0.1)
        + F.softplus((ratio - float(magnitude_ceiling)) / 0.1)
    ).mean() * 0.1

    # Match every reference to its own centered teacher effect and use every
    # other artist in the controlled batch as a negative.  K and V are kept
    # separate because their useful geometries are not interchangeable.
    relation_terms: list[torch.Tensor] = []
    relation_accuracy: list[torch.Tensor] = []
    if student_f.shape[0] > 1:
        labels = torch.arange(student_f.shape[0], device=student_f.device)
        for kind in range(student_f.shape[1]):
            student_descriptor = F.normalize(
                student_f[:, kind].flatten(1), dim=1
            )
            teacher_descriptor = F.normalize(
                teacher_f[:, kind].flatten(1), dim=1
            )
            logits = student_descriptor @ teacher_descriptor.t()
            logits = logits / float(temperature)
            relation_terms.append(F.cross_entropy(logits, labels))
            relation_accuracy.append((logits.argmax(dim=1) == labels).float().mean())
    relation = (
        torch.stack(relation_terms).mean()
        if relation_terms
        else reconstruction.new_zeros(())
    )
    accuracy = (
        torch.stack(relation_accuracy).mean()
        if relation_accuracy
        else reconstruction.new_ones(())
    )

    # The exact dataset-level centered target has zero mean.  Matching the
    # stochastic teacher-batch mean gives an unbiased anti-common gradient
    # without forcing every small batch to an artificial zero tensor.
    batch_scale = teacher_rms.mean().clamp_min(1e-5)
    common = F.smooth_l1_loss(
        student_f.mean(dim=0) / batch_scale,
        teacher_f.mean(dim=0) / batch_scale,
        beta=0.1,
    )
    loss = (
        reconstruction
        + float(direction_weight) * direction
        + float(magnitude_weight) * magnitude
        + float(relation_weight) * relation
        + float(common_weight) * common
    )
    return loss, {
        "loss": loss.detach(),
        "centered_huber": reconstruction.detach(),
        "cosine": cosine_rows.mean().detach(),
        "direction_loss": direction.detach(),
        "magnitude_band_loss": magnitude.detach(),
        "student_to_teacher_rms": ratio.mean().detach(),
        "relation_loss": relation.detach(),
        "relation_accuracy": accuracy.detach(),
        "residual_common_loss": common.detach(),
        "residual_common_ratio": (
            student_f.mean(dim=0).square().mean().sqrt()
            / student_f.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
    }


def _functional_centered_attention_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    centered_huber_weight: float,
    direction_weight: float,
    magnitude_weight: float,
    relation_weight: float,
    raw_huber_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only the K/V effect visible through native attention.

    Every row in the controlled batch uses the same context and queries.  The
    batch mean therefore represents the shared functional effect, while the
    centered rows expose reference-specific differences.  This intentionally
    avoids regressing pre-normalization K/V components that native
    normalization, softmax, or O projection subsequently discard.
    """

    student_f = student.float()
    teacher_f = teacher.detach().float()
    if student_f.shape != teacher_f.shape or student_f.shape[0] < 2:
        raise ValueError(
            "Functional centered attention loss needs matching controlled rows"
        )
    student_common = student_f.mean(dim=0, keepdim=True)
    teacher_common = teacher_f.mean(dim=0, keepdim=True)
    student_centered = student_f - student_common
    teacher_centered = teacher_f - teacher_common
    reduce_dims = tuple(range(1, teacher_f.ndim))
    row_shape = (-1,) + (1,) * (teacher_f.ndim - 1)
    teacher_rms = (
        teacher_centered.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-5)
    )
    student_rms = (
        student_centered.square().mean(dim=reduce_dims) + 1e-12
    ).sqrt()
    scale = teacher_rms.reshape(row_shape)
    centered_huber = F.smooth_l1_loss(
        student_centered / scale,
        teacher_centered / scale,
        beta=0.1,
    )
    cosine_rows = F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=1
    )
    direction = (1.0 - cosine_rows).mean()
    magnitude = F.smooth_l1_loss(
        (student_rms / teacher_rms).clamp_min(1e-4).log().clamp(-4, 4),
        torch.zeros_like(student_rms),
        beta=0.1,
    )
    if temperature <= 0:
        raise ValueError("Functional relation temperature must be positive")
    student_unit = F.normalize(student_centered.flatten(1), dim=1)
    teacher_unit = F.normalize(teacher_centered.flatten(1), dim=1)
    logits = student_unit @ teacher_unit.t() / float(temperature)
    labels = torch.arange(len(student_f), device=student_f.device)
    relation = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )
    teacher_total_rms = teacher_f.square().mean().sqrt().clamp_min(1e-5)
    raw_huber = F.smooth_l1_loss(
        student_f / teacher_total_rms,
        teacher_f / teacher_total_rms,
        beta=0.1,
    )
    loss = (
        float(centered_huber_weight) * centered_huber
        + float(direction_weight) * direction
        + float(magnitude_weight) * magnitude
        + float(relation_weight) * relation
        + float(raw_huber_weight) * raw_huber
    )
    positive = (logits.diagonal() * float(temperature)).mean()
    wrong = logits.masked_fill(
        torch.eye(len(student_f), device=student_f.device, dtype=torch.bool),
        torch.finfo(logits.dtype).min,
    ).max(dim=1).values.mul(float(temperature)).mean()
    return loss, {
        "functional_loss": loss.detach(),
        "functional_centered_huber": centered_huber.detach(),
        "functional_centered_cosine": cosine_rows.mean().detach(),
        "functional_centered_magnitude": magnitude.detach(),
        "functional_student_to_teacher_rms": (
            student_rms / teacher_rms
        ).mean().detach(),
        "functional_relation_loss": relation.detach(),
        "functional_relation_accuracy": (
            logits.argmax(dim=1) == labels
        ).float().mean().detach(),
        "functional_relation_cosine_gap": (positive - wrong).detach(),
        "functional_raw_huber": raw_huber.detach(),
        "functional_student_common_ratio": (
            student_common.square().mean().sqrt()
            / student_f.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
        "functional_teacher_common_ratio": (
            teacher_common.square().mean().sqrt()
            / teacher_f.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
    }


def prepare_kv_activation_mixture_specs(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Create diverse K/V-only mixture specs and filter unstable functions."""

    cfg = dict(config["kv_activation_mixture_teacher"])
    cache_cfg = dict(cfg["teacher_cache"])
    output = destination / str(cache_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    artist_ids, down, up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    specs = build_mixture_specs(
        len(artist_ids),
        pair_count=int(cache_cfg.get("pair_mixtures", 64)),
        triple_count=int(cache_cfg.get("triple_mixtures", 64)),
        amplified_count=int(cache_cfg.get("amplified_mixtures", 64)),
        signed_count=int(cache_cfg.get("signed_mixtures", 64)),
        amplified_sum_range=tuple(cache_cfg.get("amplified_sum_range", [1.0, 1.5])),
        signed_beta_range=tuple(cache_cfg.get("signed_beta_range", [0.05, 0.5])),
        seed=int(cache_cfg.get("seed", 20260824)),
    )
    device = str(cache_cfg.get("device", "cuda"))
    contexts = load_file(
        destination / str(cache_cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    context_count = min(int(cache_cfg.get("probe_contexts", 8)), len(contexts))
    token_count = min(int(cache_cfg.get("probe_tokens", 64)), contexts.shape[1])
    channel_count = min(int(cache_cfg.get("probe_channels", 256)), up.shape[-2])
    context_ids = torch.linspace(0, len(contexts) - 1, context_count).round().long()
    token_ids = torch.linspace(0, contexts.shape[1] - 1, token_count).round().long()
    channel_ids = torch.linspace(0, up.shape[-2] - 1, channel_count).round().long()
    contexts = contexts[context_ids][:, token_ids].to(device=device, dtype=torch.bfloat16)
    down = down.to(device=device, dtype=torch.bfloat16)
    up = up[:, :, :, channel_ids].to(device=device, dtype=torch.bfloat16)
    blocks = tuple(
        int(value)
        for value in cache_cfg.get("probe_blocks", [0, 4, 8, 12, 16, 20, 24, 27])
    )
    rms_values: dict[int, float] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for spec in specs:
            energies = []
            for block in blocks:
                value = contexts.new_zeros(
                    len(contexts), 2, token_count, channel_count
                )
                for component, weight in zip(
                    spec.components, spec.weights, strict=True
                ):
                    value.add_(
                        apply_kv_factors(
                            contexts,
                            down[component, block][None].expand(len(contexts), -1, -1, -1),
                            up[component, block][None].expand(len(contexts), -1, -1, -1),
                        ),
                        alpha=float(weight),
                    )
                energies.append(value.float().square().mean())
            rms_values[spec.index] = float(torch.stack(energies).mean().sqrt())
    single_median = float(torch.tensor([
        rms_values[index] for index in range(len(artist_ids))
    ]).median())
    minimum, maximum = (
        float(value)
        for value in cache_cfg.get("stable_activation_ratio_range", [0.2, 1.5])
    )
    rows = []
    for spec in specs:
        ratio = rms_values[spec.index] / max(single_median, 1e-8)
        enabled = spec.kind == "single" or minimum <= ratio <= maximum
        rows.append({
            "index": spec.index,
            "kind": spec.kind,
            "components": list(spec.components),
            "weights": list(spec.weights),
            "style_ids": [artist_ids[index] for index in spec.components],
            "mixture_style_id": f"kv-mixture-{spec.index:05d}",
            "activation_rms": rms_values[spec.index],
            "activation_to_single_median_ratio": ratio,
            "enabled": enabled,
        })
    write_records(output / "mixtures.parquet", rows)
    summary = {
        "artists": len(artist_ids),
        "mixtures": len(rows),
        "enabled_mixtures": sum(bool(row["enabled"]) for row in rows),
        "pair": sum(row["kind"] == "pair" for row in rows),
        "triple": sum(row["kind"] == "triple" for row in rows),
        "amplified": sum(row["kind"] == "amplified" for row in rows),
        "signed": sum(row["kind"] == "signed" for row in rows),
        "single_activation_rms_median": single_median,
        "stable_activation_ratio_range": [minimum, maximum],
        "content_contexts": len(contexts),
    }
    write_json(output / "summary.json", summary)
    return summary


def _load_reader(
    config: dict[str, Any], destination: Path, cfg: dict[str, Any], device: str
) -> DetailPreservingTypedSlotReader:
    state = torch.load(
        destination / str(cfg["reader_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    detail = _oracle_detail_config(config, dict(config["kv_lora_oracle_bootstrap"]))
    reader = DetailPreservingTypedSlotReader(**dict(detail["model"]))
    reader.load_state_dict(state["reader"], strict=True)
    return reader.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()


def _materialize_style_codes(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    reader: DetailPreservingTypedSlotReader,
    artist_ids: list[str],
    mixture_rows: list[dict[str, Any]],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    training = dict(cfg["training"])
    chunk = int(training.get("materialization_style_chunk", 16))
    single_references = int(training.get("single_reference_images", 8))
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["synthetic_reference_cache"]),
        split="train", style_ids=artist_ids, batch_size=chunk,
        references=single_references, seed=int(cfg["seed"]) ^ 0x53594E54,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    human_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train", style_ids=artist_ids, batch_size=chunk,
        references=single_references, seed=int(cfg["seed"]) ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    single_codes: dict[str, torch.Tensor] = {}
    counts: torch.Tensor | None = None
    for domain, loader in (("synthetic", synthetic_loader), ("human", human_loader)):
        codes, domain_counts = _materialize_reader_code_bank(
            reader, loader, artist_ids, reference_images=single_references,
            seed=int(cfg["seed"]) ^ (0x11111111 if domain == "synthetic" else 0x22222222),
            device=device, style_chunk_size=chunk,
        )
        single_codes[domain] = codes
        counts = domain_counts if counts is None else counts
        if not torch.equal(counts, domain_counts):
            raise RuntimeError("Single-domain reference-count views disagree")
    mixture_by_kind: dict[str, torch.Tensor] = {}
    mixture_references = int(training.get("mixture_reference_images", 4))
    for kind in ("pair", "triple", "amplified", "signed"):
        rows = [row for row in mixture_rows if str(row["kind"]) == kind]
        style_ids = [str(row["mixture_style_id"]) for row in rows]
        loader = CachedTeacherReferenceLoader(
            destination / str(cfg["mixture_reference_cache"]),
            split="train", style_ids=style_ids, batch_size=chunk,
            references=mixture_references,
            seed=int(cfg["seed"]) ^ (sum(map(ord, kind)) * 1_000_003),
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            strict_style_ids=True,
        )
        codes, _ = _materialize_reader_code_bank(
            reader, loader, style_ids, reference_images=mixture_references,
            seed=int(cfg["seed"]) ^ (sum(map(ord, kind)) * 1_000_003),
            device=device, style_chunk_size=chunk,
        )
        mixture_by_kind[kind] = codes
    assert counts is not None
    return single_codes, mixture_by_kind, counts


def _save_training_state(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    reader: nn.Module | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "step": int(step),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    if reader is not None:
        state["reader"] = {
            key: value.detach().cpu() for key, value in reader.state_dict().items()
        }
    torch.save(state, temporary)
    temporary.replace(path)


def _load_native_attention_probes(
    config: dict[str, Any], destination: Path, device: str
) -> nn.ModuleList:
    """Copy only native K/V normalization and O, then release the full DiT."""

    from .style_transfer import _resolve_anima_model

    anima = _resolve_anima_model(config, destination, device)
    anima.requires_grad_(False).eval()
    probes = nn.ModuleList(
        _NativeAttentionProbe(block.cross_attn) for block in anima.blocks
    ).to(device=device, dtype=torch.bfloat16)
    probes.requires_grad_(False).eval()
    del anima
    gc.collect()
    torch.cuda.empty_cache()
    return probes


def train_reference_conditioned_kv_activation_generator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_key: str = "kv_reference_activation_generator",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[config_key])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 3000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)), dtype=torch.float16,
    )
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)
    centered_teacher = str(cfg.get("teacher_decomposition", "full")) == "centered"
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    common_operator: torch.Tensor | None = None
    if centered_teacher:
        common_path = output / "frozen_common_operator.pt"
        if common_path.exists():
            common_operator = torch.load(
                common_path, map_location="cpu", weights_only=True
            )["common_operator"].to(device=device, dtype=torch.bfloat16)
        else:
            common_operator = _mean_teacher_operator(teacher_down, teacher_up)
            torch.save(
                {"common_operator": common_operator.cpu()}, common_path
            )
    mixture_rows = [
        row for row in read_records(destination / str(cfg["mixture_manifest"]))
        if str(row["kind"]) != "single" and bool(row.get("enabled", True))
    ]
    rows_by_kind = {
        kind: [row for row in mixture_rows if str(row["kind"]) == kind]
        for kind in ("pair", "triple", "amplified", "signed")
    }
    if any(not rows for rows in rows_by_kind.values()):
        raise RuntimeError("Every configured mixture kind needs enabled rows")
    reader = _load_reader(config, destination, cfg, device)
    single_codes, mixture_codes, reference_counts = _materialize_style_codes(
        config, destination, cfg, reader, artist_ids, mixture_rows, device
    )
    del reader
    torch.cuda.empty_cache()

    contexts = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"].to(device=device, dtype=torch.bfloat16)
    heldout_contexts = int(training.get("heldout_contexts", 32))
    train_context_count = len(contexts) - heldout_contexts
    if train_context_count <= 0:
        raise ValueError("heldout_contexts leaves no training captions")
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "direct_cross_attention"))
    model_type: type[nn.Module]
    if architecture == "direct_cross_attention":
        model_type = ReferenceConditionedKVActivationGenerator
    elif architecture == "bilinear_low_rank_operator":
        model_type = ReferenceConditionedLowRankKVOperator
    else:
        raise ValueError(f"Unsupported K/V activation architecture: {architecture}")
    model = model_type(
        style_dim=int(next(iter(single_codes.values())).shape[-1]),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 2e-4)),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    direction_weight = float(training.get("direction_weight", 0.2))
    magnitude_weight = float(training.get("magnitude_weight", 0.05))
    relation_weight = float(training.get("relation_weight", 0.0))
    common_weight = float(training.get("residual_common_weight", 0.0))
    magnitude_floor = float(training.get("magnitude_floor", 0.7))
    magnitude_ceiling = float(training.get("magnitude_ceiling", 1.3))
    relation_temperature = float(training.get("relation_temperature", 0.1))
    relation_start = int(training.get("relation_start_step", 250))
    relation_ramp = int(training.get("relation_ramp_steps", 250))
    single_only_steps = int(training.get("single_only_steps", 0))
    mixture_ramp_steps = int(training.get("mixture_ramp_steps", 1))
    base_lr = float(training.get("learning_rate", 2e-4))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every", 10))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    validation_every = int(training.get("validation_every", 250))
    validation_tokens = int(training.get("validation_tokens", 64))
    attention_start = int(training.get("attention_start_step", 1000))
    attention_ramp = int(training.get("attention_ramp_steps", 500))
    attention_weight = float(training.get("attention_weight", 0.3))
    attention_queries = int(training.get("attention_queries", 32))
    categories = ("single", "pair", "triple", "amplified", "signed")
    category_weights = tuple(
        float(value)
        for value in training.get("category_weights", [0.25, 0.1875, 0.0625, 0.25, 0.25])
    )
    if len(category_weights) != len(categories):
        raise ValueError("category_weights must cover single/pair/triple/amplified/signed")

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-activation-generator")),
            id=str(wandb_cfg.get("id", "kv-reference-activation-generator")),
            resume="allow" if start_step else "never",
            config={config_key: cfg},
        )
    running: dict[str, list[float]] = defaultdict(list)
    native_probes: nn.ModuleList | None = None
    started = time.perf_counter()
    model.train()
    try:
        for step in range(start_step + 1, steps + 1):
            rng = random.Random(seed + step * 1_000_003)
            mixture_progress = max(
                0.0,
                min(
                    1.0,
                    (step - single_only_steps) / max(1, mixture_ramp_steps),
                ),
            )
            active_category_weights = (
                (
                    1.0 - mixture_progress
                    + mixture_progress * category_weights[0],
                    *(mixture_progress * value for value in category_weights[1:]),
                )
                if centered_teacher
                else category_weights
            )
            category = rng.choices(
                categories, weights=active_category_weights, k=1
            )[0]
            if centered_teacher:
                context_indices = torch.full(
                    (batch,), rng.randrange(train_context_count),
                    device=device, dtype=torch.long,
                )
            else:
                context_indices = torch.tensor(
                    [rng.randrange(train_context_count) for _ in range(batch)],
                    device=device,
                )
            context = contexts[context_indices]
            if category == "single":
                selected_indices = (
                    rng.sample(range(len(artist_ids)), batch)
                    if centered_teacher and batch <= len(artist_ids)
                    else [rng.randrange(len(artist_ids)) for _ in range(batch)]
                )
                target_indices = torch.tensor(selected_indices, device=device)
                domain = "human" if rng.random() < 0.5 else "synthetic"
                codes = single_codes[domain]
                view_indices = torch.tensor(
                    [rng.randrange(codes.shape[1]) for _ in range(batch)], device=device
                )
                style = codes[target_indices, view_indices]
                components = target_indices[:, None]
                weights = torch.ones(batch, 1, device=device)
            else:
                source_rows = rows_by_kind[category]
                selected_rows = (
                    rng.sample(range(len(source_rows)), batch)
                    if centered_teacher and batch <= len(source_rows)
                    else [rng.randrange(len(source_rows)) for _ in range(batch)]
                )
                target_indices = torch.tensor(selected_rows, device=device)
                codes = mixture_codes[category]
                view_indices = torch.tensor(
                    [rng.randrange(codes.shape[1]) for _ in range(batch)], device=device
                )
                style = codes[target_indices, view_indices]
                selected = [source_rows[index] for index in target_indices.cpu().tolist()]
                max_components = max(len(row["components"]) for row in selected)
                components = torch.full((batch, max_components), -1, device=device, dtype=torch.long)
                weights = torch.zeros(batch, max_components, device=device)
                for index, row in enumerate(selected):
                    count = len(row["components"])
                    components[index, :count] = torch.tensor(row["components"], device=device)
                    weights[index, :count] = torch.tensor(row["weights"], device=device)
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            selected_blocks = [
                (first_block + offset) % model.blocks
                for offset in range(blocks_per_step)
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            metrics_by_block = []
            active_attention_weight = attention_weight * max(
                0.0,
                min(1.0, (step - attention_start) / max(1, attention_ramp)),
            )
            active_relation_weight = relation_weight * max(
                0.0,
                min(1.0, (step - relation_start) / max(1, relation_ramp)),
            )
            if active_attention_weight > 0 and native_probes is None:
                native_probes = _load_native_attention_probes(
                    config, destination, device
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for block in selected_blocks:
                    student = model(style, context, block)
                    teacher = _mixture_target(
                        context, teacher_down, teacher_up,
                        components, weights, block,
                    )
                    common_delta = None
                    if centered_teacher:
                        assert common_operator is not None
                        common_delta = _apply_dense_kv_operator(
                            context, common_operator[block]
                        ) * weights.sum(dim=1)[:, None, None, None]
                        teacher = teacher - common_delta
                        block_loss, block_metrics = _centered_residual_loss(
                            student,
                            teacher,
                            direction_weight=direction_weight,
                            magnitude_weight=magnitude_weight,
                            relation_weight=active_relation_weight,
                            common_weight=common_weight,
                            magnitude_floor=magnitude_floor,
                            magnitude_ceiling=magnitude_ceiling,
                            temperature=relation_temperature,
                        )
                    else:
                        block_loss, block_metrics = _normalized_activation_loss(
                            student, teacher,
                            direction_weight=direction_weight,
                            magnitude_weight=magnitude_weight,
                        )
                    if active_attention_weight > 0:
                        assert native_probes is not None
                        probe = native_probes[block]
                        query_generator = torch.Generator(device=device).manual_seed(
                            seed + step * 1_000_003 + block * 10_007
                        )
                        queries = torch.randn(
                            batch,
                            attention_queries,
                            probe.n_heads,
                            probe.head_dim,
                            device=device,
                            dtype=torch.bfloat16,
                            generator=query_generator,
                        )
                        zero = torch.zeros_like(student)
                        attention_base = (
                            common_delta if common_delta is not None else zero
                        )
                        base_key, base_value = probe.project_context(
                            context, attention_base
                        )
                        student_key, student_value = probe.project_context(
                            context, attention_base + student
                        )
                        teacher_key, teacher_value = probe.project_context(
                            context, attention_base + teacher
                        )
                        base_output = probe.attend(queries, base_key, base_value)
                        student_effect = (
                            probe.attend(queries, student_key, student_value)
                            - base_output
                        )
                        teacher_effect = (
                            probe.attend(queries, teacher_key, teacher_value)
                            - base_output
                        )
                        attention_loss, attention_metrics = (
                            _normalized_activation_loss(
                                student_effect[:, None],
                                teacher_effect[:, None],
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                        )
                        block_loss = block_loss + (
                            active_attention_weight * attention_loss
                        )
                        block_metrics.update({
                            f"attention_{key}": value
                            for key, value in attention_metrics.items()
                        })
                    losses.append(block_loss)
                    metrics_by_block.append(block_metrics)
                loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            lr = base_lr * min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            running["loss"].append(float(loss.detach()))
            running["grad_norm"].append(float(grad_norm))
            running["learning_rate"].append(lr)
            running["attention_weight"].append(active_attention_weight)
            running["relation_weight"].append(active_relation_weight)
            running["mixture_progress"].append(mixture_progress)
            running[f"category/{category}"].append(1.0)
            for key in metrics_by_block[0]:
                running[key].append(sum(float(row[key]) for row in metrics_by_block) / len(metrics_by_block))
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["context_unique_fraction"] = len(set(context_indices.tolist())) / batch
                print(f"K/V activation generator step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                model.eval()
                validation_rows: dict[str, list[float]] = defaultdict(list)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    val_artists = torch.linspace(0, len(artist_ids) - 1, min(8, len(artist_ids)), device=device).round().long().unique()
                    val_context_ids = torch.linspace(train_context_count, len(contexts) - 1, min(4, heldout_contexts), device=device).round().long().unique()
                    val_style = single_codes["synthetic"][val_artists, 0]
                    for block in range(model.blocks):
                        for context_index in val_context_ids.tolist():
                            token_ids = torch.linspace(
                                0,
                                contexts.shape[1] - 1,
                                min(validation_tokens, contexts.shape[1]),
                                device=device,
                            ).round().long().unique()
                            val_context = contexts[
                                context_index, token_ids
                            ].expand(len(val_artists), -1, -1)
                            student = model(val_style, val_context, block)
                            teacher = _mixture_target(
                                val_context, teacher_down, teacher_up,
                                val_artists[:, None], torch.ones(len(val_artists), 1, device=device), block,
                            )
                            if centered_teacher:
                                assert common_operator is not None
                                teacher = teacher - _apply_dense_kv_operator(
                                    val_context, common_operator[block]
                                )
                                _, values = _centered_residual_loss(
                                    student,
                                    teacher,
                                    direction_weight=direction_weight,
                                    magnitude_weight=magnitude_weight,
                                    relation_weight=relation_weight,
                                    common_weight=common_weight,
                                    magnitude_floor=magnitude_floor,
                                    magnitude_ceiling=magnitude_ceiling,
                                    temperature=relation_temperature,
                                )
                            else:
                                _, values = _normalized_activation_loss(
                                    student, teacher,
                                    direction_weight=direction_weight,
                                    magnitude_weight=magnitude_weight,
                                )
                            for key, value in values.items():
                                validation_rows[key].append(float(value))
                validation = {key: sum(values) / len(values) for key, values in validation_rows.items()}
                print(f"K/V activation generator validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"val/{key}": value for key, value in validation.items()}, step=step)
                model.train()
            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path, step=step, model=model, optimizer=optimizer, cfg=cfg
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "mixtures": len(mixture_rows),
        "contexts": len(contexts),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": architecture,
        "teacher_decomposition": "centered" if centered_teacher else "full",
        "elapsed_seconds": time.perf_counter() - started,
        "end_to_end_flow_training": False,
        "native_artist_teacher": False,
    }
    write_json(output / "summary.json", summary)
    return summary


_FUNCTIONAL_READER_PREFIXES = (
    "set_query",
    "set_norm",
    "reference_identity_norm",
    "reference_identity_projection",
    "pool_type_embeddings",
    "pool_type_preference",
    "set_attention",
    "set_ff_norm",
    "set_ff",
    "mixers",
)


def _open_functional_reader_pooling(
    reader: DetailPreservingTypedSlotReader,
) -> list[nn.Parameter]:
    selected: list[nn.Parameter] = []
    for name, parameter in reader.named_parameters():
        trainable = any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _FUNCTIONAL_READER_PREFIXES
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("No Reader pooling parameters were selected")
    reader.train()
    return selected


@torch.no_grad()
def _materialize_reference_token_bank(
    loader: CachedTeacherReferenceLoader,
    style_ids: list[str],
    *,
    references: int,
    seed: int,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """Keep cached frozen-Resampler tokens resident on the training GPU."""

    chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    for offset in range(0, len(style_ids), chunk_size):
        ids = style_ids[offset : offset + chunk_size]
        loaded = loader.load_styles(
            ids,
            references_per_style=references,
            seed=seed + offset * 1_000_003,
        )
        chunks.append(
            loaded["tokens"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
        )
        if offset + len(ids) == len(style_ids) or (offset // chunk_size + 1) % 5 == 0:
            print(
                "materialized raw reference tokens "
                f"{offset + len(ids)}/{len(style_ids)} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    if not chunks:
        raise ValueError("Cannot materialize an empty reference bank")
    return torch.cat(chunks, dim=0)


def _select_reference_tokens(
    bank: torch.Tensor,
    artist_indices: list[int],
    *,
    reference_counts: list[int],
    reference_start: int,
    reference_stop: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(reference_counts)
    rows = []
    mask = torch.zeros(
        len(artist_indices), maximum, device=bank.device, dtype=torch.bool
    )
    for row, (artist, count) in enumerate(
        zip(artist_indices, reference_counts, strict=True)
    ):
        choices = rng.sample(range(reference_start, reference_stop), count)
        values = bank[artist, choices]
        if count < maximum:
            values = torch.cat(
                [values, values.new_zeros(maximum - count, *values.shape[1:])]
            )
        rows.append(values)
        mask[row, :count] = True
    return torch.stack(rows), mask


def _direct_delta_artist_split(
    artist_ids: list[str],
    mixture_rows: list[dict[str, Any]],
    *,
    training_artists: int,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Keep every mixture component in train and hold out whole artists."""

    index_by_style = {style_id: index for index, style_id in enumerate(artist_ids)}
    remapped = []
    required: set[int] = set()
    for row in mixture_rows:
        style_ids = [str(value) for value in row.get("style_ids", [])]
        if not style_ids:
            raise RuntimeError("Mixture rows must retain component style_ids")
        missing = [style_id for style_id in style_ids if style_id not in index_by_style]
        if missing:
            raise RuntimeError(f"Mixture styles are absent from the 320 bank: {missing[:4]}")
        components = [index_by_style[style_id] for style_id in style_ids]
        if len(components) != len(row["weights"]):
            raise RuntimeError("Mixture component styles and weights disagree")
        required.update(components)
        remapped.append({**row, "teacher_components": components})
    if len(required) > int(training_artists):
        raise ValueError("training_artists cannot contain every mixture component")
    remaining = [index for index in range(len(artist_ids)) if index not in required]
    train = sorted(required) + remaining[: int(training_artists) - len(required)]
    train_set = set(train)
    validation = [index for index in range(len(artist_ids)) if index not in train_set]
    return train, validation, remapped


def _open_direct_delta_reader(
    reader: DetailPreservingTypedSlotReader,
) -> list[nn.Parameter]:
    """Fine-tune every Reader path used to produce style memory."""

    parameters = []
    for name, parameter in reader.named_parameters():
        trainable = not name.startswith("reconstruction_")
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("No direct-delta Reader parameters were selected")
    reader.train()
    return parameters


def train_direct_reference_kv_delta_320(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Train styled-reference-only, text-conditioned full native K/V deltas."""

    from .kv_real_query_distillation import _RealQueryBank

    cfg = copy.deepcopy(config["kv_reference_direct_delta_320"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 4000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260826))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    raw_mixture_rows = [
        row
        for row in read_records(destination / str(cfg["mixture_manifest"]))
        if str(row["kind"]) in {"pair", "triple", "amplified", "signed"}
        and bool(row.get("enabled", True))
    ]
    train_artists, validation_artists, mixture_rows = _direct_delta_artist_split(
        artist_ids,
        raw_mixture_rows,
        training_artists=int(training.get("training_artists", 256)),
    )
    if not validation_artists:
        raise RuntimeError("Direct-delta training requires held-out artists")
    rows_by_kind = {
        kind: [row for row in mixture_rows if str(row["kind"]) == kind]
        for kind in ("pair", "triple", "amplified", "signed")
    }
    if any(not rows for rows in rows_by_kind.values()):
        raise RuntimeError("Every direct-delta mixture category needs rows")
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    reader = _load_reader(config, destination, cfg, device)
    reader_parameters = _open_direct_delta_reader(reader)
    chunk = int(training.get("materialization_style_chunk", 16))
    token_lru = int(training.get("token_lru_shards", 8))
    single_images = int(training.get("single_reference_images", 8))
    single_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["synthetic_reference_cache"]),
        split="train", style_ids=artist_ids, batch_size=chunk,
        references=single_images, seed=seed ^ 0x53594E54,
        token_lru_shards=token_lru, strict_style_ids=True,
    )
    single_bank = _materialize_reference_token_bank(
        single_loader, artist_ids, references=single_images,
        seed=seed ^ 0x53594E54, chunk_size=chunk, device=device,
    )
    mixture_images = int(training.get("mixture_reference_images", 4))
    mixture_banks: dict[str, torch.Tensor] = {}
    for kind, rows in rows_by_kind.items():
        style_ids = [str(row["mixture_style_id"]) for row in rows]
        loader = CachedTeacherReferenceLoader(
            destination / str(cfg["mixture_reference_cache"]),
            split="train", style_ids=style_ids, batch_size=chunk,
            references=mixture_images,
            seed=seed ^ (sum(map(ord, kind)) * 1_000_003),
            token_lru_shards=token_lru, strict_style_ids=True,
        )
        mixture_banks[kind] = _materialize_reference_token_bank(
            loader, style_ids, references=mixture_images,
            seed=seed ^ (sum(map(ord, kind)) * 1_000_003),
            chunk_size=chunk, device=device,
        )

    contexts = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"].to(device=device, dtype=torch.bfloat16)
    heldout_contexts = int(training.get("heldout_contexts", 32))
    train_context_count = len(contexts) - heldout_contexts
    if train_context_count <= 0:
        raise ValueError("heldout_contexts leaves no direct-delta training context")
    query_cfg = dict(config["kv_real_query_bank"])
    query_bank = _RealQueryBank(
        destination / str(cfg["query_cache"]),
        destination / str(query_cfg["source_cache"]),
        device=device,
        gpu_resident=bool(training.get("gpu_resident_queries", True)),
    )
    query_heldout = int(training.get("heldout_query_contents", 8))
    train_query_count = len(query_bank.rows) - query_heldout
    if train_query_count <= 0:
        raise ValueError("heldout_query_contents leaves no real-Q training rows")
    probes = _load_native_attention_probes(config, destination, device)

    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "direct_cross_attention"))
    if architecture != "direct_cross_attention":
        raise ValueError("320 direct-delta training requires direct_cross_attention")
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    generator_lr = float(training.get("generator_learning_rate", 2e-4))
    reader_lr = float(training.get("reader_learning_rate", generator_lr * 0.2))
    optimizer = torch.optim.AdamW(
        [
            {"name": "generator", "params": list(model.parameters()), "lr": generator_lr},
            {"name": "reader", "params": reader_parameters, "lr": reader_lr},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    categories = ("single", "pair", "triple", "amplified", "signed")
    category_weights = tuple(
        float(value)
        for value in training.get("category_weights", [0.5, 0.125, 0.125, 0.125, 0.125])
    )
    single_counts = [int(value) for value in training.get("single_reference_counts", [1, 2, 4, 8])]
    single_count_weights = [float(value) for value in training.get("single_reference_count_weights", [0.30, 0.25, 0.25, 0.20])]
    mixture_counts = [int(value) for value in training.get("mixture_reference_counts", [1, 2, 4])]
    mixture_count_weights = [float(value) for value in training.get("mixture_reference_count_weights", [0.40, 0.35, 0.25])]
    direction_weight = float(training.get("direction_weight", 0.3))
    magnitude_weight = float(training.get("magnitude_weight", 0.15))
    consistency_weight = float(training.get("reference_consistency_weight", 0.1))
    attention_weight = float(training.get("attention_weight", 0.2))
    attention_ramp = int(training.get("attention_ramp_steps", 500))
    warmup = int(training.get("warmup_steps", 100))
    generator_clip = float(training.get("max_grad_norm", 10.0))
    reader_clip = float(training.get("reader_max_grad_norm", 5.0))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    validation_tokens = int(training.get("validation_tokens", 64))
    validation_blocks = [
        int(value)
        for value in training.get("validation_blocks", [0, 4, 8, 12, 16, 20, 24, 27])
    ]

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-direct-delta-320")),
            id=str(wandb_cfg.get("id", "kv-reference-direct-delta-320")),
            resume="allow" if start_step else "never",
            config={"kv_reference_direct_delta_320": cfg},
        )

    running: dict[str, list[float]] = defaultdict(list)
    generator_grad_history: list[float] = []
    reader_grad_history: list[float] = []
    started = time.perf_counter()
    model.train()
    reader.train()
    try:
        for step in range(start_step + 1, steps + 1):
            rng = random.Random(seed + step * 1_000_003)
            category = rng.choices(categories, weights=category_weights, k=1)[0]
            alternative_references = None
            alternative_mask = None
            if category == "single":
                artists = rng.sample(train_artists, batch)
                counts = rng.choices(single_counts, weights=single_count_weights, k=batch)
                references, reference_mask = _select_reference_tokens(
                    single_bank, artists, reference_counts=counts,
                    reference_start=0, reference_stop=single_images, rng=rng,
                )
                alternative_references, alternative_mask = _select_reference_tokens(
                    single_bank, artists, reference_counts=counts,
                    reference_start=0, reference_stop=single_images,
                    rng=random.Random(seed ^ (step * 97_409)),
                )
                components = torch.tensor(artists, device=device, dtype=torch.long)[:, None]
                weights = torch.ones(batch, 1, device=device)
            else:
                source_rows = rows_by_kind[category]
                selected_rows = rng.sample(range(len(source_rows)), batch)
                counts = rng.choices(mixture_counts, weights=mixture_count_weights, k=batch)
                references, reference_mask = _select_reference_tokens(
                    mixture_banks[category], selected_rows,
                    reference_counts=counts, reference_start=0,
                    reference_stop=mixture_images, rng=rng,
                )
                selected = [source_rows[index] for index in selected_rows]
                maximum = max(len(row["teacher_components"]) for row in selected)
                components = torch.full(
                    (batch, maximum), -1, device=device, dtype=torch.long
                )
                weights = torch.zeros(batch, maximum, device=device)
                for row_index, row in enumerate(selected):
                    count = len(row["teacher_components"])
                    components[row_index, :count] = torch.tensor(
                        row["teacher_components"], device=device
                    )
                    weights[row_index, :count] = torch.tensor(
                        row["weights"], device=device
                    )
            context_indices = torch.tensor(
                [rng.randrange(train_context_count) for _ in range(batch)],
                device=device, dtype=torch.long,
            )
            context = contexts[context_indices]
            query_content = rng.randrange(train_query_count)
            query_timestep = rng.randrange(len(query_bank.timesteps))
            functional_context = query_bank.context(query_content)[None].expand(
                batch, -1, -1
            )
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            selected_blocks = [
                (first_block + offset) % model.blocks for offset in range(blocks_per_step)
            ]
            active_attention = attention_weight * min(
                1.0, step / max(1, attention_ramp)
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = reader(references, reference_mask).tokens
                alternative_style = (
                    reader(alternative_references, alternative_mask).tokens
                    if alternative_references is not None and alternative_mask is not None
                    else None
                )
                block_losses = []
                block_metrics = []
                first_student = None
                first_teacher = None
                for block in selected_blocks:
                    student = model(style, context, block)
                    teacher = _mixture_target(
                        context, teacher_down, teacher_up,
                        components, weights, block,
                    )
                    raw_loss, metrics = _normalized_activation_loss(
                        student, teacher,
                        direction_weight=direction_weight,
                        magnitude_weight=magnitude_weight,
                    )
                    if first_student is None:
                        first_student, first_teacher = student, teacher
                    if active_attention > 0:
                        functional_student = model(style, functional_context, block)
                        functional_teacher = _mixture_target(
                            functional_context, teacher_down, teacher_up,
                            components, weights, block,
                        )
                        probe = probes[block]
                        queries = query_bank.query(
                            query_content, query_timestep, block
                        )[None].expand(batch, -1, -1, -1)
                        zero = torch.zeros_like(functional_student)
                        base_key, base_value = probe.project_context(
                            functional_context, zero
                        )
                        student_key, student_value = probe.project_context(
                            functional_context, functional_student
                        )
                        teacher_key, teacher_value = probe.project_context(
                            functional_context, functional_teacher
                        )
                        base_output = probe.attend(
                            queries, base_key, base_value, queries_normalized=True
                        )
                        student_effect = probe.attend(
                            queries, student_key, student_value,
                            queries_normalized=True,
                        ) - base_output
                        teacher_effect = probe.attend(
                            queries, teacher_key, teacher_value,
                            queries_normalized=True,
                        ) - base_output
                        attention_loss, attention_metrics = _normalized_activation_loss(
                            student_effect[:, None], teacher_effect[:, None],
                            direction_weight=direction_weight,
                            magnitude_weight=magnitude_weight,
                        )
                        raw_loss = raw_loss + active_attention * attention_loss
                        metrics.update({
                            f"attention_{key}": value
                            for key, value in attention_metrics.items()
                        })
                    block_losses.append(raw_loss)
                    block_metrics.append(metrics)
                loss = torch.stack(block_losses).mean()
                consistency = loss.new_zeros(())
                if alternative_style is not None:
                    assert first_student is not None and first_teacher is not None
                    alternate = model(alternative_style, context, selected_blocks[0])
                    scale = first_teacher.float().square().mean(
                        dim=(2, 3), keepdim=True
                    ).sqrt().clamp_min(1e-5)
                    consistency = F.smooth_l1_loss(
                        alternate.float() / scale,
                        first_student.float() / scale,
                        beta=0.1,
                    )
                    loss = loss + consistency_weight * consistency
            loss.backward()
            generator_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), generator_clip, foreach=True
            )
            reader_grad = torch.nn.utils.clip_grad_norm_(
                reader_parameters, reader_clip, foreach=True
            )
            generator_grad_history.append(float(generator_grad))
            reader_grad_history.append(float(reader_grad))
            lr_scale = min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = generator_lr * lr_scale
            optimizer.param_groups[1]["lr"] = reader_lr * lr_scale
            optimizer.step()

            running["loss"].append(float(loss.detach()))
            running["generator_grad_norm_unclipped"].append(float(generator_grad))
            running["reader_grad_norm_unclipped"].append(float(reader_grad))
            running["reference_count"].append(sum(counts) / len(counts))
            running["reference_consistency"].append(float(consistency.detach()))
            running["attention_weight"].append(active_attention)
            running[f"category/{category}"].append(1.0)
            for key in block_metrics[0]:
                running[key].append(
                    sum(float(values[key]) for values in block_metrics)
                    / len(block_metrics)
                )
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["generator_lr"] = optimizer.param_groups[0]["lr"]
                row["reader_lr"] = optimizer.param_groups[1]["lr"]
                row["context_unique_fraction"] = len(set(context_indices.tolist())) / batch
                print(f"Direct reference K/V delta step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()

            if validation_every > 0 and step % validation_every == 0:
                model.eval()
                reader.eval()
                validation_rows: dict[str, list[float]] = defaultdict(list)
                val_artists = validation_artists[: min(8, len(validation_artists))]
                token_ids = torch.linspace(
                    0, contexts.shape[1] - 1,
                    min(validation_tokens, contexts.shape[1]), device=device,
                ).round().long().unique()
                val_context = contexts[train_context_count, token_ids][None].expand(
                    len(val_artists), -1, -1
                )
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for reference_count in single_counts:
                        val_refs, val_mask = _select_reference_tokens(
                            single_bank, val_artists,
                            reference_counts=[reference_count] * len(val_artists),
                            reference_start=0, reference_stop=single_images,
                            rng=random.Random(seed ^ step ^ reference_count),
                        )
                        val_style = reader(val_refs, val_mask).tokens
                        artist_tensor = torch.tensor(
                            val_artists, device=device, dtype=torch.long
                        )
                        for block in validation_blocks:
                            student = model(val_style, val_context, block)
                            teacher = _mixture_target(
                                val_context, teacher_down, teacher_up,
                                artist_tensor[:, None],
                                torch.ones(len(val_artists), 1, device=device), block,
                            )
                            _, values = _normalized_activation_loss(
                                student, teacher,
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                            correct = F.cosine_similarity(
                                student.float().flatten(2),
                                teacher.float().flatten(2), dim=-1,
                            ).mean()
                            wrong = F.cosine_similarity(
                                student.float().flatten(2),
                                teacher.roll(1, dims=0).float().flatten(2), dim=-1,
                            ).mean()
                            values["correct_minus_wrong_cosine"] = correct - wrong
                            for key, value in values.items():
                                validation_rows[f"single_r{reference_count}/{key}"].append(
                                    float(value)
                                )
                    for kind, source_rows in rows_by_kind.items():
                        selected_rows = list(range(min(4, len(source_rows))))
                        mix_refs, mix_mask = _select_reference_tokens(
                            mixture_banks[kind], selected_rows,
                            reference_counts=[mixture_images] * len(selected_rows),
                            reference_start=0, reference_stop=mixture_images,
                            rng=random.Random(seed ^ step ^ sum(map(ord, kind))),
                        )
                        mix_style = reader(mix_refs, mix_mask).tokens
                        selected = [source_rows[index] for index in selected_rows]
                        maximum = max(len(row["teacher_components"]) for row in selected)
                        mix_components = torch.full(
                            (len(selected), maximum), -1,
                            device=device, dtype=torch.long,
                        )
                        mix_weights = torch.zeros(
                            len(selected), maximum, device=device
                        )
                        for row_index, source in enumerate(selected):
                            count = len(source["teacher_components"])
                            mix_components[row_index, :count] = torch.tensor(
                                source["teacher_components"], device=device
                            )
                            mix_weights[row_index, :count] = torch.tensor(
                                source["weights"], device=device
                            )
                        mix_context = val_context[:1].expand(len(selected), -1, -1)
                        for block in validation_blocks:
                            student = model(mix_style, mix_context, block)
                            teacher = _mixture_target(
                                mix_context, teacher_down, teacher_up,
                                mix_components, mix_weights, block,
                            )
                            _, values = _normalized_activation_loss(
                                student, teacher,
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                            for key, value in values.items():
                                validation_rows[f"mixture_{kind}/{key}"].append(
                                    float(value)
                                )
                validation = {
                    key: sum(values) / len(values)
                    for key, values in validation_rows.items()
                }
                print(f"Direct reference K/V delta validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/{key}": value for key, value in validation.items()},
                        step=step,
                    )
                model.train()
                reader.train()

            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path, step=step, model=model, reader=reader,
                        optimizer=optimizer, cfg=cfg,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    def grad_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"median": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "median": float(tensor.median()),
            "p95": float(tensor.quantile(0.95)),
            "p99": float(tensor.quantile(0.99)),
            "maximum": float(tensor.max()),
        }

    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "training_artists": len(train_artists),
        "validation_artists": len(validation_artists),
        "mixtures": len(mixture_rows),
        "contexts": len(contexts),
        "reference_input": "styled_only",
        "teacher_decomposition": "full",
        "common_branch": False,
        "reader_end_to_end": True,
        "generator_grad_norm": grad_summary(generator_grad_history),
        "reader_grad_norm": grad_summary(reader_grad_history),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_direct_reference_kv_delta_320(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_direct_delta_320"]
    cfg["output_directory"] = "kv_reference_direct_delta_320_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 50
    cfg["training"]["checkpoint_every"] = 100
    # Measure the true distribution first; the production config is tightened
    # to its p99 after this calibration run.
    cfg["training"]["max_grad_norm"] = 1.0e9
    cfg["training"]["reader_max_grad_norm"] = 1.0e9
    return train_direct_reference_kv_delta_320(
        effective, destination, steps_override=100
    )


@torch.no_grad()
def sample_direct_reference_kv_delta_320(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render held-out teachers against direct activation predictions."""

    from PIL import Image

    from .kv_activation_sampling import (
        NativeKVActivationInjector,
        NativeKVFactorInjector,
        _save_panel,
    )
    from .lora_functional_distillation import _preview_pixels
    from .style_transfer import (
        _load_sampling_vae,
        _optimize_frozen_anima,
        _resolve_anima_model,
    )
    from .synthetic_teacher import _sample_anima_batch
    from .dual_query_external_samples import load_dual_query_external_sample

    sample_cfg = dict(config["kv_reference_direct_delta_320_sample"])
    device = str(sample_cfg.get("device", "cuda"))
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["config"])
    training = dict(cfg["training"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)), dtype=torch.float16,
    )
    mixture_rows = [
        row
        for row in read_records(destination / str(cfg["mixture_manifest"]))
        if str(row["kind"]) in {"pair", "triple", "amplified", "signed"}
        and bool(row.get("enabled", True))
    ]
    _, validation_artists, _ = _direct_delta_artist_split(
        artist_ids, mixture_rows,
        training_artists=int(training.get("training_artists", 256)),
    )
    artist_count = min(int(sample_cfg.get("artists", 8)), len(validation_artists))
    selected = validation_artists[:artist_count]
    selected_ids = [artist_ids[index] for index in selected]

    reader = _load_reader(config, destination, cfg, device)
    reader.load_state_dict(checkpoint["reader"], strict=True)
    reader.requires_grad_(False).eval()
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "direct_cross_attention"))
    if architecture != "direct_cross_attention":
        raise RuntimeError("Direct-delta sample received a different architecture")
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    references = int(sample_cfg.get("references", 4))
    loader = CachedTeacherReferenceLoader(
        destination / str(cfg["synthetic_reference_cache"]),
        split="train", style_ids=selected_ids, batch_size=artist_count,
        references=int(training.get("single_reference_images", 8)),
        seed=int(sample_cfg.get("seed", 20260826)),
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    loaded = loader.load_styles(
        selected_ids,
        references_per_style=references,
        seed=int(sample_cfg.get("reference_seed", 20260826)),
    )
    reference_tokens = loaded["tokens"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    reference_mask = torch.ones(
        reference_tokens.shape[:2], device=device, dtype=torch.bool
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        style_memory = reader(reference_tokens, reference_mask).tokens

    prepared = load_dual_query_external_sample(config, destination)
    generation = dict(prepared["cfg"])
    positive = prepared["positive"].to(device=device, dtype=torch.bfloat16)
    negative = prepared["negative"].to(device=device, dtype=torch.bfloat16)
    if positive.ndim == 2:
        positive = positive[None]
    if negative.ndim == 2:
        negative = negative[None]
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    factor_injector = NativeKVFactorInjector(anima)
    activation_injector = NativeKVActivationInjector(anima, model)
    width, height = int(generation["width"]), int(generation["height"])
    seed = int(generation["seed"])
    steps = int(generation["steps"])
    shift = float(generation.get("flow_shift", 3.0))
    text_cfg = float(generation["cfg"])
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    noise = torch.randn(
        1, 16, 1, height // 8, width // 8,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    batch_size = int(sample_cfg.get("batch_size", 4))

    def denoise(mode: str, strength: float = 1.0) -> torch.Tensor:
        values = []
        for start in range(0, artist_count, batch_size):
            stop = min(artist_count, start + batch_size)
            rows = stop - start
            if mode == "base":
                factor_injector.disable()
                activation_injector.disable()
            elif mode == "teacher":
                activation_injector.disable()
                factor_injector.set_factors(
                    teacher_down[selected[start:stop]].to(
                        device=device, dtype=torch.bfloat16
                    ),
                    teacher_up[selected[start:stop]].to(
                        device=device, dtype=torch.bfloat16
                    ),
                    strength=strength,
                )
            elif mode == "predicted":
                factor_injector.disable()
                activation_injector.set_style(
                    style_memory[start:stop], strength=strength
                )
            else:
                raise ValueError(f"Unknown direct-delta sample mode: {mode}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values.append(_sample_anima_batch(
                    anima,
                    noise.repeat(rows, 1, 1, 1, 1),
                    positive.expand(rows, -1, -1),
                    negative.expand(rows, -1, -1),
                    sigmas,
                    text_cfg=text_cfg,
                    speed=None,
                    generation_seeds=[seed] * rows,
                ).cpu())
        return torch.cat(values)

    baseline = denoise("base")
    teacher = denoise("teacher")
    strengths = [float(value) for value in sample_cfg.get("strengths", [1.0, 2.0])]
    predicted = {strength: denoise("predicted", strength) for strength in strengths}
    factor_injector.close()
    activation_injector.close()
    torch.cuda.empty_cache()

    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(False).eval()
    latent_groups = {
        "Frozen Anima": baseline,
        "Teacher K/V-LoRA 1x": teacher,
        **{f"Predicted {strength:g}x": value for strength, value in predicted.items()},
    }
    images: dict[str, list[Image.Image]] = {}
    reference_cache = destination / str(cfg["synthetic_reference_cache"])
    manifest = {
        int(row["id"]): row
        for row in read_records(
            reference_cache.parent / "manifest.parquet"
        )
    }
    images["Styled references"] = [
        Image.open(manifest[int(ids[0])]["local_path"]).convert("RGB")
        for ids in loaded["ids"]
    ]
    for label, latents in latent_groups.items():
        decoded = []
        for start in range(0, artist_count, batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded.extend(_preview_pixels(
                    vae.decode_to_pixels(latents[start : start + batch_size].to(device))
                ))
        images[label] = decoded
    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    panel = _save_panel(
        output, selected_ids, images, list(images),
        tile_width=int(sample_cfg.get("panel_tile_width", 384)),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "panel": str(panel),
        "heldout_artist_indices": selected,
        "heldout_artist_ids": selected_ids,
        "reference_ids": [list(ids) for ids in loaded["ids"]],
        "strengths": strengths,
        "prompt": str(generation["prompt"]),
        "seed": seed,
    }
    write_json(output / "summary.json", summary)
    return summary


def train_functional_reference_kv_operator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Train a visual rank operator in native-attention functional space.

    No dataset-mean K/V operator is subtracted.  The exact rank-16 K/V LoRA
    activation is retained only as a weak auxiliary target; native
    normalization, attention softmax and O define the primary supervised
    effect.  A controlled batch shares context and Q so artist centering is
    meaningful, and Reader pooling remains trainable.
    """

    cfg = copy.deepcopy(config["kv_reference_functional_operator"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 4000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    validation_artists = int(training.get("validation_artists", 32))
    if not 1 <= validation_artists < len(artist_ids):
        raise ValueError("validation_artists must leave training artists")
    train_artist_count = len(artist_ids) - validation_artists
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    visual_reader = _load_reader(config, destination, cfg, device)
    reader_parameters = _open_functional_reader_pooling(visual_reader)
    reference_images = int(training.get("materialized_reference_images", 12))
    train_reference_images = int(training.get("train_reference_images", 8))
    if reference_images <= train_reference_images:
        raise ValueError("Reference materialization must reserve validation images")
    chunk = int(training.get("materialization_style_chunk", 16))
    reference_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=artist_ids,
        batch_size=chunk,
        references=reference_images,
        seed=seed ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    reference_bank = _materialize_reference_token_bank(
        reference_loader,
        artist_ids,
        references=reference_images,
        seed=seed ^ 0x48554D41,
        chunk_size=chunk,
        device=device,
    )

    contexts = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"].to(device=device, dtype=torch.bfloat16)
    heldout_contexts = int(training.get("heldout_contexts", 32))
    train_context_count = len(contexts) - heldout_contexts
    if train_context_count <= 0:
        raise ValueError("heldout_contexts leaves no training context")
    probes = _load_native_attention_probes(config, destination, device)

    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "bilinear_low_rank_operator"))
    if architecture != "bilinear_low_rank_operator":
        raise ValueError("Functional operator experiment requires bilinear operator")
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=int(visual_reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)

    operator_lr = float(training.get("operator_learning_rate", 2e-4))
    reader_lr = float(training.get("reader_learning_rate", 2e-5))
    optimizer = torch.optim.AdamW(
        [
            {"name": "operator", "params": list(model.parameters()), "lr": operator_lr},
            {"name": "reader", "params": reader_parameters, "lr": reader_lr},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        visual_reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    reference_values = [int(value) for value in training.get("reference_counts", [1, 2, 4])]
    reference_weights = [float(value) for value in training.get("reference_count_weights", [0.5, 0.3, 0.2])]
    attention_queries = int(training.get("attention_queries", 64))
    kv_aux_weight = float(training.get("kv_auxiliary_weight", 0.1))
    functional_weight = float(training.get("functional_weight", 1.0))
    loss_cfg = dict(training.get("functional_loss", {}))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    reader_max_grad_norm = float(training.get("reader_max_grad_norm", 0.25))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-functional-operator")),
            id=str(wandb_cfg.get("id", "kv-reference-functional-operator")),
            resume="allow" if start_step else "never",
            config={"kv_reference_functional_operator": cfg},
        )

    def functional_values(
        style: torch.Tensor,
        context: torch.Tensor,
        indices: torch.Tensor,
        block: int,
        query_seed: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_delta = model(style, context, block)
        teacher_delta = _mixture_target(
            context,
            teacher_down,
            teacher_up,
            indices[:, None],
            torch.ones(len(indices), 1, device=device),
            block,
        )
        probe = probes[block]
        generator = torch.Generator(device=device).manual_seed(query_seed)
        queries = torch.randn(
            len(indices), attention_queries, probe.n_heads, probe.head_dim,
            device=device, dtype=torch.bfloat16, generator=generator,
        )
        zero = torch.zeros_like(student_delta)
        base_key, base_value = probe.project_context(context, zero)
        student_key, student_value = probe.project_context(context, student_delta)
        teacher_key, teacher_value = probe.project_context(context, teacher_delta)
        base_output = probe.attend(queries, base_key, base_value)
        student_effect = probe.attend(queries, student_key, student_value) - base_output
        teacher_effect = probe.attend(queries, teacher_key, teacher_value) - base_output
        functional_loss, values = _functional_centered_attention_loss(
            student_effect,
            teacher_effect,
            centered_huber_weight=float(loss_cfg.get("centered_huber", 1.0)),
            direction_weight=float(loss_cfg.get("direction", 1.0)),
            magnitude_weight=float(loss_cfg.get("magnitude", 0.2)),
            relation_weight=float(loss_cfg.get("relation", 0.5)),
            raw_huber_weight=float(loss_cfg.get("raw_huber", 0.05)),
            temperature=float(loss_cfg.get("temperature", 0.1)),
        )
        kv_loss, kv_values = _normalized_activation_loss(
            student_delta,
            teacher_delta,
            direction_weight=float(training.get("kv_direction_weight", 0.1)),
            magnitude_weight=float(training.get("kv_magnitude_weight", 0.02)),
        )
        values.update({f"kv_{key}": value for key, value in kv_values.items()})
        values["weighted_functional"] = functional_loss.detach() * functional_weight
        values["weighted_kv_auxiliary"] = kv_loss.detach() * kv_aux_weight
        return functional_weight * functional_loss + kv_aux_weight * kv_loss, values

    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            rng = random.Random(seed + step * 1_000_003)
            artists = rng.sample(range(train_artist_count), batch)
            counts = rng.choices(reference_values, weights=reference_weights, k=batch)
            references, reference_mask = _select_reference_tokens(
                reference_bank,
                artists,
                reference_counts=counts,
                reference_start=0,
                reference_stop=train_reference_images,
                rng=rng,
            )
            context_index = rng.randrange(train_context_count)
            context = contexts[context_index : context_index + 1].expand(batch, -1, -1)
            artist_tensor = torch.tensor(artists, device=device, dtype=torch.long)
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            selected_blocks = [
                (first_block + offset) % model.blocks
                for offset in range(blocks_per_step)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = visual_reader(references, reference_mask).tokens
                rows = [
                    functional_values(
                        style,
                        context,
                        artist_tensor,
                        block,
                        seed + step * 1_000_003 + block * 10_007,
                    )
                    for block in selected_blocks
                ]
                loss = torch.stack([row[0] for row in rows]).mean()
            loss.backward()
            operator_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            reader_grad = torch.nn.utils.clip_grad_norm_(
                reader_parameters, reader_max_grad_norm, foreach=True
            )
            lr_scale = min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = operator_lr * lr_scale
            optimizer.param_groups[1]["lr"] = reader_lr * lr_scale
            optimizer.step()
            running["loss"].append(float(loss.detach()))
            running["operator_grad_norm"].append(float(operator_grad))
            running["reader_grad_norm"].append(float(reader_grad))
            running["reference_count"].append(sum(counts) / len(counts))
            for key in rows[0][1]:
                running[key].append(
                    sum(float(row[1][key]) for row in rows) / len(rows)
                )
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["operator_lr"] = optimizer.param_groups[0]["lr"]
                row["reader_lr"] = optimizer.param_groups[1]["lr"]
                print(f"Functional K/V operator step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()

            if validation_every > 0 and step % validation_every == 0:
                model.eval()
                visual_reader.eval()
                val_rows: dict[str, list[float]] = defaultdict(list)
                val_artists = list(range(train_artist_count, len(artist_ids)))
                val_artists = val_artists[: min(16, len(val_artists))]
                val_counts = [min(4, reference_images - train_reference_images)] * len(val_artists)
                val_rng = random.Random(seed ^ step)
                val_refs, val_mask = _select_reference_tokens(
                    reference_bank,
                    val_artists,
                    reference_counts=val_counts,
                    reference_start=train_reference_images,
                    reference_stop=reference_images,
                    rng=val_rng,
                )
                val_indices = torch.tensor(val_artists, device=device, dtype=torch.long)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    val_style = visual_reader(val_refs, val_mask).tokens
                    for context_index in range(train_context_count, min(len(contexts), train_context_count + 2)):
                        val_context = contexts[context_index : context_index + 1].expand(len(val_artists), -1, -1)
                        for block in range(model.blocks):
                            _, values = functional_values(
                                val_style,
                                val_context,
                                val_indices,
                                block,
                                seed ^ step ^ (context_index * 1009 + block * 9176),
                            )
                            for key, value in values.items():
                                val_rows[key].append(float(value))
                validation = {
                    key: sum(values) / len(values)
                    for key, values in val_rows.items()
                }
                print(f"Functional K/V operator validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"val/{key}": value for key, value in validation.items()}, step=step)
                model.train()
                visual_reader.train()

            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path,
                        step=step,
                        model=model,
                        reader=visual_reader,
                        optimizer=optimizer,
                        cfg=cfg,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "training_artists": train_artist_count,
        "validation_artists": validation_artists,
        "operator_rank": model.operator_rank,
        "operator_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reader_trainable_parameters": sum(parameter.numel() for parameter in reader_parameters),
        "teacher_decomposition": "none",
        "primary_objective": "native_attention_functional_centered",
        "kv_auxiliary_weight": kv_aux_weight,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_functional_reference_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_functional_operator"]
    cfg["output_directory"] = "kv_reference_functional_operator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    cfg["training"]["materialized_reference_images"] = 5
    cfg["training"]["train_reference_images"] = 4
    cfg["training"]["reference_counts"] = [1]
    cfg["training"]["reference_count_weights"] = [1.0]
    return train_functional_reference_kv_operator(
        effective, destination, steps_override=2
    )


def smoke_test_reference_conditioned_kv_activation_generator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_activation_generator"]
    cfg["output_directory"] = "kv_reference_activation_generator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    return train_reference_conditioned_kv_activation_generator(
        effective, destination, steps_override=2
    )


def train_reference_conditioned_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_reference_conditioned_kv_activation_generator(
        config,
        destination,
        config_key="kv_reference_bilinear_operator",
    )


def smoke_test_reference_conditioned_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_bilinear_operator"]
    cfg["output_directory"] = "kv_reference_bilinear_operator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    return train_reference_conditioned_kv_activation_generator(
        effective,
        destination,
        steps_override=2,
        config_key="kv_reference_bilinear_operator",
    )


def train_centered_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_reference_conditioned_kv_activation_generator(
        config,
        destination,
        config_key="kv_reference_centered_bilinear_operator",
    )


def smoke_test_centered_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_centered_bilinear_operator"]
    cfg["output_directory"] = "kv_reference_centered_bilinear_operator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    return train_reference_conditioned_kv_activation_generator(
        effective,
        destination,
        steps_override=2,
        config_key="kv_reference_centered_bilinear_operator",
    )


@torch.no_grad()
def sample_reference_conditioned_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render fresh held-out Human references through the learned operator."""

    from .kv_activation_sampling import sample_kv_activation_modulator
    from .kv_generalizing_modulator import _teacher_image_split

    sample_cfg = copy.deepcopy(config["kv_reference_bilinear_sample"])
    device = str(sample_cfg.get("device", "cuda"))
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = dict(checkpoint["config"])
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", ""))
    if architecture != "bilinear_low_rank_operator":
        raise RuntimeError(
            f"Expected a bilinear operator checkpoint, got {architecture!r}"
        )

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    artist_count = min(int(sample_cfg.get("artists", 7)), len(artist_ids))
    positions = torch.linspace(0, len(artist_ids) - 1, artist_count).round().long()
    selected_indices = [int(value) for value in positions.unique().tolist()]
    selected_ids = [artist_ids[index] for index in selected_indices]
    _, validation_image_ids = _teacher_image_split(
        destination / str(cfg["lora_directory"]), artist_ids
    )

    reader = _load_reader(config, destination, cfg, device)
    if "reader" in checkpoint:
        reader.load_state_dict(checkpoint["reader"], strict=True)
    operator = ReferenceConditionedLowRankKVOperator(
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    operator.load_state_dict(checkpoint["model"], strict=True)
    operator.requires_grad_(False).eval()

    maximum_references = max(
        int(value) for value in sample_cfg.get("reference_counts", [1, 4])
    )
    loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=selected_ids,
        batch_size=len(selected_ids),
        references=maximum_references,
        seed=int(sample_cfg.get("seed", 20260824)),
        token_lru_shards=int(sample_cfg.get("token_lru_shards", 8)),
        strict_style_ids=True,
        allowed_image_ids=validation_image_ids,
    )
    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for reference_count_value in sample_cfg.get("reference_counts", [1, 4]):
        reference_count = int(reference_count_value)
        loaded = loader.load_styles(
            selected_ids,
            references_per_style=reference_count,
            seed=int(sample_cfg.get("seed", 20260824))
            + reference_count * 1_000_003,
        )
        references = loaded["tokens"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        reference_mask = torch.ones(
            references.shape[:2], device=device, dtype=torch.bool
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            style_memory = reader(references, reference_mask).tokens
            down_rows: list[torch.Tensor] = []
            up_rows: list[torch.Tensor] = []
            for block in range(operator.blocks):
                down, up, sigma = operator._operator(style_memory, block)
                down_rows.append(down)
                up_rows.append(
                    up.transpose(-1, -2) * sigma[:, :, None, :]
                )
        predicted_down = torch.stack(down_rows, dim=1).cpu()
        predicted_up = torch.stack(up_rows, dim=1).cpu()
        compatibility = {
            "predicted_down": predicted_down,
            "predicted_up": predicted_up,
            "predicted_artist_indices": selected_indices,
            "config": {
                "lora_directory": cfg["lora_directory"],
                "blocks": int(cfg.get("blocks", 28)),
            },
        }
        compatibility_path = output / f"reference-{reference_count}-operator.pt"
        torch.save(compatibility, compatibility_path)

        effective = copy.deepcopy(config)
        render_output = output / f"reference-{reference_count}"
        effective["kv_activation_modulator_sample"] = {
            **dict(effective["kv_activation_modulator_sample"]),
            "checkpoint": str(compatibility_path.relative_to(destination)),
            "output_directory": str(render_output.relative_to(destination)),
            "device": device,
            "artist_indices": selected_indices,
            "predicted_strengths": [
                float(value)
                for value in sample_cfg.get("predicted_strengths", [1.0, 1.5])
            ],
            "batch_size": int(sample_cfg.get("batch_size", 4)),
            "panel_tile_width": int(sample_cfg.get("panel_tile_width", 320)),
        }
        rendered = sample_kv_activation_modulator(effective, destination)
        rendered["reference_ids"] = [list(ids) for ids in loaded["ids"]]
        results[f"{reference_count}ref"] = rendered

    del operator, reader, checkpoint, teacher_down, teacher_up
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "checkpoint": str(checkpoint_path),
        "artists": selected_ids,
        "artist_indices": selected_indices,
        "fresh_human_validation_references": True,
        "results": results,
    }
    write_json(output / "summary.json", summary)
    return summary


@torch.no_grad()
def _sample_external_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path, *, config_key: str
) -> dict[str, Any]:
    """Render the established TestSample 1-7 external references."""

    from .dual_query_external_samples import load_dual_query_external_sample
    from .kv_activation_sampling import sample_kv_activation_modulator

    sample_cfg = copy.deepcopy(config[config_key])
    device = str(sample_cfg.get("device", "cuda"))
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = dict(checkpoint["config"])
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", ""))
    if architecture != "bilinear_low_rank_operator":
        raise RuntimeError(
            f"Expected a bilinear operator checkpoint, got {architecture!r}"
        )

    prepared = load_dual_query_external_sample(config, destination)
    references = prepared["reference_tokens"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )[:, None]
    reference_mask = torch.ones(
        references.shape[:2], device=device, dtype=torch.bool
    )
    reader = _load_reader(config, destination, cfg, device)
    if "reader" in checkpoint:
        reader.load_state_dict(checkpoint["reader"], strict=True)
    _, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    operator = ReferenceConditionedLowRankKVOperator(
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    operator.load_state_dict(checkpoint["model"], strict=True)
    operator.requires_grad_(False).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        style_memory = reader(references, reference_mask).tokens
        down_rows: list[torch.Tensor] = []
        up_rows: list[torch.Tensor] = []
        for block in range(operator.blocks):
            down, up, sigma = operator._operator(style_memory, block)
            down_rows.append(down)
            up_rows.append(up.transpose(-1, -2) * sigma[:, :, None, :])
    predicted_down = torch.stack(down_rows, dim=1).cpu()
    predicted_up = torch.stack(up_rows, dim=1).cpu()

    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    compatibility_path = output / "fixed-reference-operator.pt"
    row_indices = list(range(len(prepared["paths"])))
    compatibility = {
            "predicted_down": predicted_down,
            "predicted_up": predicted_up,
            "predicted_artist_indices": row_indices,
            "config": {
                "lora_directory": cfg["lora_directory"],
                "blocks": int(cfg.get("blocks", 28)),
            },
    }
    if str(cfg.get("teacher_decomposition", "full")) == "centered":
        common_path = checkpoint_path.parent.parent / "frozen_common_operator.pt"
        compatibility["common_operator"] = torch.load(
            common_path, map_location="cpu", weights_only=True
        )["common_operator"]
    torch.save(compatibility, compatibility_path)
    effective = copy.deepcopy(config)
    labels = [f"TestSample {index + 1}" for index in row_indices]
    effective["kv_activation_modulator_sample"] = {
        **dict(effective["kv_activation_modulator_sample"]),
        "checkpoint": str(compatibility_path.relative_to(destination)),
        "output_directory": str(output.relative_to(destination)),
        "device": device,
        "artist_indices": row_indices,
        "artist_labels": labels,
        "predicted_strengths": [
            float(value)
            for value in sample_cfg.get("predicted_strengths", [1.0, 1.5])
        ],
        "batch_size": int(sample_cfg.get("batch_size", 4)),
        "panel_tile_width": int(sample_cfg.get("panel_tile_width", 320)),
        "include_teacher": False,
        "include_reference_images": True,
    }
    rendered = sample_kv_activation_modulator(effective, destination)
    rendered["checkpoint"] = str(checkpoint_path)
    rendered["reference_paths"] = [str(path) for path in prepared["paths"]]
    write_json(output / "summary.json", rendered)
    del operator, reader, checkpoint, teacher_down, teacher_up
    gc.collect()
    torch.cuda.empty_cache()
    return rendered


def sample_external_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_bilinear_fixed_sample",
    )


def sample_external_reference_centered_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_centered_bilinear_fixed_sample",
    )


def sample_external_reference_functional_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_functional_operator_fixed_sample",
    )
