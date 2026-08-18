from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .query_style_tokenizer import QueryStyleTokenizerOutput


@dataclass
class TypedMultiDescriptorOutput(QueryStyleTokenizerOutput):
    artist_tokens: torch.Tensor | None = None
    diversity_tokens: torch.Tensor | None = None
    attention_maps: torch.Tensor | None = None
    reference_conditioned_tokens: torch.Tensor | None = None
    descriptor_tokens: torch.Tensor | None = None
    output_gain: torch.Tensor | None = None


class _CrossAttentionResidualBlock(nn.Module):
    """Use learned queries for routing without emitting their common residual."""

    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_memory = self.memory_norm(memory)
        attended, attention = self.attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        # Raw learned queries are identical for every sample. Carrying them
        # into the value stream made the first experiment emit one dominant
        # common style. The queries define routing only; emitted values must be
        # conditioned on memory.
        values = attended
        return values + self.ff(self.ff_norm(values)), attention


class _DescriptorGroupHead(nn.Module):
    """Expand a small typed descriptor group into explicit conditional slots."""

    def __init__(
        self,
        *,
        dim: int,
        descriptor_tokens: int,
        output_tokens: int,
        bottleneck_dim: int,
        slot_embedding_scale: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.descriptor_tokens = int(descriptor_tokens)
        self.output_tokens = int(output_tokens)
        self.slot_embedding_scale = float(slot_embedding_scale)
        self.input_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_tokens * dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, output_tokens * dim),
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(1, output_tokens, dim)
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        batch = descriptors.shape[0]
        values = self.mlp(self.input_norm(descriptors).flatten(1)).reshape(
            batch, self.output_tokens, self.dim
        )
        return values + self.slot_embedding_scale * self.slot_embedding


class TypedMultiDescriptorCompactStyleTokenizer(nn.Module):
    """Moderately compress typed Dual-query memory before native token output.

    Every reference becomes eight aligned descriptors instead of either one
    pooled vector or all 84 cached tokens. References are pooled independently
    for each descriptor role. The final sixteen slots are produced either by
    a routing-only cross-attention layer or by typed dense group heads. The
    grouped path gives every slot a reference-conditioned output matrix while
    preserving the semantic source type until the final token expansion.
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        spatial_tokens: int = 64,
        global_tokens: int = 16,
        artist_summary_tokens: int = 4,
        include_artist_summary: bool = True,
        spatial_descriptors: int = 4,
        global_descriptors: int = 2,
        artist_descriptors: int = 2,
        output_tokens: int = 16,
        heads: int = 16,
        ff_dim: int = 2048,
        slot_modulation_scale: float = 0.25,
        output_mode: str = "attention",
        descriptor_group_size: int = 2,
        group_bottleneck_dim: int = 512,
        group_slot_embedding_scale: float = 1.0,
        output_gain_center: float = 0.15,
        output_gain_log_span: float = 0.50,
        learnable_output_gain: bool = False,
    ) -> None:
        super().__init__()
        descriptor_counts = (
            spatial_descriptors,
            global_descriptors,
            artist_descriptors,
        )
        if dim % heads:
            raise ValueError("Style width must be divisible by attention heads")
        if min(
            spatial_tokens,
            global_tokens,
            artist_summary_tokens,
            output_tokens,
            *descriptor_counts,
        ) <= 0:
            raise ValueError("Token and descriptor counts must be positive")
        if not include_artist_summary:
            raise ValueError("Typed multi-descriptor training requires artist summary")
        if output_gain_center <= 0 or output_gain_log_span < 0:
            raise ValueError("Output gain parameters must be non-negative")
        if output_mode not in {"attention", "grouped_mlp"}:
            raise ValueError(f"Unknown typed output mode: {output_mode}")
        if (
            descriptor_group_size <= 0
            or group_bottleneck_dim <= 0
            or group_slot_embedding_scale < 0
        ):
            raise ValueError("Grouped output dimensions must be positive")

        self.dim = int(dim)
        self.source_dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.include_artist_summary = True
        self.cached_tokens = (
            self.spatial_tokens
            + self.global_tokens
            + self.artist_summary_tokens
        )
        self.descriptor_counts = tuple(int(value) for value in descriptor_counts)
        self.descriptor_tokens = sum(self.descriptor_counts)
        self.output_tokens = int(output_tokens)
        self.output_mode = str(output_mode)
        self.descriptor_group_size = int(descriptor_group_size)
        self.group_bottleneck_dim = int(group_bottleneck_dim)
        self.group_slot_embedding_scale = float(group_slot_embedding_scale)
        self.slot_modulation_scale = float(slot_modulation_scale)
        self.output_gain_center = float(output_gain_center)
        self.output_gain_log_span = float(output_gain_log_span)
        self.learnable_output_gain = bool(learnable_output_gain)

        self.input_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.type_embeddings = nn.Parameter(torch.empty(3, dim))
        self.spatial_positions = nn.Parameter(torch.empty(spatial_tokens, dim))
        self.descriptor_queries = nn.Parameter(
            torch.empty(1, self.descriptor_tokens, dim)
        )
        self.descriptor_reader = _CrossAttentionResidualBlock(
            dim, heads, ff_dim
        )

        self.reference_score_norm = nn.LayerNorm(dim)
        self.reference_score_queries = nn.Parameter(
            torch.empty(self.descriptor_tokens, dim)
        )

        self.output_slot_queries: nn.Parameter | None = None
        self.output_reader: _CrossAttentionResidualBlock | None = None
        self.output_slot_modulation: nn.Parameter | None = None
        self.output_group_slices: tuple[tuple[int, int], ...] = ()
        self.output_group_heads = nn.ModuleList()
        if self.output_mode == "attention":
            self.output_slot_queries = nn.Parameter(
                torch.empty(1, output_tokens, dim)
            )
            self.output_reader = _CrossAttentionResidualBlock(
                dim, heads, ff_dim
            )
            self.output_slot_modulation = nn.Parameter(
                torch.zeros(1, output_tokens, dim)
            )
        else:
            if output_tokens % self.descriptor_tokens:
                raise ValueError(
                    "Grouped output tokens must be a multiple of descriptors"
                )
            expansion = output_tokens // self.descriptor_tokens
            group_slices = []
            start = 0
            for type_count in self.descriptor_counts:
                type_end = start + type_count
                while start < type_end:
                    end = min(start + self.descriptor_group_size, type_end)
                    descriptor_count = end - start
                    group_slices.append((start, end))
                    self.output_group_heads.append(
                        _DescriptorGroupHead(
                            dim=dim,
                            descriptor_tokens=descriptor_count,
                            output_tokens=descriptor_count * expansion,
                            bottleneck_dim=self.group_bottleneck_dim,
                            slot_embedding_scale=self.group_slot_embedding_scale,
                        )
                    )
                    start = end
            self.output_group_slices = tuple(group_slices)
        self.output_norm = nn.LayerNorm(dim)
        self.gain_norm = nn.LayerNorm(dim) if self.learnable_output_gain else None
        self.gain_head = nn.Linear(dim, 1) if self.learnable_output_gain else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.type_embeddings, std=self.dim**-0.5)
        nn.init.normal_(self.spatial_positions, std=self.dim**-0.5)
        # These values are residual-stream tokens, not low-norm parameter
        # markers. RMS-one initialization prevents common attended memory from
        # immediately erasing descriptor and output-slot identities.
        nn.init.normal_(self.descriptor_queries, std=1.0)
        if self.output_slot_queries is not None:
            nn.init.normal_(self.output_slot_queries, std=1.0)
        nn.init.normal_(self.reference_score_queries, std=self.dim**-0.5)
        for head in self.output_group_heads:
            nn.init.normal_(head.slot_embedding, std=self.dim**-0.5)
        if self.gain_head is not None:
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)

    def _read_typed_descriptors(
        self, references: torch.Tensor
    ) -> torch.Tensor:
        batch_references = references.shape[0]
        spatial_end = self.spatial_tokens
        global_end = spatial_end + self.global_tokens
        memories = (
            references[:, :spatial_end] + self.spatial_positions,
            references[:, spatial_end:global_end],
            references[:, global_end:],
        )
        query_start = 0
        descriptors = []
        for type_index, (memory, count) in enumerate(
            zip(memories, self.descriptor_counts, strict=True)
        ):
            normalized_memory = self.input_norms[type_index](memory)
            normalized_memory = normalized_memory + self.type_embeddings[type_index]
            queries = self.descriptor_queries[
                :, query_start : query_start + count
            ].expand(batch_references, -1, -1)
            values, _ = self.descriptor_reader(
                queries, normalized_memory
            )
            descriptors.append(values)
            query_start += count
        return torch.cat(descriptors, dim=1)

    def _pool_references(
        self,
        descriptors: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.reference_score_norm(descriptors)
        logits = torch.einsum(
            "brsd,sd->brs", normalized, self.reference_score_queries
        ) / math.sqrt(self.dim)
        logits = logits.masked_fill(
            ~reference_mask[:, :, None], torch.finfo(logits.dtype).min
        )
        weights = logits.softmax(dim=1)
        return torch.sum(weights[..., None] * descriptors, dim=1)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> TypedMultiDescriptorOutput:
        if reconstruct:
            raise ValueError("Reconstruction belongs to the frozen Resampler")
        if references.ndim != 4:
            raise ValueError("references must be [batch, references, tokens, dim]")
        if references.shape[2:] != (self.cached_tokens, self.dim):
            raise ValueError(
                f"Expected reference tail {(self.cached_tokens, self.dim)}, "
                f"got {tuple(references.shape[2:])}"
            )
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference_mask does not match references")
        if not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain a reference")

        batch, reference_count = references.shape[:2]
        flat_references = references.reshape(
            batch * reference_count, self.cached_tokens, self.dim
        )
        flat_descriptors = self._read_typed_descriptors(flat_references)
        per_reference = flat_descriptors.reshape(
            batch, reference_count, self.descriptor_tokens, self.dim
        )
        style_memory = self._pool_references(per_reference, reference_mask)

        if self.output_mode == "attention":
            assert self.output_slot_queries is not None
            assert self.output_reader is not None
            assert self.output_slot_modulation is not None
            output_queries = self.output_slot_queries.expand(batch, -1, -1)
            output_queries, output_attention = self.output_reader(
                output_queries, style_memory
            )
            output_queries = output_queries * (
                1.0
                + self.slot_modulation_scale
                * torch.tanh(self.output_slot_modulation)
            )
        else:
            output_queries = torch.cat(
                [
                    head(style_memory[:, start:end])
                    for head, (start, end) in zip(
                        self.output_group_heads,
                        self.output_group_slices,
                        strict=True,
                    )
                ],
                dim=1,
            )
            output_attention = None
        normalized = self.output_norm(output_queries)
        if self.gain_head is None:
            gain = torch.full(
                (batch, 1),
                self.output_gain_center,
                device=references.device,
                dtype=torch.float32,
            )
        else:
            assert self.gain_norm is not None
            gain_logits = self.gain_head(
                self.gain_norm(style_memory.mean(dim=1))
            ).float()
            gain = self.output_gain_center * torch.exp(
                self.output_gain_log_span * torch.tanh(gain_logits)
            )
        tokens = (normalized.float() * gain[:, None]).to(references.dtype)

        return TypedMultiDescriptorOutput(
            tokens=tokens,
            per_reference_tokens=per_reference,
            reconstruction=None,
            reconstruction_target=None,
            artist_tokens=style_memory[:, -self.descriptor_counts[-1] :],
            diversity_tokens=tokens,
            attention_maps=(
                output_attention[:, None]
                if output_attention is not None
                else None
            ),
            reference_conditioned_tokens=tokens,
            descriptor_tokens=style_memory,
            output_gain=gain,
        )
