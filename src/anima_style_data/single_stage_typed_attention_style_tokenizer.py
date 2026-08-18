from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .query_style_tokenizer import QueryStyleTokenizerOutput


@dataclass
class SingleStageTypedAttentionOutput(QueryStyleTokenizerOutput):
    artist_tokens: torch.Tensor | None = None


class _SharedTypedReader(nn.Module):
    """Read one token type without carrying learned queries into the output."""

    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
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
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(queries),
            memory,
            memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return attended + self.ff(self.ff_norm(attended))


class SingleStageTypedAttentionStyleTokenizer(nn.Module):
    """Read all references once as three typed, permutation-invariant sets."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        spatial_tokens: int = 64,
        global_tokens: int = 16,
        artist_summary_tokens: int = 4,
        spatial_output_tokens: int = 8,
        global_output_tokens: int = 4,
        artist_output_tokens: int = 4,
        heads: int = 16,
        ff_dim: int = 2048,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Style width must be divisible by attention heads")
        counts = (
            spatial_tokens,
            global_tokens,
            artist_summary_tokens,
            spatial_output_tokens,
            global_output_tokens,
            artist_output_tokens,
        )
        if min(counts) <= 0:
            raise ValueError("Typed token counts must be positive")

        self.dim = int(dim)
        self.source_dim = int(dim)
        self.context_dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.cached_tokens = (
            self.spatial_tokens
            + self.global_tokens
            + self.artist_summary_tokens
        )
        self.output_counts = (
            int(spatial_output_tokens),
            int(global_output_tokens),
            int(artist_output_tokens),
        )
        self.output_tokens = sum(self.output_counts)

        self.input_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.type_embeddings = nn.Parameter(torch.empty(3, dim))
        self.spatial_positions = nn.Parameter(
            torch.empty(self.spatial_tokens, dim)
        )
        self.queries = nn.ParameterList(
            nn.Parameter(torch.empty(1, count, dim))
            for count in self.output_counts
        )
        self.reader = _SharedTypedReader(dim, heads, ff_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.type_embeddings, std=self.dim**-0.5)
        nn.init.normal_(self.spatial_positions, std=self.dim**-0.5)
        for query in self.queries:
            nn.init.normal_(query, std=self.dim**-0.5)

    def _typed_memories(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        batch, reference_count, _, dim = references.shape
        spatial_end = self.spatial_tokens
        global_end = spatial_end + self.global_tokens
        groups = (
            references[:, :, :spatial_end]
            + self.spatial_positions[None, None],
            references[:, :, spatial_end:global_end],
            references[:, :, global_end:],
        )
        result = []
        for type_index, group in enumerate(groups):
            tokens_per_reference = group.shape[2]
            memory = self.input_norms[type_index](group)
            memory = memory + self.type_embeddings[type_index]
            memory = memory.reshape(
                batch, reference_count * tokens_per_reference, dim
            )
            padding_mask = (
                ~reference_mask[:, :, None]
                .expand(-1, -1, tokens_per_reference)
                .reshape(batch, reference_count * tokens_per_reference)
            )
            result.append((memory, padding_mask))
        return tuple(result)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> SingleStageTypedAttentionOutput:
        if reconstruct:
            raise ValueError(
                "Single-stage StyleTokenizer reconstruction belongs to the "
                "frozen Resampler"
            )
        if references.ndim != 4:
            raise ValueError(
                "references must have shape [batch, references, tokens, dim]"
            )
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference mask does not match reference batch")
        if references.shape[2:] != (self.cached_tokens, self.dim):
            raise ValueError(
                f"Expected reference tail {(self.cached_tokens, self.dim)}, "
                f"got {tuple(references.shape[2:])}"
            )
        if not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample needs at least one reference")

        memories = self._typed_memories(references, reference_mask)
        batch = references.shape[0]
        outputs = [
            self.reader(
                query.expand(batch, -1, -1),
                memory,
                padding_mask,
            )
            for query, (memory, padding_mask) in zip(
                self.queries, memories, strict=True
            )
        ]
        tokens = torch.cat(outputs, dim=1).to(dtype=references.dtype)
        artist_tokens = outputs[-1].to(dtype=references.dtype)
        return SingleStageTypedAttentionOutput(
            tokens=tokens,
            per_reference_tokens=references,
            reconstruction=None,
            reconstruction_target=None,
            artist_tokens=artist_tokens,
        )
