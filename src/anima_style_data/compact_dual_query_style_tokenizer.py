from __future__ import annotations

import torch

from .query_style_tokenizer import QueryStyleTokenizerOutput
from .style_tokenizer import AnimaStyleTokenizer


class CompactDualQueryStyleTokenizer(AnimaStyleTokenizer):
    """Apply the successful compact native-context tokenizer to Dual-query tokens.

    This deliberately preserves the original small StyleTokenizer's two-stage
    attention pooling, MLP expansion, final LayerNorm, and learned global RMS.
    The only changed contract is the frozen per-reference input: 80 Dual-query
    query tokens plus the optional four artist-summary tokens.
    """

    def __init__(
        self,
        *,
        source_dim: int = 1024,
        context_dim: int = 1024,
        query_tokens: int = 80,
        artist_summary_tokens: int = 4,
        include_artist_summary: bool = True,
        output_tokens: int = 16,
        bottleneck_dim: int = 512,
        score_hidden_dim: int = 256,
        output_rms_init: float = 0.15,
    ) -> None:
        super().__init__(
            source_dim=source_dim,
            context_dim=context_dim,
            output_tokens=output_tokens,
            bottleneck_dim=bottleneck_dim,
            score_hidden_dim=score_hidden_dim,
            output_rms_init=output_rms_init,
        )
        self.dim = int(context_dim)
        if query_tokens <= 0 or artist_summary_tokens < 0:
            raise ValueError("Dual-query token counts are invalid")
        self.query_tokens = int(query_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.include_artist_summary = bool(include_artist_summary)

    @property
    def cached_tokens(self) -> int:
        return self.query_tokens + self.artist_summary_tokens

    @property
    def active_tokens(self) -> int:
        return self.query_tokens + (
            self.artist_summary_tokens if self.include_artist_summary else 0
        )

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> QueryStyleTokenizerOutput:
        if reconstruct:
            raise ValueError(
                "Compact StyleTokenizer reconstruction belongs to the frozen Resampler"
            )
        if references.ndim != 4:
            raise ValueError("references must be [batch, references, tokens, dim]")
        if references.shape[2:] != (self.cached_tokens, self.source_dim):
            raise ValueError(
                f"Expected reference tail {(self.cached_tokens, self.source_dim)}, "
                f"got {tuple(references.shape[2:])}"
            )
        selected = references[:, :, : self.active_tokens]
        tokens = super().forward(selected, reference_mask)
        return QueryStyleTokenizerOutput(
            tokens=tokens,
            per_reference_tokens=selected,
            reconstruction=None,
            reconstruction_target=None,
        )
