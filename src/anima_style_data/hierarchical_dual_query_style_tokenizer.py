from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HierarchicalStyleTokenizerOutput:
    tokens: torch.Tensor
    per_reference_tokens: torch.Tensor
    reconstruction: torch.Tensor | None
    reconstruction_target: torch.Tensor | None
    artist_tokens: torch.Tensor
    diversity_tokens: torch.Tensor


class HierarchicalDualQueryStyleTokenizer(nn.Module):
    """Read each Dual-query reference before pooling references slot-wise.

    The frozen cache contains 64 spatial, 16 global, and optionally four artist
    summary tokens per image.  Keeping the per-image reader ahead of reference
    pooling preserves that hierarchy while the shared slot identities make the
    aggregation permutation invariant in the reference dimension.
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        spatial_tokens: int = 64,
        global_tokens: int = 16,
        artist_summary_tokens: int = 4,
        include_artist_summary: bool = True,
        output_tokens: int = 16,
        heads: int = 16,
        per_reference_layers: int = 1,
        per_reference_ff_dim: int = 4096,
        reference_score_dim: int = 256,
        reconstruction_layers: int = 1,
        reconstruction_ff_dim: int = 2048,
        initial_output_rms: float = 0.15,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Style width must be divisible by attention heads")
        if min(spatial_tokens, global_tokens, output_tokens, per_reference_layers) <= 0:
            raise ValueError("Token counts and per-reference depth must be positive")
        if artist_summary_tokens < 0 or reconstruction_layers < 0:
            raise ValueError("Summary token count and reconstruction depth cannot be negative")
        if initial_output_rms <= 0:
            raise ValueError("initial_output_rms must be positive")

        self.dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.include_artist_summary = bool(include_artist_summary)
        self.output_tokens = int(output_tokens)
        self.cached_tokens = (
            self.spatial_tokens + self.global_tokens + self.artist_summary_tokens
        )
        self.active_tokens = self.spatial_tokens + self.global_tokens + (
            self.artist_summary_tokens if self.include_artist_summary else 0
        )

        self.input_norms = nn.ModuleDict(
            {
                "spatial": nn.LayerNorm(dim),
                "global": nn.LayerNorm(dim),
                "summary": nn.LayerNorm(dim),
            }
        )
        self.input_projections = nn.ModuleDict(
            {
                "spatial": nn.Linear(dim, dim),
                "global": nn.Linear(dim, dim),
                "summary": nn.Linear(dim, dim),
            }
        )
        self.type_embeddings = nn.Parameter(torch.empty(3, dim))
        self.reference_queries = nn.Parameter(torch.empty(1, output_tokens, dim))
        reader_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=per_reference_ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.reference_reader = nn.TransformerDecoder(
            reader_layer,
            num_layers=per_reference_layers,
            norm=nn.LayerNorm(dim),
        )
        self.reference_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, reference_score_dim),
            nn.SiLU(),
            nn.Linear(reference_score_dim, 1, bias=False),
        )

        # These embeddings remain explicit at the native Anima-token boundary;
        # the artist/diversity losses use only reference_conditioned so a static
        # slot identity cannot satisfy them by itself.
        self.slot_embeddings = nn.Parameter(torch.empty(1, output_tokens, dim))
        self.output_input_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)

        self.reconstruction_queries: nn.Parameter | None
        self.reconstruction_decoder: nn.Module | None
        if reconstruction_layers:
            self.reconstruction_queries = nn.Parameter(
                torch.empty(1, self.active_tokens, dim)
            )
            reconstruction_layer = nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=reconstruction_ff_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.reconstruction_decoder = nn.TransformerDecoder(
                reconstruction_layer,
                num_layers=reconstruction_layers,
                norm=nn.LayerNorm(dim),
            )
        else:
            self.register_parameter("reconstruction_queries", None)
            self.reconstruction_decoder = None
        self.reset_parameters(initial_output_rms=float(initial_output_rms))

    def reset_parameters(self, *, initial_output_rms: float) -> None:
        nn.init.normal_(self.type_embeddings, std=self.dim**-0.5)
        nn.init.normal_(self.reference_queries, std=self.dim**-0.5)
        nn.init.normal_(self.slot_embeddings, std=self.dim**-0.5)
        if self.reconstruction_queries is not None:
            nn.init.normal_(self.reconstruction_queries, std=self.dim**-0.5)
        # Calibrate only the starting distribution. There is deliberately no
        # output normalization or learned global RMS constraint after this map.
        nn.init.xavier_uniform_(self.output_projection.weight, gain=initial_output_rms)
        nn.init.zeros_(self.output_projection.bias)

    def _typed_memory(
        self, references: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_end = self.spatial_tokens
        global_end = spatial_end + self.global_tokens
        pieces: list[torch.Tensor] = []
        normalized: list[torch.Tensor] = []
        ranges = (
            ("spatial", references[:, :spatial_end], 0),
            ("global", references[:, spatial_end:global_end], 1),
        )
        for name, values, type_index in ranges:
            values = self.input_norms[name](values)
            normalized.append(values)
            pieces.append(
                self.input_projections[name](values) + self.type_embeddings[type_index]
            )
        if self.include_artist_summary:
            values = self.input_norms["summary"](
                references[:, global_end : global_end + self.artist_summary_tokens]
            )
            normalized.append(values)
            pieces.append(
                self.input_projections["summary"](values) + self.type_embeddings[2]
            )
        return torch.cat(pieces, dim=1), torch.cat(normalized, dim=1)

    def _encode_valid_references(
        self, references: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, reconstruction_target = self._typed_memory(references)
        queries = self.reference_queries.expand(references.shape[0], -1, -1)
        return self.reference_reader(queries, memory), reconstruction_target

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> HierarchicalStyleTokenizerOutput:
        if references.ndim != 4:
            raise ValueError("references must be [batch, references, tokens, dim]")
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference mask does not match reference batch")
        if references.shape[2:] != (self.cached_tokens, self.dim):
            raise ValueError(
                f"Expected reference tail {(self.cached_tokens, self.dim)}, "
                f"got {tuple(references.shape[2:])}"
            )
        if not reference_mask.is_cuda and not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample needs at least one reference")

        valid_source = references[reference_mask]
        valid_encoded, valid_targets = self._encode_valid_references(valid_source)
        per_reference = valid_encoded.new_zeros(
            *references.shape[:2], self.output_tokens, self.dim
        )
        per_reference[reference_mask] = valid_encoded

        logits = self.reference_score(per_reference).squeeze(-1)
        logits = logits.masked_fill(
            ~reference_mask[..., None], torch.finfo(logits.dtype).min
        )
        weights = logits.softmax(dim=1)
        aggregated = torch.sum(weights[..., None] * per_reference, dim=1)
        reference_conditioned = self.output_projection(
            self.output_input_norm(aggregated)
        )
        tokens = reference_conditioned + self.slot_embeddings

        reconstruction = None
        reconstruction_target = None
        if reconstruct:
            if self.reconstruction_decoder is None or self.reconstruction_queries is None:
                raise RuntimeError("The reconstruction decoder is disabled")
            first = reference_mask.to(torch.int64).argmax(dim=1)
            rows = torch.arange(references.shape[0], device=references.device)
            selected_encoded = per_reference[rows, first]
            # `valid_targets` is in compact valid-reference order. Build the
            # dense target only for the selected reference of each batch row.
            dense_targets = valid_targets.new_zeros(
                *references.shape[:2], self.active_tokens, self.dim
            )
            dense_targets[reference_mask] = valid_targets
            reconstruction_target = dense_targets[rows, first].detach()
            reconstruction = self.reconstruction_decoder(
                self.reconstruction_queries.expand(references.shape[0], -1, -1),
                selected_encoded,
            )

        return HierarchicalStyleTokenizerOutput(
            tokens=tokens,
            per_reference_tokens=per_reference,
            reconstruction=reconstruction,
            reconstruction_target=reconstruction_target,
            artist_tokens=reference_conditioned,
            diversity_tokens=reference_conditioned,
        )
