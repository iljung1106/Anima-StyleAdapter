from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .dual_query_style_tokenizer import DualQueryCachedStyleLoader
from .query_style_tokenizer import QueryStyleTokenizerOutput
from .style_transfer import StyleEpisode


def cache_global_query_multimode_text(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Build the isolated six-variant post-LLM text cache for this experiment."""

    from .anima_cache import cache_anima_text_conditions

    copied = copy.deepcopy(config)
    overrides = dict(config["global_query_multimode_style_tokenizer"]["text_cache"])
    copied["anima_cache"]["text"].update(overrides)
    return cache_anima_text_conditions(copied, destination)


@dataclass
class GlobalQueryStyleTokenizerOutput(QueryStyleTokenizerOutput):
    artist_tokens: torch.Tensor | None = None
    diversity_tokens: torch.Tensor | None = None
    attention_maps: torch.Tensor | None = None
    reference_conditioned_tokens: torch.Tensor | None = None


class _PreNormMemoryBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(values)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        values = values + attended
        return values + self.ff(self.ff_norm(values))


class _PreNormQueryBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, batch_first=True
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
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.cross_attention(
            self.cross_query_norm(queries),
            self.cross_memory_norm(memory),
            self.cross_memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        queries = queries + attended
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        queries = queries + self.ff(self.ff_norm(queries))
        return queries, attention


class _SlotPreservingQueryBlock(nn.Module):
    """Read memory without mixing the identities of the output slots."""

    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, batch_first=True
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
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_memory = self.cross_memory_norm(memory)
        attended, attention = self.cross_attention(
            self.cross_query_norm(queries),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        queries = queries + attended
        return queries + self.ff(self.ff_norm(queries)), attention


class GlobalQueryMemoryStyleTokenizer(nn.Module):
    """Read typed, per-reference Dual-query memories without early pooling."""

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
        local_layers: int = 1,
        cross_layers: int = 2,
        ff_dim: int = 2048,
        output_rms_init: float = 0.15,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Style width must be divisible by attention heads")
        if min(spatial_tokens, global_tokens, output_tokens, local_layers, cross_layers) <= 0:
            raise ValueError("Token counts and depths must be positive")
        if artist_summary_tokens <= 0:
            raise ValueError("Artist-summary tokens must be present")
        if not include_artist_summary:
            raise ValueError("The global-query model requires artist-summary tokens")
        self.dim = int(dim)
        self.source_dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.include_artist_summary = True
        self.cached_tokens = self.spatial_tokens + self.global_tokens + self.artist_summary_tokens
        self.output_tokens = int(output_tokens)
        self.output_rms_init = float(output_rms_init)

        self.input_norm = nn.LayerNorm(dim)
        self.type_embedding = nn.Parameter(torch.empty(3, dim))
        self.spatial_position = nn.Parameter(torch.empty(spatial_tokens, dim))
        self.reference_register = nn.Parameter(torch.empty(1, 1, dim))
        self.local_blocks = nn.ModuleList(
            _PreNormMemoryBlock(dim, heads, ff_dim) for _ in range(local_layers)
        )
        self.output_queries = nn.Parameter(torch.empty(1, output_tokens, dim))
        self.query_blocks = nn.ModuleList(
            _PreNormQueryBlock(dim, heads, ff_dim) for _ in range(cross_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        std = self.dim**-0.5
        nn.init.normal_(self.type_embedding, std=std)
        nn.init.normal_(self.spatial_position, std=std)
        nn.init.normal_(self.reference_register, std=std)
        nn.init.normal_(self.output_queries, std=std)
        # A unit-RMS normalized hidden vector mapped by this initialization has
        # the requested RMS only at initialization. No runtime RMS constraint
        # is applied, so reference- and slot-dependent strength remains learnable.
        nn.init.normal_(
            self.output_projection.weight,
            std=self.output_rms_init / (self.dim**0.5),
        )
        nn.init.zeros_(self.output_projection.bias)

    def _typed_memory(self, references: torch.Tensor) -> torch.Tensor:
        spatial_end = self.spatial_tokens
        global_end = spatial_end + self.global_tokens
        spatial = (
            references[:, :, :spatial_end]
            + self.type_embedding[0]
            + self.spatial_position
        )
        global_values = references[:, :, spatial_end:global_end] + self.type_embedding[1]
        artist = references[:, :, global_end:] + self.type_embedding[2]
        return torch.cat((spatial, global_values, artist), dim=2)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> GlobalQueryStyleTokenizerOutput:
        if reconstruct:
            raise ValueError(
                "Reconstruction belongs to the frozen Dual-query Resampler"
            )
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
        memory = self.input_norm(self._typed_memory(references))
        memory = memory.reshape(batch * reference_count, self.cached_tokens, self.dim)
        register = self.reference_register.expand(batch * reference_count, -1, -1)
        memory = torch.cat((register, memory), dim=1)
        for block in self.local_blocks:
            memory = block(memory)
        memory_per_reference = memory.reshape(
            batch, reference_count, self.cached_tokens + 1, self.dim
        )
        memory = memory_per_reference.flatten(1, 2)
        memory_padding_mask = (~reference_mask).unsqueeze(-1).expand(
            -1, -1, self.cached_tokens + 1
        ).reshape(batch, -1)

        queries = self.output_queries.expand(batch, -1, -1)
        attentions = []
        for block in self.query_blocks:
            queries, attention = block(queries, memory, memory_padding_mask)
            attentions.append(attention)
        normalized = self.final_norm(queries)
        tokens = self.output_projection(normalized).to(dtype=references.dtype)
        attention_maps = torch.stack(attentions, dim=1)
        return GlobalQueryStyleTokenizerOutput(
            tokens=tokens,
            per_reference_tokens=memory_per_reference,
            reconstruction=None,
            reconstruction_target=None,
            artist_tokens=tokens,
            diversity_tokens=tokens,
            attention_maps=attention_maps,
            reference_conditioned_tokens=tokens,
        )


class SlotPreservingGlobalQueryStyleTokenizer(nn.Module):
    """Global memory reader with non-exchangeable final Anima token slots.

    Fixed orthogonal slot bases keep the sixteen output roles distinct.  Each
    slot receives the same shared output map plus a small slot-specific low-rank
    delta.  Query self-attention is intentionally absent: references interact
    through the memory set, while output-slot identities cannot average into a
    single common query.
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
        local_layers: int = 1,
        cross_layers: int = 2,
        ff_dim: int = 2048,
        slot_rank: int = 32,
        slot_query_delta_scale: float = 0.10,
        output_rms_init: float = 0.15,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Style width must be divisible by attention heads")
        if min(
            spatial_tokens,
            global_tokens,
            output_tokens,
            local_layers,
            cross_layers,
            slot_rank,
        ) <= 0:
            raise ValueError("Token counts, depths, and slot rank must be positive")
        if artist_summary_tokens <= 0 or not include_artist_summary:
            raise ValueError("Artist-summary tokens must be present")
        self.dim = int(dim)
        self.source_dim = int(dim)
        self.spatial_tokens = int(spatial_tokens)
        self.global_tokens = int(global_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.include_artist_summary = True
        self.cached_tokens = (
            self.spatial_tokens + self.global_tokens + self.artist_summary_tokens
        )
        self.output_tokens = int(output_tokens)
        self.slot_rank = int(slot_rank)
        self.slot_query_delta_scale = float(slot_query_delta_scale)
        self.output_rms_init = float(output_rms_init)

        self.input_norm = nn.LayerNorm(dim)
        self.type_embedding = nn.Parameter(torch.empty(3, dim))
        self.spatial_position = nn.Parameter(torch.empty(spatial_tokens, dim))
        self.reference_register = nn.Parameter(torch.empty(1, 1, dim))
        self.local_blocks = nn.ModuleList(
            _PreNormMemoryBlock(dim, heads, ff_dim) for _ in range(local_layers)
        )
        self.register_buffer(
            "slot_query_basis", torch.empty(1, output_tokens, dim), persistent=True
        )
        self.slot_query_delta = nn.Parameter(torch.zeros(1, output_tokens, dim))
        self.query_blocks = nn.ModuleList(
            _SlotPreservingQueryBlock(dim, heads, ff_dim)
            for _ in range(cross_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.shared_output = nn.Linear(dim, dim)
        self.slot_down = nn.Parameter(torch.empty(output_tokens, dim, slot_rank))
        self.slot_up = nn.Parameter(torch.zeros(output_tokens, slot_rank, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        std = self.dim**-0.5
        nn.init.normal_(self.type_embedding, std=std)
        nn.init.normal_(self.spatial_position, std=std)
        nn.init.normal_(self.reference_register, std=std)
        basis = torch.randn(
            self.dim,
            self.output_tokens,
            device=self.slot_query_basis.device,
            dtype=self.slot_query_basis.dtype,
        )
        basis = torch.linalg.qr(basis, mode="reduced").Q.transpose(0, 1)
        self.slot_query_basis.copy_(basis.unsqueeze(0))
        nn.init.zeros_(self.slot_query_delta)
        nn.init.normal_(
            self.shared_output.weight,
            std=self.output_rms_init / (self.dim**0.5),
        )
        nn.init.zeros_(self.shared_output.bias)
        nn.init.normal_(self.slot_down, std=self.dim**-0.5)
        nn.init.zeros_(self.slot_up)

    def _typed_memory(self, references: torch.Tensor) -> torch.Tensor:
        spatial_end = self.spatial_tokens
        global_end = spatial_end + self.global_tokens
        spatial = (
            references[:, :, :spatial_end]
            + self.type_embedding[0]
            + self.spatial_position
        )
        global_values = references[:, :, spatial_end:global_end]
        global_values = global_values + self.type_embedding[1]
        artist = references[:, :, global_end:] + self.type_embedding[2]
        return torch.cat((spatial, global_values, artist), dim=2)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> GlobalQueryStyleTokenizerOutput:
        if reconstruct:
            raise ValueError(
                "Reconstruction belongs to the frozen Dual-query Resampler"
            )
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
        memory = self.input_norm(self._typed_memory(references))
        memory = memory.reshape(batch * reference_count, self.cached_tokens, self.dim)
        register = self.reference_register.expand(batch * reference_count, -1, -1)
        memory = torch.cat((register, memory), dim=1)
        for block in self.local_blocks:
            memory = block(memory)
        memory_per_reference = memory.reshape(
            batch, reference_count, self.cached_tokens + 1, self.dim
        )
        memory = memory_per_reference.flatten(1, 2)
        memory_padding_mask = (~reference_mask).unsqueeze(-1).expand(
            -1, -1, self.cached_tokens + 1
        ).reshape(batch, -1)

        queries = self.slot_query_basis.expand(batch, -1, -1)
        queries = queries + self.slot_query_delta_scale * self.slot_query_delta
        attentions = []
        for block in self.query_blocks:
            queries, attention = block(queries, memory, memory_padding_mask)
            attentions.append(attention)
        normalized = self.final_norm(queries)
        shared = self.shared_output(normalized)
        low_rank = torch.einsum("bsd,sdr->bsr", normalized, self.slot_down)
        low_rank = torch.einsum("bsr,srd->bsd", low_rank, self.slot_up)
        tokens = (shared + low_rank).to(dtype=references.dtype)
        return GlobalQueryStyleTokenizerOutput(
            tokens=tokens,
            per_reference_tokens=memory_per_reference,
            reconstruction=None,
            reconstruction_target=None,
            artist_tokens=tokens,
            diversity_tokens=tokens,
            attention_maps=torch.stack(attentions, dim=1),
            reference_conditioned_tokens=tokens,
        )


def attention_map_diversity_loss(attention_maps: torch.Tensor) -> torch.Tensor:
    """Discourage different output queries from reading identical memory."""

    # [B,layers,heads,queries,memory] -> average heads and layers without
    # penalizing head specialization inside one query.
    maps = attention_maps.float().mean(dim=(1, 2))
    maps = F.normalize(maps, dim=-1)
    similarities = maps @ maps.transpose(1, 2)
    identity = torch.eye(maps.shape[1], device=maps.device, dtype=torch.bool)
    return similarities[:, ~identity].square().mean()


def reference_conditioned_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Apply diversity only to batch-varying, reference-conditioned output."""

    if tokens.shape[0] < 2:
        return tokens.float().sum() * 0.0
    centered = tokens.float() - tokens.float().mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, dim=-1)
    similarities = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool)
    return similarities[:, ~identity].square().mean()


class MultiPromptDualQueryCachedStyleLoader(DualQueryCachedStyleLoader):
    """Select cached prompt modes while keeping Empty strictly exact-self."""

    def __init__(self, destination: Path, cfg: dict[str, Any]) -> None:
        super().__init__(destination, cfg)
        prompt_cfg = dict(cfg.get("prompt_modes", {}))
        self.prompt_mode_weights = {
            "full": float(prompt_cfg.get("full", 0.30)),
            "tag_dropout": float(prompt_cfg.get("tag_dropout", 0.40)),
            "short": float(prompt_cfg.get("short", 0.20)),
            "empty": float(prompt_cfg.get("empty", 0.10)),
        }
        if any(value < 0 for value in self.prompt_mode_weights.values()):
            raise ValueError("Prompt mode weights cannot be negative")
        total = sum(self.prompt_mode_weights.values())
        if total <= 0:
            raise ValueError("Prompt mode distribution is empty")
        self.prompt_mode_weights = {
            key: value / total for key, value in self.prompt_mode_weights.items()
        }
        self.quality_probability = float(prompt_cfg.get("quality_probability", 0.50))
        if not 0 <= self.quality_probability <= 1:
            raise ValueError("quality_probability must be in [0,1]")
        self.variant_by_image_and_name = {
            (int(row["id"]), str(row.get("variant_name", row["variant"]))): int(
                row["variant"]
            )
            for row in self.text_by_key.values()
        }
        required = {
            "full", "full_quality", "tag_dropout", "tag_dropout_quality",
            "short", "short_quality",
        }
        available = {
            name for (_, name) in self.variant_by_image_and_name
        }
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(
                f"Multi-prompt text cache is missing variants {missing}"
            )
        self.empty_condition = load_file(
            self.text_root / "null_conditioning.safetensors", device="cpu"
        )["empty_prompt"]

    def episodes_for_step(self, step: int) -> list[StyleEpisode]:
        episodes = super().episodes_for_step(step)
        rng = random.Random(self.seed ^ 0x6D75_1A0D ^ (int(step) * 1_000_003))
        modes = tuple(self.prompt_mode_weights)
        weights = tuple(self.prompt_mode_weights[value] for value in modes)
        result = []
        for episode in episodes:
            mode = rng.choices(modes, weights=weights, k=1)[0]
            references = list(episode.reference_ids)
            rng.shuffle(references)
            if mode == "empty":
                result.append(
                    StyleEpisode(
                        episode.target_id,
                        (episode.target_id,),
                        episode.style_id,
                        episode.latent_shape,
                        -1,
                    )
                )
                continue
            quality = rng.random() < self.quality_probability
            variant_name = mode + ("_quality" if quality else "")
            variant = self.variant_by_image_and_name.get(
                (episode.target_id, variant_name)
            )
            if variant is None:
                raise RuntimeError(
                    f"Target {episode.target_id} lacks text variant {variant_name}"
                )
            result.append(
                StyleEpisode(
                    episode.target_id,
                    tuple(references),
                    episode.style_id,
                    episode.latent_shape,
                    variant,
                )
            )
        return result

    def _load_episode_condition(
        self, item: StyleEpisode
    ) -> tuple[torch.Tensor, int]:
        if item.text_variant == -1:
            return self.empty_condition, int(self.empty_condition.shape[0])
        return super()._load_episode_condition(item)

    def _episode_prompt_mode(self, item: StyleEpisode) -> str:
        if item.text_variant == -1:
            return "empty"
        return super()._episode_prompt_mode(item)
