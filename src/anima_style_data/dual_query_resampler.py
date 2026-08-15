from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def normalized_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return row-major image coordinates normalized independently to [-1, 1]."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def padded_grid_coordinates(
    shapes: torch.Tensor,
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build continuous 2D coordinates and a validity mask for padded grids."""
    coordinates = torch.zeros(
        (int(shapes.shape[0]), length, 2), device=device, dtype=dtype
    )
    valid = torch.zeros((int(shapes.shape[0]), length), device=device, dtype=torch.bool)
    for index, shape in enumerate(shapes.detach().cpu().tolist()):
        height, width = (int(shape[0]), int(shape[1]))
        tokens = height * width
        if tokens > length:
            raise ValueError(f"Grid {height}x{width} exceeds padded length {length}")
        coordinates[index, :tokens] = normalized_grid(
            height, width, device=device, dtype=dtype
        )
        valid[index, :tokens] = True
    return coordinates, valid


def _rotate_axis(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    axis_dim = int(values.shape[-1])
    if axis_dim % 2:
        raise ValueError("Each 2D RoPE axis needs an even channel count")
    frequencies = torch.arange(
        0, axis_dim, 2, device=values.device, dtype=torch.float32
    )
    frequencies = torch.exp(-math.log(10_000.0) * frequencies / axis_dim)
    angles = coordinates.float().unsqueeze(1).unsqueeze(-1) * math.pi * frequencies
    cosine = angles.cos().to(values.dtype)
    sine = angles.sin().to(values.dtype)
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def apply_2d_rope(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Apply continuous 2D RoPE to [batch, heads, tokens, head_dim]."""
    head_dim = int(values.shape[-1])
    rotary_dim = head_dim - (head_dim % 4)
    if rotary_dim < 4:
        return values
    axis_dim = rotary_dim // 2
    x = _rotate_axis(values[..., :axis_dim], coordinates[..., 0])
    y = _rotate_axis(values[..., axis_dim:rotary_dim], coordinates[..., 1])
    return torch.cat((x, y, values[..., rotary_dim:]), dim=-1)


class MultiheadCrossAttention(nn.Module):
    """Cross attention with independent projections and continuous 2D RoPE."""

    def __init__(self, dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        query_coordinates: torch.Tensor | None = None,
        context_coordinates: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(queries))
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))
        if query_coordinates is not None:
            q = apply_2d_rope(q, query_coordinates)
        if context_coordinates is not None:
            k = apply_2d_rope(k, context_coordinates)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        if context_mask is not None:
            if not bool(context_mask.any(dim=1).all()):
                raise ValueError("Every sample must contain at least one valid context token")
            scores.masked_fill_(~context_mask[:, None, None, :], -torch.inf)
        attention = scores.softmax(dim=-1).to(v.dtype)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(queries.shape)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim * 2)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.up(value).chunk(2, dim=-1)
        return self.down(F.silu(gate) * content)


class ConvResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden_dim = min(max(dim // 4, 16), 256)
        groups = min(32, dim)
        while dim % groups:
            groups -= 1
        hidden_groups = min(32, hidden_dim)
        while hidden_dim % hidden_groups:
            hidden_groups -= 1
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.norm2 = nn.GroupNorm(hidden_groups, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(value)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        residual = self.conv3(F.silu(residual))
        return value + residual


class DualQueryBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.semantic_query_norm = nn.LayerNorm(dim)
        self.semantic_context_norm = nn.LayerNorm(dim)
        self.semantic_attention = MultiheadCrossAttention(dim, dim, heads)
        self.vae_query_norm = nn.LayerNorm(dim)
        self.vae_context_norm = nn.LayerNorm(dim)
        self.vae_attention = MultiheadCrossAttention(dim, dim, heads)
        self.semantic_gate = nn.Parameter(torch.full((dim,), 0.5))
        self.vae_gate = nn.Parameter(torch.full((dim,), 0.5))
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = SwiGLU(dim, ff_dim)

    def forward(
        self,
        queries: torch.Tensor,
        semantic_context: torch.Tensor,
        vae_context: torch.Tensor,
        *,
        query_coordinates: torch.Tensor,
        semantic_coordinates: torch.Tensor,
        vae_coordinates: torch.Tensor,
        semantic_mask: torch.Tensor,
        vae_mask: torch.Tensor,
        semantic_keep: torch.Tensor,
        vae_keep: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        semantic = self.semantic_attention(
            self.semantic_query_norm(queries),
            self.semantic_context_norm(semantic_context),
            query_coordinates=query_coordinates,
            context_coordinates=semantic_coordinates,
            context_mask=semantic_mask,
        )
        queries = queries + semantic_keep * self.semantic_gate * semantic
        perceptual = self.vae_attention(
            self.vae_query_norm(queries),
            self.vae_context_norm(vae_context),
            query_coordinates=query_coordinates,
            context_coordinates=vae_coordinates,
            context_mask=vae_mask,
        )
        queries = queries + vae_keep * self.vae_gate * perceptual
        return queries + self.feed_forward(self.ff_norm(queries))


class ArtistPoolingBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = MultiheadCrossAttention(dim, dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = SwiGLU(dim, ff_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        queries = queries + self.cross_attention(
            self.query_norm(queries), self.context_norm(context)
        )
        return queries + self.feed_forward(self.ff_norm(queries))


class ArtistDescriptorHead(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        pooling_queries: int = 4,
        layers: int = 2,
        descriptor_dim: int = 512,
        summary_tokens: int = 4,
        ff_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling_queries = nn.Parameter(torch.randn(pooling_queries, dim) * 0.02)
        # Keep this head materially smaller than the Q-Former: it pools rather
        # than learning a second full token encoder.
        pooling_ff_dim = min(ff_dim or dim, dim)
        self.pooler = nn.ModuleList(
            [ArtistPoolingBlock(dim, heads, pooling_ff_dim) for _ in range(layers)]
        )
        self.pooling_norm = nn.LayerNorm(dim)
        hidden = min(512, dim)
        self.descriptor_pool = nn.Linear(dim, hidden)
        self.descriptor_projection = nn.Linear(pooling_queries * hidden, descriptor_dim)
        self.summary_mixer = nn.Parameter(torch.zeros(summary_tokens, pooling_queries))
        with torch.no_grad():
            self.summary_mixer.fill_(-4.0)
            for index in range(min(summary_tokens, pooling_queries)):
                self.summary_mixer[index, index] = 4.0
        self.summary_projection = nn.Linear(dim, dim)
        self.summary_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(tokens.shape[0])
        pooling = self.pooling_queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.pooler:
            pooling = block(pooling, tokens)
        pooling = self.pooling_norm(pooling)
        descriptor_hidden = self.descriptor_pool(pooling).flatten(1)
        descriptor = F.normalize(
            self.descriptor_projection(descriptor_hidden).float(), dim=-1
        )
        summary_weights = self.summary_mixer.softmax(dim=-1)
        summary = torch.einsum("sp,bpd->bsd", summary_weights, pooling)
        return descriptor, self.summary_norm(self.summary_projection(summary))


@dataclass
class PerReferenceOutput:
    tokens: torch.Tensor
    descriptor: torch.Tensor
    artist_summary: torch.Tensor
    semantic_reconstruction: dict[int, torch.Tensor] | None = None
    vae_reconstruction: torch.Tensor | None = None


class DualQueryResampler(nn.Module):
    """B-prime Q-Former over separate C-RADIO and Qwen VAE feature banks."""

    def __init__(
        self,
        *,
        semantic_layers: Sequence[int] = (18, 24),
        semantic_dim: int = 1152,
        vae_channels: int = 16,
        dim: int = 1024,
        spatial_query_grid: int = 8,
        global_queries: int = 16,
        layers: int = 4,
        heads: int = 16,
        ff_dim: int = 4096,
        artist_descriptor_dim: int = 512,
        artist_pooling_queries: int = 4,
        artist_summary_tokens: int = 4,
        artist_classes: int = 0,
        semantic_dropout: float = 0.05,
        vae_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.semantic_layers = tuple(int(layer) for layer in semantic_layers)
        self.dim = dim
        self.spatial_query_grid = spatial_query_grid
        self.spatial_queries = spatial_query_grid**2
        self.global_queries = global_queries
        self.query_count = self.spatial_queries + global_queries
        self.semantic_dropout = semantic_dropout
        self.vae_dropout = vae_dropout

        self.semantic_norms = nn.ModuleDict(
            {str(layer): nn.LayerNorm(semantic_dim) for layer in self.semantic_layers}
        )
        self.semantic_projections = nn.ModuleDict(
            {str(layer): nn.Linear(semantic_dim, dim) for layer in self.semantic_layers}
        )
        self.semantic_type_embeddings = nn.Parameter(
            torch.randn(len(self.semantic_layers), dim) * 0.02
        )
        self.vae_stem = nn.Sequential(
            nn.Conv2d(vae_channels, dim, kernel_size=2, stride=2),
            ConvResidualBlock(dim),
            ConvResidualBlock(dim),
        )
        self.spatial_query_tokens = nn.Parameter(
            torch.randn(self.spatial_queries, dim) * 0.02
        )
        self.global_query_tokens = nn.Parameter(torch.randn(global_queries, dim) * 0.02)
        self.query_type_embeddings = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.geometry_embedding = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            [DualQueryBlock(dim, heads, ff_dim) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.artist_head = ArtistDescriptorHead(
            dim,
            heads,
            pooling_queries=artist_pooling_queries,
            descriptor_dim=artist_descriptor_dim,
            summary_tokens=artist_summary_tokens,
            ff_dim=ff_dim,
        )
        self.register_buffer(
            "artist_proxies",
            F.normalize(torch.randn(artist_classes, artist_descriptor_dim), dim=-1)
            if artist_classes > 0
            else None,
            persistent=True,
        )
        self.semantic_decoder_norm = nn.LayerNorm(dim)
        self.semantic_decoder_heads = nn.ModuleDict(
            {
                str(layer): nn.Conv2d(dim, semantic_dim, kernel_size=1)
                for layer in self.semantic_layers
            }
        )
        self.vae_decoder_norm = nn.LayerNorm(dim)
        self.vae_decoder_head = nn.Sequential(
            ConvResidualBlock(dim), nn.Conv2d(dim, vae_channels, kernel_size=1)
        )
        self.global_decoder_bias = nn.Linear(dim, dim)

    def artist_proxy_loss(
        self,
        descriptors: torch.Tensor,
        labels: torch.Tensor,
        *,
        scale: float = 16.0,
        margin: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-only CosFace proxy loss that breaks episodic collapse."""
        if self.artist_proxies is None:
            raise RuntimeError("artist_classes must be positive to use proxy supervision")
        descriptors = F.normalize(descriptors.float(), dim=-1)
        proxies = F.normalize(self.artist_proxies.float(), dim=-1)
        cosine = torch.matmul(descriptors, proxies.T)
        target_margin = F.one_hot(
            labels, num_classes=int(proxies.shape[0])
        ).to(cosine.dtype)
        logits = (cosine - margin * target_margin) * scale
        loss = F.cross_entropy(logits, labels)
        top1 = (cosine.argmax(dim=1) == labels).float().mean()
        return loss, top1

    def _semantic_bank(
        self,
        features: Mapping[int, torch.Tensor],
        mask: torch.Tensor,
        grid_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = []
        for index, layer in enumerate(self.semantic_layers):
            value = features[layer]
            value = self.semantic_projections[str(layer)](
                self.semantic_norms[str(layer)](value)
            )
            projected.append(value + self.semantic_type_embeddings[index])
        coordinates, shape_mask = padded_grid_coordinates(
            grid_shapes,
            int(mask.shape[1]),
            device=mask.device,
            dtype=projected[0].dtype,
        )
        valid = mask & shape_mask
        return (
            torch.cat(projected, dim=1),
            torch.cat([valid] * len(projected), dim=1),
            torch.cat([coordinates] * len(projected), dim=1),
        )

    def _vae_bank(
        self, latents: torch.Tensor, vae_shapes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.vae_stem(latents)
        batch, channels, height, width = feature_map.shape
        context = feature_map.flatten(2).transpose(1, 2)
        downsampled_shapes = torch.div(vae_shapes, 2, rounding_mode="floor").clamp_min(1)
        coordinates = torch.zeros(
            (batch, height * width, 2), device=latents.device, dtype=context.dtype
        )
        valid = torch.zeros(
            (batch, height * width), device=latents.device, dtype=torch.bool
        )
        for index, shape in enumerate(downsampled_shapes.detach().cpu().tolist()):
            item_height, item_width = int(shape[0]), int(shape[1])
            item_grid = normalized_grid(
                item_height, item_width, device=latents.device, dtype=context.dtype
            ).view(item_height, item_width, 2)
            grid = coordinates[index].view(height, width, 2)
            grid[:item_height, :item_width] = item_grid
            valid[index].view(height, width)[:item_height, :item_width] = True
        return context, valid, coordinates

    def _query_bank(
        self, image_sizes: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(image_sizes.shape[0])
        spatial = self.spatial_query_tokens + self.query_type_embeddings[0]
        global_tokens = self.global_query_tokens + self.query_type_embeddings[1]
        queries = torch.cat((spatial, global_tokens), dim=0).unsqueeze(0).expand(
            batch, -1, -1
        )
        sizes = image_sizes.to(dtype=torch.float32)
        log_aspect = torch.log(sizes[:, 1].clamp_min(1) / sizes[:, 0].clamp_min(1))
        log_scale = torch.log(torch.sqrt(sizes.prod(dim=1)).clamp_min(1) / 512.0)
        geometry = self.geometry_embedding(torch.stack((log_aspect, log_scale), dim=-1))
        queries = queries + geometry.to(dtype).unsqueeze(1)
        spatial_coordinates = normalized_grid(
            self.spatial_query_grid,
            self.spatial_query_grid,
            device=image_sizes.device,
            dtype=dtype,
        )
        coordinates = torch.zeros(
            (batch, self.query_count, 2), device=image_sizes.device, dtype=dtype
        )
        coordinates[:, : self.spatial_queries] = spatial_coordinates
        return queries, coordinates

    def _modality_keep(
        self,
        batch: int,
        dropout: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        override: torch.Tensor | None,
    ) -> torch.Tensor:
        if override is not None:
            keep = override.to(device=device, dtype=dtype)
        elif self.training and dropout > 0:
            keep = (torch.rand(batch, device=device) >= dropout).to(dtype)
        else:
            keep = torch.ones(batch, device=device, dtype=dtype)
        return keep.view(batch, 1, 1)

    def encode(
        self,
        semantic_features: Mapping[int, torch.Tensor],
        semantic_mask: torch.Tensor,
        semantic_grid_shapes: torch.Tensor,
        vae_latents: torch.Tensor,
        vae_shapes: torch.Tensor,
        image_sizes: torch.Tensor,
        *,
        semantic_keep: torch.Tensor | None = None,
        vae_keep: torch.Tensor | None = None,
        reconstruct: bool = False,
    ) -> PerReferenceOutput:
        semantic_context, semantic_valid, semantic_coordinates = self._semantic_bank(
            semantic_features, semantic_mask, semantic_grid_shapes
        )
        vae_context, vae_valid, vae_coordinates = self._vae_bank(
            vae_latents, vae_shapes
        )
        queries, query_coordinates = self._query_bank(
            image_sizes, dtype=semantic_context.dtype
        )
        batch = int(queries.shape[0])
        semantic_factor = self._modality_keep(
            batch,
            self.semantic_dropout,
            device=queries.device,
            dtype=queries.dtype,
            override=semantic_keep,
        )
        vae_factor = self._modality_keep(
            batch,
            self.vae_dropout,
            device=queries.device,
            dtype=queries.dtype,
            override=vae_keep,
        )
        # Never erase both observations for an item during per-reference training.
        both_missing = (semantic_factor == 0) & (vae_factor == 0)
        vae_factor = torch.where(both_missing, torch.ones_like(vae_factor), vae_factor)
        for block in self.blocks:
            queries = block(
                queries,
                semantic_context,
                vae_context,
                query_coordinates=query_coordinates,
                semantic_coordinates=semantic_coordinates,
                vae_coordinates=vae_coordinates,
                semantic_mask=semantic_valid,
                vae_mask=vae_valid,
                semantic_keep=semantic_factor,
                vae_keep=vae_factor,
            )
        tokens = self.output_norm(queries)
        descriptor, artist_summary = self.artist_head(tokens)
        semantic_reconstruction = None
        vae_reconstruction = None
        if reconstruct:
            semantic_reconstruction = self.decode_semantic(tokens, semantic_grid_shapes)
            vae_reconstruction = self.decode_vae(tokens, vae_latents.shape[-2:])
        return PerReferenceOutput(
            tokens=tokens,
            descriptor=descriptor,
            artist_summary=artist_summary,
            semantic_reconstruction=semantic_reconstruction,
            vae_reconstruction=vae_reconstruction,
        )

    def _decoder_map(self, tokens: torch.Tensor) -> torch.Tensor:
        spatial = self.semantic_decoder_norm(tokens[:, : self.spatial_queries])
        spatial = spatial.transpose(1, 2).reshape(
            tokens.shape[0], self.dim, self.spatial_query_grid, self.spatial_query_grid
        )
        global_bias = self.global_decoder_bias(
            tokens[:, self.spatial_queries :].mean(dim=1)
        )
        return spatial + global_bias[:, :, None, None]

    def decode_semantic(
        self, tokens: torch.Tensor, grid_shapes: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        source = self._decoder_map(tokens)
        max_tokens = int((grid_shapes[:, 0] * grid_shapes[:, 1]).max().item())
        outputs = {
            layer: source.new_zeros(
                (tokens.shape[0], max_tokens, head.out_channels)
            )
            for layer, head in (
                (layer, self.semantic_decoder_heads[str(layer)])
                for layer in self.semantic_layers
            )
        }
        for item, shape in enumerate(grid_shapes.detach().cpu().tolist()):
            height, width = int(shape[0]), int(shape[1])
            resized = F.interpolate(
                source[item : item + 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            for layer in self.semantic_layers:
                decoded = self.semantic_decoder_heads[str(layer)](resized)
                outputs[layer][item, : height * width] = decoded.flatten(2).transpose(1, 2)[0]
        return outputs

    def decode_vae(
        self, tokens: torch.Tensor, output_shape: Sequence[int]
    ) -> torch.Tensor:
        source = self.vae_decoder_norm(tokens[:, : self.spatial_queries])
        source = source.transpose(1, 2).reshape(
            tokens.shape[0], self.dim, self.spatial_query_grid, self.spatial_query_grid
        )
        source = F.interpolate(
            source, size=tuple(output_shape), mode="bilinear", align_corners=False
        )
        return self.vae_decoder_head(source)


class MultiReferenceSetTransformer(nn.Module):
    """Permutation-invariant reference set aggregation into Anima context tokens."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        output_tokens: int = 32,
        heads: int = 16,
        cross_layers: int = 1,
        cross_slot_layers: int = 2,
        ff_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.output_queries = nn.Parameter(torch.randn(output_tokens, dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            dim,
            heads,
            ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_decoder = nn.TransformerDecoder(
            decoder_layer, cross_layers, norm=nn.LayerNorm(dim)
        )
        slot_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_slot = nn.TransformerEncoder(
            slot_layer, cross_slot_layers, norm=nn.LayerNorm(dim)
        )

    def forward(
        self,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        artist_summary: torch.Tensor | None = None,
        include_artist_summary: bool = True,
    ) -> torch.Tensor:
        if reference_tokens.ndim != 4:
            raise ValueError("reference_tokens must be [batch, references, tokens, dim]")
        batch, references, tokens, dim = reference_tokens.shape
        pieces = [reference_tokens]
        tokens_per_reference = tokens
        if include_artist_summary and artist_summary is not None:
            if artist_summary.shape[:2] != (batch, references):
                raise ValueError("artist_summary must share batch/reference dimensions")
            pieces.append(artist_summary)
            tokens_per_reference += int(artist_summary.shape[2])
        context = torch.cat(pieces, dim=2).reshape(
            batch, references * tokens_per_reference, dim
        )
        context_mask = reference_mask[:, :, None].expand(
            -1, -1, tokens_per_reference
        ).reshape(batch, -1)
        if not bool(context_mask.any(dim=1).all()):
            raise ValueError("Each item needs at least one reference")
        queries = self.output_queries.unsqueeze(0).expand(batch, -1, -1)
        queries = self.set_decoder(
            queries, context, memory_key_padding_mask=~context_mask
        )
        return self.cross_slot(queries)


@dataclass
class DualQueryStyleOutput:
    style_tokens: torch.Tensor
    reference_tokens: torch.Tensor
    descriptors: torch.Tensor
    artist_summaries: torch.Tensor


class DualQueryStyleEncoder(nn.Module):
    def __init__(
        self,
        resampler: DualQueryResampler,
        aggregator: MultiReferenceSetTransformer,
        *,
        include_artist_summary: bool = True,
    ) -> None:
        super().__init__()
        self.resampler = resampler
        self.aggregator = aggregator
        self.include_artist_summary = include_artist_summary

    def forward(
        self,
        semantic_features: Mapping[int, torch.Tensor],
        semantic_mask: torch.Tensor,
        semantic_grid_shapes: torch.Tensor,
        vae_latents: torch.Tensor,
        vae_shapes: torch.Tensor,
        image_sizes: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> DualQueryStyleOutput:
        batch, references = reference_mask.shape
        flat_features = {
            layer: value.flatten(0, 1) for layer, value in semantic_features.items()
        }
        encoded = self.resampler.encode(
            flat_features,
            semantic_mask.flatten(0, 1),
            semantic_grid_shapes.flatten(0, 1),
            vae_latents.flatten(0, 1),
            vae_shapes.flatten(0, 1),
            image_sizes.flatten(0, 1),
        )
        reference_tokens = encoded.tokens.reshape(batch, references, *encoded.tokens.shape[1:])
        descriptors = encoded.descriptor.reshape(batch, references, -1)
        summaries = encoded.artist_summary.reshape(
            batch, references, *encoded.artist_summary.shape[1:]
        )
        style_tokens = self.aggregator(
            reference_tokens,
            reference_mask,
            artist_summary=summaries,
            include_artist_summary=self.include_artist_summary,
        )
        return DualQueryStyleOutput(
            style_tokens=style_tokens,
            reference_tokens=reference_tokens,
            descriptors=descriptors,
            artist_summaries=summaries,
        )


def episodic_angular_prototype_loss(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale: float = 16.0,
    margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Leave-one-out angular prototypical classification within a batch."""
    descriptors = F.normalize(descriptors.float(), dim=-1)
    classes = torch.unique(labels, sorted=True)
    if len(classes) < 2:
        raise ValueError("Prototype loss needs at least two artists")
    counts = torch.stack([(labels == value).sum() for value in classes])
    if bool((counts < 2).any()):
        raise ValueError("Prototype loss needs at least two images per artist")
    class_sums = torch.stack(
        [descriptors[labels == value].sum(dim=0) for value in classes]
    )
    logits = []
    targets = []
    positive_values = []
    negative_values = []
    for index, (descriptor, label) in enumerate(zip(descriptors, labels)):
        class_index = int(torch.nonzero(classes == label, as_tuple=False)[0, 0])
        prototypes = class_sums.clone()
        prototypes[class_index] -= descriptor
        divisor = counts.to(descriptors.dtype).unsqueeze(1)
        divisor[class_index] -= 1
        prototypes = F.normalize(prototypes / divisor, dim=-1)
        cosine = torch.matmul(prototypes, descriptor).clamp(-1 + 1e-6, 1 - 1e-6)
        positive = cosine[class_index]
        # The equivalent acos formulation has an unbounded derivative near
        # +/-1. Early in training the descriptor head can emit nearly identical
        # vectors, so use the stable ArcFace identity and clamp its sine term.
        sine = torch.sqrt((1.0 - positive.square()).clamp_min(1e-4))
        angular_positive = (
            positive * math.cos(margin) - sine * math.sin(margin)
        )
        cosine = cosine.clone()
        cosine[class_index] = angular_positive
        logits.append(cosine * scale)
        targets.append(class_index)
        positive_values.append(positive)
        negative_values.append(
            torch.cat((cosine[:class_index], cosine[class_index + 1 :])).max()
        )
    logits_tensor = torch.stack(logits)
    targets_tensor = torch.tensor(targets, device=labels.device, dtype=torch.long)
    loss = F.cross_entropy(logits_tensor, targets_tensor)
    metrics = {
        "prototype_top1": (logits_tensor.argmax(dim=1) == targets_tensor).float().mean(),
        "prototype_positive_cosine": torch.stack(positive_values).mean(),
        "prototype_hard_negative_cosine": torch.stack(negative_values).mean(),
    }
    return loss, metrics


def supervised_contrastive_loss(
    descriptors: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.10
) -> torch.Tensor:
    descriptors = F.normalize(descriptors.float(), dim=-1)
    similarities = torch.matmul(descriptors, descriptors.T) / temperature
    identity = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    if not bool(positives.any(dim=1).all()):
        raise ValueError("Supervised contrastive loss needs a positive for every item")
    similarities = similarities.masked_fill(identity, -torch.inf)
    log_probabilities = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    return -(
        log_probabilities.masked_fill(~positives, 0).sum(dim=1)
        / positives.sum(dim=1)
    ).mean()


def token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(tokens.float(), dim=-1)
    gram = torch.matmul(normalized, normalized.transpose(-2, -1))
    identity = torch.eye(tokens.shape[-2], device=tokens.device, dtype=torch.bool)
    return gram.masked_select(~identity).square().mean()
