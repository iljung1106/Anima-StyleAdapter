from __future__ import annotations

import copy
import json
import gc
import math
import os
import random
import shutil
import sys
import threading
import time
import types
import weakref
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from PIL import Image, ImageDraw, ImageOps

from .io import read_records, write_json
from .tap_resampler import build_tap_resampler_model


@dataclass(frozen=True)
class StyleEpisode:
    target_id: int
    reference_ids: tuple[int, ...]
    style_id: str
    latent_shape: tuple[int, int]
    text_variant: int


class _TensorShardCache:
    """Small LRU for the moderately sized packed text/latent shards."""

    def __init__(self, root: Path, max_shards: int = 2):
        self.root = root
        self.max_shards = max(1, max_shards)
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str) -> dict[str, torch.Tensor]:
        with self._lock:
            tensors = self._cache.pop(name, None)
            if tensors is None:
                tensors = load_file(self.root / name, device="cpu")
            self._cache[name] = tensors
            while len(self._cache) > self.max_shards:
                self._cache.popitem(last=False)
            return tensors


def _pad_text_conditions(
    conditions: list[torch.Tensor], conditioning_length: int
) -> torch.Tensor:
    if any(value.shape[0] > conditioning_length for value in conditions):
        longest = max(value.shape[0] for value in conditions)
        raise ValueError(
            f"Cached text condition length {longest} exceeds configured length {conditioning_length}"
        )
    batch = torch.zeros(
        len(conditions),
        conditioning_length,
        conditions[0].shape[-1],
        dtype=conditions[0].dtype,
    )
    for index, value in enumerate(conditions):
        batch[index, : value.shape[0]] = value
    return batch


class ProductionStyleLoader:
    """Deterministic same-style target/reference episodes over frozen caches.

    Targets in a batch share a latent bucket, so no latent padding is required.
    Reference count, target role, and prompt variant rotate deterministically by
    step. Style shards are opened once per batch and only requested tensors are
    paged in; large C-RADIO shards are never loaded wholesale.
    """

    def __init__(self, destination: Path, cfg: dict[str, Any]):
        self.destination = destination
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 20260811))
        self.batch_size = int(cfg.get("batch_size", 1))
        self.min_references = int(cfg.get("min_references", 1))
        self.max_references = int(cfg.get("max_references", 8))
        self.split = str(cfg.get("split", "train"))
        # Anima was trained with fixed 512-token post-LLM conditioning. Its
        # cross-attention does not receive a text padding mask, so the trailing
        # zero embeddings are part of the learned softmax normalization and
        # must not be trimmed at runtime.
        self.text_conditioning_length = int(cfg.get("text_conditioning_length", 512))

        style_root = destination / str(cfg["style_cache"])
        text_root = destination / str(cfg["text_cache"])
        latent_root = destination / str(cfg["latent_cache"])
        self.style_root = style_root
        self.text_root = text_root
        self.latent_root = latent_root
        style_rows = read_records(style_root / "manifest.parquet")
        text_rows = read_records(text_root / "manifest.parquet")
        latent_rows = read_records(latent_root / "manifest.parquet")

        self.style_by_id = {int(row["id"]): row for row in style_rows}
        self.latent_by_id = {int(row["id"]): row for row in latent_rows}
        self.text_by_key = {
            (int(row["id"]), int(row["variant"])): row for row in text_rows
        }
        text_variants: dict[int, list[int]] = defaultdict(list)
        for image_id, variant in self.text_by_key:
            text_variants[image_id].append(variant)
        self.text_variants = {key: sorted(value) for key, value in text_variants.items()}

        common = set(self.style_by_id) & set(self.latent_by_id) & set(self.text_variants)
        by_style: dict[str, list[int]] = defaultdict(list)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for image_id in sorted(common):
            style_row = self.style_by_id[image_id]
            latent_row = self.latent_by_id[image_id]
            if str(style_row.get("split", "train")) != self.split:
                continue
            style_id = str(style_row.get("style_id", style_row["artist"]))
            by_style[style_id].append(image_id)
            shape = (int(latent_row["latent_height"]), int(latent_row["latent_width"]))
            buckets[shape].append(image_id)
        self.by_style = {key: value for key, value in by_style.items() if len(value) >= 2}
        valid_ids = {image_id for values in self.by_style.values() for image_id in values}
        # A training batch must contain distinct targets with an identical
        # latent shape. Rare aspect-ratio buckets can survive the cache
        # intersection with fewer rows than the configured batch size; letting
        # one of those enter weighted sampling makes a long run fail only when
        # that rare bucket is eventually drawn. References are allowed to come
        # from other buckets, so only the target bucket needs this size filter.
        self.buckets = {}
        for shape, values in buckets.items():
            eligible = [image_id for image_id in values if image_id in valid_ids]
            if len(eligible) >= self.batch_size:
                self.buckets[shape] = eligible
        if not self.by_style or not self.buckets:
            raise RuntimeError("No eligible same-style episodes exist in the cache intersection")
        self.bucket_keys = sorted(self.buckets)
        self.bucket_weights = [len(self.buckets[key]) for key in self.bucket_keys]
        self.text_shards = _TensorShardCache(text_root, int(cfg.get("text_lru_shards", 2)))
        self.latent_shards = _TensorShardCache(latent_root, int(cfg.get("latent_lru_shards", 2)))

    def episodes_for_step(self, step: int) -> list[StyleEpisode]:
        rng = random.Random(self.seed + step * 1_000_003)
        # Sampling bucket names uniformly would drastically overrepresent rare
        # extreme aspect ratios. Weight by eligible target count so each image
        # retains approximately equal target probability while batches remain
        # exact-shape.
        shape = rng.choices(self.bucket_keys, weights=self.bucket_weights, k=1)[0]
        candidates = self.buckets[shape]
        chosen: list[int] = []
        attempts = 0
        while len(chosen) < self.batch_size and attempts < max(64, self.batch_size * 32):
            target_id = candidates[rng.randrange(len(candidates))]
            style_row = self.style_by_id[target_id]
            style_id = str(style_row.get("style_id", style_row["artist"]))
            if target_id not in chosen and style_id not in {
                str(self.style_by_id[item].get("style_id", self.style_by_id[item]["artist"]))
                for item in chosen
            }:
                chosen.append(target_id)
            attempts += 1
        if len(chosen) < self.batch_size:
            # Same-style targets are still valid; distinct styles per batch are
            # an efficiency preference, not part of the data contract.
            remaining = [item for item in candidates if item not in chosen]
            rng.shuffle(remaining)
            chosen.extend(remaining[: self.batch_size - len(chosen)])
        if len(chosen) != self.batch_size:
            raise RuntimeError(f"Bucket {shape} cannot provide batch_size={self.batch_size}")

        episodes = []
        for target_id in chosen:
            row = self.style_by_id[target_id]
            style_id = str(row.get("style_id", row["artist"]))
            pool = [item for item in self.by_style[style_id] if item != target_id]
            count = rng.randint(self.min_references, min(self.max_references, len(pool)))
            references = tuple(rng.sample(pool, count))
            variants = self.text_variants[target_id]
            episodes.append(
                StyleEpisode(target_id, references, style_id, shape, variants[step % len(variants)])
            )
        return episodes

    def load_step(self, step: int) -> dict[str, Any]:
        episodes = self.episodes_for_step(step)
        latent_rows = [self.latent_by_id[item.target_id] for item in episodes]
        latents = []
        for row in latent_rows:
            shard = self.latent_shards.get(str(row["cache_shard"]))
            latents.append(shard["latents"][int(row["row_index"])])
        latent_batch = torch.stack(latents)

        conditions = []
        for item in episodes:
            row = self.text_by_key[(item.target_id, item.text_variant)]
            shard = self.text_shards.get(str(row["cache_shard"]))
            start = int(row["token_offset"])
            conditions.append(shard["conditioning"][start : start + int(row["token_length"])])
        condition_batch = _pad_text_conditions(conditions, self.text_conditioning_length)

        flat_references = [image_id for item in episodes for image_id in item.reference_ids]
        target_ids = [item.target_id for item in episodes]
        max_refs = max(len(item.reference_ids) for item in episodes)
        reference_mask = torch.zeros(len(episodes), max_refs, dtype=torch.bool)
        cursor = 0
        reference_positions = []
        for batch_index, item in enumerate(episodes):
            reference_mask[batch_index, : len(item.reference_ids)] = True
            for ref_index, _ in enumerate(item.reference_ids):
                reference_positions.append((batch_index, ref_index))
                cursor += 1

        feature_values: dict[int, dict[int, torch.Tensor]] = {}
        global_values: dict[int, torch.Tensor] = {}
        grouped: dict[str, list[int]] = defaultdict(list)
        for image_id in dict.fromkeys(flat_references + target_ids):
            grouped[str(self.style_by_id[image_id]["feature_shard"])].append(image_id)
        for shard_name, image_ids in grouped.items():
            with safe_open(self.style_root / shard_name, framework="pt", device="cpu") as handle:
                for image_id in image_ids:
                    feature_values[image_id] = {
                        18: handle.get_tensor(f"{image_id}.layer_18_spatial"),
                        24: handle.get_tensor(f"{image_id}.layer_24_spatial"),
                    }
                    global_values[image_id] = handle.get_tensor(
                        f"{image_id}.layer_24_siglip_cls"
                    )
        max_tokens = max(feature_values[item][18].shape[0] for item in flat_references)
        spatial_dim = feature_values[flat_references[0]][18].shape[-1]
        features = {
            layer: torch.zeros(len(flat_references), max_tokens, spatial_dim, dtype=torch.float16)
            for layer in (18, 24)
        }
        feature_mask = torch.zeros(len(flat_references), max_tokens, dtype=torch.bool)
        shapes = []
        for index, image_id in enumerate(flat_references):
            count = feature_values[image_id][18].shape[0]
            feature_mask[index, :count] = True
            for layer in (18, 24):
                features[layer][index, :count] = feature_values[image_id][layer]
            row = self.style_by_id[image_id]
            shapes.append((int(row["target_height"]), int(row["target_width"])))
        global_features = torch.stack([global_values[item] for item in flat_references])
        max_target_tokens = max(feature_values[item][18].shape[0] for item in target_ids)
        target_features = {
            layer: torch.zeros(len(target_ids), max_target_tokens, spatial_dim, dtype=torch.float16)
            for layer in (18, 24)
        }
        target_feature_mask = torch.zeros(len(target_ids), max_target_tokens, dtype=torch.bool)
        for index, image_id in enumerate(target_ids):
            count = feature_values[image_id][18].shape[0]
            target_feature_mask[index, :count] = True
            for layer in (18, 24):
                target_features[layer][index, :count] = feature_values[image_id][layer]
        target_global_features = torch.stack([global_values[item] for item in target_ids])
        return {
            "episodes": episodes,
            "latents": latent_batch.pin_memory(),
            "conditioning": condition_batch.pin_memory(),
            "features": {key: value.pin_memory() for key, value in features.items()},
            "feature_mask": feature_mask.pin_memory(),
            "feature_shapes": shapes,
            "global_features": global_features.pin_memory(),
            "target_features": {
                key: value.pin_memory() for key, value in target_features.items()
            },
            "target_feature_mask": target_feature_mask.pin_memory(),
            "target_global_features": target_global_features.pin_memory(),
            "reference_positions": reference_positions,
            "reference_mask": reference_mask.pin_memory(),
        }

    def prefetch(self, start_step: int, steps: int, workers: int = 2, depth: int = 4) -> Iterator[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures: dict[int, Future[dict[str, Any]]] = {}
            next_step = start_step
            for step in range(start_step, start_step + steps):
                while next_step < start_step + steps and len(futures) < max(1, depth):
                    futures[next_step] = executor.submit(self.load_step, next_step)
                    next_step += 1
                yield futures.pop(step).result()


class SetAggregatorLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attn(values)
        values = values + self.attn(
            normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False
        )[0]
        return values + self.ff(self.norm_ff(values))


class SlotSetAggregator(nn.Module):
    """Order-invariant attention over references, aligned independently per slot."""

    def __init__(
        self,
        slots: int = 16,
        dim: int = 768,
        heads: int = 12,
        layers: int = 2,
        slot_mixer_layers: int = 1,
    ):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.layers = nn.ModuleList([SetAggregatorLayer(dim, heads) for _ in range(layers)])
        self.slot_queries = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.out_ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        # Reference aggregation above is deliberately slot-aligned. Once each
        # slot has been pooled, a small Transformer lets the final style set
        # exchange information across slots without introducing reference-order
        # dependence.
        self.slot_mixers = nn.ModuleList(
            [SetAggregatorLayer(dim, heads) for _ in range(slot_mixer_layers)]
        )

    def forward(self, references: torch.Tensor, reference_mask: torch.Tensor) -> torch.Tensor:
        if references.ndim != 4:
            raise ValueError("references must have shape [batch, references, slots, dim]")
        batch, refs, slots, dim = references.shape
        if (slots, dim) != (self.slots, self.dim):
            raise ValueError(f"Expected slots/dim {(self.slots, self.dim)}, got {(slots, dim)}")
        # [B,R,S,D] -> [B*S,R,D]. No reference positional encoding: permutation invariant.
        values = references.permute(0, 2, 1, 3).reshape(batch * slots, refs, dim)
        padding = (~reference_mask[:, None, :].expand(batch, slots, refs)).reshape(batch * slots, refs)
        for layer in self.layers:
            values = layer(values, padding)
        query = self.slot_queries[None].expand(batch, -1, -1).reshape(batch * slots, 1, dim)
        pooled = query + self.pool(
            self.query_norm(query), self.value_norm(values), self.value_norm(values),
            key_padding_mask=padding, need_weights=False,
        )[0]
        pooled = pooled + self.out_ff(self.out_norm(pooled))
        pooled = pooled.reshape(batch, slots, dim)
        slot_padding = torch.zeros(batch, slots, dtype=torch.bool, device=pooled.device)
        for mixer in self.slot_mixers:
            pooled = mixer(pooled, slot_padding)
        return pooled


class SharedLowRankStyleAdapter(nn.Module):
    """Set aggregation plus shared K/V and per-block low-rank K/V deltas."""

    def __init__(
        self,
        *,
        style_dim: int = 768,
        slots: int = 16,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        output_scale: float = 0.02,
        heads: int = 16,
        blocks: int = 28,
        rank: int = 16,
        aggregator_heads: int = 12,
        aggregator_layers: int = 2,
        aggregator_slot_mixer_layers: int = 1,
        style_dropout: float = 0.12,
        gate_dim: int = 256,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.slots = slots
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_scale = float(output_scale)
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.blocks = blocks
        self.style_dropout = style_dropout
        self.aggregator = SlotSetAggregator(
            slots,
            style_dim,
            aggregator_heads,
            aggregator_layers,
            aggregator_slot_mixer_layers,
        )
        self.null_tokens = nn.Parameter(torch.empty(1, slots, style_dim))
        nn.init.normal_(self.null_tokens, std=0.02)
        self.shared_k = nn.Linear(style_dim, hidden_dim, bias=False)
        self.shared_v = nn.Linear(style_dim, hidden_dim, bias=False)
        self.k_down = nn.ModuleList([nn.Linear(style_dim, rank, bias=False) for _ in range(blocks)])
        self.k_up = nn.ModuleList([nn.Linear(rank, hidden_dim, bias=False) for _ in range(blocks)])
        self.v_down = nn.ModuleList([nn.Linear(style_dim, rank, bias=False) for _ in range(blocks)])
        self.v_up = nn.ModuleList([nn.Linear(rank, hidden_dim, bias=False) for _ in range(blocks)])
        self.shared_o = nn.Linear(hidden_dim, output_dim, bias=False)
        self.o_down = nn.ModuleList(
            [nn.Linear(hidden_dim, rank, bias=False) for _ in range(blocks)]
        )
        self.o_up = nn.ModuleList(
            [nn.Linear(rank, output_dim, bias=False) for _ in range(blocks)]
        )
        nn.init.zeros_(self.shared_o.weight)
        for modules in (self.k_up, self.v_up):
            for layer in modules:
                nn.init.zeros_(layer.weight)
        for layer in self.o_up:
            nn.init.zeros_(layer.weight)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, gate_dim), nn.SiLU(), nn.Linear(gate_dim, blocks))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self._style_tokens: torch.Tensor | None = None
        self._runtime_gate_abs: dict[int, torch.Tensor] = {}
        self._runtime_residual_ratio: dict[int, torch.Tensor] = {}

    def aggregate(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        tokens = self.aggregator(references, reference_mask)
        if apply_dropout and self.training and self.style_dropout > 0:
            dropped = torch.rand(tokens.shape[0], device=tokens.device) < self.style_dropout
            tokens = torch.where(dropped[:, None, None], self.null_tokens.expand_as(tokens), tokens)
        return tokens

    def projected_signature(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compact signature of the actual K/V tensors injected into all blocks."""
        shared_k = self.shared_k(tokens)
        shared_v = self.shared_v(tokens)
        values = []
        for index in range(self.blocks):
            key = shared_k + self.k_up[index](self.k_down[index](tokens))
            value = shared_v + self.v_up[index](self.v_down[index](tokens))
            values.extend((key.mean(1), value.mean(1)))
        return torch.cat(values, dim=-1)

    def unconditional(self, batch: int) -> torch.Tensor:
        return self.null_tokens.expand(batch, -1, -1)

    def set_style_tokens(self, tokens: torch.Tensor) -> None:
        self._style_tokens = tokens

    def clear_style_tokens(self) -> None:
        self._style_tokens = None

    def reset_runtime_stats(self) -> None:
        self._runtime_gate_abs.clear()
        self._runtime_residual_ratio.clear()

    def runtime_stats(self) -> dict[str, float]:
        if not self._runtime_gate_abs:
            return {
                "style_gate_abs_mean": 0.0,
                "style_gate_abs_max": 0.0,
                "style_block_residual_ratio_mean": 0.0,
                "style_block_residual_ratio_max": 0.0,
            }
        gates = torch.stack(list(self._runtime_gate_abs.values())).float()
        ratios = torch.stack(list(self._runtime_residual_ratio.values())).float()
        return {
            "style_gate_abs_mean": float(gates.mean()),
            "style_gate_abs_max": float(gates.max()),
            "style_block_residual_ratio_mean": float(ratios.mean()),
            "style_block_residual_ratio_max": float(ratios.max()),
        }

    def attend(
        self,
        block_index: int,
        normalized_x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        cross_attention: nn.Module,
    ) -> torch.Tensor:
        if self._style_tokens is None:
            return torch.zeros_like(normalized_x)
        style = self._style_tokens
        q = cross_attention.q_proj(normalized_x)
        k = self.shared_k(style) + self.k_up[block_index](self.k_down[block_index](style))
        v = self.shared_v(style) + self.v_up[block_index](self.v_down[block_index](style))
        q = q.reshape(q.shape[0], q.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q = cross_attention.q_norm(q)
        k = cross_attention.k_norm(k)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(normalized_x.shape[0], normalized_x.shape[1], self.hidden_dim)
        attended = self.shared_o(attended) + self.o_up[block_index](
            self.o_down[block_index](attended)
        )
        # The zero-initialized output projection makes the initial residual
        # exactly zero while still receiving a full matrix gradient. A unit-
        # centered timestep gate can learn suppression/amplification only after
        # that useful direction exists; unlike a zero scalar gate it does not
        # block the representation path at initialization.
        gate = 1.0 + torch.tanh(self.gate(timestep_embedding)[:, 0, block_index])
        result = attended * (self.output_scale * gate[:, None, None])
        self._runtime_gate_abs[block_index] = gate.detach().abs().mean()
        self._runtime_residual_ratio[block_index] = (
            result.detach().float().square().mean().sqrt()
            / normalized_x.detach().float().square().mean().sqrt().clamp_min(1e-8)
        )
        return result


def _style_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    emb: torch.Tensor,
    crossattn_emb: torch.Tensor,
    attn_params: Any,
    use_fp32: bool = False,
    rope_emb_L_1_1_D: torch.Tensor | None = None,
    adaln_lora_B_T_3D: torch.Tensor | None = None,
    extra_per_block_pos_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    if use_fp32:
        x = x.float()
    if extra_per_block_pos_emb is not None:
        x = x + extra_per_block_pos_emb
    # The standalone runner supplies normalized diffusion timesteps as FP32.
    # Under non-reentrant checkpoint replay PyTorch does not guarantee that the
    # surrounding autocast has already converted the timestep embedding. Match
    # the frozen AdaLN-LoRA weights explicitly, as the upstream block contract
    # does for its activation path.
    modulation = block.adaln_modulation_self_attn[-1]
    modulation_dtype = modulation.weight.dtype
    emb = emb.to(dtype=modulation_dtype)
    if adaln_lora_B_T_3D is not None:
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype=modulation_dtype)
    with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=use_fp32):
        if block.use_adaln_lora:
            self_mod = block.adaln_modulation_self_attn(emb) + adaln_lora_B_T_3D
            cross_mod = block.adaln_modulation_cross_attn(emb) + adaln_lora_B_T_3D
            mlp_mod = block.adaln_modulation_mlp(emb) + adaln_lora_B_T_3D
        else:
            self_mod = block.adaln_modulation_self_attn(emb)
            cross_mod = block.adaln_modulation_cross_attn(emb)
            mlp_mod = block.adaln_modulation_mlp(emb)
        shift_self, scale_self, gate_self = self_mod.chunk(3, dim=-1)
        shift_cross, scale_cross, gate_cross = cross_mod.chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = mlp_mod.chunk(3, dim=-1)
    expand = lambda value: rearrange(value, "b t d -> b t 1 1 d")
    shift_self, scale_self, gate_self = map(expand, (shift_self, scale_self, gate_self))
    shift_cross, scale_cross, gate_cross = map(expand, (shift_cross, scale_cross, gate_cross))
    shift_mlp, scale_mlp, gate_mlp = map(expand, (shift_mlp, scale_mlp, gate_mlp))
    batch, frames, height, width, _ = x.shape
    normalized = block.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
    result = rearrange(
        block.self_attn(rearrange(normalized, "b t h w d -> b (t h w) d"), attn_params, None, rope_emb=rope_emb_L_1_1_D),
        "b (t h w) d -> b t h w d", t=frames, h=height, w=width,
    )
    x = x + gate_self * result
    normalized = block.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
    result = rearrange(
        block.cross_attn(rearrange(normalized, "b t h w d -> b (t h w) d"), attn_params, crossattn_emb, rope_emb=rope_emb_L_1_1_D),
        "b (t h w) d -> b t h w d", t=frames, h=height, w=width,
    )
    x = x + gate_cross * result

    # Separate style attention, immediately after text attention. It reuses the
    # frozen text-attention Q projection, Q norm, and output projection.
    normalized_style = block.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
    controller = block.__dict__["_style_controller"]()
    style_result = controller.attend(
        block.__dict__["_style_block_index"],
        rearrange(normalized_style, "b t h w d -> b (t h w) d"),
        emb,
        block.cross_attn,
    )
    x = x + rearrange(style_result, "b (t h w) d -> b t h w d", t=frames, h=height, w=width)
    normalized = block.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
    return x + gate_mlp * block.mlp(normalized)


def attach_style_adapter(anima: nn.Module, adapter: SharedLowRankStyleAdapter) -> None:
    if len(anima.blocks) != adapter.blocks:
        raise ValueError(f"Adapter expects {adapter.blocks} blocks, Anima has {len(anima.blocks)}")
    anima.style_adapter = adapter
    for index, block in enumerate(anima.blocks):
        if "_style_original_forward" in block.__dict__:
            continue
        block.__dict__["_style_original_forward"] = block._forward
        block.__dict__["_style_controller"] = weakref.ref(adapter)
        block.__dict__["_style_block_index"] = index

        def patched(self, *args, **kwargs):
            return _style_block_forward(self, *args, **kwargs)

        block._forward = types.MethodType(patched, block)


def load_per_reference_resampler(destination: Path, cfg: dict[str, Any], device: str):
    checkpoint_path = destination / str(cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = checkpoint["training"]
    variant = checkpoint["variant"]
    # Infer C-RADIO width from the trained normalization parameter; this keeps
    # checkpoint loading tied to actual weights rather than a remembered width.
    spatial_dim = int(checkpoint["model"]["tap_norms.18.weight"].numel())
    model = build_tap_resampler_model(
        taps=[int(value) for value in variant["taps"]],
        reconstruction_taps=[18, 24],
        spatial_dim=spatial_dim,
        global_kind=str(variant["global"]),
        global_dim=spatial_dim,
        model_dim=int(training["model_dim"]),
        latent_tokens=int(training["latent_tokens"]),
        heads=int(training["heads"]),
        resampler_layers=int(training["resampler_layers"]),
        decoder_layers=int(training["decoder_layers"]),
        style_dim=int(training["style_dim"]),
        spatial_fusion=str(training["spatial_fusion"]),
        direct_style_tokens=bool(training["direct_style_tokens"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval().to(device)
    return model


def _encode_reference_tokens(model, batch: dict[str, Any], device: str) -> torch.Tensor:
    non_blocking = device.startswith("cuda")
    features = {key: value.to(device, non_blocking=non_blocking) for key, value in batch["features"].items()}
    mask = batch["feature_mask"].to(device, non_blocking=non_blocking)
    global_features = batch["global_features"].to(device, non_blocking=non_blocking)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=non_blocking):
        _, tokens = model.encode(features, mask, global_features)
    batch_size, max_refs = batch["reference_mask"].shape
    packed = tokens.new_zeros(batch_size, max_refs, tokens.shape[1], tokens.shape[2])
    for source, (batch_index, ref_index) in enumerate(batch["reference_positions"]):
        packed[batch_index, ref_index] = tokens[source]
    return packed


def _encode_target_tokens(model, batch: dict[str, Any], device: str) -> torch.Tensor:
    non_blocking = device.startswith("cuda")
    features = {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch["target_features"].items()
    }
    mask = batch["target_feature_mask"].to(device, non_blocking=non_blocking)
    global_features = batch["target_global_features"].to(
        device, non_blocking=non_blocking
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=non_blocking
    ):
        _, tokens = model.encode(features, mask, global_features)
    return tokens


def _symmetric_style_contrastive_loss(
    references: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # Slots are aligned by the frozen per-reference Resampler. Averaging their
    # cosine scores retains that alignment without letting tensor magnitude
    # solve the artist-matching task.
    references = F.normalize(references.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    logits = torch.einsum("bsd,csd->bc", references, targets)
    logits = logits / (references.shape[1] * temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def _resolve_anima_model(config: dict[str, Any], destination: Path, device: str):
    cache_cfg = config["anima_cache"]
    model_cfg = cache_cfg["models"]
    from huggingface_hub import hf_hub_download

    dit_path = hf_hub_download(
        repo_id=str(model_cfg["repo_id"]), filename=str(model_cfg["dit_filename"]),
        revision=str(model_cfg["revision"]), cache_dir=str(destination / "anima_model_cache"),
    )
    sd_scripts = Path(str(cache_cfg["sd_scripts_path"])).resolve()
    if str(sd_scripts) not in sys.path:
        sys.path.insert(0, str(sd_scripts))
    from library import anima_utils

    model = anima_utils.load_anima_model(
        device=device, dit_path=dit_path, attn_mode="torch", split_attn=False,
        loading_device=device, dit_weight_dtype=torch.bfloat16,
    )
    # load_anima_model places checkpoint tensors directly, while accelerator in
    # the upstream trainer performs the final move that also covers RoPE and
    # other non-checkpoint buffers. This standalone runner must do that step.
    return model.to(device)


def _low_precision_rmsnorm_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the legacy Anima RMSNorm wholly in the activation dtype."""
    # Timestep embedding enters parts of the legacy model as FP32 even when all
    # frozen weights and image activations are BF16. The weight dtype is the
    # model compute contract; normalize there instead of preserving that FP32.
    x = x.to(dtype=module.weight.dtype)
    output = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.eps)
    return output * module.weight


def _fused_attention_compute_qkv(
    module: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projection-fused equivalent of the legacy Anima Attention path."""
    if module.is_selfattn:
        qkv = module.qkv_proj(x)
        q, k, v = qkv.unflatten(
            -1, (3, module.n_heads, module.head_dim)
        ).unbind(dim=-3)
    else:
        q = module.q_proj(x).unflatten(-1, (module.n_heads, module.head_dim))
        context = x if context is None else context
        kv = module.kv_proj(context)
        k, v = kv.unflatten(
            -1, (2, module.n_heads, module.head_dim)
        ).unbind(dim=-3)
    q = module.q_norm(q)
    k = module.k_norm(k)
    v = module.v_norm(v)
    if module.is_selfattn and rope_emb is not None:
        rotary = module.__dict__["_style_apply_rotary"]
        q = rotary(q, rope_emb, tensor_format=module.qkv_format, fused=False)
        k = rotary(k, rope_emb, tensor_format=module.qkv_format, fused=False)
    return q, k, v


def _fuse_frozen_linears(layers: list[nn.Linear]) -> nn.Linear:
    first = layers[0]
    fused = nn.Linear(
        first.in_features,
        sum(layer.out_features for layer in layers),
        bias=any(layer.bias is not None for layer in layers),
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([layer.weight for layer in layers], dim=0))
        if fused.bias is not None:
            fused.bias.copy_(
                torch.cat(
                    [
                        layer.bias
                        if layer.bias is not None
                        else torch.zeros(
                            layer.out_features,
                            device=first.weight.device,
                            dtype=first.weight.dtype,
                        )
                        for layer in layers
                    ],
                    dim=0,
                )
            )
    return fused.requires_grad_(False)


def _final_layer_dtype_guard(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Keep legacy final-layer activations on its frozen matrix dtype."""
    dtype = module.linear.weight.dtype
    values = list(args)
    for index in range(min(2, len(values))):
        if isinstance(values[index], torch.Tensor):
            values[index] = values[index].to(dtype=dtype)
    adaln = kwargs.get("adaln_lora_B_T_3D")
    if isinstance(adaln, torch.Tensor):
        kwargs["adaln_lora_B_T_3D"] = adaln.to(dtype=dtype)
    kwargs["use_fp32"] = False
    return tuple(values), kwargs


def _optimize_frozen_anima(
    anima: nn.Module,
    *,
    low_precision_rmsnorm: bool,
    fuse_attention_projections: bool,
) -> dict[str, int]:
    """Apply inference-safe hot-path optimizations before adapter attachment."""
    counts = {
        "low_precision_rmsnorm": 0,
        "fused_self_attention": 0,
        "fused_cross_attention": 0,
        "final_layer_dtype_guard": 0,
    }
    modules = list(anima.modules())
    if low_precision_rmsnorm:
        compute_dtype = next(
            (
                parameter.dtype
                for parameter in anima.parameters()
                if parameter.dtype in (torch.bfloat16, torch.float16)
            ),
            torch.bfloat16,
        )
        rmsnorm_classes: set[type[nn.Module]] = set()
        for module in modules:
            if module.__class__.__name__ != "RMSNorm":
                continue
            # Legacy RMSNorm parameters are intentionally left FP32 by the
            # loader even when the DiT matrices are BF16. Convert the frozen
            # scale itself, otherwise a weight-dtype keyed forward stays FP32.
            module.weight.data = module.weight.data.to(dtype=compute_dtype)
            rmsnorm_classes.add(module.__class__)
            counts["low_precision_rmsnorm"] += 1
        # nn.Module resolves forward from the module class in its call path.
        # Patch each concrete legacy RMSNorm class once for this process.
        for rmsnorm_class in rmsnorm_classes:
            rmsnorm_class.forward = _low_precision_rmsnorm_forward
        if hasattr(anima, "final_layer"):
            anima.final_layer.register_forward_pre_hook(
                _final_layer_dtype_guard, with_kwargs=True
            )
            counts["final_layer_dtype_guard"] = 1
    if fuse_attention_projections:
        for module in modules:
            if not all(
                hasattr(module, name)
                for name in ("is_selfattn", "compute_qkv", "n_heads", "head_dim")
            ):
                continue
            original = module.compute_qkv
            rotary = getattr(original, "__func__", original).__globals__.get(
                "apply_rotary_pos_emb"
            )
            if module.is_selfattn:
                module.qkv_proj = _fuse_frozen_linears(
                    [module.q_proj, module.k_proj, module.v_proj]
                )
                del module.q_proj, module.k_proj, module.v_proj
                module.__dict__["_style_apply_rotary"] = rotary
                counts["fused_self_attention"] += 1
            else:
                module.kv_proj = _fuse_frozen_linears([module.k_proj, module.v_proj])
                del module.k_proj, module.v_proj
                counts["fused_cross_attention"] += 1
            module.compute_qkv = types.MethodType(_fused_attention_compute_qkv, module)
    return counts


def _parameter_grad_norm(parameters) -> float:
    values = [value.grad.detach().float().norm() for value in parameters if value.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _style_magnitude_ramp(step: int, config: dict[str, Any]) -> float:
    start_step = max(0, int(config.get("style_magnitude_start_step", 0)))
    ramp_steps = max(1, int(config.get("style_magnitude_ramp_steps", 250)))
    return min(1.0, max(0.0, (step - start_step) / ramp_steps))


def _style_bootstrap_state(step: int, config: dict[str, Any]) -> tuple[float, float, float]:
    ramp = _style_magnitude_ramp(step, config)
    anneal_start = int(config.get("style_aux_anneal_start", 10**12))
    anneal_end = max(anneal_start + 1, int(config.get("style_aux_anneal_end", anneal_start + 1)))
    if step <= anneal_start:
        auxiliary = 1.0
    elif step >= anneal_end:
        auxiliary = 0.0
    else:
        auxiliary = 1.0 - (step - anneal_start) / (anneal_end - anneal_start)
    floor = float(config.get("style_output_ratio_floor", 0.0)) * ramp * auxiliary
    target_probability = float(config.get("target_reference_probability", 0.0)) * auxiliary
    return auxiliary, floor, target_probability


def _timestep_interval_bounds(
    timesteps: torch.Tensor,
    calibration: dict[str, Any],
    *,
    lower_scale: float = 1.0,
    upper_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up the empirical artist-tag effect interval for each timestep."""
    edges = torch.as_tensor(
        calibration["timestep_edges"], device=timesteps.device, dtype=timesteps.dtype
    )
    bins = calibration["bins"]
    if edges.numel() != len(bins) + 1:
        raise ValueError("Style calibration must have one more edge than bins")
    indices = torch.bucketize(timesteps, edges[1:-1], right=False)
    lower_values = torch.as_tensor(
        [float(item["p25"]) for item in bins],
        device=timesteps.device,
        dtype=timesteps.dtype,
    )
    upper_values = torch.as_tensor(
        [float(item["p75"]) for item in bins],
        device=timesteps.device,
        dtype=timesteps.dtype,
    )
    return lower_values[indices] * lower_scale, upper_values[indices] * upper_scale


def _anneal_multiplier(step: int, start: int, end: int) -> float:
    if step <= start:
        return 1.0
    if step >= end:
        return 0.0
    return 1.0 - (step - start) / max(1, end - start)


def _self_reference_curriculum_state(step: int, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the staged self-reference curriculum for one optimizer step."""
    if not config:
        return {
            "phase": "target_excluded",
            "gate_only": False,
            "target_only": False,
            "target_probability": 0.0,
            "oracle_required": False,
            "self_reference_steps": 0,
        }
    gate_only_steps = max(0, int(config.get("gate_only_steps", 0)))
    self_reference_steps = max(gate_only_steps, int(config.get("self_reference_steps", 0)))
    target_anneal_end = max(
        self_reference_steps + 1,
        int(config.get("target_anneal_end", self_reference_steps + 1)),
    )
    oracle_distill_end = max(
        self_reference_steps,
        int(config.get("oracle_distill_end", target_anneal_end)),
    )
    if step <= gate_only_steps:
        phase = "output_bootstrap_self_reference"
    elif step <= self_reference_steps:
        phase = "full_self_reference"
    elif step < target_anneal_end:
        phase = "oracle_target_anneal"
    else:
        phase = "target_excluded"
    if step <= self_reference_steps:
        target_probability = 1.0
    elif step >= target_anneal_end:
        target_probability = 0.0
    else:
        target_probability = 1.0 - (
            (step - self_reference_steps) / (target_anneal_end - self_reference_steps)
        )
    return {
        "phase": phase,
        "gate_only": step <= gate_only_steps,
        "target_only": step <= self_reference_steps,
        "target_probability": target_probability,
        "oracle_required": self_reference_steps < step < oracle_distill_end,
        "self_reference_steps": self_reference_steps,
    }


def _set_adapter_trainable_stage(
    adapter: SharedLowRankStyleAdapter, *, gate_only: bool
) -> None:
    """Bootstrap the exact-zero output map before opening the K/V path."""
    for parameter in adapter.parameters():
        parameter.requires_grad_(not gate_only)
    if gate_only:
        bootstrap_modules = [
            adapter.gate,
            adapter.shared_o,
            adapter.o_down,
            adapter.o_up,
        ]
        for module in bootstrap_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)


@contextmanager
def _use_style_controller(anima: nn.Module, adapter: SharedLowRankStyleAdapter):
    previous = [block.__dict__["_style_controller"] for block in anima.blocks]
    controller = weakref.ref(adapter)
    for block in anima.blocks:
        block.__dict__["_style_controller"] = controller
    try:
        yield
    finally:
        for block, original in zip(anima.blocks, previous, strict=True):
            block.__dict__["_style_controller"] = original


@contextmanager
def _bypass_style_blocks(anima: nn.Module, adapter: SharedLowRankStyleAdapter):
    del anima
    adapter.clear_style_tokens()
    try:
        yield
    finally:
        adapter.clear_style_tokens()


def _soft_interval_loss(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Huber penalty outside an interval, with exactly zero cost inside it."""
    violations = F.relu(lower - values) + F.relu(values - upper)
    return F.smooth_l1_loss(
        violations, torch.zeros_like(violations), beta=max(float(beta), 1e-8)
    )


def _flow_direction_loss(
    delta: torch.Tensor,
    desired: torch.Tensor,
    base_rms: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Choose a useful residual direction even when the adapter is exactly zero."""
    dimensions = tuple(range(1, delta.ndim))
    scale = base_rms.view(-1, *([1] * (delta.ndim - 1))).clamp_min(1e-6)
    delta_normalized = delta / scale
    desired_unit = desired / (
        desired.square().mean(dim=dimensions, keepdim=True).sqrt().clamp_min(1e-6)
    )
    dot = (delta_normalized * desired_unit).mean(dim=dimensions)
    delta_norm = (
        delta_normalized.square().mean(dim=dimensions) + float(epsilon) ** 2
    ).sqrt()
    return (1.0 - dot / delta_norm).mean()


def _forward_flow_loss(
    anima: nn.Module,
    adapter: SharedLowRankStyleAdapter,
    resampler: nn.Module,
    batch: dict[str, Any],
    device: str,
    *,
    generator: torch.Generator | None = None,
    loss_config: dict[str, Any] | None = None,
    step: int = 0,
    oracle_adapter: SharedLowRankStyleAdapter | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss_config = loss_config or {}
    curriculum = _self_reference_curriculum_state(
        step, dict(loss_config.get("curriculum", {}))
    )
    auxiliary, magnitude_floor, _ = _style_bootstrap_state(
        step, loss_config
    )
    target_probability = float(curriculum["target_probability"])
    direction_multiplier = _anneal_multiplier(
        step,
        int(loss_config.get("style_direction_anneal_start", 0)),
        int(loss_config.get("style_direction_anneal_end", 1)),
    )
    latents = batch["latents"].to(device, non_blocking=True, dtype=torch.bfloat16)
    conditioning = batch["conditioning"].to(device, non_blocking=True, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        target_style_tokens = None
        if target_probability > 0 or curriculum["oracle_required"] or any(
            float(loss_config.get(key, 0.0)) > 0
            for key in ("style_token_contrastive_weight", "style_kv_contrastive_weight")
        ):
            target_style_tokens = _encode_target_tokens(resampler, batch, device)
        if curriculum["target_only"]:
            references = target_style_tokens[:, None]
            reference_mask = torch.ones(
                references.shape[:2], dtype=torch.bool, device=references.device
            )
        else:
            references = _encode_reference_tokens(resampler, batch, device)
            reference_mask = batch["reference_mask"].to(device, non_blocking=True)
            if target_probability > 0:
                include_target = (
                    torch.rand(references.shape[0], device=references.device)
                    < target_probability
                )
                references = torch.cat((references, target_style_tokens[:, None]), dim=1)
                reference_mask = torch.cat((reference_mask, include_target[:, None]), dim=1)
        raw_style_tokens = adapter.aggregate(
            references, reference_mask, apply_dropout=False
        )
        style_tokens = raw_style_tokens
        if adapter.training and adapter.style_dropout > 0:
            dropped = torch.rand(
                style_tokens.shape[0], device=style_tokens.device
            ) < adapter.style_dropout
            style_tokens = torch.where(
                dropped[:, None, None],
                adapter.null_tokens.expand_as(style_tokens),
                style_tokens,
            )
        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
        timesteps = torch.rand(latents.shape[0], device=device, dtype=torch.float32, generator=generator)
        sigma = timesteps[:, None, None, None].to(latents.dtype)
        noisy = (1 - sigma) * latents + sigma * noise
        padding_mask = torch.zeros(
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
            device=device, dtype=latents.dtype,
        )

        magnitude_start = max(0, int(loss_config.get("style_magnitude_start_step", 0)))
        magnitude_weight = (
            float(loss_config.get("style_magnitude_weight", 0.0)) * auxiliary
            if step > magnitude_start else 0.0
        )
        direction_weight = float(loss_config.get("style_flow_direction_weight", 0.0)) * direction_multiplier
        bypass_prediction = None
        if bool(loss_config.get("measure_bypass", False)) or magnitude_weight > 0 or direction_weight > 0:
            # Frozen Anima establishes an absolute zero for style intervention.
            # Unlike a shuffled reference, this baseline cannot be made worse by
            # the adapter and therefore cannot satisfy the objective by shortcut.
            with torch.no_grad():
                with _bypass_style_blocks(anima, adapter):
                    bypass_prediction = anima(
                        noisy.unsqueeze(2), timesteps.to(latents.dtype), context=conditioning,
                        padding_mask=padding_mask, target_input_ids=None,
                    ).squeeze(2).float()

        adapter.reset_runtime_stats()
        adapter.set_style_tokens(style_tokens)
        prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype), context=conditioning,
            padding_mask=padding_mask, target_input_ids=None,
        ).squeeze(2)
        prediction = prediction.float()
        target_velocity = (noise - latents).float()
        flow_loss = F.mse_loss(prediction, target_velocity)

        oracle_distill_loss = flow_loss.new_zeros(())
        oracle_distill_applied = False
        oracle_weight = float(loss_config.get("oracle_distill_weight", 0.0))
        oracle_probability = float(loss_config.get("oracle_distill_probability", 1.0))
        if curriculum["oracle_required"] and oracle_adapter is not None and oracle_weight > 0:
            apply_oracle = bool(
                torch.rand((), device=prediction.device, generator=generator)
                < oracle_probability
            )
            if apply_oracle:
                oracle_mask = torch.ones(
                    target_style_tokens.shape[0], 1,
                    dtype=torch.bool, device=target_style_tokens.device,
                )
                with torch.no_grad():
                    oracle_tokens = oracle_adapter.aggregate(
                        target_style_tokens[:, None], oracle_mask, apply_dropout=False
                    )
                    oracle_adapter.set_style_tokens(oracle_tokens)
                    with _use_style_controller(anima, oracle_adapter):
                        oracle_prediction = anima(
                            noisy.unsqueeze(2), timesteps.to(latents.dtype),
                            context=conditioning, padding_mask=padding_mask,
                            target_input_ids=None,
                        ).squeeze(2).float()
                    oracle_adapter.clear_style_tokens()
                dimensions = tuple(range(1, prediction.ndim))
                oracle_scale = (
                    oracle_prediction.square().mean(dim=dimensions, keepdim=True).sqrt()
                    * float(loss_config.get("oracle_distill_scale_floor", 0.005))
                ).clamp_min(1e-5)
                oracle_distill_loss = F.smooth_l1_loss(
                    (prediction - oracle_prediction) / oracle_scale,
                    torch.zeros_like(prediction),
                    beta=float(loss_config.get("oracle_distill_huber_beta", 0.1)),
                )
                oracle_distill_applied = True

        magnitude_loss = flow_loss.new_zeros(())
        direction_loss = flow_loss.new_zeros(())
        output_ratio = flow_loss.new_zeros(())
        if bypass_prediction is not None:
            dimensions = tuple(range(1, prediction.ndim))
            difference_rms = (
                (prediction - bypass_prediction).square().mean(dim=dimensions) + 1e-12
            ).sqrt()
            prediction_rms = (
                bypass_prediction.square().mean(dim=dimensions) + 1e-8
            ).sqrt().detach()
            ratios = difference_rms / prediction_rms
            valid = ~dropped if adapter.training and adapter.style_dropout > 0 else torch.ones_like(
                ratios, dtype=torch.bool
            )
            if valid.any():
                output_ratio = ratios[valid].mean()
                calibration = loss_config.get("_style_effect_calibration")
                if calibration is None:
                    magnitude_loss = F.relu(magnitude_floor - ratios[valid]).square().mean()
                else:
                    lower, upper = _timestep_interval_bounds(
                        timesteps[valid], calibration, lower_scale=(
                            float(loss_config.get("style_guided_effect_scale", 1.0)) * min(
                                1.0,
                                max(
                                    0.0,
                                    _style_magnitude_ramp(step, loss_config),
                                ),
                            )
                        ),
                        upper_scale=float(loss_config.get("style_effect_upper_scale", 1.0)),
                    )
                    magnitude_loss = _soft_interval_loss(
                        ratios[valid], lower, upper,
                        beta=float(loss_config.get("style_interval_huber_beta", 0.01)),
                    )
                if direction_weight > 0:
                    delta = (prediction - bypass_prediction)[valid]
                    desired = (target_velocity - bypass_prediction).detach()[valid]
                    direction_loss = _flow_direction_loss(
                        delta,
                        desired,
                        prediction_rms[valid],
                        epsilon=float(loss_config.get("style_direction_epsilon", 0.01)),
                    )

        token_weight = float(loss_config.get("style_token_contrastive_weight", 0.0))
        kv_weight = float(loss_config.get("style_kv_contrastive_weight", 0.0))
        token_contrastive = flow_loss.new_zeros(())
        kv_contrastive = flow_loss.new_zeros(())
        if token_weight > 0 or kv_weight > 0:
            temperature = float(loss_config.get("style_contrastive_temperature", 0.07))
            if token_weight > 0:
                token_contrastive = _symmetric_style_contrastive_loss(
                    raw_style_tokens, target_style_tokens, temperature
                )
            if kv_weight > 0:
                reference_signature = adapter.projected_signature(raw_style_tokens)
                target_signature = adapter.projected_signature(target_style_tokens)
                kv_contrastive = _symmetric_style_contrastive_loss(
                    reference_signature[:, None], target_signature[:, None], temperature
                )
        loss = (
            flow_loss
            + oracle_weight * oracle_distill_loss
            + magnitude_weight * magnitude_loss
            + direction_weight * direction_loss
            + token_weight * token_contrastive
            + kv_weight * kv_contrastive
        )
    return loss, {
        "references": int(reference_mask.sum()),
        "latent_shape": list(latents.shape),
        "flow_loss": float(flow_loss.detach()),
        "base_flow_loss": (
            float(F.mse_loss(bypass_prediction, target_velocity).detach())
            if bypass_prediction is not None else float("nan")
        ),
        "paired_flow_improvement": (
            float(
                (
                    F.mse_loss(bypass_prediction, target_velocity) - flow_loss
                ).div(F.mse_loss(bypass_prediction, target_velocity).clamp_min(1e-8)).detach()
            )
            if bypass_prediction is not None else float("nan")
        ),
        "curriculum_phase": str(curriculum["phase"]),
        "curriculum_gate_only": bool(curriculum["gate_only"]),
        "oracle_distill_weight": oracle_weight,
        "oracle_distill_applied": oracle_distill_applied,
        "oracle_distill_loss": float(oracle_distill_loss.detach()),
        "style_auxiliary": auxiliary,
        "style_magnitude_floor": magnitude_floor,
        "target_reference_probability": target_probability,
        "style_output_ratio": float(output_ratio.detach()),
        "style_magnitude_loss": float(magnitude_loss.detach()),
        "style_flow_direction_multiplier": direction_multiplier,
        "style_flow_direction_loss": float(direction_loss.detach()),
        "style_token_contrastive": float(token_contrastive.detach()),
        "style_kv_contrastive": float(kv_contrastive.detach()),
        **adapter.runtime_stats(),
    }


@torch.no_grad()
def _validate_style_adapter(
    anima: nn.Module,
    adapter: SharedLowRankStyleAdapter,
    resampler: nn.Module,
    loader: ProductionStyleLoader,
    device: str,
    *,
    batches: int,
    seed: int,
    step: int = 0,
    loss_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    anima.eval()
    adapter.eval()
    losses = []
    references = []
    base_losses = []
    paired_improvements = []
    output_ratios = []
    started = time.perf_counter()
    try:
        for index in range(batches):
            batch = loader.load_step(index)
            generator = torch.Generator(device=device).manual_seed(seed + index)
            loss, details = _forward_flow_loss(
                anima, adapter, resampler, batch, device, generator=generator,
                loss_config={**(loss_config or {}), "measure_bypass": True},
                step=step,
            )
            # Validation reports the actual flow objective. Auxiliary training
            # penalties are logged independently and must not contaminate the
            # comparison against frozen Anima.
            losses.append(details["flow_loss"])
            references.append(details["references"])
            base_losses.append(details["base_flow_loss"])
            paired_improvements.append(details["paired_flow_improvement"])
            output_ratios.append(details["style_output_ratio"])
            adapter.clear_style_tokens()
    finally:
        adapter.clear_style_tokens()
        anima.train()
        adapter.train()
    paired_mean = sum(paired_improvements) / len(paired_improvements)
    if len(paired_improvements) > 1:
        paired_variance = sum(
            (value - paired_mean) ** 2 for value in paired_improvements
        ) / (len(paired_improvements) - 1)
        paired_ci95 = 1.96 * math.sqrt(paired_variance / len(paired_improvements))
    else:
        paired_ci95 = 0.0
    return {
        "loss": sum(losses) / len(losses),
        "base_loss": sum(base_losses) / len(base_losses),
        "paired_improvement": paired_mean,
        "paired_improvement_ci95": paired_ci95,
        "paired_positive_fraction": sum(value > 0 for value in paired_improvements)
        / len(paired_improvements),
        "style_output_ratio": sum(output_ratios) / len(output_ratios),
        "batches": float(batches),
        "mean_references": sum(references) / len(references),
        "elapsed_s": time.perf_counter() - started,
    }


def _load_sampling_vae(config: dict[str, Any], destination: Path):
    from huggingface_hub import hf_hub_download

    cache_cfg = config["anima_cache"]
    model_cfg = cache_cfg["models"]
    path = hf_hub_download(
        repo_id=str(model_cfg["repo_id"]), filename=str(model_cfg["vae_filename"]),
        revision=str(model_cfg["revision"]), cache_dir=str(destination / "anima_model_cache"),
    )
    sd_scripts = Path(str(cache_cfg["sd_scripts_path"])).resolve()
    if str(sd_scripts) not in sys.path:
        sys.path.insert(0, str(sd_scripts))
    from library import qwen_image_autoencoder_kl_2d

    return qwen_image_autoencoder_kl_2d.load_vae(
        path, device="cpu", disable_mmap=True
    ).requires_grad_(False).eval()


def _make_sample_sheet(
    generated: Image.Image,
    loader: ProductionStyleLoader,
    batch: dict[str, Any],
    *,
    base_generated: Image.Image | None = None,
    generated_label: str | None = None,
) -> Image.Image:
    episode = batch["episodes"][0]
    sources = [("target", episode.target_id)] + [
        (f"ref {index + 1}", image_id)
        for index, image_id in enumerate(episode.reference_ids[:4])
    ]
    thumb = 160
    generated_width = generated.width + (base_generated.width if base_generated is not None else 0)
    sheet = Image.new(
        "RGB",
        (max(generated_width, thumb * len(sources)), generated.height + thumb + 28),
        "white",
    )
    if base_generated is None:
        sheet.paste(generated, ((sheet.width - generated.width) // 2, 0))
    else:
        sheet.paste(base_generated, (0, 0))
        sheet.paste(generated, (base_generated.width, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (label, image_id) in enumerate(sources):
        path = Path(str(loader.style_by_id[image_id]["local_path"]))
        with Image.open(path) as source:
            tile = ImageOps.fit(source.convert("RGB"), (thumb, thumb), method=Image.Resampling.LANCZOS)
        x = index * thumb
        sheet.paste(tile, (x, generated.height + 28))
        draw.text((x + 4, generated.height + 6), label, fill="black")
    if base_generated is None:
        draw.text((4, 6), f"generated — {episode.style_id}", fill="white", stroke_width=2, stroke_fill="black")
    else:
        draw.text((4, 6), "frozen Anima (adapter bypassed)", fill="white", stroke_width=2, stroke_fill="black")
        draw.text(
            (base_generated.width + 4, 6),
            generated_label or f"styled — {episode.style_id}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
    return sheet


@torch.no_grad()
def _sample_style_adapter(
    anima: nn.Module,
    adapter: SharedLowRankStyleAdapter,
    resampler: nn.Module,
    loader: ProductionStyleLoader,
    config: dict[str, Any],
    destination: Path,
    output: Path,
    device: str,
    step: int,
    vae: nn.Module | None,
    *,
    reference_mode: str = "heldout",
) -> tuple[Path, nn.Module, float]:
    sample_cfg = config["style_transfer"]["sampling"]
    started = time.perf_counter()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    anima.eval()
    adapter.eval()
    batch = loader.load_step(int(sample_cfg.get("episode", 0)))
    if reference_mode == "self":
        references = _encode_target_tokens(resampler, batch, device)[:, None]
        reference_mask = torch.ones(
            references.shape[:2], dtype=torch.bool, device=device
        )
    elif reference_mode == "heldout":
        references = _encode_reference_tokens(resampler, batch, device)
        reference_mask = batch["reference_mask"].to(device, non_blocking=True)
    else:
        raise ValueError(f"Unknown sample reference mode: {reference_mode}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        positive_style = adapter.aggregate(references, reference_mask)
    positive_text = batch["conditioning"][:1].to(device, dtype=torch.bfloat16)
    null_file = loader.text_root / "null_conditioning.safetensors"
    null_text = load_file(null_file, device="cpu")["empty_prompt"]
    if null_text.ndim == 2:
        null_text = null_text.unsqueeze(0)
    null_text = _pad_text_conditions(
        [null_text[0]], loader.text_conditioning_length
    )
    null_text = null_text.to(device, dtype=torch.bfloat16)
    null_style = adapter.unconditional(1)
    height = int(sample_cfg.get("height", 512))
    width = int(sample_cfg.get("width", 512))
    latent_h, latent_w = height // 8, width // 8
    generator = torch.Generator(device="cpu").manual_seed(int(sample_cfg.get("seed", 20260811)))
    initial_noise = torch.randn(1, 16, 1, latent_h, latent_w, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    steps = int(sample_cfg.get("steps", 20))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.bfloat16)
    shift = float(sample_cfg.get("flow_shift", 3.0))
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, device=device, dtype=torch.bfloat16)
    text_scale = float(sample_cfg.get("text_cfg", 4.0))
    style_scale = float(os.environ.get("ANIMA_STYLE_CFG", sample_cfg.get("style_cfg", 1.0)))

    def predict(x: torch.Tensor, text: torch.Tensor, style: torch.Tensor | None, timestep: torch.Tensor):
        if style is None:
            adapter.clear_style_tokens()
        else:
            adapter.set_style_tokens(style)
        return anima(
            x, timestep.expand(1), context=text, padding_mask=padding_mask,
            target_input_ids=None,
        ).float()

    def denoise(*, with_style: bool) -> torch.Tensor:
        x = initial_noise.clone()
        for index in range(steps):
            timestep = sigmas[index].to(torch.bfloat16)
            if with_style:
                base = predict(x, null_text, null_style, timestep)
                text_only = predict(x, positive_text, null_style, timestep)
                full = predict(x, positive_text, positive_style, timestep)
                velocity = base + text_scale * (text_only - base) + style_scale * (full - text_only)
            else:
                # A missing style token returns zero before any style projection,
                # while retaining the optimized BF16 Anima block path.
                base = predict(x, null_text, None, timestep)
                text_only = predict(x, positive_text, None, timestep)
                velocity = base + text_scale * (text_only - base)
            x = (x.float() + velocity * (sigmas[index + 1] - sigmas[index]).float()).to(torch.bfloat16)
            if not torch.isfinite(x).all():
                mode = "styled" if with_style else "base"
                raise FloatingPointError(f"Non-finite {mode} latent at sampling step {index + 1}")
        return x

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            base_x = denoise(with_style=False)
            styled_x = denoise(with_style=True)
    finally:
        adapter.clear_style_tokens()

    if vae is None:
        vae = _load_sampling_vae(config, destination)
    vae.to(device=device, dtype=torch.bfloat16)
    decoded = vae.decode_to_pixels(torch.cat((base_x, styled_x), dim=0)).float()
    target_x = batch["latents"][:1].to(device=device, dtype=torch.bfloat16).unsqueeze(2)
    target_decoded = vae.decode_to_pixels(target_x).float()
    vae.to("cpu")

    def to_image(value: torch.Tensor) -> Image.Image:
        if value.ndim == 4:
            value = value[:, 0]
        pixels = ((value.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)

    base_generated = to_image(decoded[0])
    generated = to_image(decoded[1])
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cfg_label = f"{style_scale:g}".replace(".", "p")
    raw_path = sample_dir / f"step-{step:07d}-{reference_mode}-style-cfg-{cfg_label}.png"
    sheet_path = sample_dir / (
        f"step-{step:07d}-{reference_mode}-style-cfg-{cfg_label}-sheet.png"
    )
    generated.save(raw_path)
    base_generated.save(sample_dir / f"step-{step:07d}-base.png")
    to_image(target_decoded[0]).save(sample_dir / f"step-{step:07d}-cached-target.png")
    _make_sample_sheet(
        generated,
        loader,
        batch,
        base_generated=base_generated,
        generated_label=(
            f"styled CFG {style_scale:g} ({reference_mode}) — "
            f"{batch['episodes'][0].style_id}"
        ),
    ).save(sheet_path)
    print(
        f"sample latent stats step={step} "
        f"base_mean={base_x.float().mean().item():.5f} base_std={base_x.float().std().item():.5f} "
        f"base_absmax={base_x.float().abs().max().item():.5f} "
        f"style_mean={styled_x.float().mean().item():.5f} style_std={styled_x.float().std().item():.5f} "
        f"style_absmax={styled_x.float().abs().max().item():.5f} "
        f"target_mean={target_x.float().mean().item():.5f} target_std={target_x.float().std().item():.5f} "
        f"target_absmax={target_x.float().abs().max().item():.5f}",
        flush=True,
    )
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    random.setstate(python_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    anima.train()
    adapter.train()
    return sheet_path, vae, time.perf_counter() - started


def sample_style_checkpoint(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Render a frozen-base control and styled sample from the resumable checkpoint."""
    cfg = config["style_transfer"]
    device = str(cfg["training"].get("device", "cuda"))
    loader_cfg = {**cfg["loader"], "split": "validation", "batch_size": 1}
    loader_cfg["seed"] = int(cfg.get("seed", 20260811)) ^ 0x51A7
    loader = ProductionStyleLoader(destination, loader_cfg)
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    adapter = SharedLowRankStyleAdapter(**cfg["adapter"]).to(device, dtype=torch.bfloat16)
    attach_style_adapter(anima, adapter)
    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    checkpoint_path = output / "training_state.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(state["adapter"])
    step = int(state["step"])
    sheet, _, elapsed = _sample_style_adapter(
        anima, adapter, resampler, loader, config, destination, output, device, step, None
    )
    return {"step": step, "sheet": str(sheet), "elapsed_s": elapsed}


@torch.no_grad()
def diagnose_style_reference_dependence(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Measure whether the trained adapter actually uses its references.

    Every comparison reuses the same targets, text, noise and timesteps.  The
    only changed variable is the style condition: correct references, another
    artist's references, learned null tokens, or a complete adapter bypass.
    """
    cfg = config["style_transfer"]
    diagnostic_cfg = dict(cfg.get("diagnostics", {}))
    device = str(cfg["training"].get("device", "cuda"))
    loader_cfg = {
        **cfg["loader"],
        "split": "validation",
        "batch_size": int(diagnostic_cfg.get("batch_size", 8)),
    }
    loader_cfg["seed"] = int(cfg.get("seed", 20260811)) ^ 0x51A7
    loader = ProductionStyleLoader(destination, loader_cfg)
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    adapter = SharedLowRankStyleAdapter(**cfg["adapter"]).to(device, dtype=torch.bfloat16)
    attach_style_adapter(anima, adapter)
    adapter.eval()

    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    checkpoint_name = str(diagnostic_cfg.get("checkpoint", "selected-step-0005500.pt"))
    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        output_candidate = output / checkpoint_path
        checkpoint_path = (
            output_candidate if output_candidate.exists() else destination / checkpoint_path
        )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(state["adapter"])

    records: list[dict[str, float]] = []
    batches = int(diagnostic_cfg.get("batches", 4))
    seed = int(diagnostic_cfg.get("seed", 20260811 ^ 0xD1A6))
    for index in range(batches):
        batch = loader.load_step(index)
        references = _encode_reference_tokens(resampler, batch, device)
        reference_mask = batch["reference_mask"].to(device, non_blocking=True)
        target_tokens = _encode_target_tokens(resampler, batch, device)
        target_mask = torch.ones(
            target_tokens.shape[0], 1, dtype=torch.bool, device=device
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            correct_style = adapter.aggregate(references, reference_mask)
            self_style = adapter.aggregate(target_tokens[:, None], target_mask)
        shuffled_style = correct_style.roll(1, dims=0)
        null_style = adapter.unconditional(correct_style.shape[0])

        latents = batch["latents"].to(device, non_blocking=True, dtype=torch.bfloat16)
        conditioning = batch["conditioning"].to(device, non_blocking=True, dtype=torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
        timesteps = torch.rand(latents.shape[0], device=device, dtype=torch.float32, generator=generator)
        sigma = timesteps[:, None, None, None].to(latents.dtype)
        noisy = (1 - sigma) * latents + sigma * noise
        target = (noise - latents).float()
        padding_mask = torch.zeros(
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
            device=device, dtype=latents.dtype,
        )

        def predict(style: torch.Tensor | None, *, bypass: bool = False) -> torch.Tensor:
            patched_forwards = None
            if bypass:
                adapter.clear_style_tokens()
                patched_forwards = [block._forward for block in anima.blocks]
                for block in anima.blocks:
                    block._forward = block.__dict__["_style_original_forward"]
            else:
                adapter.set_style_tokens(style)
            try:
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                ):
                    return anima(
                        noisy.unsqueeze(2), timesteps.to(latents.dtype), context=conditioning,
                        padding_mask=padding_mask, target_input_ids=None,
                    ).squeeze(2).float()
            finally:
                adapter.clear_style_tokens()
                if patched_forwards is not None:
                    for block, patched in zip(anima.blocks, patched_forwards, strict=True):
                        block._forward = patched

        self_prediction = predict(self_style)
        correct_prediction = predict(correct_style)
        shuffled_prediction = predict(shuffled_style)
        null_prediction = predict(null_style)
        bypass_prediction = predict(None, bypass=True)
        flat = F.normalize(correct_style.float().flatten(1), dim=1)
        self_flat = F.normalize(self_style.float().flatten(1), dim=1)
        similarities = flat @ flat.T
        off_diagonal = ~torch.eye(flat.shape[0], device=device, dtype=torch.bool)
        records.append({
            "self_loss": float(F.mse_loss(self_prediction, target)),
            "correct_loss": float(F.mse_loss(correct_prediction, target)),
            "shuffled_loss": float(F.mse_loss(shuffled_prediction, target)),
            "null_loss": float(F.mse_loss(null_prediction, target)),
            "bypass_loss": float(F.mse_loss(bypass_prediction, target)),
            "self_vs_correct_rms": float(
                (self_prediction - correct_prediction).square().mean().sqrt()
            ),
            "self_vs_bypass_rms": float(
                (self_prediction - bypass_prediction).square().mean().sqrt()
            ),
            "correct_vs_shuffled_rms": float(
                (correct_prediction - shuffled_prediction).square().mean().sqrt()
            ),
            "correct_vs_null_rms": float(
                (correct_prediction - null_prediction).square().mean().sqrt()
            ),
            "correct_vs_bypass_rms": float(
                (correct_prediction - bypass_prediction).square().mean().sqrt()
            ),
            "shuffled_vs_bypass_rms": float(
                (shuffled_prediction - bypass_prediction).square().mean().sqrt()
            ),
            "null_vs_bypass_rms": float(
                (null_prediction - bypass_prediction).square().mean().sqrt()
            ),
            "prediction_rms": float(correct_prediction.square().mean().sqrt()),
            "style_pairwise_cosine": float(similarities[off_diagonal].mean()),
            "self_correct_style_cosine": float((self_flat * flat).sum(dim=1).mean()),
            "self_correct_style_rms": float(
                (self_style.float() - correct_style.float()).square().mean().sqrt()
            ),
            "style_centered_rms": float(
                (correct_style.float() - correct_style.float().mean(0, keepdim=True))
                .square().mean().sqrt()
            ),
            "style_rms": float(correct_style.float().square().mean().sqrt()),
        })

    means = {
        key: sum(record[key] for record in records) / len(records)
        for key in records[0]
    }
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "step": int(state["step"]),
        "batches": batches,
        "batch_size": loader.batch_size,
        "means": means,
        "records": records,
    }
    write_json(output / "reference_dependence_diagnostic.json", result)
    return result


def _save_training_state(
    path: Path,
    step: int,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "step": step,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary)
    temporary.replace(path)


def _archive_training_state(source: Path, destination: Path) -> None:
    """Snapshot an immutable checkpoint without serializing optimizer state twice."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _save_oracle_adapter(path: Path, adapter: SharedLowRankStyleAdapter) -> None:
    """Persist the immutable end-of-bootstrap teacher once, without optimizer state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    torch.save(state, temporary)
    temporary.replace(path)


def _load_oracle_adapter(
    path: Path,
    template: SharedLowRankStyleAdapter,
    device: str,
) -> SharedLowRankStyleAdapter:
    oracle = copy.deepcopy(template)
    oracle.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    oracle.clear_style_tokens()
    return oracle.requires_grad_(False).eval().to(device, dtype=torch.bfloat16)


def train_style_adapter(config: dict[str, Any], destination: Path, *, steps_override: int | None = None) -> dict[str, Any]:
    cfg = config["style_transfer"]
    training = cfg["training"]
    calibration_path = training.get("style_effect_calibration")
    if calibration_path:
        resolved_calibration = Path(str(calibration_path))
        if not resolved_calibration.is_absolute():
            resolved_calibration = destination / resolved_calibration
        with resolved_calibration.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        if not calibration.get("bins"):
            raise ValueError(f"No timestep bins in style calibration: {resolved_calibration}")
        training["_style_effect_calibration"] = calibration
        print(
            f"loaded empirical style-effect calibration from {resolved_calibration} "
            f"with {len(calibration['bins'])} timestep bins",
            flush=True,
        )
    device = str(training.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Anima style training")
    seed = int(cfg.get("seed", 20260811))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
    loader = ProductionStyleLoader(destination, cfg["loader"])
    validation_loader_cfg = {**cfg["loader"], "split": "validation", "batch_size": 1}
    validation_loader_cfg["seed"] = seed ^ 0x51A7
    validation_loader = ProductionStyleLoader(destination, validation_loader_cfg)
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device)
    anima.requires_grad_(False).train()
    optimization_counts = _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", False)),
        fuse_attention_projections=bool(
            training.get("fuse_attention_projections", False)
        ),
    )
    print(f"frozen Anima optimizations: {optimization_counts}", flush=True)
    if bool(training.get("gradient_checkpointing", False)):
        anima.enable_gradient_checkpointing()
    # Trainable parameters must remain FP32. Plain AdamW does not maintain FP32
    # master weights for BF16 parameters; at the production LR, updates to the
    # nonzero K/V and Aggregator weights otherwise round to exactly zero.
    # Autocast still executes the expensive attention/linear kernels in BF16.
    adapter = SharedLowRankStyleAdapter(**cfg["adapter"]).to(device)
    attach_style_adapter(anima, adapter)
    output_parameters = list(adapter.shared_o.parameters())
    output_parameters += list(adapter.o_down.parameters()) + list(adapter.o_up.parameters())
    gate_parameters = list(adapter.gate.parameters())
    special_ids = {id(value) for value in output_parameters + gate_parameters}
    representation_parameters = [
        value for value in adapter.parameters() if id(value) not in special_ids
    ]
    parameters = representation_parameters + output_parameters + gate_parameters
    if len({id(value) for value in parameters}) != len(list(adapter.parameters())):
        raise RuntimeError("Style optimizer parameter groups do not cover the adapter exactly once")
    representation_lr = float(
        training.get("representation_learning_rate", training.get("learning_rate", 1e-4))
    )
    output_lr = float(training.get("output_learning_rate", representation_lr))
    gate_lr = float(training.get("gate_learning_rate", output_lr))
    weight_decay = float(training.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": representation_parameters,
                "lr": representation_lr,
                "weight_decay": weight_decay,
                "name": "representation",
            },
            {
                "params": output_parameters,
                "lr": output_lr,
                "weight_decay": weight_decay,
                "name": "output",
            },
            {
                "params": gate_parameters,
                "lr": gate_lr,
                "weight_decay": 0.0,
                "name": "gate",
            },
        ],
        fused=device.startswith("cuda"),
    )
    print(
        "style optimizer: FP32 trainable weights, BF16 autocast; "
        f"representation_lr={representation_lr:g} output_lr={output_lr:g} "
        f"gate_lr={gate_lr:g}",
        flush=True,
    )
    steps = int(steps_override if steps_override is not None else training["steps"])
    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    if steps_override is not None:
        output = output / f"smoke-{steps_override}-steps"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "training_state.pt"
    oracle_path = output / "self_reference_oracle.pt"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    resume = bool(training.get("resume", True)) and steps_override is None
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"resuming style training from step {start_step}", flush=True)
    elif steps_override is None and training.get("initial_checkpoint"):
        initial_path = Path(str(training["initial_checkpoint"]))
        if not initial_path.is_absolute():
            initial_path = destination / initial_path
        initial_state = torch.load(initial_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(initial_state["adapter"])
        print(
            f"initialized style adapter from {initial_path} at source step "
            f"{int(initial_state.get('step', -1))}",
            flush=True,
        )
    if start_step >= steps:
        raise RuntimeError(f"Checkpoint is already at step {start_step}, requested steps={steps}")

    oracle_adapter = None
    curriculum_cfg = dict(training.get("curriculum", {}))
    self_reference_steps = int(curriculum_cfg.get("self_reference_steps", 0))
    if start_step > self_reference_steps:
        if not oracle_path.exists():
            raise RuntimeError(
                f"Resuming after self-reference bootstrap requires {oracle_path}"
            )
        oracle_adapter = _load_oracle_adapter(oracle_path, adapter, device)
        print(f"loaded frozen self-reference oracle from {oracle_path}", flush=True)

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)) and steps_override is None:
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "anima-style-transfer")),
            id=str(wandb_cfg.get("id", "anima-style-transfer-l18-l24")),
            resume="allow",
            config=cfg,
        )
    metrics = []
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    accumulation_steps = max(1, int(training.get("gradient_accumulation_steps", 1)))
    iterator = iter(loader.prefetch(
        start_step * accumulation_steps,
        (steps - start_step) * accumulation_steps,
        workers=int(training.get("prefetch_workers", 2)),
        depth=int(training.get("prefetch_batches", 4)),
    ))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    validation_every = int(training.get("validation_every", 500))
    validation_batches = int(training.get("validation_batches", 8))
    full_validation_every = int(training.get("full_validation_every", 0))
    full_validation_batches = int(
        training.get("full_validation_batches", validation_batches)
    )
    sample_every = int(training.get("sample_every", 1000))
    log_every = int(training.get("log_every", 10))
    vae = None
    if start_step == 0 and steps_override is None:
        baseline = _validate_style_adapter(
            anima,
            adapter,
            resampler,
            validation_loader,
            device,
            batches=validation_batches,
            seed=seed ^ 0xA11CE,
        )
        print(
            f"validation step=0 loss={baseline['loss']:.6f} "
            f"base={baseline['base_loss']:.6f} "
            f"paired={baseline['paired_improvement']:.6f} "
            f"ci95=±{baseline['paired_improvement_ci95']:.6f} "
            f"output_ratio={baseline['style_output_ratio']:.6f} "
            f"batches={validation_batches} elapsed_s={baseline['elapsed_s']:.2f}",
            flush=True,
        )
        sheet_path, vae, sample_s = _sample_style_adapter(
            anima,
            adapter,
            resampler,
            validation_loader,
            config,
            destination,
            output,
            device,
            0,
            vae,
            reference_mode="heldout",
        )
        print(f"sample step=0 path={sheet_path} elapsed_s={sample_s:.2f}", flush=True)
        if wandb_run is not None:
            import wandb

            wandb_run.log(
                {
                    **{f"validation/{key}": value for key, value in baseline.items()},
                    "sample/image": wandb.Image(str(sheet_path)),
                    "sample/elapsed_s": sample_s,
                },
                step=0,
            )
    for zero_based_step in range(start_step, steps):
        step = zero_based_step + 1
        curriculum = _self_reference_curriculum_state(step, curriculum_cfg)
        _set_adapter_trainable_stage(adapter, gate_only=bool(curriculum["gate_only"]))
        if curriculum["oracle_required"] and oracle_adapter is None:
            if step != self_reference_steps + 1:
                raise RuntimeError(
                    "Frozen oracle was not available at the curriculum transition"
                )
            adapter.clear_style_tokens()
            _save_oracle_adapter(oracle_path, adapter)
            oracle_adapter = _load_oracle_adapter(oracle_path, adapter, device)
            print(
                f"froze self-reference oracle at step {self_reference_steps}: {oracle_path}",
                flush=True,
            )
        data_ready = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        data_wait = 0.0
        accumulated_loss = 0.0
        details = None
        for _ in range(accumulation_steps):
            wait_started = time.perf_counter()
            batch = next(iterator)
            data_wait += time.perf_counter() - wait_started
            loss, details = _forward_flow_loss(
                anima, adapter, resampler, batch, device,
                loss_config=training, step=step, oracle_adapter=oracle_adapter,
            )
            accumulated_loss += float(loss.detach()) / accumulation_steps
            (loss / accumulation_steps).backward()
            # Style tokens are part of the adapter's live autograd graph, so
            # keep them attached until each microbatch backward completes.
            adapter.clear_style_tokens()
        assert details is not None
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(training.get("max_grad_norm", 1.0)))
        group_grads = {
            "aggregator_grad": _parameter_grad_norm(adapter.aggregator.parameters()),
            "shared_kv_grad": _parameter_grad_norm(
                list(adapter.shared_k.parameters()) + list(adapter.shared_v.parameters())
            ),
            "style_output_grad": _parameter_grad_norm(
                list(adapter.shared_o.parameters())
                + list(adapter.o_down.parameters())
                + list(adapter.o_up.parameters())
            ),
            "gate_grad": _parameter_grad_norm(adapter.gate.parameters()),
        }
        optimizer.step()
        elapsed = time.perf_counter() - data_ready
        row = {
            "step": step, "loss": accumulated_loss, "grad_norm": float(grad_norm),
            "step_s": elapsed, "data_wait_s": data_wait,
            "gradient_accumulation_steps": accumulation_steps,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3) if device.startswith("cuda") else 0.0,
            **details, **group_grads,
        }
        metrics.append(row)
        metrics = metrics[-100:]
        if step == start_step + 1 or step % log_every == 0 or step == steps:
            print(
                f"style step={step}/{steps} loss={row['loss']:.6f} grad={row['grad_norm']:.4f} "
                f"phase={row['curriculum_phase']} "
                f"refs={row['references']} shape={tuple(row['latent_shape'])} step_s={elapsed:.2f} "
                f"data_wait_s={data_wait:.3f} output_ratio={row['style_output_ratio']:.4f} "
                f"mag={row['style_magnitude_loss']:.5f} dir={row['style_flow_direction_loss']:.5f} "
                f"oracle={row['oracle_distill_loss']:.5f}/{int(row['oracle_distill_applied'])} "
                f"gate={row['style_gate_abs_mean']:.4f} "
                f"block_res={row['style_block_residual_ratio_mean']:.4f} "
                f"peak_vram={row['peak_vram_gib']:.2f}GiB",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{key}": value for key, value in row.items() if key not in {"step", "latent_shape"}},
                    step=step,
                )
        if validation_every and step % validation_every == 0:
            current_validation_batches = (
                full_validation_batches
                if full_validation_every and step % full_validation_every == 0
                else validation_batches
            )
            heldout_validation = _validate_style_adapter(
                anima, adapter, resampler, validation_loader, device,
                batches=current_validation_batches, seed=seed ^ 0xA11CE,
            )
            self_validation = _validate_style_adapter(
                anima,
                adapter,
                resampler,
                validation_loader,
                device,
                batches=current_validation_batches,
                seed=seed ^ 0xA11CE,
                step=step,
                loss_config=training,
            )
            print(
                f"validation[heldout] step={step} loss={heldout_validation['loss']:.6f} "
                f"base={heldout_validation['base_loss']:.6f} "
                f"paired={heldout_validation['paired_improvement']:.6f} "
                f"ci95=±{heldout_validation['paired_improvement_ci95']:.6f} "
                f"positive={heldout_validation['paired_positive_fraction']:.3f} "
                f"output_ratio={heldout_validation['style_output_ratio']:.6f} "
                f"batches={current_validation_batches} "
                f"elapsed_s={heldout_validation['elapsed_s']:.2f}",
                flush=True,
            )
            print(
                f"validation[self] step={step} loss={self_validation['loss']:.6f} "
                f"base={self_validation['base_loss']:.6f} "
                f"paired={self_validation['paired_improvement']:.6f} "
                f"ci95=±{self_validation['paired_improvement_ci95']:.6f} "
                f"positive={self_validation['paired_positive_fraction']:.3f} "
                f"output_ratio={self_validation['style_output_ratio']:.6f} "
                f"batches={current_validation_batches} "
                f"elapsed_s={self_validation['elapsed_s']:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{
                            f"validation_heldout/{key}": value
                            for key, value in heldout_validation.items()
                        },
                        **{
                            f"validation_self/{key}": value
                            for key, value in self_validation.items()
                        },
                    },
                    step=step,
                )
        if sample_every and step % sample_every == 0:
            heldout_sheet, vae, heldout_sample_s = _sample_style_adapter(
                anima, adapter, resampler, validation_loader, config, destination,
                output, device, step, vae, reference_mode="heldout",
            )
            self_sheet, vae, self_sample_s = _sample_style_adapter(
                anima, adapter, resampler, validation_loader, config, destination,
                output, device, step, vae, reference_mode="self",
            )
            print(
                f"sample step={step} heldout={heldout_sheet} self={self_sheet} "
                f"elapsed_s={heldout_sample_s + self_sample_s:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb

                wandb_run.log(
                    {
                        "sample/heldout": wandb.Image(str(heldout_sheet)),
                        "sample/self": wandb.Image(str(self_sheet)),
                        "sample/elapsed_s": heldout_sample_s + self_sample_s,
                    },
                    step=step,
                )
        if checkpoint_every and step % checkpoint_every == 0:
            _save_training_state(checkpoint_path, step, adapter, optimizer, cfg)
            _archive_training_state(
                checkpoint_path, checkpoint_dir / f"step-{step:07d}.pt"
            )

    checkpoint = output / "checkpoint.pt"
    _save_training_state(checkpoint_path, steps, adapter, optimizer, cfg)
    _archive_training_state(checkpoint_path, checkpoint)
    summary = {
        "steps": steps, "metrics": metrics, "elapsed_s": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "trainable_parameters": sum(value.numel() for value in parameters),
    }
    write_json(output / "summary.json", summary)
    if wandb_run is not None:
        wandb_run.finish()
    return summary


def smoke_test_style_adapter(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    smoke_config = copy.deepcopy(config)
    training = smoke_config["style_transfer"]["training"]
    training["validation_every"] = 1
    training["validation_batches"] = 1
    training["checkpoint_every"] = 1
    training["sample_every"] = 1
    # Exercise both sides of the new production transition in two real steps:
    # exact-zero gate-only self-reference, then frozen-oracle distillation with
    # the complete student path open.
    training["curriculum"] = {
        "gate_only_steps": 1,
        "self_reference_steps": 1,
        "target_anneal_end": 3,
        "oracle_distill_end": 3,
    }
    training["oracle_distill_probability"] = 1.0
    smoke_config["style_transfer"]["sampling"]["steps"] = 2
    return train_style_adapter(smoke_config, destination, steps_override=2)


def benchmark_style_batches(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Measure end-to-end training throughput without validation/sample overhead."""
    benchmark_cfg = config["style_transfer"].get("benchmark", {})
    batch_sizes = [int(value) for value in benchmark_cfg.get("batch_sizes", [4, 8, 16])]
    steps = int(benchmark_cfg.get("steps", 8))
    warmup = int(benchmark_cfg.get("warmup_steps", 2))
    if warmup >= steps:
        raise ValueError("style_transfer.benchmark.warmup_steps must be smaller than steps")
    results = []
    benchmark_root = destination / "style_transfer_benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    for batch_size in batch_sizes:
        candidate = copy.deepcopy(config)
        style_cfg = candidate["style_transfer"]
        style_cfg["output_directory"] = f"style_transfer_benchmarks/batch-{batch_size}"
        style_cfg["loader"]["batch_size"] = batch_size
        training = style_cfg["training"]
        training.update(
            {
                "validation_every": 0,
                "checkpoint_every": 0,
                "sample_every": 0,
                "log_every": 1,
                "resume": False,
            }
        )
        training.setdefault("wandb", {})["enabled"] = False
        try:
            summary = train_style_adapter(candidate, destination, steps_override=steps)
            measured = summary["metrics"][warmup:]
            mean_compute = sum(row["step_s"] for row in measured) / len(measured)
            mean_wait = sum(row["data_wait_s"] for row in measured) / len(measured)
            wall_step = mean_compute + mean_wait
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "measured_steps": len(measured),
                "mean_compute_s": mean_compute,
                "mean_data_wait_s": mean_wait,
                "mean_wall_step_s": wall_step,
                "target_images_s": batch_size / wall_step,
                "mean_reference_images": sum(row["references"] for row in measured) / len(measured),
                "peak_vram_gib": max(row["peak_vram_gib"] for row in summary["metrics"]),
            }
        except torch.cuda.OutOfMemoryError as error:
            result = {"batch_size": batch_size, "status": "oom", "error": str(error)}
        results.append(result)
        print(f"batch benchmark: {json.dumps(result, ensure_ascii=False)}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = {"steps": steps, "warmup_steps": warmup, "results": results}
    write_json(benchmark_root / "batch_benchmark.json", output)
    return output
