from __future__ import annotations

import copy
import hashlib
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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from PIL import Image, ImageDraw, ImageOps

from .io import read_records, write_json, write_records
from .tap_resampler import (
    _joint_token_descriptor,
    _reconstruction_loss,
    _slot_variation_diversity_loss,
    build_tap_resampler_model,
)


def _format_optional_metric(value: float, digits: int = 4) -> str:
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "n/a"


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


def _pilot_reference_schedule_state(
    optimizer_step: int, schedule: list[dict[str, Any]]
) -> dict[str, Any]:
    if not schedule:
        raise ValueError("pilot reference schedule cannot be empty")
    return next(
        (item for item in schedule if optimizer_step <= int(item["end_step"])),
        schedule[-1],
    )


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
        self.artist_balanced = bool(cfg.get("artist_balanced", False))
        self.reference_curriculum = dict(cfg.get("reference_curriculum", {}))
        self.pilot_reference_schedule = [
            dict(item) for item in cfg.get("pilot_reference_schedule", [])
        ]
        self.reference_count_weights = cfg.get("reference_count_weights")
        self.self_reference_target_images_per_style = max(
            0, int(cfg.get("self_reference_target_images_per_style", 0))
        )
        self.gradient_accumulation_steps = max(
            1, int(cfg.get("gradient_accumulation_steps", 1))
        )
        # Anima was trained with fixed 512-token post-LLM conditioning. Its
        # cross-attention does not receive a text padding mask, so the trailing
        # zero embeddings are part of the learned softmax normalization and
        # must not be trimmed at runtime.
        self.text_conditioning_length = int(cfg.get("text_conditioning_length", 512))
        allowed_style_ids = {
            str(value) for value in cfg.get("allowed_style_ids", [])
        }

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
            if allowed_style_ids and style_id not in allowed_style_ids:
                continue
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
        self.target_sampling_weights: dict[tuple[int, int], list[float]] = {}
        if self.artist_balanced:
            for shape, image_ids in self.buckets.items():
                self.target_sampling_weights[shape] = [
                    1.0
                    / len(
                        self.by_style[
                            str(
                                self.style_by_id[image_id].get(
                                    "style_id", self.style_by_id[image_id]["artist"]
                                )
                            )
                        ]
                    )
                    for image_id in image_ids
                ]
            # Selecting buckets by their summed inverse-frequency mass and
            # rows by the same weights makes total target exposure close to
            # uniform per artist while preserving exact latent-shape batches.
            self.bucket_weights = [
                sum(self.target_sampling_weights[key]) for key in self.bucket_keys
            ]
        else:
            self.bucket_weights = [len(self.buckets[key]) for key in self.bucket_keys]
        self.self_reference_buckets: dict[tuple[int, int], list[int]] = {}
        self.self_reference_target_ids: set[int] = set()
        if self.reference_curriculum and self.self_reference_target_images_per_style > 0:
            selected_ids: set[int] = set()
            for style_id, image_ids in sorted(self.by_style.items()):
                digest = hashlib.blake2b(
                    f"{self.seed}:{style_id}".encode("utf-8"), digest_size=8
                ).digest()
                style_rng = random.Random(int.from_bytes(digest, "little"))
                selected_ids.update(
                    style_rng.sample(
                        image_ids,
                        min(self.self_reference_target_images_per_style, len(image_ids)),
                    )
                )
            for shape, values in self.buckets.items():
                selected = [image_id for image_id in values if image_id in selected_ids]
                if len(selected) >= self.batch_size:
                    self.self_reference_buckets[shape] = selected
            if not self.self_reference_buckets:
                raise RuntimeError(
                    "The self-reference target pool cannot form an exact-shape batch"
                )
            self.self_reference_target_ids = {
                image_id
                for values in self.self_reference_buckets.values()
                for image_id in values
            }
        self.self_reference_bucket_keys = sorted(self.self_reference_buckets)
        self.self_reference_bucket_weights = [
            len(self.self_reference_buckets[key])
            for key in self.self_reference_bucket_keys
        ]
        self.text_shards = _TensorShardCache(text_root, int(cfg.get("text_lru_shards", 2)))
        self.latent_shards = _TensorShardCache(latent_root, int(cfg.get("latent_lru_shards", 2)))
        token_cache = cfg.get("resampler_token_cache")
        self.token_root = (
            destination / str(token_cache) if token_cache else None
        )
        self.token_by_id: dict[int, dict[str, Any]] = {}
        if self.token_root is not None:
            token_rows = read_records(self.token_root / "manifest.parquet")
            self.token_by_id = {int(row["id"]): row for row in token_rows}
            missing = sorted(valid_ids - set(self.token_by_id))
            if missing:
                raise RuntimeError(
                    f"Resampler token cache misses {len(missing)} eligible {self.split} images; "
                    f"first IDs={missing[:8]}"
                )

    def episodes_for_step(self, step: int) -> list[StyleEpisode]:
        rng = random.Random(self.seed + step * 1_000_003)
        reference_curriculum = getattr(self, "reference_curriculum", {})
        pilot_schedule = getattr(self, "pilot_reference_schedule", [])
        if pilot_schedule:
            optimizer_step = step // getattr(self, "gradient_accumulation_steps", 1) + 1
            stage = _pilot_reference_schedule_state(optimizer_step, pilot_schedule)
            min_references = int(stage.get("min_references", 1))
            max_references = int(stage.get("max_references", self.max_references))
            reference_count_weights = stage.get("reference_count_weights")
            curriculum = None
        elif reference_curriculum:
            optimizer_step = step // getattr(self, "gradient_accumulation_steps", 1) + 1
            curriculum = _self_reference_curriculum_state(
                optimizer_step, reference_curriculum
            )
            min_references = int(curriculum["min_references"])
            max_references = int(curriculum["max_references"])
            reference_count_weights = curriculum.get("reference_count_weights")
        else:
            min_references = self.min_references
            max_references = self.max_references
            reference_count_weights = getattr(
                self, "reference_count_weights", None
            )
            curriculum = None
        # Sampling bucket names uniformly would drastically overrepresent rare
        # extreme aspect ratios. Weight by eligible target count so each image
        # retains approximately equal target probability while batches remain
        # exact-shape.
        use_self_reference_pool = bool(
            curriculum
            and curriculum["target_only"]
            and getattr(self, "self_reference_buckets", {})
        )
        if use_self_reference_pool:
            bucket_keys = self.self_reference_bucket_keys
            bucket_weights = self.self_reference_bucket_weights
            target_buckets = self.self_reference_buckets
        else:
            bucket_keys = self.bucket_keys
            bucket_weights = self.bucket_weights
            target_buckets = self.buckets
        shape = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
        candidates = target_buckets[shape]
        chosen: list[int] = []
        attempts = 0
        while len(chosen) < self.batch_size and attempts < max(64, self.batch_size * 32):
            if getattr(self, "artist_balanced", False) and not use_self_reference_pool:
                target_id = rng.choices(
                    candidates,
                    weights=self.target_sampling_weights[shape],
                    k=1,
                )[0]
            else:
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
            upper = min(max_references, len(pool))
            lower = min(min_references, upper)
            counts, probabilities = _reference_count_distribution(
                lower, upper, reference_count_weights
            )
            count = rng.choices(counts, weights=probabilities, k=1)[0]
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
        condition_lengths = []
        for item in episodes:
            row = self.text_by_key[(item.target_id, item.text_variant)]
            shard = self.text_shards.get(str(row["cache_shard"]))
            start = int(row["token_offset"])
            length = int(row["token_length"])
            conditions.append(shard["conditioning"][start : start + length])
            condition_lengths.append(length)
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

        if self.token_root is not None:
            token_values: dict[int, torch.Tensor] = {}
            grouped_tokens: dict[str, list[int]] = defaultdict(list)
            for image_id in dict.fromkeys(flat_references + target_ids):
                row = self.token_by_id[image_id]
                grouped_tokens[str(row["token_shard"])].append(image_id)
            for shard_name, image_ids in grouped_tokens.items():
                # One token shard is about 128 MiB. `load_file` would read the
                # entire tensor for a handful of random references and thrash
                # the NFS/LRU. PySafeSlice maps only the requested rows.
                with safe_open(
                    self.token_root / shard_name, framework="pt", device="cpu"
                ) as handle:
                    token_slice = handle.get_slice("tokens")
                    for image_id in image_ids:
                        token_values[image_id] = token_slice[
                            int(self.token_by_id[image_id]["token_row"])
                        ]
            reference_tokens = torch.stack(
                [token_values[image_id] for image_id in flat_references]
            )
            target_tokens = torch.stack(
                [token_values[image_id] for image_id in target_ids]
            )
            return {
                "episodes": episodes,
                "latents": latent_batch.pin_memory(),
                "conditioning": condition_batch.pin_memory(),
                "conditioning_lengths": torch.tensor(
                    condition_lengths, dtype=torch.long
                ).pin_memory(),
                "cached_reference_tokens": reference_tokens.pin_memory(),
                "cached_target_tokens": target_tokens.pin_memory(),
                "reference_positions": reference_positions,
                "reference_mask": reference_mask.pin_memory(),
            }

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
        target_feature_shapes = []
        for index, image_id in enumerate(target_ids):
            count = feature_values[image_id][18].shape[0]
            target_feature_mask[index, :count] = True
            for layer in (18, 24):
                target_features[layer][index, :count] = feature_values[image_id][layer]
            row = self.style_by_id[image_id]
            target_feature_shapes.append(
                (int(row["target_height"]), int(row["target_width"]))
            )
        target_global_features = torch.stack([global_values[item] for item in target_ids])
        return {
            "episodes": episodes,
            "latents": latent_batch.pin_memory(),
            "conditioning": condition_batch.pin_memory(),
            "conditioning_lengths": torch.tensor(
                condition_lengths, dtype=torch.long
            ).pin_memory(),
            "features": {key: value.pin_memory() for key, value in features.items()},
            "feature_mask": feature_mask.pin_memory(),
            "feature_shapes": shapes,
            "global_features": global_features.pin_memory(),
            "target_features": {
                key: value.pin_memory() for key, value in target_features.items()
            },
            "target_feature_mask": target_feature_mask.pin_memory(),
            "target_feature_shapes": target_feature_shapes,
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
        # Fixed final normalization removes the gate/value scale degeneracy.
        return F.layer_norm(pooled, (dim,))


class MinimalSlotSetAggregator(nn.Module):
    """Tiny reference-order-invariant pooling that preserves slot alignment."""

    def __init__(self, slots: int = 128, dim: int = 1024, bottleneck: int = 256):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.score = nn.Sequential(
            nn.Linear(dim, bottleneck), nn.SiLU(), nn.Linear(bottleneck, 1)
        )
        self.residual = nn.Sequential(
            nn.Linear(dim, bottleneck), nn.SiLU(), nn.Linear(bottleneck, dim)
        )

    def forward(self, references: torch.Tensor, reference_mask: torch.Tensor) -> torch.Tensor:
        if references.ndim != 4:
            raise ValueError("references must have shape [batch, references, slots, dim]")
        _, refs, slots, dim = references.shape
        if (slots, dim) != (self.slots, self.dim):
            raise ValueError(f"Expected slots/dim {(self.slots, self.dim)}, got {(slots, dim)}")
        valid_counts = reference_mask.sum(dim=1)
        # Exact-self bootstrap bypasses every learned aggregation parameter.
        # Use the mask rather than the padded reference dimension: batches can
        # mix one- and multi-reference episodes.
        if refs == 1:
            return F.layer_norm(references[:, 0], (dim,))
        scores = self.score(references).squeeze(-1)
        scores = scores.masked_fill(~reference_mask[:, :, None], -torch.inf)
        pooled = torch.einsum("brs,brsd->bsd", scores.softmax(dim=1), references)
        result = F.layer_norm(pooled + self.residual(pooled), (dim,))
        single_rows = valid_counts == 1
        single = (references * reference_mask[:, :, None, None]).sum(dim=1)
        exact = F.layer_norm(single, (dim,))
        return torch.where(single_rows[:, None, None], exact, result)


class ConnectorTransformerLayer(nn.Module):
    """Pre-norm connector block with affine-free Q/K/V normalization."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("Connector width must be divisible by its head count")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = values.shape
        q, k, v = self.qkv(self.norm_attn(values)).chunk(3, dim=-1)
        def heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            return F.layer_norm(tensor, (self.head_dim,))
        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        values = values + self.output(attended)
        return values + self.ff(self.norm_ff(values))


class QueryConditionedReferenceHead(nn.Module):
    """Read all style slots with the native Anima query at each block."""

    def __init__(
        self, style_dim: int = 1024, hidden_dim: int = 2048,
        latent_dim: int = 512, blocks: int = 28, heads: int = 8,
    ) -> None:
        super().__init__()
        if latent_dim % heads:
            raise ValueError("Reference-head latent dim must divide its head count")
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = latent_dim // heads
        self.style_kv = nn.Linear(style_dim, latent_dim * 2, bias=False)
        self.query = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.block_embedding = nn.Parameter(torch.zeros(blocks, latent_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.output[-1].weight)

    def _style_heads(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        style = F.layer_norm(tokens.float(), (tokens.shape[-1],)).to(tokens.dtype)
        key, value = self.style_kv(style).chunk(2, dim=-1)
        return tuple(
            item.reshape(batch, item.shape[1], self.heads, self.head_dim).transpose(1, 2)
            for item in (key, value)
        )

    def _query_heads(self, queries: torch.Tensor, block: int) -> torch.Tensor:
        batch, _, query_count, _ = queries.shape
        query = queries.transpose(1, 2).reshape(batch, query_count, self.hidden_dim)
        query = self.query(
            F.layer_norm(query.float(), (self.hidden_dim,)).to(query.dtype)
        ) + self.block_embedding[block][None, None]
        return query.reshape(
            batch, query_count, self.heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self, tokens: torch.Tensor, queries: torch.Tensor, block: int
    ) -> torch.Tensor:
        query = self._query_heads(queries, block)
        key, value = self._style_heads(tokens)
        attended = F.scaled_dot_product_attention(
            query, key, value
        )
        attended = attended.transpose(1, 2).flatten(2)
        return self.output(attended)

    def all_pairs(
        self, tokens: torch.Tensor, queries: torch.Tensor, block: int
    ) -> torch.Tensor:
        """Return [targets, references, queries, hidden] without repeated projections."""
        targets, references = queries.shape[0], tokens.shape[0]
        query = self._query_heads(queries, block)
        key, value = self._style_heads(tokens)
        query = query[:, None].expand(-1, references, -1, -1, -1).reshape(
            targets * references, *query.shape[1:]
        )
        key = key[None].expand(targets, -1, -1, -1, -1).reshape(
            targets * references, *key.shape[1:]
        )
        value = value[None].expand(targets, -1, -1, -1, -1).reshape_as(key)
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.output(attended.transpose(1, 2).flatten(2)).reshape(
            targets, references, attended.shape[2], self.hidden_dim
        )

    def grouped_pairs(
        self, tokens: torch.Tensor, queries: torch.Tensor, block: int,
        group_size: int,
    ) -> torch.Tensor:
        """Attend within contiguous negative groups without cross-group B² work."""
        batch = tokens.shape[0]
        if batch % group_size:
            raise ValueError("Reference-head batch must divide into equal groups")
        groups = batch // group_size
        query = self._query_heads(queries, block).reshape(
            groups, group_size, self.heads, -1, self.head_dim
        )
        key, value = self._style_heads(tokens)
        key = key.reshape(groups, group_size, self.heads, -1, self.head_dim)
        value = value.reshape_as(key)
        pair_query = query[:, :, None].expand(
            -1, -1, group_size, -1, -1, -1
        ).reshape(batch * group_size, *query.shape[2:])
        pair_key = key[:, None].expand(
            -1, group_size, -1, -1, -1, -1
        ).reshape(batch * group_size, *key.shape[2:])
        pair_value = value[:, None].expand(
            -1, group_size, -1, -1, -1, -1
        ).reshape_as(pair_key)
        attended = F.scaled_dot_product_attention(pair_query, pair_key, pair_value)
        return self.output(attended.transpose(1, 2).flatten(2)).reshape(
            batch, group_size, attended.shape[2], self.hidden_dim
        )


class SharedLowRankStyleAdapter(nn.Module):
    """Set aggregation plus learned or pretrained-coordinate style attention."""

    def __init__(
        self,
        *,
        style_dim: int = 768,
        slots: int = 16,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        heads: int = 16,
        blocks: int = 28,
        rank: int = 16,
        aggregator_heads: int = 12,
        aggregator_layers: int = 2,
        aggregator_slot_mixer_layers: int = 1,
        style_dropout: float = 0.12,
        gate_dim: int = 256,
        projection_mode: str = "learned_shared",
        context_dim: int | None = None,
        aggregator_mode: str = "transformer",
        aggregator_bottleneck: int = 256,
        connector_layers: int = 0,
        connector_heads: int = 16,
        connector_groups: int = 4,
        connector_group_layers: int = 0,
        connector_summary_tokens: bool = False,
        reference_effect_head: bool = False,
        reference_effect_latent_dim: int = 512,
        reference_effect_heads: int = 8,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.slots = slots
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.blocks = blocks
        self.style_dropout = style_dropout
        self.projection_mode = str(projection_mode)
        self.context_dim = style_dim if context_dim is None else int(context_dim)
        if self.projection_mode not in {"learned_shared", "pretrained_block_lora"}:
            raise ValueError(f"Unknown style projection mode: {self.projection_mode}")
        if self.projection_mode == "pretrained_block_lora" and self.context_dim != style_dim:
            raise ValueError("Identity-initialized style context projection requires context_dim == style_dim")
        if self.projection_mode == "learned_shared" and self.context_dim != style_dim:
            raise ValueError("The learned shared projection mode requires context_dim == style_dim")
        self.aggregator_mode = str(aggregator_mode)
        if self.aggregator_mode == "minimal":
            self.aggregator = MinimalSlotSetAggregator(slots, style_dim, aggregator_bottleneck)
        elif self.aggregator_mode == "transformer":
            self.aggregator = SlotSetAggregator(
                slots, style_dim, aggregator_heads, aggregator_layers,
                aggregator_slot_mixer_layers,
            )
        else:
            raise ValueError(f"Unknown aggregator mode: {self.aggregator_mode}")
        self.null_tokens = nn.Parameter(torch.empty(1, slots, style_dim))
        nn.init.normal_(self.null_tokens, std=0.02)
        if self.projection_mode == "learned_shared":
            self.shared_k = nn.Linear(style_dim, hidden_dim, bias=False)
            self.shared_v = nn.Linear(style_dim, hidden_dim, bias=False)
            self.shared_o = nn.Linear(hidden_dim, output_dim, bias=False)
            nn.init.zeros_(self.shared_o.weight)
        else:
            # C-RADIO/Resampler tokens do not initially share Anima's text
            # conditioning coordinates. A tiny scaled identity keeps a live
            # direction signal without letting Anima's large native
            # gate_cross(t) immediately amplify an unaligned full-strength path.
            self.style_context_proj = nn.Linear(style_dim, self.context_dim, bias=False)
            with torch.no_grad():
                self.style_context_proj.weight.copy_(
                    torch.eye(self.context_dim, style_dim) * 1e-4
                )
        self.connector_groups = int(connector_groups)
        if blocks % self.connector_groups:
            raise ValueError("Anima blocks must divide evenly across connector groups")
        self.blocks_per_group = blocks // self.connector_groups
        self.connector_enabled = connector_layers > 0 or connector_group_layers > 0
        self.connector_summary_tokens = bool(connector_summary_tokens)
        self.connector_trunk = nn.ModuleList(
            [ConnectorTransformerLayer(self.context_dim, connector_heads) for _ in range(connector_layers)]
        )
        self.connector_branches = nn.ModuleList(
            [
                nn.ModuleList(
                    [ConnectorTransformerLayer(self.context_dim, connector_heads)
                     for _ in range(connector_group_layers)]
                )
                for _ in range(self.connector_groups)
            ]
        )
        if self.connector_enabled:
            self.block_embeddings = nn.Parameter(torch.empty(blocks, self.context_dim))
            nn.init.normal_(self.block_embeddings, std=1e-6)
        else:
            self.register_parameter("block_embeddings", None)
        # The terminal o_up is the sole exact-zero boundary. Keep connector
        # residuals live but tiny so every connector parameter can receive a
        # gradient as soon as the terminal opens, without rotating the
        # pretrained Anima conditioning coordinates at initialization.
        for layer in list(self.connector_trunk) + [
            item for branch in self.connector_branches for item in branch
        ]:
            nn.init.normal_(layer.output.weight, std=1e-7)
            nn.init.normal_(layer.ff[-1].weight, std=1e-7)
        self.k_down = nn.ModuleList([nn.Linear(self.context_dim, rank, bias=False) for _ in range(blocks)])
        self.k_up = nn.ModuleList([nn.Linear(rank, hidden_dim, bias=False) for _ in range(blocks)])
        self.v_down = nn.ModuleList([nn.Linear(self.context_dim, rank, bias=False) for _ in range(blocks)])
        self.v_up = nn.ModuleList([nn.Linear(rank, hidden_dim, bias=False) for _ in range(blocks)])
        self.o_down = nn.ModuleList(
            [nn.Linear(hidden_dim, rank, bias=False) for _ in range(blocks)]
        )
        self.o_up = nn.ModuleList(
            [nn.Linear(rank, output_dim, bias=False) for _ in range(blocks)]
        )
        # K/V deltas also remain live. Their small nonzero up projections keep
        # the copied native K/V basis dominant while avoiding another staged
        # zero-init boundary behind the terminal output map.
        for modules in (self.k_up, self.v_up):
            for layer in modules:
                nn.init.normal_(layer.weight, std=1e-3)
        for layer in self.o_up:
            nn.init.zeros_(layer.weight)
        self.reference_effect_head = (
            QueryConditionedReferenceHead(
                style_dim=style_dim,
                hidden_dim=hidden_dim,
                latent_dim=reference_effect_latent_dim,
                blocks=blocks,
                heads=reference_effect_heads,
            )
            if reference_effect_head else None
        )
        self._style_tokens: torch.Tensor | None = None
        self._style_block_tokens: list[torch.Tensor] | None = None
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

    def _context_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.connector_summary_tokens:
            # Preserve every spatial/style slot while exposing the two global
            # statistics that proved artist-discriminative in the Resampler
            # probe.  This adds no learned bottleneck and only two attention
            # positions, so the connector can use artist identity without
            # having to rediscover pooling through several residual layers.
            normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],)).to(tokens.dtype)
            tokens = torch.cat(
                (
                    tokens,
                    normalized.mean(dim=1, keepdim=True),
                    normalized.std(dim=1, correction=0, keepdim=True),
                ),
                dim=1,
            )
        if self.projection_mode == "pretrained_block_lora":
            return self.style_context_proj(tokens)
        return tokens

    def _block_context_tokens(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        shared = self._context_tokens(tokens)
        if not self.connector_enabled:
            return [shared] * self.blocks
        for layer in self.connector_trunk:
            shared = layer(shared)
        groups = []
        for branch in self.connector_branches:
            values = shared
            for layer in branch:
                values = layer(values)
            groups.append(values)
        # The aggregator already provides normalized style tokens.  Preserve
        # the magnitude selected by style_context_proj here: a final
        # LayerNorm would turn the intentionally tiny 1e-4 identity bridge
        # back into an O(1) context and make the initial style residual dwarf
        # the native artist delta.  Connector layers are pre-norm internally,
        # so removing this redundant terminal norm does not destabilize them.
        return [
            groups[index // self.blocks_per_group]
            + self.block_embeddings[index][None, None]
            for index in range(self.blocks)
        ]

    def selected_block_context_tokens(
        self, tokens: torch.Tensor, block_indices: list[int]
    ) -> dict[int, torch.Tensor]:
        """Compute only connector branches needed by the selected blocks."""
        shared = self._context_tokens(tokens)
        if not self.connector_enabled:
            return {index: shared for index in block_indices}
        for layer in self.connector_trunk:
            shared = layer(shared)
        groups: dict[int, torch.Tensor] = {}
        for group_index in sorted({index // self.blocks_per_group for index in block_indices}):
            values = shared
            for layer in self.connector_branches[group_index]:
                values = layer(values)
            groups[group_index] = values
        return {
            index: groups[index // self.blocks_per_group]
            + self.block_embeddings[index][None, None]
            for index in block_indices
        }

    @staticmethod
    def _pretrained_kv(
        cross_attention: nn.Module, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(cross_attention, "kv_proj"):
            return cross_attention.kv_proj(context).chunk(2, dim=-1)
        return cross_attention.k_proj(context), cross_attention.v_proj(context)

    def projected_signature(
        self,
        tokens: torch.Tensor,
        cross_attentions: list[nn.Module] | None = None,
    ) -> torch.Tensor:
        """Compact signature of the actual K/V tensors injected into all blocks."""
        block_contexts = self._block_context_tokens(tokens)
        if self.projection_mode == "learned_shared":
            shared_k = self.shared_k(block_contexts[0])
            shared_v = self.shared_v(block_contexts[0])
        elif cross_attentions is None or len(cross_attentions) != self.blocks:
            raise ValueError("Pretrained K/V signatures require every Anima cross-attention block")
        values = []
        for index in range(self.blocks):
            if self.projection_mode == "learned_shared":
                base_k, base_v = shared_k, shared_v
            else:
                base_k, base_v = self._pretrained_kv(cross_attentions[index], block_contexts[index])
            context = block_contexts[index]
            key = base_k + self.k_up[index](self.k_down[index](context))
            value = base_v + self.v_up[index](self.v_down[index](context))
            values.extend((key.mean(1), value.mean(1)))
        return torch.cat(values, dim=-1)

    def output_parameters(self) -> list[nn.Parameter]:
        values = list(self.o_down.parameters()) + list(self.o_up.parameters())
        if self.projection_mode == "learned_shared":
            values = list(self.shared_o.parameters()) + values
        return values

    def kv_parameters(self) -> list[nn.Parameter]:
        values = (
            list(self.k_down.parameters())
            + list(self.k_up.parameters())
            + list(self.v_down.parameters())
            + list(self.v_up.parameters())
        )
        if self.projection_mode == "learned_shared":
            return list(self.shared_k.parameters()) + list(self.shared_v.parameters()) + values
        return values

    def bridge_parameters(self) -> list[nn.Parameter]:
        if self.projection_mode == "pretrained_block_lora":
            return list(self.style_context_proj.parameters())
        return []

    def gate_bootstrap_parameters(self) -> list[nn.Parameter]:
        return self.output_parameters()

    def gate_parameters(self) -> list[nn.Parameter]:
        """Compatibility group: native Anima gate_cross has no adapter parameters."""
        return []

    def unconditional(self, batch: int) -> torch.Tensor:
        return self.null_tokens.expand(batch, -1, -1)

    def set_style_tokens(self, tokens: torch.Tensor) -> None:
        # Runtime inference commonly keeps the adapter in BF16 while the
        # numerically stable aggregator returns FP32 tokens outside autocast.
        # Normalize once at this boundary instead of relying on caller state.
        parameter = next(self.parameters())
        tokens = tokens.to(device=parameter.device, dtype=parameter.dtype)
        self._style_tokens = tokens
        # Cache the expensive high-capacity connector once per Anima forward.
        # Legacy/no-connector mode retains its stateless behavior.
        self._style_block_tokens = (
            self._block_context_tokens(tokens) if self.connector_enabled else None
        )

    def clear_style_tokens(self) -> None:
        self._style_tokens = None
        self._style_block_tokens = None

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
        cross_attention: nn.Module,
        cross_gate: torch.Tensor,
    ) -> torch.Tensor:
        if self._style_tokens is None:
            return torch.zeros_like(normalized_x)
        style = (
            self._style_block_tokens[block_index]
            if self._style_block_tokens is not None
            else self._block_context_tokens(self._style_tokens)[block_index]
        )
        q = cross_attention.q_proj(normalized_x)
        if self.projection_mode == "learned_shared":
            base_k, base_v = self.shared_k(style), self.shared_v(style)
        else:
            base_k, base_v = self._pretrained_kv(cross_attention, style)
        k = base_k + self.k_up[block_index](self.k_down[block_index](style))
        v = base_v + self.v_up[block_index](self.v_down[block_index](style))
        q = q.reshape(q.shape[0], q.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q = cross_attention.q_norm(q)
        k = cross_attention.k_norm(k)
        v = cross_attention.v_norm(v)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(normalized_x.shape[0], normalized_x.shape[1], self.hidden_dim)
        if self.projection_mode == "learned_shared":
            output_delta = self.o_up[block_index](self.o_down[block_index](attended))
            attended = self.shared_o(attended) + output_delta
        else:
            # Preserve Anima's pretrained output coordinates, but never add
            # that full-rank result directly.  The rank-limited terminal is
            # the sole style residual and its zero-initialized up projection
            # makes the first forward exactly identical to frozen Anima.
            native_output = cross_attention.output_proj(attended)
            attended = self.o_up[block_index](self.o_down[block_index](native_output))
        if self.reference_effect_head is not None:
            attended = attended + self.reference_effect_head(
                self._style_tokens, q, block_index
            )
        # Reuse Anima's pretrained channel-wise text cross-attention gate.
        # There is no independent style scale or learnable timestep gate that
        # can suppress the complete conditioning path as an optimization shortcut.
        result = attended * cross_gate
        debug_label = self.__dict__.get("_debug_autograd_label")
        if block_index == 0 and debug_label:
            print(
                "style autograd probe "
                f"label={debug_label} grad_enabled={torch.is_grad_enabled()} "
                f"tokens_grad={self._style_tokens.requires_grad} "
                f"context_grad={style.requires_grad} k_grad={k.requires_grad} "
                f"v_grad={v.requires_grad} attended_grad={attended.requires_grad} "
                f"gate_grad={cross_gate.requires_grad} result_grad={result.requires_grad} "
                f"context_weight_trainable={self.style_context_proj.weight.requires_grad if self.projection_mode == 'pretrained_block_lora' else True} "
                f"kv_trainable={self.k_up[block_index].weight.requires_grad} "
                f"output_trainable={self.o_up[block_index].weight.requires_grad}",
                flush=True,
            )
        self._runtime_gate_abs[block_index] = cross_gate.detach().abs().mean()
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
    # frozen text-attention Q and its K/V/O coordinate system; pretrained mode
    # still emits only through the zero-init rank-limited terminal.
    normalized_style = block.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
    controller = block.__dict__["_style_controller"]()
    style_result = controller.attend(
        block.__dict__["_style_block_index"],
        rearrange(normalized_style, "b t h w d -> b (t h w) d"),
        block.cross_attn,
        rearrange(
            gate_cross.expand(batch, frames, height, width, -1),
            "b t h w d -> b (t h w) d",
        ),
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


def _create_and_attach_style_adapter(
    anima: nn.Module,
    adapter_config: dict[str, Any],
    device: str,
    *,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Select the legacy or isolated same-Q implementation at one boundary."""
    config = dict(adapter_config)
    architecture = str(config.pop("architecture", "legacy_low_rank"))
    if architecture == "legacy_low_rank":
        adapter = SharedLowRankStyleAdapter(**config)
        adapter = adapter.to(device=device, dtype=dtype) if dtype else adapter.to(device)
        attach_style_adapter(anima, adapter)
        return adapter
    if architecture == "same_q_full_rank":
        from .same_q_style_adapter import (
            SameQFullRankStyleAdapter,
            attach_same_q_style_adapter,
        )

        adapter = SameQFullRankStyleAdapter(**config)
        adapter = adapter.to(device=device, dtype=dtype) if dtype else adapter.to(device)
        # Attachment materializes full-rank K/V copies, so it must happen
        # before optimizer creation or checkpoint loading.
        attach_same_q_style_adapter(anima, adapter)
        return adapter
    raise ValueError(f"Unknown style adapter architecture: {architecture}")


def load_per_reference_resampler(
    destination: Path, cfg: dict[str, Any], device: str, *, trainable: bool = False
):
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
        prototype_pool=str(training.get("prototype_pool", "joint_flatten")),
    )
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(trainable).to(device)
    model.train(trainable)
    return model


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_production_resampler_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Cache frozen Resampler outputs for every production style feature."""
    cfg = config["style_transfer"]
    loader_cfg = cfg["loader"]
    cache_cfg = cfg.get("resampler_token_cache", {})
    output = destination / str(
        cache_cfg.get("output_directory", "production_resampler_tokens")
    )
    output.mkdir(parents=True, exist_ok=True)
    feature_root = destination / str(loader_cfg["style_cache"])
    feature_rows = read_records(feature_root / "manifest.parquet")
    checkpoint = destination / str(cfg["resampler"]["checkpoint"])
    checkpoint_sha256 = _file_sha256(checkpoint)
    final_manifest = output / "manifest.parquet"
    summary_path = output / "summary.json"
    if final_manifest.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("resampler_checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError("Production Resampler token cache checkpoint mismatch")
        if int(summary.get("images", -1)) == len(feature_rows):
            return {**summary, "reused": len(feature_rows)}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[str(row["feature_shard"])].append(row)
    device = str(cache_cfg.get("device", cfg["training"].get("device", "cuda")))
    resampler = load_per_reference_resampler(
        destination, cfg["resampler"], device, trainable=False
    )
    batch_size = int(cache_cfg.get("batch_size", 32))
    prefetch_shards = max(1, int(cache_cfg.get("prefetch_shards", 4)))
    records: list[dict[str, Any]] = []
    shard_groups = sorted(grouped.items())

    def read_feature_shard(item):
        feature_shard, rows = item
        batches = []
        with safe_open(feature_root / feature_shard, framework="pt", device="cpu") as handle:
            for offset in range(0, len(rows), batch_size):
                part = rows[offset : offset + batch_size]
                ids = [int(row["id"]) for row in part]
                layer18 = [
                    handle.get_tensor(f"{image_id}.layer_18_spatial")
                    for image_id in ids
                ]
                layer24 = [
                    handle.get_tensor(f"{image_id}.layer_24_spatial")
                    for image_id in ids
                ]
                batches.append(
                    {
                        "layer18": torch.nn.utils.rnn.pad_sequence(
                            layer18, batch_first=True
                        ).pin_memory(),
                        "layer24": torch.nn.utils.rnn.pad_sequence(
                            layer24, batch_first=True
                        ).pin_memory(),
                        "counts": torch.tensor(
                            [value.shape[0] for value in layer18]
                        ).pin_memory(),
                        "global": torch.stack(
                            [
                                handle.get_tensor(
                                    f"{image_id}.layer_24_siglip_cls"
                                )
                                for image_id in ids
                            ]
                        ).pin_memory(),
                    }
                )
        return feature_shard, rows, batches

    executor = ThreadPoolExecutor(max_workers=prefetch_shards)
    pending = []
    iterator = iter(shard_groups)
    for _ in range(prefetch_shards):
        item = next(iterator, None)
        if item is not None:
            pending.append(executor.submit(read_feature_shard, item))
    started = time.perf_counter()
    try:
        for shard_index in range(len(shard_groups)):
            _, rows, batches = pending.pop(0).result()
            item = next(iterator, None)
            if item is not None:
                pending.append(executor.submit(read_feature_shard, item))
            token_name = f"part-{shard_index:05d}.safetensors"
            token_path = output / token_name
            row_path = output / f"part-{shard_index:05d}.parquet"
            if token_path.exists() and row_path.exists():
                records.extend(read_records(row_path))
                print(
                    f"reused production Resampler token shard "
                    f"{shard_index + 1}/{len(shard_groups)}",
                    flush=True,
                )
                continue
            token_parts = []
            for batch in batches:
                features = {
                    18: batch["layer18"].to(device, non_blocking=True),
                    24: batch["layer24"].to(device, non_blocking=True),
                }
                counts = batch["counts"].to(device, non_blocking=True)
                mask = torch.arange(features[18].shape[1], device=device)[None] < counts[:, None]
                global_features = batch["global"].to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                ):
                    _, tokens = resampler.encode(features, mask, global_features)
                token_parts.append(tokens.to("cpu", dtype=torch.bfloat16))
            tokens = torch.cat(token_parts).contiguous()
            if tuple(tokens.shape[1:]) != (128, 1024) or not torch.isfinite(tokens).all():
                raise RuntimeError(f"Invalid production token shard: {tuple(tokens.shape)}")
            temporary = token_path.with_suffix(".safetensors.tmp")
            save_file({"tokens": tokens}, temporary)
            temporary.replace(token_path)
            shard_records = [
                {
                    "id": int(row["id"]),
                    "artist": str(row["artist"]),
                    "style_id": str(row.get("style_id", row["artist"])),
                    "split": str(row.get("split", "train")),
                    "token_shard": token_name,
                    "token_row": index,
                    "slots": 128,
                    "style_dim": 1024,
                    "resampler_checkpoint_sha256": checkpoint_sha256,
                }
                for index, row in enumerate(rows)
            ]
            write_records(row_path, shard_records)
            records.extend(shard_records)
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"cached production Resampler token shard "
                f"{shard_index + 1}/{len(shard_groups)} "
                f"({len(records)}/{len(feature_rows)} images, "
                f"{len(records) / elapsed:.1f} img/s)",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
    if {int(row["id"]) for row in records} != {
        int(row["id"]) for row in feature_rows
    }:
        raise RuntimeError("Production Resampler token cache ID set mismatch")
    write_records(final_manifest, sorted(records, key=lambda row: int(row["id"])))
    summary = {
        "images": len(records),
        "shards": len(shard_groups),
        "slots": 128,
        "style_dim": 1024,
        "dtype": "bfloat16",
        "resampler_checkpoint": str(cfg["resampler"]["checkpoint"]),
        "resampler_checkpoint_sha256": checkpoint_sha256,
        "storage_bytes": sum(
            path.stat().st_size for path in output.glob("part-*.safetensors")
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    return summary


def _pack_reference_tokens(tokens, batch: dict[str, Any]) -> torch.Tensor:
    batch_size, max_refs = batch["reference_mask"].shape
    packed = tokens.new_zeros(batch_size, max_refs, tokens.shape[1], tokens.shape[2])
    for source, (batch_index, ref_index) in enumerate(batch["reference_positions"]):
        packed[batch_index, ref_index] = tokens[source]
    return packed


def _encode_reference_tokens(model, batch: dict[str, Any], device: str) -> torch.Tensor:
    if "cached_reference_tokens" in batch:
        tokens = batch["cached_reference_tokens"].to(
            device, non_blocking=device.startswith("cuda")
        )
        return _pack_reference_tokens(tokens, batch)
    non_blocking = device.startswith("cuda")
    features = {key: value.to(device, non_blocking=non_blocking) for key, value in batch["features"].items()}
    mask = batch["feature_mask"].to(device, non_blocking=non_blocking)
    global_features = batch["global_features"].to(device, non_blocking=non_blocking)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=non_blocking):
        _, tokens = model.encode(features, mask, global_features)
    return _pack_reference_tokens(tokens, batch)


def _encode_target_tokens(model, batch: dict[str, Any], device: str) -> torch.Tensor:
    if "cached_target_tokens" in batch:
        return batch["cached_target_tokens"].to(
            device, non_blocking=device.startswith("cuda")
        )
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


def _encode_reference_tokens_trainable(
    model, batch: dict[str, Any], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.startswith("cuda")
    features = {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch["features"].items()
    }
    mask = batch["feature_mask"].to(device, non_blocking=non_blocking)
    global_features = batch["global_features"].to(
        device, non_blocking=non_blocking
    )
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=non_blocking
    ):
        _, tokens = model.encode(features, mask, global_features)
    return _pack_reference_tokens(tokens, batch), tokens


def _encode_target_tokens_trainable(
    model,
    batch: dict[str, Any],
    device: str,
    *,
    huber_weight: float,
    reconstruct: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.startswith("cuda")
    features = {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch["target_features"].items()
    }
    mask = batch["target_feature_mask"].to(device, non_blocking=non_blocking)
    global_features = batch["target_global_features"].to(
        device, non_blocking=non_blocking
    )
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=non_blocking
    ):
        if reconstruct:
            decoded, decoded_mask, tokens = model(
                features,
                mask,
                batch["target_feature_shapes"],
                global_features,
            )
            reconstruction = _reconstruction_loss(
                decoded, features, decoded_mask, huber_weight
            )
        else:
            _, tokens = model.encode(features, mask, global_features)
            reconstruction = tokens.new_zeros((), dtype=torch.float32)
    return tokens, reconstruction


def _episode_resampler_prototype_losses(
    references: torch.Tensor,
    reference_mask: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    style_ids: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify each target against its batch peers using its own references."""
    joint_targets = _joint_token_descriptor(targets)
    joint_prototypes = []
    slot_prototypes = []
    for index in range(references.shape[0]):
        valid = references[index, reference_mask[index]]
        joint_prototypes.append(
            F.normalize(_joint_token_descriptor(valid).mean(dim=0), dim=-1)
        )
        slot_prototypes.append(
            F.normalize(F.normalize(valid.float(), dim=-1).mean(dim=0), dim=-1)
        )
    joint_prototypes = torch.stack(joint_prototypes)
    slot_prototypes = torch.stack(slot_prototypes)
    style_ids = style_ids or [str(index) for index in range(targets.shape[0])]
    unique_styles = list(dict.fromkeys(style_ids))
    labels = torch.tensor(
        [unique_styles.index(value) for value in style_ids], device=targets.device
    )
    if len(unique_styles) != len(style_ids):
        joint_prototypes = torch.stack(
            [
                F.normalize(
                    joint_prototypes[
                        torch.tensor(
                            [value == style for value in style_ids],
                            device=targets.device,
                        )
                    ].mean(dim=0),
                    dim=-1,
                )
                for style in unique_styles
            ]
        )
        slot_prototypes = torch.stack(
            [
                F.normalize(
                    slot_prototypes[
                        torch.tensor(
                            [value == style for value in style_ids],
                            device=targets.device,
                        )
                    ].mean(dim=0),
                    dim=-1,
                )
                for style in unique_styles
            ]
        )
    joint_loss = F.cross_entropy(
        (joint_targets @ joint_prototypes.T) / temperature, labels
    )
    normalized_targets = F.normalize(targets.float(), dim=-1)
    slot_logits = torch.einsum(
        "bsd,asd->ba", normalized_targets, slot_prototypes
    ) / targets.shape[1]
    slot_loss = F.cross_entropy(slot_logits / temperature, labels)
    return joint_loss, slot_loss


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


def _resolve_anima_model(
    config: dict[str, Any], destination: Path, device: str, *, attn_mode: str = "torch"
):
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
        device=device, dit_path=dit_path, attn_mode=attn_mode, split_attn=False,
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


def _anima_block_dtype_guard(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Match FP32 timestep embeddings to frozen BF16 AdaLN matrices.

    The adapter-patched block path already performs this cast. Native-context
    conditioning does not patch blocks, so it needs the same contract at the
    module boundary while retaining gradients with respect to text/style
    context inputs.
    """

    values = list(args)
    modulation = module.adaln_modulation_self_attn[-1]
    dtype = modulation.weight.dtype
    if len(values) > 1 and isinstance(values[1], torch.Tensor):
        values[1] = values[1].to(dtype=dtype)
    adaln = kwargs.get("adaln_lora_B_T_3D")
    if isinstance(adaln, torch.Tensor):
        kwargs["adaln_lora_B_T_3D"] = adaln.to(dtype=dtype)
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
        "block_dtype_guard": 0,
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
        for block in getattr(anima, "blocks", []):
            if "_native_context_dtype_guard" in block.__dict__:
                continue
            handle = block.register_forward_pre_hook(
                _anima_block_dtype_guard, with_kwargs=True
            )
            block.__dict__["_native_context_dtype_guard"] = handle
            counts["block_dtype_guard"] += 1
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


def _parameter_grad_norm(parameters) -> torch.Tensor:
    parameters = list(parameters)
    values = [value.grad.detach().float().norm() for value in parameters if value.grad is not None]
    return (
        torch.stack(values).norm()
        if values
        else torch.zeros((), device=parameters[0].device if parameters else "cpu")
    )


def _clip_style_gradient_groups(
    bridge_parameters: list[nn.Parameter],
    representation_parameters: list[nn.Parameter],
    output_parameters: list[nn.Parameter],
    gate_parameters: list[nn.Parameter],
    training: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Clip functional adapter paths independently.

    Timestep-gate gradients can be much larger than the K/V and output-path
    gradients. A single global clip then suppresses every path according to
    the gate norm. Independent bounds keep the safety limit without starving
    the representation and output paths on those hard batches.
    """
    default = float(training.get("max_grad_norm", 1.0))
    groups = {
        "bridge": bridge_parameters,
        "representation": representation_parameters,
        "output": output_parameters,
        "gate": gate_parameters,
    }
    limits = {
        "bridge": float(training.get("bridge_max_grad_norm", 0.05)),
        "representation": float(training.get("representation_max_grad_norm", default)),
        "output": float(training.get("output_max_grad_norm", default)),
        "gate": float(training.get("gate_max_grad_norm", default)),
    }
    norms = {
        name: torch.nn.utils.clip_grad_norm_(parameters, limits[name])
        for name, parameters in groups.items()
    }
    norm_device = next(
        parameter.device for parameters in groups.values() for parameter in parameters
    )
    norms["combined"] = torch.stack([
        value.to(norm_device) for value in norms.values()
    ]).square().sum().sqrt()
    return norms


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


def _direction_anneal_multiplier(step: int, config: dict[str, Any]) -> float:
    if (
        "style_direction_anneal_start" not in config
        and "style_direction_anneal_end" not in config
    ):
        return 1.0
    return _anneal_multiplier(
        step,
        int(config.get("style_direction_anneal_start", 0)),
        int(config.get("style_direction_anneal_end", 1)),
    )


def _sample_flow_timesteps(
    batch_size: int,
    device: str | torch.device,
    config: dict[str, Any],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw continuous flow sigmas using Anima-compatible densities."""
    mode = str(config.get("timestep_sampling", "uniform"))
    if mode == "uniform":
        return torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    if mode not in {"sigmoid", "shift"}:
        raise ValueError(f"Unsupported timestep_sampling: {mode}")
    logits = torch.randn(
        batch_size, device=device, dtype=torch.float32, generator=generator
    )
    logits = (
        logits * float(config.get("sigmoid_scale", 1.0))
        + float(config.get("sigmoid_bias", 0.0))
    )
    timesteps = logits.sigmoid()
    if mode == "shift":
        shift = float(config.get("discrete_flow_shift", 1.0))
        if shift <= 0:
            raise ValueError("discrete_flow_shift must be positive")
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
    return timesteps


def _learning_rate_multiplier(
    step: int, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    """Linear warmup followed by cosine decay to a nonzero LR floor."""
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio must be between 0 and 1")
    warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
    if warmup_steps and step <= warmup_steps:
        return max(1, int(step)) / warmup_steps
    decay_steps = max(1, int(total_steps) - warmup_steps)
    progress = min(1.0, max(0.0, (int(step) - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _reference_count_distribution(
    lower: int,
    upper: int,
    weights: list[float] | tuple[float, ...] | None,
) -> tuple[list[int], list[float]]:
    """Resolve configured per-count weights over the active curriculum range."""

    counts = list(range(int(lower), int(upper) + 1))
    if not counts:
        raise ValueError("Reference count range must not be empty")
    if weights is None:
        probability = 1.0 / len(counts)
        return counts, [probability] * len(counts)
    if len(weights) < upper:
        raise ValueError(
            f"reference_count_weights needs at least {upper} values, got {len(weights)}"
        )
    selected = [float(weights[count - 1]) for count in counts]
    if any(value < 0 for value in selected) or sum(selected) <= 0:
        raise ValueError("Active reference_count_weights must be non-negative and nonzero")
    total = sum(selected)
    return counts, [value / total for value in selected]


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
            "min_references": 1,
            "max_references": 8,
            "reference_count_weights": None,
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
    target_mix_end_value = config.get("target_mix_end_step")
    target_mix_end = (
        max(self_reference_steps + 1, int(target_mix_end_value))
        if target_mix_end_value is not None
        else None
    )
    midpoint_probability = float(config.get("target_mix_end_probability", 0.5))
    if not 0.0 <= midpoint_probability <= 1.0:
        raise ValueError("target_mix_end_probability must be between 0 and 1")
    if step <= gate_only_steps:
        phase = "output_bootstrap_self_reference"
    elif step <= self_reference_steps:
        phase = "full_self_reference"
    elif target_mix_end is not None and step <= target_mix_end:
        phase = "target_mix"
    elif step < target_anneal_end:
        phase = "target_anneal" if target_mix_end is not None else "oracle_target_anneal"
    else:
        phase = "target_excluded"
    if step <= self_reference_steps:
        target_probability = 1.0
    elif step >= target_anneal_end:
        target_probability = 0.0
    elif target_mix_end is not None and step <= target_mix_end:
        progress = (step - self_reference_steps) / (
            target_mix_end - self_reference_steps
        )
        target_probability = 1.0 + progress * (midpoint_probability - 1.0)
    elif target_mix_end is not None:
        progress = (step - target_mix_end) / (target_anneal_end - target_mix_end)
        target_probability = midpoint_probability * (1.0 - progress)
    else:
        target_probability = 1.0 - (
            (step - self_reference_steps) / (target_anneal_end - self_reference_steps)
        )
    if step <= self_reference_steps:
        min_references = max_references = 1
    elif target_mix_end is not None and step <= target_mix_end:
        min_references = int(config.get("target_mix_min_references", 1))
        max_references = int(config.get("target_mix_max_references", 4))
    else:
        min_references = int(config.get("target_anneal_min_references", 1))
        max_references = int(config.get("target_anneal_max_references", 8))
    if min_references < 1 or max_references < min_references:
        raise ValueError("Curriculum reference counts must satisfy 1 <= min <= max")
    return {
        "phase": phase,
        "gate_only": step <= gate_only_steps,
        "target_only": step <= self_reference_steps,
        "target_probability": target_probability,
        "oracle_required": self_reference_steps < step < oracle_distill_end,
        "self_reference_steps": self_reference_steps,
        "min_references": min_references,
        "max_references": max_references,
        "reference_count_weights": config.get("reference_count_weights"),
    }


def _set_adapter_trainable_stage(
    adapter: SharedLowRankStyleAdapter, *, gate_only: bool
) -> None:
    """Bootstrap the exact-zero output map before opening the K/V path."""
    for parameter in adapter.parameters():
        parameter.requires_grad_(not gate_only)
    if gate_only:
        for parameter in adapter.gate_bootstrap_parameters():
            parameter.requires_grad_(True)


def _apply_adapter_freeze_policy(
    adapter: nn.Module, training: dict[str, Any]
) -> dict[str, int]:
    """Reapply immutable phase-level freezes after curriculum stage changes.

    The curriculum setter intentionally opens the complete adapter after a
    gate-only stage. Phase A instead needs fixed native K/V copies and fixed
    nonzero alpha for its entire run, otherwise minimizing alpha is an easier
    solution than aligning the visual tokens to Anima's context space.
    """
    groups: dict[str, list[nn.Parameter]] = {}
    if hasattr(adapter, "kv_base_parameters"):
        all_base = list(adapter.kv_base_parameters())
        trainable_base: list[nn.Parameter] = []
        if bool(training.get("train_full_rank_style_k", False)):
            trainable_base.extend(adapter.active_k_base_parameters())
        if bool(training.get("train_full_rank_style_v", False)):
            trainable_base.extend(adapter.active_v_base_parameters())
        trainable_ids = {id(parameter) for parameter in trainable_base}
        frozen_base = [
            parameter for parameter in all_base
            if id(parameter) not in trainable_ids
        ]
        if frozen_base:
            groups[
                "style_kv_base" if not trainable_base else "style_kv_base_frozen"
            ] = frozen_base
        for parameter in trainable_base:
            parameter.requires_grad_(True)
    if bool(training.get("freeze_style_kv", False)):
        groups["style_kv"] = list(adapter.kv_parameters())
    if bool(training.get("freeze_style_alpha", False)):
        groups["style_alpha"] = (
            list(adapter.alpha_parameters())
            if hasattr(adapter, "alpha_parameters")
            else list(adapter.output_parameters())
        )
    for parameters in groups.values():
        for parameter in parameters:
            parameter.requires_grad_(False)
    return {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }


def _adaptive_reference_loss_config(
    training: dict[str, Any],
    state: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Keep discrimination losses off until absolute flow improves.

    Direction-only objectives can increase reference-dependent output without
    reducing frozen Anima's error. A2.1 therefore unlocks them only after
    fixed self-reference validation is positive for consecutive evaluations.
    """
    config = dict(training)
    activation_step = state.get("activation_step")
    ramp_steps = max(1, int(training.get("reference_loss_ramp_steps", 750)))
    progress = (
        0.0
        if activation_step is None
        else min(1.0, max(0.0, (step - int(activation_step)) / ramp_steps))
    )
    config["exact_self_direction_weight"] = progress * float(
        training.get("exact_self_direction_target_weight", 0.0)
    )
    config["style_reference_direction_weight"] = progress * float(
        training.get("style_reference_direction_target_weight", 0.0)
    )
    config["_reference_loss_progress"] = progress
    return config


@torch.no_grad()
def _calibrate_same_q_bridge_output_rms(
    adapter: nn.Module,
    loader: ProductionStyleLoader,
    *,
    batches: int,
) -> dict[str, Any]:
    """Match bridge output scale to real nonzero Anima text tokens.

    Text-cache padding is exactly zero and intentionally remains in Anima's
    512-token attention context.  It must not dilute the scale that native K/V
    learned for actual conditioning tokens, so only nonzero tokens contribute
    to this calibration.
    """
    if not hasattr(adapter, "set_bridge_output_rms"):
        raise TypeError("Bridge RMS calibration requires the same-Q adapter")
    square_sum = 0.0
    value_sum = 0.0
    value_count = 0
    token_count = 0
    padded_token_count = 0
    for index in range(max(1, batches)):
        conditioning = loader.load_step(index)["conditioning"].float()
        token_square_mean = conditioning.square().mean(dim=-1)
        nonzero = token_square_mean > 1e-12
        values = conditioning[nonzero]
        square_sum += float(values.square().sum(dtype=torch.float64))
        value_sum += float(values.sum(dtype=torch.float64))
        value_count += values.numel()
        token_count += int(nonzero.sum())
        padded_token_count += int(nonzero.numel() - nonzero.sum())
    if value_count == 0:
        raise RuntimeError("Text conditioning calibration observed no nonzero tokens")
    rms = math.sqrt(square_sum / value_count)
    mean = value_sum / value_count
    standard_deviation = math.sqrt(max(0.0, square_sum / value_count - mean * mean))
    adapter.set_bridge_output_rms(rms)
    return {
        "batches": max(1, int(batches)),
        "nonzero_tokens": token_count,
        "padding_tokens": padded_token_count,
        "nonzero_text_mean": mean,
        "nonzero_text_standard_deviation": standard_deviation,
        "nonzero_text_rms": rms,
        "bridge_output_rms": rms,
    }


@torch.no_grad()
def _calibrate_same_q_alpha(
    anima: nn.Module,
    adapter: nn.Module,
    resampler: nn.Module,
    loader: ProductionStyleLoader,
    device: str,
    training: dict[str, Any],
    *,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    """Set each alpha from measured raw style/text attention RMS.

    Calibration runs with alpha exactly zero, so it observes every block but
    leaves frozen Anima's activations unperturbed. The resulting per-block
    alpha makes the *actual* pre-O attention contribution start at a common
    target ratio despite different token counts and V statistics.
    """
    if not hasattr(adapter, "begin_alpha_calibration"):
        raise TypeError("Configured alpha RMS calibration requires the same-Q adapter")
    anima.eval()
    adapter.eval()
    adapter.begin_alpha_calibration()
    try:
        calibration_loss_config = {
            **training,
            "curriculum": {"gate_only_steps": 0, "self_reference_steps": 1},
            "timestep_sampling": "uniform",
            "style_flow_loss_weight": 1.0,
            "exact_self_residual_weight": 0.0,
            "exact_self_direction_weight": 0.0,
            "style_reference_direction_weight": 0.0,
            "style_reference_rank_weight": 0.0,
            "style_magnitude_weight": 0.0,
        }
        for index in range(max(1, batches)):
            batch = loader.load_step(index)
            generator = torch.Generator(device=device).manual_seed(seed + index)
            _forward_flow_loss(
                anima,
                adapter,
                resampler,
                batch,
                device,
                generator=generator,
                loss_config=calibration_loss_config,
                step=1,
                collect_details=False,
            )
            adapter.clear_style_tokens()
        return adapter.finish_alpha_calibration(
            float(training.get("alpha_target_style_to_text_ratio", 0.02)),
            minimum_alpha=float(training.get("alpha_minimum", 1e-6)),
            maximum_alpha=float(training.get("alpha_maximum", 0.01)),
        )
    finally:
        adapter.clear_style_tokens()
        anima.train()
        adapter.train()


def _load_adapter_checkpoint(
    adapter: SharedLowRankStyleAdapter, state: dict[str, Any]
) -> None:
    """Load ordinary or offline-bootstrap adapter/reference-head checkpoints."""
    result = adapter.load_state_dict(state["adapter"], strict=False)
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    allowed_missing = {
        key for key in missing if key.startswith("reference_effect_head.")
    }
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible adapter checkpoint; missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    centered = state.get("centered_head")
    if centered is not None:
        if adapter.reference_effect_head is None:
            raise RuntimeError(
                "Checkpoint contains a reference head but adapter.reference_effect_head is disabled"
            )
        adapter.reference_effect_head.load_state_dict(centered)
    elif allowed_missing:
        raise RuntimeError(
            "Adapter enables reference_effect_head but checkpoint contains no trained head"
        )


def _set_aggregator_trainable(
    adapter: SharedLowRankStyleAdapter,
    *,
    step: int,
    start_step: int,
    gate_only: bool,
) -> bool:
    trainable = not gate_only and start_step >= 0 and step >= start_step
    for parameter in adapter.aggregator.parameters():
        parameter.requires_grad_(trainable)
    return trainable


def _replace_reference_with_target(
    references: torch.Tensor,
    reference_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    include_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert target without changing each episode's total reference count."""
    references = references.clone()
    ordinary_counts = reference_mask.sum(dim=1)
    rows = torch.nonzero(include_target, as_tuple=False).flatten()
    if rows.numel() > 0:
        replace_at = ordinary_counts[rows].sub(1).clamp_min(0)
        references[rows, replace_at] = target_tokens[rows]
    return references, reference_mask


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


@contextmanager
def _uncached_no_grad_autocast(device: str):
    """Run a detached teacher branch without poisoning autocast's weight cache.

    A trainable FP32 weight first cast under ``no_grad`` is cached as a
    detached low-precision tensor for the rest of the surrounding autocast
    region. A later student forward would then update only its inputs, not the
    weight. Disabling the nested cache keeps the subsequent correct-reference
    forward connected to every adapter parameter.
    """
    device_type = torch.device(device).type
    with torch.no_grad():
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type in {"cpu", "cuda"},
            cache_enabled=False,
        ):
            yield


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


def _exact_self_residual_losses(
    prediction: torch.Tensor,
    bypass_prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    noisy: torch.Tensor,
    clean_latent: torch.Tensor,
    sigma: torch.Tensor,
    *,
    scale_floor: float = 1e-3,
    huber_beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Directly supervise the adapter residual in the frozen-Anima coordinate system."""
    dimensions = tuple(range(1, prediction.ndim))
    student = prediction - bypass_prediction
    desired = target_velocity - bypass_prediction
    scale = desired.square().mean(dim=dimensions, keepdim=True).sqrt().clamp_min(scale_floor)
    normalized_mse = F.mse_loss(student / scale, desired / scale)
    normalized_huber = F.smooth_l1_loss(
        student / scale, desired / scale, beta=huber_beta
    )
    direction = 1 - F.cosine_similarity(
        student.flatten(1), desired.flatten(1), dim=1
    ).mean()
    student_rms = student.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    desired_rms = desired.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    log_rms = F.smooth_l1_loss(
        torch.log(student_rms), torch.log(desired_rms), beta=0.5
    )
    predicted_x0 = noisy.float() - sigma.float() * prediction
    x0 = F.mse_loss(predicted_x0, clean_latent.float())
    return {
        "normalized_mse": normalized_mse,
        "normalized_huber": normalized_huber,
        "direction": direction,
        "log_rms": log_rms,
        "x0": x0,
    }


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
    collect_details: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss_config = loss_config or {}
    curriculum = _self_reference_curriculum_state(
        step, dict(loss_config.get("curriculum", {}))
    )
    auxiliary, magnitude_floor, _ = _style_bootstrap_state(
        step, loss_config
    )
    target_probability = float(curriculum["target_probability"])
    direction_multiplier = _direction_anneal_multiplier(step, loss_config)
    latents = batch["latents"].to(device, non_blocking=True, dtype=torch.bfloat16)
    conditioning = batch["conditioning"].to(device, non_blocking=True, dtype=torch.bfloat16)
    resampler_train_start = int(loss_config.get("resampler_train_start_step", -1))
    resampler_trainable = resampler_train_start >= 0 and step >= resampler_train_start
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        target_style_tokens = None
        reference_tokens_for_aux = None
        episode_reference_tokens = None
        episode_reference_mask = None
        needs_reference_contrastive = any(
            float(loss_config.get(key, 0.0)) > 0
            for key in ("style_token_contrastive_weight", "style_kv_contrastive_weight")
        )
        resampler_auxiliary_weight = float(
            loss_config.get("resampler_auxiliary_weight", 0.0)
        )
        resampler_reconstruction = latents.new_zeros((), dtype=torch.float32)
        resampler_joint_prototype = latents.new_zeros((), dtype=torch.float32)
        resampler_slot_prototype = latents.new_zeros((), dtype=torch.float32)
        resampler_diversity = latents.new_zeros((), dtype=torch.float32)
        target_included = torch.zeros(
            latents.shape[0], dtype=torch.bool, device=latents.device
        )
        if resampler_trainable:
            needs_episode_references = (
                not curriculum["target_only"]
                or needs_reference_contrastive
                or resampler_auxiliary_weight > 0
            )
            if needs_episode_references:
                references, flat_reference_tokens = _encode_reference_tokens_trainable(
                    resampler, batch, device
                )
                reference_mask = batch["reference_mask"].to(
                    device, non_blocking=True
                )
                episode_reference_tokens = references
                episode_reference_mask = reference_mask
            target_style_tokens, resampler_reconstruction = (
                _encode_target_tokens_trainable(
                    resampler,
                    batch,
                    device,
                    huber_weight=float(loss_config.get("resampler_huber_weight", 0.10)),
                    reconstruct=resampler_auxiliary_weight > 0,
                )
            )
            if resampler_auxiliary_weight > 0:
                reference_tokens_for_aux = references
                (
                    resampler_joint_prototype,
                    resampler_slot_prototype,
                ) = _episode_resampler_prototype_losses(
                    references,
                    reference_mask,
                    target_style_tokens,
                    float(loss_config.get("resampler_prototype_temperature", 0.07)),
                    [str(item.style_id) for item in batch["episodes"]],
                )
                resampler_diversity = _slot_variation_diversity_loss(
                    torch.cat((flat_reference_tokens, target_style_tokens), dim=0),
                    float(loss_config.get("resampler_diversity_margin", 0.20)),
                )
        elif "cached_target_tokens" in batch and (
            target_probability > 0 or curriculum["oracle_required"] or any(
                float(loss_config.get(key, 0.0)) > 0
                for key in ("style_token_contrastive_weight", "style_kv_contrastive_weight")
            )
        ):
            target_style_tokens = _encode_target_tokens(resampler, batch, device)
        elif target_probability > 0 or curriculum["oracle_required"] or any(
            float(loss_config.get(key, 0.0)) > 0
            for key in ("style_token_contrastive_weight", "style_kv_contrastive_weight")
        ):
            target_style_tokens = _encode_target_tokens(resampler, batch, device)
        if needs_reference_contrastive and episode_reference_tokens is None:
            episode_reference_tokens = _encode_reference_tokens(resampler, batch, device)
            episode_reference_mask = batch["reference_mask"].to(
                device, non_blocking=True
            )
        if curriculum["target_only"]:
            references = target_style_tokens[:, None]
            reference_mask = torch.ones(
                references.shape[:2], dtype=torch.bool, device=references.device
            )
            target_included.fill_(True)
        else:
            if not resampler_trainable:
                references = (
                    episode_reference_tokens
                    if episode_reference_tokens is not None
                    else _encode_reference_tokens(resampler, batch, device)
                )
                reference_mask = batch["reference_mask"].to(device, non_blocking=True)
            if target_probability > 0:
                include_target = (
                    torch.rand(references.shape[0], device=references.device)
                    < target_probability
                )
                # The curriculum specifies the total number of style images,
                # not non-target references plus an optional extra target. If
                # target is included, replace one ordinary reference so 1--4
                # and 1--8 remain exact bounds.
                references, reference_mask = _replace_reference_with_target(
                    references,
                    reference_mask,
                    target_style_tokens,
                    include_target,
                )
                target_included = include_target
        raw_style_tokens = adapter.aggregate(
            references, reference_mask, apply_dropout=False
        )
        contrastive_style_tokens = raw_style_tokens
        if needs_reference_contrastive and curriculum["target_only"]:
            # Flow bootstrap uses the exact target as its reference, but the
            # representation objective must not learn an image-identity
            # shortcut. Match the other images from the same artist to the
            # held-out target instead.
            contrastive_style_tokens = adapter.aggregate(
                episode_reference_tokens,
                episode_reference_mask,
                apply_dropout=False,
            )
        style_tokens = raw_style_tokens
        dropped = torch.zeros(
            style_tokens.shape[0], dtype=torch.bool, device=style_tokens.device
        )
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
        timesteps = _sample_flow_timesteps(
            latents.shape[0], device, loss_config, generator
        )
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
        reference_rank_weight = float(loss_config.get("style_reference_rank_weight", 0.0))
        reference_direction_weight = float(
            loss_config.get("style_reference_direction_weight", 0.0)
        )
        reference_residual_mse_weight = float(
            loss_config.get("style_reference_residual_mse_weight", 0.0)
        )
        reference_rank_start = int(loss_config.get("style_reference_rank_start_step", 0))
        reference_rank_end = int(loss_config.get("style_reference_rank_end_step", 0))
        reference_rank_probability = float(
            loss_config.get("style_reference_rank_probability", 1.0)
        )
        reference_wrong_grad_samples = max(
            0, int(loss_config.get("style_reference_wrong_grad_samples", 0))
        )
        reference_rank_active = (
            (
                reference_rank_weight > 0
                or reference_direction_weight > 0
                or reference_residual_mse_weight > 0
            )
            and style_tokens.shape[0] > 1
            and step >= reference_rank_start
            and (reference_rank_end <= 0 or step < reference_rank_end)
            and bool(
                torch.rand((), device=style_tokens.device, generator=generator)
                < reference_rank_probability
            )
        )
        exact_self_mask = target_included & ~dropped
        exact_self_fraction = exact_self_mask.float().mean()
        exact_self_supervision_active = bool(exact_self_mask.any()) and any(
            float(loss_config.get(key, 0.0)) > 0
            for key in (
                "exact_self_residual_weight",
                "exact_self_direction_weight",
                "exact_self_log_rms_weight",
                "exact_self_x0_weight",
            )
        )
        bypass_prediction = None
        if (
            bool(loss_config.get("measure_bypass", False))
            or magnitude_weight > 0
            or direction_weight > 0
            or reference_rank_active
            or exact_self_supervision_active
        ):
            # Frozen Anima establishes an absolute zero for style intervention.
            # Unlike a shuffled reference, this baseline cannot be made worse by
            # the adapter and therefore cannot satisfy the objective by shortcut.
            with torch.no_grad():
                with _bypass_style_blocks(anima, adapter):
                    bypass_prediction = anima(
                        noisy.unsqueeze(2), timesteps.to(latents.dtype), context=conditioning,
                        padding_mask=padding_mask, target_input_ids=None,
                    ).squeeze(2).float()

        wrong_reference_prediction = None
        wrong_reference_indices = None
        if reference_rank_active:
            shuffled_tokens = raw_style_tokens.roll(1, dims=0)
            if reference_wrong_grad_samples > 0:
                # Keep a small wrong-reference branch in the graph. In the
                # direction objective, gradients through correct - wrong then
                # cancel any reference-independent path. A full second batch
                # would retain two Anima graphs and exceed practical VRAM, so
                # sample only a few non-dropped rows per optimizer microstep.
                eligible = (
                    (~dropped).nonzero(as_tuple=False).flatten()
                    if adapter.training and adapter.style_dropout > 0
                    else torch.arange(latents.shape[0], device=device)
                )
                if eligible.numel() > 0:
                    take = min(reference_wrong_grad_samples, int(eligible.numel()))
                    order = torch.randperm(
                        eligible.numel(), device=device, generator=generator
                    )[:take]
                    wrong_reference_indices = eligible[order]
                    if step <= int(loss_config.get("debug_autograd_steps", 0)):
                        adapter.__dict__["_debug_autograd_label"] = "wrong_grad"
                    adapter.set_style_tokens(shuffled_tokens[wrong_reference_indices])
                    wrong_reference_prediction = anima(
                        noisy[wrong_reference_indices].unsqueeze(2),
                        timesteps[wrong_reference_indices].to(latents.dtype),
                        context=conditioning[wrong_reference_indices],
                        padding_mask=padding_mask[wrong_reference_indices],
                        target_input_ids=None,
                    ).squeeze(2).float()
                    adapter.clear_style_tokens()
                    adapter.__dict__.pop("_debug_autograd_label", None)
            else:
                # Legacy detached comparison. This is valid for measurement
                # and ranking, but not sufficient to isolate a reference-
                # specific gradient because shared parameters can still learn
                # a common residual on the correct branch.
                with _uncached_no_grad_autocast(device):
                    if step <= int(loss_config.get("debug_autograd_steps", 0)):
                        adapter.__dict__["_debug_autograd_label"] = "wrong_no_grad"
                    adapter.set_style_tokens(shuffled_tokens)
                    wrong_reference_prediction = anima(
                        noisy.unsqueeze(2), timesteps.to(latents.dtype), context=conditioning,
                        padding_mask=padding_mask, target_input_ids=None,
                    ).squeeze(2).float()
                    adapter.clear_style_tokens()

        adapter.reset_runtime_stats()
        if step <= int(loss_config.get("debug_autograd_steps", 0)):
            adapter.__dict__["_debug_autograd_label"] = "correct_grad"
        adapter.set_style_tokens(style_tokens)
        prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype), context=conditioning,
            padding_mask=padding_mask, target_input_ids=None,
        ).squeeze(2)
        adapter.__dict__.pop("_debug_autograd_label", None)
        prediction = prediction.float()
        target_velocity = (noise - latents).float()
        flow_loss = F.mse_loss(prediction, target_velocity)
        exact_self_losses = {
            key: flow_loss.new_zeros(())
            for key in (
                "normalized_mse", "normalized_huber", "direction", "log_rms", "x0"
            )
        }
        exact_self_residual_weight = float(
            loss_config.get("exact_self_residual_weight", 0.0)
        )
        if (
            bypass_prediction is not None
            and exact_self_supervision_active
        ):
            exact_self_losses = _exact_self_residual_losses(
                prediction[exact_self_mask], bypass_prediction[exact_self_mask],
                target_velocity[exact_self_mask], noisy.float()[exact_self_mask],
                latents.float()[exact_self_mask], sigma.float()[exact_self_mask],
                scale_floor=float(loss_config.get("exact_self_scale_floor", 1e-3)),
                huber_beta=float(loss_config.get("exact_self_huber_beta", 0.1)),
            )
        reference_rank_loss = flow_loss.new_zeros(())
        reference_rank_advantage = flow_loss.new_full((), float("nan"))
        reference_direction_loss = flow_loss.new_zeros(())
        reference_residual_mse_loss = flow_loss.new_zeros(())
        if wrong_reference_prediction is not None and bypass_prediction is not None:
            if wrong_reference_indices is not None:
                correct_for_reference = prediction[wrong_reference_indices]
                wrong_for_reference = wrong_reference_prediction
                bypass_for_reference = bypass_prediction[wrong_reference_indices]
                target_for_reference = target_velocity[wrong_reference_indices]
                wrong_has_grad = True
            else:
                valid = (
                    ~dropped
                    if adapter.training and adapter.style_dropout > 0
                    else torch.ones(
                        prediction.shape[0], dtype=torch.bool, device=prediction.device
                    )
                )
                correct_for_reference = prediction[valid]
                wrong_for_reference = wrong_reference_prediction[valid]
                bypass_for_reference = bypass_prediction[valid]
                target_for_reference = target_velocity[valid]
                wrong_has_grad = False
            if correct_for_reference.shape[0] > 0:
                reference_rank_loss, reference_rank_advantage = _reference_flow_rank_loss(
                    correct_for_reference,
                    wrong_for_reference,
                    bypass_for_reference,
                    target_for_reference,
                    margin=float(loss_config.get("style_reference_rank_margin", 0.005)),
                )
                if reference_direction_weight > 0:
                    reference_direction_loss = _reference_flow_direction_loss(
                        correct_for_reference,
                        wrong_for_reference,
                        target_for_reference,
                        epsilon=float(
                            loss_config.get("style_reference_direction_epsilon", 0.05)
                        ),
                        wrong_has_grad=wrong_has_grad,
                    )
                if reference_residual_mse_weight > 0:
                    if not wrong_has_grad:
                        raise RuntimeError(
                            "style_reference_residual_mse_weight requires "
                            "style_reference_wrong_grad_samples > 0"
                        )
                    reference_residual_mse_loss = (
                        _reference_flow_residual_mse_loss(
                            correct_for_reference,
                            wrong_for_reference,
                            target_for_reference,
                            scale_floor=float(
                                loss_config.get(
                                    "style_reference_residual_scale_floor", 1e-3
                                )
                            ),
                        )
                    )

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
        flow_direction_cosine = flow_loss.new_full((), float("nan"))
        flow_desired_projection = flow_loss.new_full((), float("nan"))
        flow_delta_to_desired_ratio = flow_loss.new_full((), float("nan"))
        flow_orthogonal_to_desired_ratio = flow_loss.new_full((), float("nan"))
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
                residual_metrics = _per_sample_flow_residual_metrics(
                    prediction[valid], bypass_prediction[valid], target_velocity[valid]
                )
                flow_direction_cosine = residual_metrics["direction_cosine"].mean()
                flow_desired_projection = residual_metrics["desired_projection"].mean()
                flow_delta_to_desired_ratio = (
                    residual_metrics["delta_rms"]
                    / residual_metrics["desired_rms"].clamp_min(1e-8)
                ).mean()
                flow_orthogonal_to_desired_ratio = residual_metrics[
                    "orthogonal_to_desired_ratio"
                ].mean()
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
                    contrastive_style_tokens, target_style_tokens, temperature
                )
            if kv_weight > 0:
                cross_attentions = [block.cross_attn for block in anima.blocks]
                reference_signature = adapter.projected_signature(
                    contrastive_style_tokens, cross_attentions
                )
                target_signature = adapter.projected_signature(
                    target_style_tokens, cross_attentions
                )
                kv_contrastive = _symmetric_style_contrastive_loss(
                    reference_signature[:, None], target_signature[:, None], temperature
                )
        flow_weight = float(loss_config.get("style_flow_loss_weight", 1.0))
        loss = (
            flow_weight * flow_loss
            + exact_self_fraction
            * float(loss_config.get("exact_self_residual_mse_weight", 0.0))
            * exact_self_losses["normalized_mse"]
            + exact_self_fraction * exact_self_residual_weight
            * exact_self_losses["normalized_huber"]
            + float(loss_config.get("exact_self_direction_weight", 0.0))
            * exact_self_fraction * exact_self_losses["direction"]
            + float(loss_config.get("exact_self_log_rms_weight", 0.0))
            * exact_self_fraction * exact_self_losses["log_rms"]
            + float(loss_config.get("exact_self_x0_weight", 0.0))
            * exact_self_fraction * exact_self_losses["x0"]
            + oracle_weight * oracle_distill_loss
            + magnitude_weight * magnitude_loss
            + direction_weight * direction_loss
            + reference_rank_weight * reference_rank_loss
            + reference_direction_weight * reference_direction_loss
            + reference_residual_mse_weight * reference_residual_mse_loss
            + token_weight * token_contrastive
            + kv_weight * kv_contrastive
            + resampler_auxiliary_weight
            * (
                resampler_reconstruction
                + float(loss_config.get("resampler_joint_prototype_weight", 0.13))
                * resampler_joint_prototype
                + float(loss_config.get("resampler_slot_prototype_weight", 0.02))
                * resampler_slot_prototype
                + float(loss_config.get("resampler_diversity_weight", 0.01))
                * resampler_diversity
            )
        )
    if not collect_details:
        adapter.reset_runtime_stats()
        return loss, {}
    return loss, {
        "references": int(reference_mask.sum()),
        "latent_shape": list(latents.shape),
        "flow_loss": float(flow_loss.detach()),
        "exact_self_residual_loss": float(
            exact_self_losses["normalized_huber"].detach()
        ),
        "exact_self_residual_mse_loss": float(
            exact_self_losses["normalized_mse"].detach()
        ),
        "exact_self_direction_loss": float(exact_self_losses["direction"].detach()),
        "exact_self_log_rms_loss": float(exact_self_losses["log_rms"].detach()),
        "predicted_x0_loss": float(exact_self_losses["x0"].detach()),
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
        "bypass_measured": bypass_prediction is not None,
        "curriculum_phase": str(curriculum["phase"]),
        "curriculum_gate_only": bool(curriculum["gate_only"]),
        "oracle_distill_weight": oracle_weight,
        "oracle_distill_applied": oracle_distill_applied,
        "oracle_distill_loss": float(oracle_distill_loss.detach()),
        "style_auxiliary": auxiliary,
        "timestep_mean": float(timesteps.mean().detach()),
        "style_magnitude_floor": magnitude_floor,
        "target_reference_probability": target_probability,
        "target_reference_fraction": float(target_included.float().mean()),
        "exact_self_supervision_fraction": float(exact_self_fraction.detach()),
        "style_dropout_fraction": float(dropped.float().mean().detach()),
        "aggregator_trainable": any(
            parameter.requires_grad for parameter in adapter.aggregator.parameters()
        ),
        "resampler_trainable": resampler_trainable,
        "resampler_reconstruction": float(resampler_reconstruction.detach()),
        "resampler_joint_prototype": float(resampler_joint_prototype.detach()),
        "resampler_slot_prototype": float(resampler_slot_prototype.detach()),
        "resampler_diversity": float(resampler_diversity.detach()),
        "style_output_ratio": float(output_ratio.detach()),
        "style_flow_direction_cosine": float(flow_direction_cosine.detach()),
        "style_flow_desired_projection": float(flow_desired_projection.detach()),
        "style_flow_delta_to_desired_ratio": float(
            flow_delta_to_desired_ratio.detach()
        ),
        "style_flow_orthogonal_to_desired_ratio": float(
            flow_orthogonal_to_desired_ratio.detach()
        ),
        "style_magnitude_loss": float(magnitude_loss.detach()),
        "style_flow_direction_multiplier": direction_multiplier,
        "style_flow_direction_loss": float(direction_loss.detach()),
        "style_reference_rank_loss": float(reference_rank_loss.detach()),
        "style_reference_rank_advantage": float(reference_rank_advantage.detach()),
        "style_reference_rank_applied": bool(wrong_reference_prediction is not None),
        "style_reference_direction_loss": float(reference_direction_loss.detach()),
        "style_reference_residual_mse_loss": float(
            reference_residual_mse_loss.detach()
        ),
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
    reference_mode: str = "heldout",
) -> dict[str, float]:
    anima.eval()
    adapter.eval()
    losses = []
    references = []
    base_losses = []
    paired_improvements = []
    output_ratios = []
    reference_advantages = []
    alignment_samples: dict[str, list[float]] = defaultdict(list)
    timestep_alignment: list[tuple[float, dict[str, float]]] = []
    started = time.perf_counter()
    try:
        for index in range(batches):
            batch = loader.load_step(index)
            generator = torch.Generator(device=device).manual_seed(seed + index)
            validation_config = {**(loss_config or {}), "measure_bypass": True}
            if reference_mode == "self":
                validation_config["curriculum"] = {
                    "gate_only_steps": 0,
                    "self_reference_steps": max(1, step),
                }
            elif reference_mode == "heldout":
                validation_config["curriculum"] = {}
            elif reference_mode != "curriculum":
                raise ValueError(f"Unknown validation reference mode: {reference_mode}")
            loss, details = _forward_flow_loss(
                anima, adapter, resampler, batch, device, generator=generator,
                loss_config=validation_config,
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
            reference_advantage = float(details["style_reference_rank_advantage"])
            if math.isfinite(reference_advantage):
                reference_advantages.append(reference_advantage)
            alignment = {
                key: float(details[key])
                for key in (
                    "style_flow_direction_cosine",
                    "style_flow_desired_projection",
                    "style_flow_delta_to_desired_ratio",
                    "style_flow_orthogonal_to_desired_ratio",
                )
            }
            for key, value in alignment.items():
                if math.isfinite(value):
                    alignment_samples[key].append(value)
            timestep_alignment.append((float(details["timestep_mean"]), alignment))
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
    summary = {
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
    for key, values in alignment_samples.items():
        summary[key] = sum(values) / len(values)
    if reference_advantages:
        summary["correct_vs_wrong_advantage"] = (
            sum(reference_advantages) / len(reference_advantages)
        )
        summary["correct_vs_wrong_positive_fraction"] = sum(
            value > 0 for value in reference_advantages
        ) / len(reference_advantages)
        summary["correct_vs_wrong_samples"] = float(len(reference_advantages))
    edges = list((loss_config or {}).get(
        "validation_timestep_edges", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ))
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = [
            metrics for timestep, metrics in timestep_alignment
            if lower <= timestep < upper or (upper == edges[-1] and timestep == upper)
        ]
        if not selected:
            continue
        label = f"t{lower:.1f}_{upper:.1f}".replace(".", "p")
        for key in alignment_samples:
            values = [metrics[key] for metrics in selected if math.isfinite(metrics[key])]
            if values:
                summary[f"{label}/{key}"] = sum(values) / len(values)
        summary[f"{label}/samples"] = float(len(selected))
    return summary


@contextmanager
def _use_same_q_alpha_blocks(adapter: nn.Module, block_indices: Iterable[int]):
    """Temporarily retain calibrated same-Q alpha only for selected blocks."""
    alpha = getattr(adapter, "alpha", None)
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 1:
        raise TypeError("Block alpha ablation requires a same-Q style adapter")
    selected = sorted({int(index) for index in block_indices})
    if any(index < 0 or index >= alpha.numel() for index in selected):
        raise ValueError(
            f"Block indices must be in [0, {alpha.numel() - 1}], got {selected}"
        )
    original = alpha.detach().clone()
    mask = torch.zeros_like(alpha)
    if selected:
        mask[selected] = 1
    with torch.no_grad():
        alpha.copy_(original * mask)
    try:
        yield
    finally:
        with torch.no_grad():
            alpha.copy_(original)


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
    sources: list[tuple[str, int]] | None = None,
) -> Image.Image:
    episode = batch["episodes"][0]
    if sources is None:
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
    episode_index: int | None = None,
    batch_override: dict[str, Any] | None = None,
    batch_row: int = 0,
    sample_group: str | None = None,
    sample_seed: int | None = None,
) -> tuple[Path, nn.Module, float]:
    sample_cfg = config["style_transfer"]["sampling"]
    started = time.perf_counter()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    anima.eval()
    adapter.eval()
    episode_number = (
        int(sample_cfg.get("episode", 0))
        if episode_index is None
        else int(episode_index)
    )
    batch = (
        loader.load_step(episode_number)
        if batch_override is None
        else _slice_exact_self_batch(batch_override, batch_row)
    )
    episode = batch["episodes"][0]
    sheet_sources: list[tuple[str, int]] = [("target", episode.target_id)]
    if reference_mode == "self":
        references = _encode_target_tokens(resampler, batch, device)[:, None]
        reference_mask = torch.ones(
            references.shape[:2], dtype=torch.bool, device=device
        )
        sheet_sources.append(("exact target", episode.target_id))
    elif reference_mode == "heldout":
        references = _encode_reference_tokens(resampler, batch, device)
        reference_mask = batch["reference_mask"].to(device, non_blocking=True)
        sheet_sources.extend(
            (f"ref {index + 1}", image_id)
            for index, image_id in enumerate(episode.reference_ids[:4])
        )
    elif reference_mode in {"wrong_artist", "mixed"}:
        if len(batch["episodes"]) < 2:
            raise ValueError(f"{reference_mode} sampling requires a loader batch size of at least 2")
        heldout_references = _encode_reference_tokens(resampler, batch, device)
        heldout_mask = batch["reference_mask"].to(device, non_blocking=True)
        wrong_references = heldout_references.roll(1, dims=0)
        wrong_mask = heldout_mask.roll(1, dims=0)
        donor = batch["episodes"][-1]
        if reference_mode == "wrong_artist":
            references, reference_mask = wrong_references, wrong_mask
            sheet_sources.extend(
                (f"wrong {index + 1}", image_id)
                for index, image_id in enumerate(donor.reference_ids[:4])
            )
        else:
            references = torch.cat((heldout_references, wrong_references), dim=1)
            reference_mask = torch.cat((heldout_mask, wrong_mask), dim=1)
            sheet_sources.extend(
                (f"right {index + 1}", image_id)
                for index, image_id in enumerate(episode.reference_ids[:2])
            )
            sheet_sources.extend(
                (f"wrong {index + 1}", image_id)
                for index, image_id in enumerate(donor.reference_ids[:2])
            )
    elif reference_mode in {"null", "bypass"}:
        references = None
        reference_mask = None
    else:
        raise ValueError(f"Unknown sample reference mode: {reference_mode}")
    if references is None:
        positive_style = adapter.unconditional(1)
    else:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            positive_style = adapter.aggregate(references, reference_mask)[:1]
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
    generator = torch.Generator(device="cpu").manual_seed(
        int(sample_cfg.get("seed", 20260811))
        if sample_seed is None
        else int(sample_seed)
    )
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
            styled_x = denoise(with_style=reference_mode != "bypass")
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
    if sample_group:
        sample_dir = sample_dir / sample_group
    sample_dir.mkdir(parents=True, exist_ok=True)
    cfg_label = f"{style_scale:g}".replace(".", "p")
    episode_label = "" if episode_index is None else f"-episode-{episode_number:05d}"
    raw_path = sample_dir / (
        f"step-{step:07d}{episode_label}-{reference_mode}-style-cfg-{cfg_label}.png"
    )
    sheet_path = sample_dir / (
        f"step-{step:07d}{episode_label}-{reference_mode}-style-cfg-{cfg_label}-sheet.png"
    )
    generated.save(raw_path)
    base_generated.save(sample_dir / f"step-{step:07d}{episode_label}-base.png")
    to_image(target_decoded[0]).save(
        sample_dir / f"step-{step:07d}{episode_label}-cached-target.png"
    )
    _make_sample_sheet(
        generated,
        loader,
        batch,
        base_generated=base_generated,
        generated_label=(
            f"styled CFG {style_scale:g} ({reference_mode}) — "
            f"{batch['episodes'][0].style_id}"
        ),
        sources=sheet_sources,
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


@torch.no_grad()
def _sample_style_adapter_batch(
    anima: nn.Module,
    adapter: SharedLowRankStyleAdapter,
    resampler: nn.Module,
    requests: list[tuple[str, ProductionStyleLoader, int, int]],
    config: dict[str, Any],
    destination: Path,
    output: Path,
    device: str,
    step: int,
    vae: nn.Module | None,
    *,
    reference_mode: str,
) -> tuple[list[tuple[str, Path]], nn.Module, float]:
    """Render the fixed panel in GPU batches while keeping VAE resident."""
    sample_cfg = config["style_transfer"]["sampling"]
    started = time.perf_counter()
    anima.eval()
    adapter.eval()
    batches = [loader.load_step(episode_index) for _, loader, episode_index, _ in requests]
    positive_text = torch.cat(
        [batch["conditioning"][:1] for batch in batches]
    ).to(device, dtype=torch.bfloat16)
    positive_styles = []
    sources_by_request = []
    for batch in batches:
        episode = batch["episodes"][0]
        sources = [("target", episode.target_id)]
        if reference_mode == "self":
            references = _encode_target_tokens(resampler, batch, device)[:, None]
            mask = torch.ones(references.shape[:2], dtype=torch.bool, device=device)
            sources.append(("exact target", episode.target_id))
        else:
            references = _encode_reference_tokens(resampler, batch, device)
            mask = batch["reference_mask"].to(device, non_blocking=True)
            sources.extend(
                (f"ref {index + 1}", image_id)
                for index, image_id in enumerate(episode.reference_ids[:4])
            )
        positive_styles.append(adapter.aggregate(references, mask)[:1])
        sources_by_request.append(sources)
    positive_style = torch.cat(positive_styles)
    batch_size = len(requests)
    first_loader = requests[0][1]
    null_text = load_file(
        first_loader.text_root / "null_conditioning.safetensors", device="cpu"
    )["empty_prompt"]
    if null_text.ndim == 2:
        null_text = null_text.unsqueeze(0)
    null_text = _pad_text_conditions(
        [null_text[0]] * batch_size, first_loader.text_conditioning_length
    ).to(device, dtype=torch.bfloat16)
    null_style = adapter.unconditional(batch_size)
    height = int(sample_cfg.get("height", 768))
    width = int(sample_cfg.get("width", 768))
    latent_h, latent_w = height // 8, width // 8
    noises = []
    for _, _, _, sample_seed in requests:
        generator = torch.Generator(device="cpu").manual_seed(sample_seed)
        noises.append(
            torch.randn(
                1, 16, 1, latent_h, latent_w,
                generator=generator, dtype=torch.float32,
            )
        )
    initial_noise = torch.cat(noises).to(device=device, dtype=torch.bfloat16)
    steps = int(sample_cfg.get("steps", 30))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.bfloat16)
    shift = float(sample_cfg.get("flow_shift", 3.0))
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    padding_mask = torch.zeros(
        batch_size, 1, latent_h, latent_w, device=device, dtype=torch.bfloat16
    )
    text_scale = float(sample_cfg.get("text_cfg", 4.0))
    style_scale = float(sample_cfg.get("style_cfg", 1.0))

    def predict(x, text, style, timestep):
        adapter.set_style_tokens(style)
        return anima(
            x, timestep.expand(batch_size), context=text,
            padding_mask=padding_mask, target_input_ids=None,
        ).float()

    def denoise(with_style: bool):
        x = initial_noise.clone()
        for index in range(steps):
            timestep = sigmas[index].to(torch.bfloat16)
            if with_style:
                base = predict(x, null_text, null_style, timestep)
                text_only = predict(x, positive_text, null_style, timestep)
                full = predict(x, positive_text, positive_style, timestep)
                velocity = (
                    base + text_scale * (text_only - base)
                    + style_scale * (full - text_only)
                )
            else:
                adapter.clear_style_tokens()
                base = anima(
                    x, timestep.expand(batch_size), context=null_text,
                    padding_mask=padding_mask, target_input_ids=None,
                ).float()
                text_only = anima(
                    x, timestep.expand(batch_size), context=positive_text,
                    padding_mask=padding_mask, target_input_ids=None,
                ).float()
                velocity = base + text_scale * (text_only - base)
            x = (
                x.float() + velocity
                * (sigmas[index + 1] - sigmas[index]).float()
            ).to(torch.bfloat16)
        return x

    try:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            base_x = denoise(False)
            styled_x = denoise(True)
    finally:
        adapter.clear_style_tokens()
    if not torch.isfinite(base_x).all() or not torch.isfinite(styled_x).all():
        raise FloatingPointError("Non-finite latent in batched qualitative sampling")
    if vae is None:
        vae = _load_sampling_vae(config, destination)
    vae.to(device=device, dtype=torch.bfloat16)
    generated_latents = torch.cat((base_x, styled_x), dim=0)
    vae_batch_size = max(1, int(sample_cfg.get("vae_batch_size", 4)))
    decoded = torch.cat([
        vae.decode_to_pixels(generated_latents[offset : offset + vae_batch_size]).float()
        for offset in range(0, generated_latents.shape[0], vae_batch_size)
    ])
    # Cached targets preserve their original aspect buckets, so they cannot be
    # stacked. They are display-only and cheap relative to 30-step generation.
    target_decoded = [
        vae.decode_to_pixels(
            batch["latents"][:1].to(
                device=device, dtype=torch.bfloat16
            ).unsqueeze(2)
        ).float()[0]
        for batch in batches
    ]

    def to_image(value):
        if value.ndim == 4:
            value = value[:, 0]
        pixels = (
            (value.clamp(-1, 1) + 1) * 127.5
        ).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)

    records = []
    cfg_label = f"{style_scale:g}".replace(".", "p")
    for index, ((split_name, loader, episode_index, _), batch, sources) in enumerate(
        zip(requests, batches, sources_by_request, strict=True)
    ):
        sample_dir = output / "samples" / split_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        episode_label = f"-episode-{episode_index:05d}"
        raw_path = sample_dir / (
            f"step-{step:07d}{episode_label}-{reference_mode}-style-cfg-{cfg_label}.png"
        )
        sheet_path = sample_dir / (
            f"step-{step:07d}{episode_label}-{reference_mode}-style-cfg-{cfg_label}-sheet.png"
        )
        base_image = to_image(decoded[index])
        styled_image = to_image(decoded[batch_size + index])
        styled_image.save(raw_path)
        base_image.save(sample_dir / f"step-{step:07d}{episode_label}-base.png")
        to_image(target_decoded[index]).save(
            sample_dir / f"step-{step:07d}{episode_label}-cached-target.png"
        )
        _make_sample_sheet(
            styled_image, loader, batch, base_generated=base_image,
            generated_label=(
                f"styled CFG {style_scale:g} ({reference_mode}) — "
                f"{batch['episodes'][0].style_id}"
            ),
            sources=sources,
        ).save(sheet_path)
        records.append((split_name, sheet_path))
    return records, vae, time.perf_counter() - started


def _select_distinct_style_episode_indices(
    loader: ProductionStyleLoader, count: int
) -> list[int]:
    """Choose a deterministic, fixed panel of distinct artists for sampling."""
    selected: list[int] = []
    seen: set[str] = set()
    limit = max(1000, count * 100)
    for episode_index in range(limit):
        episode = loader.episodes_for_step(episode_index)[0]
        if episode.style_id in seen:
            continue
        selected.append(episode_index)
        seen.add(episode.style_id)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Could select only {len(selected)}/{count} distinct artists")


def sample_style_checkpoint(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Render one or more validation artists from an explicit saved checkpoint."""
    cfg = config["style_transfer"]
    device = str(cfg["training"].get("device", "cuda"))
    loader_cfg = {**cfg["loader"], "split": "validation", "batch_size": 1}
    loader_cfg["seed"] = int(cfg.get("seed", 20260811)) ^ 0x51A7
    loader = ProductionStyleLoader(destination, loader_cfg)
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    adapter = _create_and_attach_style_adapter(
        anima, cfg["adapter"], device, dtype=torch.bfloat16
    )
    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    sample_cfg = dict(cfg.get("sampling", {}))
    checkpoint_path = Path(str(sample_cfg.get("checkpoint", "training_state.pt")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = output / checkpoint_path
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _load_adapter_checkpoint(adapter, state)
    if "resampler" in state:
        resampler.load_state_dict(state["resampler"])

    step = int(state["step"])
    episodes = [
        int(value)
        for value in sample_cfg.get("episodes", [sample_cfg.get("episode", 0)])
    ]
    reference_modes = [
        str(value) for value in sample_cfg.get("reference_modes", ["heldout"])
    ]
    vae = None
    sheets: list[dict[str, Any]] = []
    elapsed_total = 0.0
    for episode_index in episodes:
        for reference_mode in reference_modes:
            sheet, vae, elapsed = _sample_style_adapter(
                anima,
                adapter,
                resampler,
                loader,
                config,
                destination,
                output,
                device,
                step,
                vae,
                reference_mode=reference_mode,
                episode_index=episode_index,
            )
            sheets.append(
                {
                    "episode": episode_index,
                    "reference_mode": reference_mode,
                    "sheet": str(sheet),
                    "elapsed_s": elapsed,
                }
            )
            elapsed_total += elapsed
    return {
        "step": step,
        "checkpoint": str(checkpoint_path),
        "sheets": sheets,
        "elapsed_s": elapsed_total,
    }


def _pixel_distance(first: Path, second: Path) -> dict[str, float]:
    """Report normalized pixel distances for paired deterministic samples."""
    with Image.open(first) as first_image, Image.open(second) as second_image:
        first_tensor = torch.frombuffer(
            bytearray(first_image.convert("RGB").tobytes()), dtype=torch.uint8
        ).float()
        second_tensor = torch.frombuffer(
            bytearray(second_image.convert("RGB").tobytes()), dtype=torch.uint8
        ).float()
    if first_tensor.shape != second_tensor.shape:
        raise ValueError(f"Cannot compare differently sized samples: {first} and {second}")
    delta = (first_tensor - second_tensor) / 255.0
    return {
        "pixel_mae": float(delta.abs().mean()),
        "pixel_rmse": float(delta.square().mean().sqrt()),
    }


def _write_comparison_grid(
    rows: list[tuple[str, list[tuple[str, Path]]]], output: Path, *, cell_size: int = 256
) -> None:
    if not rows:
        return
    columns = max(len(images) for _, images in rows)
    label_height = 24
    canvas = Image.new(
        "RGB", (columns * cell_size, len(rows) * (cell_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_label, images) in enumerate(rows):
        top = row_index * (cell_size + label_height)
        for column_index, (column_label, path) in enumerate(images):
            with Image.open(path) as source:
                thumbnail = ImageOps.fit(source.convert("RGB"), (cell_size, cell_size))
            left = column_index * cell_size
            canvas.paste(thumbnail, (left, top + label_height))
            draw.text((left + 4, top + 4), f"{row_label} | {column_label}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.no_grad()
def compare_style_checkpoint_samples(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Generate deterministic fixed-panel and reference/CFG controlled comparisons."""
    cfg = config["style_transfer"]
    comparison_cfg = dict(cfg.get("comparison", {}))
    device = str(cfg["training"].get("device", "cuda"))
    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    checkpoint = Path(str(comparison_cfg.get("checkpoint", "training_state.pt")))
    if not checkpoint.is_absolute():
        candidate = output / checkpoint
        checkpoint = candidate if candidate.exists() else destination / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    adapter = _create_and_attach_style_adapter(
        anima, cfg["adapter"], device, dtype=torch.bfloat16
    )
    _load_adapter_checkpoint(adapter, state)
    if "resampler" in state:
        resampler.load_state_dict(state["resampler"])
    adapter.eval()

    step = int(state["step"])
    comparison_root = output / "sample_comparison" / f"step-{step:07d}"
    comparison_root.mkdir(parents=True, exist_ok=True)
    sample_config = copy.deepcopy(config)
    sampling_cfg = sample_config["style_transfer"]["sampling"]
    sampling_cfg["width"] = int(comparison_cfg.get("width", sampling_cfg.get("width", 768)))
    sampling_cfg["height"] = int(comparison_cfg.get("height", sampling_cfg.get("height", 768)))
    sampling_cfg["steps"] = int(comparison_cfg.get("steps", sampling_cfg.get("steps", 30)))
    sampling_cfg["seed"] = int(comparison_cfg.get("seed", sampling_cfg.get("seed", 20260811)))

    loader_seed = int(cfg.get("seed", 20260811)) ^ 0x51A7
    loaders: dict[str, ProductionStyleLoader] = {}
    for split in ("train", "validation"):
        loader_cfg = {**cfg["loader"], "split": split, "batch_size": 1, "seed": loader_seed}
        loaders[split] = ProductionStyleLoader(destination, loader_cfg)

    panel_path = output / "sample_panel.json"
    with panel_path.open("r", encoding="utf-8") as handle:
        panel = json.load(handle)
    fixed_cfgs = [float(value) for value in comparison_cfg.get("fixed_style_cfgs", [1.0, 4.0])]
    fixed_paths: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    vae = None
    for style_cfg in fixed_cfgs:
        sampling_cfg["style_cfg"] = style_cfg
        for split, seed_offset in (("train", 0), ("validation", 100_000)):
            entries = panel[split]
            requests = [
                (
                    split,
                    loaders[split],
                    int(entry["episode"]),
                    sampling_cfg["seed"] + seed_offset + index,
                )
                for index, entry in enumerate(entries)
            ]
            records, vae, _ = _sample_style_adapter_batch(
                anima, adapter, resampler, requests, sample_config, destination,
                comparison_root, device, step, vae, reference_mode="self",
            )
            for entry, (_, sheet) in zip(entries, records, strict=True):
                raw = Path(str(sheet).replace("-sheet.png", ".png"))
                key = f"episode-{int(entry['episode']):05d}"
                fixed_paths[split].setdefault(key, {})[f"cfg-{style_cfg:g}"] = str(raw)
                fixed_paths[split][key]["sheet"] = str(sheet)
    if vae is not None:
        vae.to("cpu")

    controlled_split = str(comparison_cfg.get("controlled_split", "validation"))
    controlled_batch_size = max(2, int(comparison_cfg.get("controlled_batch_size", 4)))
    controlled_loader = ProductionStyleLoader(
        destination,
        {
            **cfg["loader"],
            "split": controlled_split,
            "batch_size": controlled_batch_size,
            "seed": loader_seed,
        },
    )
    controlled_episode = int(comparison_cfg.get("controlled_episode", 0))
    controlled_batch = controlled_loader.load_step(controlled_episode)
    controlled_target = controlled_batch["episodes"][0]
    controlled_donor = controlled_batch["episodes"][-1]
    controlled_cfgs = [
        float(value) for value in comparison_cfg.get("controlled_style_cfgs", [1.0, 2.0, 4.0])
    ]
    controlled_modes = [
        "null" if value is None else str(value)
        for value in comparison_cfg.get(
            "controlled_reference_modes",
            ["bypass", "null", "wrong_artist", "heldout", "mixed", "self"],
        )
    ]
    controlled_paths: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    vae = None
    for style_cfg in controlled_cfgs:
        sampling_cfg["style_cfg"] = style_cfg
        for mode in controlled_modes:
            sheet, vae, _ = _sample_style_adapter(
                anima, adapter, resampler, controlled_loader, sample_config,
                destination, comparison_root, device, step, vae,
                reference_mode=mode, episode_index=controlled_episode,
                sample_group=f"controlled/{mode}",
                sample_seed=sampling_cfg["seed"],
            )
            raw = Path(str(sheet).replace("-sheet.png", ".png"))
            controlled_paths[mode][f"cfg-{style_cfg:g}"] = {
                "image": str(raw), "sheet": str(sheet)
            }
    if vae is not None:
        vae.to("cpu")

    fixed_metrics: dict[str, Any] = {}
    fixed_grid_rows = []
    for split, episodes in fixed_paths.items():
        for episode, paths in episodes.items():
            first_path = Path(next(iter(paths[key] for key in paths if key.startswith("cfg-"))))
            base = first_path.parent / first_path.name.replace(
                f"-self-style-cfg-{next(key[4:] for key in paths if key.startswith('cfg-'))}.png",
                "-base.png",
            )
            target = base.parent / base.name.replace("-base.png", "-cached-target.png")
            images = [("target/reference", target), ("base", base)]
            row_metrics: dict[str, Any] = {}
            for key in sorted(value for value in paths if value.startswith("cfg-")):
                path = Path(paths[key])
                images.append((key, path))
                row_metrics[f"{key}_vs_base"] = _pixel_distance(path, base)
            cfg_keys = sorted(value for value in paths if value.startswith("cfg-"))
            if len(cfg_keys) >= 2:
                row_metrics[f"{cfg_keys[0]}_vs_{cfg_keys[-1]}"] = _pixel_distance(
                    Path(paths[cfg_keys[0]]), Path(paths[cfg_keys[-1]])
                )
            fixed_metrics[f"{split}/{episode}"] = row_metrics
            fixed_grid_rows.append((f"{split}/{episode}", images))

    controlled_metrics: dict[str, Any] = {}
    controlled_grid_rows = []
    base_path = Path(controlled_paths["bypass"][f"cfg-{controlled_cfgs[0]:g}"]["image"])
    for mode in controlled_modes:
        images = []
        mode_metrics: dict[str, Any] = {}
        for style_cfg in controlled_cfgs:
            key = f"cfg-{style_cfg:g}"
            path = Path(controlled_paths[mode][key]["image"])
            images.append((key, path))
            mode_metrics[f"{key}_vs_base"] = _pixel_distance(path, base_path)
        mode_metrics["cfg_min_vs_max"] = _pixel_distance(images[0][1], images[-1][1])
        controlled_metrics[mode] = mode_metrics
        controlled_grid_rows.append((mode, images))
    for style_cfg in controlled_cfgs:
        key = f"cfg-{style_cfg:g}"
        for first_mode, second_mode in (
            ("self", "heldout"), ("self", "wrong_artist"),
            ("heldout", "wrong_artist"), ("heldout", "mixed"),
        ):
            if first_mode in controlled_paths and second_mode in controlled_paths:
                controlled_metrics[f"{first_mode}_vs_{second_mode}/{key}"] = _pixel_distance(
                    Path(controlled_paths[first_mode][key]["image"]),
                    Path(controlled_paths[second_mode][key]["image"]),
                )

    fixed_grid = comparison_root / "fixed-panel-cfg-grid.png"
    controlled_grid = comparison_root / "controlled-reference-cfg-grid.png"
    _write_comparison_grid(fixed_grid_rows, fixed_grid)
    _write_comparison_grid(controlled_grid_rows, controlled_grid)
    result = {
        "step": step,
        "checkpoint": str(checkpoint.resolve()),
        "fixed_panel": dict(fixed_paths),
        "fixed_metrics": fixed_metrics,
        "controlled": {
            "split": controlled_split,
            "episode": controlled_episode,
            "target_id": int(controlled_target.target_id),
            "style_id": str(controlled_target.style_id),
            "same_artist_reference_ids": [int(value) for value in controlled_target.reference_ids],
            "wrong_artist_style_id": str(controlled_donor.style_id),
            "wrong_artist_reference_ids": [int(value) for value in controlled_donor.reference_ids],
            "paths": dict(controlled_paths),
            "metrics": controlled_metrics,
        },
        "fixed_grid": str(fixed_grid),
        "controlled_grid": str(controlled_grid),
    }
    write_json(comparison_root / "summary.json", result)
    return result


def _per_sample_flow_residual_metrics(
    prediction: torch.Tensor,
    bypass: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Resolve magnitude and direction of one condition relative to frozen Anima."""
    dimensions = tuple(range(1, prediction.ndim))
    delta = prediction.float() - bypass.float()
    desired = target.float() - bypass.float()
    base_mse = desired.square().mean(dim=dimensions).clamp_min(1e-12)
    condition_mse = (prediction.float() - target.float()).square().mean(dim=dimensions)
    delta_rms = delta.square().mean(dim=dimensions).sqrt()
    bypass_rms = bypass.float().square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    desired_rms = base_mse.sqrt()
    dot = (delta * desired).mean(dim=dimensions)
    cosine = dot / (delta_rms * desired_rms).clamp_min(1e-12)
    projection = dot / base_mse
    # Decompose delta into its signed desired-axis projection and orthogonal
    # remainder. Both are normalized by desired RMS, so paired improvement is
    # exactly 2 * projection - projection**2 - orthogonal**2 per sample.
    delta_to_desired = delta_rms / desired_rms
    orthogonal_to_desired = (
        delta_to_desired.square() - projection.square()
    ).clamp_min(0).sqrt()
    return {
        "loss": condition_mse,
        "paired_improvement": (base_mse - condition_mse) / base_mse,
        "delta_to_base_ratio": delta_rms / bypass_rms,
        "direction_cosine": cosine,
        "desired_projection": projection,
        "orthogonal_to_desired_ratio": orthogonal_to_desired,
        "delta_rms": delta_rms,
        "desired_rms": desired_rms,
    }


def _per_sample_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, first.ndim))
    first = first.float()
    second = second.float()
    numerator = (first * second).mean(dim=dimensions)
    denominator = (
        first.square().mean(dim=dimensions).sqrt()
        * second.square().mean(dim=dimensions).sqrt()
    ).clamp_min(1e-12)
    return numerator / denominator


def _per_sample_condition_comparison(
    first: torch.Tensor,
    second: torch.Tensor,
    bypass: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare two conditions directly on one identical flow problem.

    Positive ``first_advantage`` means the first condition has lower flow MSE.
    Direct condition differences are reported alongside the frozen-model error
    scale so a reference effect can be separated from ordinary flow magnitude.
    """
    dimensions = tuple(range(1, first.ndim))
    first = first.float()
    second = second.float()
    bypass = bypass.float()
    target = target.float()
    base_mse = (bypass - target).square().mean(dim=dimensions).clamp_min(1e-12)
    first_mse = (first - target).square().mean(dim=dimensions)
    second_mse = (second - target).square().mean(dim=dimensions)
    difference_rms = (first - second).square().mean(dim=dimensions).sqrt()
    bypass_rms = bypass.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    return {
        "first_advantage": (second_mse - first_mse) / base_mse,
        "difference_to_base_ratio": difference_rms / bypass_rms,
        "difference_to_desired_ratio": difference_rms / base_mse.sqrt(),
    }


def _reference_flow_rank_loss(
    correct: torch.Tensor,
    wrong: torch.Tensor,
    bypass: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require the correct condition to beat a wrong condition.

    The caller chooses whether the wrong branch is detached. With a live wrong
    branch this becomes a contrastive ranking loss: each style is pulled toward
    its matching target and pushed away from another target in the cyclic
    batch permutation. A reference-independent update affects both branches
    equally and therefore cannot improve their MSE margin.
    """
    comparison = _per_sample_condition_comparison(correct, wrong, bypass, target)
    advantage = comparison["first_advantage"]
    return F.relu(float(margin) - advantage).mean(), advantage.mean()


def _reference_flow_direction_loss(
    correct: torch.Tensor,
    wrong: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    wrong_has_grad: bool = False,
) -> torch.Tensor:
    """Align only the reference-specific residual with the remaining error.

    When ``wrong_has_grad`` is true, the difference path differentiates both
    conditions. A reference-independent residual then has equal Jacobians and
    cancels, while the desired direction keeps the wrong prediction detached.
    This prevents the objective from being optimized through a common gate/KV
    update. The legacy detached mode remains useful for inexpensive pilots.
    """
    dimensions = tuple(range(1, correct.ndim))
    wrong_value = wrong.float()
    delta = correct.float() - (
        wrong_value if wrong_has_grad else wrong_value.detach()
    )
    desired = target.float() - wrong_value.detach()
    desired_rms = desired.square().mean(dim=dimensions).sqrt().clamp_min(1e-6)
    return _flow_direction_loss(
        delta,
        desired,
        desired_rms,
        epsilon=float(epsilon),
    )


def _reference_flow_residual_mse_loss(
    correct: torch.Tensor,
    wrong: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_floor: float = 1e-3,
) -> torch.Tensor:
    """Regress the target residual with a reference-centered gradient.

    Numerically, ``(correct - wrong) - (target - stopgrad(wrong))`` is the
    ordinary ``correct - target`` error.  Its backward path is different:
    gradients flow through both conditions, so a reference-independent
    adapter path has identical Jacobians in ``correct`` and ``wrong`` and
    cancels exactly.  The cyclic wrong-reference branch therefore preserves
    the direct flow target while making its useful gradient reference-specific.
    """
    dimensions = tuple(range(1, correct.ndim))
    wrong_value = wrong.float()
    wrong_target = wrong_value.detach()
    student = correct.float() - wrong_value
    desired = target.float() - wrong_target
    scale = desired.square().mean(
        dim=dimensions, keepdim=True
    ).sqrt().clamp_min(float(scale_floor))
    return F.mse_loss(student / scale, desired / scale)


def _summarize_scalar_samples(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean())
    if tensor.numel() <= 1:
        ci95 = 0.0
    else:
        ci95 = float(1.96 * tensor.std(unbiased=True) / math.sqrt(tensor.numel()))
    return {
        "mean": mean,
        "ci95": ci95,
        "positive_fraction": float((tensor > 0).double().mean()),
        "samples": int(tensor.numel()),
    }


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
    checkpoint_sweep = diagnostic_cfg.get("checkpoints")
    if checkpoint_sweep:
        results = []
        for checkpoint in checkpoint_sweep:
            candidate = copy.deepcopy(config)
            candidate_diagnostics = candidate["style_transfer"]["diagnostics"]
            candidate_diagnostics.pop("checkpoints", None)
            candidate_diagnostics["checkpoint"] = str(checkpoint)
            result = diagnose_style_reference_dependence(candidate, destination)
            results.append(
                {
                    "step": int(result["step"]),
                    "checkpoint": str(result["checkpoint"]),
                }
            )
        output = destination / str(
            cfg.get("output_directory", "style_transfer_training")
        )
        index = {"checkpoint_diagnostics": results}
        write_json(output / "diagnostics" / "checkpoint_comparison_index.json", index)
        return index
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
    adapter = _create_and_attach_style_adapter(
        anima, cfg["adapter"], device, dtype=torch.bfloat16
    )
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
    _load_adapter_checkpoint(adapter, state)
    if "resampler" in state:
        resampler.load_state_dict(state["resampler"])

    alpha_block_ablation: dict[str, dict[str, float]] = {}
    configured_block_groups = diagnostic_cfg.get("alpha_block_groups", {})
    if configured_block_groups:
        if not isinstance(configured_block_groups, dict):
            raise TypeError("diagnostics.alpha_block_groups must be a mapping")
        ablation_batches = int(
            diagnostic_cfg.get("alpha_block_ablation_batches", 8)
        )
        ablation_seed = int(
            diagnostic_cfg.get("alpha_block_ablation_seed", 20260811 ^ 0xA1FA)
        )
        validation_loss_config = {
            **cfg["training"],
            "timestep_sampling": "uniform",
            "style_dropout": 0.0,
        }
        for name, block_indices in configured_block_groups.items():
            with _use_same_q_alpha_blocks(adapter, block_indices):
                alpha_block_ablation[str(name)] = _validate_style_adapter(
                    anima,
                    adapter,
                    resampler,
                    loader,
                    device,
                    batches=ablation_batches,
                    seed=ablation_seed,
                    step=int(state["step"]),
                    loss_config=validation_loss_config,
                    reference_mode="self",
                )
            adapter.eval()

    records: list[dict[str, float]] = []
    batches = int(diagnostic_cfg.get("batches", 4))
    seed = int(diagnostic_cfg.get("seed", 20260811 ^ 0xD1A6))
    timestep_values = [
        float(value)
        for value in diagnostic_cfg.get(
            "timesteps", [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        )
    ]
    timestep_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    comparison_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    direct_comparison_samples: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
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
            wrong_references = references.roll(1, dims=0)
            wrong_mask = reference_mask.roll(1, dims=0)
            wrong_style = adapter.aggregate(wrong_references, wrong_mask)
            mixed_style = adapter.aggregate(
                torch.cat((references, wrong_references), dim=1),
                torch.cat((reference_mask, wrong_mask), dim=1),
            )
        null_style = adapter.unconditional(correct_style.shape[0])

        latents = batch["latents"].to(device, non_blocking=True, dtype=torch.bfloat16)
        conditioning = batch["conditioning"].to(device, non_blocking=True, dtype=torch.bfloat16)
        padding_mask = torch.zeros(
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
            device=device, dtype=latents.dtype,
        )

        def predict(
            noisy: torch.Tensor,
            timesteps: torch.Tensor,
            style: torch.Tensor | None,
            *,
            bypass: bool = False,
        ) -> torch.Tensor:
            if bypass:
                adapter.clear_style_tokens()
            else:
                adapter.set_style_tokens(style)
            try:
                bypass_context = (
                    _bypass_style_blocks(anima, adapter)
                    if bypass else nullcontext()
                )
                with bypass_context:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16,
                        enabled=device.startswith("cuda"),
                    ):
                        return anima(
                            noisy.unsqueeze(2), timesteps.to(latents.dtype),
                            context=conditioning, padding_mask=padding_mask,
                            target_input_ids=None,
                        ).squeeze(2).float()
            finally:
                adapter.clear_style_tokens()

        flat = F.normalize(correct_style.float().flatten(1), dim=1)
        self_flat = F.normalize(self_style.float().flatten(1), dim=1)
        similarities = flat @ flat.T
        off_diagonal = ~torch.eye(flat.shape[0], device=device, dtype=torch.bool)
        record = {
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
        }
        records.append(record)

        for timestep in timestep_values:
            generator = torch.Generator(device=device).manual_seed(
                seed + index * 10_007 + round(timestep * 10_000)
            )
            noise = torch.randn(
                latents.shape, device=device, dtype=latents.dtype, generator=generator
            )
            timesteps = torch.full(
                (latents.shape[0],), timestep, device=device, dtype=torch.float32
            )
            sigma = timesteps[:, None, None, None].to(latents.dtype)
            noisy = (1 - sigma) * latents + sigma * noise
            target = (noise - latents).float()
            predictions = {
                "self": predict(noisy, timesteps, self_style),
                "heldout": predict(noisy, timesteps, correct_style),
                "wrong_artist": predict(noisy, timesteps, wrong_style),
                "mixed": predict(noisy, timesteps, mixed_style),
                "null": predict(noisy, timesteps, null_style),
            }
            bypass_prediction = predict(noisy, timesteps, None, bypass=True)
            # A second identical bypass pass measures the numerical repeatability
            # floor of a full frozen-Anima evaluation on this hardware/backend.
            bypass_repeat = predict(noisy, timesteps, None, bypass=True)
            timestep_key = f"{timestep:.2f}"
            deltas = {}
            for condition, prediction in predictions.items():
                metrics = _per_sample_flow_residual_metrics(
                    prediction, bypass_prediction, target
                )
                deltas[condition] = prediction - bypass_prediction
                for metric, values in metrics.items():
                    timestep_samples[f"{timestep_key}/{condition}"][metric].extend(
                        float(value) for value in values.cpu()
                    )
            for comparison, other in (
                ("heldout_vs_self", "self"),
                ("heldout_vs_wrong", "wrong_artist"),
                ("heldout_vs_mixed", "mixed"),
                ("heldout_vs_null", "null"),
            ):
                values = _per_sample_cosine(deltas["heldout"], deltas[other])
                comparison_samples[timestep_key][comparison].extend(
                    float(value) for value in values.cpu()
                )
            for comparison, first, second in (
                ("self_vs_wrong", "self", "wrong_artist"),
                ("heldout_vs_wrong", "heldout", "wrong_artist"),
                ("heldout_vs_null", "heldout", "null"),
                ("self_vs_heldout", "self", "heldout"),
                ("bypass_repeatability", None, None),
            ):
                first_prediction = (
                    bypass_prediction if first is None else predictions[first]
                )
                second_prediction = (
                    bypass_repeat if second is None else predictions[second]
                )
                metrics = _per_sample_condition_comparison(
                    first_prediction, second_prediction, bypass_prediction, target
                )
                for metric, values in metrics.items():
                    direct_comparison_samples[timestep_key][comparison][metric].extend(
                        float(value) for value in values.cpu()
                    )

    means = {key: sum(record[key] for record in records) / len(records) for key in records[0]}
    timestep_metrics = {
        key: {
            metric: _summarize_scalar_samples(values)
            for metric, values in metrics.items()
        }
        for key, metrics in timestep_samples.items()
    }
    delta_cosines = {
        timestep: {
            name: _summarize_scalar_samples(values)
            for name, values in comparisons.items()
        }
        for timestep, comparisons in comparison_samples.items()
    }
    direct_comparisons = {
        timestep: {
            comparison: {
                metric: _summarize_scalar_samples(values)
                for metric, values in metrics.items()
            }
            for comparison, metrics in comparisons.items()
        }
        for timestep, comparisons in direct_comparison_samples.items()
    }
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "step": int(state["step"]),
        "batches": batches,
        "batch_size": loader.batch_size,
        "means": means,
        "records": records,
        "timestep_metrics": timestep_metrics,
        "condition_delta_cosines": delta_cosines,
        "direct_condition_comparisons": direct_comparisons,
        "alpha_block_ablation": alpha_block_ablation,
    }
    diagnostic_output = output / "diagnostics" / f"step-{int(state['step']):07d}"
    diagnostic_output.mkdir(parents=True, exist_ok=True)
    result_path = diagnostic_output / "reference_dependence_diagnostic.json"
    write_json(result_path, result)
    if bool(diagnostic_cfg.get("generate", True)):
        sampling_loader_cfg = {**loader_cfg, "batch_size": max(2, loader.batch_size)}
        sampling_loader = ProductionStyleLoader(destination, sampling_loader_cfg)
        vae = None
        generated_sheets = {}
        for reference_mode in (
            "bypass", "null", "wrong_artist", "heldout", "mixed", "self"
        ):
            sheet, vae, _ = _sample_style_adapter(
                anima, adapter, resampler, sampling_loader, config, destination,
                diagnostic_output, device, int(state["step"]), vae,
                reference_mode=reference_mode,
            )
            generated_sheets[reference_mode] = str(sheet)
        result["generated_sheets"] = generated_sheets
        write_json(result_path, result)
    return result


def _roll_exact_target_features(batch: dict[str, Any]) -> dict[str, Any]:
    """Keep each training target but give it another batch item's visual condition."""
    return {
        **batch,
        "target_features": {
            layer: values.roll(1, dims=0)
            for layer, values in batch["target_features"].items()
        },
        "target_feature_mask": batch["target_feature_mask"].roll(1, dims=0),
        "target_feature_shapes": batch["target_feature_shapes"][-1:]
        + batch["target_feature_shapes"][:-1],
        "target_global_features": batch["target_global_features"].roll(1, dims=0),
    }


def _slice_exact_self_batch(batch: dict[str, Any], index: int) -> dict[str, Any]:
    """Keep one target row; exact-self sampling never consumes episode references."""
    return {
        **batch,
        "episodes": [batch["episodes"][index]],
        "latents": batch["latents"][index : index + 1],
        "conditioning": batch["conditioning"][index : index + 1],
        "target_features": {
            layer: values[index : index + 1]
            for layer, values in batch["target_features"].items()
        },
        "target_feature_mask": batch["target_feature_mask"][index : index + 1],
        "target_feature_shapes": [batch["target_feature_shapes"][index]],
        "target_global_features": batch["target_global_features"][index : index + 1],
    }


def _exact_self_batch_only(batch: dict[str, Any]) -> dict[str, Any]:
    """Drop held-out reference tensors unused by target-only flow training."""
    keys = (
        "episodes", "latents", "conditioning", "target_features",
        "target_feature_mask", "target_feature_shapes", "target_global_features",
    )
    return {key: batch[key] for key in keys}


def _collect_disjoint_exact_self_batches(
    loader: ProductionStyleLoader, *, train_batches: int, validation_batches: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize fixed, target-disjoint RAM pools for a small generalization test."""
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    step = 0
    required = train_batches + validation_batches
    while len(selected) < required and step < required * 100:
        batch = loader.load_step(step)
        ids = {int(item.target_id) for item in batch["episodes"]}
        if not ids & used:
            selected.append(_exact_self_batch_only(batch))
            used.update(ids)
        step += 1
    if len(selected) != required:
        raise RuntimeError(f"Could collect only {len(selected)}/{required} disjoint batches")
    return selected[:train_batches], selected[train_batches:]


def overfit_exact_self_batch(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Test whether the adapter can memorize a fixed exact-self flow problem."""
    cfg = config["style_transfer"]
    overfit_cfg = dict(cfg.get("overfit", {}))
    training_cfg = dict(cfg["training"])
    device = str(training_cfg.get("device", "cuda"))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The exact-self overfit diagnostic requires CUDA")
    seed = int(overfit_cfg.get("seed", 20260812))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.get("allow_tf32", True))

    loader_cfg = {
        **cfg["loader"],
        "split": "train",
        "batch_size": int(overfit_cfg.get("batch_size", 3)),
    }
    loader_cfg["seed"] = seed
    loader = ProductionStyleLoader(destination, loader_cfg)
    fixed_batch = loader.load_step(int(overfit_cfg.get("episode", 0)))
    wrong_batch = _roll_exact_target_features(fixed_batch)
    unseen_batch = loader.load_step(int(overfit_cfg.get("unseen_episode", 1)))

    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    resampler.requires_grad_(False).eval()
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).train()
    optimization_counts = _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", False)),
        fuse_attention_projections=bool(training_cfg.get("fuse_attention_projections", False)),
    )
    adapter = _create_and_attach_style_adapter(anima, cfg["adapter"], device)

    source_output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    checkpoint_value = overfit_cfg.get("checkpoint")
    state: dict[str, Any] | None = None
    checkpoint: Path | None = None
    if checkpoint_value:
        checkpoint = Path(str(checkpoint_value))
        if not checkpoint.is_absolute():
            checkpoint = source_output / checkpoint
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _load_adapter_checkpoint(adapter, state)
        if "resampler" in state:
            resampler.load_state_dict(state["resampler"])
    adapter.style_dropout = 0.0

    output_parameters = adapter.output_parameters()
    gate_parameters = adapter.gate_parameters()
    special_ids = {id(value) for value in output_parameters + gate_parameters}
    representation_parameters = [
        value for value in adapter.parameters() if id(value) not in special_ids
    ]
    weight_decay = float(overfit_cfg.get("weight_decay", 0.0))
    # The connector trunk and its low-rank K/V/O deltas also contain zero-init
    # output matrices. AdamW normalizes their first sparse signal into an
    # almost learning-rate-sized update per coordinate, which overwhelmed the
    # native cross gate even during linear warmup. RAdam keeps those earliest
    # steps SGD-like while its variance estimate is unreliable.
    optimizer = torch.optim.RAdam(
        [
            {
                "params": representation_parameters,
                "lr": float(overfit_cfg.get("representation_learning_rate", 3e-4)),
                "weight_decay": weight_decay,
            },
            {
                "params": output_parameters,
                "lr": float(overfit_cfg.get("output_learning_rate", 2e-4)),
                "weight_decay": weight_decay,
            },
            {
                "params": gate_parameters,
                "lr": float(overfit_cfg.get("gate_learning_rate", 1e-4)),
                "weight_decay": 0.0,
            },
        ],
    )
    loss_config = {
        **training_cfg,
        "resampler_train_start_step": -1,
        "resampler_auxiliary_weight": 0.0,
        "style_magnitude_weight": 0.0,
        "style_flow_loss_weight": float(overfit_cfg.get("flow_loss_weight", 1.0)),
        "style_flow_direction_weight": float(overfit_cfg.get("direction_weight", 0.0)),
        "exact_self_residual_weight": float(overfit_cfg.get("residual_weight", 1.0)),
        "exact_self_direction_weight": float(overfit_cfg.get("residual_direction_weight", 0.2)),
        "exact_self_log_rms_weight": float(overfit_cfg.get("residual_log_rms_weight", 0.05)),
        "exact_self_x0_weight": float(overfit_cfg.get("x0_weight", 1.0)),
        "exact_self_scale_floor": float(overfit_cfg.get("residual_scale_floor", 1e-3)),
        "exact_self_huber_beta": float(overfit_cfg.get("residual_huber_beta", 0.1)),
        "style_token_contrastive_weight": 0.0,
        "style_kv_contrastive_weight": 0.0,
        "measure_bypass": True,
        "curriculum": {
            "gate_only_steps": 0,
            "self_reference_steps": int(overfit_cfg.get("steps", 500)) + 1,
            "target_anneal_end": int(overfit_cfg.get("steps", 500)) + 1,
            "oracle_distill_end": int(overfit_cfg.get("steps", 500)) + 1,
        },
        "oracle_distill_weight": 0.0,
    }
    fixed_noise_seed = int(overfit_cfg.get("noise_seed", seed ^ 0xF10))

    def evaluate(batch: dict[str, Any], noise_seed: int) -> dict[str, float]:
        anima.eval()
        adapter.eval()
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        with torch.no_grad():
            _, metrics = _forward_flow_loss(
                anima, adapter, resampler, batch, device,
                generator=generator, loss_config=loss_config, step=1,
            )
        return {
            key: float(metrics[key])
            for key in (
                "flow_loss", "base_flow_loss", "paired_flow_improvement",
                "style_output_ratio", "style_flow_direction_cosine",
                "style_flow_desired_projection",
                "style_flow_delta_to_desired_ratio",
                "exact_self_residual_loss", "exact_self_direction_loss",
                "exact_self_log_rms_loss", "predicted_x0_loss",
            )
        }

    fixed_noise_each_step = bool(overfit_cfg.get("fixed_noise_each_step", False))
    default_output_name = (
        "overfit_exact_self_fixed_flow"
        if fixed_noise_each_step
        else "overfit_exact_self_random_flow"
    )
    output = source_output / str(overfit_cfg.get("output_name", default_output_name))
    output.mkdir(parents=True, exist_ok=True)
    history = []
    steps = int(overfit_cfg.get("steps", 500))
    evaluate_every = int(overfit_cfg.get("evaluate_every", 25))
    max_grad_norm = float(overfit_cfg.get("max_grad_norm", 1.0))
    initial = {
        "step": 0,
        "self": evaluate(fixed_batch, fixed_noise_seed),
        "wrong": evaluate(wrong_batch, fixed_noise_seed),
        "unseen_self": evaluate(unseen_batch, fixed_noise_seed + 1),
    }
    history.append(initial)
    print(f"exact-self overfit step=0 metrics={initial}", flush=True)
    sample_config = copy.deepcopy(config)
    sample_config["style_transfer"]["sampling"].update({
        "episode": int(overfit_cfg.get("episode", 0)),
        "width": int(overfit_cfg.get("sample_width", 512)),
        "height": int(overfit_cfg.get("sample_height", 512)),
        "steps": int(overfit_cfg.get("sample_steps", 30)),
        "seed": int(overfit_cfg.get("sample_seed", fixed_noise_seed)),
        "style_cfg": 1.0,
    })
    vae = None
    initial_sample, vae, _ = _sample_style_adapter(
        anima, adapter, resampler, loader, sample_config, destination,
        output, device, 0, vae, reference_mode="self",
    )

    started = time.perf_counter()
    for step in range(1, steps + 1):
        anima.train()
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        train_noise_seed = (
            fixed_noise_seed
            if fixed_noise_each_step
            else fixed_noise_seed + step * 10_007
        )
        generator = torch.Generator(device=device).manual_seed(train_noise_seed)
        loss, _ = _forward_flow_loss(
            anima, adapter, resampler, fixed_batch, device,
            generator=generator, loss_config=loss_config, step=step,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
        optimizer.step()
        if step % evaluate_every == 0 or step == steps:
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "self": evaluate(fixed_batch, fixed_noise_seed),
                "wrong": evaluate(wrong_batch, fixed_noise_seed),
                "unseen_self": evaluate(unseen_batch, fixed_noise_seed + 1),
            }
            history.append(row)
            write_json(output / "history.json", history)
            print(f"exact-self overfit step={step} metrics={row}", flush=True)

    checkpoint_output = output / "overfit_state.pt"
    torch.save(
        {
            "step": steps,
            "source_checkpoint": str(checkpoint) if checkpoint is not None else None,
            "adapter": adapter.state_dict(),
            "resampler": resampler.state_dict(),
        },
        checkpoint_output,
    )
    final_samples = {}
    for batch_row, episode in enumerate(fixed_batch["episodes"]):
        for style_cfg in (1.0, 4.0):
            sample_config["style_transfer"]["sampling"]["style_cfg"] = style_cfg
            key = f"target_{episode.target_id}_style_cfg_{style_cfg:g}"
            final_samples[key] = str(_sample_style_adapter(
                anima, adapter, resampler, loader, sample_config, destination,
                output, device, steps, vae, reference_mode="self",
                episode_index=int(episode.target_id),
                batch_override=fixed_batch, batch_row=batch_row,
            )[0])
    randomized_evaluation: dict[str, dict[str, dict[str, float]]] = {}
    randomized_records: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in range(int(overfit_cfg.get("generalization_seeds", 8))):
        evaluation_seed = fixed_noise_seed + 1_000_003 + index * 97
        for name, batch in (
            ("self", fixed_batch),
            ("wrong", wrong_batch),
            ("unseen_self", unseen_batch),
        ):
            values = evaluate(batch, evaluation_seed)
            for metric, value in values.items():
                randomized_records[name][metric].append(value)
    randomized_evaluation = {
        name: {
            metric: _summarize_scalar_samples(values)
            for metric, values in metrics.items()
        }
        for name, metrics in randomized_records.items()
    }
    result = {
        "source_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "source_step": int(state["step"]) if state is not None else 0,
        "steps": steps,
        "fixed_targets": [item.target_id for item in fixed_batch["episodes"]],
        "evaluation_noise_seed": fixed_noise_seed,
        "fixed_noise_and_timestep_during_training": fixed_noise_each_step,
        "resampler_trainable": False,
        "losses": {
            "flow_mse_weight": float(loss_config.get("style_flow_loss_weight", 1.0)),
            "normalized_direction_weight": float(
                loss_config.get("style_flow_direction_weight", 0.0)
            ),
        },
        "optimization_counts": optimization_counts,
        "initial": history[0],
        "final": history[-1],
        "randomized_flow_evaluation": randomized_evaluation,
        "history": str(output / "history.json"),
        "checkpoint": str(checkpoint_output),
        "initial_sample": str(initial_sample),
        "final_samples": final_samples,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", result)
    return result


def sample_exact_self_overfit_checkpoint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render every memorized target from a completed exact-self checkpoint."""
    cfg = config["style_transfer"]
    overfit_cfg = dict(cfg.get("overfit", {}))
    device = str(cfg["training"].get("device", "cuda"))
    loader_cfg = {
        **cfg["loader"], "split": "train",
        "batch_size": int(overfit_cfg.get("batch_size", 3)),
        "seed": int(overfit_cfg.get("seed", 20260812)),
    }
    loader = ProductionStyleLoader(destination, loader_cfg)
    fixed_batch = loader.load_step(int(overfit_cfg.get("episode", 0)))
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    adapter = _create_and_attach_style_adapter(
        anima, cfg["adapter"], device, dtype=torch.bfloat16
    )
    output = (
        destination / str(cfg.get("output_directory", "style_transfer_training"))
        / str(overfit_cfg.get("output_name", "overfit_exact_self"))
    )
    checkpoint = output / "overfit_state.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _load_adapter_checkpoint(adapter, state)
    resampler.load_state_dict(state["resampler"])
    sample_config = copy.deepcopy(config)
    sample_config["style_transfer"]["sampling"].update({
        "width": int(overfit_cfg.get("sample_width", 512)),
        "height": int(overfit_cfg.get("sample_height", 512)),
        "steps": int(overfit_cfg.get("sample_steps", 30)),
        "seed": int(overfit_cfg.get("sample_seed", overfit_cfg.get("noise_seed", 0))),
    })
    vae = None
    samples = {}
    for batch_row, episode in enumerate(fixed_batch["episodes"]):
        for style_cfg in (1.0, 4.0):
            sample_config["style_transfer"]["sampling"]["style_cfg"] = style_cfg
            sheet, vae, _ = _sample_style_adapter(
                anima, adapter, resampler, loader, sample_config, destination,
                output, device, int(state["step"]), vae, reference_mode="self",
                episode_index=int(episode.target_id),
                batch_override=fixed_batch, batch_row=batch_row,
            )
            samples[f"target_{episode.target_id}_style_cfg_{style_cfg:g}"] = str(sheet)
    result = {"checkpoint": str(checkpoint), "step": int(state["step"]), "samples": samples}
    write_json(output / "all_target_samples.json", result)
    return result


def train_exact_self_generalization(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Learn exact-self flow on a small image pool and validate on disjoint targets."""
    cfg = config["style_transfer"]
    run_cfg = dict(cfg.get("exact_self_generalization", {}))
    training_cfg = dict(cfg["training"])
    device = str(training_cfg.get("device", "cuda"))
    seed = int(run_cfg.get("seed", 20260815))
    random.seed(seed)
    torch.manual_seed(seed)
    loader_cfg = {
        **cfg["loader"], "split": "train",
        "batch_size": int(run_cfg.get("batch_size", 3)), "seed": seed,
    }
    loader = ProductionStyleLoader(destination, loader_cfg)
    train_batches, validation_batches = _collect_disjoint_exact_self_batches(
        loader,
        train_batches=int(run_cfg.get("train_batches", 32)),
        validation_batches=int(run_cfg.get("validation_batches", 8)),
    )
    train_ids = [item.target_id for batch in train_batches for item in batch["episodes"]]
    validation_ids = [
        item.target_id for batch in validation_batches for item in batch["episodes"]
    ]
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Exact-self train and validation targets overlap")

    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    resampler.requires_grad_(False).eval()
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).train()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", False)),
        fuse_attention_projections=bool(training_cfg.get("fuse_attention_projections", False)),
    )
    adapter = _create_and_attach_style_adapter(anima, cfg["adapter"], device)
    adapter.style_dropout = 0.0
    output_parameters = adapter.output_parameters()
    gate_parameters = adapter.gate_parameters()
    special = {id(value) for value in output_parameters + gate_parameters}
    representation_parameters = [
        value for value in adapter.parameters() if id(value) not in special
    ]
    optimizer = torch.optim.RAdam([
        {"params": representation_parameters, "lr": float(run_cfg.get("representation_learning_rate", 1e-4))},
        {"params": output_parameters, "lr": float(run_cfg.get("output_learning_rate", 1e-4))},
        {"params": gate_parameters, "lr": float(run_cfg.get("gate_learning_rate", 0.0))},
    ])
    steps = int(run_cfg.get("steps", 8000))
    loss_config = {
        **training_cfg,
        "measure_bypass": True,
        "resampler_train_start_step": -1,
        "resampler_auxiliary_weight": 0.0,
        "style_magnitude_weight": 0.0,
        "style_flow_loss_weight": float(run_cfg.get("flow_loss_weight", 0.25)),
        "style_flow_direction_weight": 0.0,
        "style_token_contrastive_weight": 0.0,
        "style_kv_contrastive_weight": 0.0,
        "exact_self_residual_weight": float(run_cfg.get("residual_weight", 1.0)),
        "exact_self_direction_weight": float(run_cfg.get("residual_direction_weight", 0.2)),
        "exact_self_log_rms_weight": float(run_cfg.get("residual_log_rms_weight", 0.05)),
        "exact_self_x0_weight": float(run_cfg.get("x0_weight", 1.0)),
        "exact_self_scale_floor": float(run_cfg.get("residual_scale_floor", 1e-3)),
        "curriculum": {
            "gate_only_steps": 0, "self_reference_steps": steps + 1,
            "target_anneal_end": steps + 2, "oracle_distill_end": steps + 1,
        },
        "oracle_distill_weight": 0.0,
    }
    output = (
        destination / str(cfg.get("output_directory", "style_transfer_training"))
        / str(run_cfg.get("output_name", "exact_self_generalization"))
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split.json", {
        "train_ids": train_ids, "validation_ids": validation_ids,
    })
    wandb_run = None
    wandb_cfg = dict(run_cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "exact-self-generalization-96-24")),
            id=str(wandb_cfg.get("id", "exact-self-generalization-96-24-v1")),
            resume="never",
            config={"style_transfer": cfg, "experiment": run_cfg},
        )

    metric_keys = (
        "flow_loss", "base_flow_loss", "paired_flow_improvement",
        "style_output_ratio", "style_flow_direction_cosine",
        "style_flow_delta_to_desired_ratio", "exact_self_residual_loss",
        "predicted_x0_loss",
    )
    def evaluate_pool(pool: list[dict[str, Any]], evaluation_seed: int) -> dict[str, float]:
        anima.eval(); adapter.eval()
        records = []
        with torch.no_grad():
            for index, batch in enumerate(pool):
                generator = torch.Generator(device=device).manual_seed(
                    evaluation_seed + index * 97
                )
                _, metrics = _forward_flow_loss(
                    anima, adapter, resampler, batch, device,
                    generator=generator, loss_config=loss_config, step=1,
                )
                records.append(metrics)
        return {key: sum(float(row[key]) for row in records) / len(records) for key in metric_keys}

    sample_config = copy.deepcopy(config)
    sample_config["style_transfer"]["sampling"].update({
        "width": int(run_cfg.get("sample_width", 512)),
        "height": int(run_cfg.get("sample_height", 512)),
        "steps": int(run_cfg.get("sample_steps", 30)),
        "seed": int(run_cfg.get("sample_seed", seed ^ 0x5151)),
        "style_cfg": float(run_cfg.get("sample_style_cfg", 1.0)),
    })
    validation_every = int(run_cfg.get("validation_every", 250))
    sample_every = int(run_cfg.get("sample_every", 500))
    checkpoint_every = int(run_cfg.get("checkpoint_every", 500))
    max_grad_norm = float(run_cfg.get("max_grad_norm", 1.0))
    history = []
    vae = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        anima.train(); adapter.train(); optimizer.zero_grad(set_to_none=True)
        batch = train_batches[(step - 1) % len(train_batches)]
        generator = torch.Generator(device=device).manual_seed(seed + step * 10_007)
        loss, metrics = _forward_flow_loss(
            anima, adapter, resampler, batch, device,
            generator=generator, loss_config=loss_config, step=step,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
        optimizer.step()
        step_s = time.perf_counter() - step_started
        if wandb_run is not None:
            wandb_run.log({
                "train/loss": float(loss.detach()),
                "train/grad_norm": float(grad_norm),
                "train/step_s": step_s,
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{f"train/{key}": float(metrics[key]) for key in metric_keys},
            }, step=step)
        if step % validation_every == 0 or step == steps:
            validation = evaluate_pool(validation_batches, seed ^ 0xA11CE)
            train_probe = evaluate_pool(train_batches[:4], seed ^ 0xBEEF)
            row = {"step": step, "train_loss": float(loss.detach()),
                   "grad_norm": float(grad_norm), "train_probe": train_probe,
                   "validation": validation}
            history.append(row); write_json(output / "history.json", history)
            print(f"exact-self generalization step={step} metrics={row}", flush=True)
            if wandb_run is not None:
                wandb_run.log({
                    **{f"train_probe/{key}": value for key, value in train_probe.items()},
                    **{f"validation/{key}": value for key, value in validation.items()},
                }, step=step)
        if step % checkpoint_every == 0 or step == steps:
            torch.save({"step": step, "adapter": adapter.state_dict(),
                        "resampler": resampler.state_dict(), "config": run_cfg},
                       output / f"checkpoint-{step:07d}.pt")
        if step % sample_every == 0 or step == steps:
            sheet, vae, _ = _sample_style_adapter(
                anima, adapter, resampler, loader, sample_config, destination,
                output, device, step, vae, reference_mode="self",
                episode_index=int(validation_batches[0]["episodes"][0].target_id),
                batch_override=validation_batches[0], batch_row=0,
            )
            print(f"exact-self generalization sample={sheet}", flush=True)
            if wandb_run is not None:
                import wandb
                wandb_run.log({"sample/unseen_exact_self": wandb.Image(str(sheet))}, step=step)
    result = {
        "steps": steps, "train_images": len(train_ids),
        "validation_images": len(validation_ids), "final": history[-1],
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", result)
    if wandb_run is not None:
        wandb_run.finish()
    return result


def sample_exact_self_generalization_checkpoint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render evenly spaced train and unseen targets from a fixed checkpoint."""
    cfg = config["style_transfer"]
    run_cfg = dict(cfg.get("exact_self_generalization", {}))
    training_cfg = dict(cfg["training"])
    device = str(training_cfg.get("device", "cuda"))
    seed = int(run_cfg.get("seed", 20260815))
    random.seed(seed)
    torch.manual_seed(seed)
    loader_cfg = {
        **cfg["loader"], "split": "train",
        "batch_size": int(run_cfg.get("batch_size", 3)), "seed": seed,
    }
    loader = ProductionStyleLoader(destination, loader_cfg)
    train_batches, validation_batches = _collect_disjoint_exact_self_batches(
        loader,
        train_batches=int(run_cfg.get("train_batches", 32)),
        validation_batches=int(run_cfg.get("validation_batches", 8)),
    )
    source = (
        destination / str(cfg.get("output_directory", "style_transfer_training"))
        / str(run_cfg.get("output_name", "exact_self_generalization"))
    )
    split_path = source / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    actual_train = [item.target_id for batch in train_batches for item in batch["episodes"]]
    actual_validation = [
        item.target_id for batch in validation_batches for item in batch["episodes"]
    ]
    if actual_train != split["train_ids"] or actual_validation != split["validation_ids"]:
        raise RuntimeError("Reconstructed exact-self pools do not match checkpoint split.json")

    checkpoint_step = int(run_cfg.get("sample_checkpoint_step", 2000))
    checkpoint = source / f"checkpoint-{checkpoint_step:07d}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["step"]) != checkpoint_step:
        raise RuntimeError(f"Checkpoint step mismatch: {state['step']} != {checkpoint_step}")
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    resampler.requires_grad_(False).eval()
    if "resampler" in state:
        resampler.load_state_dict(state["resampler"])
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", False)),
        fuse_attention_projections=bool(training_cfg.get("fuse_attention_projections", False)),
    )
    adapter = _create_and_attach_style_adapter(
        anima, cfg["adapter"], device, dtype=torch.bfloat16
    )
    _load_adapter_checkpoint(adapter, state)
    adapter.requires_grad_(False).eval()

    sample_config = copy.deepcopy(config)
    sample_cfg = sample_config["style_transfer"]["sampling"]
    sample_cfg.update({
        "width": int(run_cfg.get("sample_width", 512)),
        "height": int(run_cfg.get("sample_height", 512)),
        "steps": int(run_cfg.get("sample_steps", 30)),
        "style_cfg": float(run_cfg.get("sample_style_cfg", 1.0)),
    })
    evaluation = source / f"evaluation-step-{checkpoint_step:07d}"
    base_seed = int(run_cfg.get("sample_seed", seed ^ 0x5151))
    vae = None
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "unseen": []}

    def evenly_spaced_indices(total: int, count: int) -> list[int]:
        count = max(1, min(count, total))
        if count == 1:
            return [total // 2]
        return [round(index * (total - 1) / (count - 1)) for index in range(count)]

    for group, batches, count in (
        ("train", train_batches, int(run_cfg.get("train_sample_count", 6))),
        ("unseen", validation_batches, int(run_cfg.get("validation_sample_count", 6))),
    ):
        rows = [(batch, row) for batch in batches for row in range(len(batch["episodes"]))]
        group_output = evaluation / group
        for order, index in enumerate(evenly_spaced_indices(len(rows), count)):
            batch, batch_row = rows[index]
            episode = batch["episodes"][batch_row]
            sample_config["style_transfer"]["sampling"]["seed"] = (
                base_seed + (0 if group == "train" else 1_000_003) + order * 10_007
            )
            sheet, vae, elapsed = _sample_style_adapter(
                anima, adapter, resampler, loader, sample_config, destination,
                group_output, device, checkpoint_step, vae,
                reference_mode="self", episode_index=int(episode.target_id),
                batch_override=batch, batch_row=batch_row,
            )
            samples[group].append({
                "target_id": int(episode.target_id), "style_id": str(episode.style_id),
                "seed": int(sample_config["style_transfer"]["sampling"]["seed"]),
                "sheet": str(sheet), "elapsed_s": elapsed,
            })
            print(
                f"exact-self checkpoint sample group={group} target={episode.target_id} "
                f"sheet={sheet}", flush=True,
            )
    result = {
        "checkpoint": str(checkpoint), "step": checkpoint_step,
        "train_pool": len(actual_train), "unseen_pool": len(actual_validation),
        "samples": samples,
    }
    write_json(evaluation / "summary.json", result)
    return result


def _save_training_state(
    path: Path,
    step: int,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    resampler: nn.Module | None = None,
    extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "step": step,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra_optimizers": {
            name: value.state_dict() for name, value in (extra_optimizers or {}).items()
        },
        "extra_state": dict(extra_state or {}),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if resampler is not None:
        state["resampler"] = resampler.state_dict()
    torch.save(state, temporary)
    temporary.replace(path)


def _save_final_model(
    path: Path,
    step: int,
    adapter: nn.Module,
    resampler: nn.Module,
    cfg: dict[str, Any],
) -> None:
    """Write the deployable bundle once, without training-only optimizer state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "adapter": adapter.state_dict(),
            "resampler": resampler.state_dict(),
            "config": cfg,
        },
        temporary,
    )
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


def _resolve_training_resume_checkpoint(
    output_checkpoint: Path,
    destination: Path,
    training: dict[str, Any],
    steps_override: int | None,
) -> Path | None:
    """Resolve an in-place or explicitly imported full training state."""
    if steps_override is not None or not bool(training.get("resume", True)):
        return None
    if output_checkpoint.exists():
        return output_checkpoint
    configured = training.get("resume_checkpoint")
    if not configured:
        return None
    checkpoint = Path(str(configured))
    if not checkpoint.is_absolute():
        checkpoint = destination / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training resume checkpoint does not exist: {checkpoint}")
    return checkpoint


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
    loader_cfg = {
        **cfg["loader"],
        "reference_curriculum": dict(training.get("curriculum", {})),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 1)
        ),
    }
    loader = ProductionStyleLoader(destination, loader_cfg)
    if loader.self_reference_target_ids:
        print(
            "self-reference target pool: "
            f"{len(loader.self_reference_target_ids)} images across "
            f"{len(loader.self_reference_buckets)} latent buckets; "
            f"limit={loader.self_reference_target_images_per_style}/style",
            flush=True,
        )
    validation_loader_cfg = {
        **cfg["loader"],
        "split": "validation",
        "batch_size": int(training.get("validation_batch_size", 1)),
    }
    validation_loader_cfg.pop("reference_curriculum", None)
    validation_loader_cfg.pop("gradient_accumulation_steps", None)
    validation_loader_cfg["seed"] = seed ^ 0x51A7
    validation_loader = ProductionStyleLoader(destination, validation_loader_cfg)
    train_validation_batches = max(
        0, int(training.get("train_validation_batches", 0))
    )
    train_validation_loader = None
    if train_validation_batches:
        # Keep the training curriculum on this loader. During exact-self
        # bootstrap it therefore evaluates the deterministic representative
        # target pool; after the bootstrap it automatically expands to the
        # complete train split. This separates underfitting from failure to
        # generalize to artist-disjoint validation rows.
        train_validation_loader_cfg = {
            **loader_cfg,
            "batch_size": int(training.get("validation_batch_size", 1)),
            "seed": seed ^ 0x7A11,
        }
        train_validation_loader = ProductionStyleLoader(
            destination, train_validation_loader_cfg
        )
    train_sample_loader_cfg = {
        **validation_loader_cfg,
        "split": "train",
        "seed": seed ^ 0x71A1,
    }
    train_sample_loader = ProductionStyleLoader(destination, train_sample_loader_cfg)
    resampler_train_start = int(training.get("resampler_train_start_step", -1))
    has_resampler_token_cache = bool(loader_cfg.get("resampler_token_cache"))
    # With a complete token cache the frozen Resampler is only needed when the
    # final deployable checkpoint is assembled.  Keep it on CPU so it consumes
    # neither training VRAM nor optimizer/checkpoint state during the run.
    resampler_device = (
        "cpu" if resampler_train_start < 0 and has_resampler_token_cache else device
    )
    resampler = load_per_reference_resampler(
        destination,
        cfg["resampler"],
        resampler_device,
        trainable=resampler_train_start >= 0,
    )
    if resampler_device == "cpu":
        print("frozen Resampler retained on CPU; training uses cached tokens", flush=True)
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
    adapter = _create_and_attach_style_adapter(anima, cfg["adapter"], device)
    # Establish immutable phase-level freezes before optimizer construction.
    # The same policy is reapplied after every curriculum transition below.
    _set_aggregator_trainable(
        adapter,
        step=0,
        start_step=int(training.get("aggregator_train_start_step", 0)),
        gate_only=False,
    )
    frozen_groups = _apply_adapter_freeze_policy(adapter, training)
    if frozen_groups:
        print(f"immutable style freeze policy: {frozen_groups}", flush=True)
    output_parameters = adapter.output_parameters()
    gate_parameters = adapter.gate_parameters()
    bridge_parameters = adapter.bridge_parameters()
    style_kv_parameters = adapter.kv_parameters()
    style_k_base_parameters: list[nn.Parameter] = []
    style_v_base_parameters: list[nn.Parameter] = []
    if bool(training.get("train_full_rank_style_k", False)):
        style_k_base_parameters.extend(adapter.active_k_base_parameters())
    if bool(training.get("train_full_rank_style_v", False)):
        style_v_base_parameters.extend(adapter.active_v_base_parameters())
    style_kv_base_parameters = style_k_base_parameters + style_v_base_parameters
    special_ids = {
        id(value)
        for value in (
            bridge_parameters + style_kv_parameters + style_kv_base_parameters
            + output_parameters + gate_parameters
        )
    }
    representation_parameters = [
        value for value in adapter.parameters() if id(value) not in special_ids
    ]
    parameters = (
        bridge_parameters + representation_parameters + style_kv_parameters
        + style_kv_base_parameters
        + output_parameters + gate_parameters
    )
    if len({id(value) for value in parameters}) != len(list(adapter.parameters())):
        raise RuntimeError("Style optimizer parameter groups do not cover the adapter exactly once")
    representation_lr = float(
        training.get("representation_learning_rate", training.get("learning_rate", 1e-4))
    )
    style_kv_lr = float(training.get("style_kv_learning_rate", representation_lr))
    style_k_base_lr = float(
        training.get(
            "style_k_base_learning_rate",
            training.get("style_kv_base_learning_rate", style_kv_lr),
        )
    )
    style_v_base_lr = float(
        training.get(
            "style_v_base_learning_rate",
            training.get("style_kv_base_learning_rate", style_kv_lr),
        )
    )
    bridge_lr = float(training.get("bridge_learning_rate", 1e-5))
    output_lr = float(training.get("output_learning_rate", representation_lr))
    gate_lr = float(training.get("gate_learning_rate", output_lr))
    resampler_lr = float(training.get("resampler_learning_rate", representation_lr * 0.1))
    resampler_parameters = (
        list(resampler.parameters())
        if resampler_train_start >= 0
        else []
    )
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
                "params": style_kv_parameters,
                "lr": style_kv_lr,
                "weight_decay": weight_decay,
                "name": "style_kv",
            },
            *(
                [{
                    "params": style_k_base_parameters,
                    "lr": style_k_base_lr,
                    "weight_decay": weight_decay,
                    "name": "style_k_base",
                }]
                if style_k_base_parameters else []
            ),
            *(
                [{
                    "params": style_v_base_parameters,
                    "lr": style_v_base_lr,
                    "weight_decay": weight_decay,
                    "name": "style_v_base",
                }]
                if style_v_base_parameters else []
            ),
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
            *(
                [{
                    "params": resampler_parameters,
                    "lr": resampler_lr,
                    "weight_decay": weight_decay,
                    "name": "resampler",
                }]
                if resampler_parameters else []
            ),
        ],
        eps=float(training.get("adapter_adam_eps", 1e-6)),
    )
    # Keep the bridge optimizer selectable.  The conservative RAdam path is
    # useful for tiny near-identity starts, while the literature-faithful
    # IP-Adapter baseline trains its image projection and copied K/V with the
    # same AdamW family from the first update.
    bridge_optimizer_name = str(
        training.get("bridge_optimizer", "radam")
    ).lower()
    bridge_optimizer_class = {
        "adamw": torch.optim.AdamW,
        "radam": torch.optim.RAdam,
    }.get(bridge_optimizer_name)
    if bridge_optimizer_class is None:
        raise ValueError(
            "style_transfer.training.bridge_optimizer must be adamw or radam"
        )
    bridge_optimizer = bridge_optimizer_class(
        bridge_parameters,
        lr=bridge_lr,
        betas=tuple(training.get("bridge_betas", (0.9, 0.999))),
        eps=float(training.get("bridge_adam_eps", 1e-6)),
        weight_decay=float(training.get("bridge_weight_decay", 0.0)),
    )
    base_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    bridge_base_learning_rate = bridge_lr
    print(
        "style optimizer: FP32 trainable weights, BF16 autocast; adapter=AdamW "
        f"bridge={bridge_optimizer_name}(lr={bridge_lr:g},warmup={int(training.get('bridge_warmup_steps', 400))},"
        f"eps={float(training.get('bridge_adam_eps', 1e-6)):g}) "
        f"representation_lr={representation_lr:g} style_kv_lr={style_kv_lr:g} "
        f"style_k_base_lr={style_k_base_lr:g}/"
        f"{'trainable' if style_k_base_parameters else 'frozen'} "
        f"style_v_base_lr={style_v_base_lr:g}/"
        f"{'trainable' if style_v_base_parameters else 'frozen'} "
        f"output_lr={output_lr:g} "
        f"gate_lr={gate_lr:g} "
        f"resampler={'AdamW(lr=' + format(resampler_lr, 'g') + ')' if resampler_parameters else 'frozen/external'}",
        flush=True,
    )
    schedule_steps = int(training["steps"])
    steps = int(steps_override if steps_override is not None else schedule_steps)
    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    if steps_override is not None:
        output = output / f"smoke-{steps_override}-steps"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "training_state.pt"
    oracle_path = output / "self_reference_oracle.pt"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    adaptive_reference_state: dict[str, Any] = {
        "positive_validations": 0,
        "activation_step": None,
    }
    resume_checkpoint = _resolve_training_resume_checkpoint(
        checkpoint_path, destination, training, steps_override
    )
    resumed = resume_checkpoint is not None
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        _load_adapter_checkpoint(adapter, state)
        if "resampler" in state:
            resampler.load_state_dict(state["resampler"])
        optimizer.load_state_dict(state["optimizer"])
        if "bridge" in state.get("extra_optimizers", {}):
            bridge_optimizer.load_state_dict(state["extra_optimizers"]["bridge"])
        start_step = int(state["step"])
        adaptive_reference_state.update(
            state.get("extra_state", {}).get("adaptive_reference", {})
        )
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(
            f"resuming style training from {resume_checkpoint} at step {start_step}",
            flush=True,
        )
    elif training.get("initial_checkpoint"):
        initial_path = Path(str(training["initial_checkpoint"]))
        if not initial_path.is_absolute():
            initial_path = destination / initial_path
        initial_state = torch.load(initial_path, map_location="cpu", weights_only=False)
        _load_adapter_checkpoint(adapter, initial_state)
        if "resampler" in initial_state:
            resampler.load_state_dict(initial_state["resampler"])
        print(
            f"initialized style adapter from {initial_path} at source step "
            f"{int(initial_state.get('step', -1))}",
            flush=True,
        )
    if (
        start_step == 0
        and bool(training.get("calibrate_bridge_output_rms_from_text", False))
    ):
        bridge_calibration = _calibrate_same_q_bridge_output_rms(
            adapter,
            validation_loader,
            batches=int(training.get("bridge_rms_calibration_batches", 4)),
        )
        write_json(output / "bridge_rms_calibration.json", bridge_calibration)
        print(
            "calibrated bridge output from nonzero text conditioning: "
            f"tokens={bridge_calibration['nonzero_tokens']} "
            f"padding={bridge_calibration['padding_tokens']} "
            f"mean={bridge_calibration['nonzero_text_mean']:.6f} "
            f"std={bridge_calibration['nonzero_text_standard_deviation']:.6f} "
            f"rms={bridge_calibration['bridge_output_rms']:.6f}",
            flush=True,
        )
    if (
        start_step == 0
        and bool(training.get("calibrate_alpha_from_attention_rms", False))
    ):
        alpha_calibration = _calibrate_same_q_alpha(
            anima,
            adapter,
            resampler,
            validation_loader,
            device,
            training,
            batches=int(training.get("alpha_calibration_batches", 4)),
            seed=seed ^ 0xA17A,
        )
        write_json(output / "alpha_calibration.json", alpha_calibration)
        active_calibration_blocks = alpha_calibration["active_blocks"]
        def mean_active(name: str) -> float:
            return sum(
                alpha_calibration[name][index]
                for index in active_calibration_blocks
            ) / len(active_calibration_blocks)
        print(
            "calibrated block alpha from raw style/text RMS: "
            f"target={alpha_calibration['target_style_to_text_ratio']:.4f} "
            f"raw_mean={sum(alpha_calibration['raw_style_to_text_ratio']) / len(alpha_calibration['raw_style_to_text_ratio']):.4f} "
            f"alpha_min={min(alpha_calibration['alpha']):.6g} "
            f"alpha_max={max(alpha_calibration['alpha']):.6g} "
            f"entropy=text:{mean_active('text_attention_normalized_entropy'):.4f}/"
            f"style:{mean_active('style_attention_normalized_entropy'):.4f} "
            f"top1=text:{mean_active('text_attention_top1_probability'):.4f}/"
            f"style:{mean_active('style_attention_top1_probability'):.4f}",
            flush=True,
        )
    if start_step >= steps:
        raise RuntimeError(f"Checkpoint is already at step {start_step}, requested steps={steps}")

    oracle_adapter = None
    curriculum_cfg = dict(training.get("curriculum", {}))
    self_reference_steps = int(curriculum_cfg.get("self_reference_steps", 0))
    oracle_distill_enabled = (
        float(training.get("oracle_distill_weight", 0.0)) > 0
        and int(curriculum_cfg.get("oracle_distill_end", self_reference_steps))
        > self_reference_steps
    )
    if start_step > self_reference_steps and oracle_distill_enabled:
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
            # Do not silently append a fresh optimizer run to stale history
            # merely because a human-readable ID was reused. True checkpoint
            # resumes retain the existing W&B run; fresh runs must use a new
            # ID and fail loudly otherwise.
            resume="allow" if resumed else "never",
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
    train_sample_artists = int(training.get("train_sample_artists", 4))
    validation_sample_artists = int(training.get("validation_sample_artists", 4))
    train_sample_episodes = _select_distinct_style_episode_indices(
        train_sample_loader, train_sample_artists
    )
    validation_sample_episodes = _select_distinct_style_episode_indices(
        validation_loader, validation_sample_artists
    )
    sample_panel = {
        split_name: [
            {
                "episode": episode_index,
                "target_id": int(sample_loader.episodes_for_step(episode_index)[0].target_id),
                "style_id": str(sample_loader.episodes_for_step(episode_index)[0].style_id),
            }
            for episode_index in episode_indices
        ]
        for split_name, sample_loader, episode_indices in (
            ("train", train_sample_loader, train_sample_episodes),
            ("validation", validation_loader, validation_sample_episodes),
        )
    }
    write_json(output / "sample_panel.json", sample_panel)
    print(f"fixed qualitative sample panel: {sample_panel}", flush=True)
    log_every = int(training.get("log_every", 10))
    vae = None
    if start_step == 0 and steps_override is None:
        baseline_loss_config = _adaptive_reference_loss_config(
            training, adaptive_reference_state, 0
        )
        baseline = _validate_style_adapter(
            anima,
            adapter,
            resampler,
            validation_loader,
            device,
            batches=validation_batches,
            seed=seed ^ 0xA11CE,
            loss_config={**baseline_loss_config, "timestep_sampling": "uniform"},
        )
        print(
            f"validation step=0 loss={baseline['loss']:.6f} "
            f"base={baseline['base_loss']:.6f} "
            f"paired={baseline['paired_improvement']:.6f} "
            f"ci95=±{baseline['paired_improvement_ci95']:.6f} "
            f"output_ratio={baseline['style_output_ratio']:.6f} "
            "ref_adv="
            f"{_format_optional_metric(baseline.get('correct_vs_wrong_advantage', float('nan')), 6)}/"
            f"{_format_optional_metric(baseline.get('correct_vs_wrong_positive_fraction', float('nan')), 3)} "
            f"batches={validation_batches} elapsed_s={baseline['elapsed_s']:.2f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    **{f"validation/{key}": value for key, value in baseline.items()},
                },
                step=0,
            )
    for zero_based_step in range(start_step, steps):
        step = zero_based_step + 1
        active_loss_config = _adaptive_reference_loss_config(
            training, adaptive_reference_state, step
        )
        lr_multiplier = _learning_rate_multiplier(
            step,
            schedule_steps,
            int(training.get("warmup_steps", 0)),
            float(training.get("minimum_lr_ratio", 1.0)),
        )
        for group, base_lr in zip(
            optimizer.param_groups, base_learning_rates, strict=True
        ):
            group["lr"] = base_lr * lr_multiplier
        bridge_lr_multiplier = _learning_rate_multiplier(
            step,
            schedule_steps,
            int(training.get("bridge_warmup_steps", 400)),
            float(training.get("bridge_minimum_lr_ratio", training.get("minimum_lr_ratio", 1.0))),
        )
        for group in bridge_optimizer.param_groups:
            group["lr"] = bridge_base_learning_rate * bridge_lr_multiplier
        curriculum = _self_reference_curriculum_state(step, curriculum_cfg)
        _set_adapter_trainable_stage(adapter, gate_only=bool(curriculum["gate_only"]))
        _set_aggregator_trainable(
            adapter,
            step=step,
            start_step=int(training.get("aggregator_train_start_step", 0)),
            gate_only=bool(curriculum["gate_only"]),
        )
        _apply_adapter_freeze_policy(adapter, training)
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
        bridge_optimizer.zero_grad(set_to_none=True)
        data_wait = 0.0
        accumulated_loss = 0.0
        details = None
        collect_step_details = (
            step == start_step + 1
            or step % log_every == 0
            or step == steps
            or (validation_every > 0 and step % validation_every == 0)
            or (sample_every > 0 and step % sample_every == 0)
            or (checkpoint_every > 0 and step % checkpoint_every == 0)
        )
        numeric_detail_samples: dict[str, list[float]] = defaultdict(list)
        boolean_detail_any: dict[str, bool] = defaultdict(bool)
        for _ in range(accumulation_steps):
            wait_started = time.perf_counter()
            batch = next(iterator)
            data_wait += time.perf_counter() - wait_started
            loss, details = _forward_flow_loss(
                anima, adapter, resampler, batch, device,
                loss_config=active_loss_config, step=step, oracle_adapter=oracle_adapter,
                collect_details=collect_step_details,
            )
            if collect_step_details:
                accumulated_loss += float(loss.detach()) / accumulation_steps
            for key, value in details.items():
                if isinstance(value, bool):
                    boolean_detail_any[key] = boolean_detail_any[key] or value
                elif key != "references" and isinstance(value, (int, float)):
                    numeric = float(value)
                    if math.isfinite(numeric):
                        numeric_detail_samples[key].append(numeric)
            (loss / accumulation_steps).backward()
            # Style tokens are part of the adapter's live autograd graph, so
            # keep them attached until each microbatch backward completes.
            adapter.clear_style_tokens()
        assert details is not None
        details = dict(details)
        for key, values in numeric_detail_samples.items():
            details[key] = sum(values) / len(values)
        details.update(boolean_detail_any)
        if bool(training.get("separate_gradient_clipping", False)):
            clipped_norms = _clip_style_gradient_groups(
                bridge_parameters,
                representation_parameters + style_kv_parameters
                + style_kv_base_parameters,
                output_parameters,
                gate_parameters,
                training,
            )
            grad_norm = clipped_norms["combined"]
        else:
            global_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(training.get("max_grad_norm", 1.0))
            )
            clipped_norms = {
                "representation": float("nan"),
                "output": float("nan"),
                "gate": float("nan"),
            }
            grad_norm = float(global_norm)
        resampler_grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                resampler_parameters,
                float(training.get("resampler_max_grad_norm", 0.25)),
            )
            if resampler_parameters
            else 0.0
        )
        group_grad_tensors = {
            "bridge_grad_norm": clipped_norms.get("bridge", torch.tensor(float("nan"))),
            "representation_grad_norm": clipped_norms["representation"],
            "output_grad_norm": clipped_norms["output"],
            "gate_grad_norm": clipped_norms["gate"],
            "combined_grad_norm": clipped_norms["combined"],
        }
        if collect_step_details:
            group_grad_tensors.update({
                "aggregator_grad": _parameter_grad_norm(adapter.aggregator.parameters()),
                "shared_kv_grad": _parameter_grad_norm(adapter.kv_parameters()),
                "full_rank_k_grad": _parameter_grad_norm(style_k_base_parameters),
                "full_rank_v_grad": _parameter_grad_norm(style_v_base_parameters),
                "style_output_grad": _parameter_grad_norm(adapter.output_parameters()),
                "gate_grad": _parameter_grad_norm(adapter.gate_parameters()),
                "resampler_grad": _parameter_grad_norm(resampler_parameters),
                "resampler_grad_norm": torch.as_tensor(resampler_grad_norm),
            })
            bridge_weights_before = [
                value.detach().clone() for value in bridge_parameters
            ]
            bridge_weight_norm_before_tensor = torch.stack([
                value.detach().float().square().sum() for value in bridge_parameters
            ]).sum().sqrt()
        optimizer.step()
        bridge_optimizer.step()
        if not collect_step_details:
            continue
        bridge_update_norm_tensor = torch.stack([
            (value.detach().float() - before.float()).square().sum()
            for value, before in zip(
                bridge_parameters, bridge_weights_before, strict=True
            )
        ]).sum().sqrt()
        bridge_weight_norm_tensor = torch.stack([
            value.detach().float().square().sum() for value in bridge_parameters
        ]).sum().sqrt()
        # This is intentionally restricted to logging/checkpoint/validation
        # steps. Synchronizing these scalars on every microbatch previously
        # serialized the otherwise asynchronous CUDA pipeline.
        group_grads = {
            name: float(torch.as_tensor(value))
            for name, value in group_grad_tensors.items()
        }
        bridge_update_norm = float(bridge_update_norm_tensor)
        bridge_weight_norm = float(bridge_weight_norm_tensor)
        bridge_weight_norm_before = float(bridge_weight_norm_before_tensor)
        bridge_update_to_weight_ratio = bridge_update_norm / max(
            bridge_weight_norm_before, 1e-12
        )
        elapsed = time.perf_counter() - data_ready
        row = {
            "step": step, "loss": accumulated_loss,
            "grad_norm": group_grads.get("combined_grad_norm", float(grad_norm)),
            "step_s": elapsed, "data_wait_s": data_wait,
            "gradient_accumulation_steps": accumulation_steps,
            "lr_multiplier": lr_multiplier,
            "bridge_lr_multiplier": bridge_lr_multiplier,
            "bridge_lr": bridge_optimizer.param_groups[0]["lr"],
            "bridge_weight_norm": bridge_weight_norm,
            "bridge_update_norm": bridge_update_norm,
            "bridge_update_to_weight_ratio": bridge_update_to_weight_ratio,
            "reference_loss_progress": float(
                active_loss_config.get("_reference_loss_progress", 0.0)
            ),
            "active_exact_self_direction_weight": float(
                active_loss_config.get("exact_self_direction_weight", 0.0)
            ),
            "active_reference_direction_weight": float(
                active_loss_config.get("style_reference_direction_weight", 0.0)
            ),
            **{
                f"{group['name']}_lr": float(group["lr"])
                for group in optimizer.param_groups
                if group.get("name")
            },
            "resampler_lr": (
                next(
                    group["lr"] for group in optimizer.param_groups
                    if group.get("name") == "resampler"
                )
                if resampler_parameters else 0.0
            ),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3) if device.startswith("cuda") else 0.0,
            **details, **group_grads,
        }
        metrics.append(row)
        metrics = metrics[-100:]
        if step == start_step + 1 or step % log_every == 0 or step == steps:
            bypass_measured = bool(row.get("bypass_measured", False))
            output_ratio = (
                _format_optional_metric(row["style_output_ratio"])
                if bypass_measured else "n/a"
            )
            projection = (
                _format_optional_metric(row["style_flow_desired_projection"])
                if bypass_measured else "n/a"
            )
            direction_cosine = (
                _format_optional_metric(row["style_flow_direction_cosine"])
                if bypass_measured else "n/a"
            )
            orthogonal_ratio = (
                _format_optional_metric(row["style_flow_orthogonal_to_desired_ratio"])
                if bypass_measured else "n/a"
            )
            print(
                f"style step={step}/{steps} loss={row['loss']:.6f} grad={row['grad_norm']:.4f} "
                f"phase={row['curriculum_phase']} "
                f"total_refs={row['references']} shape={tuple(row['latent_shape'])} step_s={elapsed:.2f} "
                f"data_wait_s={data_wait:.3f} output_ratio={output_ratio} "
                f"res_mse={row['exact_self_residual_mse_loss']:.4f} "
                f"align=proj:{projection}/cos:{direction_cosine}/orth:{orthogonal_ratio} "
                f"mag={row['style_magnitude_loss']:.5f} dir={row['style_flow_direction_loss']:.5f} "
                f"rank={row['style_reference_rank_loss']:.5f}/"
                f"{_format_optional_metric(row['style_reference_rank_advantage'], 5)} "
                f"ref_dir={row['style_reference_direction_loss']:.5f} "
                f"ref_mse={row['style_reference_residual_mse_loss']:.5f} "
                f"oracle={row['oracle_distill_loss']:.5f}/{int(row['oracle_distill_applied'])} "
                f"gate={row['style_gate_abs_mean']:.4f} "
                f"context_rms={row['style_context_rms']:.4f} "
                f"block_res={row['style_block_residual_ratio_mean']:.4f} "
                f"out_delta={row.get('style_output_delta_ratio_mean', 0.0):.4f} "
                f"bridge=grad:{row['bridge_grad_norm']:.4g}/"
                f"update:{row['bridge_update_norm']:.3g}/rel:{row['bridge_update_to_weight_ratio']:.3g} "
                f"grads=agg:{row['aggregator_grad']:.4g}/kv:{row['shared_kv_grad']:.4g}/"
                f"full_k:{row['full_rank_k_grad']:.4g}/"
                f"full_v:{row['full_rank_v_grad']:.4g}/"
                f"o:{row['style_output_grad']:.4g}/gate:{row['gate_grad']:.4g}/"
                f"res:{row['resampler_grad']:.4g} "
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
                loss_config={**active_loss_config, "timestep_sampling": "uniform"},
                reference_mode="heldout",
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
                # Compare self and heldout references on the same fixed
                # uniform validation distribution, regardless of the training
                # timestep sampler. Only the curriculum/reference mode should
                # differ between these paired reports.
                loss_config={**active_loss_config, "timestep_sampling": "uniform"},
                reference_mode="self",
            )
            train_self_validation = None
            if train_validation_loader is not None:
                train_self_validation = _validate_style_adapter(
                    anima,
                    adapter,
                    resampler,
                    train_validation_loader,
                    device,
                    batches=train_validation_batches,
                    seed=seed ^ 0x7A11CE,
                    step=step,
                    loss_config={
                        **active_loss_config,
                        "timestep_sampling": "uniform",
                    },
                    reference_mode="self",
                )
            print(
                f"validation[heldout] step={step} loss={heldout_validation['loss']:.6f} "
                f"base={heldout_validation['base_loss']:.6f} "
                f"paired={heldout_validation['paired_improvement']:.6f} "
                f"ci95=±{heldout_validation['paired_improvement_ci95']:.6f} "
                f"positive={heldout_validation['paired_positive_fraction']:.3f} "
                f"output_ratio={heldout_validation['style_output_ratio']:.6f} "
                f"projection={heldout_validation['style_flow_desired_projection']:.6f} "
                f"cosine={heldout_validation['style_flow_direction_cosine']:.6f} "
                f"orthogonal={heldout_validation['style_flow_orthogonal_to_desired_ratio']:.6f} "
                "ref_adv="
                f"{_format_optional_metric(heldout_validation.get('correct_vs_wrong_advantage', float('nan')), 6)}/"
                f"{_format_optional_metric(heldout_validation.get('correct_vs_wrong_positive_fraction', float('nan')), 3)} "
                f"batches={current_validation_batches} "
                f"elapsed_s={heldout_validation['elapsed_s']:.2f}",
                flush=True,
            )
            if bool(training.get("adaptive_reference_loss", False)):
                threshold = float(
                    training.get("reference_loss_activation_paired_threshold", 0.0)
                )
                lower_confidence_bound = (
                    self_validation["paired_improvement"]
                    - self_validation["paired_improvement_ci95"]
                )
                minimum_positive_fraction = float(
                    training.get(
                        "reference_loss_activation_positive_fraction", 0.75
                    )
                )
                validation_qualifies = (
                    lower_confidence_bound > threshold
                    and self_validation["paired_positive_fraction"]
                    >= minimum_positive_fraction
                )
                if validation_qualifies:
                    adaptive_reference_state["positive_validations"] = int(
                        adaptive_reference_state.get("positive_validations", 0)
                    ) + 1
                else:
                    adaptive_reference_state["positive_validations"] = 0
                required = max(
                    1,
                    int(training.get("reference_loss_activation_validations", 4)),
                )
                if (
                    adaptive_reference_state.get("activation_step") is None
                    and adaptive_reference_state["positive_validations"] >= required
                ):
                    adaptive_reference_state["activation_step"] = step
                    print(
                        "activated direction/reference loss ramp at "
                        f"step={step} after {required} statistically positive "
                        "self validations",
                        flush=True,
                    )
                print(
                    "reference-loss gate "
                    f"qualified={int(validation_qualifies)} "
                    f"streak={adaptive_reference_state['positive_validations']}/{required} "
                    f"self_lcb95={lower_confidence_bound:.6f} "
                    f"positive={self_validation['paired_positive_fraction']:.3f} "
                    f"activation_step={adaptive_reference_state.get('activation_step')}",
                    flush=True,
                )
            print(
                f"validation[self] step={step} loss={self_validation['loss']:.6f} "
                f"base={self_validation['base_loss']:.6f} "
                f"paired={self_validation['paired_improvement']:.6f} "
                f"ci95=±{self_validation['paired_improvement_ci95']:.6f} "
                f"positive={self_validation['paired_positive_fraction']:.3f} "
                f"output_ratio={self_validation['style_output_ratio']:.6f} "
                f"projection={self_validation['style_flow_desired_projection']:.6f} "
                f"cosine={self_validation['style_flow_direction_cosine']:.6f} "
                f"orthogonal={self_validation['style_flow_orthogonal_to_desired_ratio']:.6f} "
                "ref_adv="
                f"{_format_optional_metric(self_validation.get('correct_vs_wrong_advantage', float('nan')), 6)}/"
                f"{_format_optional_metric(self_validation.get('correct_vs_wrong_positive_fraction', float('nan')), 3)} "
                f"batches={current_validation_batches} "
                f"elapsed_s={self_validation['elapsed_s']:.2f}",
                flush=True,
            )
            if train_self_validation is not None:
                print(
                    "validation[train_self] "
                    f"step={step} loss={train_self_validation['loss']:.6f} "
                    f"base={train_self_validation['base_loss']:.6f} "
                    f"paired={train_self_validation['paired_improvement']:.6f} "
                    f"ci95=±{train_self_validation['paired_improvement_ci95']:.6f} "
                    f"positive={train_self_validation['paired_positive_fraction']:.3f} "
                    f"output_ratio={train_self_validation['style_output_ratio']:.6f} "
                    f"projection={train_self_validation['style_flow_desired_projection']:.6f} "
                    f"cosine={train_self_validation['style_flow_direction_cosine']:.6f} "
                    f"orthogonal={train_self_validation['style_flow_orthogonal_to_desired_ratio']:.6f} "
                    f"batches={train_validation_batches} "
                    f"elapsed_s={train_self_validation['elapsed_s']:.2f}",
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
                        **(
                            {
                                f"validation_train_self/{key}": value
                                for key, value in train_self_validation.items()
                            }
                            if train_self_validation is not None
                            else {}
                        ),
                        "curriculum/reference_validation_qualified": int(
                            bool(training.get("adaptive_reference_loss", False))
                            and validation_qualifies
                        ),
                        "curriculum/reference_positive_streak": int(
                            adaptive_reference_state.get("positive_validations", 0)
                        ),
                        "curriculum/reference_activation_step": int(
                            adaptive_reference_state.get("activation_step") or 0
                        ),
                    },
                    step=step,
                )
        if sample_every and step % sample_every == 0:
            # During the first stage the model is trained only on exact-self;
            # later panels switch to target-excluded references to expose
            # whether generalization appears as target inclusion is annealed.
            sample_mode = "self" if step <= self_reference_steps else "heldout"
            sample_records: list[tuple[str, Path]] = []
            sample_requests = []
            for split_name, sample_loader, episode_indices, seed_offset in (
                ("train", train_sample_loader, train_sample_episodes, 0),
                (
                    "validation",
                    validation_loader,
                    validation_sample_episodes,
                    100_000,
                ),
            ):
                for artist_index, episode_index in enumerate(episode_indices):
                    sample_requests.append(
                        (
                            split_name,
                            sample_loader,
                            episode_index,
                            int(
                            cfg.get("sampling", {}).get("seed", seed)
                            )
                            + seed_offset
                            + artist_index,
                        )
                    )
            sample_batch_size = max(
                1, int(training.get("sample_batch_size", 4))
            )
            sample_elapsed_s = 0.0
            split_counts = defaultdict(int)
            for offset in range(0, len(sample_requests), sample_batch_size):
                records, vae, elapsed_s = _sample_style_adapter_batch(
                    anima, adapter, resampler,
                    sample_requests[offset : offset + sample_batch_size],
                    config, destination, output, device, step, vae,
                    reference_mode=sample_mode,
                )
                sample_elapsed_s += elapsed_s
                for split_name, sheet in records:
                    split_counts[split_name] += 1
                    sample_records.append(
                        (
                            f"sample/{split_name}_artist_{split_counts[split_name]}",
                            sheet,
                        )
                    )
            if vae is not None:
                vae.to("cpu")
            anima.train()
            adapter.train()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                f"sample step={step} mode={sample_mode} "
                f"train_artists={len(train_sample_episodes)} "
                f"validation_artists={len(validation_sample_episodes)} "
                f"elapsed_s={sample_elapsed_s:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb

                wandb_run.log(
                    {
                        **{
                            key: wandb.Image(str(sheet))
                            for key, sheet in sample_records
                        },
                        "sample/elapsed_s": sample_elapsed_s,
                    },
                    step=step,
                )
        if checkpoint_every and step % checkpoint_every == 0:
            _save_training_state(
                checkpoint_path, step, adapter, optimizer, cfg,
                resampler if resampler_parameters else None,
                {"bridge": bridge_optimizer},
                {"adaptive_reference": adaptive_reference_state},
            )
            _archive_training_state(
                checkpoint_path, checkpoint_dir / f"step-{step:07d}.pt"
            )

    checkpoint = output / "checkpoint.pt"
    _save_training_state(
        checkpoint_path, steps, adapter, optimizer, cfg,
        resampler if resampler_parameters else None,
        {"bridge": bridge_optimizer},
        {"adaptive_reference": adaptive_reference_state},
    )
    _save_final_model(checkpoint, steps, adapter, resampler, cfg)
    summary = {
        "steps": steps, "metrics": metrics, "elapsed_s": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "trainable_parameters": sum(
            value.numel() for value in parameters + resampler_parameters
            if value.requires_grad
        ),
        "resampler_checkpoint": str(cfg["resampler"]["checkpoint"]),
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
    # The smoke contract is forward/backward/checkpoint validity. Full image
    # sampling is substantially slower and is tested by style-sample itself.
    training["sample_every"] = int(training.get("smoke_sample_every", 0))
    # Keep the production LR and curriculum clocks. Compressing an 8000-step
    # schedule into two steps makes step 1 use half the peak LR and turns this
    # safety check into an artificial first-update explosion test.
    smoke_config["style_transfer"]["sampling"]["steps"] = 2
    return train_style_adapter(smoke_config, destination, steps_override=2)


def benchmark_style_batches(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Measure end-to-end training throughput without validation/sample overhead."""
    benchmark_cfg = config["style_transfer"].get("benchmark", {})
    candidates = benchmark_cfg.get("candidates")
    if candidates is None:
        candidates = [
            {
                "batch_size": int(value),
                "gradient_accumulation_steps": int(
                    config["style_transfer"]["training"].get(
                        "gradient_accumulation_steps", 1
                    )
                ),
            }
            for value in benchmark_cfg.get("batch_sizes", [4, 8, 16])
        ]
    steps = int(benchmark_cfg.get("steps", 8))
    warmup = int(benchmark_cfg.get("warmup_steps", 2))
    if warmup >= steps:
        raise ValueError("style_transfer.benchmark.warmup_steps must be smaller than steps")
    results = []
    benchmark_root = destination / "style_transfer_benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    for benchmark_candidate in candidates:
        batch_size = int(benchmark_candidate["batch_size"])
        accumulation = int(benchmark_candidate["gradient_accumulation_steps"])
        candidate = copy.deepcopy(config)
        style_cfg = candidate["style_transfer"]
        style_cfg["output_directory"] = (
            f"style_transfer_benchmarks/batch-{batch_size}-accum-{accumulation}"
        )
        style_cfg["loader"]["batch_size"] = batch_size
        training = style_cfg["training"]
        training.update(
            {
                "validation_every": 0,
                "checkpoint_every": 0,
                "sample_every": 0,
                "log_every": 1,
                "resume": False,
                "gradient_accumulation_steps": accumulation,
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
                "gradient_accumulation_steps": accumulation,
                "status": "ok",
                "measured_steps": len(measured),
                "mean_compute_s": mean_compute,
                "mean_data_wait_s": mean_wait,
                "mean_wall_step_s": wall_step,
                "target_images_s": batch_size * accumulation / wall_step,
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
