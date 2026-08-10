from __future__ import annotations

import json
import math
import random
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
        self.buckets = {
            shape: [image_id for image_id in values if image_id in valid_ids]
            for shape, values in buckets.items()
            if any(image_id in valid_ids for image_id in values)
        }
        if not self.by_style or not self.buckets:
            raise RuntimeError("No eligible same-style episodes exist in the cache intersection")
        self.bucket_keys = sorted(self.buckets)
        self.text_shards = _TensorShardCache(text_root, int(cfg.get("text_lru_shards", 2)))
        self.latent_shards = _TensorShardCache(latent_root, int(cfg.get("latent_lru_shards", 2)))

    def episodes_for_step(self, step: int) -> list[StyleEpisode]:
        rng = random.Random(self.seed + step * 1_000_003)
        shape = self.bucket_keys[step % len(self.bucket_keys)]
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
        max_text = max(value.shape[0] for value in conditions)
        condition_batch = torch.zeros(
            len(conditions), max_text, conditions[0].shape[-1], dtype=conditions[0].dtype
        )
        for index, value in enumerate(conditions):
            condition_batch[index, : value.shape[0]] = value

        flat_references = [image_id for item in episodes for image_id in item.reference_ids]
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
        for image_id in flat_references:
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
        return {
            "episodes": episodes,
            "latents": latent_batch.pin_memory(),
            "conditioning": condition_batch.pin_memory(),
            "features": {key: value.pin_memory() for key, value in features.items()},
            "feature_mask": feature_mask.pin_memory(),
            "feature_shapes": shapes,
            "global_features": global_features.pin_memory(),
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

    def __init__(self, slots: int = 16, dim: int = 768, heads: int = 12, layers: int = 2):
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
        return pooled.reshape(batch, slots, dim)


class SharedLowRankStyleAdapter(nn.Module):
    """Set aggregation plus shared K/V and per-block low-rank K/V deltas."""

    def __init__(
        self,
        *,
        style_dim: int = 768,
        slots: int = 16,
        hidden_dim: int = 2048,
        heads: int = 16,
        blocks: int = 28,
        rank: int = 16,
        aggregator_heads: int = 12,
        aggregator_layers: int = 2,
        style_dropout: float = 0.12,
        gate_dim: int = 256,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.slots = slots
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.blocks = blocks
        self.style_dropout = style_dropout
        self.aggregator = SlotSetAggregator(slots, style_dim, aggregator_heads, aggregator_layers)
        self.null_tokens = nn.Parameter(torch.empty(1, slots, style_dim))
        nn.init.normal_(self.null_tokens, std=0.02)
        self.shared_k = nn.Linear(style_dim, hidden_dim, bias=False)
        self.shared_v = nn.Linear(style_dim, hidden_dim, bias=False)
        self.k_down = nn.ModuleList([nn.Linear(style_dim, rank, bias=False) for _ in range(blocks)])
        self.k_up = nn.ModuleList([nn.Linear(rank, hidden_dim, bias=False) for _ in range(blocks)])
        self.v_down = nn.ModuleList([nn.Linear(style_dim, rank, bias=False) for _ in range(blocks)])
        self.v_up = nn.ModuleList([nn.Linear(rank, hidden_dim, bias=False) for _ in range(blocks)])
        for modules in (self.k_up, self.v_up):
            for layer in modules:
                nn.init.zeros_(layer.weight)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, gate_dim), nn.SiLU(), nn.Linear(gate_dim, blocks))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self._style_tokens: torch.Tensor | None = None

    def aggregate(self, references: torch.Tensor, reference_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.aggregator(references, reference_mask)
        if self.training and self.style_dropout > 0:
            dropped = torch.rand(tokens.shape[0], device=tokens.device) < self.style_dropout
            tokens = torch.where(dropped[:, None, None], self.null_tokens.expand_as(tokens), tokens)
        return tokens

    def unconditional(self, batch: int) -> torch.Tensor:
        return self.null_tokens.expand(batch, -1, -1)

    def set_style_tokens(self, tokens: torch.Tensor) -> None:
        self._style_tokens = tokens

    def clear_style_tokens(self) -> None:
        self._style_tokens = None

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
        attended = cross_attention.output_dropout(cross_attention.output_proj(attended))
        gate = torch.tanh(self.gate(timestep_embedding)[:, 0, block_index])
        return attended * gate[:, None, None]


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


def train_style_adapter(config: dict[str, Any], destination: Path, *, steps_override: int | None = None) -> dict[str, Any]:
    cfg = config["style_transfer"]
    training = cfg["training"]
    device = str(training.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Anima style training")
    seed = int(cfg.get("seed", 20260811))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
    loader = ProductionStyleLoader(destination, cfg["loader"])
    resampler = load_per_reference_resampler(destination, cfg["resampler"], device)
    anima = _resolve_anima_model(config, destination, device)
    anima.requires_grad_(False).train()
    if bool(training.get("gradient_checkpointing", True)):
        anima.enable_gradient_checkpointing()
    adapter = SharedLowRankStyleAdapter(**cfg["adapter"]).to(device, dtype=torch.bfloat16)
    attach_style_adapter(anima, adapter)
    parameters = [value for value in adapter.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=device.startswith("cuda"),
    )
    steps = int(steps_override if steps_override is not None else training["steps"])
    output = destination / str(cfg.get("output_directory", "style_transfer_training"))
    if steps_override is not None:
        output = output / f"smoke-{steps_override}-steps"
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    started = time.perf_counter()
    iterator = loader.prefetch(
        0, steps, workers=int(training.get("prefetch_workers", 2)),
        depth=int(training.get("prefetch_batches", 4)),
    )
    for step, batch in enumerate(iterator, start=1):
        data_ready = time.perf_counter()
        references = _encode_reference_tokens(resampler, batch, device)
        reference_mask = batch["reference_mask"].to(device, non_blocking=True)
        latents = batch["latents"].to(device, non_blocking=True, dtype=torch.bfloat16)
        conditioning = batch["conditioning"].to(device, non_blocking=True, dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            style_tokens = adapter.aggregate(references, reference_mask)
            adapter.set_style_tokens(style_tokens)
            noise = torch.randn_like(latents)
            timesteps = torch.rand(latents.shape[0], device=device, dtype=torch.float32)
            sigma = timesteps[:, None, None, None].to(latents.dtype)
            noisy = (1 - sigma) * latents + sigma * noise
            padding_mask = torch.zeros(
                latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
                device=device, dtype=latents.dtype,
            )
            prediction = anima(
                noisy.unsqueeze(2), timesteps, context=conditioning,
                padding_mask=padding_mask, target_input_ids=None,
            ).squeeze(2)
            target = noise - latents
            loss = F.mse_loss(prediction.float(), target.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(training.get("max_grad_norm", 1.0)))
        optimizer.step()
        adapter.clear_style_tokens()
        elapsed = time.perf_counter() - data_ready
        row = {
            "step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
            "step_s": elapsed, "references": int(reference_mask.sum()),
            "latent_shape": list(latents.shape),
        }
        metrics.append(row)
        print(
            f"style step={step}/{steps} loss={row['loss']:.6f} grad={row['grad_norm']:.4f} "
            f"refs={row['references']} shape={tuple(latents.shape)} step_s={elapsed:.2f}", flush=True,
        )
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {"step": steps, "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg},
        checkpoint,
    )
    summary = {
        "steps": steps, "metrics": metrics, "elapsed_s": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "trainable_parameters": sum(value.numel() for value in parameters),
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_style_adapter(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    return train_style_adapter(config, destination, steps_override=2)
