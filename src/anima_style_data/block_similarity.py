"""Functional clustering of Anima cross-attention blocks for shared Style K/V."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .detail_style_teacher_context import NativeArtistContextCache
from .io import write_json
from .native_centered_teacher import NativeCenteredTeacherBank
from .same_q_style_adapter import attach_same_q_style_adapter
from .same_q_style_adapter.adapter import (
    _match_native_attention_dtypes,
    _run_attention,
)
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


def centered_linear_cka(features: list[torch.Tensor], device: str) -> torch.Tensor:
    """Pairwise linear CKA for equally sampled [observations, features] matrices."""

    if not features:
        raise ValueError("features cannot be empty")
    shape = features[0].shape
    if any(value.ndim != 2 or value.shape != shape for value in features):
        raise ValueError("all CKA feature matrices must have the same 2D shape")
    values = torch.stack(features).to(device=device, dtype=torch.float32)
    values = values - values.mean(dim=1, keepdim=True)
    cross = torch.einsum("and,bne->abde", values, values)
    hsic = cross.square().sum(dim=(-1, -2))
    diagonal = hsic.diagonal().clamp_min(1e-12)
    result = hsic / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    return result.clamp(0.0, 1.0).cpu()


def correlation_similarity(features: list[torch.Tensor]) -> torch.Tensor:
    values = torch.stack([value.float().flatten() for value in features])
    values = values - values.mean(dim=1, keepdim=True)
    values = F.normalize(values, dim=1, eps=1e-8)
    return ((values @ values.T).clamp(-1.0, 1.0) + 1.0) * 0.5


def k_medoids(similarity: torch.Tensor, clusters: int) -> tuple[list[int], list[list[int]]]:
    """Deterministic PAM-style clustering for the 28-block similarity matrix."""

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    count = int(similarity.shape[0])
    if not 1 <= clusters <= count:
        raise ValueError("invalid cluster count")
    distance = (1.0 - similarity.double()).clamp_min(0.0)
    best: tuple[float, list[int], list[list[int]]] | None = None
    for first in range(count):
        medoids = [first]
        while len(medoids) < clusters:
            nearest = distance[:, medoids].min(dim=1).values
            nearest[medoids] = -1
            medoids.append(int(nearest.argmax()))
        for _ in range(64):
            labels = distance[:, medoids].argmin(dim=1)
            updated = []
            for label in range(clusters):
                members = torch.where(labels == label)[0]
                if members.numel() == 0:
                    raise RuntimeError("k-medoids produced an empty cluster")
                costs = distance[members][:, members].sum(dim=1)
                updated.append(int(members[int(costs.argmin())]))
            if updated == medoids:
                break
            medoids = updated
        labels = distance[:, medoids].argmin(dim=1)
        groups = [
            sorted(torch.where(labels == label)[0].tolist())
            for label in range(clusters)
        ]
        cost = float(sum(distance[group, medoid].sum() for group, medoid in zip(groups, medoids)))
        ordered = sorted(zip(medoids, groups), key=lambda item: min(item[1]))
        candidate = (cost, [item[0] for item in ordered], [item[1] for item in ordered])
        if best is None or candidate[0] < best[0] - 1e-12 or (
            abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] < best[1]
        ):
            best = candidate
    assert best is not None
    return best[1], best[2]


class _FunctionalBlockProbe(nn.Module):
    """Run the native text path while recording same-Q artist effects."""

    def __init__(self, *, blocks: int, token_samples: int, projection_dim: int, seed: int):
        super().__init__()
        self.blocks = int(blocks)
        self.token_samples = int(token_samples)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)
        self._teacher_context: torch.Tensor | None = None
        self._pending: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.query_features: list[list[torch.Tensor]] = [[] for _ in range(blocks)]
        self.effect_features: list[list[torch.Tensor]] = [[] for _ in range(blocks)]
        self.effect_rms: list[list[torch.Tensor]] = [[] for _ in range(blocks)]
        self._projections: dict[tuple[int, str], torch.Tensor] = {}
        self._initialized = False

    def initialize_from_anima(self, anima: nn.Module) -> None:
        if len(anima.blocks) != self.blocks:
            raise ValueError(f"expected {self.blocks} blocks, got {len(anima.blocks)}")
        self._initialized = True

    def set_teacher_context(self, context: torch.Tensor | None) -> None:
        self._teacher_context = context

    def _project(self, values: torch.Tensor, kind: str) -> torch.Tensor:
        key = (int(values.shape[-1]), kind)
        projection = self._projections.get(key)
        if projection is None:
            generator = torch.Generator(device=values.device).manual_seed(
                self.seed ^ int(values.shape[-1]) ^ (0x51A7 if kind == "query" else 0xEFFE)
            )
            projection = torch.randn(
                values.shape[-1], self.projection_dim,
                device=values.device, dtype=torch.float32, generator=generator,
            ) / math.sqrt(values.shape[-1])
            self._projections[key] = projection
        return values.float() @ projection

    @staticmethod
    def _native_context_kv(cross_attention: nn.Module, context: torch.Tensor):
        if hasattr(cross_attention, "kv_proj"):
            key, value = cross_attention.kv_proj(context).unflatten(
                -1, (2, cross_attention.n_heads, cross_attention.head_dim)
            ).unbind(dim=-3)
        else:
            key = cross_attention.k_proj(context).unflatten(
                -1, (cross_attention.n_heads, cross_attention.head_dim)
            )
            value = cross_attention.v_proj(context).unflatten(
                -1, (cross_attention.n_heads, cross_attention.head_dim)
            )
        return cross_attention.k_norm(key), cross_attention.v_norm(value)

    def merged_cross_attention(
        self,
        block_index: int,
        normalized_x: torch.Tensor,
        text_context: torch.Tensor,
        cross_attention: nn.Module,
        attn_params: Any,
    ) -> torch.Tensor:
        if self._teacher_context is None:
            raise RuntimeError("teacher context is not set")
        query, text_key, text_value = cross_attention.compute_qkv(
            normalized_x, text_context
        )
        teacher_key, teacher_value = self._native_context_kv(
            cross_attention, self._teacher_context
        )
        query, text_key, text_value, teacher_key, teacher_value = (
            _match_native_attention_dtypes(
                query, text_key, text_value, teacher_key, teacher_value, attn_params
            )
        )
        text_attended = _run_attention(
            cross_attention, query, text_key, text_value, attn_params
        )
        teacher_attended = _run_attention(
            cross_attention, query, teacher_key, teacher_value, attn_params
        )
        teacher_delta = F.linear(
            teacher_attended - text_attended,
            cross_attention.output_proj.weight,
            None,
        )
        self._pending[int(block_index)] = (query.detach(), teacher_delta.detach())
        return cross_attention.output_dropout(cross_attention.output_proj(text_attended))

    def record_gated_internal_teacher(
        self,
        block_index: int,
        gate_cross: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> None:
        query, effect = self._pending.pop(int(block_index))
        frames, height, width = spatial_shape
        gate = gate_cross.expand(-1, frames, height, width, -1).reshape(
            effect.shape[0], -1, effect.shape[-1]
        )
        effect = effect * gate
        if effect.shape[0] > 1:
            effect = effect - effect.mean(dim=0, keepdim=True)
        positions = torch.linspace(
            0, query.shape[1] - 1,
            min(self.token_samples, query.shape[1]),
            device=query.device,
        ).round().long().unique()
        query_values = query[0, positions].flatten(1)
        effect_values = effect[:, positions].reshape(-1, effect.shape[-1])
        self.query_features[block_index].append(
            self._project(query_values, "query").to("cpu", dtype=torch.float32)
        )
        self.effect_features[block_index].append(
            self._project(effect_values, "effect").to("cpu", dtype=torch.float32)
        )
        self.effect_rms[block_index].append(
            effect.float().square().mean(dim=(1, 2)).sqrt().cpu()
        )

    def matrices(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self._pending:
            raise RuntimeError("incomplete block probe forward")
        return (
            [torch.cat(values) for values in self.query_features],
            [torch.cat(values) for values in self.effect_features],
            [torch.cat(values) for values in self.effect_rms],
        )


def _native_kv_weights(anima: nn.Module, kind: str) -> list[torch.Tensor]:
    result = []
    for block in anima.blocks:
        cross = block.cross_attn
        if hasattr(cross, "kv_proj"):
            output_dim = int(cross.n_heads) * int(cross.head_dim)
            key, value = cross.kv_proj.weight.unflatten(0, (2, output_dim)).unbind(0)
        else:
            key, value = cross.k_proj.weight, cross.v_proj.weight
        result.append((key if kind == "key" else value).detach())
    return result


@torch.no_grad()
def _subspace_similarity(weights: list[torch.Tensor], device: str) -> torch.Tensor:
    covariances = []
    for weight in weights:
        value = weight.to(device=device, dtype=torch.float32)
        covariance = value.T @ value
        covariances.append(F.normalize(covariance.flatten(), dim=0))
    stacked = torch.stack(covariances)
    return (stacked @ stacked.T).clamp(0.0, 1.0).cpu()


@torch.no_grad()
def analyze_anima_block_similarity(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["detail_preserving_style_cross_attention"])
    probe_cfg = dict(cfg["block_similarity"])
    teacher_cfg = dict(cfg["teacher"])
    device = str(probe_cfg.get("device", cfg["training"].get("device", "cuda")))
    seed = int(probe_cfg.get("seed", 20260819))
    random.seed(seed)
    output = destination / str(probe_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)

    bank_key = str(teacher_cfg.get("bank_config_key", "dual_domain_native_teacher"))
    bank = NativeCenteredTeacherBank.load(config, destination, config_key=bank_key)
    contexts = NativeArtistContextCache(
        destination / str(teacher_cfg["context_cache"]),
        capacity=int(teacher_cfg.get("context_lru_shards", 8)),
    )
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(cfg["training"].get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(cfg["training"].get("fuse_attention_projections", True)),
    )
    probe = _FunctionalBlockProbe(
        blocks=len(anima.blocks),
        token_samples=int(probe_cfg.get("token_samples", 16)),
        projection_dim=int(probe_cfg.get("projection_dim", 256)),
        seed=seed,
    ).to(device)
    attach_same_q_style_adapter(anima, probe)

    tensors = bank.tensors
    contents = int(tensors["noisy_inputs"].shape[0])
    timesteps = int(tensors["noisy_inputs"].shape[1])
    batches = int(probe_cfg.get("probe_batches", contents * timesteps))
    rows = int(probe_cfg.get("artists_per_batch", 4))
    style_ids = list(bank.summary["train_style_ids"])
    rng = random.Random(seed ^ 0xB10C_51A1)
    started = time.perf_counter()
    for index in range(batches):
        content_index = index % contents
        timestep_index = (index // contents) % timesteps
        selected = rng.sample(style_ids, rows)
        tagged = contexts.get(selected, content_index).to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        noisy = tensors["noisy_inputs"][content_index, timestep_index].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        timestep = tensors["timesteps"][timestep_index].to(
            device=device, dtype=torch.bfloat16
        )
        base_context = tensors["base_context"][content_index : content_index + 1].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        probe.set_teacher_context(tagged)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            anima(
                noisy.expand(rows, -1, -1, -1).unsqueeze(2),
                timestep.expand(rows),
                context=base_context.expand(rows, -1, -1),
                padding_mask=torch.zeros(
                    rows, 1, noisy.shape[-2], noisy.shape[-1],
                    device=device, dtype=noisy.dtype,
                ),
                target_input_ids=None,
            )
        print(
            f"block-similarity probe {index + 1}/{batches} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    queries, effects, effect_rms = probe.matrices()
    query_cka = centered_linear_cka(queries, device)
    effect_cka = centered_linear_cka(effects, device)
    rms_correlation = correlation_similarity(effect_rms)
    key_subspace = _subspace_similarity(_native_kv_weights(anima, "key"), device)
    value_subspace = _subspace_similarity(_native_kv_weights(anima, "value"), device)
    weights = {
        "effect_cka": float(probe_cfg.get("effect_cka_weight", 0.35)),
        "query_cka": float(probe_cfg.get("query_cka_weight", 0.25)),
        "key_subspace": float(probe_cfg.get("key_subspace_weight", 0.15)),
        "value_subspace": float(probe_cfg.get("value_subspace_weight", 0.15)),
        "effect_rms_correlation": float(probe_cfg.get("effect_rms_weight", 0.10)),
    }
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("block similarity weights must sum to one")
    composite = (
        weights["effect_cka"] * effect_cka
        + weights["query_cka"] * query_cka
        + weights["key_subspace"] * key_subspace
        + weights["value_subspace"] * value_subspace
        + weights["effect_rms_correlation"] * rms_correlation
    )
    composite.fill_diagonal_(1.0)
    medoids, groups = k_medoids(composite, int(probe_cfg.get("clusters", 4)))
    block_rms = torch.tensor([float(value.median()) for value in effect_rms])
    relative_gain = (block_rms / block_rms.median().clamp_min(1e-8)).clamp(0.5, 2.0)

    matrices = {
        "composite": composite,
        "teacher_effect_cka": effect_cka,
        "query_cka": query_cka,
        "key_subspace": key_subspace,
        "value_subspace": value_subspace,
        "teacher_effect_rms_correlation": rms_correlation,
    }
    result = {
        "version": "anima-block-similarity-v1",
        "probe_batches": batches,
        "artists_per_batch": rows,
        "token_samples": probe.token_samples,
        "projection_dim": probe.projection_dim,
        "weights": weights,
        "medoid_blocks": medoids,
        "clusters": [
            {"base": index, "medoid_block": medoid, "blocks": group}
            for index, (medoid, group) in enumerate(zip(medoids, groups))
        ],
        "teacher_effect_rms": block_rms.tolist(),
        "relative_block_gain": relative_gain.tolist(),
        "matrices": {key: value.tolist() for key, value in matrices.items()},
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", result)
    return result
