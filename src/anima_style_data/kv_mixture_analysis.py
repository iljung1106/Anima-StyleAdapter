"""Measure whether multi-artist LoRA mixtures contain transferable signal.

This is deliberately an analysis stage rather than another training run.  It
asks whether coefficients obtained from frozen visual Reader codes reproduce
the held-out artist's *function* (native text K/V activation), and compares
that with an oracle linear-span upper bound.  LoRA factors themselves are not
mixed because their rank coordinates are not the represented function.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import write_json
from .kv_activation_modulation import (
    apply_kv_factors,
    canonicalize_lora_factor_bank,
    load_kv_lora_factor_bank,
)
from .lora_oracle_bootstrap import (
    _materialize_reader_code_bank,
    _oracle_detail_config,
)


def _ridge_coefficients(
    train: torch.Tensor,
    query: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    """Fit one shared signed coefficient vector per query in centered space."""

    train_f = train.float()
    query_f = query.float()
    train_f = train_f - train_f.mean(dim=0, keepdim=True)
    query_f = query_f - train.mean(dim=0, keepdim=True).float()
    gram = train_f @ train_f.t() / train_f.shape[1]
    cross = query_f @ train_f.t() / train_f.shape[1]
    scale = gram.diagonal().mean().clamp_min(1e-8)
    regularized = gram + float(ridge) * scale * torch.eye(
        gram.shape[0], device=gram.device
    )
    return torch.linalg.solve(regularized, cross.t()).t()


def _knn_coefficients(
    train: torch.Tensor,
    query: torch.Tensor,
    *,
    neighbors: int,
    temperature: float,
) -> torch.Tensor:
    """Return non-negative, sum-one visual-neighbour mixture coefficients."""

    common = train.float().mean(dim=0, keepdim=True)
    train_n = F.normalize(train.float() - common, dim=-1)
    query_n = F.normalize(query.float() - common, dim=-1)
    similarity = query_n @ train_n.t()
    values, indices = similarity.topk(min(neighbors, train.shape[0]), dim=-1)
    local = F.softmax(values / float(temperature), dim=-1)
    weights = torch.zeros_like(similarity)
    return weights.scatter(-1, indices, local)


def _sparse_ridge_coefficients(
    train: torch.Tensor,
    query: torch.Tensor,
    *,
    neighbors: int,
    ridge: float,
) -> torch.Tensor:
    """Refit a signed affine ridge solution on its strongest dictionary atoms."""

    train_centered = train.float() - train.float().mean(dim=0, keepdim=True)
    query_centered = query.float() - train.float().mean(dim=0, keepdim=True)
    full = _ridge_coefficients(train, query, ridge=ridge)
    keep = min(int(neighbors), int(train.shape[0]))
    selected = full.abs().topk(keep, dim=-1).indices
    result = torch.zeros_like(full)
    for row in range(query.shape[0]):
        atoms = train_centered[selected[row]]
        gram = atoms @ atoms.t() / atoms.shape[1]
        cross = atoms @ query_centered[row] / atoms.shape[1]
        scale = gram.diagonal().mean().clamp_min(1e-8)
        local = torch.linalg.solve(
            gram
            + float(ridge) * scale * torch.eye(
                keep, device=train.device, dtype=gram.dtype
            ),
            cross,
        )
        result[row].scatter_(0, selected[row], local)
    return result


def _activation_from_coefficients(
    train_activation: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    affine_centered: bool,
) -> torch.Tensor:
    """Mix gauge-invariant K/V activations, optionally around their mean."""

    if affine_centered:
        common = train_activation.float().mean(dim=0, keepdim=True)
        centered = train_activation.float() - common
        return common + torch.einsum("va,a...->v...", coefficients, centered)
    return torch.einsum("va,a...->v...", coefficients, train_activation.float())


def _effect_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    prediction_f = prediction.float()
    target_f = target.float()
    cosine = F.cosine_similarity(
        prediction_f.flatten(2), target_f.flatten(2), dim=-1
    )
    target_rms = target_f.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-8)
    prediction_rms = prediction_f.square().mean(dim=(-2, -1)).sqrt()
    error = (prediction_f - target_f).square().mean(dim=(-2, -1)).sqrt()
    return {
        "cosine": float(cosine.mean()),
        "k_cosine": float(cosine[:, 0].mean()),
        "v_cosine": float(cosine[:, 1].mean()),
        "rms_ratio": float((prediction_rms / target_rms).mean()),
        "relative_rms_error": float((error / target_rms).mean()),
    }


def _mixture_rank_energy_retention(
    down: torch.Tensor,
    up: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_rank: int,
) -> float:
    """Return the Frobenius energy retained by the best target-rank mixture.

    ``sum_g weight_g * up_g @ down_g`` can have ``groups * teacher_rank``
    non-zero singular values. Computing the large dense delta is unnecessary:
    thin QR decompositions reduce its singular spectrum to a small core matrix.
    """

    if down.ndim != 3 or up.ndim != 3:
        raise ValueError("Expected [groups, rank, input] and [groups, output, rank]")
    if down.shape[0] != up.shape[0] or down.shape[0] != weights.numel():
        raise ValueError("Mixture group dimensions do not match")
    groups, rank, _ = down.shape
    if up.shape[-1] != rank:
        raise ValueError("LoRA down/up ranks do not match")

    left = (
        up.float() * weights.float()[:, None, None]
    ).permute(1, 0, 2).reshape(up.shape[1], groups * rank)
    right_t = down.float().reshape(groups * rank, down.shape[-1]).t()
    _, left_r = torch.linalg.qr(left, mode="reduced")
    _, right_r = torch.linalg.qr(right_t, mode="reduced")
    singular = torch.linalg.svdvals(left_r @ right_r.t())
    energy = singular.square()
    keep = min(int(target_rank), int(energy.numel()))
    return float(energy[:keep].sum() / energy.sum().clamp_min(1e-12))


def _measure_mixture_rank_retention(
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    seed: int,
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed ^ 0x52414E4B)
    blocks = tuple(
        min(max(0, int(value)), int(teacher_down.shape[1]) - 1)
        for value in config.get("mixture_rank_blocks", [0, 7, 14, 21, 27])
    )
    samples = int(config.get("mixture_rank_samples", 4))
    result: dict[str, dict[str, float]] = {}
    for group_size in tuple(
        int(value) for value in config.get("mixture_rank_sizes", [2, 3])
    ):
        values: list[float] = []
        for _ in range(samples):
            local = torch.randperm(
                int(train_indices.numel()), generator=generator
            )[:group_size]
            artists = train_indices[local.to(train_indices.device)]
            weights = -torch.rand(
                group_size, generator=generator
            ).clamp_min(1e-6).log().to(train_indices.device)
            weights /= weights.sum()
            for block in blocks:
                for kind in range(2):
                    values.append(_mixture_rank_energy_retention(
                        teacher_down[artists, block, kind],
                        teacher_up[artists, block, kind],
                        weights,
                        target_rank=int(teacher_down.shape[-2]),
                    ))
        ordered = sorted(values)
        result[str(group_size)] = {
            "mean": float(sum(values) / len(values)),
            "minimum": float(ordered[0]),
            "p10": float(ordered[int(0.10 * (len(ordered) - 1))]),
        }
    return result


@torch.no_grad()
def analyze_kv_lora_mixture_generalization(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["kv_activation_mixture_analysis"])
    device = str(cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    modulator_state = torch.load(
        destination / str(cfg["modulator_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    mod_cfg = dict(modulator_state["config"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(mod_cfg["lora_directory"]),
        blocks=int(mod_cfg.get("blocks", 28)),
    )
    teacher_down, teacher_up = canonicalize_lora_factor_bank(
        teacher_down.to(device),
        teacher_up.to(device),
        chunk_size=int(cfg.get("canonicalization_chunk_size", 64)),
    )
    teacher_down = teacher_down.to(dtype=torch.bfloat16)
    teacher_up = teacher_up.to(dtype=torch.bfloat16)
    anchors = modulator_state["style_codes"].to(
        device=device, dtype=torch.bfloat16
    )
    del modulator_state

    oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
    reader_state = torch.load(
        destination / str(mod_cfg["style_code_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    detail_cfg = _oracle_detail_config(config, oracle_cfg)
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(
        device=device, dtype=torch.bfloat16
    )
    reader.load_state_dict(reader_state["reader"], strict=True)
    reader.requires_grad_(False).eval()
    del reader_state
    gc.collect()

    validation_count = int(cfg.get("validation_artists", 16))
    validation_list = [
        int(value) for value in torch.linspace(
            0, len(artist_ids) - 1, validation_count
        ).round().long().unique()
    ]
    validation_set = set(validation_list)
    train_list = [
        index for index in range(len(artist_ids)) if index not in validation_set
    ]
    train_indices = torch.tensor(train_list, device=device)
    validation_indices = torch.tensor(validation_list, device=device)

    # A convex mixture of two/three rank-R teachers can have rank 2R/3R,
    # while the current student emits rank R. Measure the best possible rank-R
    # compression before interpreting mixture-training loss.
    rank_retention = _measure_mixture_rank_retention(
        teacher_down,
        teacher_up,
        train_indices,
        seed=seed,
        config=cfg,
    )

    reference_images = int(cfg.get("materialized_reference_images", 8))
    loader_kwargs = {
        "split": "train",
        "style_ids": artist_ids,
        "batch_size": int(cfg.get("materialization_artist_chunk", 16)),
        "references": reference_images,
        "token_lru_shards": int(cfg.get("token_lru_shards", 8)),
        "strict_style_ids": True,
    }
    code_banks: list[torch.Tensor] = []
    reference_counts: torch.Tensor | None = None
    for domain, cache_key, domain_seed in (
        ("human", "human_reference_cache", 0x48554D41),
        ("synthetic", "synthetic_reference_cache", 0x53594E54),
    ):
        loader = CachedTeacherReferenceLoader(
            destination / str(oracle_cfg[cache_key]),
            seed=seed ^ domain_seed,
            **loader_kwargs,
        )
        codes, counts = _materialize_reader_code_bank(
            reader,
            loader,
            artist_ids,
            reference_images=reference_images,
            seed=seed ^ domain_seed ^ 0x33333333,
            device=device,
            style_chunk_size=int(cfg.get("materialization_artist_chunk", 16)),
        )
        if reference_counts is not None and not torch.equal(reference_counts, counts):
            raise RuntimeError("Human and synthetic reference views disagree")
        reference_counts = counts
        code_banks.append(codes)
        print(f"materialized mixture-analysis {domain} Reader codes", flush=True)
    del reader
    torch.cuda.empty_cache()
    assert reference_counts is not None
    code_bank = torch.stack(code_banks)
    max_references = int(reference_counts.max())
    max_views = torch.nonzero(
        reference_counts == max_references, as_tuple=False
    ).flatten()
    visual_codes = code_bank[:, :, max_views].mean(dim=2)

    contexts_all = load_file(
        destination / str(mod_cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    heldout_contexts = int(mod_cfg["training"].get("heldout_contexts", 32))
    contexts_all = contexts_all[-heldout_contexts:]
    context_count = int(cfg.get("contexts", 4))
    context_indices = torch.linspace(
        0, heldout_contexts - 1, context_count
    ).round().long().unique()
    contexts = contexts_all[context_indices].to(device=device, dtype=torch.bfloat16)
    token_stride = int(cfg.get("token_stride", 8))
    output_stride = int(cfg.get("output_stride", 8))
    contexts = contexts[:, ::token_stride]
    teacher_up_sampled = teacher_up[..., ::output_stride, :]

    anchor_flat = anchors.flatten(1)
    train_anchor = anchor_flat[train_indices]
    methods_by_domain: dict[str, dict[str, torch.Tensor]] = {}
    code_metrics: dict[str, dict[str, float]] = {}
    for domain_index, domain_name in enumerate(("human", "synthetic")):
        visual_flat = visual_codes[domain_index].flatten(1)
        query = visual_flat[validation_indices]
        methods: dict[str, torch.Tensor] = {
            "visual_ridge": _ridge_coefficients(
                train_anchor, query, ridge=float(cfg.get("visual_ridge", 0.05))
            )
        }
        for neighbors in cfg.get("mixture_neighbors", [2, 4, 8]):
            methods[f"visual_knn_{int(neighbors)}"] = _knn_coefficients(
                train_anchor,
                query,
                neighbors=int(neighbors),
                temperature=float(cfg.get("knn_temperature", 0.1)),
            )
        methods_by_domain[domain_name] = methods
        common = train_anchor.float().mean(dim=0, keepdim=True)
        reconstructed = common + methods["visual_ridge"] @ (
            train_anchor.float() - common
        )
        query_centered = query.float() - common
        reconstructed_centered = reconstructed - common
        code_metrics[domain_name] = {
            "reference_count": float(max_references),
            "ridge_code_cosine": float(F.cosine_similarity(
                reconstructed, query.float(), dim=-1
            ).mean()),
            "ridge_centered_code_cosine": float(F.cosine_similarity(
                reconstructed_centered, query_centered, dim=-1
            ).mean()),
            "ridge_centered_relative_rms_error": float(
                (reconstructed_centered - query_centered).square().mean(dim=-1).sqrt()
                .div(query_centered.square().mean(dim=-1).sqrt().clamp_min(1e-8))
                .mean()
            ),
        }

    # Fit a global signed artist-span coefficient using one held-out text
    # context and every block, then test it on different contexts.  This is an
    # oracle upper bound: inference cannot use the target K/V activation.
    train_count = len(train_list)
    val_count = len(validation_list)
    gram = torch.zeros(train_count, train_count, device=device)
    cross = torch.zeros(val_count, train_count, device=device)
    fit_context = contexts[0].expand(len(artist_ids), -1, -1)
    for block in range(teacher_down.shape[1]):
        activation = apply_kv_factors(
            fit_context,
            teacher_down[:, block],
            teacher_up_sampled[:, block],
        ).float()
        train_activation = activation[train_indices]
        common = train_activation.mean(dim=0, keepdim=True)
        train_centered = (train_activation - common).flatten(1)
        validation_centered = (
            activation[validation_indices] - common
        ).flatten(1)
        dimensions = train_centered.shape[1]
        gram.add_(train_centered @ train_centered.t() / dimensions)
        cross.add_(validation_centered @ train_centered.t() / dimensions)
    ridge_scale = gram.diagonal().mean().clamp_min(1e-8)
    oracle_coefficients = torch.linalg.solve(
        gram + float(cfg.get("oracle_ridge", 0.01)) * ridge_scale * torch.eye(
            train_count, device=device
        ),
        cross.t(),
    ).t()

    rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    evaluation_contexts = contexts[1:] if contexts.shape[0] > 1 else contexts
    for context in evaluation_contexts:
        expanded_context = context.expand(len(artist_ids), -1, -1)
        for block in range(teacher_down.shape[1]):
            activation = apply_kv_factors(
                expanded_context,
                teacher_down[:, block],
                teacher_up_sampled[:, block],
            )
            train_activation = activation[train_indices]
            target = activation[validation_indices]
            all_methods = {
                "oracle_activation_ridge": (oracle_coefficients, True),
            }
            for domain_name, methods in methods_by_domain.items():
                for method_name, coefficients in methods.items():
                    all_methods[f"{domain_name}_{method_name}"] = (
                        coefficients,
                        method_name == "visual_ridge",
                    )
            for method_name, (coefficients, affine_centered) in all_methods.items():
                prediction = _activation_from_coefficients(
                    train_activation,
                    coefficients,
                    affine_centered=affine_centered,
                )
                metrics = _effect_metrics(prediction, target)
                for key, value in metrics.items():
                    rows[method_name][key].append(value)

    results = {
        method: {
            key: sum(values) / len(values) for key, values in metrics.items()
        }
        for method, metrics in rows.items()
    }
    summary = {
        "train_artists": len(train_list),
        "validation_artists": len(validation_list),
        "fit_contexts": 1,
        "evaluation_contexts": int(evaluation_contexts.shape[0]),
        "blocks": int(teacher_down.shape[1]),
        "sampled_tokens": int(contexts.shape[1]),
        "sampled_output_channels": int(teacher_up_sampled.shape[-2]),
        "student_rank": int(teacher_down.shape[-2]),
        "mixture_rank_energy_retention": rank_retention,
        "code_metrics": code_metrics,
        "activation_metrics": results,
    }
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    return summary


@torch.no_grad()
def analyze_generalizing_kv_mixture_signal(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Measure genuine visual-to-LoRA signal with disjoint artist/image splits."""

    from .kv_generalizing_modulator import _teacher_image_split

    cfg = dict(config["kv_activation_generalizing_signal_analysis"])
    general_cfg = dict(config["kv_activation_generalizing_modulator"])
    training = dict(general_cfg["training"])
    device = str(cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    lora_directory = destination / str(general_cfg["lora_directory"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        lora_directory, blocks=int(general_cfg.get("blocks", 28))
    )
    teacher_down, teacher_up = canonicalize_lora_factor_bank(
        teacher_down.to(device),
        teacher_up.to(device),
        chunk_size=int(cfg.get("canonicalization_chunk_size", 64)),
    )
    teacher_down = teacher_down.to(dtype=torch.bfloat16)
    teacher_up = teacher_up.to(dtype=torch.bfloat16)

    reader_state = torch.load(
        destination / str(general_cfg["reader_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
    detail_cfg = _oracle_detail_config(config, oracle_cfg)
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(
        device=device, dtype=torch.bfloat16
    )
    reader.load_state_dict(reader_state["reader"], strict=True)
    reader.requires_grad_(False).eval()
    del reader_state

    teacher_train_ids, teacher_validation_ids = _teacher_image_split(
        lora_directory, artist_ids
    )
    common_loader = {
        "split": "train",
        "style_ids": artist_ids,
        "batch_size": int(training.get("materialization_artist_chunk", 16)),
        "seed": seed,
        "token_lru_shards": int(training.get("token_lru_shards", 8)),
        "strict_style_ids": True,
    }
    train_images = int(training.get("materialized_reference_images", 8))
    validation_images = int(training.get("validation_reference_images", 4))
    train_loader = CachedTeacherReferenceLoader(
        destination / str(general_cfg["human_reference_cache"]),
        references=train_images,
        allowed_image_ids=teacher_train_ids,
        **common_loader,
    )
    validation_loader = CachedTeacherReferenceLoader(
        destination / str(general_cfg["human_reference_cache"]),
        references=validation_images,
        allowed_image_ids=teacher_validation_ids,
        **common_loader,
    )
    train_codes, train_counts = _materialize_reader_code_bank(
        reader,
        train_loader,
        artist_ids,
        reference_images=train_images,
        seed=seed ^ 0x54524149,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    validation_codes, validation_counts = _materialize_reader_code_bank(
        reader,
        validation_loader,
        artist_ids,
        reference_images=validation_images,
        seed=seed ^ 0x56414C49,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    del reader
    torch.cuda.empty_cache()

    validation_count = int(training.get("validation_artists", 32))
    validation_list = [
        int(value)
        for value in torch.linspace(0, len(artist_ids) - 1, validation_count)
        .round()
        .long()
        .unique()
    ]
    validation_set = set(validation_list)
    train_list = [
        index for index in range(len(artist_ids)) if index not in validation_set
    ]
    train_indices = torch.tensor(train_list, device=device)
    validation_indices = torch.tensor(validation_list, device=device)
    rank_retention = _measure_mixture_rank_retention(
        teacher_down,
        teacher_up,
        train_indices,
        seed=seed,
        config=cfg,
    )

    coefficient_methods: dict[str, tuple[torch.Tensor, bool]] = {}
    code_metrics: dict[str, dict[str, float]] = {}
    for reference_count in tuple(int(value) for value in cfg.get("reference_counts", [1, 4])):
        train_views = torch.nonzero(train_counts == reference_count).flatten()
        validation_views = torch.nonzero(
            validation_counts == reference_count
        ).flatten()
        if not len(train_views) or not len(validation_views):
            raise RuntimeError(f"Missing {reference_count}-reference Reader views")
        train_visual = train_codes[:, train_views].float().mean(dim=1).flatten(1)
        validation_visual = (
            validation_codes[:, validation_views].float().mean(dim=1).flatten(1)
        )
        train_anchor = train_visual[train_indices]
        query = validation_visual[validation_indices]
        ridge = _ridge_coefficients(
            train_anchor, query, ridge=float(cfg.get("visual_ridge", 0.05))
        )
        coefficient_methods[f"{reference_count}ref_visual_ridge"] = (ridge, True)
        for neighbors in cfg.get("ridge_sparse_neighbors", [8, 16, 32]):
            coefficient_methods[
                f"{reference_count}ref_visual_sparse_ridge_{int(neighbors)}"
            ] = (
                _sparse_ridge_coefficients(
                    train_anchor,
                    query,
                    neighbors=int(neighbors),
                    ridge=float(cfg.get("visual_ridge", 0.05)),
                ),
                True,
            )
        for neighbors in cfg.get("mixture_neighbors", [2, 4, 8]):
            coefficient_methods[
                f"{reference_count}ref_visual_knn_{int(neighbors)}"
            ] = (
                _knn_coefficients(
                    train_anchor,
                    query,
                    neighbors=int(neighbors),
                    temperature=float(cfg.get("knn_temperature", 0.1)),
                ),
                False,
            )
        common = train_anchor.mean(dim=0, keepdim=True)
        reconstructed = common + ridge @ (train_anchor - common)
        query_centered = query - common
        reconstructed_centered = reconstructed - common
        code_metrics[f"{reference_count}ref"] = {
            "ridge_code_cosine": float(
                F.cosine_similarity(reconstructed, query, dim=-1).mean()
            ),
            "ridge_centered_code_cosine": float(
                F.cosine_similarity(
                    reconstructed_centered, query_centered, dim=-1
                ).mean()
            ),
            "ridge_centered_relative_rms_error": float(
                (reconstructed_centered - query_centered)
                .square()
                .mean(dim=-1)
                .sqrt()
                .div(
                    query_centered.square()
                    .mean(dim=-1)
                    .sqrt()
                    .clamp_min(1e-8)
                )
                .mean()
            ),
        }

    contexts_all = load_file(
        destination / str(general_cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    heldout_contexts = int(training.get("heldout_contexts", 32))
    contexts_all = contexts_all[-heldout_contexts:]
    context_indices = torch.linspace(
        0,
        heldout_contexts - 1,
        int(cfg.get("contexts", 4)),
    ).round().long().unique()
    contexts = contexts_all[context_indices].to(
        device=device, dtype=torch.bfloat16
    )[:, :: int(cfg.get("token_stride", 8))]
    teacher_up_sampled = teacher_up[..., :: int(cfg.get("output_stride", 8)), :]

    train_count = len(train_list)
    validation_count = len(validation_list)
    gram = torch.zeros(train_count, train_count, device=device)
    cross = torch.zeros(validation_count, train_count, device=device)
    fit_context = contexts[0].expand(len(artist_ids), -1, -1)
    for block in range(teacher_down.shape[1]):
        activation = apply_kv_factors(
            fit_context,
            teacher_down[:, block],
            teacher_up_sampled[:, block],
        ).float()
        train_activation = activation[train_indices]
        common = train_activation.mean(dim=0, keepdim=True)
        train_centered = (train_activation - common).flatten(1)
        validation_centered = (activation[validation_indices] - common).flatten(1)
        dimensions = train_centered.shape[1]
        gram.add_(train_centered @ train_centered.t() / dimensions)
        cross.add_(validation_centered @ train_centered.t() / dimensions)
    ridge_scale = gram.diagonal().mean().clamp_min(1e-8)
    oracle_coefficients = torch.linalg.solve(
        gram
        + float(cfg.get("oracle_ridge", 0.01))
        * ridge_scale
        * torch.eye(train_count, device=device),
        cross.t(),
    ).t()

    rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    evaluation_contexts = contexts[1:] if contexts.shape[0] > 1 else contexts
    for context in evaluation_contexts:
        expanded = context.expand(len(artist_ids), -1, -1)
        for block in range(teacher_down.shape[1]):
            activation = apply_kv_factors(
                expanded,
                teacher_down[:, block],
                teacher_up_sampled[:, block],
            )
            train_activation = activation[train_indices]
            target = activation[validation_indices]
            methods = {
                "oracle_activation_ridge": (oracle_coefficients, True),
                **coefficient_methods,
            }
            for method, (coefficients, affine_centered) in methods.items():
                prediction = _activation_from_coefficients(
                    train_activation,
                    coefficients,
                    affine_centered=affine_centered,
                )
                for key, value in _effect_metrics(prediction, target).items():
                    rows[method][key].append(value)

    activation_metrics = {
        method: {
            key: sum(values) / len(values) for key, values in metrics.items()
        }
        for method, metrics in rows.items()
    }
    summary = {
        "train_artists": len(train_list),
        "validation_artists": len(validation_list),
        "teacher_reference_images_disjoint": True,
        "evaluation_contexts": int(evaluation_contexts.shape[0]),
        "blocks": int(teacher_down.shape[1]),
        "sampled_tokens": int(contexts.shape[1]),
        "sampled_output_channels": int(teacher_up_sampled.shape[-2]),
        "student_rank": int(teacher_down.shape[-2]),
        "mixture_rank_energy_retention": rank_retention,
        "code_metrics": code_metrics,
        "activation_metrics": activation_metrics,
    }
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    return summary
