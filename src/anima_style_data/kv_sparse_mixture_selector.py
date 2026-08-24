"""Learn a visual metric for sparse mixtures of functional K/V LoRA teachers.

The direct factor hypernetwork must invent two bilinear rank factors for an
unseen style.  This stage instead keeps the trained K/V-only LoRAs as a
functional dictionary and learns which small convex mixture best explains a
reference.  A leave-one-out curriculum removes the source artist from the
dictionary, so training cannot collapse into artist-ID retrieval.
"""

from __future__ import annotations

import copy
import gc
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import write_json
from .kv_activation_modulation import (
    apply_kv_factors,
    kv_activation_objective,
    load_kv_lora_factor_bank,
)
from .kv_generalizing_modulator import (
    _stratified_view_indices,
    _teacher_image_split,
    _view_probabilities,
)
from .lora_oracle_bootstrap import (
    _materialize_reader_code_bank,
    _oracle_detail_config,
)


class SparseLoRAMixtureSelector(nn.Module):
    """Blend the frozen Reader cosine metric with a learned low-rank metric."""

    def __init__(
        self,
        *,
        slots: int,
        style_dim: int,
        hidden_dim: int = 512,
        learned_mix_initial: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0.0 < learned_mix_initial < 1.0:
            raise ValueError("learned_mix_initial must be between zero and one")
        self.slots = int(slots)
        self.style_dim = int(style_dim)
        flat_dim = self.slots * self.style_dim
        self.style_norm = nn.LayerNorm(style_dim)
        self.projection = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        initial_logit = math.log(
            float(learned_mix_initial) / (1.0 - float(learned_mix_initial))
        )
        self.learned_mix_logit = nn.Parameter(torch.tensor(initial_logit))

    def similarities(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query.shape[1:] != (self.slots, self.style_dim):
            raise ValueError("Query Reader-code shape does not match selector")
        if anchors.shape[1:] != (self.slots, self.style_dim):
            raise ValueError("Anchor Reader-code shape does not match selector")
        anchor_flat = self.style_norm(anchors).flatten(1)
        query_flat = self.style_norm(query).flatten(1)
        common = anchor_flat.mean(dim=0, keepdim=True)
        anchor_centered = anchor_flat - common
        query_centered = query_flat - common
        raw = F.normalize(query_centered.float(), dim=-1) @ F.normalize(
            anchor_centered.float(), dim=-1
        ).t()
        projected_anchor = self.projection(anchor_centered)
        projected_query = self.projection(query_centered)
        learned = F.normalize(projected_query.float(), dim=-1) @ F.normalize(
            projected_anchor.float(), dim=-1
        ).t()
        blend = self.learned_mix_logit.sigmoid()
        similarity = (1.0 - blend) * raw + blend * learned
        return similarity, {
            "raw_similarity": raw.mean().detach(),
            "learned_similarity": learned.mean().detach(),
            "learned_metric_fraction": blend.detach(),
        }


def sparse_mixture_coefficients(
    similarity: torch.Tensor,
    *,
    excluded: torch.Tensor | None,
    neighbors: int,
    temperature: float,
) -> torch.Tensor:
    """Top-k softmax coefficients with optional per-row dictionary masking."""

    logits = similarity.float()
    if excluded is not None:
        if excluded.shape != logits.shape:
            raise ValueError("Excluded dictionary mask shape does not match logits")
        logits = logits.masked_fill(excluded, float("-inf"))
    keep = min(int(neighbors), int(logits.shape[-1]))
    values, indices = logits.topk(keep, dim=-1)
    if not torch.isfinite(values).all():
        raise RuntimeError("Too many dictionary entries were excluded")
    local = F.softmax(values / float(temperature), dim=-1)
    return torch.zeros_like(logits).scatter(-1, indices, local)


def _save_state(
    path: Path,
    *,
    step: int,
    model: SparseLoRAMixtureSelector,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    train_indices: list[int],
    validation_indices: list[int],
    best_cosine: float,
    best_step: int,
    best_validation: dict[str, float] | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "architecture": {
            "slots": model.slots,
            "style_dim": model.style_dim,
        },
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "best_heldout_centered_cosine": float(best_cosine),
        "best_step": int(best_step),
        "best_validation": best_validation,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _validate_selector(
    model: SparseLoRAMixtureSelector,
    anchor_codes: torch.Tensor,
    validation_codes: torch.Tensor,
    reference_counts: torch.Tensor,
    validation_indices: torch.Tensor,
    contexts: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    neighbors: int,
    temperature: float,
    views_per_count: int,
    tokens: int,
    output_channels: int,
    direction_weight: float,
    magnitude_weight: float,
) -> dict[str, float]:
    model.eval()
    rows: dict[str, list[float]] = defaultdict(list)
    view_indices = _stratified_view_indices(reference_counts, views_per_count)
    token_indices = torch.linspace(
        0, contexts.shape[1] - 1, min(tokens, contexts.shape[1]),
        device=contexts.device,
    ).round().long().unique()
    output_indices = torch.linspace(
        0, teacher_up.shape[-2] - 1,
        min(output_channels, teacher_up.shape[-2]),
        device=teacher_up.device,
    ).round().long().unique()
    train_down = teacher_down[train_indices]
    train_up = teacher_up[train_indices].index_select(-2, output_indices)
    target_down = teacher_down[validation_indices]
    target_up = teacher_up[validation_indices].index_select(-2, output_indices)
    anchors = anchor_codes[train_indices]
    for view in view_indices.tolist():
        reference_count = int(reference_counts[view])
        query = validation_codes[validation_indices, view]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            similarity, similarity_metrics = model.similarities(query, anchors)
        coefficients = sparse_mixture_coefficients(
            similarity,
            excluded=None,
            neighbors=neighbors,
            temperature=temperature,
        )
        for block in range(int(teacher_down.shape[1])):
            for context in contexts:
                sampled_context = context[token_indices]
                basis = apply_kv_factors(
                    sampled_context[None].expand(len(train_indices), -1, -1),
                    train_down[:, block],
                    train_up[:, block],
                ).float()
                target = apply_kv_factors(
                    sampled_context[None].expand(len(validation_indices), -1, -1),
                    target_down[:, block],
                    target_up[:, block],
                ).float()
                prediction = torch.einsum("ba,atno->btno", coefficients, basis)
                prediction_centered = prediction - prediction.mean(
                    dim=0, keepdim=True
                )
                target_centered = target - target.mean(dim=0, keepdim=True)
                _, raw_metrics = kv_activation_objective(
                    prediction,
                    target,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
                _, centered_metrics = kv_activation_objective(
                    prediction_centered,
                    target_centered,
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
                for key, value in raw_metrics.items():
                    rows[f"raw_{key}"].append(float(value))
                    rows[f"{reference_count}ref_raw_{key}"].append(float(value))
                for key, value in centered_metrics.items():
                    rows[f"centered_{key}"].append(float(value))
                    rows[f"{reference_count}ref_centered_{key}"].append(
                        float(value)
                    )
        entropy = -(coefficients.clamp_min(1e-8).log() * coefficients).sum(-1)
        rows["coefficient_entropy"].append(float(entropy.mean()))
        rows["effective_neighbors"].append(float(entropy.exp().mean()))
        for key, value in similarity_metrics.items():
            rows[key].append(float(value))
    model.train()
    return {key: sum(values) / len(values) for key, values in rows.items()}


def train_sparse_kv_lora_mixture_selector(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config["kv_sparse_mixture_selector"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 4000))
    seed = int(cfg.get("seed", 20260824))
    device = str(training.get("device", "cuda"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    lora_directory = destination / str(cfg["lora_directory"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        lora_directory, blocks=int(cfg.get("blocks", 28))
    )
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)
    validation_count = int(training.get("validation_artists", 32))
    validation_list = [
        int(value)
        for value in torch.linspace(0, len(artist_ids) - 1, validation_count)
        .round().long().unique()
    ]
    validation_set = set(validation_list)
    train_list = [
        index for index in range(len(artist_ids)) if index not in validation_set
    ]
    train_indices = torch.tensor(train_list, device=device)
    validation_indices = torch.tensor(validation_list, device=device)

    reader_state = torch.load(
        destination / str(cfg["reader_checkpoint"]),
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
    materialized_images = int(training.get("materialized_reference_images", 8))
    loader_common = {
        "split": "train",
        "style_ids": artist_ids,
        "batch_size": int(training.get("materialization_artist_chunk", 16)),
        "seed": seed,
        "token_lru_shards": int(training.get("token_lru_shards", 8)),
        "strict_style_ids": True,
    }
    train_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        references=materialized_images,
        allowed_image_ids=teacher_train_ids,
        **loader_common,
    )
    validation_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        references=int(training.get("validation_reference_images", 4)),
        allowed_image_ids=teacher_validation_ids,
        **loader_common,
    )
    code_bank, reference_counts = _materialize_reader_code_bank(
        reader,
        train_loader,
        artist_ids,
        reference_images=materialized_images,
        seed=seed ^ 0x53454C54,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    validation_codes, validation_counts = _materialize_reader_code_bank(
        reader,
        validation_loader,
        artist_ids,
        reference_images=int(training.get("validation_reference_images", 4)),
        seed=seed ^ 0x56414C53,
        device=device,
        style_chunk_size=int(training.get("materialization_artist_chunk", 16)),
    )
    del reader
    gc.collect()
    torch.cuda.empty_cache()
    code_bank = code_bank.to(device=device, dtype=torch.bfloat16)
    validation_codes = validation_codes.to(device=device, dtype=torch.bfloat16)
    anchor_codes = code_bank.mean(dim=1)
    view_probabilities = _view_probabilities(
        reference_counts, dict(training.get("reference_count_weights", {}))
    )

    context_bank = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    heldout_contexts = int(training.get("heldout_contexts", 32))
    train_contexts = context_bank[:-heldout_contexts].to(
        device=device, dtype=torch.bfloat16
    )
    heldout = context_bank[-heldout_contexts:]
    validation_context_indices = torch.linspace(
        0, heldout_contexts - 1,
        int(training.get("validation_contexts", 3)),
    ).round().long().unique()
    validation_contexts = heldout[validation_context_indices].to(
        device=device, dtype=torch.bfloat16
    )

    model = SparseLoRAMixtureSelector(
        slots=int(code_bank.shape[-2]),
        style_dim=int(code_bank.shape[-1]),
        **dict(cfg.get("model", {})),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    best_cosine = float("-inf")
    best_step = 0
    best_validation: dict[str, float] | None = None
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_cosine = float(state.get("best_heldout_centered_cosine", -math.inf))
        best_step = int(state.get("best_step", 0))
        best_validation = state.get("best_validation")
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch_artists = int(training.get("batch_artists", 16))
    mixture_probability = float(training.get("mixture_probability", 0.25))
    mixture_sizes = tuple(int(value) for value in training.get("mixture_sizes", [2, 3]))
    train_neighbors = int(training.get("training_neighbors", 32))
    validation_neighbors = int(training.get("validation_neighbors", 8))
    temperature = float(training.get("temperature", 0.1))
    exclude_start = int(training.get("exclude_source_start_step", 250))
    exclude_end = int(training.get("exclude_source_end_step", 1500))
    contexts_per_step = int(training.get("contexts_per_step", 2))
    tokens_per_step = int(training.get("tokens_per_step", 64))
    channels_per_step = int(training.get("output_channels_per_step", 256))
    raw_weight = float(training.get("raw_function_weight", 0.25))
    centered_weight = float(training.get("centered_function_weight", 1.0))
    direction_weight = float(training.get("direction_weight", 1.0))
    magnitude_weight = float(training.get("magnitude_weight", 0.2))
    base_lr = float(training.get("learning_rate", 2e-4))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 2.0))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    validation_history: list[dict[str, Any]] = []

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-sparse-mixture-selector")),
            id=str(wandb_cfg.get("id", "kv-sparse-mixture-selector")),
            resume="allow",
            config={"kv_sparse_mixture_selector": cfg},
        )
    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        model.train()
        for step in range(start_step + 1, steps + 1):
            learning_rate = base_lr * min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = learning_rate
            generator = torch.Generator().manual_seed(seed + step * 1_000_003)
            is_mixture = float(torch.rand((), generator=generator)) < mixture_probability
            group_size = (
                mixture_sizes[int(torch.randint(len(mixture_sizes), (1,), generator=generator))]
                if is_mixture else 1
            )
            needed = batch_artists * group_size
            local_groups = torch.randperm(
                len(train_list), generator=generator
            )[:needed].reshape(batch_artists, group_size).to(device)
            global_groups = train_indices[local_groups]
            views = torch.multinomial(
                view_probabilities,
                needed,
                replacement=True,
                generator=generator,
            ).reshape(batch_artists, group_size).to(device)
            if group_size == 1:
                mixture_weights = torch.ones(batch_artists, 1, device=device)
            else:
                mixture_weights = -torch.rand(
                    batch_artists, group_size, generator=generator
                ).clamp_min(1e-6).log().to(device)
                mixture_weights /= mixture_weights.sum(dim=-1, keepdim=True)
            visual_groups = code_bank[global_groups, views]
            visual = torch.einsum(
                "bg,bgsd->bsd", mixture_weights.to(visual_groups), visual_groups
            )
            block = (step - 1) % int(teacher_down.shape[1])
            context_indices = torch.randperm(
                len(train_contexts), generator=generator
            )[:contexts_per_step].to(device)
            token_indices = torch.randperm(
                train_contexts.shape[1], generator=generator
            )[:tokens_per_step].to(device)
            output_indices = torch.randperm(
                teacher_up.shape[-2], generator=generator
            )[:channels_per_step].to(device)
            sampled_contexts = train_contexts[context_indices][:, token_indices]

            exclude_probability = max(
                0.0,
                min(1.0, (step - exclude_start) / max(1, exclude_end - exclude_start)),
            )
            excluded = torch.zeros(
                batch_artists, len(train_list), device=device, dtype=torch.bool
            )
            if float(torch.rand((), generator=generator)) < exclude_probability:
                excluded.scatter_(1, local_groups, True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                similarity, similarity_metrics = model.similarities(
                    visual, anchor_codes[train_indices]
                )
            coefficients = sparse_mixture_coefficients(
                similarity,
                excluded=excluded,
                neighbors=train_neighbors,
                temperature=temperature,
            )
            basis_context = sampled_contexts[None].expand(
                len(train_list), -1, -1, -1
            ).reshape(
                len(train_list) * contexts_per_step,
                tokens_per_step,
                sampled_contexts.shape[-1],
            )
            basis_down = teacher_down[train_indices, block][:, None].expand(
                -1, contexts_per_step, -1, -1, -1
            ).reshape(len(train_list) * contexts_per_step, *teacher_down.shape[2:])
            selected_up = teacher_up[train_indices, block].index_select(
                -2, output_indices
            )
            basis_up = selected_up[:, None].expand(
                -1, contexts_per_step, -1, -1, -1
            ).reshape(len(train_list) * contexts_per_step, *selected_up.shape[1:])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                basis = apply_kv_factors(
                    basis_context, basis_down, basis_up
                ).reshape(
                    len(train_list), contexts_per_step, 2,
                    tokens_per_step, channels_per_step,
                )
            prediction = torch.einsum(
                "ba,actno->bctno", coefficients.to(basis), basis
            ).flatten(0, 1)
            target = (
                basis[local_groups]
                * mixture_weights[:, :, None, None, None, None].to(basis)
            ).sum(dim=1).flatten(0, 1)
            raw_loss, raw_metrics = kv_activation_objective(
                prediction,
                target,
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
            )
            prediction_by_artist = prediction.reshape(
                batch_artists, contexts_per_step, *prediction.shape[1:]
            ).float()
            target_by_artist = target.reshape(
                batch_artists, contexts_per_step, *target.shape[1:]
            ).float()
            centered_prediction = prediction_by_artist - prediction_by_artist.mean(
                dim=0, keepdim=True
            )
            centered_target = target_by_artist - target_by_artist.mean(
                dim=0, keepdim=True
            )
            centered_loss, centered_metrics = kv_activation_objective(
                centered_prediction.flatten(0, 1),
                centered_target.flatten(0, 1),
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
            )
            loss = raw_weight * raw_loss + centered_weight * centered_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            optimizer.step()
            entropy = -(coefficients.clamp_min(1e-8).log() * coefficients).sum(-1)
            metrics = {
                "loss": loss.detach(),
                "raw_loss": raw_loss.detach(),
                "centered_loss": centered_loss.detach(),
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **{
                    f"centered_{key}": value
                    for key, value in centered_metrics.items()
                },
                **similarity_metrics,
                "coefficient_entropy": entropy.mean().detach(),
                "effective_neighbors": entropy.exp().mean().detach(),
                "source_exclusion_probability": torch.tensor(exclude_probability),
                "source_excluded": excluded.any(dim=1).float().mean(),
                "mixture": torch.tensor(float(is_mixture)),
                "mixture_size": torch.tensor(float(group_size)),
                "mean_references": reference_counts[views].float().mean(),
                "grad_norm": grad_norm.detach(),
                "learning_rate": torch.tensor(learning_rate),
                "block": torch.tensor(float(block)),
            }
            for key, value in metrics.items():
                running[key].append(float(value))
            if step % log_every == 0:
                row = {
                    key: sum(values) / len(values) for key, values in running.items()
                }
                print(f"K/V sparse selector step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/kv_sparse/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                validation = _validate_selector(
                    model,
                    anchor_codes,
                    validation_codes,
                    validation_counts,
                    validation_indices,
                    validation_contexts,
                    teacher_down,
                    teacher_up,
                    train_indices,
                    neighbors=validation_neighbors,
                    temperature=temperature,
                    views_per_count=int(training.get("validation_views", 2)),
                    tokens=int(training.get("validation_tokens", 64)),
                    output_channels=int(training.get("validation_output_channels", 256)),
                    direction_weight=direction_weight,
                    magnitude_weight=magnitude_weight,
                )
                validation_history.append({"step": step, **validation})
                write_json(output / "validation_history.json", validation_history)
                heldout_cosine = float(validation["centered_cosine"])
                print(
                    f"K/V sparse selector validation step={step} {validation}",
                    flush=True,
                )
                if heldout_cosine > best_cosine:
                    best_cosine = heldout_cosine
                    best_step = step
                    best_validation = dict(validation)
                    _save_state(
                        output / "best.pt",
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        train_indices=train_list,
                        validation_indices=validation_list,
                        best_cosine=best_cosine,
                        best_step=best_step,
                        best_validation=best_validation,
                    )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/kv_sparse/{key}": value for key, value in validation.items()},
                        step=step,
                    )
            if step % checkpoint_every == 0 or step == steps:
                for path in (
                    state_path,
                    checkpoints / f"step-{step:07d}.pt",
                ):
                    _save_state(
                        path,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        train_indices=train_list,
                        validation_indices=validation_list,
                        best_cosine=best_cosine,
                        best_step=best_step,
                        best_validation=best_validation,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "train_artists": len(train_list),
        "validation_artists": len(validation_list),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "best_step": best_step,
        "best_heldout_centered_cosine": best_cosine,
        "best_validation": best_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(state_path),
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_sparse_kv_lora_mixture_selector(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_sparse_mixture_selector"]
    cfg["output_directory"] += "_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["wandb"]["enabled"] = False
    return train_sparse_kv_lora_mixture_selector(
        effective, destination, steps_override=2
    )
