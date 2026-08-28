from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from torch.nn import functional as F

from .dual_query_training import (
    CacheEpisode,
    _intersect_cache_rows,
    _load_cache_episode,
    _model_from_config,
)
from .io import read_records, write_json


@dataclass
class RecoveryBatch:
    inputs: CacheEpisode
    target_tokens: torch.Tensor
    target_descriptors: torch.Tensor

    def to(self, device: str) -> "RecoveryBatch":
        non_blocking = device.startswith("cuda")
        return RecoveryBatch(
            inputs=self.inputs.to(device),
            target_tokens=self.target_tokens.to(device, non_blocking=non_blocking),
            target_descriptors=self.target_descriptors.to(
                device, non_blocking=non_blocking
            ),
        )


def _cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum_ratio: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    multiplier = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
    )
    return peak * multiplier


def _attach_teacher_rows(
    rows: list[dict[str, Any]], token_manifest: Path
) -> list[dict[str, Any]]:
    teacher_rows = {
        int(row["id"]): row for row in read_records(token_manifest)
    }
    joined = []
    for row in rows:
        teacher = teacher_rows.get(int(row["id"]))
        if teacher is None:
            continue
        joined.append(
            {
                **row,
                "teacher_shard": str(teacher["token_shard"]),
                "teacher_row": int(teacher["token_row"]),
            }
        )
    if not joined:
        raise RuntimeError("Feature, latent, and teacher-token caches do not intersect")
    return joined


def _load_teacher_bank(
    root: Path, rows: list[dict[str, Any]], *, pin_memory: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    id_to_index = {int(row["id"]): index for index, row in enumerate(ordered)}
    tokens = torch.empty(len(ordered), 84, 1024, dtype=torch.bfloat16)
    descriptors = torch.empty(len(ordered), 512, dtype=torch.bfloat16)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        groups[str(row["teacher_shard"])].append(row)
    for shard, shard_rows in groups.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            # Network-volume random slices are much slower than one contiguous
            # read. Each shard is only ~84 MiB, so materialize it once and copy
            # all selected rows together.
            shard_tokens = handle.get_tensor("tokens")
            shard_descriptors = handle.get_tensor("descriptors")
            source = torch.tensor(
                [int(row["teacher_row"]) for row in shard_rows], dtype=torch.long
            )
            target = torch.tensor(
                [id_to_index[int(row["id"])] for row in shard_rows], dtype=torch.long
            )
            tokens.index_copy_(0, target, shard_tokens.index_select(0, source))
            descriptors.index_copy_(
                0, target, shard_descriptors.index_select(0, source)
            )
    if pin_memory:
        tokens = tokens.pin_memory()
        descriptors = descriptors.pin_memory()
    return tokens, descriptors, id_to_index


def _split_styles(
    rows: list[dict[str, Any]], *, validation_styles: int, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["style_id"])].append(row)
    styles = sorted(grouped)
    if len(styles) <= validation_styles:
        raise RuntimeError(
            f"Need more than {validation_styles} styles, found {len(styles)}"
        )
    generator = random.Random(seed ^ 0x4D31)
    heldout = set(generator.sample(styles, validation_styles))
    train = {
        style: sorted(values, key=lambda row: int(row["id"]))
        for style, values in grouped.items()
        if style not in heldout
    }
    validation = {
        style: sorted(values, key=lambda row: int(row["id"]))
        for style, values in grouped.items()
        if style in heldout
    }
    return train, validation


def _sample_rows(
    grouped: Mapping[str, list[dict[str, Any]]],
    *,
    batch_size: int,
    seed: int,
    step: int,
) -> list[dict[str, Any]]:
    generator = random.Random(seed + step * 1_000_003)
    styles = sorted(grouped)
    chosen_styles = (
        generator.sample(styles, batch_size)
        if len(styles) >= batch_size
        else [generator.choice(styles) for _ in range(batch_size)]
    )
    return [
        {
            **generator.choice(grouped[style]),
            "episode_label": index,
            "artist_class_label": -1,
        }
        for index, style in enumerate(chosen_styles)
    ]


def _recovery_loss(
    model: torch.nn.Module,
    batch: RecoveryBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    item_count = int(batch.target_tokens.shape[0])
    keep = torch.ones(
        item_count,
        device=batch.target_tokens.device,
        dtype=batch.inputs.vae_latents.dtype,
    )
    output = model.encode(
        batch.inputs.semantic_features,
        batch.inputs.semantic_mask,
        batch.inputs.semantic_grid_shapes,
        batch.inputs.vae_latents,
        batch.inputs.vae_shapes,
        batch.inputs.image_sizes,
        semantic_keep=keep,
        vae_keep=keep,
        reconstruct=False,
    )
    predicted_queries = output.tokens.float()
    predicted_summary = output.artist_summary.float()
    predicted_descriptors = output.descriptor.float()
    target_queries = batch.target_tokens[:, :80].float()
    target_summary = batch.target_tokens[:, 80:].float()
    target_descriptors = batch.target_descriptors.float()

    query_huber = F.smooth_l1_loss(predicted_queries, target_queries, beta=0.1)
    summary_huber = F.smooth_l1_loss(
        predicted_summary, target_summary, beta=0.1
    )
    descriptor_huber = F.smooth_l1_loss(
        predicted_descriptors, target_descriptors, beta=0.02
    )
    query_cosine = F.cosine_similarity(
        predicted_queries, target_queries, dim=-1
    ).mean()
    summary_cosine = F.cosine_similarity(
        predicted_summary, target_summary, dim=-1
    ).mean()
    descriptor_cosine = F.cosine_similarity(
        predicted_descriptors, target_descriptors, dim=-1
    ).mean()
    total = (
        query_huber
        + summary_huber
        + 0.25 * (1.0 - query_cosine)
        + 0.50 * (1.0 - summary_cosine)
        + 0.25 * descriptor_huber
        + 0.25 * (1.0 - descriptor_cosine)
    )
    predicted_combined = torch.cat((predicted_queries, predicted_summary), dim=1)
    target_combined = batch.target_tokens.float()
    flat_cosine = F.cosine_similarity(
        predicted_combined.flatten(1), target_combined.flatten(1), dim=-1
    ).mean()
    similarity = F.normalize(predicted_descriptors, dim=-1) @ F.normalize(
        target_descriptors, dim=-1
    ).T
    retrieval_top1 = (
        similarity.argmax(dim=1)
        == torch.arange(item_count, device=similarity.device)
    ).float().mean()
    metrics = {
        "loss": total.detach(),
        "query_huber": query_huber.detach(),
        "summary_huber": summary_huber.detach(),
        "descriptor_huber": descriptor_huber.detach(),
        "query_cosine": query_cosine.detach(),
        "summary_cosine": summary_cosine.detach(),
        "descriptor_cosine": descriptor_cosine.detach(),
        "flat_token_cosine": flat_cosine.detach(),
        "query_rmse": (predicted_queries - target_queries).square().mean().sqrt().detach(),
        "summary_rmse": (predicted_summary - target_summary).square().mean().sqrt().detach(),
        "descriptor_retrieval_top1": retrieval_top1.detach(),
    }
    return total, metrics


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    grouped: Mapping[str, list[dict[str, Any]]],
    *,
    load_batch: Any,
    batches: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    for index in range(batches):
        rows = _sample_rows(
            grouped,
            batch_size=batch_size,
            seed=seed ^ 0x731A,
            step=index,
        )
        batch = load_batch(rows).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            _, metrics = _recovery_loss(model, batch)
        for key, value in metrics.items():
            totals[key] += float(value)
    model.train()
    return {key: value / batches for key, value in totals.items()}


def recover_dual_query_resampler(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["dual_query_resampler_recovery"])
    training = dict(cfg["training"])
    source_root = destination / str(cfg["source_directory"])
    cache_cfg = {
        "feature_directory": str(source_root / str(cfg.get("feature_directory", "style_features"))),
        "latent_directory": str(source_root / str(cfg.get("latent_directory", "latents"))),
    }
    feature_root, latent_root, rows = _intersect_cache_rows(destination, cache_cfg)
    token_root = source_root / str(cfg["teacher_token_cache"])
    rows = _attach_teacher_rows(rows, token_root / "manifest.parquet")
    seed = int(cfg.get("seed", 20260829))
    train_groups, validation_groups = _split_styles(
        rows,
        validation_styles=int(training.get("validation_styles", 32)),
        seed=seed,
    )
    device = str(training.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Resampler recovery")
    torch.manual_seed(seed)
    random.seed(seed)
    semantic_layers = tuple(
        int(value)
        for value in config["dual_query_resampler"]["model"].get(
            "semantic_layers", [18, 24]
        )
    )
    with safe_open(
        latent_root / str(rows[0]["latent_shard"]), framework="pt", device="cpu"
    ) as handle:
        vae_channels = int(handle.get_slice("latents").get_shape()[1])
    semantic_dim = int(rows[0]["spatial_dim"])
    model = _model_from_config(
        config["dual_query_resampler"], semantic_dim, vae_channels
    ).to(device)
    for module in (
        model.semantic_decoder_norm,
        model.semantic_decoder_heads,
        model.vae_decoder_norm,
        model.vae_decoder_head,
        model.global_decoder_bias,
    ):
        module.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    peak_lr = float(training.get("learning_rate", 1e-4))
    optimizer_kwargs: dict[str, Any] = {
        "lr": peak_lr,
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "betas": tuple(training.get("betas", [0.9, 0.95])),
        "eps": float(training.get("epsilon", 1e-8)),
    }
    if device.startswith("cuda") and bool(training.get("fused_adamw", True)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)

    print(f"loading {len(rows)} frozen teacher-token rows into host RAM", flush=True)
    target_tokens, target_descriptors, target_indices = _load_teacher_bank(
        token_root, rows, pin_memory=device.startswith("cuda")
    )
    print(
        f"teacher-token bank ready: {target_tokens.numel() * target_tokens.element_size() / 2**30:.2f} GiB",
        flush=True,
    )

    def load_batch(selected: list[dict[str, Any]]) -> RecoveryBatch:
        episode = _load_cache_episode(
            selected,
            feature_root,
            latent_root,
            semantic_layers,
            pin_memory=device.startswith("cuda"),
        )
        indices = torch.tensor(
            [target_indices[image_id] for image_id in episode.image_ids],
            dtype=torch.long,
        )
        return RecoveryBatch(
            inputs=episode,
            target_tokens=target_tokens.index_select(0, indices),
            target_descriptors=target_descriptors.index_select(0, indices),
        )

    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    history_path = output / "validation_history.json"
    history = json.loads(history_path.read_text("utf-8")) if history_path.exists() else []
    start_step = 0
    if state_path.exists() and bool(training.get("resume", True)):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"resuming Resampler recovery from step {start_step}", flush=True)

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "dual-query-resampler-recovery")),
            id=str(wandb_cfg.get("id", "dual-query-resampler-recovery")),
            resume="allow",
            config={
                "recovery": cfg,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "train_styles": len(train_groups),
                "validation_styles": len(validation_groups),
                "images": len(rows),
            },
        )

    steps = int(training.get("steps", 10_000))
    batch_size = int(training.get("batch_size", 16))
    prefetch = max(1, int(training.get("prefetch_batches", 4)))
    workers = max(1, int(training.get("prefetch_workers", 2)))
    log_every = int(training.get("log_every", 20))
    running: dict[str, torch.Tensor | float] = defaultdict(float)
    log_started = time.perf_counter()

    def load_step(step: int) -> RecoveryBatch:
        selected = _sample_rows(
            train_groups,
            batch_size=batch_size,
            seed=seed,
            step=step,
        )
        return load_batch(selected)

    futures: dict[int, Future[RecoveryBatch]] = {}
    next_submit = start_step

    def fill(executor: ThreadPoolExecutor, anchor: int) -> None:
        nonlocal next_submit
        while next_submit < steps and next_submit < anchor + prefetch:
            futures[next_submit] = executor.submit(load_step, next_submit)
            next_submit += 1

    model.train()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fill(executor, start_step)
        for step in range(start_step, steps):
            wait_started = time.perf_counter()
            batch = futures.pop(step).result().to(device)
            data_wait = time.perf_counter() - wait_started
            fill(executor, step + 1)
            learning_rate = _cosine_learning_rate(
                step,
                total_steps=steps,
                warmup_steps=int(training.get("warmup_steps", 250)),
                peak=peak_lr,
                minimum_ratio=float(training.get("minimum_lr_ratio", 0.1)),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                loss, metrics = _recovery_loss(model, batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite recovery loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(training.get("max_grad_norm", 1.0))
            )
            optimizer.step()
            for key, value in metrics.items():
                running[key] = running[key] + value.detach()
            running["grad_norm"] = running["grad_norm"] + grad_norm.detach()
            running["data_wait_s"] = float(running["data_wait_s"]) + data_wait
            completed = step + 1
            if completed % log_every == 0:
                elapsed = time.perf_counter() - log_started
                logged = {
                    f"train/{key}": float(value / log_every)
                    for key, value in running.items()
                    if key != "data_wait_s"
                }
                logged.update(
                    {
                        "train/learning_rate": learning_rate,
                        "perf/step_s": elapsed / log_every,
                        "perf/data_wait_s": float(running["data_wait_s"]) / log_every,
                    }
                )
                print(
                    f"recovery step={completed}/{steps} loss={logged['train/loss']:.4f} "
                    f"query_cos={logged['train/query_cosine']:.4f} "
                    f"summary_cos={logged['train/summary_cosine']:.4f} "
                    f"descriptor_cos={logged['train/descriptor_cosine']:.4f} "
                    f"step_s={logged['perf/step_s']:.3f} "
                    f"wait_s={logged['perf/data_wait_s']:.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(logged, step=completed)
                running.clear()
                log_started = time.perf_counter()
            validation_every = int(training.get("validation_every", 250))
            if validation_every and completed % validation_every == 0:
                validation = _validate(
                    model,
                    validation_groups,
                    load_batch=load_batch,
                    batches=int(training.get("validation_batches", 8)),
                    batch_size=batch_size,
                    device=device,
                    seed=seed,
                )
                history.append({"step": completed, **validation})
                write_json(history_path, history)
                print(
                    f"recovery validation step={completed} "
                    f"loss={validation['loss']:.4f} "
                    f"query_cos={validation['query_cosine']:.4f} "
                    f"summary_cos={validation['summary_cosine']:.4f} "
                    f"descriptor_cos={validation['descriptor_cosine']:.4f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"validation/{key}": value for key, value in validation.items()},
                        step=completed,
                    )
            checkpoint_every = int(training.get("checkpoint_every", 250))
            if checkpoint_every and completed % checkpoint_every == 0:
                payload = {
                    "step": completed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": dict(config["dual_query_resampler"]["model"]),
                    "recovery_config": cfg,
                }
                temporary = state_path.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(state_path)
                torch.save(
                    {key: value for key, value in payload.items() if key != "optimizer"},
                    checkpoints / f"step-{completed:06d}.pt",
                )
    if wandb_run is not None:
        wandb_run.finish()
    summary = {
        "steps": steps,
        "images": len(rows),
        "train_styles": len(train_groups),
        "validation_styles": len(validation_groups),
        "output_directory": str(output.resolve()),
        "last_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", summary)
    return summary
