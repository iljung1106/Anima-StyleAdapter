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

from .dual_query_resampler import (
    DualQueryResampler,
    episodic_angular_prototype_loss,
    supervised_contrastive_loss,
    token_diversity_loss,
)
from .io import read_records, write_json


@dataclass
class CacheEpisode:
    semantic_features: dict[int, torch.Tensor]
    semantic_mask: torch.Tensor
    semantic_grid_shapes: torch.Tensor
    vae_latents: torch.Tensor
    vae_shapes: torch.Tensor
    image_sizes: torch.Tensor
    labels: torch.Tensor
    class_labels: torch.Tensor
    image_ids: list[int]

    def to(self, device: str) -> "CacheEpisode":
        non_blocking = device.startswith("cuda")
        float_dtype = None if non_blocking else torch.float32

        def move(value: torch.Tensor) -> torch.Tensor:
            dtype = float_dtype if value.is_floating_point() else None
            return value.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return CacheEpisode(
            semantic_features={
                layer: move(value)
                for layer, value in self.semantic_features.items()
            },
            semantic_mask=move(self.semantic_mask),
            semantic_grid_shapes=move(self.semantic_grid_shapes),
            vae_latents=move(self.vae_latents),
            vae_shapes=move(self.vae_shapes),
            image_sizes=move(self.image_sizes),
            labels=move(self.labels),
            class_labels=move(self.class_labels),
            image_ids=self.image_ids,
        )


def _resolve_cache_path(destination: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else destination / path


def _intersect_cache_rows(
    destination: Path, cfg: Mapping[str, Any]
) -> tuple[Path, Path, list[dict[str, Any]]]:
    feature_root = _resolve_cache_path(destination, cfg["feature_directory"])
    latent_root = _resolve_cache_path(destination, cfg["latent_directory"])
    feature_rows = {
        int(row["id"]): row for row in read_records(feature_root / "manifest.parquet")
    }
    latent_rows = {
        int(row["id"]): row for row in read_records(latent_root / "manifest.parquet")
    }
    rows = []
    for image_id in sorted(feature_rows.keys() & latent_rows.keys()):
        feature = feature_rows[image_id]
        latent = latent_rows[image_id]
        feature_style = str(feature.get("style_id", feature["artist"]))
        latent_style = str(latent.get("style_id", latent["artist"]))
        if feature_style != latent_style:
            raise RuntimeError(f"Style mismatch for cached image {image_id}")
        rows.append(
            {
                "id": image_id,
                "style_id": feature_style,
                "artist": feature.get("artist", feature_style),
                "split": feature.get("split", latent.get("split", "train")),
                "feature_shard": feature["feature_shard"],
                "semantic_height": int(feature["target_height"]) // 16,
                "semantic_width": int(feature["target_width"]) // 16,
                "spatial_tokens": int(feature["spatial_tokens"]),
                "spatial_dim": int(feature["spatial_dim"]),
                "latent_shard": latent["cache_shard"],
                "latent_row": int(latent["row_index"]),
                "latent_height": int(latent["latent_height"]),
                "latent_width": int(latent["latent_width"]),
                "image_height": int(latent["target_height"]),
                "image_width": int(latent["target_width"]),
            }
        )
    if not rows:
        raise RuntimeError("C-RADIO and Qwen VAE caches do not share any image IDs")
    return feature_root, latent_root, rows


def _group_by_style(
    rows: list[dict[str, Any]],
    split: str,
    minimum_images: int,
    *,
    artist_limit: int | None = None,
    images_per_artist_limit: int | None = None,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["split"]) == split:
            grouped[str(row["style_id"])].append(row)
    eligible = {
        style: sorted(values, key=lambda row: int(row["id"]))
        for style, values in grouped.items()
        if len(values) >= minimum_images
    }
    generator = random.Random(seed)
    artists = sorted(eligible)
    if artist_limit is not None:
        if len(artists) < artist_limit:
            raise RuntimeError(
                f"Split {split!r} has {len(artists)} eligible artists, fewer than "
                f"the requested {artist_limit}"
            )
        artists = sorted(generator.sample(artists, artist_limit))
    selected = {}
    for artist in artists:
        values = eligible[artist]
        if images_per_artist_limit is not None and len(values) > images_per_artist_limit:
            values = generator.sample(values, images_per_artist_limit)
        selected[artist] = sorted(values, key=lambda row: int(row["id"]))
    if len(selected) < 2:
        raise RuntimeError(
            f"Split {split!r} needs at least two artists with {minimum_images} images"
        )
    return selected


def _episode_rows(
    grouped: Mapping[str, list[dict[str, Any]]],
    *,
    artists_per_batch: int,
    images_per_artist: int,
    seed: int,
    step: int,
) -> list[dict[str, Any]]:
    generator = random.Random(seed + step * 1_000_003)
    artists = sorted(grouped)
    selected = (
        generator.sample(artists, artists_per_batch)
        if len(artists) >= artists_per_batch
        else [generator.choice(artists) for _ in range(artists_per_batch)]
    )
    rows = []
    for label, artist in enumerate(selected):
        candidates = grouped[artist]
        chosen = (
            generator.sample(candidates, images_per_artist)
            if len(candidates) >= images_per_artist
            else [generator.choice(candidates) for _ in range(images_per_artist)]
        )
        rows.extend({**row, "episode_label": label} for row in chosen)
    return rows


def _load_cache_episode(
    rows: list[dict[str, Any]],
    feature_root: Path,
    latent_root: Path,
    semantic_layers: tuple[int, ...],
    *,
    pin_memory: bool,
) -> CacheEpisode:
    feature_values: dict[int, dict[int, torch.Tensor]] = {
        int(row["id"]): {} for row in rows
    }
    feature_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        feature_groups[str(row["feature_shard"])].append(row)
        latent_groups[str(row["latent_shard"])].append(row)
    for shard, shard_rows in feature_groups.items():
        with safe_open(feature_root / shard, framework="pt", device="cpu") as handle:
            for row in shard_rows:
                image_id = int(row["id"])
                for layer in semantic_layers:
                    key = f"{image_id}.layer_{layer:02d}_spatial"
                    feature_values[image_id][layer] = handle.get_tensor(key)
    latent_values: dict[int, torch.Tensor] = {}
    for shard, shard_rows in latent_groups.items():
        with safe_open(latent_root / shard, framework="pt", device="cpu") as handle:
            latents = handle.get_slice("latents")
            for row in shard_rows:
                latent_values[int(row["id"])] = latents[
                    int(row["latent_row"]) : int(row["latent_row"]) + 1
                ][0]

    maximum_tokens = max(int(row["spatial_tokens"]) for row in rows)
    semantic_dim = int(rows[0]["spatial_dim"])
    semantic = {
        layer: torch.zeros(
            len(rows), maximum_tokens, semantic_dim, dtype=torch.float16
        )
        for layer in semantic_layers
    }
    semantic_mask = torch.zeros(len(rows), maximum_tokens, dtype=torch.bool)
    maximum_latent_height = max(int(row["latent_height"]) for row in rows)
    maximum_latent_width = max(int(row["latent_width"]) for row in rows)
    vae_channels = int(next(iter(latent_values.values())).shape[0])
    vae_latents = torch.zeros(
        len(rows),
        vae_channels,
        maximum_latent_height,
        maximum_latent_width,
        dtype=torch.float16,
    )
    semantic_shapes = []
    vae_shapes = []
    image_sizes = []
    labels = []
    class_labels = []
    image_ids = []
    for index, row in enumerate(rows):
        image_id = int(row["id"])
        count = int(row["spatial_tokens"])
        expected = int(row["semantic_height"]) * int(row["semantic_width"])
        if count != expected:
            raise RuntimeError(f"Invalid C-RADIO grid metadata for image {image_id}")
        for layer in semantic_layers:
            value = feature_values[image_id][layer]
            if tuple(value.shape) != (count, semantic_dim):
                raise RuntimeError(
                    f"Unexpected layer {layer} shape for image {image_id}: {tuple(value.shape)}"
                )
            semantic[layer][index, :count].copy_(value)
        semantic_mask[index, :count] = True
        latent = latent_values[image_id]
        height, width = int(latent.shape[-2]), int(latent.shape[-1])
        vae_latents[index, :, :height, :width].copy_(latent)
        semantic_shapes.append((int(row["semantic_height"]), int(row["semantic_width"])))
        vae_shapes.append((height, width))
        image_sizes.append((int(row["image_height"]), int(row["image_width"])))
        labels.append(int(row["episode_label"]))
        class_labels.append(int(row.get("artist_class_label", -1)))
        image_ids.append(image_id)
    episode = CacheEpisode(
        semantic_features=semantic,
        semantic_mask=semantic_mask,
        semantic_grid_shapes=torch.tensor(semantic_shapes, dtype=torch.long),
        vae_latents=vae_latents,
        vae_shapes=torch.tensor(vae_shapes, dtype=torch.long),
        image_sizes=torch.tensor(image_sizes, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        class_labels=torch.tensor(class_labels, dtype=torch.long),
        image_ids=image_ids,
    )
    if pin_memory:
        episode.semantic_features = {
            layer: value.pin_memory() for layer, value in episode.semantic_features.items()
        }
        episode.semantic_mask = episode.semantic_mask.pin_memory()
        episode.semantic_grid_shapes = episode.semantic_grid_shapes.pin_memory()
        episode.vae_latents = episode.vae_latents.pin_memory()
        episode.vae_shapes = episode.vae_shapes.pin_memory()
        episode.image_sizes = episode.image_sizes.pin_memory()
        episode.labels = episode.labels.pin_memory()
        episode.class_labels = episode.class_labels.pin_memory()
    return episode


def _semantic_reconstruction_loss(
    predictions: Mapping[int, torch.Tensor],
    targets: Mapping[int, torch.Tensor],
    mask: torch.Tensor,
    *,
    sample_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_counts = mask.sum(dim=1).clamp_min(1)
    probabilities = (sample_tokens / valid_counts.float()).clamp_max(1.0)
    sampled = mask & (
        torch.rand(mask.shape, device=mask.device) < probabilities[:, None]
    )
    # The probability is only tiny for large grids; retain one deterministic token.
    sampled[:, 0] |= mask[:, 0]
    cosine_losses = []
    huber_losses = []
    for layer, prediction in predictions.items():
        target = targets[layer]
        cosine_losses.append(
            (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1))[
                sampled
            ].mean()
        )
        huber_losses.append(
            F.smooth_l1_loss(
                prediction[sampled].float(), target[sampled].float(), beta=0.1
            )
        )
    cosine = torch.stack(cosine_losses).mean()
    huber = torch.stack(huber_losses).mean()
    return cosine + 0.25 * huber, cosine, huber


def _vae_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shapes: torch.Tensor,
) -> torch.Tensor:
    height, width = prediction.shape[-2:]
    valid = torch.zeros(
        (prediction.shape[0], 1, height, width),
        device=prediction.device,
        dtype=torch.bool,
    )
    for index, shape in enumerate(shapes.detach().cpu().tolist()):
        valid[index, :, : int(shape[0]), : int(shape[1])] = True
    # A 2x low-pass target keeps the weak decoder focused on local appearance,
    # rather than turning the style encoder into a pixel-copy autoencoder.
    prediction_low = F.avg_pool2d(prediction.float(), 2, 2)
    target_low = F.avg_pool2d(target.float(), 2, 2)
    valid_low = F.avg_pool2d(valid.float(), 2, 2) > 0.99
    return F.smooth_l1_loss(
        prediction_low.expand_as(target_low)[valid_low.expand_as(target_low)],
        target_low[valid_low.expand_as(target_low)],
        beta=0.1,
    )


def _losses(
    model: DualQueryResampler,
    episode: CacheEpisode,
    cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model.encode(
        episode.semantic_features,
        episode.semantic_mask,
        episode.semantic_grid_shapes,
        episode.vae_latents,
        episode.vae_shapes,
        episode.image_sizes,
        reconstruct=True,
    )
    semantic, semantic_cosine, semantic_huber = _semantic_reconstruction_loss(
        output.semantic_reconstruction,
        episode.semantic_features,
        episode.semantic_mask,
        sample_tokens=int(cfg.get("semantic_reconstruction_sample_tokens", 192)),
    )
    vae = _vae_reconstruction_loss(
        output.vae_reconstruction, episode.vae_latents, episode.vae_shapes
    )
    prototype, prototype_metrics = episodic_angular_prototype_loss(
        output.descriptor,
        episode.labels,
        scale=float(cfg.get("prototype_scale", 16.0)),
        margin=float(cfg.get("prototype_margin", 0.10)),
    )
    contrastive = supervised_contrastive_loss(
        output.descriptor,
        episode.labels,
        temperature=float(cfg.get("contrastive_temperature", 0.10)),
    )
    proxy = output.descriptor.new_zeros(())
    proxy_top1 = output.descriptor.new_zeros(())
    if model.artist_proxies is not None and bool((episode.class_labels >= 0).all()):
        proxy, proxy_top1 = model.artist_proxy_loss(
            output.descriptor,
            episode.class_labels,
            scale=float(cfg.get("artist_proxy_scale", 16.0)),
            margin=float(cfg.get("artist_proxy_margin", 0.10)),
        )
    artist = (
        prototype
        + float(cfg.get("contrastive_fraction", 0.25)) * contrastive
        + float(cfg.get("artist_proxy_fraction", 0.50)) * proxy
    )
    diversity = token_diversity_loss(output.tokens)
    total = (
        semantic
        + float(cfg.get("vae_reconstruction_weight", 0.10)) * vae
        + float(cfg.get("artist_weight", 0.05)) * artist
        + float(cfg.get("token_diversity_weight", 0.01)) * diversity
    )
    metrics = {
        "loss": total,
        "semantic_reconstruction": semantic,
        "semantic_cosine": semantic_cosine,
        "semantic_huber": semantic_huber,
        "vae_reconstruction": vae,
        "artist": artist,
        "prototype": prototype,
        "supervised_contrastive": contrastive,
        "artist_proxy": proxy,
        "artist_proxy_top1": proxy_top1,
        "token_diversity": diversity,
        **prototype_metrics,
    }
    return total, metrics


def _model_from_config(cfg: Mapping[str, Any], semantic_dim: int, vae_channels: int):
    model_cfg = cfg["model"]
    return DualQueryResampler(
        semantic_layers=tuple(int(value) for value in model_cfg.get("semantic_layers", [18, 24])),
        semantic_dim=semantic_dim,
        vae_channels=vae_channels,
        dim=int(model_cfg.get("dim", 1024)),
        spatial_query_grid=int(model_cfg.get("spatial_query_grid", 8)),
        global_queries=int(model_cfg.get("global_queries", 16)),
        layers=int(model_cfg.get("layers", 4)),
        heads=int(model_cfg.get("heads", 16)),
        ff_dim=int(model_cfg.get("ff_dim", 4096)),
        artist_descriptor_dim=int(model_cfg.get("artist_descriptor_dim", 512)),
        artist_pooling_queries=int(model_cfg.get("artist_pooling_queries", 4)),
        artist_summary_tokens=int(model_cfg.get("artist_summary_tokens", 4)),
        artist_classes=int(cfg.get("training", {}).get("training_artist_count", 0)),
        semantic_dropout=float(model_cfg.get("semantic_dropout", 0.05)),
        vae_dropout=float(model_cfg.get("vae_dropout", 0.10)),
    )


def _cosine_learning_rate(
    step: int, *, total_steps: int, warmup_steps: int, peak: float, minimum_ratio: float
) -> float:
    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    multiplier = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
    )
    return peak * multiplier


@torch.no_grad()
def _validate(
    model: DualQueryResampler,
    grouped: Mapping[str, list[dict[str, Any]]],
    feature_root: Path,
    latent_root: Path,
    semantic_layers: tuple[int, ...],
    cfg: Mapping[str, Any],
    *,
    device: str,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    batches = int(cfg.get("validation_batches", 8))
    for index in range(batches):
        rows = _episode_rows(
            grouped,
            artists_per_batch=int(cfg["artists_per_batch"]),
            images_per_artist=int(cfg["images_per_artist"]),
            seed=seed ^ 0x71A5,
            step=index,
        )
        episode = _load_cache_episode(
            rows,
            feature_root,
            latent_root,
            semantic_layers,
            pin_memory=device.startswith("cuda"),
        ).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            _, metrics = _losses(model, episode, cfg)
        for key, value in metrics.items():
            totals[key] += float(value)
    model.train()
    return {key: value / batches for key, value in totals.items()}


def train_dual_query_resampler(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["dual_query_resampler"]
    training = cfg["training"]
    feature_root, latent_root, rows = _intersect_cache_rows(destination, cfg)
    semantic_layers = tuple(int(value) for value in cfg["model"].get("semantic_layers", [18, 24]))
    images_per_artist = int(training.get("images_per_artist", 2))
    image_limit = int(training.get("images_per_artist_limit", 15))
    train_groups = _group_by_style(
        rows,
        str(training.get("train_split", "train")),
        images_per_artist,
        artist_limit=int(training.get("training_artist_count", 3000)),
        images_per_artist_limit=image_limit,
        seed=int(cfg.get("seed", 20260815)) ^ 0x2A11,
    )
    train_groups = {
        artist: [
            {**row, "artist_class_label": class_index} for row in artist_rows
        ]
        for class_index, (artist, artist_rows) in enumerate(sorted(train_groups.items()))
    }
    validation_split = str(training.get("validation_split", "validation"))
    validation_groups = _group_by_style(
        rows,
        validation_split,
        images_per_artist,
        artist_limit=int(training.get("validation_artist_count", 150)),
        images_per_artist_limit=image_limit,
        seed=int(cfg.get("seed", 20260815)) ^ 0x7A11,
    )
    device = str(training.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured dual-query training")
    seed = int(cfg.get("seed", 20260815))
    torch.manual_seed(seed)
    random.seed(seed)
    first_latent_row = rows[0]
    with safe_open(
        latent_root / first_latent_row["latent_shard"], framework="pt", device="cpu"
    ) as handle:
        vae_channels = int(handle.get_slice("latents").get_shape()[1])
    semantic_dim = int(rows[0]["spatial_dim"])
    model = _model_from_config(cfg, semantic_dim, vae_channels).to(device)
    peak_lr = float(training.get("learning_rate", 1e-4))
    optimizer_kwargs: dict[str, Any] = {
        "lr": peak_lr,
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "betas": tuple(training.get("betas", [0.9, 0.95])),
        "eps": float(training.get("epsilon", 1e-8)),
    }
    if device.startswith("cuda") and bool(training.get("fused_adamw", True)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    output = destination / str(cfg.get("output_directory", "dual_query_resampler_bprime"))
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    history_path = output / "validation_history.json"
    history = json.loads(history_path.read_text("utf-8")) if history_path.exists() else []
    start_step = 0
    if state_path.exists() and bool(training.get("resume", True)):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"resuming dual-query Resampler from step {start_step}", flush=True)

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "dual-query-resampler-bprime")),
            id=str(wandb_cfg.get("id", "dual-query-resampler-bprime-v1")),
            resume="allow",
            config={
                "dual_query_resampler": cfg,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_artists": len(train_groups),
                "validation_artists": len(validation_groups),
                "train_images": sum(len(values) for values in train_groups.values()),
                "validation_images": sum(
                    len(values) for values in validation_groups.values()
                ),
            },
        )

    steps = int(training.get("steps", 20_000))
    artists_per_batch = int(training.get("artists_per_batch", 4))
    prefetch = max(1, int(training.get("prefetch_batches", 2)))
    workers = max(1, int(training.get("prefetch_workers", 2)))
    running: dict[str, torch.Tensor | float] = defaultdict(float)
    log_every = int(training.get("log_every", 20))
    log_started = time.perf_counter()

    def load_step(step: int) -> CacheEpisode:
        selected = _episode_rows(
            train_groups,
            artists_per_batch=artists_per_batch,
            images_per_artist=images_per_artist,
            seed=seed,
            step=step,
        )
        return _load_cache_episode(
            selected,
            feature_root,
            latent_root,
            semantic_layers,
            pin_memory=device.startswith("cuda"),
        )

    futures: dict[int, Future[CacheEpisode]] = {}
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
            episode = futures.pop(step).result().to(device)
            data_wait = time.perf_counter() - wait_started
            fill(executor, step + 1)
            learning_rate = _cosine_learning_rate(
                step,
                total_steps=steps,
                warmup_steps=int(training.get("warmup_steps", 500)),
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
                loss, metrics = _losses(model, episode, training)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite dual-query loss at step {step + 1}: "
                    + ", ".join(
                        f"{key}={float(value.detach())}"
                        for key, value in metrics.items()
                    )
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("max_grad_norm", 1.0))
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"Non-finite dual-query gradient norm at step {step + 1}"
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
                    f"dual-query step={completed}/{steps} loss={logged['train/loss']:.4f} "
                    f"sem={logged['train/semantic_reconstruction']:.4f} "
                    f"vae={logged['train/vae_reconstruction']:.4f} "
                    f"proto={logged['train/prototype']:.4f} "
                    f"top1={logged['train/prototype_top1']:.3f} "
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
                    feature_root,
                    latent_root,
                    semantic_layers,
                    training,
                    device=device,
                    seed=seed,
                )
                history.append({"step": completed, **validation})
                write_json(history_path, history)
                print(
                    f"validation step={completed} loss={validation['loss']:.4f} "
                    f"prototype_top1={validation['prototype_top1']:.3f} "
                    f"positive={validation['prototype_positive_cosine']:.3f} "
                    f"hard_negative={validation['prototype_hard_negative_cosine']:.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"validation/{key}": value for key, value in validation.items()},
                        step=completed,
                    )
            checkpoint_every = int(training.get("checkpoint_every", 500))
            if checkpoint_every and completed % checkpoint_every == 0:
                payload = {
                    "step": completed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": dict(cfg["model"]),
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
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_artists": len(train_groups),
        "validation_artists": len(validation_groups),
        "train_images": sum(len(values) for values in train_groups.values()),
        "validation_images": sum(len(values) for values in validation_groups.values()),
        "cache_images": len(rows),
        "output_directory": str(output.resolve()),
        "last_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_dual_query_resampler(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    torch.manual_seed(19)
    model = DualQueryResampler(
        semantic_layers=(18, 24),
        semantic_dim=12,
        vae_channels=4,
        dim=32,
        spatial_query_grid=4,
        global_queries=4,
        layers=2,
        heads=4,
        ff_dim=64,
        artist_descriptor_dim=16,
        artist_pooling_queries=2,
        artist_summary_tokens=2,
        semantic_dropout=0.05,
        vae_dropout=0.10,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        episode = CacheEpisode(
            semantic_features={
                18: torch.randn(4, 12, 12),
                24: torch.randn(4, 12, 12),
            },
            semantic_mask=torch.ones(4, 12, dtype=torch.bool),
            semantic_grid_shapes=torch.tensor([[3, 4]] * 4),
            vae_latents=torch.randn(4, 4, 8, 10),
            vae_shapes=torch.tensor([[8, 10]] * 4),
            image_sizes=torch.tensor([[64, 80]] * 4),
            labels=torch.tensor([0, 0, 1, 1]),
            class_labels=torch.full((4,), -1, dtype=torch.long),
            image_ids=[0, 1, 2, 3],
        )
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _losses(
            model,
            episode,
            {
                "semantic_reconstruction_sample_tokens": 12,
                "vae_reconstruction_weight": 0.10,
                "artist_weight": 0.05,
                "token_diversity_weight": 0.01,
            },
        )
        loss.backward()
        optimizer.step()
        losses.append(float(metrics["loss"].detach()))
    summary = {
        "steps": 2,
        "losses": losses,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite": all(math.isfinite(value) for value in losses),
    }
    write_json(destination / "dual_query_resampler_smoke.json", summary)
    return summary
