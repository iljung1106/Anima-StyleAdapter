"""Reconstruction pretraining for the detail-preserving style Reader."""

from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction_f = prediction.float()
    target_f = target.float()
    cosine = 1.0 - (
        F.normalize(prediction_f, dim=-1) * F.normalize(target_f, dim=-1)
    ).sum(dim=-1).mean()
    huber = F.smooth_l1_loss(prediction_f, target_f, beta=0.1)
    return cosine + 0.1 * huber, cosine.detach(), huber.detach()


def _references(batch: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["reference_mask"].to(device, non_blocking=True)
    flat = batch["cached_reference_tokens"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    references = flat.reshape(mask.shape[0], mask.shape[1], *flat.shape[1:])
    return references, mask


def _save(path: Path, step: int, reader: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "reader": {key: value.detach().cpu() for key, value in reader.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _evaluate(reader: DetailPreservingTypedSlotReader, loader: CachedTeacherReferenceLoader, device: str, batches: int) -> dict[str, float]:
    reader.eval()
    totals: Counter[str] = Counter()
    for index in range(batches):
        references, mask = _references(loader.load_step(index), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            output = reader(references, mask, reconstruct=True)
        per, per_cos, per_huber = _loss(output.reconstruction, output.reconstruction_target)
        pooled, pooled_cos, pooled_huber = _loss(
            output.pooled_reconstruction, output.pooled_reconstruction_target
        )
        for key, value in {
            "loss": per + pooled,
            "per_reference_cosine": per_cos,
            "per_reference_huber": per_huber,
            "pooled_cosine": pooled_cos,
            "pooled_huber": pooled_huber,
            "style_token_rms": output.tokens.float().square().mean().sqrt(),
        }.items():
            totals[key] += float(value)
    reader.train()
    return {key: value / batches for key, value in totals.items()}


def pretrain_detail_style_reader(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    cfg = dict(config["detail_preserving_style_cross_attention"])
    training = dict(cfg["reader_pretraining"])
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 0)) ^ 0x51EAD
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))

    token_root = destination / str(cfg["cache"]["output_directory"])
    rows = read_records(token_root / "manifest.parquet")
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        counts.setdefault(str(row.get("split", "train")), Counter())[str(row["style_id"])] += 1
    references = int(training.get("references", 4))
    train_split = str(cfg.get("train_split", "train"))
    validation_split = str(cfg.get("validation_split", "validation"))
    train_styles = sorted(style for style, count in counts.get(train_split, {}).items() if count >= references)
    validation_styles = sorted(style for style, count in counts.get(validation_split, {}).items() if count >= references)
    loader_kwargs = {
        "token_root": token_root,
        "batch_size": int(training.get("batch_size", 32)),
        "references": references,
        "token_lru_shards": int(training.get("token_lru_shards", 8)),
        "ram_resident_tokens": bool(training.get("ram_resident_tokens", True)),
        "ram_preload_workers": int(training.get("ram_preload_workers", 8)),
    }
    train_loader = CachedTeacherReferenceLoader(
        split=train_split, style_ids=train_styles, seed=seed, **loader_kwargs
    )
    validation_loader = CachedTeacherReferenceLoader(
        split=validation_split, style_ids=validation_styles, seed=seed ^ 0xA11CE,
        **loader_kwargs,
    )

    reader = DetailPreservingTypedSlotReader(**dict(cfg["model"])).to(device)
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=float(training.get("learning_rate", 1e-4)),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True) and device.startswith("cuda")),
    )
    output = destination / str(training["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "detail-style-reader-pretrain")),
            id=str(wandb_cfg.get("id", "detail-style-reader-pretrain")),
            resume="allow" if start_step else "never",
            config={"reader_pretraining": training, "model": cfg["model"]},
        )

    steps = int(training.get("steps", 4_000))
    warmup = int(training.get("warmup_steps", 200))
    pooled_weight = float(training.get("pooled_reconstruction_weight", 1.0))
    prefetch = train_loader.prefetch(
        start_step, steps - start_step,
        workers=int(training.get("prefetch_workers", 2)),
        depth=int(training.get("prefetch_batches", 6)),
    )
    started = time.perf_counter()
    for step, batch in enumerate(prefetch, start=start_step + 1):
        ratio = min(1.0, step / max(1, warmup))
        for group in optimizer.param_groups:
            group["lr"] = float(training.get("learning_rate", 1e-4)) * ratio
        references_tensor, mask = _references(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            result = reader(references_tensor, mask, reconstruct=True)
            per, per_cos, per_huber = _loss(result.reconstruction, result.reconstruction_target)
            pooled, pooled_cos, pooled_huber = _loss(
                result.pooled_reconstruction, result.pooled_reconstruction_target
            )
            loss = per + pooled_weight * pooled
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), float(training.get("max_grad_norm", 1.0)))
        optimizer.step()
        if step % int(training.get("log_every", 10)) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            metrics = {
                "loss": float(loss.detach()), "per_reference_cosine": float(per_cos),
                "per_reference_huber": float(per_huber), "pooled_cosine": float(pooled_cos),
                "pooled_huber": float(pooled_huber), "grad_norm": float(grad_norm),
                "style_token_rms": float(result.tokens.detach().float().square().mean().sqrt()),
                "steps_per_s": (step - start_step) / elapsed,
            }
            print(f"detail-reader-pretrain step={step}/{steps} {metrics}", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"reader_pretrain/train/{key}": value for key, value in metrics.items()}, step=step)
        if step % int(training.get("validation_every", 250)) == 0 or step == steps:
            validation = _evaluate(reader, validation_loader, device, int(training.get("validation_batches", 8)))
            print(f"detail-reader-pretrain validation step={step} {validation}", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"reader_pretrain/val/{key}": value for key, value in validation.items()}, step=step)
        if step % int(training.get("checkpoint_every", 500)) == 0 or step == steps:
            _save(checkpoints / f"step-{step:07d}.pt", step, reader, optimizer, training)
            _save(state_path, step, reader, optimizer, training)
    if wandb_run is not None:
        wandb_run.finish()
    summary = {
        "steps": steps,
        "train_styles": len(train_styles),
        "validation_styles": len(validation_styles),
        "checkpoint": str(checkpoints / f"step-{steps:07d}.pt"),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary
