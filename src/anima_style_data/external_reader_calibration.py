"""Large-batch content-invariant calibration for external LoRA references."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json, write_records
from .kv_activation_generator import (
    _ema_checkpoint_state,
    _load_reader,
    _materialize_reference_token_bank,
    _open_direct_delta_reader,
    _resolved_experiment_config,
    _stage_reference_token_cache,
    _update_parameter_ema,
)


def _external_style_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    validation_single_styles: int,
    minimum_effect_rms: float,
) -> tuple[list[str], list[str], list[str]]:
    """Match the full trainer's component-disjoint external split exactly."""

    singles = [
        row
        for row in rows
        if str(row["kind"]) == "single"
        and bool(row.get("enabled", True))
        and float(row.get("effect_rms", 0.0)) >= minimum_effect_rms
    ]
    if not 0 < validation_single_styles < len(singles):
        raise ValueError("validation_single_styles is invalid")
    split_rng = random.Random(int(seed) ^ 0x45585641)
    validation_positions = set(
        split_rng.sample(range(len(singles)), validation_single_styles)
    )
    validation_rows = [
        row for index, row in enumerate(singles) if index in validation_positions
    ]
    heldout_components = {
        int(row["components"][0]) for row in validation_rows
    }
    train_rows = [
        row
        for row in rows
        if bool(row.get("enabled", True))
        and float(row.get("effect_rms", 0.0)) >= minimum_effect_rms
        and not any(
            int(component) in heldout_components for component in row["components"]
        )
    ]
    train_singles = [
        str(row["mixture_style_id"])
        for row in train_rows
        if str(row["kind"]) == "single"
    ]
    train_mixtures = [
        str(row["mixture_style_id"])
        for row in train_rows
        if str(row["kind"]) != "single"
    ]
    validation = [str(row["mixture_style_id"]) for row in validation_rows]
    if not train_singles or not train_mixtures or not validation:
        raise RuntimeError("External Reader calibration split is empty")
    return train_singles, train_mixtures, validation


def _retrieval(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric two-view InfoNCE with gradients through both Reader views."""

    left_unit = F.normalize(left.float().flatten(1), dim=-1)
    right_unit = F.normalize(right.float().flatten(1), dim=-1)
    cosine = left_unit @ right_unit.T
    labels = torch.arange(len(left), device=left.device)
    loss = 0.5 * (
        F.cross_entropy(cosine / temperature, labels)
        + F.cross_entropy(cosine.T / temperature, labels)
    )
    diagonal = cosine.diagonal()
    off_diagonal = ~torch.eye(
        len(left), device=left.device, dtype=torch.bool
    )
    wrong = cosine.masked_fill(~off_diagonal, -torch.inf)
    hardest_wrong = wrong.max(dim=1).values
    pairwise = (left_unit @ left_unit.T)[off_diagonal]
    return loss, {
        "loss": loss.detach(),
        "same_subset_cosine": diagonal.mean().detach(),
        "hardest_wrong_cosine": hardest_wrong.mean().detach(),
        "cosine_gap": (diagonal - hardest_wrong).mean().detach(),
        "retrieval_accuracy": (
            cosine.argmax(dim=1) == labels
        ).float().mean().detach(),
        "positive_pairwise_cosine": F.relu(pairwise).mean().detach(),
    }


def _reader_forward(
    reader: nn.Module,
    references: torch.Tensor,
    mask: torch.Tensor,
    parameters: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if parameters is None:
        return reader(references, mask).tokens
    return functional_call(reader, parameters, (references, mask)).tokens


@torch.no_grad()
def _evaluate(
    reader: nn.Module,
    bank: torch.Tensor,
    indices: list[int],
    *,
    temperature: float,
    parameters: dict[str, torch.Tensor] | None,
) -> dict[str, float]:
    """Average all three disjoint 2+2 partitions of four references."""

    partitions = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    selected = torch.tensor(indices, device=bank.device, dtype=torch.long)
    mask = torch.ones(len(indices), 2, device=bank.device, dtype=torch.bool)
    totals: dict[str, float] = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for left_slots, right_slots in partitions:
            left = _reader_forward(
                reader, bank[selected][:, left_slots], mask, parameters
            )
            right = _reader_forward(
                reader, bank[selected][:, right_slots], mask, parameters
            )
            _, metrics = _retrieval(left, right, temperature=temperature)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / len(partitions) for key, value in totals.items()}


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    reader: nn.Module,
    ema_reader: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    best_score: float,
    stale_validations: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": int(step),
            "reader": {
                key: value.detach().cpu()
                for key, value in reader.state_dict().items()
            },
            "ema_reader": {
                key: value.detach().cpu()
                for key, value in _ema_checkpoint_state(
                    reader, ema_reader
                ).items()
            },
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "best_score": float(best_score),
            "stale_validations": int(stale_validations),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    temporary.replace(path)


def train_external_lora_reader_calibration(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["external_lora_reader_calibration"])
    source_cfg = _resolved_experiment_config(
        config, str(cfg["source_experiment_key"])
    )
    device = str(cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", source_cfg.get("seed", 20260910)))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    external_cfg = dict(
        source_cfg["training"]["multi_domain_distillation"]["external_lora"]
    )
    functional_root = destination / str(external_cfg["functional_teacher_cache"])
    functional_rows = read_records(functional_root / "mixtures.parquet")
    train_singles, train_mixtures, validation_styles = _external_style_split(
        functional_rows,
        seed=seed,
        validation_single_styles=int(
            external_cfg.get("validation_single_styles", 32)
        ),
        minimum_effect_rms=float(external_cfg.get("minimum_effect_rms", 0.0)),
    )
    all_styles = train_singles + train_mixtures + validation_styles

    reference_root = destination / str(external_cfg["reference_cache"])
    local_cache = cfg.get("local_cache_directory")
    if local_cache:
        reference_root = _stage_reference_token_cache(
            reference_root, Path(str(local_cache))
        )
    loader = CachedTeacherReferenceLoader(
        reference_root,
        split="train",
        style_ids=all_styles,
        batch_size=int(cfg.get("materialization_chunk", 32)),
        references=4,
        seed=seed,
        token_lru_shards=int(cfg.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    bank = _materialize_reference_token_bank(
        loader,
        all_styles,
        references=4,
        seed=seed,
        chunk_size=int(cfg.get("materialization_chunk", 32)),
        device=device,
    )
    index_by_style = {style_id: index for index, style_id in enumerate(all_styles)}
    single_indices = [index_by_style[value] for value in train_singles]
    mixture_indices = [index_by_style[value] for value in train_mixtures]
    validation_indices = [index_by_style[value] for value in validation_styles]
    train_probe_indices = single_indices[:32] + mixture_indices[:32]

    reader = _load_reader(config, destination, source_cfg, device)
    reader = reader.to(dtype=torch.float32)
    parameters = _open_direct_delta_reader(reader)
    initial = torch.load(
        destination / str(cfg["initial_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    initial_key = (
        "ema_reader" if bool(cfg.get("initial_use_ema_reader", False)) else "reader"
    )
    reader.load_state_dict(initial[initial_key], strict=True)

    learning_rate = float(cfg.get("learning_rate", 2e-5))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=bool(cfg.get("fused_adamw", True)),
    )
    ema_decay = float(cfg.get("ema_decay", 0.995))
    ema_reader = {
        name: parameter.detach().float().clone()
        for name, parameter in reader.named_parameters()
        if parameter.requires_grad
    }

    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    best_score = -math.inf
    stale_validations = 0
    if bool(cfg.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        ema_reader = {
            name: state["ema_reader"][name].to(device=device, dtype=torch.float32)
            for name, _ in reader.named_parameters()
            if name in ema_reader
        }
        start_step = int(state["step"])
        best_score = float(state.get("best_score", -math.inf))
        stale_validations = int(state.get("stale_validations", 0))
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    steps = int(cfg.get("steps", 1000))
    batch_size = int(cfg.get("batch_size", 64))
    single_batch = min(
        int(round(batch_size * float(cfg.get("single_fraction", 0.5)))),
        len(single_indices),
    )
    mixture_batch = batch_size - single_batch
    if mixture_batch > len(mixture_indices):
        raise ValueError("Reader calibration mixture batch exceeds train styles")
    temperature = float(cfg.get("temperature", 0.07))
    cosine_floor = float(cfg.get("same_style_cosine_floor", 0.80))
    cosine_floor_weight = float(cfg.get("same_style_cosine_weight", 0.10))
    pairwise_cap = float(cfg.get("pairwise_cosine_cap", 0.70))
    pairwise_weight = float(cfg.get("pairwise_overload_weight", 0.05))
    rms_lower = float(cfg.get("rms_ratio_lower", 0.80))
    rms_upper = float(cfg.get("rms_ratio_upper", 1.25))
    rms_weight = float(cfg.get("rms_band_weight", 0.05))
    validation_every = int(cfg.get("validation_every", 50))
    checkpoint_every = int(cfg.get("checkpoint_every", 100))
    early_minimum = int(cfg.get("early_stop_minimum_step", 300))
    early_patience = int(cfg.get("early_stop_patience", 8))
    minimum_improvement = float(cfg.get("minimum_score_improvement", 0.002))
    warmup_steps = int(cfg.get("warmup_steps", 50))
    decay_start = int(cfg.get("decay_start_step", 500))
    final_lr = float(cfg.get("final_learning_rate", learning_rate * 0.1))
    max_grad_norm = float(cfg.get("max_grad_norm", 100.0))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline_mask = torch.ones(
            len(train_probe_indices), 2, device=device, dtype=torch.bool
        )
        baseline_memory = reader(
            bank[train_probe_indices, :2], baseline_mask
        ).tokens
        baseline_rms = float(baseline_memory.float().square().mean().sqrt())

    wandb_run = None
    wandb_cfg = dict(cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "external-reader-calibration")),
            id=str(wandb_cfg.get("id", "external-reader-calibration")),
            resume=str(wandb_cfg.get("resume", "allow")),
            config=cfg,
        )

    metrics_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed_step = start_step
    try:
        for step in range(start_step + 1, steps + 1):
            rng = random.Random(seed + step * 1_000_003)
            selected_indices = (
                rng.sample(single_indices, single_batch)
                + rng.sample(mixture_indices, mixture_batch)
            )
            rng.shuffle(selected_indices)
            left_slots = []
            right_slots = []
            for _ in selected_indices:
                slots = rng.sample(range(4), 4)
                left_slots.append(slots[:2])
                right_slots.append(slots[2:])
            rows = torch.tensor(selected_indices, device=device, dtype=torch.long)
            left_slot_tensor = torch.tensor(
                left_slots, device=device, dtype=torch.long
            )
            right_slot_tensor = torch.tensor(
                right_slots, device=device, dtype=torch.long
            )
            batch_rows = rows[:, None].expand(-1, 2)
            left_references = bank[batch_rows, left_slot_tensor]
            right_references = bank[batch_rows, right_slot_tensor]
            mask = torch.ones(batch_size, 2, device=device, dtype=torch.bool)

            if step <= warmup_steps:
                current_lr = learning_rate * step / max(1, warmup_steps)
            elif step <= decay_start:
                current_lr = learning_rate
            else:
                progress = min(
                    1.0, (step - decay_start) / max(1, steps - decay_start)
                )
                current_lr = final_lr + 0.5 * (learning_rate - final_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            optimizer.param_groups[0]["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                left_memory = reader(left_references, mask).tokens
                right_memory = reader(right_references, mask).tokens
                retrieval_loss, train_metrics = _retrieval(
                    left_memory, right_memory, temperature=temperature
                )
                left_unit = F.normalize(left_memory.float().flatten(1), dim=-1)
                right_unit = F.normalize(right_memory.float().flatten(1), dim=-1)
                same_cosine = (left_unit * right_unit).sum(dim=-1).mean()
                pairwise_mask = ~torch.eye(
                    batch_size, device=device, dtype=torch.bool
                )
                positive_pairwise = F.relu(
                    (left_unit @ left_unit.T)[pairwise_mask]
                ).mean()
                same_floor = F.relu(
                    cosine_floor - same_cosine
                ).square()
                pairwise_overload = F.relu(
                    positive_pairwise - pairwise_cap
                ).square()
                current_rms = 0.5 * (
                    left_memory.float().square().mean().sqrt()
                    + right_memory.float().square().mean().sqrt()
                )
                rms_ratio = current_rms / max(baseline_rms, 1e-8)
                rms_band = (
                    F.relu(rms_lower - rms_ratio).square()
                    + F.relu(rms_ratio - rms_upper).square()
                )
                loss = (
                    retrieval_loss
                    + cosine_floor_weight * same_floor
                    + pairwise_weight * pairwise_overload
                    + rms_weight * rms_band
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_grad_norm, foreach=True
            )
            optimizer.step()
            _update_parameter_ema(ema_reader, reader, ema_decay)
            completed_step = step

            if step == 1 or step % int(cfg.get("log_every", 10)) == 0:
                row = {
                    "step": step,
                    "train/loss": float(loss.detach()),
                    **{
                        f"train/{key}": float(value)
                        for key, value in train_metrics.items()
                    },
                    "train/same_floor": float(same_floor.detach()),
                    "train/pairwise_overload": float(pairwise_overload.detach()),
                    "train/rms_ratio": float(rms_ratio.detach()),
                    "train/rms_band": float(rms_band.detach()),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": current_lr,
                }
                metrics_rows.append(row)
                print(f"External Reader calibration {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {key: value for key, value in row.items() if key != "step"},
                        step=step,
                    )

            if step % validation_every == 0 or step == steps:
                reader.eval()
                raw_validation = _evaluate(
                    reader,
                    bank,
                    validation_indices,
                    temperature=temperature,
                    parameters=None,
                )
                ema_validation = _evaluate(
                    reader,
                    bank,
                    validation_indices,
                    temperature=temperature,
                    parameters=ema_reader,
                )
                train_validation = _evaluate(
                    reader,
                    bank,
                    train_probe_indices,
                    temperature=temperature,
                    parameters=ema_reader,
                )
                reader.train()
                score = (
                    ema_validation["cosine_gap"]
                    + 0.25 * ema_validation["retrieval_accuracy"]
                )
                improved = score > best_score + minimum_improvement
                if improved:
                    best_score = score
                    stale_validations = 0
                    _save_checkpoint(
                        output / "best.pt",
                        step=step,
                        reader=reader,
                        ema_reader=ema_reader,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_score=best_score,
                        stale_validations=stale_validations,
                    )
                else:
                    stale_validations += 1
                validation_row = {
                    "step": step,
                    **{
                        f"validation/raw/{key}": value
                        for key, value in raw_validation.items()
                    },
                    **{
                        f"validation/ema/{key}": value
                        for key, value in ema_validation.items()
                    },
                    **{
                        f"validation/train_probe/{key}": value
                        for key, value in train_validation.items()
                    },
                    "validation/score": score,
                    "validation/best_score": best_score,
                    "validation/stale": stale_validations,
                }
                metrics_rows.append(validation_row)
                print(f"External Reader validation {validation_row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            key: value
                            for key, value in validation_row.items()
                            if key != "step"
                        },
                        step=step,
                    )
                write_records(output / "metrics.parquet", metrics_rows)

            if step % checkpoint_every == 0 or step == steps:
                for checkpoint_path in (
                    state_path,
                    checkpoints / f"step-{step:07d}.pt",
                ):
                    _save_checkpoint(
                        checkpoint_path,
                        step=step,
                        reader=reader,
                        ema_reader=ema_reader,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_score=best_score,
                        stale_validations=stale_validations,
                    )
            if step >= early_minimum and stale_validations >= early_patience:
                print(
                    f"External Reader calibration plateau at step {step}; "
                    f"best_score={best_score:.6f}",
                    flush=True,
                )
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps_requested": steps,
        "steps_completed": completed_step,
        "train_single_styles": len(train_singles),
        "train_mixture_styles": len(train_mixtures),
        "validation_single_styles": len(validation_styles),
        "batch_size": batch_size,
        "references_per_view": 2,
        "best_score": best_score,
        "plateau": completed_step < steps,
        "elapsed_s": time.perf_counter() - started,
        "best_checkpoint": str(output / "best.pt"),
    }
    write_json(output / "summary.json", summary)
    return summary
