"""Distill visual K/V operators under real frozen-Anima query trajectories.

The earlier functional pilot evaluated teacher and student K/V under Gaussian
queries.  That makes a useful algebraic capacity probe, but it is not the
function used by Anima.  This module captures normalized native queries from
matched ``(x_t, timestep, prompt)`` trajectories, uses them for dense block
supervision, and periodically checks the complete frozen DiT velocity path.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json, write_records
from .kv_activation_generator import (
    ReferenceConditionedLowRankKVOperator,
    _NativeAttentionProbe,
    _functional_centered_attention_loss,
    _load_reader,
    _materialize_reference_token_bank,
    _mixture_target,
    _normalized_activation_loss,
    _save_training_state,
    _select_reference_tokens,
)
from .kv_activation_modulation import load_kv_lora_factor_bank


def _selected_content_indices(total: int, count: int) -> list[int]:
    if not 1 <= count <= total:
        raise ValueError(f"content count must be in [1, {total}]")
    if count == total:
        return list(range(total))
    return [
        int(value)
        for value in torch.linspace(0, total - 1, count).round().long().tolist()
    ]


def cache_real_anima_query_bank(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Capture post-q_norm Anima queries paired with cached text/noise rows."""

    from .style_transfer import _optimize_frozen_anima, _resolve_anima_model

    cfg = dict(config["kv_real_query_bank"])
    device = str(cfg.get("device", "cuda"))
    source_root = destination / str(cfg["source_cache"])
    source_path = source_root / "base.safetensors"
    source = load_file(source_path, device="cpu")
    contexts = source["base_context"]
    noisy = source["noisy_inputs"]
    timesteps = source["timesteps"].float()
    if noisy.shape[:2] != (len(contexts), len(timesteps)):
        raise RuntimeError("Functional source contexts/noisy/timesteps disagree")

    content_indices = _selected_content_indices(
        len(contexts), int(cfg.get("contents", 64))
    )
    queries_per_block = int(cfg.get("queries_per_block", 64))
    shard_contents = int(cfg.get("shard_contents", 4))
    forward_batch_rows = int(cfg.get("forward_batch_rows", 2))
    seed = int(cfg.get("seed", 20260825))
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.parquet"
    summary_path = output / "summary.json"

    expected_shards = math.ceil(len(content_indices) / shard_contents)
    if summary_path.exists() and manifest_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            int(summary.get("contents", -1)) == len(content_indices)
            and int(summary.get("shards", -1)) == expected_shards
            and all(
                (output / f"part-{index:04d}.safetensors").exists()
                for index in range(expected_shards)
            )
        ):
            return {**summary, "reused": True}

    anima = _resolve_anima_model(config, destination, device)
    anima.requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    heads = int(anima.blocks[0].cross_attn.n_heads)
    head_dim = int(anima.blocks[0].cross_attn.head_dim)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for block_index, block in enumerate(anima.blocks):
        def hook(_module, _inputs, result, *, index=block_index):
            captured[index] = result.detach()

        handles.append(block.cross_attn.q_norm.register_forward_hook(hook))

    rows: list[dict[str, Any]] = []
    stored_bytes = 0
    started = time.perf_counter()
    try:
        for shard_index, offset in enumerate(
            range(0, len(content_indices), shard_contents)
        ):
            path = output / f"part-{shard_index:04d}.safetensors"
            shard_indices = content_indices[offset : offset + shard_contents]
            if path.exists():
                tensors = load_file(path, device="cpu")
                if tensors["queries"].shape[:3] != (
                    len(shard_indices), len(timesteps), 28
                ):
                    raise RuntimeError(f"Stale real-Q shard shape: {path}")
                stored_bytes += path.stat().st_size
            else:
                queries = torch.empty(
                    len(shard_indices), len(timesteps), 28,
                    queries_per_block, heads, head_dim,
                    dtype=torch.float16,
                )
                conditions = [
                    (local, timestep_index, source_index)
                    for local, source_index in enumerate(shard_indices)
                    for timestep_index in range(len(timesteps))
                ]
                for batch_offset in range(0, len(conditions), forward_batch_rows):
                    batch_rows = conditions[
                        batch_offset : batch_offset + forward_batch_rows
                    ]
                    batch_noisy = torch.stack([
                        noisy[source_index, timestep_index]
                        for _, timestep_index, source_index in batch_rows
                    ]).to(device=device, dtype=torch.bfloat16)
                    batch_context = torch.stack([
                        contexts[source_index]
                        for _, _, source_index in batch_rows
                    ]).to(device=device, dtype=torch.bfloat16)
                    batch_timestep = torch.tensor(
                        [float(timesteps[timestep_index]) for _, timestep_index, _ in batch_rows],
                        device=device, dtype=torch.bfloat16,
                    )
                    padding = torch.zeros(
                        len(batch_rows), 1, batch_noisy.shape[-2], batch_noisy.shape[-1],
                        device=device, dtype=batch_noisy.dtype,
                    )
                    captured.clear()
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        anima(
                            batch_noisy.unsqueeze(2), batch_timestep,
                            context=batch_context, padding_mask=padding,
                            target_input_ids=None,
                        )
                    if len(captured) != 28:
                        raise RuntimeError(
                            f"Captured {len(captured)} of 28 Anima query tensors"
                        )
                    for block_index in range(28):
                        values = captured[block_index]
                        if values.ndim != 4 or values.shape[0] != len(batch_rows):
                            raise RuntimeError(
                                f"Unexpected block-{block_index} query shape {tuple(values.shape)}"
                            )
                        if values.shape[1] < queries_per_block:
                            raise RuntimeError("Requested more real queries than latent tokens")
                        for row_index, (local, timestep_index, source_index) in enumerate(batch_rows):
                            generator = torch.Generator(device="cpu").manual_seed(
                                seed + source_index * 1_000_003
                                + timestep_index * 10_007 + block_index * 101
                            )
                            positions = torch.randperm(
                                values.shape[1], generator=generator
                            )[:queries_per_block].to(values.device)
                            queries[local, timestep_index, block_index].copy_(
                                values[row_index].index_select(0, positions).to(
                                    device="cpu", dtype=torch.float16
                                )
                            )
                save_file({
                    "queries": queries.contiguous(),
                    "content_indices": torch.tensor(shard_indices, dtype=torch.int64),
                    "timesteps": timesteps,
                }, path)
                stored_bytes += path.stat().st_size

            for local, source_index in enumerate(shard_indices):
                rows.append({
                    "content_slot": offset + local,
                    "source_content_index": source_index,
                    "query_shard": path.name,
                    "query_row": local,
                })
            print(
                f"real Anima Q cache {offset + len(shard_indices)}/{len(content_indices)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
        del anima
        gc.collect()
        torch.cuda.empty_cache()

    write_records(manifest_path, rows)
    summary = {
        "contents": len(content_indices),
        "source_contents": len(contexts),
        "timesteps": [float(value) for value in timesteps],
        "blocks": 28,
        "queries_per_block": queries_per_block,
        "heads": heads,
        "head_dim": head_dim,
        "shards": expected_shards,
        "storage_bytes": stored_bytes,
        "source_cache": str(cfg["source_cache"]),
        "post_q_norm": True,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    return summary


class _RealQueryBank:
    def __init__(
        self,
        root: Path,
        source_root: Path,
        *,
        device: str,
        gpu_resident: bool,
    ) -> None:
        self.rows = sorted(
            read_records(root / "manifest.parquet"),
            key=lambda row: int(row["content_slot"]),
        )
        target = device if gpu_resident else "cpu"
        shards: dict[str, torch.Tensor] = {}
        query_rows = []
        source_indices = []
        for row in self.rows:
            name = str(row["query_shard"])
            if name not in shards:
                shards[name] = load_file(root / name, device="cpu")["queries"].to(target)
            query_rows.append(shards[name][int(row["query_row"])])
            source_indices.append(int(row["source_content_index"]))
        self.queries = torch.stack(query_rows)
        self.source_indices = torch.tensor(source_indices, dtype=torch.long)
        source = load_file(source_root / "base.safetensors", device="cpu")
        self.contexts = source["base_context"].index_select(
            0, self.source_indices
        ).to(target)
        self.noisy = source["noisy_inputs"].index_select(
            0, self.source_indices
        )
        self.base_predictions = source["base_predictions"].index_select(
            0, self.source_indices
        )
        self.timesteps = source["timesteps"].float()
        self.device = device

    def query(self, content: int, timestep: int, block: int) -> torch.Tensor:
        return self.queries[content, timestep, block].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True
        )

    def context(self, content: int) -> torch.Tensor:
        return self.contexts[content].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True
        )


def _last_pooling_parameters(reader: torch.nn.Module) -> list[torch.nn.Parameter]:
    selected = []
    for name, parameter in reader.named_parameters():
        trainable = name == "mixers" or name.startswith("mixers.")
        parameter.requires_grad_(False)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("Reader exposes no final mixer parameters")
    return selected


def _operator_factors(
    model: ReferenceConditionedLowRankKVOperator,
    style: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    down_rows = []
    up_rows = []
    for block in range(model.blocks):
        down, up, sigma = model._operator(style, block)
        down_rows.append(down)
        up_rows.append(up.transpose(-1, -2) * sigma[:, :, None, :])
    return torch.stack(down_rows, dim=1), torch.stack(up_rows, dim=1)


def train_real_query_reference_kv_operator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Train dense real-Q block effects plus sparse full-DiT velocity effects."""

    from .kv_activation_sampling import NativeKVFactorInjector
    from .style_transfer import _optimize_frozen_anima, _resolve_anima_model

    cfg = copy.deepcopy(config["kv_reference_real_query_operator"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 4000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260825))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)), dtype=torch.float16,
    )
    validation_artists = int(training.get("validation_artists", 32))
    train_artist_count = len(artist_ids) - validation_artists
    if train_artist_count < int(training.get("batch_size", 8)):
        raise ValueError("Not enough training artists for a controlled batch")
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    reader = _load_reader(config, destination, cfg, device)
    reader_parameters = _last_pooling_parameters(reader)
    reader_unfreeze_step = int(training.get("reader_unfreeze_step", 1500))

    chunk = int(training.get("materialization_style_chunk", 16))
    human_images = int(training.get("human_reference_images", 12))
    human_train_images = int(training.get("human_train_reference_images", 8))
    synthetic_images = int(training.get("synthetic_reference_images", 8))
    synthetic_train_images = int(training.get("synthetic_train_reference_images", 6))

    def materialize(cache: str, images: int, domain_seed: int) -> torch.Tensor:
        loader = CachedTeacherReferenceLoader(
            destination / cache, split="train", style_ids=artist_ids,
            batch_size=chunk, references=images, seed=domain_seed,
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            strict_style_ids=True,
        )
        return _materialize_reference_token_bank(
            loader, artist_ids, references=images, seed=domain_seed,
            chunk_size=chunk, device=device,
        )

    reference_banks = {
        "human": materialize(
            str(cfg["human_reference_cache"]), human_images, seed ^ 0x48554D41
        ),
        "synthetic": materialize(
            str(cfg["synthetic_reference_cache"]), synthetic_images,
            seed ^ 0x53594E54,
        ),
    }
    query_cfg = dict(config["kv_real_query_bank"])
    query_bank = _RealQueryBank(
        destination / str(cfg["query_cache"]),
        destination / str(query_cfg["source_cache"]),
        device=device,
        gpu_resident=bool(training.get("gpu_resident_queries", True)),
    )
    heldout_contents = int(training.get("heldout_contents", 8))
    train_content_count = len(query_bank.rows) - heldout_contents
    if train_content_count <= 0:
        raise ValueError("heldout_contents leaves no real-Q training conditions")

    anima = _resolve_anima_model(config, destination, device)
    anima.requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    probes = torch.nn.ModuleList(
        _NativeAttentionProbe(block.cross_attn) for block in anima.blocks
    ).to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
    injector = NativeKVFactorInjector(anima)

    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture"))
    if architecture != "bilinear_low_rank_operator":
        raise ValueError("Real-Q training requires the bilinear operator")
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=int(reader.dim), context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]), blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)

    operator_lr = float(training.get("operator_learning_rate", 2e-4))
    reader_lr = float(training.get("reader_learning_rate", 5e-6))
    optimizer = torch.optim.AdamW(
        [
            {"name": "operator", "params": list(model.parameters()), "lr": operator_lr},
            {"name": "reader_mixer", "params": reader_parameters, "lr": reader_lr},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    reference_values = [int(value) for value in training.get("reference_counts", [1, 2, 4])]
    reference_weights = [float(value) for value in training.get("reference_count_weights", [0.5, 0.3, 0.2])]
    dense_cfg = dict(training.get("dense_functional_loss", {}))
    final_cfg = dict(training.get("final_velocity_loss", {}))
    functional_weight = float(training.get("functional_weight", 1.0))
    kv_aux_weight = float(training.get("kv_auxiliary_weight", 0.05))
    final_weight = float(training.get("final_velocity_weight", 0.25))
    final_every = int(training.get("final_velocity_every", 8))
    final_start = int(training.get("final_velocity_start_step", 500))
    final_batch = int(training.get("final_velocity_batch_size", 2))
    warmup = int(training.get("warmup_steps", 100))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    reader_max_grad_norm = float(training.get("reader_max_grad_norm", 0.10))

    def loss_kwargs(values: dict[str, Any]) -> dict[str, float]:
        return {
            "centered_huber_weight": float(values.get("centered_huber", 1.0)),
            "direction_weight": float(values.get("direction", 1.0)),
            "magnitude_weight": float(values.get("magnitude", 0.2)),
            "relation_weight": float(values.get("relation", 0.5)),
            "raw_huber_weight": float(values.get("raw_huber", 0.05)),
            "temperature": float(values.get("temperature", 0.1)),
        }

    def dense_loss(
        style: torch.Tensor, context: torch.Tensor, artists: torch.Tensor,
        content_index: int, timestep_index: int, block: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_delta = model(style, context, block)
        teacher_delta = _mixture_target(
            context, teacher_down, teacher_up, artists[:, None],
            torch.ones(len(artists), 1, device=device), block,
        )
        probe = probes[block]
        queries = query_bank.query(content_index, timestep_index, block)[None].expand(
            len(artists), -1, -1, -1
        )
        zero = torch.zeros_like(student_delta)
        base_key, base_value = probe.project_context(context, zero)
        student_key, student_value = probe.project_context(context, student_delta)
        teacher_key, teacher_value = probe.project_context(context, teacher_delta)
        base_output = probe.attend(
            queries, base_key, base_value, queries_normalized=True
        )
        student_effect = probe.attend(
            queries, student_key, student_value, queries_normalized=True
        ) - base_output
        teacher_effect = probe.attend(
            queries, teacher_key, teacher_value, queries_normalized=True
        ) - base_output
        functional, values = _functional_centered_attention_loss(
            student_effect, teacher_effect, **loss_kwargs(dense_cfg)
        )
        kv_loss, kv_values = _normalized_activation_loss(
            student_delta, teacher_delta,
            direction_weight=float(training.get("kv_direction_weight", 0.05)),
            magnitude_weight=float(training.get("kv_magnitude_weight", 0.01)),
        )
        values.update({f"kv_{key}": value for key, value in kv_values.items()})
        values["weighted_dense_functional"] = functional.detach() * functional_weight
        values["weighted_kv_auxiliary"] = kv_loss.detach() * kv_aux_weight
        return functional * functional_weight + kv_loss * kv_aux_weight, values

    def velocity_loss(
        style: torch.Tensor, context: torch.Tensor, artists: torch.Tensor,
        content_index: int, timestep_index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = min(final_batch, len(artists))
        style = style[:count]
        context = context[:count]
        artists = artists[:count]
        noisy = query_bank.noisy[content_index, timestep_index].to(
            device=device, dtype=torch.bfloat16
        )[None].expand(count, -1, -1, -1)
        timestep_value = float(query_bank.timesteps[timestep_index])
        timestep = torch.full(
            (count,), timestep_value, device=device, dtype=torch.bfloat16
        )
        padding = torch.zeros(
            count, 1, noisy.shape[-2], noisy.shape[-1],
            device=device, dtype=noisy.dtype,
        )
        base = query_bank.base_predictions[content_index, timestep_index].to(
            device=device, dtype=torch.float32
        )[None].expand(count, -1, -1, -1)
        injector.set_factors(
            teacher_down.index_select(0, artists),
            teacher_up.index_select(0, artists), strength=1.0,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_prediction = anima(
                noisy.unsqueeze(2), timestep, context=context,
                padding_mask=padding, target_input_ids=None,
            ).squeeze(2).float()
        predicted_down, predicted_up = _operator_factors(model, style)
        injector.set_factors(predicted_down, predicted_up, strength=1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_prediction = anima(
                noisy.unsqueeze(2), timestep, context=context,
                padding_mask=padding, target_input_ids=None,
            ).squeeze(2).float()
        injector.disable()
        loss, values = _functional_centered_attention_loss(
            student_prediction - base,
            teacher_prediction - base,
            **loss_kwargs(final_cfg),
        )
        return loss, {f"final_{key}": value for key, value in values.items()}

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-real-query-operator")),
            id=str(wandb_cfg.get("id", "kv-real-query-operator")),
            resume="allow" if start_step else "never",
            config={"kv_reference_real_query_operator": cfg},
        )

    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            reader_open = step >= reader_unfreeze_step
            for parameter in reader_parameters:
                parameter.requires_grad_(reader_open)
            reader.train(reader_open)
            rng = random.Random(seed + step * 1_000_003)
            domain = "human" if step % 2 else "synthetic"
            bank = reference_banks[domain]
            train_stop = (
                human_train_images if domain == "human" else synthetic_train_images
            )
            artists = rng.sample(range(train_artist_count), batch)
            counts = rng.choices(reference_values, weights=reference_weights, k=batch)
            references, reference_mask = _select_reference_tokens(
                bank, artists, reference_counts=counts,
                reference_start=0, reference_stop=train_stop, rng=rng,
            )
            content_index = rng.randrange(train_content_count)
            timestep_index = rng.randrange(len(query_bank.timesteps))
            context = query_bank.context(content_index)[None].expand(batch, -1, -1)
            artist_tensor = torch.tensor(artists, device=device, dtype=torch.long)
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            blocks = [
                (first_block + offset) % model.blocks
                for offset in range(blocks_per_step)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = reader(references, reference_mask).tokens
                dense_rows = [
                    dense_loss(
                        style, context, artist_tensor, content_index,
                        timestep_index, block,
                    )
                    for block in blocks
                ]
                loss = torch.stack([row[0] for row in dense_rows]).mean()
                final_values = None
                if (
                    final_every > 0 and step >= final_start
                    and step % final_every == 0
                ):
                    final_loss, final_values = velocity_loss(
                        style, context, artist_tensor,
                        content_index, timestep_index,
                    )
                    loss = loss + final_weight * final_loss
            loss.backward()
            operator_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            reader_grad = torch.nn.utils.clip_grad_norm_(
                reader_parameters, reader_max_grad_norm, foreach=True
            ) if reader_open else torch.zeros(())
            lr_scale = min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = operator_lr * lr_scale
            optimizer.param_groups[1]["lr"] = reader_lr * lr_scale if reader_open else 0.0
            optimizer.step()

            running["loss"].append(float(loss.detach()))
            running["operator_grad_norm"].append(float(operator_grad))
            running["reader_grad_norm"].append(float(reader_grad))
            running["reader_open"].append(float(reader_open))
            running[f"domain_{domain}"].append(1.0)
            for key in dense_rows[0][1]:
                running[key].append(
                    sum(float(row[1][key]) for row in dense_rows) / len(dense_rows)
                )
            if final_values is not None:
                for key, value in final_values.items():
                    running[key].append(float(value))
                running["weighted_final_velocity"].append(
                    float(final_values["final_functional_loss"]) * final_weight
                )

            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["operator_lr"] = optimizer.param_groups[0]["lr"]
                row["reader_lr"] = optimizer.param_groups[1]["lr"]
                print(f"Real-Q K/V operator step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()

            if validation_every > 0 and step % validation_every == 0:
                model.eval()
                reader.eval()
                validation: dict[str, list[float]] = defaultdict(list)
                val_artists = list(range(train_artist_count, len(artist_ids)))[:16]
                val_indices = torch.tensor(val_artists, device=device, dtype=torch.long)
                val_rng = random.Random(seed ^ step)
                content_index = train_content_count + (
                    step // validation_every
                ) % heldout_contents
                timestep_index = (step // validation_every) % len(query_bank.timesteps)
                val_context = query_bank.context(content_index)[None].expand(
                    len(val_artists), -1, -1
                )
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for domain, bank in reference_banks.items():
                        start = human_train_images if domain == "human" else synthetic_train_images
                        stop = human_images if domain == "human" else synthetic_images
                        refs, mask = _select_reference_tokens(
                            bank, val_artists,
                            reference_counts=[min(2, stop - start)] * len(val_artists),
                            reference_start=start, reference_stop=stop, rng=val_rng,
                        )
                        val_style = reader(refs, mask).tokens
                        for block in range(model.blocks):
                            _, values = dense_loss(
                                val_style, val_context, val_indices,
                                content_index, timestep_index, block,
                            )
                            for key, value in values.items():
                                validation[f"{domain}/{key}"].append(float(value))
                metrics = {
                    key: sum(values) / len(values)
                    for key, values in validation.items()
                }
                print(f"Real-Q validation step={step} {metrics}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
                model.train()

            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path, step=step, model=model, reader=reader,
                        optimizer=optimizer, cfg=cfg,
                    )
    finally:
        injector.close()
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps": steps,
        "artists": len(artist_ids),
        "training_artists": train_artist_count,
        "validation_artists": validation_artists,
        "query_contents": len(query_bank.rows),
        "real_queries": True,
        "synthetic_and_human_domains_separate": True,
        "reader_unfreeze_step": reader_unfreeze_step,
        "final_velocity_every": final_every,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_real_query_reference_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_real_query_operator"]
    cfg["output_directory"] = "kv_reference_real_query_operator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["steps"] = 2
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["final_velocity_start_step"] = 1
    cfg["training"]["final_velocity_every"] = 1
    cfg["training"]["final_velocity_batch_size"] = 2
    cfg["training"]["reader_unfreeze_step"] = 2
    cfg["training"]["wandb"]["enabled"] = False
    return train_real_query_reference_kv_operator(
        effective, destination, steps_override=2
    )


def sample_external_reference_real_query_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    from .kv_activation_generator import (
        _sample_external_reference_bilinear_kv_operator,
    )

    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_real_query_operator_fixed_sample",
    )
