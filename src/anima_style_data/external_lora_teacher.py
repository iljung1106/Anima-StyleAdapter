"""Whole-model functional caches for heterogeneous external Anima LoRAs."""

from __future__ import annotations

import gc
import json
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file

from .io import read_records, write_json, write_records
from .lora_functional_distillation import (
    MixtureSpec,
    _cached_training_probe_bank,
    _preview_pixels,
    _predict_frozen_anima_in_chunks,
    build_mixture_specs,
)
from .style_calibration import _encode_prompts
from .style_transfer import (
    _load_sampling_vae,
    _optimize_frozen_anima,
    _resolve_anima_model,
)
from .synthetic_teacher import _sample_anima_batch


class _EmbeddingLoRA(torch.nn.Module):
    """Factorized LoRA delta for an ``nn.Embedding`` weight matrix."""

    def __init__(
        self,
        lora_name: str,
        embedding: torch.nn.Embedding,
        down: torch.Tensor,
        up: torch.Tensor,
        alpha: torch.Tensor | None,
    ) -> None:
        super().__init__()
        rank = int(down.shape[0])
        if tuple(down.shape) != (rank, embedding.embedding_dim):
            raise RuntimeError(
                f"{lora_name} down shape {tuple(down.shape)} does not match "
                f"Embedding(*, {embedding.embedding_dim})"
            )
        if tuple(up.shape) != (embedding.num_embeddings, rank):
            raise RuntimeError(
                f"{lora_name} up shape {tuple(up.shape)} does not match "
                f"Embedding({embedding.num_embeddings}, *)"
            )
        self.lora_name = lora_name
        self.lora_down = torch.nn.Linear(embedding.embedding_dim, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, embedding.num_embeddings, bias=False)
        self.lora_down.weight.data.copy_(down)
        self.lora_up.weight.data.copy_(up)
        alpha_value = rank if alpha is None or float(alpha) == 0 else float(alpha)
        self.register_buffer("alpha", torch.tensor(alpha_value), persistent=True)
        self.scale = alpha_value / rank
        self.multiplier = 1.0
        self.org_module = [embedding]
        self.org_forward = embedding.forward

    def apply_to(self) -> None:
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self) -> None:
        self.org_module[0].forward = self.org_forward

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        base = self.org_forward(input, *args, **kwargs)
        low_rank_rows = torch.nn.functional.embedding(input, self.lora_up.weight)
        delta = torch.nn.functional.linear(
            low_rank_rows, self.lora_down.weight.transpose(0, 1)
        )
        return base + delta.to(base.dtype) * self.multiplier * self.scale


class _ExternalAdapterPool:
    """Construct each external adapter once and keep inactive weights in CPU RAM."""

    def __init__(
        self,
        config: dict[str, Any],
        anima: torch.nn.Module,
        device: str,
        *,
        prefetch_workers: int = 4,
    ) -> None:
        self.anima = anima
        self.device = device
        self.networks: dict[int, torch.nn.Module] = {}
        self.pending: dict[int, Future[tuple[torch.nn.Module, int]]] = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, int(prefetch_workers)),
            thread_name_prefix="external-lora",
        )
        self.embedding_targets = {
            "lora_unet_" + name.replace(".", "_"): module
            for name, module in anima.named_modules()
            if isinstance(module, torch.nn.Embedding)
        }
        self.cache_hits = 0
        self.cache_misses = 0
        sd_root = Path(str(config["anima_cache"]["sd_scripts_path"])).resolve()
        if str(sd_root) not in sys.path:
            sys.path.insert(0, str(sd_root))
        try:
            from lycoris.kohya import create_network_from_weights
        except ImportError as error:
            raise RuntimeError(
                "External LoRA caching requires lycoris_lora>=3.4"
            ) from error
        self.create_network_from_weights = create_network_from_weights

    def _build(self, item: dict[str, Any]) -> tuple[torch.nn.Module, int]:
        state = _normalize_external_lora_state(
            load_file(str(item["resolved_weight_path"]), device="cpu")
        )
        expected = _expected_unet_modules(state)
        factory_state = dict(state)
        embedding_loras = []
        for lora_name, embedding in self.embedding_targets.items():
            up_key = f"{lora_name}.lora_up.weight"
            down_key = f"{lora_name}.lora_down.weight"
            if up_key not in factory_state:
                continue
            embedding_loras.append(
                _EmbeddingLoRA(
                    lora_name,
                    embedding,
                    factory_state[down_key],
                    factory_state[up_key],
                    factory_state.get(f"{lora_name}.alpha"),
                )
            )
            for suffix in ("lora_down.weight", "lora_up.weight", "alpha"):
                factory_state.pop(f"{lora_name}.{suffix}", None)
        network, _ = self.create_network_from_weights(
            1.0,
            str(item["resolved_weight_path"]),
            None,
            [],
            self.anima,
            weights_sd=factory_state,
            for_inference=True,
        )
        network.unet_loras.extend(embedding_loras)
        actual = {str(module.lora_name) for module in network.unet_loras}
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(
                f"{item['style_id']} has {len(missing)} unmatched Anima modules; "
                f"first={missing[:3]}"
            )
        # create_network_from_weights has already copied every tensor through
        # make_module_from_state_dict. Register the adapter modules without
        # patching Anima or redundantly loading the full state dict a second time.
        network.text_encoder_loras = []
        network.loras = list(network.unet_loras)
        for module in network.loras:
            network.add_module(str(module.lora_name), module)
        network.to(device="cpu", dtype=torch.bfloat16).requires_grad_(False).eval()
        del state, factory_state
        return network, len(actual)

    def prefetch(self, item: dict[str, Any]) -> None:
        index = int(item["index"])
        with self.lock:
            if index in self.networks or index in self.pending:
                return
            self.cache_misses += 1
            self.pending[index] = self.executor.submit(self._build, item)

    def _load(self, item: dict[str, Any]) -> tuple[torch.nn.Module, int]:
        index = int(item["index"])
        with self.lock:
            network = self.networks.get(index)
            if network is not None:
                self.cache_hits += 1
                return network, len(network.unet_loras)
            future = self.pending.pop(index, None)
            if future is None:
                self.cache_misses += 1
        result = future.result() if future is not None else self._build(item)
        with self.lock:
            self.networks[index] = result[0]
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    @contextmanager
    def applied(
        self, components: list[tuple[dict[str, Any], float]]
    ) -> Iterator[dict[str, int]]:
        active: list[torch.nn.Module] = []
        loaded_modules = 0
        try:
            for item, multiplier in components:
                network, count = self._load(item)
                network.set_multiplier(float(multiplier))
                network.to(device=self.device, dtype=torch.bfloat16)
                network.apply_to(
                    [], self.anima, apply_text_encoder=False, apply_unet=True
                )
                active.append(network)
                loaded_modules += count
            yield {"loaded_modules": loaded_modules}
        finally:
            for network in reversed(active):
                network.restore()
                network.to(device="cpu", dtype=torch.bfloat16)


def _load_bank(destination: Path, path: str) -> list[dict[str, Any]]:
    payload = json.loads((destination / path).read_text(encoding="utf-8"))
    rows = [dict(item) for item in payload["items"]]
    if [int(item["index"]) for item in rows] != list(range(len(rows))):
        raise RuntimeError("External LoRA bank indices must be dense and ordered")
    for item in rows:
        weight = Path(str(item["weight_path"]))
        item["resolved_weight_path"] = str(
            weight if weight.is_absolute() else destination / weight
        )
        if not Path(item["resolved_weight_path"]).exists():
            raise FileNotFoundError(item["resolved_weight_path"])
    return rows


def _normalize_diffusers_lora(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Anima diffusers ``lora_A/B`` keys to Kohya keys in memory."""

    if not any(key.endswith(".lora_A.weight") for key in state):
        return state
    converted: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    for key, value in state.items():
        if not key.startswith("diffusion_model."):
            continue
        body = key.removeprefix("diffusion_model.")
        if body.endswith(".lora_A.weight"):
            name = body.removesuffix(".lora_A.weight")
            prefix = "lora_unet_" + name.replace(".", "_")
            converted[prefix + ".lora_down.weight"] = value
            ranks[prefix] = int(value.shape[0])
        elif body.endswith(".lora_B.weight"):
            name = body.removesuffix(".lora_B.weight")
            prefix = "lora_unet_" + name.replace(".", "_")
            converted[prefix + ".lora_up.weight"] = value
    for prefix, rank in ranks.items():
        converted[prefix + ".alpha"] = torch.tensor(float(rank))
    if not converted:
        raise RuntimeError("Diffusers LoRA contained no convertible Anima weights")
    return converted


def _normalize_external_lora_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize external Anima variants to the LyCORIS inference schema."""

    state = _normalize_diffusers_lora(state)
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        # networks.lora_anima saves RMSNorm weight deltas as Full ``diff``.
        # LyCORIS Full supports Linear/Conv only; its Norm module represents the
        # same additive weight delta as ``w_norm`` and supports Anima RMSNorm.
        if key.endswith("_norm.diff"):
            key = key.removesuffix(".diff") + ".w_norm"
        normalized[key] = value
    return normalized


def _expected_unet_modules(state: dict[str, torch.Tensor]) -> set[str]:
    return {
        key.split(".", 1)[0]
        for key in state
        if key.startswith("lora_unet_") and "." in key
    }


@contextmanager
def _applied_adapters(
    config: dict[str, Any],
    anima: torch.nn.Module,
    components: list[tuple[dict[str, Any], float]],
    device: str,
) -> Iterator[dict[str, int]]:
    """Attach only the active 1--3 adapters, then fully remove their wrappers."""

    sd_root = Path(str(config["anima_cache"]["sd_scripts_path"])).resolve()
    if str(sd_root) not in sys.path:
        sys.path.insert(0, str(sd_root))
    try:
        from lycoris.kohya import create_network_from_weights
    except ImportError as error:
        raise RuntimeError(
            "External LoRA caching requires lycoris_lora>=3.4"
        ) from error

    networks = []
    loaded_modules = 0
    expected_modules = 0
    try:
        for item, multiplier in components:
            state = _normalize_external_lora_state(
                load_file(str(item["resolved_weight_path"]), device="cpu")
            )
            expected = _expected_unet_modules(state)
            network, _ = create_network_from_weights(
                float(multiplier),
                str(item["resolved_weight_path"]),
                None,
                [],
                anima,
                weights_sd=state,
                for_inference=True,
            )
            actual = {str(module.lora_name) for module in network.unet_loras}
            missing = sorted(expected - actual)
            if missing:
                raise RuntimeError(
                    f"{item['style_id']} has {len(missing)} unmatched Anima modules; "
                    f"first={missing[:3]}"
                )
            network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
            incompatible = network.load_state_dict(state, strict=False)
            unexpected_unet = [
                key
                for key in incompatible.unexpected_keys
                if key.startswith("lora_unet_")
            ]
            if unexpected_unet:
                raise RuntimeError(
                    f"{item['style_id']} has unexpected Anima keys: "
                    f"{unexpected_unet[:3]}"
                )
            network.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
            networks.append(network)
            expected_modules += len(expected)
            loaded_modules += len(actual)
        yield {"expected_modules": expected_modules, "loaded_modules": loaded_modules}
    finally:
        for network in reversed(networks):
            network.restore()
        del networks


def _spec_records(
    bank: list[dict[str, Any]], specs: list[MixtureSpec]
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        component_rows = [bank[index] for index in spec.components]
        trigger_words: list[str] = []
        for item in component_rows:
            for trigger in item.get("trigger_words") or []:
                value = str(trigger).strip()
                if value and value.casefold() not in {
                    existing.casefold() for existing in trigger_words
                }:
                    trigger_words.append(value)
        rows.append(
            {
                "index": spec.index,
                "kind": spec.kind,
                "components": list(spec.components),
                "weights": list(spec.weights),
                "component_count": len(spec.components),
                "coefficient_sum": sum(spec.weights),
                "coefficient_l1": sum(abs(value) for value in spec.weights),
                "style_ids": [item["style_id"] for item in component_rows],
                "mixture_style_id": (
                    str(component_rows[0]["style_id"])
                    if spec.kind == "single"
                    else f"civitai-mixture-{spec.index:05d}"
                ),
                "trigger_words": trigger_words,
            }
        )
    return rows


def _trigger_prompt(caption: str, words: list[str]) -> str:
    suffix = ", ".join(value for value in words if value)
    return f"{caption}, {suffix}" if suffix else caption


def _prepare_trigger_contexts(
    config: dict[str, Any],
    destination: Path,
    output: Path,
    records: list[dict[str, Any]],
    content_rows: list[dict[str, Any]],
    *,
    trigger_contents: int,
    shard_rows: int,
    device: str,
    text_batch_size: int,
) -> None:
    expected_shards = (len(records) + shard_rows - 1) // shard_rows
    if all(
        (output / f"trigger-contexts-{index:05d}.safetensors").exists()
        for index in range(expected_shards)
    ):
        return
    prompts = []
    indices = []
    contents = len(content_rows)
    for record in records:
        selected = [
            (int(record["index"]) * trigger_contents + offset) % contents
            for offset in range(trigger_contents)
        ]
        indices.append(selected)
        prompts.extend(
            _trigger_prompt(
                str(content_rows[index]["caption"]),
                list(record["trigger_words"]),
            )
            for index in selected
        )
    encoded_flat = _encode_prompts(
        config, destination, prompts, device, batch_size=text_batch_size
    )
    encoded = encoded_flat.reshape(
        len(records),
        trigger_contents,
        encoded_flat.shape[1],
        encoded_flat.shape[2],
    )
    index_tensor = torch.tensor(indices, dtype=torch.int64)
    for shard_index, offset in enumerate(range(0, len(records), shard_rows)):
        end = min(offset + shard_rows, len(records))
        save_file(
            {
                "contexts": encoded[offset:end].contiguous(),
                "content_indices": index_tensor[offset:end].contiguous(),
                "mixture_indices": torch.arange(offset, end, dtype=torch.int64),
            },
            output / f"trigger-contexts-{shard_index:05d}.safetensors",
        )
    del encoded
    gc.collect()


def cache_external_lora_flow_inputs(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Materialize clean-query text and latent manifests for synthetic flow.

    External references were rendered with their trigger words, but the flow
    query deliberately uses ``content_prompt``.  The reference images must
    therefore carry the style signal instead of letting the text trigger solve
    the training example.  Existing VAE latents and frozen reference tokens are
    reused without copying their tensor shards.
    """

    reference_cfg = dict(config["external_civitai_lora_references"])
    flow_cfg = dict(reference_cfg.get("flow_cache", {}))
    reference_root = destination / str(reference_cfg["output_directory"])
    source_manifest = reference_root / "manifest.parquet"
    rows = sorted(read_records(source_manifest), key=lambda row: int(row["id"]))
    if not rows:
        raise RuntimeError(f"External reference manifest is empty: {source_manifest}")

    text_root = reference_root / str(
        flow_cfg.get("text_output_directory", "flow_text_clean_content_v1")
    )
    latent_root = reference_root / "latents"
    summary_path = text_root / "summary.json"
    expected_images = len(rows)
    if summary_path.exists() and (text_root / "manifest.parquet").exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(previous.get("images", -1)) == expected_images:
            return {**previous, "reused": True}

    text_root.mkdir(parents=True, exist_ok=True)
    latent_root.mkdir(parents=True, exist_ok=True)
    latent_rows = [
        {
            **row,
            "cache_shard": str(row["latent_shard"]),
            "row_index": int(row["latent_row"]),
        }
        for row in rows
    ]
    write_records(latent_root / "manifest.parquet", latent_rows)

    started = time.perf_counter()
    encoded = _encode_prompts(
        config,
        destination,
        [str(row["content_prompt"]) for row in rows],
        str(flow_cfg.get("device", reference_cfg.get("device", "cuda"))),
        batch_size=int(flow_cfg.get("text_batch_size", 32)),
    )
    shard_rows = int(flow_cfg.get("shard_rows", 256))
    output_rows: list[dict[str, Any]] = []
    storage_bytes = 0
    for shard_index, offset in enumerate(range(0, len(rows), shard_rows)):
        end = min(offset + shard_rows, len(rows))
        shard = encoded[offset:end]
        lengths = shard.abs().amax(dim=-1).ne(0).sum(dim=-1).clamp_min(1)
        conditions = [
            shard[index, : int(length)].contiguous()
            for index, length in enumerate(lengths.tolist())
        ]
        offsets = [0]
        for condition in conditions:
            offsets.append(offsets[-1] + int(condition.shape[0]))
        path = text_root / f"part-{shard_index:05d}.safetensors"
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(
            {
                "conditioning": torch.cat(conditions, dim=0),
                "offsets": torch.tensor(offsets, dtype=torch.int64),
                "ids": torch.tensor(
                    [int(row["id"]) for row in rows[offset:end]],
                    dtype=torch.int64,
                ),
                "variants": torch.zeros(end - offset, dtype=torch.int16),
            },
            temporary,
        )
        temporary.replace(path)
        storage_bytes += path.stat().st_size
        for local_index, row in enumerate(rows[offset:end]):
            output_rows.append(
                {
                    "id": int(row["id"]),
                    "variant": 0,
                    "variant_name": "full",
                    "cache_shard": path.name,
                    "token_offset": offsets[local_index],
                    "token_length": int(lengths[local_index]),
                    "prompt": str(row["content_prompt"]),
                }
            )
    write_records(text_root / "manifest.parquet", output_rows)
    summary = {
        "images": expected_images,
        "text_rows": len(output_rows),
        "shards": (expected_images + shard_rows - 1) // shard_rows,
        "prompt_policy": "content_prompt_without_external_trigger",
        "latent_manifest": str(latent_root / "manifest.parquet"),
        "text_manifest": str(text_root / "manifest.parquet"),
        "storage_bytes": storage_bytes,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    return summary


def cache_external_lora_functional_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["external_civitai_lora_teacher"])
    cache_cfg = dict(cfg["teacher_cache"])
    output = destination / str(cache_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    bank = _load_bank(destination, str(cfg["bank_manifest"]))
    specs = build_mixture_specs(
        len(bank),
        pair_count=int(cache_cfg.get("pair_mixtures", 128)),
        triple_count=int(cache_cfg.get("triple_mixtures", 128)),
        amplified_count=int(cache_cfg.get("amplified_mixtures", 64)),
        signed_count=int(cache_cfg.get("signed_mixtures", 64)),
        amplified_sum_range=tuple(cache_cfg.get("amplified_sum_range", [1.05, 1.5])),
        signed_beta_range=tuple(cache_cfg.get("signed_beta_range", [0.05, 0.25])),
        signed_l1_maximum=float(cache_cfg.get("signed_l1_maximum", 1.5)),
        seed=int(cache_cfg.get("seed", 20260828)),
    )
    summary_path = output / "summary.json"
    if summary_path.exists() and (output / "base.safetensors").exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(previous.get("complete_mixtures", -1)) == len(specs):
            return {**previous, "reused": True}
    records = _spec_records(bank, specs)
    write_records(output / "mixtures.parquet", records)

    contents = int(cache_cfg.get("contents", 24))
    timesteps = [float(value) for value in cache_cfg["timesteps"]]
    latents, contexts, content_rows = _cached_training_probe_bank(
        destination, cache_cfg, contents
    )
    write_records(output / "content_manifest.parquet", content_rows)
    device = str(cache_cfg.get("device", "cuda"))
    shard_rows = int(cache_cfg.get("shard_mixtures", 8))
    trigger_contents = int(cache_cfg.get("trigger_probe_contents", 4))
    trigger_timestep_indices = [
        int(value)
        for value in cache_cfg.get("trigger_timestep_indices", [0, 2, 5, 7])
    ]
    if trigger_contents:
        _prepare_trigger_contexts(
            config,
            destination,
            output,
            records,
            content_rows,
            trigger_contents=trigger_contents,
            shard_rows=shard_rows,
            device=device,
            text_batch_size=int(cache_cfg.get("text_batch_size", 32)),
        )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    latent_device = latents.to(device=device, dtype=torch.bfloat16)
    context_device = contexts.to(device=device, dtype=torch.bfloat16)
    seed = int(cache_cfg.get("seed", 20260828))
    noisy_rows, context_rows, timestep_rows = [], [], []
    for content_index in range(contents):
        for timestep_index, timestep in enumerate(timesteps):
            noise = torch.randn(
                latent_device[content_index].shape,
                generator=torch.Generator(device=device).manual_seed(
                    seed + content_index * 100_003 + timestep_index * 1009
                ),
                device=device,
                dtype=torch.bfloat16,
            )
            noisy_rows.append(
                (1 - timestep) * latent_device[content_index] + timestep * noise
            )
            context_rows.append(context_device[content_index])
            timestep_rows.append(timestep)
    noisy = torch.stack(noisy_rows)
    clean_context = torch.stack(context_rows)
    timestep_tensor = torch.tensor(
        timestep_rows, device=device, dtype=torch.bfloat16
    )
    batch_rows = int(cache_cfg.get("condition_batch_rows", 64))
    base = _predict_frozen_anima_in_chunks(
        anima, noisy, clean_context, timestep_tensor, batch_rows=batch_rows
    )
    save_file(
        {
            "base_context": contexts.to(torch.float16),
            "noisy_inputs": noisy.reshape(
                contents, len(timesteps), *noisy.shape[1:]
            ).to("cpu", dtype=torch.float16),
            "base_predictions": base.reshape(
                contents, len(timesteps), *base.shape[1:]
            ).to(torch.float16),
            "timesteps": torch.tensor(timesteps, dtype=torch.float32),
        },
        output / "base.safetensors",
    )

    completed = 0
    started = time.perf_counter()
    effect_rms: dict[int, float] = {}
    prefetch_workers = int(cache_cfg.get("adapter_prefetch_workers", 4))
    adapter_pool = _ExternalAdapterPool(
        config,
        anima,
        device,
        prefetch_workers=prefetch_workers,
    )
    for shard_index, offset in enumerate(range(0, len(specs), shard_rows)):
        shard_path = output / f"effects-{shard_index:05d}.safetensors"
        part = specs[offset : offset + shard_rows]
        if shard_path.exists():
            existing = load_file(shard_path, device="cpu")["effects"].float()
            rms = existing.square().mean(dim=tuple(range(1, existing.ndim))).sqrt()
            effect_rms.update(
                {spec.index: float(value) for spec, value in zip(part, rms, strict=True)}
            )
            completed += len(part)
            continue
        trigger_cache = (
            load_file(
                output / f"trigger-contexts-{shard_index:05d}.safetensors",
                device="cpu",
            )
            if trigger_contents
            else None
        )
        clean_effects, triggered_effects, triggered_bases = [], [], []
        loaded_counts = []
        trigger_noisy_all = trigger_context_all = trigger_time_all = None
        trigger_base_all = None
        if trigger_cache is not None:
            trigger_noisy_parts = []
            for content_indices in trigger_cache["content_indices"]:
                trigger_noisy_parts.append(
                    torch.stack(
                        [
                            noisy[int(content) * len(timesteps) + timestep_index]
                            for content in content_indices
                            for timestep_index in trigger_timestep_indices
                        ]
                    )
                )
            trigger_noisy_all = torch.cat(trigger_noisy_parts)
            trigger_context_all = trigger_cache["contexts"].to(
                device=device, dtype=torch.bfloat16
            ).reshape(-1, *trigger_cache["contexts"].shape[2:]).repeat_interleave(
                len(trigger_timestep_indices), dim=0
            )
            trigger_time_all = torch.tensor(
                [
                    timesteps[index]
                    for _ in range(len(part) * trigger_contents)
                    for index in trigger_timestep_indices
                ],
                device=device,
                dtype=torch.bfloat16,
            )
            trigger_base_all = _predict_frozen_anima_in_chunks(
                anima,
                trigger_noisy_all,
                trigger_context_all,
                trigger_time_all,
                batch_rows=batch_rows,
            )
        trigger_rows = trigger_contents * len(trigger_timestep_indices)
        for spec in specs[offset : min(offset + prefetch_workers, len(specs))]:
            for component_index in spec.components:
                adapter_pool.prefetch(bank[component_index])
        for local_index, spec in enumerate(part):
            components = [
                (bank[index], weight)
                for index, weight in zip(spec.components, spec.weights, strict=True)
            ]
            with adapter_pool.applied(components) as audit:
                next_index = offset + local_index + prefetch_workers
                if next_index < len(specs):
                    for component_index in specs[next_index].components:
                        adapter_pool.prefetch(bank[component_index])
                prediction = _predict_frozen_anima_in_chunks(
                    anima, noisy, clean_context, timestep_tensor, batch_rows=batch_rows
                )
                loaded_counts.append(audit["loaded_modules"])
                if trigger_cache is not None:
                    start = local_index * trigger_rows
                    end = start + trigger_rows
                    triggered = _predict_frozen_anima_in_chunks(
                        anima,
                        trigger_noisy_all[start:end],
                        trigger_context_all[start:end],
                        trigger_time_all[start:end],
                        batch_rows=batch_rows,
                    )
            clean_effects.append(
                (prediction - base).reshape(
                    contents, len(timesteps), *base.shape[1:]
                ).to(torch.bfloat16)
            )
            if trigger_cache is not None:
                start = local_index * trigger_rows
                end = start + trigger_rows
                trigger_base = trigger_base_all[start:end]
                triggered_bases.append(trigger_base.to(torch.bfloat16))
                triggered_effects.append(
                    (triggered - trigger_base).reshape(
                        trigger_contents,
                        len(trigger_timestep_indices),
                        *triggered.shape[1:],
                    ).to(torch.bfloat16)
                )
        stacked = torch.stack(clean_effects)
        payload = {
            "effects": stacked,
            "mixture_indices": torch.tensor(
                [spec.index for spec in part], dtype=torch.int64
            ),
            "loaded_module_counts": torch.tensor(loaded_counts, dtype=torch.int64),
        }
        if trigger_cache is not None:
            payload.update(
                {
                    "trigger_effects": torch.stack(triggered_effects),
                    "trigger_base_predictions": torch.stack(triggered_bases),
                    "trigger_content_indices": trigger_cache["content_indices"],
                    "trigger_timestep_indices": torch.tensor(
                        trigger_timestep_indices, dtype=torch.int64
                    ),
                }
            )
        save_file(payload, shard_path)
        rms = stacked.float().square().mean(
            dim=tuple(range(1, stacked.ndim))
        ).sqrt()
        effect_rms.update(
            {spec.index: float(value) for spec, value in zip(part, rms, strict=True)}
        )
        completed += len(part)
        print(
            f"external LoRA functional cache {completed}/{len(specs)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    single = torch.tensor([effect_rms[index] for index in range(len(bank))])
    median = float(single.median())
    lower, upper = [
        float(value)
        for value in cache_cfg.get("stable_effect_ratio_range", [0.35, 3.0])
    ]
    for record in records:
        rms = effect_rms[int(record["index"])]
        ratio = rms / max(median, 1e-8)
        record["effect_rms"] = rms
        record["effect_to_single_median_ratio"] = ratio
        record["enabled"] = record["kind"] == "single" or lower <= ratio <= upper
    write_records(output / "mixtures.parquet", records)
    summary = {
        "mixtures": len(specs),
        "individual": len(bank),
        "pairs": sum(spec.kind == "pair" for spec in specs),
        "triples": sum(spec.kind == "triple" for spec in specs),
        "amplified": sum(spec.kind == "amplified" for spec in specs),
        "signed": sum(spec.kind == "signed" for spec in specs),
        "contents": contents,
        "timesteps": timesteps,
        "trigger_probe_contents": trigger_contents,
        "trigger_timestep_indices": trigger_timestep_indices,
        "complete_mixtures": completed,
        "single_effect_rms_median": median,
        "disabled_unstable_mixtures": sum(not row["enabled"] for row in records),
        "query_policy": "clean functional target plus isolated trigger-conditioned probe",
        "actual_multi_adapter_forward": True,
        "adapter_application": "native-rank removable LyCORIS wrappers",
        "adapter_pool_resident": len(adapter_pool.networks),
        "adapter_pool_hits": adapter_pool.cache_hits,
        "adapter_pool_misses": adapter_pool.cache_misses,
        "adapter_prefetch_workers": prefetch_workers,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    adapter_pool.close()
    del anima, noisy, clean_context, base
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def generate_external_lora_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render trigger-conditioned references for every external LoRA mixture.

    The functional cache already contains four independently selected content
    prompts per mixture, encoded with the union of the active adapters' trigger
    words. Reusing those exact contexts keeps reference generation aligned with
    the teacher while avoiding a second multi-gigabyte text-conditioning cache.
    """

    cfg = dict(config["external_civitai_lora_references"])
    teacher_cfg = dict(config[str(cfg["functional_config_key"])])
    cache_cfg = dict(teacher_cfg["teacher_cache"])
    teacher_root = destination / str(cache_cfg["output_directory"])
    teacher_summary_path = teacher_root / "summary.json"
    if not teacher_summary_path.exists():
        raise FileNotFoundError(teacher_summary_path)
    teacher_summary = json.loads(teacher_summary_path.read_text(encoding="utf-8"))
    expected_mixtures = int(teacher_summary["mixtures"])
    if int(teacher_summary.get("complete_mixtures", -1)) != expected_mixtures:
        raise RuntimeError("External functional teacher cache is incomplete")

    records = sorted(
        read_records(teacher_root / "mixtures.parquet"),
        key=lambda row: int(row["index"]),
    )
    if [int(row["index"]) for row in records] != list(range(expected_mixtures)):
        raise RuntimeError("External mixture manifest is not dense and ordered")
    content_rows = read_records(teacher_root / "content_manifest.parquet")
    images_per_style = int(cfg.get("images_per_style", 4))
    trigger_contents = int(cache_cfg.get("trigger_probe_contents", 4))
    if images_per_style != trigger_contents:
        raise ValueError(
            "images_per_style must equal teacher_cache.trigger_probe_contents "
            "so every image uses a cached trigger-conditioned content"
        )
    shard_rows = int(cache_cfg.get("shard_mixtures", 8))
    expected_context_shards = (len(records) + shard_rows - 1) // shard_rows
    missing_contexts = [
        index
        for index in range(expected_context_shards)
        if not (teacher_root / f"trigger-contexts-{index:05d}.safetensors").exists()
    ]
    if missing_contexts:
        raise RuntimeError(
            f"Missing trigger context shards: first={missing_contexts[:8]}"
        )

    output = destination / str(cfg["output_directory"])
    for name in ("images", "latents", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)
    bank = _load_bank(destination, str(teacher_cfg["bank_manifest"]))
    device = str(cfg.get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    pool = _ExternalAdapterPool(
        config,
        anima,
        device,
        prefetch_workers=int(cfg.get("adapter_prefetch_workers", 4)),
    )
    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    vae.requires_grad_(False).eval()
    negative = load_file(
        destination / str(cfg["negative_conditioning_file"]), device="cpu"
    )["conditioning"].to(device=device, dtype=torch.bfloat16)
    negative = negative.expand(images_per_style, -1, -1)
    width, height = int(cfg.get("width", 512)), int(cfg.get("height", 512))
    steps = int(cfg.get("steps", 20))
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    text_cfg = float(cfg.get("text_cfg", 4.0))
    seed_base = int(cfg.get("seed", 20260829))
    image_id_base = int(cfg.get("image_id_base", 40_000_000_000))
    webp_quality = int(cfg.get("webp_quality", 95))
    completed_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    context_shard_index = -1
    context_shard: dict[str, torch.Tensor] | None = None

    try:
        for position, record in enumerate(records):
            mixture_index = int(record["index"])
            part_path = output / "manifests" / f"part-{mixture_index:05d}.parquet"
            if part_path.exists():
                completed_rows.extend(read_records(part_path))
                continue

            # Keep CPU construction overlapped with the current GPU render.
            for upcoming in records[position : position + int(cfg.get("prefetch_styles", 8))]:
                for component in upcoming["components"]:
                    pool.prefetch(bank[int(component)])

            required_shard = mixture_index // shard_rows
            if context_shard is None or required_shard != context_shard_index:
                context_shard = load_file(
                    teacher_root / f"trigger-contexts-{required_shard:05d}.safetensors",
                    device="cpu",
                )
                context_shard_index = required_shard
            local_index = mixture_index - required_shard * shard_rows
            cached_index = int(context_shard["mixture_indices"][local_index])
            if cached_index != mixture_index:
                raise RuntimeError(
                    f"Trigger context index mismatch: {cached_index} != {mixture_index}"
                )
            positive = context_shard["contexts"][local_index].to(
                device=device, dtype=torch.bfloat16
            )
            content_indices = [
                int(value)
                for value in context_shard["content_indices"][local_index].tolist()
            ]
            seeds = [
                seed_base + mixture_index * 100_003 + image_index * 1009
                for image_index in range(images_per_style)
            ]
            noise = torch.stack(
                [
                    torch.randn(
                        16,
                        1,
                        height // 8,
                        width // 8,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    for seed in seeds
                ]
            )
            components = [
                (bank[int(component)], float(weight))
                for component, weight in zip(
                    record["components"], record["weights"], strict=True
                )
            ]
            with pool.applied(components), torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                latents = _sample_anima_batch(
                    anima,
                    noise,
                    positive,
                    negative,
                    sigmas,
                    text_cfg=text_cfg,
                    speed=None,
                    generation_seeds=seeds,
                )
                decoded = vae.decode_to_pixels(latents)
            if not torch.isfinite(latents).all():
                raise RuntimeError(
                    f"Non-finite external reference latent at mixture {mixture_index}"
                )
            images = _preview_pixels(decoded)
            latent_name = f"part-{mixture_index:05d}.safetensors"
            save_file(
                {
                    "latents": latents[:, :, 0]
                    .to("cpu", dtype=torch.float16)
                    .contiguous()
                },
                output / "latents" / latent_name,
            )
            style_id = str(record["mixture_style_id"])
            image_dir = output / "images" / style_id
            image_dir.mkdir(exist_ok=True)
            rows = []
            for image_index, (image, content_index, seed) in enumerate(
                zip(images, content_indices, seeds, strict=True)
            ):
                image_path = image_dir / f"content-{image_index:02d}.webp"
                image.save(image_path, format="WEBP", quality=webp_quality)
                kind = "artist" if str(record["kind"]) == "single" else "lora_mixture"
                content_prompt = str(
                    content_rows[content_index].get(
                        "caption", content_rows[content_index].get("prompt", "")
                    )
                )
                triggered_prompt = _trigger_prompt(
                    content_prompt, list(record.get("trigger_words") or [])
                )
                rows.append(
                    {
                        "id": image_id_base
                        + mixture_index * images_per_style
                        + image_index,
                        "kind": kind,
                        "mixture_kind": str(record["kind"]),
                        "mixture_index": mixture_index,
                        "artist_index": mixture_index,
                        "artist": style_id,
                        "style_id": style_id,
                        "artist_split": "train",
                        "split": "train",
                        "content_index": image_index,
                        "source_content_index": content_index,
                        "generation_seed": seed,
                        "content_prompt": content_prompt,
                        "artist_prompt": triggered_prompt,
                        "artist_tag": ", ".join(record.get("trigger_words") or []),
                        "trigger_words": list(record.get("trigger_words") or []),
                        "components": list(record["components"]),
                        "weights": list(record["weights"]),
                        "local_path": str(image_path.resolve()),
                        "width": width,
                        "height": height,
                        "latent_height": height // 8,
                        "latent_width": width // 8,
                        "steps": steps,
                        "text_cfg": text_cfg,
                        "flow_shift": shift,
                        "attention_backend": str(cfg.get("attention_mode", "torch")),
                        "latent_shard": latent_name,
                        "latent_row": image_index,
                    }
                )
            write_records(part_path, rows)
            completed_rows.extend(rows)
            print(
                f"external LoRA references {position + 1}/{len(records)} "
                f"kind={record['kind']} images={len(completed_rows)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    finally:
        pool.close()

    completed_rows.sort(key=lambda row: int(row["id"]))
    expected_images = len(records) * images_per_style
    if len(completed_rows) != expected_images:
        raise RuntimeError(
            f"External reference manifest incomplete: "
            f"{len(completed_rows)}/{expected_images}"
        )
    write_records(output / "manifest.parquet", completed_rows)
    summary = {
        "styles": len(records),
        "single_styles": sum(str(row["kind"]) == "single" for row in records),
        "mixture_styles": sum(str(row["kind"]) != "single" for row in records),
        "images": len(completed_rows),
        "images_per_style": images_per_style,
        "content_policy": "four distinct cached-training contents per style",
        "prompt_policy": "content plus union of component trigger words",
        "functional_teacher_cache": str(teacher_root),
        "adapter_pool_hits": pool.cache_hits,
        "adapter_pool_misses": pool.cache_misses,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "generation_summary.json", summary)
    del anima, vae
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def validate_external_lora_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Run one real Anima forward for each adapter storage format."""

    cfg = dict(config["external_civitai_lora_teacher"])
    cache_cfg = dict(cfg["teacher_cache"])
    bank = _load_bank(destination, str(cfg["bank_manifest"]))
    representatives = []
    seen = set()
    for item in bank:
        if item["format"] not in seen:
            representatives.append(item)
            seen.add(item["format"])
    latents, contexts, _ = _cached_training_probe_bank(destination, cache_cfg, 1)
    device = str(cache_cfg.get("device", "cuda"))
    latent = latents[:1].to(device=device, dtype=torch.bfloat16)
    context = contexts[:1].to(device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
    noise = torch.randn(
        latent.shape,
        generator=torch.Generator(device=device).manual_seed(20260828),
        device=device,
        dtype=torch.bfloat16,
    )
    noisy = 0.5 * latent + 0.5 * noise
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    base = _predict_frozen_anima_in_chunks(
        anima, noisy, context, timestep, batch_rows=1
    )
    adapter_pool = _ExternalAdapterPool(config, anima, device)
    for item in representatives:
        adapter_pool.prefetch(item)
    results = []
    for representative_index, item in enumerate(representatives):
        with adapter_pool.applied([(item, 1.0)]) as audit:
            styled = _predict_frozen_anima_in_chunks(
                anima, noisy, context, timestep, batch_rows=1
            )
        restored = _predict_frozen_anima_in_chunks(
            anima, noisy, context, timestep, batch_rows=1
        )
        if not torch.equal(restored, base):
            raise RuntimeError(f"{item['format']} adapter did not restore exactly")
        if representative_index == 0:
            with adapter_pool.applied([(item, 1.0)]):
                reused = _predict_frozen_anima_in_chunks(
                    anima, noisy, context, timestep, batch_rows=1
                )
            if not torch.equal(reused, styled):
                raise RuntimeError("Re-applied adapter changed its prediction")
        with _applied_adapters(config, anima, [(item, 1.0)], device):
            legacy = _predict_frozen_anima_in_chunks(
                anima, noisy, context, timestep, batch_rows=1
            )
        if not torch.equal(legacy, styled):
            maximum_error = float((legacy - styled).float().abs().max())
            raise RuntimeError(
                f"Fast {item['format']} adapter differs from legacy loader: "
                f"max_abs={maximum_error}"
            )
        rms = float((styled - base).float().square().mean().sqrt())
        if not torch.isfinite(styled).all() or rms <= 0:
            raise RuntimeError(
                f"Invalid {item['format']} teacher effect: rms={rms}"
            )
        results.append(
            {
                "format": item["format"],
                "style_id": item["style_id"],
                "loaded_modules": audit["loaded_modules"],
                "effect_rms": rms,
            }
        )
    del anima
    adapter_pool.close()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "formats": len(results),
        "adapter_pool_hits": adapter_pool.cache_hits,
        "adapter_pool_misses": adapter_pool.cache_misses,
        "results": results,
    }
