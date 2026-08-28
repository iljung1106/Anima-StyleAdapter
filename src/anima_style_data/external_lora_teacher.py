"""Whole-model functional caches for heterogeneous external Anima LoRAs."""

from __future__ import annotations

import gc
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file

from .io import write_json, write_records
from .lora_functional_distillation import (
    MixtureSpec,
    _cached_training_probe_bank,
    _predict_frozen_anima_in_chunks,
    build_mixture_specs,
)
from .style_calibration import _encode_prompts
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


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
            state = _normalize_diffusers_lora(
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
        for local_index, spec in enumerate(part):
            components = [
                (bank[index], weight)
                for index, weight in zip(spec.components, spec.weights, strict=True)
            ]
            with _applied_adapters(config, anima, components, device) as audit:
                prediction = _predict_frozen_anima_in_chunks(
                    anima, noisy, clean_context, timestep_tensor, batch_rows=batch_rows
                )
                loaded_counts.append(audit["loaded_modules"])
                if trigger_cache is not None:
                    content_indices = trigger_cache["content_indices"][local_index]
                    trigger_context = trigger_cache["contexts"][local_index].to(
                        device=device, dtype=torch.bfloat16
                    )
                    trigger_noisy = torch.stack(
                        [
                            noisy[int(content) * len(timesteps) + timestep_index]
                            for content in content_indices
                            for timestep_index in trigger_timestep_indices
                        ]
                    )
                    trigger_time = torch.tensor(
                        [timesteps[index] for _ in content_indices for index in trigger_timestep_indices],
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    trigger_context_flat = trigger_context.repeat_interleave(
                        len(trigger_timestep_indices), dim=0
                    )
                    triggered = _predict_frozen_anima_in_chunks(
                        anima,
                        trigger_noisy,
                        trigger_context_flat,
                        trigger_time,
                        batch_rows=batch_rows,
                    )
            clean_effects.append(
                (prediction - base).reshape(
                    contents, len(timesteps), *base.shape[1:]
                ).to(torch.bfloat16)
            )
            if trigger_cache is not None:
                trigger_base = _predict_frozen_anima_in_chunks(
                    anima,
                    trigger_noisy,
                    trigger_context_flat,
                    trigger_time,
                    batch_rows=batch_rows,
                )
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
        "actual_merged_lora_forward": True,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    del anima, noisy, clean_context, base
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
    results = []
    for item in representatives:
        with _applied_adapters(config, anima, [(item, 1.0)], device) as audit:
            styled = _predict_frozen_anima_in_chunks(
                anima, noisy, context, timestep, batch_rows=1
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
    gc.collect()
    torch.cuda.empty_cache()
    return {"formats": len(results), "results": results}
