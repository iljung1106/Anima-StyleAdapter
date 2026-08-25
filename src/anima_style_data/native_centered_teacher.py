from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .anima_cache import _caption_rows
from .io import read_records, write_json
from .style_calibration import _artist_prompt, _encode_prompts
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model
from .synthetic_teacher import synthetic_artist_split_map


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cached_prompt_conditions(
    text_root: Path, prompts: list[str]
) -> torch.Tensor | None:
    """Load an exact synthetic prompt cache instead of running Qwen again."""
    manifest_path = text_root / "manifest.parquet"
    if not manifest_path.exists():
        return None
    rows = sorted(
        read_records(manifest_path), key=lambda row: int(row["condition_id"])
    )
    if len(rows) != len(prompts) or any(
        str(row["prompt"]) != prompt
        for row, prompt in zip(rows, prompts, strict=True)
    ):
        return None
    shards: dict[str, torch.Tensor] = {}
    values = []
    for row in rows:
        name = str(row["cache_shard"])
        if name not in shards:
            shards[name] = load_file(text_root / name, device="cpu")["conditioning"]
        values.append(shards[name][int(row["row_index"])])
    print(
        f"reused {len(values)} exact synthetic text conditions for native teacher",
        flush=True,
    )
    return torch.stack(values).contiguous()


def _load_probe_latents(
    destination: Path,
    config: dict[str, Any],
    probe_ids: list[int],
) -> torch.Tensor:
    root = destination / str(config["style_transfer"]["loader"]["latent_cache"])
    rows = read_records(root / "manifest.parquet")
    by_id = {int(row["id"]): row for row in rows}
    shards: dict[str, dict[str, torch.Tensor]] = {}
    values = []
    for image_id in probe_ids:
        row = by_id[image_id]
        shard_name = str(row["cache_shard"])
        if shard_name not in shards:
            shards[shard_name] = load_file(root / shard_name, device="cpu")
        values.append(shards[shard_name]["latents"][int(row["row_index"])])
    result = torch.stack(values).to(dtype=torch.float16).contiguous()
    if len({tuple(value.shape) for value in result}) != 1:
        raise RuntimeError("Native teacher probe latents must share one shape")
    return result


def _load_manifest_probe_latents(
    destination: Path,
    cfg: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> torch.Tensor:
    root = destination / str(cfg["probe_latent_directory"])
    shards: dict[str, dict[str, torch.Tensor]] = {}
    values = []
    for row in probe_rows:
        shard_name = str(row["latent_shard"])
        if shard_name not in shards:
            shards[shard_name] = load_file(root / shard_name, device="cpu")
        values.append(shards[shard_name]["latents"][int(row["latent_row"])])
    result = torch.stack(values).to(dtype=torch.float16).contiguous()
    if len({tuple(value.shape) for value in result}) != 1:
        raise RuntimeError("Configured teacher probe latents must share one shape")
    return result


def _teacher_spec(
    cfg: dict[str, Any], destination: Path, config: dict[str, Any]
) -> tuple[list[str], list[str], list[str], list[int], dict[str, Any]]:
    requested_contents = int(cfg.get("content_count", 4))
    if cfg.get("artist_manifest"):
        source_path = destination / str(cfg["artist_manifest"])
        all_source_rows = read_records(source_path)
        source_rows = [
            row
            for row in all_source_rows
            if str(row.get("kind", "artist")) == "artist"
        ]
        by_artist: dict[str, dict[str, Any]] = {}
        for row in source_rows:
            artist = str(row["artist"])
            by_artist.setdefault(artist, row)
        artists = sorted(
            by_artist,
            key=lambda value: int(by_artist[value].get("artist_index", 0)),
        )
        requested_artists = int(cfg.get("artist_count", len(artists)))
        if len(artists) != requested_artists:
            raise RuntimeError(
                f"Teacher manifest has {len(artists)} artists, expected "
                f"{requested_artists}"
            )
        style_ids = [str(by_artist[artist]["style_id"]) for artist in artists]
        fallback_splits = (
            synthetic_artist_split_map(config, source_rows)
            if any(
                not (row.get("artist_split") or row.get("split"))
                for row in source_rows
            )
            else {}
        )
        splits = [
            "test"
            if str(
                by_artist[artist].get("artist_split")
                or by_artist[artist].get("split")
                or fallback_splits.get(artist, "train")
            )
            == "meta_test"
            else str(
                by_artist[artist].get("artist_split")
                or by_artist[artist].get("split")
                or fallback_splits.get(artist, "train")
            )
            for artist in artists
        ]
        if cfg.get("probe_manifest"):
            control_rows = all_source_rows
            if not any(
                str(row.get("kind")) == "content_control"
                for row in control_rows
            ):
                control_rows = read_records(
                    destination / str(cfg["probe_manifest"])
                )
            controls = {
                int(row["content_index"]): int(row["id"])
                for row in control_rows
                if str(row.get("kind")) == "content_control"
                and int(row.get("seed_index", 0)) == 0
            }
            probe_ids = [controls[index] for index in sorted(controls)][
                :requested_contents
            ]
        else:
            content_ids = {
                int(row["content_index"]): int(row["content_source_id"])
                for row in source_rows
            }
            probe_ids = [content_ids[index] for index in sorted(content_ids)][
                :requested_contents
            ]
        if len(probe_ids) != requested_contents:
            raise RuntimeError("Artist manifest does not contain enough probe contents")
        source_signature = {
            "artist_manifest": str(cfg["artist_manifest"]),
            "artist_manifest_sha256": _sha256(source_path),
        }
        if cfg.get("probe_manifest"):
            probe_manifest_path = destination / str(cfg["probe_manifest"])
            source_signature["probe_manifest_sha256"] = _sha256(
                probe_manifest_path
            )
        return artists, style_ids, splits, probe_ids, source_signature

    calibration_path = destination / str(cfg["calibration_file"])
    style_manifest = destination / str(cfg["style_manifest"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    retained = [str(value) for value in calibration["retained_artists"]]
    probe_ids = [int(value) for value in calibration["probe_ids"]][
        :requested_contents
    ]
    if len(probe_ids) != requested_contents:
        raise RuntimeError("Calibration does not contain enough probe contents")

    split_by_artist: dict[str, tuple[str, str]] = {}
    for row in read_records(style_manifest):
        artist = str(row["artist"])
        if artist in retained:
            split_by_artist[artist] = (
                str(row["style_id"]), str(row.get("split", "train"))
            )
    if set(split_by_artist) != set(retained):
        missing = sorted(set(retained) - set(split_by_artist))
        raise RuntimeError(f"Teacher artists missing from style cache: {missing}")
    return (
        retained,
        [split_by_artist[artist][0] for artist in retained],
        [split_by_artist[artist][1] for artist in retained],
        probe_ids,
        {
            "calibration_sha256": _sha256(calibration_path),
            "style_manifest": str(cfg["style_manifest"]),
        },
    )


def _cache_native_centered_teacher(
    config: dict[str, Any], destination: Path, *, config_key: str
) -> dict[str, Any]:
    cfg = dict(config[config_key])
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "teacher_bank.safetensors"
    summary_path = output / "summary.json"
    artists, style_ids, splits, probe_ids, source_signature = _teacher_spec(
        cfg, destination, config
    )
    timesteps = [float(value) for value in cfg["timesteps"]]
    signature = {
        "version": "native-artist-centered-flow-v1",
        "config_key": config_key,
        **source_signature,
        "artists": artists,
        "style_ids": style_ids,
        "splits": splits,
        "probe_ids": probe_ids,
        "timesteps": timesteps,
        "seed": int(cfg.get("seed", 20260817)),
    }
    if tensor_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") == signature:
            return {**summary, "reused": True}

    if cfg.get("probe_manifest"):
        probe_manifest_path = destination / str(cfg["probe_manifest"])
        probe_manifest = {
            int(row["id"]): row for row in read_records(probe_manifest_path)
        }
        probe_rows = [
            {
                **probe_manifest[image_id],
                "anima_caption": str(probe_manifest[image_id]["content_prompt"]),
            }
            for image_id in probe_ids
        ]
        latents = _load_manifest_probe_latents(destination, cfg, probe_rows)
    else:
        captions = {int(row["id"]): row for row in _caption_rows(destination)}
        probe_rows = [captions[image_id] for image_id in probe_ids]
        latents = _load_probe_latents(destination, config, probe_ids)
    prompts = [str(row["anima_caption"]) for row in probe_rows]
    if cfg.get("probe_manifest"):
        prompts.extend(
            f"{row['anima_caption']}, "
            f"@{' '.join(artist.replace('_', ' ').split())}"
            for artist in artists
            for row in probe_rows
        )
    else:
        prompts.extend(
            _artist_prompt(row, artist)
            for artist in artists
            for row in probe_rows
        )
    device = str(cfg.get("device", "cuda"))
    conditions = None
    if cfg.get("probe_manifest"):
        conditions = _load_cached_prompt_conditions(
            (destination / str(cfg["probe_manifest"])).parent / "text",
            prompts,
        )
    if conditions is None:
        conditions = _encode_prompts(
            config,
            destination,
            prompts,
            device,
            int(cfg.get("text_batch_size", 64)),
        )
    contents = len(probe_rows)
    base_context = conditions[:contents].to(dtype=torch.float16).contiguous()
    tagged_context = conditions[contents:].reshape(
        len(artists), contents, *conditions.shape[1:]
    )
    base_lengths = (
        base_context.float().abs().sum(dim=-1) > 0
    ).sum(dim=-1).to(torch.int64)
    latent_shape = tuple(int(value) for value in latents.shape[1:])
    noisy_inputs = torch.empty(
        contents, len(timesteps), *latent_shape, dtype=torch.float16
    )
    base_predictions = torch.empty_like(noisy_inputs)
    centered_teacher = torch.empty(
        len(artists), contents, len(timesteps), *latent_shape,
        dtype=torch.float16,
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    base_context_device = base_context
    if bool(cfg.get("gpu_resident_text", False)) and device.startswith("cuda"):
        base_context_device = base_context.to(device=device, dtype=torch.bfloat16)
        tagged_context = tagged_context.to(device=device, dtype=torch.bfloat16)
        print(
            "resident native-teacher text context "
            f"{(base_context_device.numel() + tagged_context.numel()) * 2 / 2**30:.2f} GiB",
            flush=True,
        )

    artist_batch = max(1, int(cfg.get("artist_batch_size", 8)))
    seed = int(cfg.get("seed", 20260817))
    candidates = sorted({
        int(value) for value in cfg.get("artist_batch_candidates", [artist_batch])
        if 0 < int(value) <= len(artists)
    })
    if device.startswith("cuda") and len(candidates) > 1:
        benchmark_rows = []
        probe_latent = latents[:1].to(device=device, dtype=torch.bfloat16)
        probe_context = base_context_device[:1].to(
            device=device, dtype=torch.bfloat16
        )
        probe_padding = torch.zeros(
            1, 1, probe_latent.shape[-2], probe_latent.shape[-1],
            device=device, dtype=probe_latent.dtype,
        )
        probe_timestep = torch.full(
            (1,), timesteps[0], device=device, dtype=probe_latent.dtype
        )
        probe_generator = torch.Generator(device=device).manual_seed(seed)
        probe_noise = torch.randn(
            probe_latent.shape, device=device, dtype=probe_latent.dtype,
            generator=probe_generator,
        )
        probe_noisy = (1 - timesteps[0]) * probe_latent + timesteps[0] * probe_noise
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for candidate in candidates:
                try:
                    tagged = tagged_context[:candidate, 0].to(
                        device=device, dtype=torch.bfloat16
                    )
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    # One warmup removes lazy kernel and allocator setup from timing.
                    anima(
                        probe_noisy.expand(candidate, -1, -1, -1).unsqueeze(2),
                        probe_timestep.expand(candidate), context=tagged,
                        padding_mask=probe_padding.expand(candidate, -1, -1, -1),
                        target_input_ids=None,
                    )
                    torch.cuda.synchronize()
                    probe_started = time.perf_counter()
                    prediction = anima(
                        probe_noisy.expand(candidate, -1, -1, -1).unsqueeze(2),
                        probe_timestep.expand(candidate), context=tagged,
                        padding_mask=probe_padding.expand(candidate, -1, -1, -1),
                        target_input_ids=None,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - probe_started
                    benchmark_rows.append({
                        "batch_size": candidate,
                        "artists_s": candidate / max(elapsed, 1e-9),
                        "elapsed_s": elapsed,
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                    })
                    del prediction, tagged
                except torch.cuda.OutOfMemoryError:
                    benchmark_rows.append({"batch_size": candidate, "oom": True})
                    torch.cuda.empty_cache()
        valid_rows = [row for row in benchmark_rows if not row.get("oom")]
        if not valid_rows:
            raise RuntimeError("Every native-teacher artist batch candidate OOMed")
        artist_batch = int(max(valid_rows, key=lambda row: row["artists_s"])["batch_size"])
        write_json(output / "artist_batch_autotune.json", {
            "selected_batch_size": artist_batch, "results": benchmark_rows
        })
        print(
            f"selected native-teacher artist batch {artist_batch}: {benchmark_rows}",
            flush=True,
        )
        del probe_latent, probe_context, probe_padding, probe_noise, probe_noisy
        torch.cuda.empty_cache()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for content_index in range(contents):
            latent = latents[content_index : content_index + 1].to(
                device=device, dtype=torch.bfloat16
            )
            context = base_context_device[content_index : content_index + 1].to(
                device=device, dtype=torch.bfloat16
            )
            padding = torch.zeros(
                1, 1, latent.shape[-2], latent.shape[-1],
                device=device, dtype=latent.dtype,
            )
            for timestep_index, timestep_value in enumerate(timesteps):
                generator = torch.Generator(device=device).manual_seed(
                    seed + content_index * 100_003 + timestep_index * 1009
                )
                noise = torch.randn(
                    latent.shape, device=device, dtype=latent.dtype,
                    generator=generator,
                )
                timestep = torch.full(
                    (1,), timestep_value, device=device, dtype=latent.dtype
                )
                noisy = (1 - timestep_value) * latent + timestep_value * noise
                base = anima(
                    noisy.unsqueeze(2), timestep, context=context,
                    padding_mask=padding, target_input_ids=None,
                ).squeeze(2).float()
                effects = torch.empty(
                    len(artists), *base.shape[1:],
                    device=device, dtype=torch.float32,
                )
                for offset in range(0, len(artists), artist_batch):
                    count = min(artist_batch, len(artists) - offset)
                    tagged = tagged_context[
                        offset : offset + count, content_index
                    ].to(device=device, dtype=torch.bfloat16)
                    prediction = anima(
                        noisy.expand(count, -1, -1, -1).unsqueeze(2),
                        timestep.expand(count),
                        context=tagged,
                        padding_mask=padding.expand(count, -1, -1, -1),
                        target_input_ids=None,
                    ).squeeze(2).float()
                    effects[offset : offset + count].copy_(prediction - base)
                centered = effects - effects.mean(dim=0, keepdim=True)
                noisy_inputs[content_index, timestep_index] = noisy[0].cpu()
                base_predictions[content_index, timestep_index] = base[0].cpu()
                centered_teacher[:, content_index, timestep_index] = centered.to(
                    device="cpu", dtype=torch.float16
                )
                print(
                    "native centered teacher "
                    f"content={content_index + 1}/{contents} "
                    f"timestep={timestep_index + 1}/{len(timesteps)}",
                    flush=True,
                )
    del anima, conditions, tagged_context, base_context_device, effects, centered
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    tensors = {
        "base_context": base_context,
        "base_lengths": base_lengths,
        "noisy_inputs": noisy_inputs,
        "base_predictions": base_predictions,
        "centered_teacher": centered_teacher,
        "timesteps": torch.tensor(timesteps, dtype=torch.float32),
    }
    save_file(tensors, tensor_path)
    dimensions = tuple(range(3, centered_teacher.ndim))
    teacher_rms = centered_teacher.float().square().mean(dim=dimensions).sqrt()
    summary = {
        "signature": signature,
        "tensor_path": str(tensor_path),
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "teacher_rms_mean": float(teacher_rms.mean()),
        "teacher_rms_min": float(teacher_rms.min()),
        "teacher_rms_max": float(teacher_rms.max()),
        "train_style_ids": [
            style_id for style_id, split in zip(style_ids, splits, strict=True)
            if split == "train"
        ],
        "validation_style_ids": [
            style_id for style_id, split in zip(style_ids, splits, strict=True)
            if split == "validation"
        ],
        "test_style_ids": [
            style_id for style_id, split in zip(style_ids, splits, strict=True)
            if split == "test"
        ],
        "storage_bytes": tensor_path.stat().st_size,
    }
    write_json(summary_path, summary)
    return {**summary, "reused": False}


def cache_native_centered_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _cache_native_centered_teacher(
        config, destination, config_key="native_centered_teacher"
    )


def cache_dual_domain_centered_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _cache_native_centered_teacher(
        config, destination, config_key="dual_domain_native_teacher"
    )


@dataclass
class NativeCenteredTeacherBank:
    tensors: dict[str, torch.Tensor]
    summary: dict[str, Any]
    artist_to_index: dict[str, int]
    root: Path

    def train_population_offset(self) -> torch.Tensor:
        cached = self.tensors.get("train_population_offset")
        if cached is not None:
            return cached
        cache_path = self.root / "train_population_offset.safetensors"
        if cache_path.exists():
            cached = load_file(cache_path, device="cpu")["train_population_offset"]
            expected_shape = tuple(self.tensors["centered_teacher"].shape[1:])
            if tuple(cached.shape) != expected_shape:
                raise RuntimeError(
                    "Native population offset cache shape does not match teacher bank"
                )
            self.tensors["train_population_offset"] = cached
            print(f"reused native train-population offset {cache_path}", flush=True)
            return cached
        indices = [
            self.artist_to_index[str(value)]
            for value in self.summary["train_style_ids"]
        ]
        if len(indices) < 2:
            raise RuntimeError("Native population Common needs at least two train artists")
        source = self.tensors["centered_teacher"]
        total = torch.zeros_like(source[0], dtype=torch.float32)
        for offset in range(0, len(indices), 16):
            part = torch.tensor(indices[offset : offset + 16], dtype=torch.long)
            total.add_(source.index_select(0, part).float().sum(dim=0))
        cached = (total / len(indices)).to(torch.float16)
        temporary = cache_path.with_name(
            f".{cache_path.name}.tmp-{os.getpid()}"
        )
        save_file({"train_population_offset": cached.contiguous()}, temporary)
        temporary.replace(cache_path)
        self.tensors["train_population_offset"] = cached
        print(f"cached native train-population offset {cache_path}", flush=True)
        return cached

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        destination: Path,
        *,
        config_key: str = "native_centered_teacher",
    ) -> "NativeCenteredTeacherBank":
        cfg = dict(config[config_key])
        root = destination / str(cfg["output_directory"])
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        tensors = load_file(root / "teacher_bank.safetensors", device="cpu")
        style_ids = [str(value) for value in summary["signature"]["style_ids"]]
        return cls(
            tensors=tensors,
            summary=summary,
            artist_to_index={value: index for index, value in enumerate(style_ids)},
            root=root,
        )
