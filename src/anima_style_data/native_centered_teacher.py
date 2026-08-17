from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .anima_cache import _caption_rows
from .io import read_records, write_json
from .style_calibration import _artist_prompt, _encode_prompts
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def cache_native_centered_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["native_centered_teacher"])
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "teacher_bank.safetensors"
    summary_path = output / "summary.json"
    calibration_path = destination / str(cfg["calibration_file"])
    style_manifest = destination / str(cfg["style_manifest"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    retained = [str(value) for value in calibration["retained_artists"]]
    requested_contents = int(cfg.get("content_count", 4))
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
    artists = retained
    style_ids = [split_by_artist[artist][0] for artist in artists]
    splits = [split_by_artist[artist][1] for artist in artists]
    timesteps = [float(value) for value in cfg["timesteps"]]
    signature = {
        "version": "native-artist-centered-flow-v1",
        "calibration_sha256": _sha256(calibration_path),
        "style_manifest": str(cfg["style_manifest"]),
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

    captions = {int(row["id"]): row for row in _caption_rows(destination)}
    probe_rows = [captions[image_id] for image_id in probe_ids]
    prompts = [str(row["anima_caption"]) for row in probe_rows]
    prompts.extend(
        _artist_prompt(row, artist)
        for artist in artists
        for row in probe_rows
    )
    device = str(cfg.get("device", "cuda"))
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
    latents = _load_probe_latents(destination, config, probe_ids)
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
    artist_batch = max(1, int(cfg.get("artist_batch_size", 8)))
    seed = int(cfg.get("seed", 20260817))
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for content_index in range(contents):
            latent = latents[content_index : content_index + 1].to(
                device=device, dtype=torch.bfloat16
            )
            context = base_context[content_index : content_index + 1].to(
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
                deltas = []
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
                    deltas.append((prediction - base).cpu())
                effects = torch.cat(deltas)
                centered = effects - effects.mean(dim=0, keepdim=True)
                noisy_inputs[content_index, timestep_index] = noisy[0].cpu()
                base_predictions[content_index, timestep_index] = base[0].cpu()
                centered_teacher[:, content_index, timestep_index] = centered
                print(
                    "native centered teacher "
                    f"content={content_index + 1}/{contents} "
                    f"timestep={timestep_index + 1}/{len(timesteps)}",
                    flush=True,
                )
    del anima, conditions, tagged_context
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


@dataclass
class NativeCenteredTeacherBank:
    tensors: dict[str, torch.Tensor]
    summary: dict[str, Any]
    artist_to_index: dict[str, int]

    @classmethod
    def load(
        cls, config: dict[str, Any], destination: Path
    ) -> "NativeCenteredTeacherBank":
        cfg = dict(config["native_centered_teacher"])
        root = destination / str(cfg["output_directory"])
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        tensors = load_file(root / "teacher_bank.safetensors", device="cpu")
        style_ids = [str(value) for value in summary["signature"]["style_ids"]]
        return cls(
            tensors=tensors,
            summary=summary,
            artist_to_index={value: index for index, value in enumerate(style_ids)},
        )
