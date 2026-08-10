from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

from .io import read_records, write_json, write_records


@dataclass(frozen=True)
class AnimaImageGeometry:
    resized_height: int
    resized_width: int
    target_height: int
    target_width: int
    crop_top: int
    crop_left: int


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _stable_keep(seed: int, image_id: int, tag: str, keep_probability: float) -> bool:
    digest = hashlib.blake2b(
        f"{seed}:{image_id}:{tag}".encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value < keep_probability


def build_anima_caption_variants(
    row: dict[str, Any], cfg: dict[str, Any]
) -> list[tuple[int, str, str]]:
    """Return deterministic caption variants without leaking artist identity."""
    count = max(1, int(cfg.get("variants", 2)))
    result = [(0, "full", str(row["anima_caption"]))]
    if count == 1:
        return result

    rating = str(row.get("rating_anima") or "").strip()
    counts = [str(value) for value in row.get("count_tags") or []]
    characters = [str(value) for value in row.get("character_tags") or []]
    general = [str(value) for value in row.get("general_tags") or []]
    drop_rate = float(cfg.get("general_tag_dropout", 0.15))
    keep_probability = 1.0 - drop_rate
    seed = int(cfg.get("variant_seed", 20260811))
    image_id = int(row["id"])
    kept = [
        tag
        for tag in general
        if _stable_keep(seed, image_id, tag, keep_probability)
    ]
    if general and not kept:
        kept = [general[0]]
    parts = ([rating] if rating else []) + counts + characters + kept
    result.append((1, "general_dropout", ", ".join(parts)))
    # Additional variants deliberately reuse the same conservative policy until
    # a generation ablation justifies more aggressive caption perturbations.
    return result[:count]


def compute_anima_geometry(
    height: int, width: int, cfg: dict[str, Any]
) -> AnimaImageGeometry:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    step = int(cfg.get("bucket_step", 64))
    min_side = int(cfg.get("min_side", 256))
    max_side = int(cfg.get("max_side", 1536))
    max_pixels = int(cfg.get("max_pixels", 1024 * 1024))
    allow_upscale = bool(cfg.get("allow_upscale", False))
    upscale_below_min = bool(cfg.get("upscale_below_min", True))
    scale = min(max_side / max(height, width), math.sqrt(max_pixels / (height * width)))
    if not allow_upscale:
        scale = min(scale, 1.0)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    target_height = (resized_height // step) * step
    target_width = (resized_width // step) * step
    if target_height < min_side or target_width < min_side:
        if allow_upscale or upscale_below_min:
            boost = max(min_side / resized_height, min_side / resized_width)
            resized_height = int(math.ceil(resized_height * boost))
            resized_width = int(math.ceil(resized_width * boost))
            target_height = max(min_side, (resized_height // step) * step)
            target_width = max(min_side, (resized_width // step) * step)
        else:
            raise ValueError(
                f"Image {width}x{height} is too small for min_side={min_side} without upscaling"
            )
    target_height = min(target_height, resized_height)
    target_width = min(target_width, resized_width)
    return AnimaImageGeometry(
        resized_height=resized_height,
        resized_width=resized_width,
        target_height=target_height,
        target_width=target_width,
        crop_top=(resized_height - target_height) // 2,
        crop_left=(resized_width - target_width) // 2,
    )


def _decode_anima_image(payload: bytes, cfg: dict[str, Any]):
    decode_started = time.monotonic()
    with Image.open(BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        geometry = compute_anima_geometry(image.height, image.width, cfg)
        decode_s = time.monotonic() - decode_started
        resize_started = time.monotonic()
        if image.size != (geometry.resized_width, geometry.resized_height):
            image = image.resize(
                (geometry.resized_width, geometry.resized_height), Image.Resampling.LANCZOS
            )
        image = image.crop(
            (
                geometry.crop_left,
                geometry.crop_top,
                geometry.crop_left + geometry.target_width,
                geometry.crop_top + geometry.target_height,
            )
        )
        array = np.asarray(image, dtype=np.float32)
        array = np.ascontiguousarray(array.transpose(2, 0, 1) / 127.5 - 1.0)
    return array, geometry, decode_s, time.monotonic() - resize_started


def _caption_rows(destination: Path) -> list[dict[str, Any]]:
    paths = sorted((destination / "captions").glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError("Anima cache requires completed caption shards")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_records(path))
    return rows


def _load_completed(manifest_dir: Path, signature: str, key_fields: tuple[str, ...]):
    rows: list[dict[str, Any]] = []
    numbers: list[int] = []
    for path in sorted(manifest_dir.glob("part-*.parquet")):
        numbers.append(int(path.stem.split("-")[-1]))
        part = read_records(path)
        if any(row.get("cache_signature") != signature for row in part):
            raise RuntimeError(
                f"Existing cache under {manifest_dir.parent} has a different signature"
            )
        rows.extend(part)
    completed = {tuple(row[field] for field in key_fields) for row in rows}
    return rows, completed, (max(numbers) + 1 if numbers else 0)


def _import_sd_scripts(path: str):
    root = str(Path(path).resolve())
    if not Path(root, "library", "anima_utils.py").exists():
        raise FileNotFoundError(f"sd-scripts Anima implementation not found at {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    from library import anima_models, anima_utils, qwen_image_autoencoder_kl_2d

    return anima_models, anima_utils, qwen_image_autoencoder_kl_2d


def _verify_sd_scripts_revision(path: str, expected: str) -> str:
    actual = subprocess.check_output(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(
            f"sd-scripts revision mismatch: expected {expected}, found {actual}"
        )
    return actual


def _resolve_model_files(cfg: dict[str, Any], destination: Path) -> dict[str, str]:
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = str(cfg.get("repo_id", "circlestone-labs/Anima"))
    revision = str(cfg.get("revision", "main"))
    cache_dir = destination / "anima_model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "qwen3": str(
            cfg.get(
                "qwen3_filename",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
            )
        ),
        "dit": str(
            cfg.get(
                "dit_filename",
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
            )
        ),
        "vae": str(
            cfg.get("vae_filename", "split_files/vae/qwen_image_vae.safetensors")
        ),
    }
    resolved_revision = HfApi().model_info(repo_id, revision=revision).sha
    return {
        **{
            name: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=resolved_revision,
                cache_dir=cache_dir,
            )
            for name, filename in filenames.items()
        },
        "repo_id": repo_id,
        "revision": resolved_revision,
    }


def _load_llm_adapter(anima_models, checkpoint: str, device: str, dtype):
    import torch
    from safetensors import safe_open

    adapter = anima_models.LLMAdapter(
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        num_heads=16,
        self_attn=True,
    )
    state = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            marker = "llm_adapter."
            if marker in key:
                state[key.split(marker, 1)[1]] = handle.get_tensor(key)
    if not state:
        raise RuntimeError(f"No llm_adapter weights found in {checkpoint}")
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    meaningful_missing = [key for key in missing if "rotary_emb.inv_freq" not in key]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"LLM Adapter checkpoint mismatch: missing={meaningful_missing}, unexpected={unexpected}"
        )
    return adapter.requires_grad_(False).eval().to(device=device, dtype=dtype)


def _write_text_shard(path: Path, manifest_path: Path, items, signature: str):
    import torch
    from safetensors.torch import save_file

    started = time.monotonic()
    offsets = [0]
    conditions = []
    records = []
    for row, condition in items:
        conditions.append(condition.contiguous())
        offsets.append(offsets[-1] + int(condition.shape[0]))
        records.append({**row, "token_offset": offsets[-2], "token_length": int(condition.shape[0])})
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {
            "conditioning": torch.cat(conditions, dim=0),
            "offsets": torch.tensor(offsets, dtype=torch.int64),
            "ids": torch.tensor([int(row["id"]) for row, _ in items], dtype=torch.int64),
            "variants": torch.tensor([int(row["variant"]) for row, _ in items], dtype=torch.int16),
        },
        temporary,
    )
    temporary.replace(path)
    for row in records:
        row["cache_shard"] = path.name
        row["cache_signature"] = signature
    write_records(manifest_path, records)
    return records, path.stat().st_size, time.monotonic() - started


def _write_latent_shard(path: Path, manifest_path: Path, items, signature: str):
    import torch
    from safetensors.torch import save_file

    started = time.monotonic()
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {
            "latents": torch.stack([latent for _, latent in items]),
            "ids": torch.tensor([int(row["id"]) for row, _ in items], dtype=torch.int64),
        },
        temporary,
    )
    temporary.replace(path)
    records = []
    for index, (row, _) in enumerate(items):
        record = {**row, "row_index": index, "cache_shard": path.name, "cache_signature": signature}
        records.append(record)
    write_records(manifest_path, records)
    return records, path.stat().st_size, time.monotonic() - started


def cache_anima_text_conditions(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    cfg = config["anima_cache"]
    text_cfg = cfg["text"]
    models = _resolve_model_files(cfg["models"], destination)
    sd_root = str(cfg["sd_scripts_path"])
    _verify_sd_scripts_revision(sd_root, str(cfg["sd_scripts_revision"]))
    anima_models, anima_utils, _ = _import_sd_scripts(sd_root)
    dtype = torch.bfloat16
    device = str(text_cfg.get("device", "cuda"))
    qwen, qwen_tokenizer = anima_utils.load_qwen3_text_encoder(
        models["qwen3"], dtype=dtype, device=device
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer()
    adapter = _load_llm_adapter(anima_models, models["dit"], device, dtype)

    signature_payload = {
        "kind": "anima-post-llm-text-v1",
        "model_repo": models["repo_id"],
        "model_revision": models["revision"],
        "sd_scripts_revision": str(cfg["sd_scripts_revision"]),
        "qwen_max_length": int(text_cfg.get("qwen_max_length", 512)),
        "t5_max_length": int(text_cfg.get("t5_max_length", 512)),
        "variants": int(text_cfg.get("variants", 2)),
        "general_tag_dropout": float(text_cfg.get("general_tag_dropout", 0.15)),
        "variant_seed": int(text_cfg.get("variant_seed", 20260811)),
        "storage_dtype": "float16",
        "max_images": text_cfg.get("max_images"),
    }
    signature = _signature(signature_payload)
    output = destination / str(text_cfg.get("output_directory", "anima_text_cache"))
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    completed_rows, completed, shard_index = _load_completed(
        manifests, signature, ("id", "variant")
    )

    work = []
    caption_rows = sorted(_caption_rows(destination), key=lambda row: int(row["id"]))
    if text_cfg.get("max_images") is not None:
        caption_rows = caption_rows[: int(text_cfg["max_images"])]
    for row in caption_rows:
        for variant, variant_name, caption in build_anima_caption_variants(row, text_cfg):
            if (int(row["id"]), variant) in completed:
                continue
            work.append(
                {
                    "id": int(row["id"]),
                    "artist": row["artist"],
                    "style_id": row.get("style_id", row["artist"]),
                    "split": row.get("split", "train"),
                    "variant": variant,
                    "variant_name": variant_name,
                    "caption": caption,
                    "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
                }
            )
    work.sort(key=lambda row: (len(row["caption"]), int(row["id"]), int(row["variant"])))
    batch_size = int(text_cfg.get("batch_size", 64))
    shard_rows = int(text_cfg.get("shard_rows", 512))
    max_qwen = int(text_cfg.get("qwen_max_length", 512))
    max_t5 = int(text_cfg.get("t5_max_length", 512))
    writer_workers = int(text_cfg.get("writer_workers", 2))
    pending_limit = int(text_cfg.get("pending_writes", 4))
    writer = ThreadPoolExecutor(max_workers=writer_workers)
    pending: deque[Future] = deque()
    shard_buffer = []
    output_rows = list(completed_rows)
    written_bytes = 0
    write_s = 0.0
    write_wait_s = 0.0
    started = time.monotonic()

    def consume(future: Future):
        nonlocal written_bytes, write_s, write_wait_s
        wait_started = time.monotonic()
        rows, size, duration = future.result()
        write_wait_s += time.monotonic() - wait_started
        output_rows.extend(rows)
        written_bytes += size
        write_s += duration

    def flush():
        nonlocal shard_buffer, shard_index
        if not shard_buffer:
            return
        path = output / f"part-{shard_index:05d}.safetensors"
        manifest_path = manifests / f"part-{shard_index:05d}.parquet"
        pending.append(writer.submit(_write_text_shard, path, manifest_path, shard_buffer, signature))
        shard_buffer = []
        shard_index += 1
        if len(pending) >= pending_limit:
            consume(pending.popleft())
        while pending and pending[0].done():
            consume(pending.popleft())

    torch.backends.cuda.matmul.allow_tf32 = bool(text_cfg.get("allow_tf32", True))
    processed = 0
    for offset in range(0, len(work), batch_size):
        batch = work[offset : offset + batch_size]
        captions = [row["caption"] for row in batch]
        qwen_tokens = qwen_tokenizer(
            captions,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_qwen,
        )
        t5_tokens = t5_tokenizer(
            captions,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_t5,
        )
        qwen_ids = qwen_tokens["input_ids"].to(device, non_blocking=True)
        qwen_mask = qwen_tokens["attention_mask"].to(device, non_blocking=True)
        t5_ids = t5_tokens["input_ids"].to(device, non_blocking=True)
        t5_mask = t5_tokens["attention_mask"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
            hidden = qwen(input_ids=qwen_ids, attention_mask=qwen_mask).last_hidden_state
            hidden[~qwen_mask.bool()] = 0
            condition = adapter(
                hidden,
                t5_ids,
                target_attention_mask=t5_mask,
                source_attention_mask=qwen_mask,
            )
            condition[~t5_mask.bool()] = 0
        condition = condition.to("cpu", dtype=torch.float16)
        lengths = t5_mask.sum(dim=1).to("cpu").tolist()
        for row, tensor, length in zip(batch, condition, lengths):
            shard_buffer.append((row, tensor[: int(length)].clone()))
            if len(shard_buffer) >= shard_rows:
                flush()
        processed += len(batch)
        if processed % max(1000, batch_size) == 0 or processed == len(work):
            elapsed = time.monotonic() - started
            print(
                f"cached Anima text {processed}/{len(work)} ({processed / max(elapsed, 1e-6):.2f} items/s) "
                f"qwen_tokens={qwen_ids.shape[1]} t5_tokens={t5_ids.shape[1]}",
                flush=True,
            )

    flush()
    while pending:
        consume(pending.popleft())
    writer.shutdown(wait=True)
    output_rows.sort(key=lambda row: (int(row["id"]), int(row["variant"])))
    if output_rows:
        write_records(output / "manifest.parquet", output_rows)

    # Reproduce sd-scripts' Anima caption-dropout path exactly: zero Qwen
    # context plus a one-token T5 </s> target. Dynamic tokenization of an empty
    # string produces a zero-length Qwen sequence, which Qwen3 cannot execute.
    null_source = torch.zeros((1, 1, 1024), dtype=dtype, device=device)
    null_source_mask = torch.zeros((1, 1), dtype=torch.long, device=device)
    null_target_ids = torch.ones((1, 1), dtype=torch.long, device=device)
    null_target_mask = torch.ones_like(null_target_ids)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
        null_condition = adapter(
            null_source,
            null_target_ids,
            target_attention_mask=null_target_mask,
            source_attention_mask=null_source_mask,
        )
    save_file(
        {
            "empty_prompt": null_condition[0].half().cpu(),
            "caption_dropout_null": null_condition[0].half().cpu(),
        },
        output / "null_conditioning.safetensors",
    )
    total_bytes = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    summary = {
        "items": len(output_rows),
        "images": len({int(row["id"]) for row in output_rows}),
        "previously_cached": len(completed_rows),
        "newly_cached": len(work),
        "storage_bytes": total_bytes,
        "cache_signature": signature,
        "signature_payload": signature_payload,
        "model_revision": models["revision"],
        "throughput_items_s": len(work) / max(time.monotonic() - started, 1e-6),
        "write_s": write_s,
        "write_wait_s": write_wait_s,
    }
    write_json(output / "summary.json", summary)
    del adapter, qwen
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def cache_anima_vae_latents(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch

    cfg = config["anima_cache"]
    latent_cfg = cfg["latents"]
    models = _resolve_model_files(cfg["models"], destination)
    _verify_sd_scripts_revision(
        str(cfg["sd_scripts_path"]), str(cfg["sd_scripts_revision"])
    )
    _, _, qwen_vae = _import_sd_scripts(str(cfg["sd_scripts_path"]))
    device = str(latent_cfg.get("device", "cuda"))
    dtype = torch.bfloat16
    vae = qwen_vae.load_vae(models["vae"], device="cpu", disable_mmap=True)
    vae = vae.requires_grad_(False).eval().to(device=device, dtype=dtype)
    preprocess_cfg = dict(latent_cfg["preprocess"])
    signature_payload = {
        "kind": "anima-qwen-image-vae-2d-v1",
        "model_repo": models["repo_id"],
        "model_revision": models["revision"],
        "sd_scripts_revision": str(cfg["sd_scripts_revision"]),
        "preprocess": preprocess_cfg,
        "storage_dtype": "float16",
        "posterior": "mode",
        "max_images": latent_cfg.get("max_images"),
    }
    signature = _signature(signature_payload)
    output = destination / str(latent_cfg.get("output_directory", "anima_latent_cache"))
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    completed_rows, completed, shard_index = _load_completed(manifests, signature, ("id",))
    completed_ids = {int(key[0]) for key in completed}
    inventory = {
        int(row["id"]): row
        for row in read_records(destination / "final_manifest.parquet")
    }
    rows = []
    for caption_row in _caption_rows(destination):
        image_id = int(caption_row["id"])
        if image_id in completed_ids:
            continue
        source = inventory[image_id]
        rows.append(
            {
                **caption_row,
                "width": source.get("width"),
                "height": source.get("height"),
                "decoded_width": source.get("decoded_width"),
                "decoded_height": source.get("decoded_height"),
            }
        )

    def predicted(row):
        height = int(row.get("decoded_height") or row.get("height") or 0)
        width = int(row.get("decoded_width") or row.get("width") or 0)
        if not height or not width:
            with Image.open(row["local_path"]) as image:
                height, width = image.height, image.width
        geometry = compute_anima_geometry(height, width, preprocess_cfg)
        return geometry.target_height, geometry.target_width

    rows.sort(key=lambda row: (*predicted(row), int(row["id"])))
    if latent_cfg.get("max_images") is not None:
        rows = rows[: int(latent_cfg["max_images"])]
    batch_size = int(latent_cfg.get("batch_size", 4))
    shard_rows = int(latent_cfg.get("shard_rows", 512))
    reader_workers = int(latent_cfg.get("reader_workers", 32))
    decoder_workers = int(latent_cfg.get("decoder_workers", 16))
    read_prefetch = int(latent_cfg.get("read_prefetch", 4096))
    decode_prefetch = int(latent_cfg.get("decode_prefetch", 512))
    writer_workers = int(latent_cfg.get("writer_workers", 4))
    pending_limit = int(latent_cfg.get("pending_writes", 8))
    pinned_count = max(2, int(latent_cfg.get("pinned_buffers", 2)))
    writer = ThreadPoolExecutor(max_workers=writer_workers)
    gpu_executor = ThreadPoolExecutor(max_workers=1)
    pending_writes: deque[Future] = deque()
    pending_gpu: deque[tuple[Future, int]] = deque()
    output_rows = list(completed_rows)
    shard_buffer = []
    current_shard_shape = None
    host_buffers = []
    host_shape = None
    free_buffers: deque[int] = deque()
    timing = {key: 0.0 for key in ("read_s", "decode_s", "resize_s", "gpu_s", "write_s", "write_wait_s")}
    written_bytes = 0
    started = time.monotonic()
    encoded = 0

    def consume_write(future: Future):
        nonlocal written_bytes
        wait_started = time.monotonic()
        records, size, duration = future.result()
        timing["write_wait_s"] += time.monotonic() - wait_started
        timing["write_s"] += duration
        written_bytes += size
        output_rows.extend(records)

    def flush_shard():
        nonlocal shard_buffer, shard_index, current_shard_shape
        if not shard_buffer:
            return
        h = int(shard_buffer[0][0]["target_height"])
        w = int(shard_buffer[0][0]["target_width"])
        path = output / f"part-{shard_index:05d}-{w:04d}x{h:04d}.safetensors"
        manifest_path = manifests / f"part-{shard_index:05d}.parquet"
        pending_writes.append(writer.submit(_write_latent_shard, path, manifest_path, shard_buffer, signature))
        shard_buffer = []
        current_shard_shape = None
        shard_index += 1
        if len(pending_writes) >= pending_limit:
            consume_write(pending_writes.popleft())
        while pending_writes and pending_writes[0].done():
            consume_write(pending_writes.popleft())

    def accept_latents(items, latents, gpu_s):
        nonlocal current_shard_shape, encoded
        timing["gpu_s"] += gpu_s
        for (row, _, geometry), latent in zip(items, latents):
            shape = tuple(latent.shape)
            if current_shard_shape is not None and shape != current_shard_shape:
                flush_shard()
            current_shard_shape = shape
            record = {
                "id": int(row["id"]),
                "artist": row["artist"],
                "style_id": row.get("style_id", row["artist"]),
                "split": row.get("split", "train"),
                "local_path": row["local_path"],
                "target_height": geometry.target_height,
                "target_width": geometry.target_width,
                "resized_height": geometry.resized_height,
                "resized_width": geometry.resized_width,
                "crop_top": geometry.crop_top,
                "crop_left": geometry.crop_left,
                "latent_height": int(latent.shape[-2]),
                "latent_width": int(latent.shape[-1]),
            }
            shard_buffer.append((record, latent))
            encoded += 1
            if len(shard_buffer) >= shard_rows:
                flush_shard()

    def consume_gpu():
        future, buffer_index = pending_gpu.popleft()
        items, latents, duration = future.result()
        accept_latents(items, latents, duration)
        free_buffers.append(buffer_index)

    def run_gpu(items, host_images):
        gpu_started = time.monotonic()
        images = host_images.to(device, non_blocking=device.startswith("cuda"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
            latents = vae.encode_pixels_to_latents(images)
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        latents = latents.to("cpu", dtype=torch.float16)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return items, latents, time.monotonic() - gpu_started

    def submit_batch(items):
        nonlocal host_buffers, host_shape, free_buffers
        shape = tuple(items[0][1].shape)
        if host_shape != shape:
            while pending_gpu:
                consume_gpu()
            host_buffers = [
                torch.empty((batch_size, *shape), dtype=torch.float32, pin_memory=device.startswith("cuda"))
                for _ in range(pinned_count)
            ]
            free_buffers = deque(range(pinned_count))
            host_shape = shape
        if not free_buffers:
            consume_gpu()
        index = free_buffers.popleft()
        host = host_buffers[index][: len(items)]
        torch.stack([item[1] for item in items], out=host)
        pending_gpu.append((gpu_executor.submit(run_gpu, items, host), index))

    def read_image(index, row):
        read_started = time.monotonic()
        payload = Path(row["local_path"]).read_bytes()
        return index, row, payload, time.monotonic() - read_started

    def decode_image(item):
        index, row, payload, read_s = item
        array, geometry, decode_s, resize_s = _decode_anima_image(payload, preprocess_cfg)
        return index, row, torch.from_numpy(array), geometry, read_s, decode_s, resize_s

    buckets: dict[tuple[int, int], list] = {}
    with ThreadPoolExecutor(max_workers=reader_workers) as readers, ThreadPoolExecutor(max_workers=decoder_workers) as decoders:
        ready_reads: Queue[Future] = Queue()
        ready_decodes: Queue[Future] = Queue()
        outstanding_reads = outstanding_decodes = next_read = processed = 0
        decoded_ready = {}

        def submit_decode(future):
            nonlocal outstanding_reads, outstanding_decodes
            outstanding_reads -= 1
            decode_future = decoders.submit(decode_image, future.result())
            decode_future.add_done_callback(ready_decodes.put)
            outstanding_decodes += 1

        while processed < len(rows):
            while next_read < len(rows) and outstanding_reads < read_prefetch:
                future = readers.submit(read_image, next_read, rows[next_read])
                future.add_done_callback(ready_reads.put)
                outstanding_reads += 1
                next_read += 1
            while outstanding_decodes < decode_prefetch:
                try:
                    submit_decode(ready_reads.get_nowait())
                except Empty:
                    break
            if not outstanding_decodes and outstanding_reads:
                submit_decode(ready_reads.get())
            completed_decodes = [ready_decodes.get()]
            while True:
                try:
                    completed_decodes.append(ready_decodes.get_nowait())
                except Empty:
                    break
            outstanding_decodes -= len(completed_decodes)
            for future in completed_decodes:
                result = future.result()
                decoded_ready[result[0]] = result[1:]
            while processed in decoded_ready:
                row, tensor, geometry, read_s, decode_s, resize_s = decoded_ready.pop(processed)
                timing["read_s"] += read_s
                timing["decode_s"] += decode_s
                timing["resize_s"] += resize_s
                key = (geometry.target_height, geometry.target_width)
                bucket = buckets.setdefault(key, [])
                bucket.append((row, tensor, geometry))
                if len(bucket) >= batch_size:
                    submit_batch(bucket[:batch_size])
                    del bucket[:batch_size]
                processed += 1
                if processed % 1000 == 0:
                    while pending_gpu and pending_gpu[0][0].done():
                        consume_gpu()
                    elapsed = time.monotonic() - started
                    print(
                        f"prepared Anima latents {processed}/{len(rows)} ({processed / max(elapsed, 1e-6):.2f} images/s) "
                        f"encoded={encoded} pending_gpu={len(pending_gpu)}",
                        flush=True,
                    )

    for bucket in buckets.values():
        if bucket:
            submit_batch(bucket)
    while pending_gpu:
        consume_gpu()
    gpu_executor.shutdown(wait=True)
    flush_shard()
    while pending_writes:
        consume_write(pending_writes.popleft())
    writer.shutdown(wait=True)
    output_rows.sort(key=lambda row: int(row["id"]))
    if output_rows:
        write_records(output / "manifest.parquet", output_rows)
    total_bytes = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    summary = {
        "items": len(output_rows),
        "previously_cached": len(completed_rows),
        "newly_cached": encoded,
        "storage_bytes": total_bytes,
        "cache_signature": signature,
        "signature_payload": signature_payload,
        "model_revision": models["revision"],
        "throughput_images_s": encoded / max(time.monotonic() - started, 1e-6),
        "timing": timing,
        "pipeline": {
            "batch_size": batch_size,
            "shard_rows": shard_rows,
            "reader_workers": reader_workers,
            "decoder_workers": decoder_workers,
            "read_prefetch": read_prefetch,
            "decode_prefetch": decode_prefetch,
            "writer_workers": writer_workers,
            "pinned_buffers": pinned_count,
            "shape_sorted": True,
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def cache_all_anima_inputs(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    text = cache_anima_text_conditions(config, destination)
    latents = cache_anima_vae_latents(config, destination)
    summary = {"text": text, "latents": latents}
    write_json(destination / "anima_cache_summary.json", summary)
    return summary
