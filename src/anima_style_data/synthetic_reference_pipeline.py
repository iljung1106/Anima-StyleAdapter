from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file

from .cradio import _load_cradio, preprocess_cradio_image
from .dual_query_style_tokenizer import (
    _file_sha256,
    _load_resampler,
    _write_token_shard,
)
from .io import read_records, write_json, write_records


class _LatentShardCache:
    def __init__(self, root: Path, capacity: int = 2) -> None:
        self.root = root
        self.capacity = max(1, int(capacity))
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, name: str) -> torch.Tensor:
        value = self.values.pop(name, None)
        if value is None:
            value = load_file(self.root / name, device="cpu")["latents"]
        self.values[name] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def _decode_cradio_image(path: str, radio_cfg: dict[str, Any]) -> torch.Tensor:
    with Image.open(path) as image:
        array, _ = preprocess_cradio_image(image, radio_cfg)
    return torch.from_numpy(array)


def _teacher_split(row: dict[str, Any]) -> str:
    value = str(row.get("artist_split", row.get("teacher_split", "train")))
    return "test" if value == "meta_test" else value


def _cache_synthetic_dual_query_tokens(
    config: dict[str, Any], destination: Path, *, config_key: str
) -> dict[str, Any]:
    """Encode synthetic images directly into frozen Resampler tokens.

    C-RADIO intermediates remain on the GPU and are consumed by the Resampler
    in the same forward pipeline. Only the compact 84x1024 reference tokens
    and 512-D descriptors are materialized on the network volume.
    """

    cfg = dict(config[config_key])
    cache_cfg = dict(cfg["dual_query_token_cache"])
    device = str(cache_cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for direct synthetic token caching")
    root = destination / str(cfg["output_directory"])
    manifest_path = root / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Synthetic generation is incomplete: {manifest_path} is absent"
        )
    rows = [
        {
            **row,
            "split": _teacher_split(row),
        }
        for row in read_records(manifest_path)
        if str(row.get("kind")) == "artist"
    ]
    if not rows:
        raise RuntimeError("Synthetic manifest contains no artist images")
    rows.sort(
        key=lambda row: (
            str(row["latent_shard"]),
            int(row["latent_row"]),
            int(row["id"]),
        )
    )

    checkpoint_value = cache_cfg.get(
        "resampler_checkpoint",
        config["dual_query_style_tokenizer"]["resampler_checkpoint"],
    )
    checkpoint = destination / str(checkpoint_value)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = _file_sha256(checkpoint)
    semantic_layers = tuple(
        int(value)
        for value in config["dual_query_resampler"]["model"].get(
            "semantic_layers", [18, 24]
        )
    )
    radio_cfg = {
        **dict(config["cradio"]),
        **dict(cache_cfg.get("preprocess", {"max_side": 512, "max_pixels": 262144})),
    }
    signature_payload = {
        "kind": "direct-synthetic-dual-query-reference-tokens-v1",
        "config_key": config_key,
        "checkpoint_sha256": checkpoint_sha256,
        "radio_model": radio_cfg["model_version"],
        "radio_ref": radio_cfg["torchhub_ref"],
        "semantic_layers": semantic_layers,
        "width": int(cfg["width"]),
        "height": int(cfg["height"]),
        "image_id_base": int(cfg["image_id_base"]),
        "slots": 84,
        "dim": 1024,
        "dtype": "bfloat16",
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    output = root / str(cache_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("cache_signature") != signature:
            raise RuntimeError("Existing direct token cache has another signature")
        if int(summary.get("images", 0)) == len(rows):
            return {**summary, "reused": True}

    batch_size = int(cache_cfg.get("batch_size", 32))
    shard_rows = int(cache_cfg.get("shard_rows", 512))
    reader_workers = int(cache_cfg.get("reader_workers", 16))
    prefetch_batches = int(cache_cfg.get("prefetch_batches", 8))
    writer_workers = int(cache_cfg.get("writer_workers", 2))
    maximum_pending_writes = int(cache_cfg.get("pending_writes", 2))
    latent_cache = _LatentShardCache(
        root / "latents", int(cache_cfg.get("latent_lru_shards", 2))
    )
    shard_groups = [
        rows[offset : offset + shard_rows]
        for offset in range(0, len(rows), shard_rows)
    ]
    completed_records: list[dict[str, Any]] = []
    written_bytes = 0
    pending_writes: list[Future[tuple[list[dict[str, Any]], int]]] = []
    writer = ThreadPoolExecutor(max_workers=max(1, writer_workers))
    started = time.perf_counter()
    timing = {
        "decode_wait_s": 0.0,
        "latent_copy_s": 0.0,
        "gpu_s": 0.0,
        "write_wait_s": 0.0,
    }

    def consume_write(future: Future[tuple[list[dict[str, Any]], int]]) -> None:
        nonlocal written_bytes
        wait_started = time.perf_counter()
        records, size = future.result()
        timing["write_wait_s"] += time.perf_counter() - wait_started
        completed_records.extend(records)
        written_bytes += size

    radio = _load_cradio(
        radio_cfg,
        Path(cache_cfg.get("model_cache_directory", destination / "cradio_model_cache")),
    )[0].requires_grad_(False).eval()
    semantic_dim = int(cache_cfg.get("semantic_dim", 1536))
    resampler, checkpoint_step = _load_resampler(
        config,
        destination,
        checkpoint,
        semantic_dim,
        int(cache_cfg.get("vae_channels", 16)),
        device,
    )
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(cache_cfg.get("amp_dtype", "bfloat16"))]
    height, width = int(cfg["height"]), int(cfg["width"])
    target_height, target_width = height, width
    image_buffers = [
        torch.empty(
            batch_size,
            3,
            target_height,
            target_width,
            dtype=torch.float32,
            pin_memory=device.startswith("cuda"),
        )
        for _ in range(2)
    ]
    latent_buffers = [
        torch.empty(
            batch_size,
            int(cache_cfg.get("vae_channels", 16)),
            height // 8,
            width // 8,
            dtype=torch.float16,
            pin_memory=device.startswith("cuda"),
        )
        for _ in range(2)
    ]
    copy_events: list[torch.cuda.Event | None] = [None, None]

    incomplete_groups: list[tuple[int, list[dict[str, Any]]]] = []
    for shard_index, shard_items in enumerate(shard_groups):
        tensor_path = output / f"part-{shard_index:05d}.safetensors"
        row_path = output / f"part-{shard_index:05d}.parquet"
        if tensor_path.exists() and row_path.exists():
            records = read_records(row_path)
            if any(row.get("cache_signature") != signature for row in records):
                raise RuntimeError(f"Token shard {shard_index} signature mismatch")
            completed_records.extend(records)
            written_bytes += tensor_path.stat().st_size
        else:
            incomplete_groups.append((shard_index, shard_items))

    flattened = [row for _, items in incomplete_groups for row in items]
    decode_futures: dict[int, Future[torch.Tensor]] = {}
    next_decode = 0
    decode_depth = max(batch_size, batch_size * prefetch_batches)

    def fill_decode(executor: ThreadPoolExecutor, consumed: int) -> None:
        nonlocal next_decode
        limit = min(len(flattened), consumed + decode_depth)
        while next_decode < limit:
            row = flattened[next_decode]
            decode_futures[next_decode] = executor.submit(
                _decode_cradio_image, str(row["local_path"]), radio_cfg
            )
            next_decode += 1

    flat_offset = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, reader_workers)) as reader:
            fill_decode(reader, 0)
            for shard_index, shard_items in incomplete_groups:
                token_parts = []
                descriptor_parts = []
                for local_offset in range(0, len(shard_items), batch_size):
                    batch = shard_items[local_offset : local_offset + batch_size]
                    count = len(batch)
                    buffer_index = (flat_offset // batch_size) % 2
                    if copy_events[buffer_index] is not None:
                        copy_events[buffer_index].synchronize()
                    decode_started = time.perf_counter()
                    for index in range(count):
                        image_buffers[buffer_index][index].copy_(
                            decode_futures.pop(flat_offset + index).result()
                        )
                    timing["decode_wait_s"] += time.perf_counter() - decode_started
                    fill_decode(reader, flat_offset + count)

                    latent_started = time.perf_counter()
                    for index, row in enumerate(batch):
                        shard = latent_cache.get(str(row["latent_shard"]))
                        latent_buffers[buffer_index][index].copy_(
                            shard[int(row["latent_row"])]
                        )
                    timing["latent_copy_s"] += time.perf_counter() - latent_started

                    gpu_started = time.perf_counter()
                    images = image_buffers[buffer_index][:count].to(
                        device, non_blocking=device.startswith("cuda")
                    )
                    latents = latent_buffers[buffer_index][:count].to(
                        device, non_blocking=device.startswith("cuda")
                    )
                    if device.startswith("cuda"):
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(device))
                        copy_events[buffer_index] = event
                    with torch.inference_mode(), torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=amp_dtype,
                        enabled=device.startswith("cuda"),
                    ):
                        intermediate = radio.forward_intermediates(
                            images,
                            indices=list(semantic_layers),
                            return_prefix_tokens=False,
                            norm=True,
                            stop_early=True,
                            output_fmt="NLC",
                            intermediates_only=True,
                            aggregation="sparse",
                        )
                        semantic_features = {
                            layer: (
                                value.features
                                if hasattr(value, "features")
                                else value
                            )
                            for layer, value in zip(
                                semantic_layers, intermediate, strict=True
                            )
                        }
                        tokens = int(next(iter(semantic_features.values())).shape[1])
                        grid_height, remainder = divmod(target_height, 16)
                        grid_width = target_width // 16
                        if remainder or tokens != grid_height * grid_width:
                            raise RuntimeError(
                                f"Unexpected C-RADIO token grid: {tokens}"
                            )
                        encoded = resampler.encode(
                            semantic_features,
                            torch.ones(count, tokens, dtype=torch.bool, device=device),
                            torch.tensor(
                                [[grid_height, grid_width]] * count,
                                dtype=torch.long,
                                device=device,
                            ),
                            latents,
                            torch.tensor(
                                [[height // 8, width // 8]] * count,
                                dtype=torch.long,
                                device=device,
                            ),
                            torch.tensor(
                                [[height, width]] * count,
                                dtype=torch.long,
                                device=device,
                            ),
                            reconstruct=False,
                        )
                    combined = torch.cat(
                        (encoded.tokens, encoded.artist_summary), dim=1
                    )
                    if tuple(combined.shape[1:]) != (84, 1024):
                        raise RuntimeError(
                            f"Unexpected direct token shape {tuple(combined.shape)}"
                        )
                    token_parts.append(combined.to("cpu", dtype=torch.bfloat16))
                    descriptor_parts.append(
                        encoded.descriptor.to("cpu", dtype=torch.bfloat16)
                    )
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    timing["gpu_s"] += time.perf_counter() - gpu_started
                    flat_offset += count

                pending_writes.append(
                    writer.submit(
                        _write_token_shard,
                        output,
                        shard_index,
                        torch.cat(token_parts),
                        torch.cat(descriptor_parts),
                        shard_items,
                        signature,
                    )
                )
                if len(pending_writes) >= maximum_pending_writes:
                    consume_write(pending_writes.pop(0))
                processed = len(completed_records) + sum(
                    len(shard_groups[index])
                    for index, _ in incomplete_groups
                    if index <= shard_index
                )
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"direct synthetic tokens {min(processed, len(rows))}/{len(rows)} "
                    f"({min(processed, len(rows)) / elapsed:.1f} img/s) ",
                    f"decode_wait={timing['decode_wait_s']:.1f}s "
                    f"gpu={timing['gpu_s']:.1f}s",
                    flush=True,
                )
        for future in pending_writes:
            consume_write(future)
    finally:
        writer.shutdown(wait=True)
        del radio, resampler
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    completed_records.sort(key=lambda row: int(row["id"]))
    expected_ids = {int(row["id"]) for row in rows}
    if {int(row["id"]) for row in completed_records} != expected_ids:
        raise RuntimeError("Direct synthetic token cache ID set mismatch")
    write_records(output / "manifest.parquet", completed_records)
    summary = {
        "images": len(completed_records),
        "shards": len(shard_groups),
        "slots": 84,
        "query_slots": 80,
        "artist_summary_slots": 4,
        "style_dim": 1024,
        "descriptor_dim": 512,
        "dtype": "bfloat16",
        "resampler_checkpoint": str(checkpoint_value),
        "resampler_checkpoint_step": checkpoint_step,
        "resampler_checkpoint_sha256": checkpoint_sha256,
        "cache_signature": signature,
        "storage_bytes": written_bytes,
        "elapsed_s": time.perf_counter() - started,
        "timing": timing,
        "materialized_cradio_features": False,
    }
    write_json(summary_path, summary)
    return summary


def cache_additional_synthetic_dual_query_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _cache_synthetic_dual_query_tokens(
        config, destination, config_key="synthetic_teacher_additional"
    )


def cache_lora_teacher_dual_query_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _cache_synthetic_dual_query_tokens(
        config, destination, config_key="lora_teacher_references"
    )
