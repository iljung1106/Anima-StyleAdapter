from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from threading import Condition
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .io import read_records, write_json, write_records


@dataclass(frozen=True)
class PreprocessInfo:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    target_height: int
    target_width: int
    crop_left: int
    crop_top: int


def compute_cradio_size(
    height: int,
    width: int,
    *,
    max_side: int,
    max_pixels: int,
    step: int = 16,
    min_side: int = 16,
) -> tuple[int, int, int, int]:
    if height < min_side or width < min_side:
        raise ValueError(f"Image is too small for C-RADIO: {width}x{height}")
    scale = min(
        1.0,
        max_side / max(height, width),
        math.sqrt(max_pixels / (height * width)),
    )
    resized_h = max(min_side, int(math.floor(height * scale)))
    resized_w = max(min_side, int(math.floor(width * scale)))
    target_h = (resized_h // step) * step
    target_w = (resized_w // step) * step
    if target_h < min_side or target_w < min_side:
        raise ValueError(f"Aligned C-RADIO size is too small: {target_w}x{target_h}")
    return resized_h, resized_w, target_h, target_w


def preprocess_cradio_image(
    image: Image.Image, cfg: dict[str, Any]
) -> tuple[np.ndarray, PreprocessInfo]:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")
    original_w, original_h = image.size
    resized_h, resized_w, target_h, target_w = compute_cradio_size(
        original_h,
        original_w,
        max_side=int(cfg["max_side"]),
        max_pixels=int(cfg["max_pixels"]),
        step=int(cfg["patch_size"]),
        min_side=int(cfg["min_side"]),
    )
    if (resized_w, resized_h) != image.size:
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    if target_w != resized_w or target_h != resized_h:
        image = image.crop((left, top, left + target_w, top + target_h))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return array, PreprocessInfo(
        original_height=original_h,
        original_width=original_w,
        resized_height=resized_h,
        resized_width=resized_w,
        target_height=target_h,
        target_width=target_w,
        crop_left=left,
        crop_top=top,
    )


class _ByteBudget:
    """Bound compressed-image RAM independently from the number of futures."""

    def __init__(self, limit_bytes: int):
        self.limit_bytes = int(limit_bytes)
        self.used_bytes = 0
        self.peak_bytes = 0
        self._condition = Condition()

    def acquire(self, size: int) -> None:
        with self._condition:
            while self.used_bytes and self.used_bytes + size > self.limit_bytes:
                self._condition.wait()
            self.used_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def release(self, size: int) -> None:
        with self._condition:
            self.used_bytes -= size
            self._condition.notify_all()


def _decode_preprocess_bytes(
    payload: bytes, cfg: dict[str, Any]
) -> tuple[np.ndarray, PreprocessInfo, float, float]:
    """Decode WebP bytes, then preserve the established PIL preprocessing path."""
    with Image.open(BytesIO(payload)) as image:
        decode_started = time.monotonic()
        image.load()
        decode_s = time.monotonic() - decode_started
        resize_started = time.monotonic()
        array, info = preprocess_cradio_image(image, cfg)
        resize_s = time.monotonic() - resize_started
    return array, info, decode_s, resize_s


def _load_cradio(cfg: dict[str, Any], cache_dir: Path):
    import torch

    device = str(cfg["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for C-RADIO but is not available")
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = (cache_dir / "huggingface").resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HOME = str(hf_cache)
        hub_constants.HF_HUB_CACHE = str(hf_cache / "hub")
    except ImportError:
        pass

    torchhub_dir = (cache_dir / "torchhub").resolve()
    torchhub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torchhub_dir))
    repo = f"{cfg['torchhub_repo']}:{cfg['torchhub_ref']}"
    model = torch.hub.load(
        repo,
        "radio_model",
        source="github",
        trust_repo=True,
        version=cfg["model_version"],
        adaptor_names=[cfg["adaptor_name"]],
        progress=True,
        skip_validation=True,
    )
    model.eval().to(device)
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    return model, device


def _summary_features(output: Any) -> tuple[Any, Any]:
    if hasattr(output, "summary") and hasattr(output, "features"):
        return output.summary, output.features
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output[0], output[1]
    raise TypeError(f"Unsupported RADIO output type: {type(output)!r}")


def content_subtracted_summary(
    visual_summary,
    text_summary,
    *,
    scale: float = 1.0,
    normalize_residual: bool = True,
):
    import torch.nn.functional as F

    visual = F.normalize(visual_summary.float(), dim=-1)
    text = F.normalize(text_summary.float(), dim=-1)
    residual = visual - float(scale) * text
    if normalize_residual:
        residual = F.normalize(residual, dim=-1)
    return visual, text, residual


def _storage_dtype(name: str):
    import torch

    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported feature storage dtype: {name}")
    return mapping[name]


def _existing_feature_ids(manifest_dir: Path) -> tuple[set[int], int]:
    completed: set[int] = set()
    numbers = []
    for path in sorted(manifest_dir.glob("part-*.parquet")):
        numbers.append(int(path.stem.split("-")[-1]))
        completed.update(int(row["id"]) for row in read_records(path))
    return completed, (max(numbers) + 1 if numbers else 0)


def _caption_rows(destination: Path) -> list[dict[str, Any]]:
    shards = sorted((destination / "captions").glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError("No Anima caption shards found; run the caption stage first")
    rows = []
    for shard in shards:
        rows.extend(read_records(shard))
    return rows


def _selected_style_feature_signature(
    radio_cfg: dict[str, Any], feature_cfg: dict[str, Any]
) -> str:
    payload = {
        "model_version": radio_cfg["model_version"],
        "torchhub_ref": radio_cfg["torchhub_ref"],
        "spatial_layers": sorted(int(value) for value in feature_cfg["spatial_layers"]),
        "statistics_layers": sorted(
            int(value) for value in feature_cfg.get("statistics_layers", [])
        ),
        "summary_layers": sorted(
            int(value) for value in feature_cfg.get("summary_layers", [])
        ),
        "statistics": list(feature_cfg.get("statistics", ["mean", "std"])),
        "storage_dtype": feature_cfg["storage_dtype"],
        "preprocess": {
            key: int(radio_cfg[key])
            for key in ("patch_size", "min_side", "max_side", "max_pixels")
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_style_tensors(
    intermediate: list[Any],
    layers: list[int],
    spatial_layers: set[int],
    statistics_layers: set[int],
    summary_layers: set[int],
    storage_dtype: Any,
) -> list[dict[str, Any]]:
    """Collect production features with one reduction/transfer per layer batch."""
    batched: dict[str, Any] = {}
    for layer, output in zip(layers, intermediate):
        spatial = output.features if hasattr(output, "features") else output
        spatial = spatial.detach()
        if layer in spatial_layers:
            batched[f"layer_{layer:02d}_spatial"] = spatial.to(
                "cpu", dtype=storage_dtype
            )
        if layer in statistics_layers:
            stable = spatial.float()
            batched[f"layer_{layer:02d}_mean"] = stable.mean(dim=1).to(
                "cpu", dtype=storage_dtype
            )
            batched[f"layer_{layer:02d}_std"] = stable.std(
                dim=1, correction=0
            ).to("cpu", dtype=storage_dtype)
        if layer in summary_layers:
            if not hasattr(output, "summary"):
                raise RuntimeError(
                    f"C-RADIO layer {layer} did not return prefix/summary tokens"
                )
            # C-RADIO summary concatenates teacher slots. The first backbone-width
            # slice is the internal SigLIP2-g teacher CLS representation.
            siglip_cls = output.summary.detach()[..., : spatial.shape[-1]]
            batched[f"layer_{layer:02d}_siglip_cls"] = siglip_cls.to(
                "cpu", dtype=storage_dtype
            )

    batch_size = int(next(iter(batched.values())).shape[0])
    return [
        {name: values[item_index] for name, values in batched.items()}
        for item_index in range(batch_size)
    ]


def extract_selected_style_features(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Cache the configured production spatial and teacher-summary features."""
    import torch
    from safetensors.torch import save_file

    feature_cfg = config["style_features"]
    radio_cfg = {**config["cradio"], **feature_cfg.get("preprocess", {})}
    spatial_layers = {int(value) for value in feature_cfg["spatial_layers"]}
    statistics_layers = {
        int(value) for value in feature_cfg.get("statistics_layers", [])
    }
    summary_layers = {int(value) for value in feature_cfg.get("summary_layers", [])}
    statistics = set(feature_cfg.get("statistics", ["mean", "std"]))
    if statistics != {"mean", "std"}:
        raise ValueError("style_features.statistics must contain exactly mean and std")
    layers = sorted(spatial_layers | statistics_layers | summary_layers)
    signature = _selected_style_feature_signature(radio_cfg, feature_cfg)

    features_dir = destination / str(feature_cfg.get("output_directory", "style_features"))
    manifest_dir = features_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    completed_rows: list[dict[str, Any]] = []
    shard_numbers: list[int] = []
    for path in sorted(manifest_dir.glob("part-*.parquet")):
        shard_numbers.append(int(path.stem.split("-")[-1]))
        part = read_records(path)
        mismatched = [row for row in part if row.get("feature_signature") != signature]
        if mismatched:
            raise RuntimeError(
                "Existing style feature shards use a different configuration; "
                "choose a new output directory or move the old cache"
            )
        completed_rows.extend(part)
    completed_ids = {int(row["id"]) for row in completed_rows}
    # This cache is image-only. Read the post-dedup inventory directly so it
    # neither consumes captions nor depends on the caption stage.
    manifest_path = Path(
        feature_cfg.get("manifest_path", destination / "final_manifest.parquet")
    )
    if not manifest_path.is_absolute():
        manifest_path = destination / manifest_path
    all_rows = read_records(manifest_path)
    rows = [row for row in all_rows if int(row["id"]) not in completed_ids]
    shard_index = max(shard_numbers) + 1 if shard_numbers else 0

    if not rows:
        total_bytes = sum(path.stat().st_size for path in features_dir.glob("*.safetensors"))
        summary = {
            "total": len(completed_rows),
            "newly_encoded": 0,
            "spatial_layers": sorted(spatial_layers),
            "statistics_layers": sorted(statistics_layers),
            "summary_layers": sorted(summary_layers),
            "storage_bytes": total_bytes,
            "feature_signature": signature,
        }
        write_json(features_dir / "summary.json", summary)
        return summary

    def predicted_target_shape(row: dict[str, Any]) -> tuple[int, int]:
        height = int(row.get("decoded_height") or row["height"])
        width = int(row.get("decoded_width") or row["width"])
        _, _, target_height, target_width = compute_cradio_size(
            height,
            width,
            max_side=int(radio_cfg["max_side"]),
            max_pixels=int(radio_cfg["max_pixels"]),
            step=int(radio_cfg["patch_size"]),
            min_side=int(radio_cfg["min_side"]),
        )
        return target_height, target_width

    # Exact-shape batches avoid flushing many partially filled aspect buckets.
    # The decoder still verifies the actual post-EXIF shape and uses that as the
    # runtime bucket key, so imperfect source metadata cannot corrupt a batch.
    rows.sort(key=lambda row: (*predicted_target_shape(row), int(row["id"])))

    model_cache = Path(
        feature_cfg.get("model_cache_directory", destination / "cradio_model_cache")
    )
    if not model_cache.is_absolute():
        model_cache = destination / model_cache
    model, device = _load_cradio(radio_cfg, model_cache)
    if min(layers) < 0 or max(layers) >= len(model.model.blocks):
        raise ValueError(f"Feature layers must be in [0, {len(model.model.blocks) - 1}]")

    batch_size = int(feature_cfg.get("batch_size", radio_cfg["batch_size"]))
    shard_rows = int(feature_cfg.get("shard_rows", 128))
    max_open_buckets = int(feature_cfg.get("max_open_buckets", 64))
    reader_workers = int(feature_cfg.get("reader_workers", 64))
    decoder_workers = int(feature_cfg.get("decoder_workers", 32))
    read_prefetch = max(
        reader_workers, int(feature_cfg.get("read_prefetch", 32768))
    )
    decode_prefetch = max(
        decoder_workers, int(feature_cfg.get("decode_prefetch", 4096))
    )
    raw_budget = _ByteBudget(
        int(float(feature_cfg.get("raw_queue_gib", 12)) * (1024**3))
    )
    writer_workers = int(feature_cfg.get("writer_workers", 2))
    pending_writes_limit = max(
        writer_workers, int(feature_cfg.get("pending_writes", writer_workers * 2))
    )
    pinned_buffer_count = max(2, int(feature_cfg.get("pinned_buffers", 2)))
    storage_dtype = _storage_dtype(feature_cfg["storage_dtype"])
    amp_name = str(feature_cfg.get("amp_dtype", radio_cfg["amp_dtype"]))
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[amp_name]
    amp_enabled = device.startswith("cuda") and amp_name != "float32"
    buckets: dict[
        tuple[int, int], list[tuple[dict[str, Any], torch.Tensor, PreprocessInfo]]
    ] = defaultdict(list)
    tensor_buffer: dict[str, torch.Tensor] = {}
    record_buffer: list[dict[str, Any]] = []
    output_rows = list(completed_rows)
    newly_encoded = 0
    started = time.monotonic()
    timing = {
        "read_s": 0.0,
        "read_wait_s": 0.0,
        "decode_s": 0.0,
        "decode_wait_s": 0.0,
        "resize_s": 0.0,
        "stack_s": 0.0,
        "pinned_alloc_s": 0.0,
        "gpu_s": 0.0,
        "write_s": 0.0,
        "write_wait_s": 0.0,
        "written_bytes": 0,
    }
    gpu_batches = 0
    gpu_batch_images = 0
    pinned_allocations = 0
    write_executor = ThreadPoolExecutor(max_workers=writer_workers)
    pending_writes: deque[
        Future[tuple[list[dict[str, Any]], float, int]]
    ] = deque()

    def write_shard(
        feature_path: Path,
        manifest_path: Path,
        tensors: dict[str, torch.Tensor],
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float, int]:
        write_started = time.monotonic()
        temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
        save_file(
            {key: value.contiguous() for key, value in tensors.items()}, temporary
        )
        temporary.replace(feature_path)
        for record in records:
            record["feature_shard"] = feature_path.name
        write_records(manifest_path, records)
        print(
            f"wrote selected style feature shard {feature_path.stem.split('-')[-1]} "
            f"({len(records)} images)",
            flush=True,
        )
        return records, time.monotonic() - write_started, feature_path.stat().st_size

    def consume_write(future: Future[tuple[list[dict[str, Any]], float, int]]) -> None:
        wait_started = time.monotonic()
        records, write_s, written_bytes = future.result()
        timing["write_wait_s"] += time.monotonic() - wait_started
        timing["write_s"] += write_s
        timing["written_bytes"] += written_bytes
        output_rows.extend(records)

    def collect_write(*, block: bool) -> None:
        if block and pending_writes:
            consume_write(pending_writes.popleft())
        while pending_writes and pending_writes[0].done():
            consume_write(pending_writes.popleft())

    def flush_shard() -> None:
        nonlocal tensor_buffer, record_buffer, shard_index
        if not record_buffer:
            return
        feature_path = features_dir / f"part-{shard_index:05d}.safetensors"
        manifest_path = manifest_dir / f"part-{shard_index:05d}.parquet"
        pending_writes.append(
            write_executor.submit(
                write_shard,
                feature_path,
                manifest_path,
                tensor_buffer,
                record_buffer,
            )
        )
        tensor_buffer = {}
        record_buffer = []
        shard_index += 1
        collect_write(block=len(pending_writes) >= pending_writes_limit)

    def run_batch(
        items: list[tuple[dict[str, Any], torch.Tensor, PreprocessInfo]],
        host_images: torch.Tensor,
    ) -> None:
        nonlocal newly_encoded
        gpu_started = time.monotonic()
        images = host_images.to(device, non_blocking=device.startswith("cuda"))
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
        ):
            intermediate = model.forward_intermediates(
                images,
                indices=layers,
                return_prefix_tokens=bool(summary_layers),
                norm=True,
                stop_early=True,
                output_fmt="NLC",
                intermediates_only=True,
                aggregation="sparse",
            )
        selected = _selected_style_tensors(
            intermediate,
            layers,
            spatial_layers,
            statistics_layers,
            summary_layers,
            storage_dtype,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timing["gpu_s"] += time.monotonic() - gpu_started
        for (row, _, info), tensors in zip(items, selected):
            image_id = int(row["id"])
            for name, tensor in tensors.items():
                tensor_buffer[f"{image_id}.{name}"] = tensor
            record_buffer.append(
                {
                    "id": image_id,
                    "artist": row["artist"],
                    "style_id": row.get("style_id", row["artist"]),
                    "split": row.get("split", "train"),
                    "local_path": row["local_path"],
                    "target_height": info.target_height,
                    "target_width": info.target_width,
                    "spatial_tokens": (info.target_height // 16)
                    * (info.target_width // 16),
                    "spatial_dim": int(next(iter(tensors.values())).shape[-1]),
                    "storage_dtype": feature_cfg["storage_dtype"],
                    "feature_signature": signature,
                    "model_version": radio_cfg["model_version"],
                    "torchhub_ref": radio_cfg["torchhub_ref"],
                }
            )
            newly_encoded += 1
        if len(record_buffer) >= shard_rows:
            flush_shard()

    gpu_executor = ThreadPoolExecutor(max_workers=1)
    pending_gpu: deque[tuple[Future[None], int]] = deque()
    free_buffers: deque[int] = deque()
    host_buffers: list[torch.Tensor] = []
    host_shape: tuple[int, ...] | None = None

    def consume_gpu() -> None:
        future, buffer_index = pending_gpu.popleft()
        future.result()
        free_buffers.append(buffer_index)

    def drain_gpu() -> None:
        while pending_gpu:
            consume_gpu()

    def submit_batch(
        items: list[tuple[dict[str, Any], torch.Tensor, PreprocessInfo]],
    ) -> None:
        nonlocal host_buffers, host_shape, free_buffers, gpu_batches, gpu_batch_images
        nonlocal pinned_allocations
        shape = tuple(items[0][1].shape)
        if host_shape != shape:
            # Shape-sorted rows make this infrequent. Drain before replacing the
            # two reusable host buffers so an asynchronous H2D copy cannot race
            # with reuse or deallocation.
            drain_gpu()
            allocation_started = time.monotonic()
            host_buffers = [
                torch.empty(
                    (batch_size, *shape),
                    dtype=items[0][1].dtype,
                    pin_memory=device.startswith("cuda"),
                )
                for _ in range(pinned_buffer_count)
            ]
            timing["pinned_alloc_s"] += time.monotonic() - allocation_started
            pinned_allocations += pinned_buffer_count
            free_buffers = deque(range(pinned_buffer_count))
            host_shape = shape
        if not free_buffers:
            consume_gpu()
        buffer_index = free_buffers.popleft()
        host_images = host_buffers[buffer_index][: len(items)]
        stack_started = time.monotonic()
        torch.stack([item[1] for item in items], out=host_images)
        timing["stack_s"] += time.monotonic() - stack_started
        pending_gpu.append(
            (gpu_executor.submit(run_batch, items, host_images), buffer_index)
        )
        gpu_batches += 1
        gpu_batch_images += len(items)

    def flush_bucket(key: tuple[int, int]) -> None:
        items = buckets.pop(key)
        for offset in range(0, len(items), batch_size):
            submit_batch(items[offset : offset + batch_size])

    def read_image(row_index: int, row: dict[str, Any]):
        read_started = time.monotonic()
        payload = Path(row["local_path"]).read_bytes()
        read_s = time.monotonic() - read_started
        raw_budget.acquire(len(payload))
        return row_index, row, payload, read_s

    def decode_image(item: tuple[int, dict[str, Any], bytes, float]):
        row_index, row, payload, read_s = item
        payload_size = len(payload)
        try:
            array, info, decode_s, resize_s = _decode_preprocess_bytes(
                payload, radio_cfg
            )
        finally:
            raw_budget.release(payload_size)
        return row_index, row, torch.from_numpy(array), info, read_s, decode_s, resize_s

    with (
        ThreadPoolExecutor(max_workers=reader_workers) as read_executor,
        ThreadPoolExecutor(max_workers=decoder_workers) as decode_executor,
    ):
        ready_reads: Queue[Future[Any]] = Queue()
        ready_decodes: Queue[Future[Any]] = Queue()
        outstanding_reads = 0
        outstanding_decodes = 0
        next_read = 0
        processed = 0
        active_bucket_key: tuple[int, int] | None = None
        decoded_ready: dict[
            int,
            tuple[dict[str, Any], torch.Tensor, PreprocessInfo, float, float, float],
        ] = {}

        def submit_decode(future: Future[Any]) -> None:
            nonlocal outstanding_reads, outstanding_decodes
            outstanding_reads -= 1
            decode_future = decode_executor.submit(decode_image, future.result())
            decode_future.add_done_callback(ready_decodes.put)
            outstanding_decodes += 1

        while processed < len(rows):
            while next_read < len(rows) and outstanding_reads < read_prefetch:
                read_future = read_executor.submit(read_image, next_read, rows[next_read])
                read_future.add_done_callback(ready_reads.put)
                outstanding_reads += 1
                next_read += 1

            while outstanding_decodes < decode_prefetch:
                try:
                    submit_decode(ready_reads.get_nowait())
                except Empty:
                    break

            if not outstanding_decodes and outstanding_reads:
                wait_started = time.monotonic()
                submit_decode(ready_reads.get())
                timing["read_wait_s"] += time.monotonic() - wait_started

            if outstanding_decodes:
                wait_started = time.monotonic()
                completed_decodes = [ready_decodes.get()]
                timing["decode_wait_s"] += time.monotonic() - wait_started
                while True:
                    try:
                        completed_decodes.append(ready_decodes.get_nowait())
                    except Empty:
                        break
                outstanding_decodes -= len(completed_decodes)
                for future in completed_decodes:
                    row_index, row, tensor, info, read_s, decode_s, resize_s = future.result()
                    decoded_ready[row_index] = (
                        row,
                        tensor,
                        info,
                        read_s,
                        decode_s,
                        resize_s,
                    )
                while processed in decoded_ready:
                    row, tensor, info, read_s, decode_s, resize_s = decoded_ready.pop(
                        processed
                    )
                    timing["read_s"] += read_s
                    timing["decode_s"] += decode_s
                    timing["resize_s"] += resize_s
                    key = (info.target_height, info.target_width)
                    if (
                        active_bucket_key is not None
                        and key != active_bucket_key
                        and active_bucket_key in buckets
                    ):
                        flush_bucket(active_bucket_key)
                    active_bucket_key = key
                    buckets[key].append((row, tensor, info))
                    if len(buckets[key]) >= batch_size:
                        flush_bucket(key)
                    elif len(buckets) > max_open_buckets:
                        fullest = max(
                            buckets, key=lambda bucket: len(buckets[bucket])
                        )
                        flush_bucket(fullest)
                    processed += 1
                    if processed % 1000 == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"prepared selected style images {processed}/{len(rows)} "
                            f"({processed / elapsed:.2f} images/s) "
                            f"read={timing['read_s'] / processed:.4f}s/img "
                            f"read_wait={timing['read_wait_s']:.2f}s "
                            f"decode={timing['decode_s'] / processed:.4f}s/img "
                            f"decode_wait={timing['decode_wait_s']:.2f}s "
                            f"resize={timing['resize_s'] / processed:.4f}s/img "
                            f"stack={timing['stack_s'] / processed:.4f}s/img "
                            f"pinned_alloc={timing['pinned_alloc_s']:.2f}s "
                            f"gpu={timing['gpu_s'] / max(newly_encoded, 1):.4f}s/img "
                            f"mean_batch={gpu_batch_images / max(gpu_batches, 1):.1f} "
                            f"write={timing['write_s'] / max(len(output_rows) - len(completed_rows), 1):.4f}s/img "
                            f"write_wait={timing['write_wait_s']:.2f}s "
                            f"raw_peak={raw_budget.peak_bytes / (1024**3):.2f}GiB",
                            flush=True,
                        )

    for key in list(buckets):
        flush_bucket(key)
    drain_gpu()
    gpu_executor.shutdown(wait=True)
    flush_shard()
    while pending_writes:
        collect_write(block=True)
    write_executor.shutdown(wait=True)
    output_rows.sort(key=lambda row: int(row["id"]))
    write_records(features_dir / "manifest.parquet", output_rows)
    total_bytes = sum(path.stat().st_size for path in features_dir.glob("*.safetensors"))
    summary = {
        "total": len(output_rows),
        "previously_encoded": len(completed_rows),
        "newly_encoded": newly_encoded,
        "spatial_layers": sorted(spatial_layers),
        "statistics_layers": sorted(statistics_layers),
        "summary_layers": sorted(summary_layers),
        "statistics": sorted(statistics),
        "storage_dtype": feature_cfg["storage_dtype"],
        "storage_bytes": total_bytes,
        "average_bytes_per_image": total_bytes / len(output_rows),
        "feature_signature": signature,
        "manifest": str((features_dir / "manifest.parquet").resolve()),
        "pipeline": {
            "batch_size": batch_size,
            "shard_rows": shard_rows,
            "reader_workers": reader_workers,
            "decoder_workers": decoder_workers,
            "read_prefetch": read_prefetch,
            "decode_prefetch": decode_prefetch,
            "raw_queue_limit_bytes": raw_budget.limit_bytes,
            "raw_queue_peak_bytes": raw_budget.peak_bytes,
            "writer_workers": writer_workers,
            "pending_writes": pending_writes_limit,
            "shape_sorted": True,
            "pinned_buffers": pinned_buffer_count,
            "pinned_allocations": pinned_allocations,
            "gpu_batches": gpu_batches,
            "mean_gpu_batch": gpu_batch_images / max(gpu_batches, 1),
        },
        "timing": timing,
    }
    write_json(features_dir / "summary.json", summary)
    return summary


def extract_cradio_features(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    cfg = config["cradio"]
    features_dir = destination / "cradio_features"
    manifest_dir = features_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    completed, shard_index = _existing_feature_ids(manifest_dir)
    rows = [row for row in _caption_rows(destination) if int(row["id"]) not in completed]
    if not rows:
        summary = {
            "total": len(completed),
            "newly_encoded": 0,
            "model_version": cfg["model_version"],
            "torchhub_ref": cfg["torchhub_ref"],
        }
        write_json(destination / "cradio_summary.json", summary)
        return summary

    model, device = _load_cradio(cfg, destination / "cradio_model_cache")
    adaptor_name = cfg["adaptor_name"]
    adaptor = model.adaptors[adaptor_name]
    batch_size = int(cfg["batch_size"])
    max_open_buckets = int(cfg.get("max_open_buckets", 16))
    storage_dtype = _storage_dtype(cfg["storage_dtype"])
    shard_rows = int(cfg["shard_rows"])
    amp_name = str(cfg["amp_dtype"])
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[amp_name]
    amp_enabled = device.startswith("cuda") and amp_name != "float32"
    buckets: dict[tuple[int, int], list[tuple[dict[str, Any], torch.Tensor, PreprocessInfo]]] = defaultdict(list)
    tensor_buffer: dict[str, torch.Tensor] = {}
    record_buffer: list[dict[str, Any]] = []
    newly_encoded = 0

    def flush_shard() -> None:
        nonlocal tensor_buffer, record_buffer, shard_index
        if not record_buffer:
            return
        feature_path = features_dir / f"part-{shard_index:05d}.safetensors"
        manifest_path = manifest_dir / f"part-{shard_index:05d}.parquet"
        temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
        save_file(tensor_buffer, temporary)
        temporary.replace(feature_path)
        for record in record_buffer:
            record["feature_shard"] = feature_path.name
        write_records(manifest_path, record_buffer)
        print(f"wrote C-RADIO shard {shard_index} ({len(record_buffer)} rows)", flush=True)
        shard_index += 1
        tensor_buffer = {}
        record_buffer = []

    def run_batch(items: list[tuple[dict[str, Any], torch.Tensor, PreprocessInfo]]) -> None:
        nonlocal newly_encoded
        images = torch.stack([item[1] for item in items]).to(device, non_blocking=True)
        captions = [item[0]["content_caption"] for item in items]
        tokenized = adaptor.tokenizer(captions).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(images)
                text_summary = adaptor.encode_text(tokenized, normalize=True)
            backbone_summary, backbone_spatial = _summary_features(output["backbone"])
            siglip_summary, _ = _summary_features(output[adaptor_name])
            visual, text, residual = content_subtracted_summary(
                siglip_summary,
                text_summary,
                scale=float(cfg["content_scale"]),
                normalize_residual=bool(cfg["normalize_residual"]),
            )

        for index, (row, _, info) in enumerate(items):
            image_id = int(row["id"])
            prefix = str(image_id)
            spatial = backbone_spatial[index].detach().to("cpu", dtype=storage_dtype).contiguous()
            summary = backbone_summary[index].detach().to("cpu", dtype=storage_dtype).contiguous()
            vis = visual[index].detach().to("cpu", dtype=storage_dtype).contiguous()
            txt = text[index].detach().to("cpu", dtype=storage_dtype).contiguous()
            resid = residual[index].detach().to("cpu", dtype=storage_dtype).contiguous()
            tensor_buffer[f"{prefix}.backbone_spatial"] = spatial
            tensor_buffer[f"{prefix}.backbone_summary"] = summary
            tensor_buffer[f"{prefix}.siglip_visual"] = vis
            tensor_buffer[f"{prefix}.siglip_text"] = txt
            tensor_buffer[f"{prefix}.siglip_residual"] = resid
            # RADIO's bundled SigLIP2-g tokenizer pads to a fixed length but
            # currently returns no attention mask. Preserve the true count if
            # a compatible tokenizer provides one, and always record slots.
            token_count = None
            if "attention_mask" in tokenized:
                token_count = int(tokenized["attention_mask"][index].sum().item())
            token_slots = int(tokenized["input_ids"][index].numel())
            record_buffer.append(
                {
                    "id": image_id,
                    "artist": row["artist"],
                    "style_id": row.get("style_id", row["artist"]),
                    "split": row.get("split", "train"),
                    "local_path": row["local_path"],
                    "caption_config_hash": row["caption_config_hash"],
                    "content_caption_sha256": hashlib.sha256(
                        row["content_caption"].encode("utf-8")
                    ).hexdigest(),
                    "text_token_count": token_count,
                    "text_token_slots": token_slots,
                    "original_height": info.original_height,
                    "original_width": info.original_width,
                    "resized_height": info.resized_height,
                    "resized_width": info.resized_width,
                    "target_height": info.target_height,
                    "target_width": info.target_width,
                    "crop_left": info.crop_left,
                    "crop_top": info.crop_top,
                    "spatial_tokens": int(spatial.shape[0]),
                    "spatial_dim": int(spatial.shape[-1]),
                    "residual_dim": int(resid.shape[-1]),
                    "storage_dtype": cfg["storage_dtype"],
                    "content_scale": float(cfg["content_scale"]),
                    "model_version": cfg["model_version"],
                    "torchhub_ref": cfg["torchhub_ref"],
                    "adaptor_name": adaptor_name,
                }
            )
            newly_encoded += 1
        if len(record_buffer) >= shard_rows:
            flush_shard()

    def flush_bucket(key: tuple[int, int]) -> None:
        items = buckets.pop(key)
        for offset in range(0, len(items), batch_size):
            run_batch(items[offset : offset + batch_size])

    for row in rows:
        with Image.open(row["local_path"]) as image:
            array, info = preprocess_cradio_image(image, cfg)
        tensor = torch.from_numpy(array)
        key = (info.target_height, info.target_width)
        buckets[key].append((row, tensor, info))
        if len(buckets[key]) >= batch_size:
            flush_bucket(key)
        elif len(buckets) > max_open_buckets:
            fullest = max(buckets, key=lambda bucket_key: len(buckets[bucket_key]))
            flush_bucket(fullest)

    for key in list(buckets):
        flush_bucket(key)
    flush_shard()
    summary = {
        "total": len(completed) + newly_encoded,
        "previously_encoded": len(completed),
        "newly_encoded": newly_encoded,
        "model_version": cfg["model_version"],
        "torchhub_ref": cfg["torchhub_ref"],
        "adaptor_name": adaptor_name,
        "content_scale": float(cfg["content_scale"]),
    }
    write_json(destination / "cradio_summary.json", summary)
    return summary


def build_style_feature_combiner(spatial_dim: int, residual_dim: int, output_dim: int):
    """Create the trainable fusion used before the future Style Resampler."""
    import torch
    from torch import nn

    class StyleFeatureCombiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial_norm = nn.LayerNorm(spatial_dim)
            self.residual_norm = nn.LayerNorm(residual_dim)
            self.spatial_projection = nn.Linear(spatial_dim, output_dim)
            self.residual_projection = nn.Linear(residual_dim, output_dim)
            self.type_embedding = nn.Parameter(torch.zeros(2, output_dim))
            nn.init.normal_(self.type_embedding, std=0.02)

        def forward(self, backbone_spatial, siglip_residual):
            spatial = self.spatial_projection(self.spatial_norm(backbone_spatial))
            residual = self.residual_projection(self.residual_norm(siglip_residual)).unsqueeze(1)
            residual = residual + self.type_embedding[0]
            spatial = spatial + self.type_embedding[1]
            return torch.cat((residual, spatial), dim=1)

    return StyleFeatureCombiner()
