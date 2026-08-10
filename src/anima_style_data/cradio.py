from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
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
    storage_dtype: Any,
) -> list[dict[str, Any]]:
    """Collect the deliberately small, production-selected C-RADIO feature set."""
    outputs: list[dict[str, Any]] = []
    batch_size = int(intermediate[0].features.shape[0])
    for item_index in range(batch_size):
        tensors: dict[str, Any] = {}
        for layer, output in zip(layers, intermediate):
            spatial = output.features[item_index]
            if layer in spatial_layers:
                tensors[f"layer_{layer:02d}_spatial"] = spatial.detach().to(
                    "cpu", dtype=storage_dtype
                )
            if layer in statistics_layers:
                stable = spatial.detach().float()
                tensors[f"layer_{layer:02d}_mean"] = stable.mean(dim=0).to(
                    "cpu", dtype=storage_dtype
                )
                tensors[f"layer_{layer:02d}_std"] = stable.std(
                    dim=0, correction=0
                ).to("cpu", dtype=storage_dtype)
        outputs.append(tensors)
    return outputs


def extract_selected_style_features(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Cache L20/L24 spatial tokens and compact low-level style statistics."""
    import torch
    from safetensors.torch import save_file

    feature_cfg = config["style_features"]
    radio_cfg = {**config["cradio"], **feature_cfg.get("preprocess", {})}
    spatial_layers = {int(value) for value in feature_cfg["spatial_layers"]}
    statistics_layers = {
        int(value) for value in feature_cfg.get("statistics_layers", [])
    }
    statistics = set(feature_cfg.get("statistics", ["mean", "std"]))
    if statistics != {"mean", "std"}:
        raise ValueError("style_features.statistics must contain exactly mean and std")
    layers = sorted(spatial_layers | statistics_layers)
    signature = _selected_style_feature_signature(radio_cfg, feature_cfg)

    features_dir = destination / "style_features"
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
    all_rows = read_records(destination / "final_manifest.parquet")
    rows = [row for row in all_rows if int(row["id"]) not in completed_ids]
    shard_index = max(shard_numbers) + 1 if shard_numbers else 0

    if not rows:
        total_bytes = sum(path.stat().st_size for path in features_dir.glob("*.safetensors"))
        summary = {
            "total": len(completed_rows),
            "newly_encoded": 0,
            "spatial_layers": sorted(spatial_layers),
            "statistics_layers": sorted(statistics_layers),
            "storage_bytes": total_bytes,
            "feature_signature": signature,
        }
        write_json(features_dir / "summary.json", summary)
        return summary

    model, device = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    if min(layers) < 0 or max(layers) >= len(model.model.blocks):
        raise ValueError(f"Feature layers must be in [0, {len(model.model.blocks) - 1}]")

    batch_size = int(feature_cfg.get("batch_size", radio_cfg["batch_size"]))
    shard_rows = int(feature_cfg.get("shard_rows", 128))
    max_open_buckets = int(feature_cfg.get("max_open_buckets", 64))
    preprocess_workers = int(feature_cfg.get("preprocess_workers", 8))
    prefetch_images = max(
        preprocess_workers, int(feature_cfg.get("prefetch_images", 64))
    )
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

    def flush_shard() -> None:
        nonlocal tensor_buffer, record_buffer, shard_index
        if not record_buffer:
            return
        feature_path = features_dir / f"part-{shard_index:05d}.safetensors"
        temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
        save_file(
            {key: value.contiguous() for key, value in tensor_buffer.items()}, temporary
        )
        temporary.replace(feature_path)
        for record in record_buffer:
            record["feature_shard"] = feature_path.name
        write_records(manifest_dir / f"part-{shard_index:05d}.parquet", record_buffer)
        output_rows.extend(record_buffer)
        print(
            f"wrote selected style feature shard {shard_index} "
            f"({len(record_buffer)} images; total_new={newly_encoded})",
            flush=True,
        )
        tensor_buffer = {}
        record_buffer = []
        shard_index += 1

    def run_batch(items: list[tuple[dict[str, Any], torch.Tensor, PreprocessInfo]]) -> None:
        nonlocal newly_encoded
        images = torch.stack([item[1] for item in items]).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
        ):
            _, intermediate = model.forward_intermediates(
                images,
                indices=layers,
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NLC",
                aggregation="sparse",
            )
        selected = _selected_style_tensors(
            intermediate, layers, spatial_layers, statistics_layers, storage_dtype
        )
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

    def flush_bucket(key: tuple[int, int]) -> None:
        items = buckets.pop(key)
        for offset in range(0, len(items), batch_size):
            run_batch(items[offset : offset + batch_size])

    def load_image(row: dict[str, Any]):
        with Image.open(row["local_path"]) as image:
            array, info = preprocess_cradio_image(image, radio_cfg)
        return row, torch.from_numpy(array), info

    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        pending: dict[int, Future[Any]] = {}
        next_submit = 0
        for index in range(len(rows)):
            while next_submit < len(rows) and next_submit < index + prefetch_images:
                pending[next_submit] = executor.submit(load_image, rows[next_submit])
                next_submit += 1
            row, tensor, info = pending.pop(index).result()
            key = (info.target_height, info.target_width)
            buckets[key].append((row, tensor, info))
            if len(buckets[key]) >= batch_size:
                flush_bucket(key)
            elif len(buckets) > max_open_buckets:
                fullest = max(buckets, key=lambda bucket: len(buckets[bucket]))
                flush_bucket(fullest)
            if (index + 1) % 1000 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"prepared selected style images {index + 1}/{len(rows)} "
                    f"({(index + 1) / elapsed:.2f} images/s)",
                    flush=True,
                )

    for key in list(buckets):
        flush_bucket(key)
    flush_shard()
    output_rows.sort(key=lambda row: int(row["id"]))
    write_records(features_dir / "manifest.parquet", output_rows)
    total_bytes = sum(path.stat().st_size for path in features_dir.glob("*.safetensors"))
    summary = {
        "total": len(output_rows),
        "previously_encoded": len(completed_rows),
        "newly_encoded": newly_encoded,
        "spatial_layers": sorted(spatial_layers),
        "statistics_layers": sorted(statistics_layers),
        "statistics": sorted(statistics),
        "storage_dtype": feature_cfg["storage_dtype"],
        "storage_bytes": total_bytes,
        "average_bytes_per_image": total_bytes / len(output_rows),
        "feature_signature": signature,
        "manifest": str((features_dir / "manifest.parquet").resolve()),
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
