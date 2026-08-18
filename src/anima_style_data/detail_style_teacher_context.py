"""Resumable post-LLM native artist-context cache for same-Q supervision."""

from __future__ import annotations

import gc
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .anima_cache import (
    _import_sd_scripts,
    _load_llm_adapter,
    _resolve_model_files,
    _verify_sd_scripts_revision,
)
from .io import read_records, write_json, write_records


def _normalized_artist(value: str) -> str:
    # Text is passed directly to Qwen/T5; parentheses are literal here and are
    # not interpreted by a ComfyUI prompt-weight parser.
    return " ".join(value.replace("_", " ").split())


def _content_prompts(destination: Path, cfg: dict[str, Any]) -> list[str]:
    rows = read_records(destination / str(cfg["probe_manifest"]))
    controls = {
        int(row["content_index"]): str(row["content_prompt"])
        for row in rows
        if str(row.get("kind")) == "content_control"
        and int(row.get("seed_index", 0)) == 0
    }
    count = int(cfg.get("content_count", 4))
    prompts = [controls[index] for index in sorted(controls)[:count]]
    if len(prompts) != count:
        raise RuntimeError(f"Expected {count} native-teacher content controls")
    return prompts


def cache_detail_style_teacher_contexts(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Cache content+@artist post-LLM contexts without retaining Qwen in training."""

    cfg = dict(config["detail_preserving_style_cross_attention"])
    teacher_cfg = dict(cfg["teacher"])
    bank_key = str(teacher_cfg.get("bank_config_key", "dual_domain_native_teacher"))
    bank_cfg = dict(config[bank_key])
    bank_root = destination / str(bank_cfg["output_directory"])
    bank_summary = json.loads((bank_root / "summary.json").read_text(encoding="utf-8"))
    signature = bank_summary["signature"]
    artists = [str(value) for value in signature["artists"]]
    style_ids = [str(value) for value in signature["style_ids"]]
    splits = [str(value) for value in signature["splits"]]
    prompts = _content_prompts(destination, bank_cfg)

    output = destination / str(teacher_cfg["context_cache"])
    output.mkdir(parents=True, exist_ok=True)
    shard_artists = int(teacher_cfg.get("context_shard_artists", 64))
    expected = {
        "version": "detail-style-native-context-v1",
        "bank_signature": signature,
        "contents": prompts,
        "shard_artists": shard_artists,
        "storage_dtype": "float16",
    }
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.parquet"
    if summary_path.exists() and manifest_path.exists():
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        if current.get("signature") == expected and int(current["artists"]) == len(artists):
            return {**current, "reused": True}

    cache_cfg = config["anima_cache"]
    text_cfg = cache_cfg["text"]
    models = _resolve_model_files(cache_cfg["models"], destination)
    sd_root = str(cache_cfg["sd_scripts_path"])
    _verify_sd_scripts_revision(sd_root, str(cache_cfg["sd_scripts_revision"]))
    anima_models, anima_utils, _ = _import_sd_scripts(sd_root)
    device = str(teacher_cfg.get("context_cache_device", "cuda"))
    dtype = torch.bfloat16
    qwen, qwen_tokenizer = anima_utils.load_qwen3_text_encoder(
        models["qwen3"], dtype=dtype, device=device
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer()
    adapter = _load_llm_adapter(anima_models, models["dit"], device, dtype)
    max_qwen = int(text_cfg.get("qwen_max_length", 512))
    max_t5 = int(text_cfg.get("t5_max_length", 512))
    batch_size = int(teacher_cfg.get("context_batch_size", 64))
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for shard_index, start in enumerate(range(0, len(artists), shard_artists)):
            shard_path = output / f"part-{shard_index:05d}.safetensors"
            stop = min(len(artists), start + shard_artists)
            expected_shape = (stop - start, len(prompts), max_t5, 1024)
            if shard_path.exists():
                cached = load_file(shard_path, device="cpu")["contexts"]
                if tuple(cached.shape) != expected_shape:
                    raise RuntimeError(f"Invalid existing teacher context shard {shard_path}")
            else:
                shard_prompts = [
                    f"{content}, @{_normalized_artist(artist)}"
                    for artist in artists[start:stop]
                    for content in prompts
                ]
                contexts = torch.zeros(expected_shape, dtype=torch.float16)
                flat = contexts.view(-1, max_t5, 1024)
                for offset in range(0, len(shard_prompts), batch_size):
                    values = shard_prompts[offset : offset + batch_size]
                    source = qwen_tokenizer(
                        values, return_tensors="pt", truncation=True, padding=True,
                        max_length=max_qwen,
                    )
                    target = t5_tokenizer(
                        values, return_tensors="pt", truncation=True, padding=True,
                        max_length=max_t5,
                    )
                    source_ids = source["input_ids"].to(device)
                    source_mask = source["attention_mask"].to(device)
                    target_ids = target["input_ids"].to(device)
                    target_mask = target["attention_mask"].to(device)
                    with torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=dtype,
                        enabled=torch.device(device).type == "cuda",
                    ):
                        hidden = qwen(
                            input_ids=source_ids, attention_mask=source_mask
                        ).last_hidden_state
                        hidden[~source_mask.bool()] = 0
                        condition = adapter(
                            hidden,
                            target_ids,
                            target_attention_mask=target_mask,
                            source_attention_mask=source_mask,
                        )
                        condition[~target_mask.bool()] = 0
                    flat[offset : offset + len(values), : condition.shape[1]].copy_(
                        condition.to(device="cpu", dtype=torch.float16)
                    )
                temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
                save_file({"contexts": contexts.contiguous()}, temporary)
                temporary.replace(shard_path)
            for row, artist_index in enumerate(range(start, stop)):
                records.append({
                    "artist_index": artist_index,
                    "artist": artists[artist_index],
                    "style_id": style_ids[artist_index],
                    "split": splits[artist_index],
                    "context_shard": shard_path.name,
                    "context_row": row,
                })
            print(
                f"detail teacher contexts {stop}/{len(artists)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    write_records(manifest_path, records)
    total_bytes = sum(path.stat().st_size for path in output.glob("part-*.safetensors"))
    result = {
        "signature": expected,
        "artists": len(artists),
        "contents": len(prompts),
        "shards": len(list(output.glob("part-*.safetensors"))),
        "storage_bytes": total_bytes,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, result)
    del adapter, qwen
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {**result, "reused": False}


class NativeArtistContextCache:
    """Small LRU over artist-major post-LLM context shards."""

    def __init__(self, root: Path, *, capacity: int = 8) -> None:
        self.root = Path(root)
        self.capacity = max(1, int(capacity))
        rows = read_records(self.root / "manifest.parquet")
        self.by_style = {str(row["style_id"]): row for row in rows}
        self._shards: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _get_shard(self, name: str) -> torch.Tensor:
        value = self._shards.pop(name, None)
        if value is None:
            value = load_file(self.root / name, device="cpu")["contexts"]
            while len(self._shards) >= self.capacity:
                self._shards.popitem(last=False)
        self._shards[name] = value
        return value

    def get(self, style_ids: list[str], content_index: int) -> torch.Tensor:
        values = []
        for style_id in style_ids:
            row = self.by_style.get(str(style_id))
            if row is None:
                raise RuntimeError(f"No native artist context for style {style_id}")
            values.append(
                self._get_shard(str(row["context_shard"]))[
                    int(row["context_row"]), int(content_index)
                ]
            )
        result = torch.stack(values)
        return result.pin_memory() if torch.cuda.is_available() else result
