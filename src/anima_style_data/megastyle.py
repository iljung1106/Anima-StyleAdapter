"""Download and production-cache contract for Tencent MegaStyle-1.4M."""

from __future__ import annotations

import bisect
import copy
import hashlib
import heapq
import io
import json
import pickle
import re
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from .io import read_records, write_json, write_records


def download_megastyle(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    cfg = dict(config["megastyle"])
    source = (destination / str(cfg["source_directory"])).resolve()
    source.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    resolved = snapshot_download(
        repo_id=str(cfg.get("repo_id", "tencent/MegaStyle-1.4M")),
        repo_type="dataset",
        revision=str(cfg.get("revision", "main")),
        local_dir=source,
        max_workers=int(cfg.get("download_workers", 16)),
    )
    files = [path for path in source.rglob("*") if path.is_file()]
    result = {
        "repo_id": str(cfg.get("repo_id", "tencent/MegaStyle-1.4M")),
        "requested_revision": str(cfg.get("revision", "main")),
        "resolved_directory": str(resolved),
        "files": len(files),
        "storage_bytes": sum(path.stat().st_size for path in files),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(source.parent / "download_summary.json", result)
    return result


def _source_parts(source: Path) -> list[Path]:
    parts = sorted(source.glob("train-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No MegaStyle parquet parts under {source}")
    return parts


def _parse_source_id(value: str) -> tuple[str, str]:
    if not value.startswith("s") or "_c" not in value:
        raise ValueError(f"Unexpected MegaStyle id: {value}")
    style, content = value[1:].split("_c", 1)
    if not style or not content:
        raise ValueError(f"Unexpected MegaStyle id: {value}")
    return style, content


def _stable_order(seed: int, value: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _select_styles(
    style_indices: dict[str, list[int]],
    source_ids: list[str],
    *,
    style_count: int,
    images_per_style: int,
    validation_styles: int,
    seed: int,
    seed_styles: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select whole styles while maximizing repeated-content connectivity."""

    content_frequency = Counter(_parse_source_id(value)[1] for value in source_ids)
    styles: list[dict[str, Any]] = []
    content_to_styles: dict[str, list[int]] = defaultdict(list)
    for description, global_indices in style_indices.items():
        if len(global_indices) != images_per_style:
            continue
        ids = [source_ids[int(index)] for index in global_indices]
        parsed = [_parse_source_id(value) for value in ids]
        style_ids = {style for style, _ in parsed}
        if len(style_ids) != 1:
            raise RuntimeError("MegaStyle style_indices crosses source style IDs")
        contents = tuple(content for _, content in parsed)
        style_index = len(styles)
        styles.append({
            "style_key": next(iter(style_ids)),
            "description": str(description),
            "global_indices": tuple(int(value) for value in global_indices),
            "source_ids": tuple(ids),
            "contents": contents,
        })
        for content in set(contents):
            content_to_styles[content].append(style_index)
    if len(styles) < style_count:
        raise RuntimeError(
            f"Only {len(styles)} complete MegaStyle styles, need {style_count}"
        )

    static_scores = [
        sum(min(content_frequency[content] - 1, 7) for content in style["contents"])
        for style in styles
    ]
    tie_breakers = [
        _stable_order(seed, str(style["style_key"])) for style in styles
    ]
    initial = sorted(
        range(len(styles)),
        key=lambda index: (-static_scores[index], tie_breakers[index]),
    )[: min(style_count, max(1, seed_styles))]
    selected = set(initial)
    represented: set[str] = set()
    overlap = [0] * len(styles)

    def unlock(style_index: int) -> None:
        for content in set(styles[style_index]["contents"]):
            if content in represented:
                continue
            represented.add(content)
            for neighbor in content_to_styles[content]:
                if neighbor not in selected:
                    overlap[neighbor] += 1

    for index in initial:
        unlock(index)
    heap = [
        (-overlap[index], -static_scores[index], tie_breakers[index], index)
        for index in range(len(styles))
        if index not in selected
    ]
    heapq.heapify(heap)
    while len(selected) < style_count:
        while True:
            negative_overlap, negative_static, tie, index = heapq.heappop(heap)
            if index in selected:
                continue
            current = (-overlap[index], -static_scores[index], tie_breakers[index], index)
            if (negative_overlap, negative_static, tie, index) != current:
                heapq.heappush(heap, current)
                continue
            break
        selected.add(index)
        before = set(represented)
        unlock(index)
        for content in represented - before:
            for neighbor in content_to_styles[content]:
                if neighbor not in selected:
                    heapq.heappush(heap, (
                        -overlap[neighbor], -static_scores[neighbor],
                        tie_breakers[neighbor], neighbor,
                    ))

    selected_indices = sorted(selected, key=lambda index: str(styles[index]["style_key"]))
    selected_content_counts = Counter(
        content
        for index in selected_indices
        for content in styles[index]["contents"]
    )
    # A validation style is only removed from train if every one of its content
    # IDs still has a train counterpart. This yields a style-disjoint but
    # content-controlled validation split.
    remaining = selected_content_counts.copy()
    validation: set[int] = set()
    validation_candidates = set(selected_indices)
    while len(validation) < validation_styles:
        index = min(
            validation_candidates,
            key=lambda value: (
                -sum(remaining[content] >= 2 for content in styles[value]["contents"]),
                tie_breakers[value],
            ),
        )
        validation_candidates.remove(index)
        validation.add(index)
        for content in styles[index]["contents"]:
            remaining[content] -= 1

    records = []
    for style_rank, index in enumerate(selected_indices):
        style = styles[index]
        split = "validation" if index in validation else "train"
        for image_index, (global_index, source_id, content_id) in enumerate(zip(
            style["global_indices"], style["source_ids"], style["contents"], strict=True
        )):
            records.append({
                "id": int(global_index) + 1,
                "global_index": int(global_index),
                "source_id": str(source_id),
                "artist": f"megastyle:{style['style_key']}",
                "style_id": f"megastyle:{style['style_key']}",
                "source_style_id": str(style["style_key"]),
                "content_id": str(content_id),
                "style_description": str(style["description"]),
                "split": split,
                "style_rank": style_rank,
                "artist_image_index": image_index,
            })
    selected_content_counts = Counter(row["content_id"] for row in records)
    summary = {
        "styles": len(selected_indices),
        "images": len(records),
        "train_styles": len(selected_indices) - len(validation),
        "validation_styles": len(validation),
        "unique_contents": len(selected_content_counts),
        "images_with_repeated_content": sum(
            count for count in selected_content_counts.values() if count > 1
        ),
        "repeated_content_image_fraction": sum(
            count for count in selected_content_counts.values() if count > 1
        ) / len(records),
        "validation_content_overlap_fraction": sum(
            remaining[row["content_id"]] > 0
            for row in records if row["split"] == "validation"
        ) / max(1, validation_styles * images_per_style),
    }
    return records, summary


def _content_tags(caption: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", caption)


def _materialize_image(
    image_bytes: bytes, path: Path
) -> tuple[int, int, str, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(image_bytes)
        temporary.replace(path)
    return width, height, hashlib.sha256(image_bytes).hexdigest(), len(image_bytes)


def _selection_signature(cfg: dict[str, Any], source: Path) -> str:
    payload = "|".join([
        str(source.resolve()),
        str(cfg.get("style_count", 5000)),
        str(cfg.get("images_per_style", 8)),
        str(cfg.get("validation_styles", 250)),
        str(cfg.get("seed", 20260819)),
        str(cfg.get("seed_styles", 256)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def prepare_megastyle_subset(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Select 5k complete styles and materialize only selected parquet rows."""

    cfg = dict(config["megastyle"])
    subset_cfg = dict(cfg["subset"])
    source = (destination / str(cfg["source_directory"])).resolve()
    output = (destination / str(subset_cfg["output_directory"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = _selection_signature(subset_cfg, source)
    summary_path = output / "prepare_summary.json"
    final_manifest = output / "final_manifest.parquet"
    caption_manifest = output / "captions" / "part-00000.parquet"
    if summary_path.exists() and final_manifest.exists() and caption_manifest.exists():
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = read_records(final_manifest)
        if (
            recorded.get("selection_signature") == signature
            and len(rows) == int(subset_cfg.get("style_count", 5000))
            * int(subset_cfg.get("images_per_style", 8))
            and all(Path(row["local_path"]).exists() for row in rows)
        ):
            return recorded

    parts = _source_parts(source)
    file_starts = []
    row_group_starts: list[list[int]] = []
    source_ids: list[str] = []
    total = 0
    for part in parts:
        parquet = pq.ParquetFile(part)
        file_starts.append(total)
        starts = []
        local = 0
        for index in range(parquet.metadata.num_row_groups):
            starts.append(local)
            local += parquet.metadata.row_group(index).num_rows
        row_group_starts.append(starts)
        values = parquet.read(columns=["id"])["id"].to_pylist()
        if len(values) != local:
            raise RuntimeError(f"Parquet metadata mismatch for {part}")
        source_ids.extend(str(value) for value in values)
        total += local

    with (source / "style_indices.pkl").open("rb") as handle:
        style_indices = pickle.load(handle)
    selected, selection_summary = _select_styles(
        style_indices,
        source_ids,
        style_count=int(subset_cfg.get("style_count", 5000)),
        images_per_style=int(subset_cfg.get("images_per_style", 8)),
        validation_styles=int(subset_cfg.get("validation_styles", 250)),
        seed=int(subset_cfg.get("seed", 20260819)),
        seed_styles=int(subset_cfg.get("seed_styles", 256)),
    )
    for row in selected:
        file_index = bisect.bisect_right(file_starts, int(row["global_index"])) - 1
        local_index = int(row["global_index"]) - file_starts[file_index]
        starts = row_group_starts[file_index]
        row_group = bisect.bisect_right(starts, local_index) - 1
        row.update({
            "source_file": str(parts[file_index]),
            "source_file_index": file_index,
            "source_row_group": row_group,
            "source_row_offset": local_index - starts[row_group],
            "selection_signature": signature,
        })
    selection_manifest = output / "selection_manifest.parquet"
    write_records(selection_manifest, sorted(selected, key=lambda row: int(row["id"])))

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(int(row["source_file_index"]), int(row["source_row_group"]))].append(row)
    workers = max(1, int(subset_cfg.get("image_workers", 32)))
    pending_limit = max(workers, int(subset_cfg.get("pending_writes", 256)))
    pending: deque[tuple[Future, dict[str, Any], str, str]] = deque()
    rows = []
    storage_bytes = 0
    started = time.perf_counter()

    def consume(item: tuple[Future, dict[str, Any], str, str]) -> None:
        nonlocal storage_bytes
        future, selected_row, content, image_name = item
        width, height, sha256, byte_count = future.result()
        storage_bytes += byte_count
        rows.append({
            **selected_row,
            "record_id": selected_row["source_id"],
            "width": width,
            "height": height,
            "decoded_width": width,
            "decoded_height": height,
            "local_path": image_name,
            "sha256": sha256,
            "download_status": "materialized",
            "content_caption": content,
        })

    with ThreadPoolExecutor(max_workers=workers) as writer:
        for (file_index, row_group), selected_rows in sorted(grouped.items()):
            table = pq.ParquetFile(parts[file_index]).read_row_group(
                row_group, columns=["id", "image", "content", "style"]
            )
            values = table.to_pylist()
            for selected_row in selected_rows:
                value = values[int(selected_row["source_row_offset"])]
                if str(value["id"]) != str(selected_row["source_id"]):
                    raise RuntimeError("MegaStyle parquet row locator mismatch")
                image = value["image"]
                image_bytes = image["bytes"]
                suffix = Path(str(image.get("path") or "image.jpg")).suffix or ".jpg"
                image_path = (
                    output / "images" / f"{int(selected_row['id']) % 256:03d}"
                    / f"{selected_row['source_id']}{suffix.lower()}"
                ).resolve()
                pending.append((
                    writer.submit(_materialize_image, image_bytes, image_path),
                    selected_row,
                    str(value["content"]),
                    str(image_path),
                ))
                if len(pending) >= pending_limit:
                    consume(pending.popleft())
            while pending and pending[0][0].done():
                consume(pending.popleft())
        while pending:
            consume(pending.popleft())

    rows.sort(key=lambda row: int(row["id"]))
    if len(rows) != int(subset_cfg.get("style_count", 5000)) * int(
        subset_cfg.get("images_per_style", 8)
    ):
        raise RuntimeError(f"Materialized {len(rows)} unexpected MegaStyle rows")
    write_records(final_manifest, rows)
    caption_rows = []
    for row in rows:
        tags = _content_tags(str(row["content_caption"]))
        caption_rows.append({
            "id": int(row["id"]),
            "artist": str(row["artist"]),
            "style_id": str(row["style_id"]),
            "split": str(row["split"]),
            "local_path": str(row["local_path"]),
            "rating_source": "",
            "rating_anima": "",
            "count_tags": [],
            "character_tags": [],
            "general_tags": tags,
            "anima_caption": str(row["content_caption"]),
            "content_caption": str(row["content_caption"]),
            "tagger_revision": "none:megastyle-content-caption",
            "tagger_threshold": 0.0,
            "caption_version": "megastyle-content-v1",
            "caption_config_hash": signature,
            "tag_source_shard": Path(row["source_file"]).name,
        })
    write_records(caption_manifest, caption_rows)
    result = {
        **selection_summary,
        "selection_signature": signature,
        "source_rows": total,
        "source_parts": len(parts),
        "storage_bytes": storage_bytes,
        "output_directory": str(output),
        "selection_manifest": str(selection_manifest),
        "final_manifest": str(final_manifest),
        "caption_manifest": str(caption_manifest),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, result)
    return result


def _effective_cache_config(
    config: dict[str, Any], destination: Path, output: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cache_cfg = dict(config["megastyle"]["subset"])
    effective["style_features"].update({
        "output_directory": "style_features_l18_l24_siglip_l24",
        "manifest_path": str(output / "final_manifest.parquet"),
        "model_cache_directory": str((destination / "cradio_model_cache").resolve()),
    })
    effective["anima_cache"]["models"]["cache_directory"] = str(
        (destination / "anima_model_cache").resolve()
    )
    effective["anima_cache"]["latents"]["output_directory"] = (
        "anima_latent_cache_qwen_2d"
    )
    effective["anima_cache"]["text"].update(
        dict(config["global_query_multimode_style_tokenizer"]["text_cache"])
    )
    effective["anima_cache"]["text"]["output_directory"] = (
        "anima_text_cache_post_llm_multimode_v1"
    )
    effective["dual_query_resampler"].update({
        "feature_directory": "style_features_l18_l24_siglip_l24",
        "latent_directory": "anima_latent_cache_qwen_2d",
    })
    effective["dual_query_style_tokenizer"]["resampler_checkpoint"] = str(
        (destination / str(config["dual_query_style_tokenizer"]["resampler_checkpoint"])).resolve()
    )
    effective["dual_query_style_tokenizer"]["cache"].update({
        "output_directory": "dual_query_reference_tokens_v8_step10000",
        "batch_size": int(cache_cfg.get("resampler_batch_size", 64)),
    })
    return effective


def cache_megastyle_subset(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Run the existing production cache stages over the selected 40k rows."""

    from .anima_cache import cache_anima_text_conditions, cache_anima_vae_latents
    from .cradio import extract_selected_style_features
    from .dual_query_style_tokenizer import cache_dual_query_style_tokens

    preparation = prepare_megastyle_subset(config, destination)
    output = Path(preparation["output_directory"])
    effective = _effective_cache_config(config, destination, output)
    started = time.perf_counter()
    features = extract_selected_style_features(effective, output)
    latents = cache_anima_vae_latents(effective, output)
    tokens = cache_dual_query_style_tokens(effective, output)
    text = cache_anima_text_conditions(effective, output)
    result = {
        "images": int(preparation["images"]),
        "styles": int(preparation["styles"]),
        "prepare": preparation,
        "style_features": features,
        "vae_latents": latents,
        "resampler_tokens": tokens,
        "multimode_text": text,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "cache_summary.json", result)
    return result
