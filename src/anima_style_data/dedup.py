from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .io import read_records, write_json, write_records


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float32)
    factor = math.pi / (2 * size)
    scale0 = math.sqrt(1 / size)
    scale = math.sqrt(2 / size)
    for row in range(size):
        row_scale = scale0 if row == 0 else scale
        for column in range(size):
            matrix[row, column] = row_scale * math.cos((2 * column + 1) * row * factor)
    return matrix


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    image = ImageOps.exif_transpose(image).convert("L")
    phash_image = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    transform = _dct_matrix(32)
    low = (transform @ phash_image @ transform.T)[:8, :8]
    median = np.median(low.reshape(-1)[1:])
    phash = _bits_to_hex(low > median)

    dhash_image = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = _bits_to_hex(dhash_image[:, 1:] > dhash_image[:, :-1])
    return phash, dhash


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_near_duplicate(
    row: dict[str, Any], kept: dict[str, Any], cfg: dict[str, Any]
) -> tuple[bool, int, int]:
    p_distance = hamming(row["phash"], kept["phash"])
    d_distance = hamming(row["dhash"], kept["dhash"])
    p_close = p_distance <= int(cfg["phash_distance"])
    d_close = d_distance <= int(cfg["dhash_distance"])
    duplicate = p_close and d_close if cfg.get("require_both_hashes", True) else p_close or d_close
    return duplicate, p_distance, d_distance


def _enrich_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = Path(row["local_path"])
    try:
        with Image.open(path) as image:
            image.load()
            phash, dhash = perceptual_hashes(image)
            width, height = image.size
        return (
            {
                **row,
                "decoded_width": width,
                "decoded_height": height,
                "phash": phash,
                "dhash": dhash,
                # The Anima500k extractor already verifies this digest against
                # source JSON. Legacy manifests fall back to hashing the file.
                "sha256": row.get("sha256") or _file_sha256(path),
            },
            None,
        )
    except Exception as exc:
        return None, {**row, "dedup_status": "invalid", "dedup_reason": str(exc)}


def deduplicate(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_records(destination / "download_manifest.parquet")
        if row["download_status"] != "failed"
    ]
    enriched: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    workers = int(config["dedup"].get("workers", 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_enrich_row, rows, chunksize=int(config["dedup"].get("chunksize", 32)))
        for index, (valid, failed) in enumerate(results, start=1):
            if valid is not None:
                enriched.append(valid)
            if failed is not None:
                invalid.append(failed)
            if index % 1000 == 0 or index == len(rows):
                print(f"hashed {index}/{len(rows)} images with {workers} workers", flush=True)

    final_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = list(invalid)
    target = int(config["selection"]["images_per_artist"])
    shortfalls: dict[str, int] = {}
    cfg = config["dedup"]

    # Exact encoded-image duplicates are removed globally. Prefer train over
    # held-out splits so leakage is eliminated without unnecessarily shrinking
    # the training corpus.
    split_priority = {"train": 0, "validation": 1, "test": 2}
    enriched.sort(
        key=lambda row: (
            split_priority.get(row.get("split", "train"), 3),
            row.get("style_id", row["artist"]),
            row["selection_rank"],
        )
    )
    globally_unique: list[dict[str, Any]] = []
    seen_sha256: dict[str, dict[str, Any]] = {}
    for row in enriched:
        duplicate_of = seen_sha256.get(row["sha256"])
        if duplicate_of is not None:
            audit_rows.append(
                {
                    **row,
                    "dedup_status": "removed",
                    "dedup_reason": "exact_sha256_global",
                    "kept_id": duplicate_of["id"],
                    "phash_distance": 0,
                    "dhash_distance": 0,
                }
            )
            continue
        seen_sha256[row["sha256"]] = row
        globally_unique.append(row)

    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in globally_unique:
        by_artist[row.get("style_id", row["artist"])].append(row)

    for artist, artist_rows in sorted(by_artist.items()):
        artist_rows.sort(key=lambda row: (row["selection_rank"], -int(row["id"])))
        kept: list[dict[str, Any]] = []
        seen_md5: dict[str, dict[str, Any]] = {}
        for row in artist_rows:
            duplicate_of = None
            reason = None
            p_distance = None
            d_distance = None
            if row["md5"] in seen_md5:
                duplicate_of = seen_md5[row["md5"]]
                reason = "exact_md5"
            else:
                for existing in kept:
                    duplicate, p_distance, d_distance = _is_near_duplicate(row, existing, cfg)
                    if duplicate:
                        duplicate_of = existing
                        reason = "near_duplicate"
                        break
            if duplicate_of is not None:
                audit_rows.append(
                    {
                        **row,
                        "dedup_status": "removed",
                        "dedup_reason": reason,
                        "kept_id": duplicate_of["id"],
                        "phash_distance": p_distance,
                        "dhash_distance": d_distance,
                    }
                )
                continue
            seen_md5[row["md5"]] = row
            kept.append(row)

        selected = kept[:target]
        for final_rank, row in enumerate(selected, start=1):
            final_rows.append({**row, "final_rank": final_rank})
            audit_rows.append(
                {
                    **row,
                    "dedup_status": "kept",
                    "dedup_reason": None,
                    "kept_id": row["id"],
                    "phash_distance": None,
                    "dhash_distance": None,
                }
            )
        for row in kept[target:]:
            audit_rows.append(
                {
                    **row,
                    "dedup_status": "reserve",
                    "dedup_reason": "artist_target_reached",
                    "kept_id": row["id"],
                    "phash_distance": None,
                    "dhash_distance": None,
                }
            )
        if len(selected) < target:
            shortfalls[artist] = target - len(selected)

    if not final_rows:
        raise RuntimeError("No images survived decoding and duplicate removal")
    final_rows.sort(
        key=lambda row: (
            split_priority.get(row.get("split", "train"), 3),
            row.get("style_id", row["artist"]),
            row["final_rank"],
        )
    )
    write_records(destination / "dedup_manifest.parquet", audit_rows)
    write_records(destination / "final_manifest.parquet", final_rows)
    summary = {
        "decoded": len(enriched),
        "invalid": len(invalid),
        "exact_duplicates_removed": len(enriched) - len(globally_unique),
        "final_images": len(final_rows),
        "artists_with_shortfall": len(shortfalls),
        "shortfalls": shortfalls,
    }
    write_json(destination / "dedup_summary.json", summary)
    return summary
