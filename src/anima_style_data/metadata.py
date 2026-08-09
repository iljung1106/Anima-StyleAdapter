from __future__ import annotations

import fnmatch
import glob
import hashlib
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from .io import write_json, write_records


SCAN_COLUMNS = [
    "id",
    "created_at",
    "source",
    "md5",
    "image_width",
    "image_height",
    "file_ext",
    "file_size",
    "tag_count_artist",
    "is_pending",
    "is_flagged",
    "is_deleted",
    "is_banned",
    "pixiv_id",
    "parent_id",
    "tag_string_artist",
    "tag_string_meta",
    "file_url",
    "large_file_url",
    "original_url",
]


@dataclass
class ArtistStat:
    count: int = 0
    earliest_ordinal: int = 10**9
    latest_ordinal: int = 0

    def add(self, ordinal: int) -> None:
        self.count += 1
        self.earliest_ordinal = min(self.earliest_ordinal, ordinal)
        self.latest_ordinal = max(self.latest_ordinal, ordinal)


def _stable_uniform(seed: int, namespace: str) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{namespace}".encode("utf-8"), digest_size=8
    ).digest()
    integer = int.from_bytes(digest, "big")
    return (integer + 1) / (2**64 + 1)


def _weighted_key(seed: int, namespace: str, weight: float) -> float:
    return math.log(_stable_uniform(seed, namespace)) / max(weight, 1e-12)


def _date_ordinal(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    try:
        return date.fromisoformat(text[:10]).toordinal()
    except (ValueError, TypeError):
        return None


def _resolve_files(metadata_cfg: dict[str, Any]) -> list[tuple[str, bool]]:
    local_glob = metadata_cfg.get("local_glob")
    include = set(metadata_cfg.get("include_files") or [])
    if local_glob:
        files = [Path(name) for name in sorted(glob.glob(local_glob))]
        if include:
            files = [path for path in files if path.name in include]
        if not files:
            raise FileNotFoundError(f"No metadata files matched {local_glob!r}")
        return [(str(path.resolve()), False) for path in files]

    repo_id = metadata_cfg["repo_id"]
    revision = metadata_cfg.get("revision") or "main"
    pattern = metadata_cfg.get("file_pattern", "*.parquet")
    names = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    names = sorted(name for name in names if fnmatch.fnmatch(name, pattern))
    if include:
        names = [name for name in names if name in include]
    if not names:
        raise FileNotFoundError(f"No Parquet files found in dataset {repo_id}")
    prefix = f"datasets/{repo_id}@{revision}"
    return [(f"{prefix}/{name}", True) for name in names]


def _iter_batches(metadata_cfg: dict[str, Any]) -> Iterator[pa.RecordBatch]:
    files = _resolve_files(metadata_cfg)
    batch_rows = int(metadata_cfg.get("batch_rows", 65536))
    max_batches = metadata_cfg.get("max_batches")
    yielded = 0
    hf_fs = HfFileSystem() if any(remote for _, remote in files) else None

    for path, remote in files:
        handle = hf_fs.open(path, "rb") if remote else open(path, "rb")
        try:
            parquet = pq.ParquetFile(handle)
            available = set(parquet.schema_arrow.names)
            columns = [name for name in SCAN_COLUMNS if name in available]
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
                yield batch
                yielded += 1
                if max_batches is not None and yielded >= int(max_batches):
                    return
        finally:
            handle.close()


def _valid_row(row: dict[str, Any], selection_cfg: dict[str, Any]) -> bool:
    if not row.get("id") or not row.get("md5"):
        return False
    if not (row.get("file_url") or row.get("large_file_url") or row.get("original_url")):
        return False
    artist = row.get("tag_string_artist")
    if not artist:
        return False
    if selection_cfg.get("require_single_artist", True):
        if row.get("tag_count_artist") != 1 or " " in artist.strip():
            return False
    extension = str(row.get("file_ext") or "").lower()
    allowed = {str(ext).lower() for ext in selection_cfg.get("allowed_extensions", [])}
    if allowed and extension not in allowed:
        return False
    for field, option in (
        ("is_pending", "exclude_pending"),
        ("is_flagged", "exclude_flagged"),
        ("is_deleted", "exclude_deleted"),
        ("is_banned", "exclude_banned"),
    ):
        if selection_cfg.get(option, True) and bool(row.get(field)):
            return False
    return _date_ordinal(row.get("created_at")) is not None


def _iter_valid_rows(
    metadata_cfg: dict[str, Any], selection_cfg: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    for batch_index, batch in enumerate(_iter_batches(metadata_cfg), start=1):
        for row in batch.to_pylist():
            if _valid_row(row, selection_cfg):
                row["created_ordinal"] = _date_ordinal(row["created_at"])
                yield row
        print(f"metadata batch {batch_index} processed", flush=True)


def _choose_artist_pool(
    stats: dict[str, ArtistStat], selection_cfg: dict[str, Any]
) -> list[tuple[str, float]]:
    seed = int(selection_cfg["seed"])
    target_images = math.ceil(
        selection_cfg["images_per_artist"] * selection_cfg["candidate_multiplier"]
    )
    eligible = [(artist, stat) for artist, stat in stats.items() if stat.count >= target_images]
    requested = int(selection_cfg["artist_count"])
    pool_size = min(
        len(eligible), math.ceil(requested * selection_cfg["artist_pool_multiplier"])
    )
    if len(eligible) < requested:
        raise RuntimeError(
            f"Only {len(eligible)} artists have at least {target_images} candidates; "
            f"{requested} requested"
        )
    snapshot = max(stat.latest_ordinal for _, stat in eligible)
    tau = float(selection_cfg["artist_recency_tau_days"])
    scored = []
    for artist, stat in eligible:
        weight = math.exp(-(snapshot - stat.latest_ordinal) / tau)
        key = _weighted_key(seed, f"artist:{artist}", weight)
        scored.append((key, artist))
    scored.sort(reverse=True)
    return [(artist, key) for key, artist in scored[:pool_size]]


def _collect_pool_rows(
    pool: set[str],
    stats: dict[str, ArtistStat],
    metadata_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    cap = int(selection_cfg["per_artist_scan_cap"])
    tau = float(selection_cfg["image_recency_tau_days"])
    seed = int(selection_cfg["seed"])
    heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    for row in _iter_valid_rows(metadata_cfg, selection_cfg):
        artist = row["tag_string_artist"].strip()
        if artist not in pool:
            continue
        weight = math.exp(-(stats[artist].latest_ordinal - row["created_ordinal"]) / tau)
        key = _weighted_key(seed, f"scan:{artist}:{row['id']}", weight)
        item = (key, int(row["id"]), row)
        heap = heaps[artist]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    return {artist: [item[2] for item in heap] for artist, heap in heaps.items()}


def _select_artist_candidates(
    artist: str,
    rows: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], int] | None:
    target = math.ceil(
        selection_cfg["images_per_artist"] * selection_cfg["candidate_multiplier"]
    )
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["created_ordinal"], item["id"]), reverse=True):
        unique.setdefault(row["md5"], row)
    rows = list(unique.values())
    if len(rows) < target:
        return None

    latest = max(row["created_ordinal"] for row in rows)
    selected_window = None
    window_rows: list[dict[str, Any]] = []
    for window_days in selection_cfg["date_windows_days"]:
        candidates = [row for row in rows if latest - row["created_ordinal"] <= window_days]
        if len(candidates) >= target:
            selected_window = int(window_days)
            window_rows = candidates
            break
    if selected_window is None:
        return None

    seed = int(selection_cfg["seed"])
    tau = float(selection_cfg["image_recency_tau_days"])
    scored = []
    for row in window_rows:
        weight = math.exp(-(latest - row["created_ordinal"]) / tau)
        key = _weighted_key(seed, f"image:{artist}:{row['id']}", weight)
        scored.append((key, int(row["id"]), row))
    scored.sort(reverse=True)

    result = []
    for rank, (key, _, row) in enumerate(scored[:target], start=1):
        result.append(
            {
                "id": int(row["id"]),
                "artist": artist,
                "created_at": row["created_at"],
                "md5": row["md5"],
                "file_ext": str(row["file_ext"]).lower(),
                "file_size": row.get("file_size"),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
                "source": row.get("source"),
                "pixiv_id": row.get("pixiv_id"),
                "parent_id": row.get("parent_id"),
                "tag_string_meta": row.get("tag_string_meta"),
                "download_url": row.get("file_url")
                or row.get("large_file_url")
                or row.get("original_url"),
                "selection_rank": rank,
                "selection_key": key,
                "date_window_days": selected_window,
            }
        )
    return result, selected_window


def select_candidates(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    metadata_cfg = config["metadata"]
    selection_cfg = config["selection"]
    stats: dict[str, ArtistStat] = defaultdict(ArtistStat)
    valid_rows = 0
    for row in _iter_valid_rows(metadata_cfg, selection_cfg):
        artist = row["tag_string_artist"].strip()
        stats[artist].add(row["created_ordinal"])
        valid_rows += 1

    artist_pool = _choose_artist_pool(stats, selection_cfg)
    pool_rank = {artist: rank for rank, (artist, _) in enumerate(artist_pool, start=1)}
    pool_keys = dict(artist_pool)
    collected = _collect_pool_rows(
        set(pool_rank), stats, metadata_cfg, selection_cfg
    )

    feasible: list[tuple[int, str, list[dict[str, Any]], int]] = []
    for artist, _ in artist_pool:
        selected = _select_artist_candidates(
            artist, collected.get(artist, []), selection_cfg
        )
        if selected is not None:
            rows, window = selected
            feasible.append((pool_rank[artist], artist, rows, window))

    requested = int(selection_cfg["artist_count"])
    if len(feasible) < requested:
        raise RuntimeError(
            f"Only {len(feasible)} artists remain feasible after applying date windows; "
            f"{requested} requested. Increase artist_pool_multiplier or date_windows_days."
        )
    feasible.sort()
    feasible = feasible[:requested]

    candidates: list[dict[str, Any]] = []
    artist_records: list[dict[str, Any]] = []
    for final_rank, (_, artist, rows, window) in enumerate(feasible, start=1):
        for row in rows:
            row["artist_rank"] = final_rank
            candidates.append(row)
        stat = stats[artist]
        artist_records.append(
            {
                "artist": artist,
                "artist_rank": final_rank,
                "metadata_count": stat.count,
                "earliest_date": date.fromordinal(stat.earliest_ordinal).isoformat(),
                "latest_date": date.fromordinal(stat.latest_ordinal).isoformat(),
                "selected_window_days": window,
                "candidate_count": len(rows),
                "artist_selection_key": pool_keys[artist],
            }
        )

    write_records(destination / "artist_stats.parquet", artist_records)
    write_records(destination / "candidate_manifest.parquet", candidates)
    summary = {
        "valid_metadata_rows": valid_rows,
        "artists_observed": len(stats),
        "artist_pool": len(artist_pool),
        "artists_selected": len(artist_records),
        "candidates_selected": len(candidates),
        "metadata_revision": metadata_cfg.get("revision"),
        "seed": selection_cfg["seed"],
    }
    write_json(destination / "selection_summary.json", summary)
    return summary
