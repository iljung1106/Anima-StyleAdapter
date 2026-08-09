from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from .download import _image_path, _md5
from .io import read_records, write_json, write_records


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _file_md5(path: Path) -> str:
    return _md5(path)


def _eligible(row: dict[str, Any], cutoff_date: str) -> bool:
    return str(row["created_at"])[:10] <= cutoff_date


def _import_staged_files(
    staged_dir: Path,
    row_by_id: dict[int, dict[str, Any]],
    images_dir: Path,
) -> dict[int, dict[str, Any]]:
    imported: dict[int, dict[str, Any]] = {}
    if not staged_dir.exists():
        return imported

    for path in staged_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            image_id = int(path.stem)
        except ValueError:
            continue
        row = row_by_id.get(image_id)
        if row is None:
            continue

        actual_md5 = _file_md5(path)
        target = _image_path(images_dir, row)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        imported[image_id] = {
            **row,
            "local_path": str(target.resolve()),
            "download_status": "downloaded",
            "download_source": "deepghs/danbooru2024",
            "actual_md5": actual_md5,
            "metadata_md5_match": actual_md5 == str(row["md5"]).lower(),
            "download_error": None,
        }
    return imported


def download_deepghs_candidates(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    from cheesechaser.datapool import Danbooru2024DataPool
    from huggingface_hub import get_token

    cfg = config["deepghs"]
    cutoff_date = str(cfg["cutoff_date"])
    rows = [
        row
        for row in read_records(destination / "candidate_manifest.parquet")
        if _eligible(row, cutoff_date)
    ]
    row_by_id = {int(row["id"]): row for row in rows}
    images_dir = destination / "images"
    results: dict[int, dict[str, Any]] = {}

    import_dir = cfg.get("import_dir")
    if import_dir:
        results.update(
            _import_staged_files(Path(import_dir), row_by_id, images_dir)
        )

    pending: list[dict[str, Any]] = []
    for row in rows:
        image_id = int(row["id"])
        if image_id in results:
            continue
        target = _image_path(images_dir, row)
        if target.exists() and target.stat().st_size > 0:
            actual_md5 = _file_md5(target)
            results[image_id] = {
                **row,
                "local_path": str(target.resolve()),
                "download_status": "cached",
                "download_source": "existing",
                "actual_md5": actual_md5,
                "metadata_md5_match": actual_md5 == str(row["md5"]).lower(),
                "download_error": None,
            }
        else:
            pending.append(row)

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token detected. Set HF_HOME to the directory containing token."
        )
    pool = Danbooru2024DataPool(hf_token=token)
    batch_size = max(int(cfg.get("batch_size", 1000)), 1)
    max_workers = max(int(cfg.get("max_workers", 12)), 1)
    started_at = time.monotonic()

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        with tempfile.TemporaryDirectory(
            prefix="deepghs-", dir=destination
        ) as temp_dir:
            pool.batch_download_to_directory(
                resource_ids=[int(row["id"]) for row in batch],
                dst_dir=temp_dir,
                max_workers=max_workers,
                save_metainfo=False,
                silent=True,
            )
            imported = _import_staged_files(
                Path(temp_dir), row_by_id, images_dir
            )
            results.update(imported)

        for row in batch:
            image_id = int(row["id"])
            if image_id not in imported:
                results[image_id] = {
                    **row,
                    "local_path": None,
                    "download_status": "unavailable",
                    "download_source": "deepghs/danbooru2024",
                    "actual_md5": None,
                    "metadata_md5_match": None,
                    "download_error": "resource not present in mirror",
                }

        completed = min(offset + len(batch), len(pending))
        successful = sum(
            result["download_status"] in {"downloaded", "cached"}
            for result in results.values()
        )
        elapsed = max(time.monotonic() - started_at, 1e-6)
        print(
            f"deepghs {completed}/{len(pending)} pending checked; "
            f"available={successful}/{len(rows)}, "
            f"batch_rate={completed / elapsed:.2f} items/s",
            flush=True,
        )
        write_json(
            destination / "deepghs_progress.json",
            {
                "eligible": len(rows),
                "pending_checked": completed,
                "available": successful,
                "elapsed_seconds": elapsed,
            },
        )

    ordered = [results[int(row["id"])] for row in rows]
    write_records(destination / "deepghs_manifest.parquet", ordered)
    successful = sum(
        row["download_status"] in {"downloaded", "cached"} for row in ordered
    )
    mismatched = sum(row["metadata_md5_match"] is False for row in ordered)
    summary = {
        "eligible": len(rows),
        "successful": successful,
        "unavailable": len(rows) - successful,
        "metadata_md5_mismatches": mismatched,
        "cutoff_date": cutoff_date,
        "repo_id": cfg["repo_id"],
    }
    write_json(destination / "deepghs_summary.json", summary)
    return summary
