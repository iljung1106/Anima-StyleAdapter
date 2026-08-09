from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import aiohttp

from .io import read_records, write_json, write_records


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_path(images_dir: Path, row: dict[str, Any]) -> Path:
    image_id = int(row["id"])
    shard = f"{image_id % 1000:03d}"
    extension = str(row["file_ext"]).lower()
    return images_dir / shard / f"{image_id}.{extension}"


async def _download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    images_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    target = _image_path(images_dir, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = str(row["md5"]).lower()
    base = dict(row)
    base["local_path"] = str(target.resolve())

    if target.exists() and _md5(target) == expected_md5:
        return {**base, "download_status": "cached", "download_error": None}

    max_bytes = int(float(cfg["max_file_mb"]) * 1024 * 1024)
    retries = int(cfg["retries"])
    error = "unknown"
    async with semaphore:
        for attempt in range(1, retries + 1):
            temp = target.with_suffix(target.suffix + ".part")
            try:
                timeout = aiohttp.ClientTimeout(total=float(cfg["timeout_seconds"]))
                async with session.get(row["download_url"], timeout=timeout) as response:
                    response.raise_for_status()
                    declared = response.content_length
                    if declared is not None and declared > max_bytes:
                        raise ValueError(f"declared file size {declared} exceeds limit")
                    digest = hashlib.md5(usedforsecurity=False)
                    size = 0
                    with temp.open("wb") as handle:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            size += len(chunk)
                            if size > max_bytes:
                                raise ValueError(f"download exceeded {max_bytes} bytes")
                            digest.update(chunk)
                            handle.write(chunk)
                actual_md5 = digest.hexdigest()
                if actual_md5 != expected_md5:
                    raise ValueError(
                        f"md5 mismatch: expected {expected_md5}, received {actual_md5}"
                    )
                temp.replace(target)
                return {
                    **base,
                    "download_status": "downloaded",
                    "download_error": None,
                    "downloaded_bytes": size,
                }
            except Exception as exc:  # network failures are recorded per image
                error = f"{type(exc).__name__}: {exc}"
                if temp.exists():
                    temp.unlink()
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))
    return {**base, "download_status": "failed", "download_error": error}


async def _run_downloads(
    rows: list[dict[str, Any]], images_dir: Path, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    concurrency = int(cfg["concurrency"])
    semaphore = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": cfg["user_agent"]}
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        results = []
        # Keep only a small multiple of concurrency as live Tasks. Creating one
        # Task per production candidate would otherwise allocate ~300k Tasks.
        chunk_size = max(concurrency * 4, 1)
        for offset in range(0, len(rows), chunk_size):
            chunk = rows[offset : offset + chunk_size]
            chunk_results = await asyncio.gather(
                *(
                    _download_one(session, semaphore, row, images_dir, cfg)
                    for row in chunk
                )
            )
            results.extend(chunk_results)
            completed = min(offset + len(chunk), len(rows))
            if completed % 100 == 0 or completed == len(rows):
                print(f"downloaded/checked {completed}/{len(rows)}", flush=True)
        return results


def download_candidates(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    rows = read_records(destination / "candidate_manifest.parquet")
    images_dir = destination / "images"
    results = asyncio.run(_run_downloads(rows, images_dir, config["download"]))
    results.sort(key=lambda row: (row["artist_rank"], row["selection_rank"]))
    write_records(destination / "download_manifest.parquet", results)
    success = sum(row["download_status"] != "failed" for row in results)
    summary = {"requested": len(results), "successful": success, "failed": len(results) - success}
    write_json(destination / "download_summary.json", summary)
    return summary
