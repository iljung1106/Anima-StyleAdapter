from __future__ import annotations

import asyncio
import hashlib
import time
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

    if target.exists() and await asyncio.to_thread(_md5, target) == expected_md5:
        return {**base, "download_status": "cached", "download_error": None}

    max_bytes = int(float(cfg["max_file_mb"]) * 1024 * 1024)
    retries = int(cfg["retries"])
    error = "unknown"
    for attempt in range(1, retries + 1):
        temp = target.with_suffix(target.suffix + ".part")
        try:
            async with semaphore:
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
                # Do not reserve a network slot while this item backs off.
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
    return {**base, "download_status": "failed", "download_error": error}


async def _run_downloads(
    rows: list[dict[str, Any]], images_dir: Path, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    concurrency = int(cfg["concurrency"])
    semaphore = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": cfg["user_agent"]}
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        results: list[dict[str, Any] | None] = [None] * len(rows)
        next_index = 0
        completed = 0
        started_at = time.monotonic()
        report_every = max(int(cfg.get("progress_every", 1000)), 1)

        async def worker() -> None:
            nonlocal next_index, completed
            while next_index < len(rows):
                # There is no await between claiming an index and incrementing it,
                # so each event-loop worker receives a unique row.
                index = next_index
                next_index += 1
                results[index] = await _download_one(
                    session, semaphore, rows[index], images_dir, cfg
                )
                completed += 1
                if completed % report_every == 0 or completed == len(rows):
                    elapsed = max(time.monotonic() - started_at, 1e-6)
                    print(
                        f"downloaded/checked {completed}/{len(rows)} "
                        f"({completed / elapsed:.2f} items/s)",
                        flush=True,
                    )

        await asyncio.gather(*(worker() for _ in range(min(concurrency, len(rows)))))
        return [result for result in results if result is not None]


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
