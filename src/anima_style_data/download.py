from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

import aiohttp

from .io import read_records, write_json, write_records


class AdaptiveRequestLimiter:
    """Pace request starts and globally cool down when the CDN returns 429."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.rate = float(cfg.get("requests_per_second", 2.0))
        self.min_rate = float(cfg.get("min_requests_per_second", 0.5))
        self.max_rate = float(cfg.get("max_requests_per_second", self.rate))
        self.increase_every = max(int(cfg.get("rate_increase_every", 200)), 1)
        self.increase_step = float(cfg.get("rate_increase_step", 0.25))
        self.backoff_factor = float(cfg.get("rate_limit_backoff_factor", 0.5))
        self.cooldown = float(cfg.get("rate_limit_cooldown_seconds", 60))
        self._successes = 0
        self._next_request = 0.0
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    async def wait_for_slot(self) -> None:
        async with self._lock:
            now = time.monotonic()
            ready_at = max(self._next_request, self._blocked_until)
            if ready_at > now:
                await asyncio.sleep(ready_at - now)
            self._next_request = time.monotonic() + 1.0 / self.rate

    async def succeeded(self) -> None:
        async with self._lock:
            self._successes += 1
            if self._successes < self.increase_every:
                return
            self._successes = 0
            previous = self.rate
            self.rate = min(self.max_rate, self.rate + self.increase_step)
            if self.rate != previous:
                print(
                    f"CDN request rate increased to {self.rate:.2f} requests/s",
                    flush=True,
                )

    async def rate_limited(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # Several already-started requests can observe the same challenge.
            # Penalize the shared rate only once per active cooldown window.
            if self._blocked_until > now:
                return
            self.rate = max(self.min_rate, self.rate * self.backoff_factor)
            self._successes = 0
            self._blocked_until = now + self.cooldown
            self._next_request = max(self._next_request, self._blocked_until)
            print(
                f"CDN rate limit detected; pausing all requests for "
                f"{self.cooldown:.0f}s, then resuming at {self.rate:.2f} requests/s",
                flush=True,
            )


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
    limiter: AdaptiveRequestLimiter,
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
    attempt = 1
    while attempt <= retries:
        temp = target.with_suffix(target.suffix + ".part")
        try:
            await limiter.wait_for_slot()
            async with semaphore:
                timeout = aiohttp.ClientTimeout(total=float(cfg["timeout_seconds"]))
                async with session.get(row["download_url"], timeout=timeout) as response:
                    if response.status == 429 or response.headers.get("cf-mitigated") == "challenge":
                        await response.read()
                        await limiter.rate_limited()
                        # A CDN challenge is transient and must not consume the
                        # ordinary per-image retry budget or become a failed row.
                        continue
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
            await limiter.succeeded()
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
            attempt += 1
    return {**base, "download_status": "failed", "download_error": error}


async def _run_downloads(
    rows: list[dict[str, Any]], images_dir: Path, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    concurrency = int(cfg["concurrency"])
    semaphore = asyncio.Semaphore(concurrency)
    limiter = AdaptiveRequestLimiter(cfg)
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
        status_counts = {"downloaded": 0, "cached": 0, "failed": 0}
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
                    session, semaphore, rows[index], images_dir, cfg, limiter
                )
                status = results[index]["download_status"]
                status_counts[status] += 1
                completed += 1
                if completed % report_every == 0 or completed == len(rows):
                    elapsed = max(time.monotonic() - started_at, 1e-6)
                    print(
                        f"downloaded/checked {completed}/{len(rows)} "
                        f"({completed / elapsed:.2f} items/s; "
                        f"downloaded={status_counts['downloaded']}, "
                        f"cached={status_counts['cached']}, "
                        f"failed={status_counts['failed']}, "
                        f"request_rate={limiter.rate:.2f}/s)",
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
