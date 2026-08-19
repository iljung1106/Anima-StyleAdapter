"""Download contract for Tencent MegaStyle-1.4M."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .io import write_json


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
