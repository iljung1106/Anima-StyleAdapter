from __future__ import annotations

import asyncio
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from anima_style_data.download import _run_downloads


def test_parallel_download_and_cached_resume(tmp_path: Path) -> None:
    payloads = {f"/{index}": f"image-{index}".encode() for index in range(6)}
    request_counts: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_counts[self.path] = request_counts.get(self.path, 0) + 1
            if self.path == "/0" and request_counts[self.path] == 1:
                self.send_response(429)
                self.send_header("cf-mitigated", "challenge")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = payloads[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = [
            {
                "id": index,
                "file_ext": "jpg",
                "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                "download_url": f"http://127.0.0.1:{server.server_port}/{index}",
            }
            for index, payload in enumerate(payloads.values())
        ]
        cfg = {
            "concurrency": 3,
            "timeout_seconds": 5,
            "retries": 1,
            "max_file_mb": 1,
            "user_agent": "test",
            "progress_every": 2,
            "requests_per_second": 100,
            "min_requests_per_second": 100,
            "max_requests_per_second": 100,
            "rate_limit_cooldown_seconds": 0.01,
        }
        images_dir = tmp_path / "images"

        first = asyncio.run(_run_downloads(rows, images_dir, cfg))
        second = asyncio.run(_run_downloads(rows, images_dir, cfg))

        assert [item["download_status"] for item in first] == ["downloaded"] * 6
        assert [item["download_status"] for item in second] == ["cached"] * 6
        assert request_counts["/0"] == 2
    finally:
        server.shutdown()
        thread.join()
