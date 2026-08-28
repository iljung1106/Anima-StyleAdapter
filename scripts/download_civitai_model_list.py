"""Download the CivitAI model-version links listed in a trigger-word text file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_entries(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    entries: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        url = line.strip()
        if not url.startswith(("https://", "http://")):
            continue
        parsed = urllib.parse.urlparse(url)
        version_id = int(parsed.path.rstrip("/").split("/")[-1])
        query = urllib.parse.parse_qs(parsed.query)
        file_id = int(query["fileId"][0]) if query.get("fileId") else None
        trigger = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if trigger.startswith(("https://", "http://")):
            trigger = ""
        entries.append(
            {
                "source_url": url,
                "source_trigger_words": trigger,
                "version_id": version_id,
                "file_id": file_id,
            }
        )
    unique: dict[tuple[int, int | None], dict[str, object]] = {}
    for entry in entries:
        unique[(int(entry["version_id"]), entry["file_id"])] = entry
    return list(unique.values())


def request_json(url: str, token: str | None, attempts: int = 5) -> dict:
    headers = {"User-Agent": "Anima-StyleAdapter-CivitAI-Downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=90
            ) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def enrich_entry(entry: dict[str, object], token: str | None) -> dict[str, object]:
    version_id = int(entry["version_id"])
    metadata = request_json(
        f"https://civitai.red/api/v1/model-versions/{version_id}", token
    )
    files = list(metadata.get("files") or [])
    file_id = entry.get("file_id")
    selected = next(
        (item for item in files if int(item.get("id", -1)) == file_id), None
    )
    if selected is None:
        selected = next((item for item in files if item.get("primary")), None)
    if selected is None:
        raise RuntimeError(f"No downloadable file in version {version_id}")
    safe_name = Path(str(selected["name"])).name
    entry.update(
        {
            "version_name": metadata.get("name"),
            "base_model": metadata.get("baseModel"),
            "api_trigger_words": metadata.get("trainedWords") or [],
            "version_description": metadata.get("description"),
            "file_id": int(selected["id"]),
            "file_name": safe_name,
            "size_kb": selected.get("sizeKB"),
            "sha256": (selected.get("hashes") or {}).get("SHA256"),
            "format": (selected.get("metadata") or {}).get("format"),
        }
    )
    return entry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def valid_existing(path: Path, entry: dict[str, object]) -> bool:
    if not path.exists():
        return False
    size_kb = entry.get("size_kb")
    if size_kb is not None:
        expected = round(float(size_kb) * 1024)
        if abs(path.stat().st_size - expected) > 8:
            return False
    expected_hash = entry.get("sha256")
    return not expected_hash or sha256(path) == str(expected_hash).upper()


def download_one(
    number: int,
    total: int,
    entry: dict[str, object],
    output: Path,
    token: str | None,
) -> tuple[int, str, str]:
    target = output / (
        f'{int(entry["version_id"])}_{int(entry["file_id"])}_'
        f'{entry["file_name"]}'
    )
    if valid_existing(target, entry):
        return number, "reused", target.name
    partial = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "Anima-StyleAdapter-CivitAI-Downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(
                str(entry["source_url"]), headers=request_headers
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                # A server may ignore Range and return a complete 200 response.
                # In that case overwrite the partial instead of appending it.
                append = offset > 0 and getattr(response, "status", 200) == 206
                with partial.open("ab" if append else "wb") as handle:
                    while chunk := response.read(8 * 1024 * 1024):
                        handle.write(chunk)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    expected_hash = entry.get("sha256")
    if expected_hash and sha256(partial) != str(expected_hash).upper():
        raise RuntimeError(f"SHA-256 mismatch: {partial}")
    partial.replace(target)
    return number, "downloaded", target.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("list_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("CIVITAI_API_TOKEN")
    entries = parse_entries(args.list_file)
    print(f"parsed {len(entries)} unique download entries", flush=True)

    enriched: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(enrich_entry, entry, token): entry for entry in entries}
        for completed, future in enumerate(as_completed(futures), 1):
            enriched.append(future.result())
            if completed % 10 == 0 or completed == len(entries):
                print(f"metadata {completed}/{len(entries)}", flush=True)
    enriched.sort(key=lambda item: entries.index(futures_entry(entries, item)))

    total_gib = sum(float(item.get("size_kb") or 0) for item in enriched) / 1024**2
    manifest = args.output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"source": str(args.list_file), "total_gib": total_gib, "items": enriched},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"expected download size: {total_gib:.2f} GiB", flush=True)

    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, i, len(enriched), item, args.output, token): item
            for i, item in enumerate(enriched, 1)
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                number, status, _ = future.result()
                print(
                    f'[{number}/{len(enriched)}] {status}: '
                    f'version={item["version_id"]}',
                    flush=True,
                )
            except Exception as error:
                error_summary = type(error).__name__
                failures.append(
                    {"version_id": item["version_id"], "error": error_summary}
                )
                print(
                    f'FAILED version={item["version_id"]}: {error_summary}',
                    flush=True,
                )
    if failures:
        (args.output / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"{len(failures)} downloads failed")
    print(f"completed {len(enriched)} downloads", flush=True)


def futures_entry(
    original: list[dict[str, object]], enriched: dict[str, object]
) -> dict[str, object]:
    key = (enriched["version_id"], enriched["file_id"])
    return next(
        item
        for item in original
        if (item["version_id"], item["file_id"]) == key
    )


if __name__ == "__main__":
    main()
