from __future__ import annotations

import hashlib
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from PIL import Image

from .io import read_records, write_json, write_records


def _source_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["anima500k"]
    if cfg.get("source") != "human":
        raise ValueError("Only the human source is enabled; synthetic data is intentionally disabled")
    return cfg


def download_anima500k_human(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Download only human tar shards and human verification metadata."""
    cfg = _source_config(config)
    source_dir = destination / cfg.get("source_dir", "source")
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="dataset",
        revision=cfg["revision"],
        local_dir=source_dir,
        allow_patterns=["human/*.tar", "metadata/human_*.json", "README.md"],
        max_workers=int(cfg.get("download_workers", 16)),
    )
    shards = sorted((Path(snapshot_path) / "human").glob("*.tar"))
    summary = {
        "source": "human",
        "synthetic_included": False,
        "shards": len(shards),
        "downloaded_bytes": sum(path.stat().st_size for path in shards),
        "snapshot_path": str(Path(snapshot_path).resolve()),
    }
    write_json(destination / "anima500k_download_summary.json", summary)
    return summary


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_shard(
    shard: Path, destination: Path, manifest_dir: Path
) -> tuple[Path, int, int]:
    manifest_path = manifest_dir / f"{shard.stem}.parquet"
    if manifest_path.exists():
        rows = read_records(manifest_path)
        return manifest_path, len(rows), 0

    rows: list[dict[str, Any]] = []
    written_bytes = 0
    with tarfile.open(shard, "r") as archive:
        json_members = [member for member in archive if member.isfile() and member.name.endswith(".json")]
        for member in json_members:
            metadata_file = archive.extractfile(member)
            if metadata_file is None:
                raise RuntimeError(f"Could not read {member.name} from {shard}")
            metadata = json.load(metadata_file)
            if metadata.get("source") != "human" or not str(metadata.get("style_id", "")).startswith("human:"):
                raise ValueError(f"Non-human record found in human shard: {member.name}")

            full = metadata["full_image"]
            image_member = archive.getmember(full["filename"])
            image_file = archive.extractfile(image_member)
            if image_file is None:
                raise RuntimeError(f"Could not read {full['filename']} from {shard}")
            image_bytes = image_file.read()
            digest = _sha256(image_bytes)
            if digest != full["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {full['filename']} in {shard}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()

            post_id = int(metadata["danbooru_post_id"])
            image_dir = destination / "images" / f"{post_id % 1000:03d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            local_path = image_dir / full["filename"]
            if not local_path.exists() or local_path.stat().st_size != len(image_bytes):
                local_path.write_bytes(image_bytes)
                written_bytes += len(image_bytes)

            rows.append(
                {
                    "id": post_id,
                    "record_id": metadata["record_id"],
                    "artist": metadata["artist"],
                    "style_id": metadata["style_id"],
                    "split": metadata["split"],
                    "artist_image_index": int(metadata["artist_image_index"]),
                    "selection_rank": int(metadata["artist_image_index"]) + 1,
                    "created_at": metadata["created_at"],
                    "rating": metadata["rating"],
                    "gender": metadata["gender"],
                    "md5": metadata["md5"],
                    "sha256": digest,
                    "width": int(full["width"]),
                    "height": int(full["height"]),
                    "tag_string_general": metadata.get("tag_string_general", ""),
                    "tag_string_character": metadata.get("tag_string_character", ""),
                    "tag_string_copyright": metadata.get("tag_string_copyright", ""),
                    "tag_string_meta": metadata.get("tag_string_meta", ""),
                    "local_path": str(local_path.resolve()),
                    "source_shard": str(shard.resolve()),
                    "source_member": full["filename"],
                    "download_status": "extracted",
                }
            )
    write_records(manifest_path, rows)
    return manifest_path, len(rows), written_bytes


def extract_anima500k_human(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Extract full human images only, with one resumable manifest per tar shard."""
    cfg = _source_config(config)
    source_dir = destination / cfg.get("source_dir", "source")
    shards = sorted((source_dir / "human").glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No human tar shards found under {source_dir / 'human'}")
    manifest_dir = destination / "extract_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    completed: list[Path] = []
    records = 0
    written_bytes = 0
    workers = int(cfg.get("extract_workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_shard, shard, destination, manifest_dir): shard
            for shard in shards
        }
        for index, future in enumerate(as_completed(futures), start=1):
            manifest, count, added = future.result()
            completed.append(manifest)
            records += count
            written_bytes += added
            print(
                f"extracted shards {index}/{len(shards)}; records={records}; "
                f"new_bytes={written_bytes}",
                flush=True,
            )

    all_rows: list[dict[str, Any]] = []
    for manifest in sorted(completed):
        all_rows.extend(read_records(manifest))
    all_rows.sort(key=lambda row: (row["split"], row["style_id"], row["artist_image_index"]))
    write_records(destination / "download_manifest.parquet", all_rows)
    split_counts: dict[str, int] = {}
    for row in all_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "source": "human",
        "synthetic_included": False,
        "face_images_extracted": 0,
        "shards": len(shards),
        "records": len(all_rows),
        "splits": split_counts,
        "new_image_bytes": written_bytes,
    }
    write_json(destination / "anima500k_extract_summary.json", summary)
    return summary
