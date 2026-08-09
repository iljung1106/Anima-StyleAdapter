from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .io import read_records, write_json, write_records


COUNT_TAG = re.compile(r"^(?:\d+|multiple)\s*(?:girls?|boys?|others?)$")
SPECIAL_COUNT_TAGS = {"no humans", "multiple girls", "multiple boys"}


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        normalized = " ".join(str(item).replace("_", " ").lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _is_count_tag(tag: str) -> bool:
    return tag in SPECIAL_COUNT_TAGS or bool(COUNT_TAG.fullmatch(tag))


def build_anima_caption(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    general = _unique(row.get("general_tags") or [])
    characters = _unique(row.get("character_tags") or [])
    counts = [tag for tag in general if _is_count_tag(tag)]
    remaining_general = [tag for tag in general if tag not in set(counts)]
    rating = cfg.get("rating_map", {}).get(row.get("rating"), row.get("rating"))

    anima_parts: list[str] = []
    if cfg.get("include_rating", True) and rating:
        anima_parts.append(rating)
    anima_parts.extend(counts)
    if cfg.get("include_characters", True):
        anima_parts.extend(characters)
    if cfg.get("include_general", True):
        anima_parts.extend(remaining_general)
    anima_parts = _unique(anima_parts)

    # Safety/rating affects generation but is not visual content to subtract.
    content_parts = _unique([*counts, *characters, *remaining_general])
    return {
        "id": int(row["id"]),
        "artist": row["artist"],
        "local_path": row["local_path"],
        "rating_source": row.get("rating"),
        "rating_anima": rating,
        "count_tags": counts,
        "character_tags": characters,
        "general_tags": remaining_general,
        "anima_caption": ", ".join(anima_parts),
        "content_caption": ", ".join(content_parts),
        "tagger_revision": row.get("tagger_revision"),
        "tagger_threshold": row.get("tagger_threshold"),
    }


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def create_anima_captions(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    cfg = config["caption"]
    config_hash = _config_hash(cfg)
    tags_dir = destination / "tags"
    caption_dir = destination / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    source_shards = sorted(tags_dir.glob("part-*.parquet"))
    if not source_shards:
        raise FileNotFoundError(f"No tag shards found under {tags_dir}")

    written = 0
    reused = 0
    rows_total = 0
    for source in source_shards:
        target = caption_dir / source.name
        if target.exists():
            existing = read_records(target)
            if existing and existing[0].get("caption_config_hash") == config_hash:
                reused += 1
                rows_total += len(existing)
                continue
        records = []
        for row in read_records(source):
            record = build_anima_caption(row, cfg)
            record["caption_version"] = cfg["version"]
            record["caption_config_hash"] = config_hash
            record["tag_source_shard"] = source.name
            records.append(record)
        rows_total += write_records(target, records)
        written += 1

    summary = {
        "rows": rows_total,
        "source_shards": len(source_shards),
        "written_shards": written,
        "reused_shards": reused,
        "caption_version": cfg["version"],
        "caption_config_hash": config_hash,
    }
    write_json(destination / "caption_summary.json", summary)
    return summary
