"""Select recent, popular Anima style LoRAs from the CivitAI REST API."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from download_civitai_model_list import request_json


STYLE_TERMS = (
    "style",
    "artist",
    "artstyle",
    "art style",
    "画风",
    "画風",
    "絵柄",
    "作画",
    "스타일",
    "화풍",
)
REJECT_TERMS = (
    "character",
    "celebrity",
    "real person",
    "costume",
    "clothing",
    "outfit",
    "pose",
    "expression",
    "vehicle",
    "object",
    "location",
    "background",
    "body slider",
    "breast",
    "turbo",
    "lightning",
    "lcm",
    "step distill",
    "detail tweaker",
    "detail slider",
    "quality modifier",
    "aesthetic boost",
    "enhancer",
    "highres",
    "upscale",
    "real skin",
    "lighting",
    # Non-style control/adaptor LoRAs can still carry a generic ``style`` tag.
    # Their effect is a scalar edit or utility operation, not an artist/style
    # distribution suitable for reference-conditioned style distillation.
    "slider",
    "tweaker",
    "detailer",
    "utility",
    "tool",
    "concept",
    "pov",
    "angle",
    "rotation",
    "aesthetic improvement",
)
TRIGGER_PATTERNS = (
    re.compile(
        r"(?:trigger(?:\s+word)?|activation\s+word|keyword)\s*"
        r"(?:is|are|:|=|-)?\s*[`\"']?([^\n,;<>]{1,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:use|prompt)\s*[`\"']([^`\"']{1,80})[`\"']",
        re.IGNORECASE,
    ),
)


def plain_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text, flags=re.IGNORECASE)
    return re.sub(r"<[^>]+>", " ", text)


def inferred_triggers(*descriptions: object) -> list[str]:
    found: list[str] = []
    for description in descriptions:
        text = plain_text(description)
        for pattern in TRIGGER_PATTERNS:
            for match in pattern.finditer(text):
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" `\"'.:-")
                if 1 <= len(candidate) <= 64 and candidate.lower() not in {
                    item.lower() for item in found
                }:
                    found.append(candidate)
    return found[:4]


def style_decision(model: dict) -> tuple[bool, str]:
    tags = [str(tag).strip().lower() for tag in model.get("tags") or []]
    name = str(model.get("name") or "")
    description = plain_text(model.get("description"))[:8000]
    searchable = " ".join([name, description, *tags]).lower()
    identity = " ".join([name, *tags]).lower()
    positives = sorted({term for term in STYLE_TERMS if term in searchable})
    # Descriptions of valid style LoRAs routinely mention characters, outfits,
    # backgrounds, or lighting as supported subjects.  Reject only when those
    # concepts identify the model itself through its title or tags.
    rejects = sorted({term for term in REJECT_TERMS if term in identity})
    if rejects:
        return False, "rejected:" + ",".join(rejects)
    if not positives:
        return False, "no-style-signal"
    if model.get("allowDerivatives") is False:
        return False, "derivatives-disallowed"
    return True, "style:" + ",".join(positives)


def parse_time(value: object) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def primary_safetensor(version: dict) -> dict | None:
    files = [
        file
        for file in version.get("files") or []
        if str((file.get("metadata") or {}).get("format", "")).lower()
        == "safetensor"
        and str(file.get("type", "Model")) == "Model"
    ]
    if not files:
        return None
    return next((file for file in files if file.get("primary")), files[0])


def load_existing(manifests: list[Path]) -> tuple[set[int], set[str]]:
    versions: set[int] = set()
    hashes: set[str] = set()
    for path in manifests:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            if item.get("version_id") is not None:
                versions.add(int(item["version_id"]))
            if item.get("sha256"):
                hashes.add(str(item["sha256"]).upper())
    return versions, hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cutoff", default="2026-05-28T00:00:00+00:00")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--candidate-models", type=int, default=1200)
    parser.add_argument("--existing-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    token = os.environ.get("CIVITAI_API_TOKEN")
    if not token:
        raise SystemExit("CIVITAI_API_TOKEN is required")
    cutoff = datetime.fromisoformat(args.cutoff).astimezone(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    existing_versions, existing_hashes = load_existing(args.existing_manifest)

    url = (
        "https://civitai.red/api/v1/models?"
        + urllib.parse.urlencode(
            {
                "types": "LORA",
                "baseModels": "Anima",
                "sort": "Most Downloaded",
                "period": "Year",
                "limit": 100,
            }
        )
    )
    summaries: list[dict] = []
    seen: set[int] = set()
    while url and len(summaries) < args.candidate_models:
        page = request_json(url, token)
        for model in page.get("items") or []:
            model_id = int(model["id"])
            if model_id not in seen:
                seen.add(model_id)
                summaries.append(model)
        url = (page.get("metadata") or {}).get("nextPage")
        if url:
            url = str(url).replace("https://civitai.com", "https://civitai.red")
        print(f"listed {len(summaries)} candidate models", flush=True)

    details: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(
                request_json,
                f'https://civitai.red/api/v1/models/{model["id"]}',
                token,
            ): model
            for model in summaries
        }
        for index, future in enumerate(as_completed(futures), 1):
            details.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"detailed {index}/{len(futures)}", flush=True)

    eligible: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    for model in details:
        accepted, reason = style_decision(model)
        if not accepted:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        choices: list[dict[str, object]] = []
        for version in model.get("modelVersions") or []:
            created = parse_time(version.get("createdAt") or version.get("publishedAt"))
            if created is None or created < cutoff or version.get("baseModel") != "Anima":
                continue
            file = primary_safetensor(version)
            if file is None:
                continue
            version_id = int(version["id"])
            sha = str((file.get("hashes") or {}).get("SHA256") or "").upper()
            if version_id in existing_versions or (sha and sha in existing_hashes):
                continue
            api_triggers = [str(value).strip() for value in version.get("trainedWords") or [] if str(value).strip()]
            description_triggers = inferred_triggers(
                version.get("description"), model.get("description")
            )
            choices.append(
                {
                    "model_id": int(model["id"]),
                    "model_name": model.get("name"),
                    "model_url": f'https://civitai.red/models/{model["id"]}',
                    "creator": (model.get("creator") or {}).get("username"),
                    "model_description": model.get("description"),
                    "tags": model.get("tags") or [],
                    "allow_derivatives": model.get("allowDerivatives"),
                    "allow_commercial_use": model.get("allowCommercialUse"),
                    "selection_reason": reason,
                    "version_id": version_id,
                    "version_name": version.get("name"),
                    "created_at": created.isoformat(),
                    "version_description": version.get("description"),
                    "download_count": int((version.get("stats") or {}).get("downloadCount") or 0),
                    "model_download_count": int((model.get("stats") or {}).get("downloadCount") or 0),
                    "api_trigger_words": api_triggers,
                    "description_trigger_words": description_triggers,
                    "effective_trigger_words": api_triggers or description_triggers,
                    "file_id": int(file["id"]),
                    "file_name": file.get("name"),
                    "size_kb": file.get("sizeKB"),
                    "sha256": sha,
                    "source_url": f'https://civitai.red/api/download/models/{version_id}?fileId={int(file["id"])}',
                }
            )
        if choices:
            eligible.append(max(choices, key=lambda item: int(item["download_count"])))

    eligible.sort(
        key=lambda item: (
            int(item["download_count"]),
            int(item["model_download_count"]),
        ),
        reverse=True,
    )
    selected = eligible[: args.count]
    if len(selected) != args.count:
        raise SystemExit(
            f"Only {len(selected)} eligible recent style LoRAs found; "
            f"increase --candidate-models (eligible={len(eligible)})"
        )
    total_gib = sum(float(item.get("size_kb") or 0) for item in selected) / 1024**2
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoff": cutoff.isoformat(),
        "ranking": "version.stats.downloadCount descending",
        "candidate_models": len(summaries),
        "eligible_models": len(eligible),
        "selected_models": len(selected),
        "minimum_download_count": min(int(item["download_count"]) for item in selected),
        "expected_gib": total_gib,
        "rejected": rejected,
        "items": selected,
    }
    (args.output / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines: list[str] = []
    for item in selected:
        lines.extend(
            [
                str(item["source_url"]),
                ", ".join(item["effective_trigger_words"]),
                "",
            ]
        )
    (args.output / "download_list.txt").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"selected {len(selected)} models; min_downloads="
        f'{manifest["minimum_download_count"]}; expected={total_gib:.2f} GiB',
        flush=True,
    )


if __name__ == "__main__":
    main()
