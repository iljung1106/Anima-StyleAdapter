"""Combine verified CivitAI download manifests into one external teacher bank."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


def weight_format(path: Path) -> str:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if any(key.endswith(".lora_down.weight") for key in keys):
        return "kohya_lora"
    if any(key.endswith(".lora_A.weight") for key in keys):
        return "diffusers_lora"
    if any(".lokr_" in key for key in keys):
        return "lokr"
    if any(".hada_" in key for key in keys):
        return "loha"
    raise RuntimeError(f"Unsupported adapter format: {path}")


def triggers(item: dict, selected: dict | None) -> list[str]:
    values = []
    for source in (
        item.get("source_trigger_words", "").split(","),
        item.get("api_trigger_words") or [],
        (selected or {}).get("effective_trigger_words") or [],
    ):
        for value in source:
            value = str(value).strip()
            if value and value.casefold() not in {item.casefold() for item in values}:
                values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--weights", action="append", required=True, type=Path)
    parser.add_argument("--selection", action="append", type=Path, default=[])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if len(args.manifest) != len(args.weights):
        raise SystemExit("--manifest and --weights counts must match")

    selection_by_version: dict[int, dict] = {}
    for path in args.selection:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selection_by_version.update(
            {int(item["version_id"]): item for item in payload.get("items", [])}
        )

    items = []
    seen_versions: set[int] = set()
    for manifest_path, weights_root in zip(args.manifest, args.weights, strict=True):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for source in payload.get("items", []):
            version_id = int(source["version_id"])
            if version_id in seen_versions:
                continue
            seen_versions.add(version_id)
            name = (
                f'{version_id}_{int(source["file_id"])}_'
                f'{Path(str(source["file_name"])).name}'
            )
            path = (weights_root / name).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            selected = selection_by_version.get(version_id)
            try:
                relative = path.relative_to(args.root.resolve())
                recorded_path = str(relative)
            except ValueError:
                recorded_path = str(path)
            items.append(
                {
                    "index": len(items),
                    "style_id": f"civitai-v{version_id}",
                    "artist": str((selected or {}).get("model_name") or path.stem),
                    "version_id": version_id,
                    "file_id": int(source["file_id"]),
                    "weight_path": recorded_path,
                    "sha256": source.get("sha256"),
                    "format": weight_format(path),
                    "trigger_words": triggers(source, selected),
                    "download_count": (selected or {}).get("download_count"),
                    "created_at": (selected or {}).get("created_at"),
                    "source_url": source.get("source_url"),
                }
            )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "teachers": len(items),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = {kind: sum(item["format"] == kind for item in items) for kind in sorted({item["format"] for item in items})}
    print(json.dumps({"teachers": len(items), "formats": counts}, indent=2))


if __name__ == "__main__":
    main()
