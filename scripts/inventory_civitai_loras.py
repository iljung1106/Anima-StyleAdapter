"""Inventory downloaded Anima LoRAs without materializing their tensors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def inspect(path: Path) -> dict[str, object]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
        down_keys = [key for key in keys if key.endswith(".lora_down.weight")]
        ranks = Counter(
            int(handle.get_slice(key).get_shape()[0]) for key in down_keys
        )
    prefixes = [key.removesuffix(".lora_down.weight") for key in down_keys]
    unet = [value for value in prefixes if value.startswith("lora_unet_")]
    text = [value for value in prefixes if value.startswith("lora_te")]
    unknown = [value for value in prefixes if value not in unet and value not in text]
    return {
        "path": str(path.resolve()),
        "file_name": path.name,
        "bytes": path.stat().st_size,
        "network_module": metadata.get("ss_network_module"),
        "base_model": metadata.get("ss_base_model_version"),
        "declared_rank": metadata.get("ss_network_dim"),
        "modules": len(prefixes),
        "unet_modules": len(unet),
        "text_modules": len(text),
        "unknown_modules": unknown,
        "rank_counts": {str(key): value for key, value in sorted(ranks.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    files = sorted(
        path
        for root in args.roots
        for path in root.glob("*.safetensors")
        if path.is_file()
    )
    items = [inspect(path) for path in files]
    rank_files: Counter[str] = Counter()
    network_modules: Counter[str] = Counter()
    for item in items:
        rank_files.update(item["rank_counts"].keys())
        network_modules[str(item["network_module"])] += 1
    summary = {
        "files": len(items),
        "gib": sum(int(item["bytes"]) for item in items) / 1024**3,
        "files_containing_rank": dict(sorted(rank_files.items(), key=lambda row: int(row[0]))),
        "network_modules": dict(network_modules.most_common()),
        "files_with_text_modules": sum(int(item["text_modules"]) > 0 for item in items),
        "files_with_unknown_modules": sum(bool(item["unknown_modules"]) for item in items),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
