from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    required = {"metadata", "selection", "download", "dedup", "tagger", "output_dir"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    config["_config_path"] = str(config_path)
    return config


def output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["output_dir"])
    if not path.is_absolute():
        # Paths are intentionally relative to the invocation directory, which is
        # convenient on both a local workstation and a mounted RunPod volume.
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
