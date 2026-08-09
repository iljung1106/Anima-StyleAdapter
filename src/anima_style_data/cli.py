from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .anima500k import download_anima500k_human, extract_anima500k_human
from .caption import create_anima_captions
from .config import load_config, output_dir
from .cradio import extract_cradio_features
from .dedup import deduplicate
from .deepghs import download_deepghs_candidates
from .download import download_candidates
from .metadata import select_candidates
from .tagger import tag_images


Stage = Callable[[dict[str, Any], Path], dict[str, Any]]


def _run(stage: Stage, config: dict[str, Any], destination: Path) -> None:
    summary = stage(config, destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anima-data",
        description="Prepare Danbooru artist data for Anima Style Adapter training.",
    )
    parser.add_argument("--config", required=True, help="YAML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("select", "Select artists and oversampled image candidates"),
        ("download", "Download and checksum selected candidates"),
        ("deepghs", "Download eligible candidates from the indexed Danbooru2024 mirror"),
        ("anima500k-download", "Download human shards from anima-style-embedding-500k"),
        ("anima500k-extract", "Extract full human images and build a manifest"),
        ("dedup", "Decode, hash, remove near-duplicates, and make final manifest"),
        ("tag", "Run the WD EVA02 tagger into resumable Parquet shards"),
        ("caption", "Build ordered Anima and content caption shards"),
        ("features", "Cache C-RADIO spatial and SigLIP2-g residual features"),
        ("prepare", "Run selection, download, and duplicate removal"),
        ("all", "Run every stage, including tagger and C-RADIO models"),
    ):
        subparsers.add_parser(command, help=help_text)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    destination = output_dir(config)
    if args.command == "select":
        _run(select_candidates, config, destination)
    elif args.command == "download":
        _run(download_candidates, config, destination)
    elif args.command == "deepghs":
        _run(download_deepghs_candidates, config, destination)
    elif args.command == "anima500k-download":
        _run(download_anima500k_human, config, destination)
    elif args.command == "anima500k-extract":
        _run(extract_anima500k_human, config, destination)
    elif args.command == "dedup":
        _run(deduplicate, config, destination)
    elif args.command == "tag":
        _run(tag_images, config, destination)
    elif args.command == "caption":
        _run(create_anima_captions, config, destination)
    elif args.command == "features":
        _run(extract_cradio_features, config, destination)
    elif args.command in {"prepare", "all"}:
        for stage in (select_candidates, download_candidates, deduplicate):
            _run(stage, config, destination)
        if args.command == "all":
            _run(tag_images, config, destination)
            _run(create_anima_captions, config, destination)
            _run(extract_cradio_features, config, destination)


if __name__ == "__main__":
    main()
