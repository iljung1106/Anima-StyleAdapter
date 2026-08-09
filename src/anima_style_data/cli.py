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
from .feature_probe import evaluate_probe_features, extract_probe_features, run_feature_probe
from .metadata import select_candidates
from .tagger import tag_images
from .tap_resampler import (
    evaluate_selected_tap_variant,
    extract_tap_features,
    run_tap_resampler_experiment,
    train_tap_resampler_variants,
)


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
        ("probe-extract", "Extract pooled C-RADIO layer candidates on a style subset"),
        ("probe-evaluate", "Evaluate multi-reference artist retrieval for probe features"),
        ("probe", "Extract and evaluate C-RADIO feature candidates"),
        ("tap-extract", "Cache spatial taps and SigLIP globals for the resampler experiment"),
        ("tap-train", "Train and evaluate configured tap-resampler variants"),
        ("tap-test", "Evaluate the validation-selected tap variant on meta-test artists"),
        ("tap-experiment", "Extract features, then train all tap-resampler variants"),
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
    elif args.command == "probe-extract":
        _run(extract_probe_features, config, destination)
    elif args.command == "probe-evaluate":
        _run(evaluate_probe_features, config, destination)
    elif args.command == "probe":
        _run(run_feature_probe, config, destination)
    elif args.command == "tap-extract":
        _run(extract_tap_features, config, destination)
    elif args.command == "tap-train":
        _run(train_tap_resampler_variants, config, destination)
    elif args.command == "tap-test":
        _run(evaluate_selected_tap_variant, config, destination)
    elif args.command == "tap-experiment":
        _run(run_tap_resampler_experiment, config, destination)
    elif args.command in {"prepare", "all"}:
        for stage in (select_candidates, download_candidates, deduplicate):
            _run(stage, config, destination)
        if args.command == "all":
            _run(tag_images, config, destination)
            _run(create_anima_captions, config, destination)
            _run(extract_cradio_features, config, destination)


if __name__ == "__main__":
    main()
