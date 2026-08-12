from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .anima500k import download_anima500k_human, extract_anima500k_human
from .anima_cache import (
    cache_all_anima_inputs,
    cache_anima_text_conditions,
    cache_anima_vae_latents,
    validate_anima_caches,
)
from .caption import create_anima_captions
from .config import load_config, output_dir
from .cradio import extract_cradio_features, extract_selected_style_features
from .dedup import deduplicate
from .deepghs import download_deepghs_candidates
from .download import download_candidates
from .feature_probe import evaluate_probe_features, extract_probe_features, run_feature_probe
from .metadata import select_candidates
from .tagger import tag_images
from .stylenet import (
    evaluate_stylenet_layer_features,
    extract_stylenet_layer_features,
    prepare_stylenet,
    run_stylenet_layer_benchmark,
)
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
        ("style-extract", "Cache selected production C-RADIO style features"),
        ("probe-extract", "Extract pooled C-RADIO layer candidates on a style subset"),
        ("probe-evaluate", "Evaluate multi-reference artist retrieval for probe features"),
        ("probe", "Extract and evaluate C-RADIO feature candidates"),
        ("tap-extract", "Cache spatial taps and SigLIP globals for the resampler experiment"),
        ("tap-train", "Train and evaluate configured tap-resampler variants"),
        ("tap-test", "Evaluate the validation-selected tap variant on meta-test artists"),
        ("tap-experiment", "Extract features, then train all tap-resampler variants"),
        ("stylenet-prepare", "Download and index the controlled StyleNet benchmark"),
        ("stylenet-extract", "Extract pooled C-RADIO layer features on StyleNet"),
        ("stylenet-evaluate", "Evaluate controlled StyleNet style ranking"),
        ("stylenet-benchmark", "Prepare, extract, and evaluate StyleNet layers"),
        ("anima-text-cache", "Cache packed post-LLM Anima text conditioning"),
        ("anima-latent-cache", "Cache packed Qwen-Image VAE latents"),
        ("anima-cache", "Cache all frozen Anima training inputs"),
        ("anima-cache-validate", "Validate packed Anima cache manifests and tensors"),
        ("style-train", "Train the multi-reference Anima style adapter"),
        ("style-smoke", "Run two real Anima style-adapter training steps"),
        ("style-benchmark", "Benchmark production style training batch sizes"),
        ("style-sample", "Render frozen-base and styled controls from the current checkpoint"),
        ("style-diagnose", "Measure correct, shuffled, null, and bypass style conditioning"),
        ("style-overfit", "Overfit a fixed exact-self batch to diagnose style-flow capacity"),
        ("style-calibrate", "Measure empirical Anima artist-tag velocity effect ranges"),
        ("synthetic-teacher", "Generate and fully cache artist-tag teacher images"),
        ("synthetic-teacher-benchmark", "Validate SPEED and batched VAE before production"),
        ("synthetic-teacher-kv", "Cache factorized native Anima K/V/O teacher targets"),
        ("synthetic-validate", "Validate synthetic corpus, artist effects, and fixed splits"),
        ("synthetic-query-probes", "Capture native Anima query probe bank"),
        ("synthetic-style-tokens", "Cache frozen per-reference Resampler tokens"),
        ("offline-kvo-smoke", "Run two offline K/V/O bootstrap steps"),
        ("offline-kvo-train", "Train and validate offline K/V/O connector bootstrap"),
        ("offline-kvo-phase-b", "Jointly tune the Resampler top and connector"),
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
    elif args.command == "style-extract":
        _run(extract_selected_style_features, config, destination)
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
    elif args.command == "stylenet-prepare":
        _run(prepare_stylenet, config, destination)
    elif args.command == "stylenet-extract":
        _run(extract_stylenet_layer_features, config, destination)
    elif args.command == "stylenet-evaluate":
        _run(evaluate_stylenet_layer_features, config, destination)
    elif args.command == "stylenet-benchmark":
        _run(run_stylenet_layer_benchmark, config, destination)
    elif args.command == "anima-text-cache":
        _run(cache_anima_text_conditions, config, destination)
    elif args.command == "anima-latent-cache":
        _run(cache_anima_vae_latents, config, destination)
    elif args.command == "anima-cache":
        _run(cache_all_anima_inputs, config, destination)
    elif args.command == "anima-cache-validate":
        _run(validate_anima_caches, config, destination)
    elif args.command in {
        "style-train", "style-smoke", "style-benchmark", "style-sample", "style-diagnose",
        "style-overfit",
        "style-calibrate",
    }:
        # Keep torch/sd-scripts optional for metadata-only commands.
        from .style_transfer import (
            benchmark_style_batches,
            diagnose_style_reference_dependence,
            overfit_exact_self_batch,
            sample_style_checkpoint,
            smoke_test_style_adapter,
            train_style_adapter,
        )
        from .style_calibration import calibrate_artist_tag_velocity

        stage = {
            "style-train": train_style_adapter,
            "style-smoke": smoke_test_style_adapter,
            "style-benchmark": benchmark_style_batches,
            "style-sample": sample_style_checkpoint,
            "style-diagnose": diagnose_style_reference_dependence,
            "style-overfit": overfit_exact_self_batch,
            "style-calibrate": calibrate_artist_tag_velocity,
        }[args.command]
        _run(stage, config, destination)
    elif args.command == "synthetic-teacher":
        from .synthetic_teacher import build_synthetic_teacher_cache

        _run(build_synthetic_teacher_cache, config, destination)
    elif args.command == "synthetic-teacher-benchmark":
        from .synthetic_teacher import benchmark_synthetic_teacher_cache

        _run(benchmark_synthetic_teacher_cache, config, destination)
    elif args.command == "synthetic-teacher-kv":
        from .synthetic_teacher import build_synthetic_teacher_kv_cache

        _run(build_synthetic_teacher_kv_cache, config, destination)
    elif args.command == "synthetic-validate":
        from .synthetic_bootstrap import validate_synthetic_teacher_corpus

        _run(validate_synthetic_teacher_corpus, config, destination)
    elif args.command == "synthetic-query-probes":
        from .synthetic_bootstrap import build_anima_query_probe_bank

        _run(build_anima_query_probe_bank, config, destination)
    elif args.command == "synthetic-style-tokens":
        from .synthetic_bootstrap import cache_synthetic_resampler_tokens

        _run(cache_synthetic_resampler_tokens, config, destination)
    elif args.command in {"offline-kvo-smoke", "offline-kvo-train", "offline-kvo-phase-b"}:
        from .synthetic_bootstrap import (
            smoke_offline_kvo_bootstrap,
            train_offline_kvo_bootstrap,
            train_offline_kvo_phase_b,
        )

        stage = {
            "offline-kvo-smoke": smoke_offline_kvo_bootstrap,
            "offline-kvo-train": train_offline_kvo_bootstrap,
            "offline-kvo-phase-b": train_offline_kvo_phase_b,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {"prepare", "all"}:
        for stage in (select_candidates, download_candidates, deduplicate):
            _run(stage, config, destination)
        if args.command == "all":
            _run(tag_images, config, destination)
            _run(create_anima_captions, config, destination)
            _run(extract_cradio_features, config, destination)


if __name__ == "__main__":
    main()
