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
        ("megastyle-download", "Download Tencent MegaStyle-1.4M Parquet shards"),
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
        ("style-token-cache", "Cache frozen production Resampler tokens"),
        ("style-smoke", "Run two real Anima style-adapter training steps"),
        ("style-benchmark", "Benchmark production style training batch sizes"),
        ("style-sample", "Render frozen-base and styled controls from the current checkpoint"),
        ("style-compare", "Render a fixed-panel and controlled-reference CFG comparison"),
        ("style-diagnose", "Measure correct, shuffled, null, and bypass style conditioning"),
        ("style-overfit", "Overfit a fixed exact-self batch to diagnose style-flow capacity"),
        ("style-overfit-sample", "Render every target from the completed exact-self overfit"),
        ("style-exact-self-generalize", "Train 96 targets and validate exact-self on 24 unseen targets"),
        ("style-exact-self-sample", "Render train and unseen targets from an exact-self checkpoint"),
        ("style-tokenizer-train", "Train native-context StyleTokenizer from frozen Resampler tokens"),
        ("style-tokenizer-generalize", "Train StyleTokenizer with target-excluded same-artist references"),
        ("style-tokenizer-generalize-smoke", "Smoke-test multi-reference StyleTokenizer auxiliaries"),
        ("style-tokenizer-lr-probe", "Resume an isolated high-LR StyleTokenizer branch"),
        ("style-tokenizer-lr10x-scratch", "Train the 10x-LR StyleTokenizer ablation from scratch"),
        ("style-tokenizer-select", "Re-evaluate and select a StyleTokenizer checkpoint"),
        ("style-tokenizer-export", "Export the selected StyleTokenizer inference bundle"),
        ("style-tokenizer-compare-artists", "Render different artists with one prompt and seed"),
        ("style-tokenizer-smoke", "Run two real Anima StyleTokenizer training steps"),
        ("query-style-tokenizer-train", "Jointly train the 32-slot query StyleTokenizer and Anima style K/V"),
        ("query-style-tokenizer-smoke", "Run two real Anima query StyleTokenizer training steps"),
        (
            "detail-style-teacher-context-cache",
            "Cache native content+artist post-LLM contexts for same-Q supervision",
        ),
        (
            "detail-style-train",
            "Train detail-preserving typed-slot Style Cross-Attention",
        ),
        (
            "detail-style-block-similarity",
            "Cluster Anima blocks for four shared Style K/V bases",
        ),
        (
            "detail-style-fixed-samples",
            "Backfill fixed TestSample1-7 panels from detail-style checkpoints",
        ),
        (
            "detail-style-smoke",
            "Run two real detail-preserving Style Cross-Attention steps",
        ),
        ("pure-token-style-tokenizer-train", "Train the 32-slot tokenizer by native text-context injection"),
        ("pure-token-style-tokenizer-smoke", "Smoke-test native-context pure token injection"),
        ("dual-query-resampler-train", "Pretrain the C-RADIO/Qwen-VAE dual-query Resampler"),
        ("dual-query-resampler-smoke", "Run two synthetic dual-query Resampler steps"),
        ("dual-query-style-cache", "Cache frozen Dual-query Resampler outputs"),
        ("dual-query-style-external-cache", "Cache the seven fixed external references"),
        (
            "native-centered-teacher-cache",
            "Cache centered native @artist velocity targets",
        ),
        (
            "dual-domain-centered-teacher-cache",
            "Cache 500-artist centered native flow targets",
        ),
        (
            "synthetic-dual-query-style-cache",
            "Cache Dual-query Resampler tokens for synthetic references",
        ),
        (
            "dual-domain-style-distill",
            "Train human and synthetic native-effect distillation from scratch",
        ),
        (
            "dual-domain-style-distill-smoke",
            "Smoke-test independent human and synthetic distillation",
        ),
        (
            "global-query-multimode-text-cache",
            "Cache Full/Dropout/Short quality and plain post-LLM conditions",
        ),
        (
            "global-query-multimode-train",
            "Train the typed-memory global-query Style Tokenizer",
        ),
        (
            "global-query-multimode-smoke",
            "Smoke-test the global-query multi-prompt training recipe",
        ),
        (
            "slot-preserving-global-query-train",
            "Train the slot-preserving artist-specific Style Tokenizer",
        ),
        (
            "slot-preserving-global-query-smoke",
            "Smoke-test slot identity and dense artist supervision",
        ),
        (
            "typed-multi-descriptor-train",
            "Train the compact typed multi-descriptor Style Tokenizer",
        ),
        (
            "typed-multi-descriptor-smoke",
            "Smoke-test typed descriptor pooling and projected supervision",
        ),
        (
            "single-stage-typed-attention-train",
            "Train the compact single-stage typed-attention Style Tokenizer",
        ),
        (
            "single-stage-typed-attention-smoke",
            "Smoke-test single-stage typed multi-reference attention",
        ),
        ("dual-query-style-ablate", "Compare artist-summary token delivery"),
        ("dual-query-style-train", "Train the multi-reference Dual-query Style Tokenizer"),
        ("dual-query-style-pilot", "Run the summary-ON 10k Dual-query Style Tokenizer pilot"),
        ("dual-query-style-exact-teacher", "Train the isolated 3k exact-self residual teacher"),
        (
            "dual-query-style-hierarchical-train",
            "Train the target-excluded hierarchical 16-token Style Tokenizer",
        ),
        (
            "dual-query-style-compact-train",
            "Train the small native-context tokenizer on Dual-query tokens",
        ),
        (
            "dual-query-style-compact-smoke",
            "Smoke-test the compact Dual-query Style Tokenizer",
        ),
        (
            "dual-query-style-compact-aligned-train",
            "Train the reference-balanced aligned compact tokenizer",
        ),
        (
            "dual-query-style-compact-aligned-smoke",
            "Smoke-test the aligned compact tokenizer recipe",
        ),
        (
            "dual-query-style-native-teacher-continue",
            "Continue the compact tokenizer on centered native artist effects",
        ),
        (
            "dual-query-style-native-teacher-smoke",
            "Smoke-test centered native artist-effect continuation",
        ),
        ("dual-query-style-pilot-smoke", "Exercise all 10k pilot loss branches on real caches"),
        ("dual-query-style-smoke", "Smoke-test the Dual-query Style Tokenizer"),
        ("style-calibrate", "Measure empirical Anima artist-tag velocity effect ranges"),
        ("synthetic-teacher", "Generate and fully cache artist-tag teacher images"),
        (
            "synthetic-reference-additional",
            "Generate the non-overlapping additional synthetic reference corpus",
        ),
        (
            "synthetic-reference-token-cache",
            "Directly cache C-RADIO plus frozen Resampler synthetic tokens",
        ),
        ("synthetic-teacher-benchmark", "Validate SPEED and batched VAE before production"),
        ("synthetic-teacher-kv", "Cache factorized native Anima K/V/O teacher targets"),
        ("real-artist-teacher-kv", "Cache 5k real-artist text and native K/V/O teacher targets"),
        ("real-artist-style-tokens", "Cache Resampler tokens for fixed 5k-artist references"),
        ("real-artist-offline-kvo", "Train reference-discriminative K/V/O head on 5k real artists"),
        ("real-artist-capacity-probe", "Overexpose 64 train artists to diagnose head capacity"),
        ("real-artist-offline-smoke", "Run two 5k real-artist offline training steps"),
        ("synthetic-validate", "Validate synthetic corpus, artist effects, and fixed splits"),
        ("synthetic-query-probes", "Capture native Anima query probe bank"),
        ("synthetic-style-tokens", "Cache frozen per-reference Resampler tokens"),
        ("offline-kvo-smoke", "Run two offline K/V/O bootstrap steps"),
        ("offline-kvo-capacity-probe", "Overexpose 64 synthetic train artists to test A0 capacity"),
        ("offline-kvo-train", "Train and validate offline K/V/O connector bootstrap"),
        ("offline-kvo-phase-b", "Jointly tune the Resampler top and connector"),
        ("offline-kvo-phase-b-smoke", "Run two Phase-B joint-tuning steps"),
        ("prepare", "Run selection, download, and duplicate removal"),
        ("megastyle-prepare", "Select and materialize the content-overlap 40k subset"),
        ("megastyle-cache", "Cache C-RADIO, VAE, Resampler and text for MegaStyle 40k"),
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
    elif args.command == "megastyle-download":
        from .megastyle import download_megastyle

        _run(download_megastyle, config, destination)
    elif args.command in {"megastyle-prepare", "megastyle-cache"}:
        from .megastyle import cache_megastyle_subset, prepare_megastyle_subset

        stage = {
            "megastyle-prepare": prepare_megastyle_subset,
            "megastyle-cache": cache_megastyle_subset,
        }[args.command]
        _run(stage, config, destination)
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
        "style-train", "style-token-cache", "style-smoke", "style-benchmark", "style-sample", "style-compare", "style-diagnose",
        "style-overfit", "style-overfit-sample", "style-exact-self-generalize",
        "style-exact-self-sample",
        "style-calibrate",
    }:
        # Keep torch/sd-scripts optional for metadata-only commands.
        from .style_transfer import (
            benchmark_style_batches,
            cache_production_resampler_tokens,
            diagnose_style_reference_dependence,
            overfit_exact_self_batch,
            sample_exact_self_overfit_checkpoint,
            sample_exact_self_generalization_checkpoint,
            compare_style_checkpoint_samples,
            train_exact_self_generalization,
            sample_style_checkpoint,
            smoke_test_style_adapter,
            train_style_adapter,
        )
        from .style_calibration import calibrate_artist_tag_velocity

        stage = {
            "style-train": train_style_adapter,
            "style-token-cache": cache_production_resampler_tokens,
            "style-smoke": smoke_test_style_adapter,
            "style-benchmark": benchmark_style_batches,
            "style-sample": sample_style_checkpoint,
            "style-compare": compare_style_checkpoint_samples,
            "style-diagnose": diagnose_style_reference_dependence,
            "style-overfit": overfit_exact_self_batch,
            "style-overfit-sample": sample_exact_self_overfit_checkpoint,
            "style-exact-self-generalize": train_exact_self_generalization,
            "style-exact-self-sample": sample_exact_self_generalization_checkpoint,
            "style-calibrate": calibrate_artist_tag_velocity,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {
        "style-tokenizer-train", "style-tokenizer-generalize",
        "style-tokenizer-lr-probe",
        "style-tokenizer-lr10x-scratch",
        "style-tokenizer-select",
        "style-tokenizer-export",
        "style-tokenizer-compare-artists",
        "style-tokenizer-smoke", "style-tokenizer-generalize-smoke",
    }:
        from .style_tokenizer import (
            smoke_test_style_tokenizer,
            smoke_test_style_tokenizer_generalization,
            export_style_tokenizer_checkpoint,
            compare_style_tokenizer_artists,
            select_style_tokenizer_checkpoint,
            train_style_tokenizer,
            train_style_tokenizer_generalization,
            train_style_tokenizer_lr_probe,
            train_style_tokenizer_lr10x_scratch,
        )

        stage = {
            "style-tokenizer-train": train_style_tokenizer,
            "style-tokenizer-generalize": train_style_tokenizer_generalization,
            "style-tokenizer-lr-probe": train_style_tokenizer_lr_probe,
            "style-tokenizer-lr10x-scratch": train_style_tokenizer_lr10x_scratch,
            "style-tokenizer-select": select_style_tokenizer_checkpoint,
            "style-tokenizer-export": export_style_tokenizer_checkpoint,
            "style-tokenizer-compare-artists": compare_style_tokenizer_artists,
            "style-tokenizer-smoke": smoke_test_style_tokenizer,
            "style-tokenizer-generalize-smoke": smoke_test_style_tokenizer_generalization,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {
        "detail-style-teacher-context-cache",
        "detail-style-train",
        "detail-style-block-similarity",
        "detail-style-fixed-samples",
        "detail-style-smoke",
    }:
        from .detail_style_teacher_context import (
            cache_detail_style_teacher_contexts,
        )
        from .detail_style_training import (
            backfill_detail_style_fixed_samples,
            smoke_test_detail_style_cross_attention,
            train_detail_style_cross_attention,
        )
        from .block_similarity import analyze_anima_block_similarity

        stage = {
            "detail-style-teacher-context-cache": (
                cache_detail_style_teacher_contexts
            ),
            "detail-style-train": train_detail_style_cross_attention,
            "detail-style-block-similarity": analyze_anima_block_similarity,
            "detail-style-fixed-samples": backfill_detail_style_fixed_samples,
            "detail-style-smoke": smoke_test_detail_style_cross_attention,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {
        "query-style-tokenizer-train", "query-style-tokenizer-smoke",
    }:
        from .query_style_tokenizer import (
            smoke_test_query_style_tokenizer,
            train_query_style_tokenizer,
        )

        stage = {
            "query-style-tokenizer-train": train_query_style_tokenizer,
            "query-style-tokenizer-smoke": smoke_test_query_style_tokenizer,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {
        "pure-token-style-tokenizer-train",
        "pure-token-style-tokenizer-smoke",
    }:
        from .pure_token_injection import (
            smoke_test_pure_token_style_tokenizer,
            train_pure_token_style_tokenizer,
        )

        stage = {
            "pure-token-style-tokenizer-train": train_pure_token_style_tokenizer,
            "pure-token-style-tokenizer-smoke": smoke_test_pure_token_style_tokenizer,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {
        "dual-query-resampler-train",
        "dual-query-resampler-smoke",
    }:
        from .dual_query_training import (
            smoke_test_dual_query_resampler,
            train_dual_query_resampler,
        )

        stage = {
            "dual-query-resampler-train": train_dual_query_resampler,
            "dual-query-resampler-smoke": smoke_test_dual_query_resampler,
        }[args.command]
        _run(stage, config, destination)
    elif args.command in {
        "dual-query-style-cache",
        "dual-query-style-external-cache",
        "native-centered-teacher-cache",
        "dual-domain-centered-teacher-cache",
        "synthetic-dual-query-style-cache",
        "dual-domain-style-distill",
        "dual-domain-style-distill-smoke",
        "global-query-multimode-text-cache",
        "global-query-multimode-train",
        "global-query-multimode-smoke",
        "slot-preserving-global-query-train",
        "slot-preserving-global-query-smoke",
        "typed-multi-descriptor-train",
        "typed-multi-descriptor-smoke",
        "single-stage-typed-attention-train",
        "single-stage-typed-attention-smoke",
        "dual-query-style-ablate",
        "dual-query-style-train",
        "dual-query-style-pilot",
        "dual-query-style-exact-teacher",
        "dual-query-style-hierarchical-train",
        "dual-query-style-compact-train",
        "dual-query-style-compact-smoke",
        "dual-query-style-compact-aligned-train",
        "dual-query-style-compact-aligned-smoke",
        "dual-query-style-native-teacher-continue",
        "dual-query-style-native-teacher-smoke",
        "dual-query-style-pilot-smoke",
        "dual-query-style-smoke",
    }:
        from .dual_query_style_tokenizer import (
            cache_dual_query_style_tokens,
            cache_synthetic_dual_query_style_tokens,
        )
        from .dual_query_external_samples import cache_dual_query_external_references
        from .global_query_style_tokenizer import (
            cache_global_query_multimode_text,
        )
        from .native_centered_teacher import (
            cache_dual_domain_centered_teacher,
            cache_native_centered_teacher,
        )
        from .dual_query_style_training import (
            compare_artist_summary_tokens,
            smoke_test_dual_query_style_tokenizer,
            smoke_test_dual_query_style_tokenizer_pilot,
            train_dual_query_style_tokenizer,
            train_dual_query_exact_self_teacher,
            train_hierarchical_dual_query_style_tokenizer,
            train_compact_dual_query_style_tokenizer,
            smoke_test_compact_dual_query_style_tokenizer,
            train_aligned_compact_dual_query_style_tokenizer,
            smoke_test_aligned_compact_dual_query_style_tokenizer,
            train_native_teacher_compact_continuation,
            smoke_test_native_teacher_compact_continuation,
            train_dual_query_style_tokenizer_pilot,
            train_dual_domain_native_distillation,
            smoke_test_dual_domain_native_distillation,
            train_global_query_multimode_style_tokenizer,
            smoke_test_global_query_multimode_style_tokenizer,
            train_slot_preserving_global_query_style_tokenizer,
            smoke_test_slot_preserving_global_query_style_tokenizer,
            train_typed_multi_descriptor_style_tokenizer,
            smoke_test_typed_multi_descriptor_style_tokenizer,
            train_single_stage_typed_attention_style_tokenizer,
            smoke_test_single_stage_typed_attention_style_tokenizer,
        )

        stage = {
            "dual-query-style-cache": cache_dual_query_style_tokens,
            "dual-query-style-external-cache": cache_dual_query_external_references,
            "native-centered-teacher-cache": cache_native_centered_teacher,
            "dual-domain-centered-teacher-cache": (
                cache_dual_domain_centered_teacher
            ),
            "synthetic-dual-query-style-cache": (
                cache_synthetic_dual_query_style_tokens
            ),
            "dual-domain-style-distill": train_dual_domain_native_distillation,
            "dual-domain-style-distill-smoke": (
                smoke_test_dual_domain_native_distillation
            ),
            "global-query-multimode-text-cache": (
                cache_global_query_multimode_text
            ),
            "global-query-multimode-train": (
                train_global_query_multimode_style_tokenizer
            ),
            "global-query-multimode-smoke": (
                smoke_test_global_query_multimode_style_tokenizer
            ),
            "slot-preserving-global-query-train": (
                train_slot_preserving_global_query_style_tokenizer
            ),
            "slot-preserving-global-query-smoke": (
                smoke_test_slot_preserving_global_query_style_tokenizer
            ),
            "typed-multi-descriptor-train": (
                train_typed_multi_descriptor_style_tokenizer
            ),
            "typed-multi-descriptor-smoke": (
                smoke_test_typed_multi_descriptor_style_tokenizer
            ),
            "single-stage-typed-attention-train": (
                train_single_stage_typed_attention_style_tokenizer
            ),
            "single-stage-typed-attention-smoke": (
                smoke_test_single_stage_typed_attention_style_tokenizer
            ),
            "dual-query-style-ablate": compare_artist_summary_tokens,
            "dual-query-style-train": train_dual_query_style_tokenizer,
            "dual-query-style-pilot": train_dual_query_style_tokenizer_pilot,
            "dual-query-style-exact-teacher": train_dual_query_exact_self_teacher,
            "dual-query-style-hierarchical-train": (
                train_hierarchical_dual_query_style_tokenizer
            ),
            "dual-query-style-compact-train": (
                train_compact_dual_query_style_tokenizer
            ),
            "dual-query-style-compact-smoke": (
                smoke_test_compact_dual_query_style_tokenizer
            ),
            "dual-query-style-compact-aligned-train": (
                train_aligned_compact_dual_query_style_tokenizer
            ),
            "dual-query-style-compact-aligned-smoke": (
                smoke_test_aligned_compact_dual_query_style_tokenizer
            ),
            "dual-query-style-native-teacher-continue": (
                train_native_teacher_compact_continuation
            ),
            "dual-query-style-native-teacher-smoke": (
                smoke_test_native_teacher_compact_continuation
            ),
            "dual-query-style-pilot-smoke": smoke_test_dual_query_style_tokenizer_pilot,
            "dual-query-style-smoke": smoke_test_dual_query_style_tokenizer,
        }[args.command]
        _run(stage, config, destination)
    elif args.command == "synthetic-teacher":
        from .synthetic_teacher import build_synthetic_teacher_cache

        _run(build_synthetic_teacher_cache, config, destination)
    elif args.command == "synthetic-reference-additional":
        from .synthetic_teacher import build_additional_synthetic_reference_images

        _run(build_additional_synthetic_reference_images, config, destination)
    elif args.command == "synthetic-reference-token-cache":
        from .synthetic_reference_pipeline import (
            cache_additional_synthetic_dual_query_tokens,
        )

        _run(cache_additional_synthetic_dual_query_tokens, config, destination)
    elif args.command == "synthetic-teacher-benchmark":
        from .synthetic_teacher import benchmark_synthetic_teacher_cache

        _run(benchmark_synthetic_teacher_cache, config, destination)
    elif args.command == "synthetic-teacher-kv":
        from .synthetic_teacher import build_synthetic_teacher_kv_cache

        _run(build_synthetic_teacher_kv_cache, config, destination)
    elif args.command == "real-artist-teacher-kv":
        from .synthetic_teacher import build_real_artist_teacher_kv_cache

        _run(build_real_artist_teacher_kv_cache, config, destination)
    elif args.command == "real-artist-style-tokens":
        from .synthetic_bootstrap import cache_real_artist_resampler_tokens

        _run(cache_real_artist_resampler_tokens, config, destination)
    elif args.command in {
        "real-artist-offline-kvo", "real-artist-offline-smoke",
        "real-artist-capacity-probe",
    }:
        from .synthetic_bootstrap import train_offline_kvo_bootstrap

        _run(
            lambda cfg, out: train_offline_kvo_bootstrap(
                cfg,
                out,
                phase="b",
                real_artist=True,
                capacity_probe=args.command == "real-artist-capacity-probe",
                steps_override=(
                    2 if args.command == "real-artist-offline-smoke" else None
                ),
            ),
            config,
            destination,
        )
    elif args.command == "synthetic-validate":
        from .synthetic_bootstrap import validate_synthetic_teacher_corpus

        _run(validate_synthetic_teacher_corpus, config, destination)
    elif args.command == "synthetic-query-probes":
        from .synthetic_bootstrap import build_anima_query_probe_bank

        _run(build_anima_query_probe_bank, config, destination)
    elif args.command == "synthetic-style-tokens":
        from .synthetic_bootstrap import cache_synthetic_resampler_tokens

        _run(cache_synthetic_resampler_tokens, config, destination)
    elif args.command in {"offline-kvo-smoke", "offline-kvo-capacity-probe", "offline-kvo-train", "offline-kvo-phase-b", "offline-kvo-phase-b-smoke"}:
        from .synthetic_bootstrap import (
            smoke_offline_kvo_bootstrap,
            smoke_offline_kvo_phase_b,
            train_offline_kvo_bootstrap,
            train_offline_kvo_phase_b,
        )

        stage = {
            "offline-kvo-smoke": smoke_offline_kvo_bootstrap,
            "offline-kvo-capacity-probe": lambda config, destination: train_offline_kvo_bootstrap(
                config, destination, capacity_probe=True
            ),
            "offline-kvo-train": train_offline_kvo_bootstrap,
            "offline-kvo-phase-b": train_offline_kvo_phase_b,
            "offline-kvo-phase-b-smoke": smoke_offline_kvo_phase_b,
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
