from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ScalarMetrics = Mapping[str, float | int]


def _copy_metrics(
    destination: dict[str, float | int],
    source: ScalarMetrics,
    mapping: tuple[tuple[str, str], ...],
) -> None:
    for source_name, destination_name in mapping:
        if source_name in source:
            destination[destination_name] = source[source_name]


_TRAIN_OBJECTIVE = (
    ("loss", "train_objective/total"),
    ("flow_loss", "train_objective/flow_weighted"),
    ("flow_loss_unweighted", "train_objective/flow_unweighted"),
    ("total_auxiliary_weighted_loss", "train_objective/auxiliary_weighted"),
)

_TRAIN_FLOW = (
    ("base_flow_loss", "train_flow/base_mse"),
    ("paired_flow_improvement", "train_flow/paired_improvement"),
    ("paired_positive_fraction", "train_flow/positive_fraction"),
    ("style_flow_direction_cosine", "train_flow/direction_cosine"),
    (
        "style_flow_delta_to_desired_ratio",
        "train_flow/delta_to_desired_rms",
    ),
    (
        "style_flow_orthogonal_to_desired_ratio",
        "train_flow/orthogonal_to_desired_rms",
    ),
    ("style_output_ratio", "train_flow/output_to_base_rms"),
)

_TRAIN_STYLE = (
    ("style_token_rms", "train_style/token_rms"),
    ("style_token_sample_rms_std", "train_style/sample_rms_std"),
    ("style_token_slot_rms_std", "train_style/slot_rms_std"),
    ("style_output_gain_mean", "train_style/output_gain_mean"),
    ("style_output_gain_std", "train_style/output_gain_std"),
)

_TRAIN_BATCH = (
    ("references", "train_batch/references_mean"),
    ("reference_count_1_fraction", "train_batch/reference_1_fraction"),
    ("reference_count_2_fraction", "train_batch/reference_2_fraction"),
    ("target_inclusion", "train_batch/target_inclusion"),
    ("prompt_mode_full_fraction", "train_batch/prompt_full_fraction"),
    (
        "prompt_mode_tag_dropout_fraction",
        "train_batch/prompt_tag_dropout_fraction",
    ),
    ("prompt_mode_short_fraction", "train_batch/prompt_short_fraction"),
    ("prompt_mode_empty_fraction", "train_batch/prompt_empty_fraction"),
    ("prompt_quality_fraction", "train_batch/quality_prefix_fraction"),
)

_TRAIN_SCHEDULE = (
    ("learning_rate", "train_schedule/learning_rate"),
    ("timestep_mean", "train_schedule/flow_timestep_mean"),
    ("flow_timestep_weight_mean", "train_schedule/flow_weight_mean"),
    ("flow_timestep_weight_min", "train_schedule/flow_weight_min"),
    ("flow_timestep_weight_max", "train_schedule/flow_weight_max"),
    ("dual_domain_teacher_every", "train_schedule/teacher_every"),
    (
        "dual_domain_teacher_gradient_scale",
        "train_schedule/teacher_gradient_scale",
    ),
    ("artist_direction_weight", "train_schedule/ranking_weight"),
)

_TRAIN_SYSTEM = (
    ("grad_norm", "system/gradient_norm"),
    ("step_s", "system/step_seconds"),
)

_RANKING = (
    ("artist_direction_weighted_loss", "train_ranking/weighted_loss"),
    ("artist_direction_ranking_loss", "train_ranking/direction_loss"),
    ("artist_flow_ranking_loss", "train_ranking/flow_loss"),
    (
        "artist_correct_direction_cosine",
        "train_ranking/correct_direction_cosine",
    ),
    (
        "artist_wrong_direction_cosine",
        "train_ranking/wrong_direction_cosine",
    ),
    (
        "artist_centered_direction_cosine",
        "train_ranking/centered_direction_cosine",
    ),
    (
        "artist_correct_flow_improvement",
        "train_ranking/correct_flow_improvement",
    ),
    (
        "artist_wrong_flow_improvement",
        "train_ranking/wrong_flow_improvement",
    ),
    (
        "artist_flow_improvement_advantage",
        "train_ranking/flow_improvement_advantage",
    ),
)


def _teacher_training_metrics(
    source: ScalarMetrics,
    *,
    source_prefix: str,
    section: str,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    suffixes = (
        ("timestep_weighted_loss", "objective_weighted"),
        ("timestep_unweighted_loss", "objective_unweighted"),
        ("cosine", "direction_cosine"),
        ("projection_coefficient", "projection_coefficient"),
        ("orthogonal_ratio", "orthogonal_ratio"),
        ("student_to_target_rms", "student_to_target_rms"),
        ("student_rms", "student_rms"),
        ("target_rms", "target_rms"),
        ("common_output_ratio", "common_output_ratio"),
        ("artist_retrieval_top1", "artist_retrieval_top1"),
        ("artist_positive_cosine", "artist_positive_cosine"),
        (
            "artist_hard_negative_cosine",
            "artist_hard_negative_cosine",
        ),
        ("reference_count", "references"),
        ("timestep", "timestep"),
        ("timestep_weight", "timestep_weight"),
    )
    for suffix, label in suffixes:
        source_name = f"{source_prefix}_{suffix}"
        if source_name in source:
            result[f"{section}/{label}"] = source[source_name]
    return result


def compact_training_metrics(metrics: ScalarMetrics) -> dict[str, float | int]:
    """Return the small, stable W&B schema used by StyleTokenizer training.

    The trainer still writes every raw diagnostic to its local JSON history.
    This function intentionally exposes only metrics that answer a concrete
    training question and groups them into dashboard-friendly sections.
    """

    result: dict[str, float | int] = {}
    for mapping in (
        _TRAIN_OBJECTIVE,
        _TRAIN_FLOW,
        _TRAIN_STYLE,
        _TRAIN_BATCH,
        _TRAIN_SCHEDULE,
        _TRAIN_SYSTEM,
    ):
        _copy_metrics(result, metrics, mapping)
    result.update(
        _teacher_training_metrics(
            metrics,
            source_prefix="human_teacher",
            section="train_teacher_human",
        )
    )
    result.update(
        _teacher_training_metrics(
            metrics,
            source_prefix="synthetic_teacher",
            section="train_teacher_synthetic",
        )
    )
    if float(metrics.get("artist_direction_weight", 0.0)) > 0.0:
        _copy_metrics(result, metrics, _RANKING)
    return result


_VALIDATION_FLOW = (
    ("flow_loss", "flow_mse"),
    ("paired_flow_improvement", "paired_improvement"),
    ("paired_positive_fraction", "positive_fraction"),
    ("style_flow_direction_cosine", "direction_cosine"),
    ("style_flow_delta_to_desired_ratio", "delta_to_desired_rms"),
    (
        "style_flow_orthogonal_to_desired_ratio",
        "orthogonal_to_desired_rms",
    ),
    ("style_output_ratio", "output_to_base_rms"),
)

_VALIDATION_TEACHER = (
    ("cosine", "direction_cosine"),
    ("projection_coefficient", "projection_coefficient"),
    ("orthogonal_ratio", "orthogonal_ratio"),
    ("student_to_target_rms", "student_to_target_rms"),
    ("timestep_weighted_loss", "objective_weighted"),
    ("timestep_unweighted_loss", "objective_unweighted"),
    ("common_output_ratio", "common_output_ratio"),
    ("artist_retrieval_top1", "artist_retrieval_top1"),
)


def _validation_flow_metrics(
    result: dict[str, float | int],
    values: ScalarMetrics,
    *,
    section: str,
) -> None:
    for source_name, label in _VALIDATION_FLOW:
        if source_name in values:
            result[f"{section}/{label}"] = values[source_name]


def _validation_teacher_metrics(
    result: dict[str, float | int],
    values: ScalarMetrics,
    *,
    domain: str,
    source_prefix: str,
) -> None:
    for reference_count in (1, 2, 4):
        for suffix, label in _VALIDATION_TEACHER:
            source_name = (
                f"references_{reference_count}/"
                f"{source_prefix}_{suffix}"
            )
            if source_name in values:
                result[
                    f"val_teacher_{domain}/{label}_r{reference_count}"
                ] = values[source_name]


def compact_validation_metrics(row: Mapping[str, Any]) -> dict[str, float | int]:
    """Flatten a validation row into a curated, readable W&B dashboard."""

    result: dict[str, float | int] = {}
    for row_name, section in (
        ("validation_heldout", "val_flow_heldout"),
        ("validation_self", "val_flow_self"),
        ("validation_wrong_artist", "val_flow_wrong"),
    ):
        values = row.get(row_name, {})
        if isinstance(values, Mapping):
            _validation_flow_metrics(result, values, section=section)

    for source_name, destination_name in (
        ("selection_score", "val_summary/selection_score"),
        (
            "correct_vs_wrong_paired_advantage",
            "val_summary/correct_vs_wrong_advantage",
        ),
    ):
        if source_name in row:
            result[destination_name] = row[source_name]

    for row_name, domain, source_prefix in (
        ("validation_human_teacher", "human", "human_teacher"),
        (
            "validation_synthetic_teacher",
            "synthetic",
            "synthetic_teacher",
        ),
        ("validation_native_teacher", "native", "native_teacher"),
    ):
        values = row.get(row_name, {})
        if isinstance(values, Mapping):
            _validation_teacher_metrics(
                result,
                values,
                domain=domain,
                source_prefix=source_prefix,
            )

    reference_sweep = row.get("reference_count_evaluation", {})
    if isinstance(reference_sweep, Mapping):
        for reference_count, values in reference_sweep.items():
            if not isinstance(values, Mapping):
                continue
            for source_name, label in (
                ("paired_flow_improvement", "paired_improvement"),
                ("style_flow_direction_cosine", "direction_cosine"),
                (
                    "style_flow_delta_to_desired_ratio",
                    "delta_to_desired_rms",
                ),
            ):
                if source_name in values:
                    result[
                        f"val_reference_sweep/{label}_r{reference_count}"
                    ] = values[source_name]

    controlled = row.get("controlled_artist_consistency", {})
    if isinstance(controlled, Mapping):
        for source_name in (
            "within_artist_centered_cosine",
            "between_artist_centered_cosine",
            "within_between_margin",
            "artist_retrieval_top1",
            "artist_retrieval_margin",
            "common_output_ratio",
            "reference_view_difference_ratio",
            "artist_effect_rms",
        ):
            if source_name in controlled:
                result[f"val_controlled/{source_name}"] = controlled[source_name]
    return result


def define_wandb_metric_summaries(run: Any) -> None:
    """Set useful min/max summaries without creating redundant charts."""

    for name, summary in (
        ("train_objective/total", "min"),
        ("train_flow/paired_improvement", "max"),
        ("val_summary/selection_score", "max"),
        ("val_flow_heldout/paired_improvement", "max"),
        ("val_summary/correct_vs_wrong_advantage", "max"),
        ("val_controlled/within_between_margin", "max"),
        ("val_controlled/artist_retrieval_top1", "max"),
        ("val_controlled/common_output_ratio", "min"),
    ):
        run.define_metric(name, summary=summary)
