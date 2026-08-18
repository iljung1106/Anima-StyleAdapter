from anima_style_data.wandb_metric_schema import (
    compact_training_metrics,
    compact_validation_metrics,
)


def test_training_schema_keeps_core_metrics_and_drops_disabled_diagnostics():
    raw = {
        "loss": 1.2,
        "flow_loss": 0.8,
        "flow_loss_unweighted": 0.9,
        "paired_flow_improvement": 0.03,
        "style_token_rms": 0.7,
        "human_teacher_cosine": 0.4,
        "human_teacher_projection_coefficient": 0.2,
        "human_teacher_timestep_weighted_loss": 0.1,
        "human_teacher_common_output_ratio": 0.6,
        "human_teacher_artist_retrieval_top1": 0.75,
        "synthetic_teacher_cosine": 0.5,
        "artist_direction_weight": 0.0,
        "artist_flow_improvement_advantage": 0.8,
        "reconstruction_loss": 99.0,
        "functional_probe_per_step_loss": 99.0,
        "step_s": 0.42,
    }

    compact = compact_training_metrics(raw)

    assert compact["train_objective/total"] == 1.2
    assert compact["train_flow/paired_improvement"] == 0.03
    assert compact["train_teacher_human/direction_cosine"] == 0.4
    assert compact["train_teacher_human/common_output_ratio"] == 0.6
    assert compact["train_teacher_human/artist_retrieval_top1"] == 0.75
    assert compact["train_teacher_synthetic/direction_cosine"] == 0.5
    assert compact["system/step_seconds"] == 0.42
    assert not any(key.startswith("train_ranking/") for key in compact)
    assert "reconstruction_loss" not in compact
    assert "functional_probe_per_step_loss" not in compact


def test_training_schema_exposes_ranking_only_after_its_ramp_starts():
    compact = compact_training_metrics(
        {
            "artist_direction_weight": 0.00025,
            "artist_direction_weighted_loss": 0.01,
            "artist_correct_direction_cosine": 0.4,
            "artist_wrong_direction_cosine": 0.1,
            "artist_flow_improvement_advantage": 0.02,
        }
    )

    assert compact["train_schedule/ranking_weight"] == 0.00025
    assert compact["train_ranking/correct_direction_cosine"] == 0.4
    assert compact["train_ranking/flow_improvement_advantage"] == 0.02


def test_validation_schema_groups_flow_teacher_reference_and_controlled_metrics():
    flow = {
        "flow_loss": 0.7,
        "paired_flow_improvement": 0.04,
        "style_flow_direction_cosine": 0.3,
        "elapsed_s": 12.0,
    }
    row = {
        "validation_heldout": flow,
        "validation_self": flow,
        "validation_wrong_artist": flow,
        "selection_score": 0.05,
        "correct_vs_wrong_paired_advantage": 0.01,
        "validation_human_teacher": {
            "references_1/human_teacher_cosine": 0.2,
            "references_4/human_teacher_projection_coefficient": 0.4,
            "references_4/human_teacher_probe_position": 17,
        },
        "validation_synthetic_teacher": {
            "references_2/synthetic_teacher_student_to_target_rms": 0.8,
        },
        "reference_count_evaluation": {
            "1": {"paired_flow_improvement": 0.01},
            "8": {"style_flow_direction_cosine": 0.5},
        },
        "controlled_artist_consistency": {
            "within_between_margin": 0.3,
            "common_output_ratio": 0.6,
            "artists": 4,
        },
    }

    compact = compact_validation_metrics(row)

    assert compact["val_flow_heldout/paired_improvement"] == 0.04
    assert compact["val_summary/selection_score"] == 0.05
    assert compact["val_teacher_human/direction_cosine_r1"] == 0.2
    assert compact["val_teacher_human/projection_coefficient_r4"] == 0.4
    assert compact["val_teacher_synthetic/student_to_target_rms_r2"] == 0.8
    assert compact["val_reference_sweep/paired_improvement_r1"] == 0.01
    assert compact["val_controlled/common_output_ratio"] == 0.6
    assert not any("elapsed" in key for key in compact)
    assert not any("probe_position" in key for key in compact)
    assert "val_controlled/artists" not in compact
