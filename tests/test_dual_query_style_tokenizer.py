from __future__ import annotations

import pytest
from safetensors.torch import save_file

torch = pytest.importorskip("torch")

from anima_style_data.dual_query_style_tokenizer import (  # noqa: E402
    CachedTeacherReferenceLoader,
    DualQuerySetStyleTokenizer,
    _load_resampler,
)
from anima_style_data.dual_query_training import _model_from_config  # noqa: E402
from anima_style_data.io import write_records  # noqa: E402
from anima_style_data.dual_query_style_training import (  # noqa: E402
    _artist_flow_ranking_loss,
    _aligned_projection_target_loss,
    _bounded_aligned_effect_loss,
    _centered_artist_effect_loss,
    _common_output_loss,
    _native_teacher_alignment_loss,
    _pilot_alignment_state,
    _pilot_stage,
    _same_artist_functional_loss,
)


def _model(*, include_summary: bool) -> DualQuerySetStyleTokenizer:
    return DualQuerySetStyleTokenizer(
        dim=32,
        query_tokens=8,
        artist_summary_tokens=2,
        include_artist_summary=include_summary,
        output_tokens=4,
        heads=4,
        cross_layers=1,
        cross_slot_layers=1,
        ff_dim=64,
    ).eval()


def test_resampler_loader_uses_checkpoint_input_dimensions(tmp_path):
    resampler_cfg = {
        "model": {
            "semantic_layers": [18, 24],
            "dim": 32,
            "spatial_query_grid": 2,
            "global_queries": 2,
            "layers": 1,
            "heads": 4,
            "ff_dim": 64,
            "artist_descriptor_dim": 16,
            "artist_pooling_queries": 2,
            "artist_summary_tokens": 2,
            "semantic_dropout": 0.0,
            "vae_dropout": 0.0,
        },
        "training": {"artist_proxy_fraction": 0.0},
    }
    expected = _model_from_config(resampler_cfg, semantic_dim=12, vae_channels=4)
    checkpoint = tmp_path / "resampler.pt"
    torch.save(
        {
            "step": 17,
            "model": expected.state_dict(),
            "model_config": dict(resampler_cfg["model"]),
        },
        checkpoint,
    )

    loaded, step = _load_resampler(
        {"dual_query_resampler": resampler_cfg},
        tmp_path,
        checkpoint,
        semantic_dim=99,
        vae_channels=99,
        device="cpu",
    )

    assert step == 17
    assert loaded.semantic_norms["18"].normalized_shape == (12,)
    assert loaded.vae_stem[0].in_channels == 4


def test_teacher_reference_loader_can_use_available_style_intersection(tmp_path):
    rows = [
        {
            "id": index,
            "style_id": "human:available",
            "split": "train",
            "token_shard": "part-00000.safetensors",
            "token_row": index,
        }
        for index in range(4)
    ]
    write_records(tmp_path / "manifest.parquet", rows)
    save_file(
        {"tokens": torch.randn(4, 3, 8)},
        tmp_path / "part-00000.safetensors",
    )

    loader = CachedTeacherReferenceLoader(
        tmp_path,
        split="train",
        style_ids=["human:available", "human:missing"],
        batch_size=1,
        references=2,
        seed=7,
        strict_style_ids=False,
    )
    batch = loader.load_step(0)
    assert batch["episodes"][0].style_id == "human:available"
    assert batch["cached_reference_tokens"].shape == (2, 3, 8)

    with pytest.raises(RuntimeError, match="missing 1 teacher artists"):
        CachedTeacherReferenceLoader(
            tmp_path,
            split="train",
            style_ids=["human:available", "human:missing"],
            batch_size=1,
            references=2,
            seed=7,
        )


def test_teacher_reference_loader_filters_disjoint_image_ids(tmp_path):
    rows = [
        {
            "id": index,
            "style_id": "human:artist",
            "split": "train",
            "token_shard": "part-00000.safetensors",
            "token_row": index,
        }
        for index in range(6)
    ]
    write_records(tmp_path / "manifest.parquet", rows)
    save_file(
        {"tokens": torch.arange(6).reshape(6, 1, 1).float()},
        tmp_path / "part-00000.safetensors",
    )
    loader = CachedTeacherReferenceLoader(
        tmp_path,
        split="train",
        style_ids=["human:artist"],
        batch_size=1,
        references=2,
        seed=13,
        allowed_image_ids={4, 5},
    )

    batch = loader.load_step(0)

    assert set(batch["episodes"][0].reference_ids) == {4, 5}
    assert set(batch["cached_reference_tokens"].flatten().tolist()) == {4.0, 5.0}


def test_teacher_reference_loader_combines_disjoint_cache_roots(tmp_path):
    roots = [tmp_path / "old", tmp_path / "additional"]
    styles = ["human:old", "human:new"]
    for root, style, offset in zip(roots, styles, (0, 100), strict=True):
        root.mkdir()
        write_records(
            root / "manifest.parquet",
            [
                {
                    "id": offset + index,
                    "style_id": style,
                    "split": "train",
                    "token_shard": "part-00000.safetensors",
                    "token_row": index,
                }
                for index in range(2)
            ],
        )
        save_file(
            {"tokens": torch.full((2, 3, 8), float(offset + 1))},
            root / "part-00000.safetensors",
        )

    loader = CachedTeacherReferenceLoader(
        roots,
        split="train",
        style_ids=styles,
        batch_size=2,
        references=2,
        seed=7,
    )
    batch = loader.load_step(0)

    assert {episode.style_id for episode in batch["episodes"]} == set(styles)
    assert set(batch["cached_reference_tokens"][:, 0, 0].tolist()) == {1.0, 101.0}


def test_teacher_reference_loader_samples_configured_reference_counts(tmp_path):
    rows = [
        {
            "id": style_index * 10 + image_index,
            "style_id": f"human:{style_index}",
            "split": "train",
            "token_shard": "part-00000.safetensors",
            "token_row": style_index * 4 + image_index,
        }
        for style_index in range(2)
        for image_index in range(4)
    ]
    write_records(tmp_path / "manifest.parquet", rows)
    save_file(
        {"tokens": torch.randn(8, 3, 8)},
        tmp_path / "part-00000.safetensors",
    )

    single = CachedTeacherReferenceLoader(
        tmp_path,
        split="train",
        style_ids=["human:0", "human:1"],
        batch_size=2,
        references=4,
        reference_count_weights=[1.0, 0.0, 0.0, 0.0],
        seed=11,
    ).load_step(0)
    assert single["reference_count"] == 1
    assert single["reference_mask"].shape == (2, 1)
    assert single["cached_reference_tokens"].shape == (2, 3, 8)

    four = CachedTeacherReferenceLoader(
        tmp_path,
        split="train",
        style_ids=["human:0", "human:1"],
        batch_size=2,
        references=4,
        reference_count_weights=[0.0, 0.0, 0.0, 1.0],
        seed=11,
    ).load_step(0)
    assert four["reference_count"] == 4
    assert four["reference_mask"].shape == (2, 4)
    assert four["cached_reference_tokens"].shape == (8, 3, 8)

    overridden = CachedTeacherReferenceLoader(
        tmp_path,
        split="train",
        style_ids=["human:0", "human:1"],
        batch_size=2,
        references=4,
        reference_count_weights=[0.0, 0.0, 0.0, 1.0],
        seed=11,
    ).load_step(0, reference_count_weights=[1.0, 0.0, 0.0, 0.0])
    assert overridden["reference_count"] == 1
    assert overridden["reference_mask"].shape == (2, 1)


def test_reference_set_is_order_invariant():
    torch.manual_seed(31)
    model = _model(include_summary=True)
    references = torch.randn(2, 3, 10, 32)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    permutation = torch.tensor([2, 0, 1])

    expected = model(references, mask).tokens
    actual = model(references[:, permutation], mask[:, permutation]).tokens

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_masked_reference_does_not_change_output():
    torch.manual_seed(37)
    model = _model(include_summary=True)
    references = torch.randn(1, 3, 10, 32)
    mask = torch.tensor([[True, True, False]])
    expected = model(references, mask).tokens
    references[:, 2].normal_(mean=1000, std=100)

    actual = model(references, mask).tokens

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_summary_ablation_changes_only_summary_dependency():
    torch.manual_seed(41)
    without = _model(include_summary=False)
    with_summary = _model(include_summary=True)
    with_summary.load_state_dict(without.state_dict())
    references = torch.randn(2, 2, 10, 32)
    mask = torch.ones(2, 2, dtype=torch.bool)

    query_only_before = without(references, mask).tokens
    summary_before = with_summary(references, mask).tokens
    references[:, :, 8:].add_(100)
    query_only_after = without(references, mask).tokens
    summary_after = with_summary(references, mask).tokens

    assert torch.equal(query_only_before, query_only_after)
    assert not torch.allclose(summary_before, summary_after)


def test_style_tokens_have_finite_gradient_and_configured_shape():
    model = _model(include_summary=True)
    references = torch.randn(3, 2, 10, 32, requires_grad=True)
    mask = torch.tensor([[True, False], [True, True], [True, True]])

    output = model(references, mask)
    output.tokens.square().mean().backward()

    assert output.tokens.shape == (3, 4, 32)
    assert torch.isfinite(output.tokens).all()
    assert torch.isfinite(references.grad).all()


def test_pilot_schedule_and_alignment_switch_at_documented_boundaries():
    training = {
        "steps": 10_000,
        "exact_self_end_step": 500,
        "reference_schedule": [
            {"name": "exact", "end_step": 500},
            {"name": "one_two", "end_step": 1000},
            {"name": "one_four", "end_step": 4000},
            {"name": "one_eight", "end_step": 10_000},
        ],
    }

    assert _pilot_stage(500, training)["name"] == "exact"
    assert _pilot_stage(501, training)["name"] == "one_two"
    assert _pilot_stage(4001, training)["name"] == "one_eight"
    assert _pilot_alignment_state(500, training)["coefficient_floor"] == pytest.approx(0.15)
    assert _pilot_alignment_state(501, training)["coefficient_floor"] == pytest.approx(0.03)
    assert _pilot_alignment_state(10_000, training)["coefficient_floor"] == pytest.approx(0.06)


def test_bounded_effect_rewards_in_range_alignment_and_penalizes_orthogonal_output():
    base = torch.zeros(2, 1, 2, 2)
    target = torch.ones_like(base)
    aligned = 0.1 * target
    orthogonal = aligned.clone()
    orthogonal[:, :, 0, 0] += 0.4
    orthogonal[:, :, 0, 1] -= 0.4
    orthogonal[:, :, 1, 0] += 0.4
    orthogonal[:, :, 1, 1] -= 0.4

    aligned_loss, _ = _bounded_aligned_effect_loss(
        aligned,
        base,
        target,
        minimum=0.05,
        maximum=0.20,
        orthogonal_maximum=0.12,
        orthogonal_weight=0.25,
        scale_floor=1e-4,
    )
    orthogonal_loss, metrics = _bounded_aligned_effect_loss(
        orthogonal,
        base,
        target,
        minimum=0.05,
        maximum=0.20,
        orthogonal_maximum=0.12,
        orthogonal_weight=0.25,
        scale_floor=1e-4,
    )

    assert aligned_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert orthogonal_loss > aligned_loss
    assert metrics["bounded_orthogonal_ratio"] > 0.12


def test_projection_target_matches_useful_residual_magnitude():
    base = torch.zeros(2, 1, 2, 2)
    target = torch.ones_like(base)

    exact_loss, exact_metrics = _aligned_projection_target_loss(
        target,
        base,
        target,
        coefficient_target=1.0,
        huber_beta=0.1,
        scale_floor=1e-4,
    )
    weak_loss, weak_metrics = _aligned_projection_target_loss(
        0.1 * target,
        base,
        target,
        coefficient_target=1.0,
        huber_beta=0.1,
        scale_floor=1e-4,
    )

    assert exact_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert exact_metrics["projection_target_coefficient"] == pytest.approx(1.0)
    assert weak_loss > exact_loss
    assert weak_metrics["projection_target_coefficient"] == pytest.approx(0.1)


def test_common_output_hinge_distinguishes_shared_and_centered_artist_effects():
    shared = torch.ones(4, 2, 2)
    centered = torch.stack(
        (torch.ones(2, 2), -torch.ones(2, 2), torch.eye(2), -torch.eye(2))
    )

    shared_loss, shared_metrics = _common_output_loss(shared, threshold=0.70)
    centered_loss, centered_metrics = _common_output_loss(centered, threshold=0.70)

    assert shared_metrics["common_output_ratio"] == pytest.approx(1.0)
    assert shared_loss > 0
    assert centered_metrics["common_output_ratio"] == pytest.approx(0.0, abs=1e-7)
    assert centered_loss.item() == pytest.approx(0.0, abs=1e-7)


def test_same_artist_functional_loss_matches_direction_and_magnitude():
    first = torch.stack((torch.ones(2, 2), torch.eye(2)))
    matching = first.clone()
    mismatching = torch.stack((-torch.ones(2, 2), 3.0 * torch.eye(2)))
    valid = torch.ones(2, dtype=torch.bool)

    matching_loss, matching_metrics = _same_artist_functional_loss(
        first,
        matching,
        valid,
        direction_fraction=0.75,
        huber_beta=0.10,
    )
    mismatching_loss, mismatching_metrics = _same_artist_functional_loss(
        first,
        mismatching,
        valid,
        direction_fraction=0.75,
        huber_beta=0.10,
    )

    assert matching_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert matching_metrics["functional_same_artist_cosine"] == pytest.approx(1.0)
    assert mismatching_loss > matching_loss
    assert mismatching_metrics["functional_same_artist_cosine"] < 1.0
    assert mismatching_metrics["functional_same_artist_log_rms_error"] > 0


def test_centered_effect_floor_rejects_a_shared_artist_output():
    shared = torch.ones(4, 2, 2)
    distinct = torch.stack(
        (torch.ones(2, 2), -torch.ones(2, 2), torch.eye(2), -torch.eye(2))
    )

    shared_loss, shared_metrics = _centered_artist_effect_loss(shared, floor=0.50)
    distinct_loss, distinct_metrics = _centered_artist_effect_loss(
        distinct, floor=0.50
    )

    assert shared_metrics["functional_centered_effect_ratio"] == pytest.approx(0.0)
    assert shared_loss > 0
    assert distinct_metrics["functional_centered_effect_ratio"] > 0.50
    assert distinct_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert distinct_metrics["functional_between_artist_cosine"] < 1.0


def test_artist_flow_ranking_prefers_the_correct_reference_without_moving_wrong():
    target = torch.ones(2, 1, 2, 2)
    base = torch.zeros_like(target)
    correct = torch.full_like(target, 0.2, requires_grad=True)
    wrong = torch.full_like(target, 0.4, requires_grad=True)

    loss, metrics = _artist_flow_ranking_loss(
        correct, wrong, base, target, margin=0.10
    )
    loss.backward()

    assert loss > 0
    assert metrics["artist_flow_improvement_advantage"] < 0
    assert correct.grad is not None and correct.grad.abs().sum() > 0
    assert torch.isfinite(correct.grad).all()
    assert wrong.grad is None


def test_native_teacher_alignment_requires_direction_and_absolute_magnitude():
    teacher = torch.stack((torch.ones(2, 2), -torch.eye(2)))
    matching = teacher.clone().requires_grad_(True)
    weak = (0.1 * teacher).requires_grad_(True)
    orthogonal = torch.flip(teacher, dims=(-1,)).requires_grad_(True)

    matching_loss, matching_metrics = _native_teacher_alignment_loss(
        matching,
        teacher,
        huber_beta=0.10,
        scale_floor=1e-4,
        direction_weight=0.10,
        magnitude_weight=0.05,
    )
    weak_loss, weak_metrics = _native_teacher_alignment_loss(
        weak,
        teacher,
        huber_beta=0.10,
        scale_floor=1e-4,
        direction_weight=0.10,
        magnitude_weight=0.05,
    )
    orthogonal_loss, orthogonal_metrics = _native_teacher_alignment_loss(
        orthogonal,
        teacher,
        huber_beta=0.10,
        scale_floor=1e-4,
        direction_weight=0.10,
        magnitude_weight=0.05,
    )

    assert matching_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert matching_metrics["native_teacher_projection_coefficient"] == pytest.approx(1.0)
    assert weak_loss > matching_loss
    assert weak_metrics["native_teacher_student_to_target_rms"] == pytest.approx(0.1)
    assert orthogonal_loss > matching_loss
    assert orthogonal_metrics["native_teacher_cosine"] < 1.0

    weak_loss.backward()
    assert weak.grad is not None and torch.isfinite(weak.grad).all()
