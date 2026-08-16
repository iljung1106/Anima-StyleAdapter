from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.dual_query_style_tokenizer import (  # noqa: E402
    DualQuerySetStyleTokenizer,
)
from anima_style_data.dual_query_style_training import (  # noqa: E402
    _artist_flow_ranking_loss,
    _bounded_aligned_effect_loss,
    _centered_artist_effect_loss,
    _common_output_loss,
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
