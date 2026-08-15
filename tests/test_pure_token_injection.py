from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.pure_token_injection import (
    _SamplingTokenView,
    _aligned_velocity_losses,
    _controlled_direction_metrics,
)
from anima_style_data.query_style_tokenizer import QueryStyleTokenizerV2


def test_aligned_floor_rejects_zero_and_orthogonal_velocity():
    base = torch.zeros(2, 1, 1, 2)
    target = torch.tensor([
        [[[1.0, 0.0]]],
        [[[1.0, 0.0]]],
    ])
    included = torch.ones(2, dtype=torch.bool)
    zero = torch.zeros_like(target, requires_grad=True)
    aligned = target * 0.25
    orthogonal = torch.tensor([
        [[[0.0, 0.25]]],
        [[[0.0, 0.25]]],
    ])

    zero_normalized, zero_floor, zero_metrics = _aligned_velocity_losses(
        zero, base, target, included,
        coefficient_floor=0.25, huber_beta=0.1, scale_floor=1e-4,
    )
    aligned_normalized, aligned_floor, aligned_metrics = _aligned_velocity_losses(
        aligned, base, target, included,
        coefficient_floor=0.25, huber_beta=0.1, scale_floor=1e-4,
    )
    _, orthogonal_floor, orthogonal_metrics = _aligned_velocity_losses(
        orthogonal, base, target, included,
        coefficient_floor=0.25, huber_beta=0.1, scale_floor=1e-4,
    )

    assert aligned_normalized < zero_normalized
    assert float(aligned_floor.detach()) == pytest.approx(0.0)
    assert float(zero_floor.detach()) == pytest.approx(0.25**2)
    assert float(orthogonal_floor.detach()) == pytest.approx(
        float(zero_floor.detach())
    )
    assert float(aligned_metrics["aligned_coefficient"]) == pytest.approx(0.25)
    assert float(zero_metrics["aligned_coefficient"]) == pytest.approx(0.0)
    assert float(orthogonal_metrics["aligned_coefficient"]) == pytest.approx(0.0)

    (zero_normalized + zero_floor).backward()
    assert float((zero.grad * target).sum()) < 0


def test_target_mask_anneals_direct_velocity_losses_with_inclusion_rate():
    base = torch.zeros(2, 1, 1, 2)
    target = torch.tensor([
        [[[1.0, 0.0]]],
        [[[1.0, 0.0]]],
    ])
    prediction = torch.zeros_like(target)
    all_rows = torch.ones(2, dtype=torch.bool)
    one_row = torch.tensor([True, False])

    all_normalized, all_floor, _ = _aligned_velocity_losses(
        prediction, base, target, all_rows,
        coefficient_floor=0.2, huber_beta=0.1, scale_floor=1e-4,
    )
    one_normalized, one_floor, metrics = _aligned_velocity_losses(
        prediction, base, target, one_row,
        coefficient_floor=0.2, huber_beta=0.1, scale_floor=1e-4,
    )

    assert float(one_normalized) == pytest.approx(float(all_normalized) * 0.5)
    assert float(one_floor) == pytest.approx(float(all_floor) * 0.5)
    assert float(metrics["target_auxiliary_fraction"]) == pytest.approx(0.5)


def test_sampling_view_exposes_only_high_capacity_final_tokens():
    tokenizer = QueryStyleTokenizerV2(
        source_dim=16,
        context_dim=16,
        source_tokens=8,
        output_tokens=4,
        heads=4,
        per_reference_layers=1,
        per_reference_ff_dim=32,
        cross_slot_layers=1,
        cross_slot_ff_dim=32,
        reconstruction_layers=1,
        reconstruction_ff_dim=32,
        reference_score_dim=8,
        output_rms_init=0.2,
    )
    view = _SamplingTokenView(tokenizer)
    references = torch.randn(2, 3, 8, 16)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    tokens = view(references, mask)

    assert tokens.shape == (2, 4, 16)
    assert torch.isfinite(tokens).all()


def test_controlled_direction_metrics_reward_repeatable_artist_effects():
    first = torch.tensor([
        [[[1.0, 0.0]]],
        [[[-1.0, 0.0]]],
    ])
    second = first.clone()

    metrics = _controlled_direction_metrics(first, second, ["a", "b"])

    assert float(metrics["within_artist_centered_cosine"]) == pytest.approx(1.0)
    assert float(metrics["between_artist_centered_cosine"]) == pytest.approx(-1.0)
    assert float(metrics["within_between_margin"]) == pytest.approx(2.0)
    assert float(metrics["artist_retrieval_top1"]) == pytest.approx(1.0)
    assert float(metrics["common_output_ratio"]) == pytest.approx(0.0)
    assert float(metrics["reference_view_difference_ratio"]) == pytest.approx(0.0)
