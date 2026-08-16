from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.hierarchical_dual_query_style_tokenizer import (  # noqa: E402
    HierarchicalDualQueryStyleTokenizer,
)
from anima_style_data.style_transfer import _pilot_reference_schedule_state  # noqa: E402


def _model() -> HierarchicalDualQueryStyleTokenizer:
    return HierarchicalDualQueryStyleTokenizer(
        dim=32,
        spatial_tokens=8,
        global_tokens=2,
        artist_summary_tokens=2,
        output_tokens=4,
        heads=4,
        per_reference_layers=1,
        per_reference_ff_dim=64,
        reference_score_dim=8,
        reconstruction_layers=1,
        reconstruction_ff_dim=64,
        initial_output_rms=0.15,
    )


def test_hierarchical_tokenizer_preserves_reference_groups_and_order_invariance():
    torch.manual_seed(71)
    model = _model().eval()
    references = torch.randn(2, 3, 12, 32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    permutation = torch.tensor([1, 0, 2])

    expected = model(references, mask)
    actual = model(references[:, permutation], mask[:, permutation])

    assert expected.tokens.shape == (2, 4, 32)
    assert expected.per_reference_tokens.shape == (2, 3, 4, 32)
    assert torch.allclose(actual.tokens, expected.tokens, atol=2e-6, rtol=2e-5)
    assert not hasattr(model, "log_output_rms")


def test_reconstruction_and_reference_conditioned_slots_receive_gradients():
    torch.manual_seed(73)
    model = _model()
    references = torch.randn(2, 2, 12, 32)
    output = model(
        references,
        torch.ones(2, 2, dtype=torch.bool),
        reconstruct=True,
    )

    assert output.reconstruction is not None
    assert output.reconstruction_target is not None
    loss = output.tokens.square().mean() + torch.nn.functional.smooth_l1_loss(
        output.reconstruction, output.reconstruction_target
    )
    loss.backward()

    assert output.reconstruction.shape == (2, 12, 32)
    assert model.reference_queries.grad is not None
    assert model.output_projection.weight.grad is not None
    assert model.reconstruction_queries is not None
    assert model.reconstruction_queries.grad is not None
    assert torch.isfinite(loss)


def test_reference_schedule_uses_optimizer_step_boundaries():
    schedule = [
        {"name": "one_two", "end_step": 500, "max_references": 2},
        {"name": "one_four", "end_step": 2000, "max_references": 4},
        {"name": "one_eight", "end_step": 10000, "max_references": 8},
    ]

    assert _pilot_reference_schedule_state(500, schedule)["max_references"] == 2
    assert _pilot_reference_schedule_state(501, schedule)["max_references"] == 4
    assert _pilot_reference_schedule_state(2001, schedule)["max_references"] == 8
