from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.dual_query_style_tokenizer import (  # noqa: E402
    DualQuerySetStyleTokenizer,
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
