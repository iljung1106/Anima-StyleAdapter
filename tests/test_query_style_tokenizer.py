from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.query_style_tokenizer import (
    QueryStyleTokenizerV2,
    _artist_contrastive_loss,
)


def _small_model() -> QueryStyleTokenizerV2:
    return QueryStyleTokenizerV2(
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


def test_query_tokenizer_preserves_slots_and_is_reference_order_invariant():
    model = _small_model().eval()
    references = torch.randn(2, 3, 8, 16)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    expected = model(references, mask).tokens
    permutation = torch.tensor([1, 0, 2])
    actual = model(references[:, permutation], mask[:, permutation]).tokens

    assert expected.shape == (2, 4, 16)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert float(expected.detach().float().square().mean().sqrt()) == pytest.approx(
        0.2, rel=0.05
    )
    assert not torch.allclose(expected[:, 0], expected[:, 1])


def test_weak_reconstruction_path_backpropagates_into_reference_queries():
    model = _small_model()
    references = torch.randn(2, 2, 8, 16)
    output = model(
        references, torch.ones(2, 2, dtype=torch.bool), reconstruct=True
    )
    assert output.reconstruction is not None
    assert output.reconstruction_target is not None
    loss = torch.nn.functional.smooth_l1_loss(
        output.reconstruction, output.reconstruction_target
    )
    loss.backward()

    assert output.reconstruction.shape == (2, 8, 16)
    assert torch.isfinite(loss)
    assert model.reference_queries.grad is not None


def test_artist_contrastive_loss_prefers_matching_reference_artist():
    targets = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[-1.0, 0.0], [0.0, -1.0]],
    ])
    matching = targets.clone()
    swapped = targets.flip(0)

    matching_loss, metrics = _artist_contrastive_loss(
        targets, matching, ["a", "b"], 0.1
    )
    swapped_loss, _ = _artist_contrastive_loss(
        targets, swapped, ["a", "b"], 0.1
    )

    assert matching_loss < swapped_loss
    assert metrics["artist_positive_similarity"] > metrics[
        "artist_negative_similarity"
    ]
