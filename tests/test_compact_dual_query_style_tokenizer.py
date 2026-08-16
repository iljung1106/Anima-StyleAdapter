from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.compact_dual_query_style_tokenizer import (  # noqa: E402
    CompactDualQueryStyleTokenizer,
)


def test_compact_dual_query_matches_small_tokenizer_contract():
    torch.manual_seed(83)
    model = CompactDualQueryStyleTokenizer(
        source_dim=32,
        context_dim=32,
        query_tokens=8,
        artist_summary_tokens=2,
        output_tokens=4,
        bottleneck_dim=16,
        score_hidden_dim=8,
        output_rms_init=0.15,
    ).eval()
    references = torch.randn(2, 3, 10, 32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    permutation = torch.tensor([1, 0, 2])

    output = model(references, mask)
    permuted = model(references[:, permutation], mask[:, permutation])

    assert output.tokens.shape == (2, 4, 32)
    assert output.per_reference_tokens.shape == (2, 3, 10, 32)
    assert torch.allclose(output.tokens, permuted.tokens, atol=2e-6, rtol=2e-5)
    assert float(output.tokens.detach().float().square().mean().sqrt()) == pytest.approx(
        0.15, rel=0.03
    )


def test_compact_dual_query_final_tokens_receive_gradients():
    torch.manual_seed(89)
    model = CompactDualQueryStyleTokenizer(
        source_dim=32,
        context_dim=32,
        query_tokens=8,
        artist_summary_tokens=2,
        output_tokens=4,
        bottleneck_dim=16,
        score_hidden_dim=8,
    )
    references = torch.randn(2, 2, 10, 32)
    output = model(references, torch.ones(2, 2, dtype=torch.bool))
    loss = output.tokens.square().mean()
    loss.backward()

    assert model.tokenizer[-1].weight.grad is not None
    assert model.log_output_rms.grad is not None
    assert torch.isfinite(loss)
