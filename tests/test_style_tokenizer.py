from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.style_tokenizer import (
    AnimaStyleTokenizer,
    insert_style_tokens,
)


def test_style_tokenizer_is_reference_order_invariant_and_compact():
    tokenizer = AnimaStyleTokenizer(
        source_dim=16,
        context_dim=12,
        output_tokens=4,
        bottleneck_dim=8,
        score_hidden_dim=4,
        output_rms_init=0.2,
    ).eval()
    references = torch.randn(2, 3, 5, 16)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = tokenizer(references, mask)
    permutation = torch.tensor([1, 0, 2])
    actual = tokenizer(references[:, permutation], mask[:, permutation])

    assert expected.shape == (2, 4, 12)
    assert torch.isfinite(expected).all()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert float(expected.detach().float().square().mean().sqrt()) == pytest.approx(
        0.2, rel=0.05
    )


def test_style_tokenizer_accepts_bfloat16_cache_under_autocast():
    tokenizer = AnimaStyleTokenizer(
        source_dim=16,
        context_dim=16,
        output_tokens=2,
        bottleneck_dim=8,
        score_hidden_dim=4,
    ).eval()
    references = torch.randn(1, 1, 4, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = tokenizer(references, torch.ones(1, 1, dtype=torch.bool))

    assert output.shape == (1, 2, 16)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_style_token_insertion_preserves_text_and_backpropagates():
    conditioning = torch.zeros(2, 8, 4)
    conditioning[0, :3] = 1
    conditioning[1, :5] = 2
    style = torch.randn(2, 2, 4, requires_grad=True)

    result = insert_style_tokens(
        conditioning, torch.tensor([3, 5]), style
    )

    assert torch.equal(result[0, :3], conditioning[0, :3])
    assert torch.equal(result[1, :5], conditioning[1, :5])
    assert torch.equal(result[0, 3:5], style[0])
    assert torch.equal(result[1, 5:7], style[1])
    assert torch.count_nonzero(result[0, 5:]) == 0
    result.sum().backward()
    assert torch.equal(style.grad, torch.ones_like(style))


def test_style_token_insertion_rejects_full_context():
    with pytest.raises(ValueError, match="enough unused positions"):
        insert_style_tokens(
            torch.zeros(1, 4, 8),
            torch.tensor([3]),
            torch.zeros(1, 2, 8),
        )
