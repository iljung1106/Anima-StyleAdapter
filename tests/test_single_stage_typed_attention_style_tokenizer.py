from __future__ import annotations

import torch

from anima_style_data.single_stage_typed_attention_style_tokenizer import (
    SingleStageTypedAttentionStyleTokenizer,
)


def _model() -> SingleStageTypedAttentionStyleTokenizer:
    torch.manual_seed(101)
    return SingleStageTypedAttentionStyleTokenizer(
        dim=32,
        spatial_tokens=8,
        global_tokens=4,
        artist_summary_tokens=2,
        spatial_output_tokens=4,
        global_output_tokens=2,
        artist_output_tokens=2,
        heads=4,
        ff_dim=64,
    )


def test_single_stage_typed_attention_is_set_invariant_and_masks_padding():
    model = _model().eval()
    references = torch.randn(2, 3, 14, 32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output = model(references, mask)

    permutation = torch.tensor([1, 0, 2])
    permuted = model(references[:, permutation], mask[:, permutation])
    assert output.tokens.shape == (2, 8, 32)
    assert output.artist_tokens is not None
    assert output.artist_tokens.shape == (2, 2, 32)
    assert torch.allclose(output.tokens, permuted.tokens, atol=2e-6, rtol=2e-5)

    changed_padding = references.clone()
    changed_padding[0, 2].fill_(1_000)
    padded = model(changed_padding, mask)
    assert torch.allclose(output.tokens[0], padded.tokens[0], atol=2e-6, rtol=2e-5)


def test_single_stage_typed_attention_keeps_type_paths_separate_and_trainable():
    model = _model()
    references = torch.randn(2, 2, 14, 32)
    mask = torch.ones(2, 2, dtype=torch.bool)
    before = model(references, mask).tokens

    changed_artist = references.clone()
    changed_artist[:, :, 12:].add_(5.0 * torch.randn_like(changed_artist[:, :, 12:]))
    after = model(changed_artist, mask).tokens
    assert torch.allclose(before[:, :6], after[:, :6], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(before[:, 6:], after[:, 6:])
    assert not hasattr(model, "log_output_rms")
    assert not hasattr(model, "output_norm")

    before.square().mean().backward()
    assert all(query.grad is not None for query in model.queries)
    assert model.reader.attention.in_proj_weight.grad is not None

