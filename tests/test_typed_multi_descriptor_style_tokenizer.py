from __future__ import annotations

import torch

from anima_style_data.dual_query_style_training import (
    _teacher_projected_effect_loss,
)
from anima_style_data.global_query_style_tokenizer import (
    scheduled_prompt_mode_weights,
)
from anima_style_data.typed_multi_descriptor_style_tokenizer import (
    TypedMultiDescriptorCompactStyleTokenizer,
)


def _model() -> TypedMultiDescriptorCompactStyleTokenizer:
    torch.manual_seed(7)
    return TypedMultiDescriptorCompactStyleTokenizer(
        dim=32,
        spatial_tokens=8,
        global_tokens=4,
        artist_summary_tokens=2,
        spatial_descriptors=2,
        global_descriptors=1,
        artist_descriptors=1,
        output_tokens=4,
        heads=4,
        ff_dim=64,
        output_gain_center=0.15,
    )


def test_typed_tokenizer_is_reference_permutation_invariant_and_masks_padding():
    model = _model().eval()
    references = torch.randn(2, 3, 14, 32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    first = model(references, mask)

    permutation = torch.tensor([1, 0, 2])
    permuted = model(references[:, permutation], mask[:, permutation])
    assert torch.allclose(first.tokens, permuted.tokens, atol=2e-6, rtol=2e-6)

    changed_padding = references.clone()
    changed_padding[0, 2].fill_(1_000)
    padded = model(changed_padding, mask)
    assert torch.allclose(first.tokens[0], padded.tokens[0], atol=2e-6, rtol=2e-6)


def test_typed_tokenizer_preserves_descriptor_and_output_slot_shapes():
    model = _model()
    references = torch.randn(3, 2, 14, 32)
    mask = torch.ones(3, 2, dtype=torch.bool)
    output = model(references, mask)

    assert output.per_reference_tokens.shape == (3, 2, 4, 32)
    assert output.descriptor_tokens is not None
    assert output.descriptor_tokens.shape == (3, 4, 32)
    assert output.artist_tokens is not None
    assert output.artist_tokens.shape == (3, 1, 32)
    assert output.tokens.shape == (3, 4, 32)
    assert output.output_gain is not None
    assert torch.allclose(
        output.output_gain,
        torch.full_like(output.output_gain, 0.15),
        atol=1e-6,
    )
    # Explicit output slots must not initialize as one shared vector.
    assert not torch.allclose(output.tokens[:, 0], output.tokens[:, 1])

    output.tokens.square().mean().backward()
    assert model.gain_head.weight.grad is not None
    assert torch.isfinite(model.gain_head.weight.grad).all()
    assert model.descriptor_queries.grad is not None
    assert torch.isfinite(model.descriptor_queries.grad).all()


def test_teacher_projected_loss_rejects_orthogonal_energy_shortcut():
    teacher = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[-1.0, 0.0], [-1.0, 0.0]]],
        ]
    )
    aligned = 0.20 * teacher
    orthogonal = torch.tensor(
        [
            [[[0.0, 0.20], [0.0, 0.20]]],
            [[[0.0, -0.20], [0.0, -0.20]]],
        ]
    )
    aligned_loss, aligned_metrics = _teacher_projected_effect_loss(
        aligned,
        teacher,
        coefficient_minimum=0.15,
        coefficient_maximum=1.25,
        orthogonal_maximum=0.10,
        orthogonal_weight=1.0,
    )
    orthogonal_loss, orthogonal_metrics = _teacher_projected_effect_loss(
        orthogonal,
        teacher,
        coefficient_minimum=0.15,
        coefficient_maximum=1.25,
        orthogonal_maximum=0.10,
        orthogonal_weight=1.0,
    )
    assert aligned_loss < 1e-7
    assert orthogonal_loss > aligned_loss
    assert aligned_metrics["native_teacher_projection_coefficient"] > 0.19
    assert orthogonal_metrics["native_teacher_projection_coefficient"].abs() < 1e-7


def test_prompt_warmup_uses_optimizer_steps_and_preserves_distribution():
    base = {"full": 0.30, "tag_dropout": 0.40, "short": 0.20, "empty": 0.10}
    warm = scheduled_prompt_mode_weights(
        base,
        data_step=1_999,
        gradient_accumulation_steps=4,
        empty_warmup_steps=500,
        empty_warmup_weight=0.20,
    )
    after = scheduled_prompt_mode_weights(
        base,
        data_step=2_000,
        gradient_accumulation_steps=4,
        empty_warmup_steps=500,
        empty_warmup_weight=0.20,
    )
    assert abs(sum(warm.values()) - 1.0) < 1e-8
    assert warm["empty"] == 0.20
    assert after == base

