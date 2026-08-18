from __future__ import annotations

import torch

from anima_style_data.dual_query_style_training import (
    _native_artist_teacher_objective,
    _same_artist_functional_loss,
    _scheduled_teacher_gradient_scale,
    _teacher_projected_effect_loss,
)
from anima_style_data.global_query_style_tokenizer import (
    scheduled_prompt_mode_weights,
)
from anima_style_data.typed_multi_descriptor_style_tokenizer import (
    TypedMultiDescriptorCompactStyleTokenizer,
)


def _model(
    output_mode: str = "attention",
    *,
    group_slot_embedding_scale: float = 1.0,
) -> TypedMultiDescriptorCompactStyleTokenizer:
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
        output_mode=output_mode,
        group_bottleneck_dim=32,
        group_slot_embedding_scale=group_slot_embedding_scale,
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
    token_rms = output.tokens.float().square().mean(dim=(1, 2)).sqrt()
    assert torch.allclose(token_rms, torch.full_like(token_rms, 0.15), atol=2e-5)
    # Explicit output slots must not initialize as one shared vector.
    assert not torch.allclose(output.tokens[:, 0], output.tokens[:, 1])
    assert not torch.allclose(output.tokens[0], output.tokens[1])

    output.tokens[:, 0, 0].mean().backward()
    assert model.descriptor_queries.grad is not None
    assert torch.isfinite(model.descriptor_queries.grad).all()


def test_grouped_mlp_preserves_typed_groups_and_conditional_output_slots():
    model = _model("grouped_mlp")
    references = torch.randn(3, 2, 14, 32)
    mask = torch.ones(3, 2, dtype=torch.bool)
    output = model(references, mask)

    assert model.output_reader is None
    assert model.output_group_slices == ((0, 2), (2, 3), (3, 4))
    assert [head.output_tokens for head in model.output_group_heads] == [2, 1, 1]
    assert output.attention_maps is None
    assert output.tokens.shape == (3, 4, 32)
    token_rms = output.tokens.float().square().mean(dim=(1, 2)).sqrt()
    assert torch.allclose(token_rms, torch.full_like(token_rms, 0.15), atol=2e-5)
    assert not torch.allclose(output.tokens[:, 0], output.tokens[:, 1])
    assert not torch.allclose(output.tokens[0], output.tokens[1])

    output.tokens.square().mean().backward()
    assert model.descriptor_queries.grad is not None
    assert all(
        head.mlp[-1].weight.grad is not None
        for head in model.output_group_heads
    )


def test_grouped_mlp_can_disable_sample_independent_slot_bias():
    model = _model("grouped_mlp", group_slot_embedding_scale=0.0).eval()
    references = torch.randn(2, 3, 14, 32)
    mask = torch.ones(2, 3, dtype=torch.bool)
    before = model(references, mask).tokens
    with torch.no_grad():
        for head in model.output_group_heads:
            head.slot_embedding.fill_(1000.0)
    after = model(references, mask).tokens
    assert torch.allclose(before, after, atol=2e-6, rtol=2e-6)


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


def test_common_output_can_be_bounded_against_teacher_rms_not_tiny_projection():
    teacher = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    common = torch.tensor([[[0.0, 0.20]], [[0.0, 0.20]]])
    student = 0.20 * teacher + common
    _, metrics = _native_artist_teacher_objective(
        student,
        teacher,
        {
            "center_student_teacher": True,
            "native_teacher_weight": 0.0,
            "native_teacher_ramp_steps": 0,
            "teacher_projected_effect_weight": 0.0,
            "common_output_denominator": "teacher_centered_rms",
            "common_output_start_step": 1,
            "common_output_ramp_end_step": 1,
            "common_output_weight": 1.0,
            "common_output_threshold_start": 0.10,
            "common_output_threshold_end": 0.10,
            "centered_energy_weight": 0.0,
            "artist_teacher_contrastive_weight": 0.0,
            "artist_teacher_ranking_weight": 0.0,
        },
        step=1,
    )
    assert torch.allclose(metrics["common_output_ratio"], torch.tensor(0.20))
    assert torch.allclose(
        metrics["native_teacher_common_to_aligned_ratio"], torch.tensor(1.0)
    )


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


def test_sparse_teacher_updates_can_preserve_expected_gradient_pressure():
    assert _scheduled_teacher_gradient_scale(4, {}) == 1.0
    assert _scheduled_teacher_gradient_scale(
        4, {"dual_domain_teacher_scale_by_cadence": True}
    ) == 4.0


def test_centered_functional_consistency_ignores_view_common_effects():
    artist_effect = torch.tensor(
        [[[[1.0, 0.0]]], [[[-1.0, 0.0]]], [[[0.0, 1.0]]], [[[0.0, -1.0]]]]
    )
    first = artist_effect + torch.tensor([[[[2.0, 2.0]]]])
    second = artist_effect + torch.tensor([[[[-3.0, 4.0]]]])
    loss, metrics = _same_artist_functional_loss(
        first,
        second,
        torch.ones(4, dtype=torch.bool),
        direction_fraction=0.75,
        huber_beta=0.10,
        center_across_artists=True,
    )
    assert float(loss) < 1e-7
    assert float(metrics["functional_same_artist_cosine"]) > 0.999


def test_centered_functional_consistency_does_not_reward_global_collapse():
    collapsed = torch.ones(4, 1, 1, 2)
    raw_loss, _ = _same_artist_functional_loss(
        collapsed,
        collapsed,
        torch.ones(4, dtype=torch.bool),
        direction_fraction=0.75,
        huber_beta=0.10,
    )
    centered_loss, _ = _same_artist_functional_loss(
        collapsed,
        collapsed,
        torch.ones(4, dtype=torch.bool),
        direction_fraction=0.75,
        huber_beta=0.10,
        center_across_artists=True,
    )
    assert float(raw_loss) < 1e-7
    assert float(centered_loss) > 0.70
