from __future__ import annotations

import torch
from torch import nn

from anima_style_data.kv_activation_generator import (
    ReferenceConditionedKVActivationGenerator,
    ReferenceConditionedLowRankKVOperator,
    _NativeAttentionProbe,
    _apply_dense_kv_operator,
    _centered_residual_loss,
    _functional_centered_attention_loss,
    _mean_teacher_operator,
    _mixture_target,
)
from anima_style_data.kv_activation_modulation import apply_kv_factors
from anima_style_data.kv_real_query_distillation import (
    _operator_factors,
    _selected_content_indices,
)


def test_activation_generator_is_reference_conditioned_and_backpropagates() -> None:
    torch.manual_seed(7)
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=32,
        context_dim=24,
        output_dim=40,
        blocks=3,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
    )
    context = torch.randn(2, 11, 24)
    first_style = torch.randn(2, 5, 32)
    second_style = first_style.clone()
    second_style[1].add_(torch.randn_like(second_style[1]) * 0.5)
    first = model(first_style, context, 1)
    second = model(second_style, context, 1)
    assert first.shape == (2, 2, 11, 40)
    assert not torch.allclose(first[1], second[1])
    second.square().mean().backward()
    assert model.output_head[1].weight.grad is not None
    assert model.output_head[0].weight.grad is None


def test_mixture_target_matches_exact_weighted_factor_effect() -> None:
    torch.manual_seed(11)
    context = torch.randn(2, 7, 6)
    down = torch.randn(3, 2, 2, 4, 6)
    up = torch.randn(3, 2, 2, 8, 4)
    components = torch.tensor([[0, 1, -1], [2, 0, 1]])
    weights = torch.tensor([[1.2, -0.2, 0.0], [0.5, 0.75, -0.25]])
    actual = _mixture_target(
        context, down, up, components, weights, block=1
    )
    expected = []
    for row in range(2):
        value = torch.zeros(2, 7, 8)
        for component, weight in zip(components[row], weights[row]):
            if int(component) >= 0:
                value += float(weight) * apply_kv_factors(
                    context[row : row + 1],
                    down[int(component), 1][None],
                    up[int(component), 1][None],
                )[0]
        expected.append(value)
    assert torch.allclose(actual, torch.stack(expected), atol=1e-5)


def test_bilinear_operator_is_linear_in_context_and_reference_conditioned() -> None:
    torch.manual_seed(17)
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=20,
        context_dim=12,
        output_dim=14,
        blocks=3,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        operator_layers=2,
        operator_rank=3,
    )
    style = torch.randn(2, 7, 20)
    left = torch.randn(2, 5, 12)
    right = torch.randn(2, 5, 12)
    combined = model(style, 0.25 * left - 0.75 * right, 1)
    expected = 0.25 * model(style, left, 1) - 0.75 * model(style, right, 1)
    assert combined.shape == (2, 2, 5, 14)
    assert torch.allclose(combined, expected, atol=2e-5, rtol=2e-4)

    changed_style = style.clone()
    changed_style[1].add_(torch.randn_like(changed_style[1]))
    changed = model(changed_style, left, 1)
    assert not torch.allclose(changed[1], model(style, left, 1)[1])
    changed.square().mean().backward()
    assert model.down_output[1][0].weight.grad is not None
    assert model.down_output[0][0].weight.grad is None


def test_bilinear_operator_respects_configured_rank() -> None:
    torch.manual_seed(23)
    rank = 3
    dimensions = 9
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=16,
        context_dim=dimensions,
        output_dim=11,
        blocks=1,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        operator_layers=1,
        operator_rank=rank,
    )
    style = torch.randn(1, 6, 16)
    identity_context = torch.eye(dimensions)[None]
    operator_matrix = model(style, identity_context, 0)[0]
    for kind in range(2):
        assert int(torch.linalg.matrix_rank(operator_matrix[kind], tol=1e-5)) <= rank


def test_mean_teacher_operator_matches_mean_composed_function() -> None:
    torch.manual_seed(29)
    context = torch.randn(3, 5, 7)
    down = torch.randn(4, 2, 2, 3, 7)
    up = torch.randn(4, 2, 2, 9, 3)
    common = _mean_teacher_operator(down, up)
    actual = _apply_dense_kv_operator(context, common[1]).float()
    expected = torch.stack([
        apply_kv_factors(
            context,
            down[artist, 1][None].expand(len(context), -1, -1, -1),
            up[artist, 1][None].expand(len(context), -1, -1, -1),
        )
        for artist in range(len(down))
    ]).mean(dim=0)
    assert torch.allclose(actual, expected, atol=0.04, rtol=0.01)


def test_centered_loss_rejects_common_collapse_and_tiny_output() -> None:
    torch.manual_seed(31)
    teacher = torch.randn(4, 2, 6, 8)
    teacher = teacher - teacher.mean(dim=0, keepdim=True)
    matching, matching_metrics = _centered_residual_loss(
        teacher,
        teacher,
        direction_weight=0.4,
        magnitude_weight=0.2,
        relation_weight=0.2,
        common_weight=0.1,
        magnitude_floor=0.7,
        magnitude_ceiling=1.3,
        temperature=0.1,
    )
    collapsed = teacher.mean(dim=0, keepdim=True).expand_as(teacher)
    collapsed_loss, collapsed_metrics = _centered_residual_loss(
        collapsed,
        teacher,
        direction_weight=0.4,
        magnitude_weight=0.2,
        relation_weight=0.2,
        common_weight=0.1,
        magnitude_floor=0.7,
        magnitude_ceiling=1.3,
        temperature=0.1,
    )
    assert matching < collapsed_loss
    assert matching_metrics["relation_accuracy"] == 1
    assert collapsed_metrics["student_to_teacher_rms"] < 0.01


def test_functional_centered_loss_prefers_correct_artist_effects() -> None:
    torch.manual_seed(37)
    teacher = torch.randn(8, 6, 24)
    common = torch.randn(1, 6, 24) * 0.5
    teacher = teacher - teacher.mean(dim=0, keepdim=True) + common
    matching, metrics = _functional_centered_attention_loss(
        teacher,
        teacher,
        centered_huber_weight=1.0,
        direction_weight=1.0,
        magnitude_weight=0.2,
        relation_weight=0.5,
        raw_huber_weight=0.05,
        temperature=0.1,
    )
    collapsed = teacher.mean(dim=0, keepdim=True).expand_as(teacher)
    collapsed_loss, collapsed_metrics = _functional_centered_attention_loss(
        collapsed,
        teacher,
        centered_huber_weight=1.0,
        direction_weight=1.0,
        magnitude_weight=0.2,
        relation_weight=0.5,
        raw_huber_weight=0.05,
        temperature=0.1,
    )
    assert matching < collapsed_loss
    assert metrics["functional_relation_accuracy"] == 1
    assert collapsed_metrics["functional_student_to_teacher_rms"] < 0.01


def test_real_query_content_selection_is_even_and_unique() -> None:
    assert _selected_content_indices(10, 4) == [0, 3, 6, 9]
    selected = _selected_content_indices(256, 64)
    assert len(selected) == len(set(selected)) == 64
    assert selected[0] == 0 and selected[-1] == 255


def test_operator_factor_export_matches_direct_activation() -> None:
    torch.manual_seed(41)
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=20,
        context_dim=12,
        output_dim=14,
        blocks=2,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        operator_layers=1,
        operator_rank=3,
    )
    style = torch.randn(2, 7, 20)
    context = torch.randn(2, 5, 12)
    down, up = _operator_factors(model, style)
    expected = model(style, context, 1)
    actual = apply_kv_factors(context, down[:, 1], up[:, 1])
    assert down.shape == (2, 2, 2, 3, 12)
    assert up.shape == (2, 2, 2, 14, 3)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)


class _ScaleNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


class _FakeCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = 2
        self.head_dim = 4
        self.k_proj = nn.Linear(6, 8, bias=False)
        self.v_proj = nn.Linear(6, 8, bias=False)
        self.q_norm = _ScaleNorm()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(8, 8, bias=False)


def test_native_probe_does_not_renormalize_cached_real_queries() -> None:
    torch.manual_seed(43)
    probe = _NativeAttentionProbe(_FakeCrossAttention())
    context = torch.randn(2, 5, 6)
    delta = torch.zeros(2, 2, 5, 8)
    key, value = probe.project_context(context, delta)
    real_queries = torch.randn(2, 3, 2, 4)
    cached_result = probe.attend(
        real_queries, key, value, queries_normalized=True
    )
    renormalized_result = probe.attend(
        real_queries, key, value, queries_normalized=False
    )
    assert not torch.allclose(cached_result, renormalized_result)
