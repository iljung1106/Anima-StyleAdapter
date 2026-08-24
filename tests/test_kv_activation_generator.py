from __future__ import annotations

import torch

from anima_style_data.kv_activation_generator import (
    ReferenceConditionedKVActivationGenerator,
    _mixture_target,
)
from anima_style_data.kv_activation_modulation import apply_kv_factors


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
