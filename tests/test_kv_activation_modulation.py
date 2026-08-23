import torch

from anima_style_data.kv_activation_modulation import (
    NativeKVFactorModulator,
    apply_kv_factors,
    canonicalize_lora_factors,
    kv_activation_objective,
    kv_factor_objective,
)


def test_apply_kv_factors_matches_explicit_low_rank_linears():
    torch.manual_seed(5)
    context = torch.randn(3, 7, 6)
    down = torch.randn(3, 2, 2, 6)
    up = torch.randn(3, 2, 8, 2)

    actual = apply_kv_factors(context, down, up)
    expected = torch.stack([
        torch.stack([
            (context[row] @ down[row, kind].t()) @ up[row, kind].t()
            for kind in range(2)
        ])
        for row in range(3)
    ])

    torch.testing.assert_close(actual, expected)


def test_canonical_factors_preserve_the_exact_weight_delta():
    torch.manual_seed(11)
    down = torch.randn(3, 7)
    up = torch.randn(9, 3)

    canonical_down, canonical_up = canonicalize_lora_factors(down, up)

    torch.testing.assert_close(
        canonical_up @ canonical_down,
        up @ down,
        atol=1e-5,
        rtol=1e-5,
    )


def test_modulator_emits_independent_kv_factors_and_backpropagates():
    model = NativeKVFactorModulator(
        style_dim=12,
        blocks=4,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
    )
    style = torch.randn(3, 5, 12)

    down, up = model(style, 2)
    loss = down.square().mean() + up.square().mean()
    loss.backward()

    assert down.shape == (3, 2, 2, 6)
    assert up.shape == (3, 2, 8, 2)
    assert model.style_projection.weight.grad is not None
    assert model.down_head.weight.grad is not None
    assert model.up_head.weight.grad is not None


def test_activation_objective_prefers_exact_teacher_delta():
    teacher = torch.randn(4, 2, 7, 8)
    exact_loss, exact = kv_activation_objective(
        teacher.clone(), teacher, direction_weight=0.5, magnitude_weight=0.1
    )
    collapsed = torch.zeros_like(teacher, requires_grad=True)
    collapsed_loss, collapsed_metrics = kv_activation_objective(
        collapsed, teacher, direction_weight=0.5, magnitude_weight=0.1
    )

    assert float(exact["cosine"]) > 0.999
    assert float(exact["relative_rms_error"]) < 1e-6
    assert float(collapsed_loss) > float(exact_loss) + 1.0
    collapsed_loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_factor_objective_prefers_canonical_teacher_factors():
    teacher = torch.randn(4, 2, 3, 8)
    exact_loss, exact = kv_factor_objective(
        teacher.clone(), teacher, direction_weight=0.5, magnitude_weight=0.1
    )
    wrong_loss, wrong = kv_factor_objective(
        teacher.roll(1, dims=0),
        teacher,
        direction_weight=0.5,
        magnitude_weight=0.1,
    )

    assert float(exact["cosine"]) > 0.999
    assert float(wrong["cosine"]) < 0.5
    assert float(wrong_loss) > float(exact_loss) + 0.5
