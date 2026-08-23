import torch
import torch.nn.functional as F

from anima_style_data.kv_activation_modulation import (
    NativeKVFactorModulator,
    apply_kv_factors,
    canonicalize_lora_factor_bank,
    canonicalize_lora_factors,
    kv_activation_objective,
    kv_factor_objective,
)
from anima_style_data.kv_activation_sampling import NativeKVFactorInjector


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


def test_batched_canonicalization_preserves_every_weight_delta():
    torch.manual_seed(13)
    down = torch.randn(2, 3, 2, 7)
    up = torch.randn(2, 3, 9, 2)

    canonical_down, canonical_up = canonicalize_lora_factor_bank(
        down, up, chunk_size=4
    )

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


def test_modulator_applies_block_rank_scales_without_losing_gradients():
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
    down_scale = torch.full((4, 2, 2), 0.02)
    up_scale = torch.full((4, 2, 2), 0.003)
    model.set_factor_scales(down_scale, up_scale)

    down, up = model(torch.randn(3, 5, 12), 1)
    down_rms = down.square().mean(dim=-1).sqrt()
    up_rms = up.transpose(-1, -2).square().mean(dim=-1).sqrt()

    torch.testing.assert_close(
        down_rms, torch.full_like(down_rms, 0.02), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        up_rms, torch.full_like(up_rms, 0.003), atol=1e-5, rtol=1e-5
    )


def test_block_specific_heads_only_update_the_selected_block():
    model = NativeKVFactorModulator(
        style_dim=12,
        blocks=3,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
        block_specific_heads=True,
    )

    down, up = model(torch.randn(2, 5, 12), 1)
    (down.square().mean() + up.square().mean()).backward()

    assert model.down_head[2].weight.grad is not None
    assert model.up_head[3].weight.grad is not None
    assert model.down_head[0].weight.grad is None
    assert model.up_head[5].weight.grad is None


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


class _DummyCrossAttention(torch.nn.Module):
    def __init__(self, context_dim: int, output_dim: int):
        super().__init__()
        self.kv_proj = torch.nn.Linear(context_dim, output_dim * 2, bias=False)


class _DummyBlock(torch.nn.Module):
    def __init__(self, context_dim: int, output_dim: int):
        super().__init__()
        self.cross_attn = _DummyCrossAttention(context_dim, output_dim)


class _DummyAnima(torch.nn.Module):
    def __init__(self, blocks: int, context_dim: int, output_dim: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            _DummyBlock(context_dim, output_dim) for _ in range(blocks)
        )


def test_native_kv_factor_injector_matches_exact_low_rank_delta_with_cfg_rows():
    torch.manual_seed(23)
    anima = _DummyAnima(blocks=2, context_dim=6, output_dim=8)
    injector = NativeKVFactorInjector(anima)
    context = torch.randn(4, 7, 6)
    down = torch.randn(2, 2, 2, 3, 6)
    up = torch.randn(2, 2, 2, 8, 3)
    injector.set_factors(down, up, strength=0.75)

    native = F.linear(context, anima.blocks[1].cross_attn.kv_proj.weight)
    actual = anima.blocks[1].cross_attn.kv_proj(context)
    repeated_down = down[:, 1].repeat(2, 1, 1, 1)
    repeated_up = up[:, 1].repeat(2, 1, 1, 1)
    delta = apply_kv_factors(context, repeated_down, repeated_up)
    expected = native + 0.75 * torch.cat((delta[:, 0], delta[:, 1]), dim=-1)

    torch.testing.assert_close(actual, expected)
    injector.disable()
    torch.testing.assert_close(
        anima.blocks[1].cross_attn.kv_proj(context), native
    )
    injector.close()
