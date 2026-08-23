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
from anima_style_data.kv_mixture_analysis import (
    _activation_from_coefficients,
    _knn_coefficients,
)
from anima_style_data.kv_generalizing_modulator import (
    _stratified_view_indices,
    build_mixed_activation_batch,
)
from anima_style_data.few_shot_kv_adapter import FewShotNativeKVStyleAdapter


def test_generalizing_validation_samples_every_reference_count():
    counts = torch.tensor([1, 1, 1, 1, 2, 2, 4], dtype=torch.float32)

    selected = _stratified_view_indices(counts, views_per_count=2)

    assert selected.tolist() == [0, 3, 4, 5, 6]
    assert counts[selected].tolist() == [1.0, 1.0, 2.0, 2.0, 4.0]


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


def test_shared_heads_with_block_low_rank_delta_backpropagate_selected_block():
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
        block_delta_rank=3,
    )

    down, up = model(torch.randn(2, 5, 12), 1)
    (down.square().mean() + up.square().mean()).backward()

    assert model.down_head.weight.grad is not None
    assert model.up_head.weight.grad is not None
    assert torch.count_nonzero(model.down_delta_b.grad[1]) > 0
    assert torch.count_nonzero(model.up_delta_b.grad[1]) > 0
    assert torch.count_nonzero(model.down_delta_b.grad[0]) == 0
    assert torch.count_nonzero(model.up_delta_b.grad[2]) == 0


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


def test_convex_activation_mixture_uses_visual_neighbor_weights():
    train_codes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = torch.tensor([[0.9, 0.1]])
    weights = _knn_coefficients(
        train_codes, query, neighbors=2, temperature=0.05
    )
    activations = torch.arange(3 * 2 * 1 * 1).reshape(3, 2, 1, 1).float()
    mixed = _activation_from_coefficients(
        activations, weights, affine_centered=False
    )

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    assert torch.count_nonzero(weights) == 2
    torch.testing.assert_close(
        mixed,
        torch.einsum("va,akno->vkno", weights, activations),
    )


def test_generalizing_batch_builds_exact_weighted_teacher_activation():
    torch.manual_seed(37)
    batch, contexts, groups = 2, 2, 3
    tokens, context_dim, output_dim, rank = 5, 7, 11, 3
    sampled_contexts = torch.randn(contexts, tokens, context_dim)
    predicted_down = torch.randn(batch, 2, rank, context_dim)
    predicted_up = torch.randn(batch, 2, output_dim, rank)
    group_down = torch.randn(batch, groups, 2, rank, context_dim)
    group_up = torch.randn(batch, groups, 2, output_dim, rank)
    weights = torch.rand(batch, groups)
    weights /= weights.sum(dim=-1, keepdim=True)
    output_indices = torch.tensor([1, 4, 8])

    student, target = build_mixed_activation_batch(
        sampled_contexts,
        predicted_down,
        predicted_up,
        group_down,
        group_up,
        weights,
        output_indices,
    )

    assert student.shape == (batch * contexts, 2, tokens, 3)
    expected_rows = []
    for artist in range(batch):
        for context in sampled_contexts:
            group_rows = apply_kv_factors(
                context.expand(groups, -1, -1),
                group_down[artist],
                group_up[artist].index_select(2, output_indices),
            )
            expected_rows.append(
                torch.einsum("g,gkno->kno", weights[artist], group_rows)
            )
    torch.testing.assert_close(target, torch.stack(expected_rows))


class _DummyReader(torch.nn.Module):
    def __init__(self, style_dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(style_dim))

    def forward(self, tokens, mask):
        if tokens.ndim != 4:
            raise ValueError("Expected real [batch,references,tokens,dim] contract")
        weights = mask.to(tokens.dtype)
        pooled = (tokens * weights[:, :, None, None]).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1)[:, None, None]
        return type("ReaderOutput", (), {"tokens": pooled * self.scale})()


def test_few_shot_adapter_activates_and_rescales_native_kv_hook():
    torch.manual_seed(41)
    anima = _DummyAnima(blocks=2, context_dim=6, output_dim=8)
    reader = _DummyReader(style_dim=12)
    modulator = NativeKVFactorModulator(
        style_dim=12,
        blocks=2,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
    )
    adapter = FewShotNativeKVStyleAdapter(
        reader=reader, modulator=modulator, anima=anima
    )
    context = torch.randn(1, 5, 6)
    baseline = anima.blocks[0].cross_attn.kv_proj(context)

    codes = adapter.set_references(torch.randn(1, 4, 5, 12), strength=0.5)
    styled_half = anima.blocks[0].cross_attn.kv_proj(context)
    adapter.set_strength(1.0)
    styled_full = anima.blocks[0].cross_attn.kv_proj(context)

    assert codes.shape == (1, 5, 12)
    torch.testing.assert_close(styled_full - baseline, 2 * (styled_half - baseline))
    adapter.disable()
    torch.testing.assert_close(anima.blocks[0].cross_attn.kv_proj(context), baseline)
    adapter.close()
