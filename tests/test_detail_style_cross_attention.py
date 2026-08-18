from __future__ import annotations

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from anima_style_data.detail_style_cross_attention import (  # noqa: E402
    DetailPreservingTypedSlotReader,
    FreshKVStyleCrossAttention,
)
from anima_style_data.detail_style_training import _audit_student_prompts  # noqa: E402


class _CountingLinear(nn.Linear):
    def __init__(self, features: int):
        super().__init__(features, features, bias=True)
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return super().forward(values)


class _CrossAttention(nn.Module):
    def __init__(self, hidden: int = 8, context: int = 6, heads: int = 2):
        super().__init__()
        self.n_heads = heads
        self.head_dim = hidden // heads
        self.qkv_format = "bshd"
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(context, hidden, bias=False)
        self.v_proj = nn.Linear(context, hidden, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()
        self.output_proj = _CountingLinear(hidden)
        self.output_dropout = nn.Identity()
        self.qkv_calls = 0

    def compute_qkv(self, x, context=None, rope_emb=None):
        del rope_emb
        self.qkv_calls += 1
        context = x if context is None else context
        return tuple(
            projection(values).unflatten(-1, (self.n_heads, self.head_dim))
            for projection, values in (
                (self.q_proj, x),
                (self.k_proj, context),
                (self.v_proj, context),
            )
        )

    def forward(self, x, attn_params, context=None, rope_emb=None):
        del attn_params
        query, key, value = self.compute_qkv(x, context, rope_emb)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2).flatten(-2)
        return self.output_dropout(self.output_proj(attended))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _CrossAttention()


class _Anima(nn.Module):
    def __init__(self, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(blocks))


def _reader() -> DetailPreservingTypedSlotReader:
    return DetailPreservingTypedSlotReader(
        dim=32,
        spatial_tokens=9,
        global_tokens=2,
        summary_tokens=1,
        output_tokens=4,
        heads=4,
        reader_layers=2,
        reader_ff_dim=64,
        mixer_ff_dim=64,
        slot_type_counts=(2, 1, 1),
        strict_v1=False,
    )


def test_reader_is_reference_order_invariant_and_reconstructs_every_valid_view():
    torch.manual_seed(101)
    model = _reader()
    references = torch.randn(2, 3, 12, 32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    permutation = torch.tensor([1, 0, 2])

    output = model(references, mask, reconstruct=True)
    permuted = model(references[:, permutation], mask[:, permutation])

    assert output.tokens.shape == (2, 4, 32)
    assert output.per_reference_tokens.shape == (2, 3, 4, 32)
    assert output.reconstruction is not None
    assert output.reconstruction_target is not None
    assert output.reconstruction.shape == (5, 12, 32)
    torch.testing.assert_close(output.tokens, permuted.tokens, atol=3e-6, rtol=3e-5)

    loss = output.tokens.square().mean() + F.smooth_l1_loss(
        output.reconstruction, output.reconstruction_target
    )
    loss.backward()
    assert model.slot_identity.grad is not None
    assert model.input_projections[0].weight.grad is not None
    assert model.reconstruction_output.weight.grad is not None
    assert not hasattr(model, "log_output_rms")


def test_fresh_kv_has_no_bias_and_native_no_style_path_is_exact():
    torch.manual_seed(103)
    anima = _Anima().requires_grad_(False)
    adapter = FreshKVStyleCrossAttention(
        context_dim=6, blocks=2, initial_alpha=0.1
    )
    adapter.initialize_from_anima(anima)

    assert all(module.bias is None for module in adapter.style_k)
    assert all(module.bias is None for module in adapter.style_v)
    assert not torch.equal(
        adapter.style_k[0].weight, anima.blocks[0].cross_attn.k_proj.weight
    )
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)
    cross = anima.blocks[0].cross_attn
    expected = cross(hidden, None, text)
    actual = adapter.merged_cross_attention(0, hidden, text, cross, None)
    torch.testing.assert_close(actual, expected)


def test_style_uses_separate_softmax_one_native_o_and_trains_fresh_kv():
    torch.manual_seed(107)
    anima = _Anima().requires_grad_(False)
    adapter = FreshKVStyleCrossAttention(
        context_dim=6, blocks=2, initial_alpha=0.2
    )
    adapter.initialize_from_anima(anima)
    adapter.set_style_context(
        torch.randn(2, 3, 6), enabled=torch.tensor([True, False])
    )
    cross = anima.blocks[0].cross_attn
    cross.qkv_calls = 0
    cross.output_proj.calls = 0

    output = adapter.merged_cross_attention(
        0, torch.randn(2, 5, 8), torch.randn(2, 4, 6), cross, None
    )
    output.square().mean().backward()

    assert cross.qkv_calls == 1
    assert cross.output_proj.calls == 1
    assert adapter.style_k[0].weight.grad is not None
    assert adapter.style_v[0].weight.grad is not None
    assert anima.blocks[0].cross_attn.output_proj.weight.grad is None
    assert output.shape == (2, 5, 8)


def test_same_q_internal_teacher_produces_live_gradient_and_calibrates_alpha():
    torch.manual_seed(109)
    anima = _Anima().requires_grad_(False)
    adapter = FreshKVStyleCrossAttention(
        context_dim=6, blocks=2, initial_alpha=0.2
    )
    adapter.initialize_from_anima(anima)
    style = torch.randn(3, 3, 6, requires_grad=True)
    teacher = torch.randn(3, 4, 6)
    adapter.set_style_context(style)
    adapter.set_teacher_context(teacher)
    adapter.begin_alpha_calibration()
    adapter.alpha[0] = 0

    for block_index, block in enumerate(anima.blocks):
        block.cross_attn.qkv_calls = 0
        adapter.merged_cross_attention(
            block_index,
            torch.randn(3, 5, 8),
            torch.randn(3, 4, 6),
            block.cross_attn,
            None,
        )
        adapter.record_gated_internal_teacher(
            block_index,
            torch.ones(3, 1, 1, 1, 8),
            (1, 1, 5),
        )
        assert block.cross_attn.qkv_calls == 1
    loss, metrics = adapter.internal_teacher_loss(rho_min=0.25)
    loss.backward()
    calibration = adapter.finish_alpha_calibration()

    assert torch.isfinite(loss)
    assert style.grad is not None and style.grad.norm() > 0
    assert torch.isfinite(style.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in adapter.parameters()
    )
    assert metrics["internal_teacher_cosine"].isfinite()
    assert len(calibration["alpha"]) == 2
    assert all(0 <= value <= 2.0 for value in calibration["alpha"])


def test_student_prompt_audit_uses_tag_boundaries_not_substrings():
    clean = SimpleNamespace(text_by_key={
        (1, 0): {
            "id": 1,
            "artist": "tri",
            "caption": "safe, triangular headpiece, @ @, 1girl",
        }
    })
    _audit_student_prompts(clean)

    leaked = SimpleNamespace(text_by_key={
        (2, 0): {
            "id": 2,
            "artist": "some_artist",
            "caption": "safe, 1girl, @some artist",
        }
    })
    with pytest.raises(RuntimeError, match="Artist leakage"):
        _audit_student_prompts(leaked)
