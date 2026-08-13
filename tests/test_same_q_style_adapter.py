import torch
import torch.nn.functional as F
from torch import nn

from anima_style_data.same_q_style_adapter import SameQFullRankStyleAdapter


class _CountingLinear(nn.Linear):
    def __init__(self, features: int):
        super().__init__(features, features, bias=False)
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
                (self.q_proj, x), (self.k_proj, context), (self.v_proj, context)
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


def _adapter(connector_layers: int = 1):
    return SameQFullRankStyleAdapter(
        style_dim=6,
        context_dim=6,
        slots=3,
        heads=2,
        blocks=2,
        alpha_init=0.01,
        style_dropout=0.0,
        aggregator_mode="minimal",
        aggregator_bottleneck=3,
        connector_layers=connector_layers,
        connector_heads=2,
    )


def test_native_kv_are_copied_and_alpha_is_small_nonzero():
    torch.manual_seed(3)
    anima = _Anima().requires_grad_(False)
    expected_keys = [block.cross_attn.k_proj.weight.detach().clone() for block in anima.blocks]
    expected_values = [block.cross_attn.v_proj.weight.detach().clone() for block in anima.blocks]
    adapter = _adapter(connector_layers=0)

    adapter.initialize_from_anima(anima)

    for actual, expected in zip(adapter.style_k, expected_keys, strict=True):
        torch.testing.assert_close(actual.weight, expected)
    for actual, expected in zip(adapter.style_v, expected_values, strict=True):
        torch.testing.assert_close(actual.weight, expected)
    torch.testing.assert_close(adapter.alpha, torch.full((2,), 0.01))
    assert not hasattr(adapter, "o_down")
    assert not hasattr(adapter, "o_up")


def test_native_kv_copy_supports_production_fused_projection():
    torch.manual_seed(5)
    anima = _Anima().requires_grad_(False)
    expected = []
    for block in anima.blocks:
        cross = block.cross_attn
        expected.append((cross.k_proj.weight.detach().clone(), cross.v_proj.weight.detach().clone()))
        fused = nn.Linear(6, 16, bias=False)
        with torch.no_grad():
            fused.weight.copy_(torch.cat((cross.k_proj.weight, cross.v_proj.weight)))
        cross.kv_proj = fused
        del cross.k_proj, cross.v_proj
    adapter = _adapter(connector_layers=0)

    adapter.initialize_from_anima(anima)

    for index, (key, value) in enumerate(expected):
        torch.testing.assert_close(adapter.style_k[index].weight, key)
        torch.testing.assert_close(adapter.style_v[index].weight, value)


def test_text_and_style_share_one_q_and_merge_before_one_native_output_projection():
    torch.manual_seed(11)
    anima = _Anima().requires_grad_(False)
    adapter = _adapter()
    adapter.initialize_from_anima(anima)
    adapter.set_style_tokens(torch.randn(2, 3, 6))
    cross = anima.blocks[0].cross_attn
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)

    output = adapter.merged_cross_attention(0, hidden, text, cross, None)
    output.square().mean().backward()

    # compute_qkv produces the exact Q used by both separate softmax calls.
    assert cross.qkv_calls == 1
    # Text and style are merged before the frozen full-rank native O.
    assert cross.output_proj.calls == 1
    assert output.shape == hidden.shape

    # alpha is nonzero, so all representation parameters learn on step one.
    assert adapter.alpha.grad is not None and adapter.alpha.grad.norm() > 0
    assert adapter.bridge.projection.weight.grad is not None
    assert adapter.bridge.projection.weight.grad.norm() > 0
    assert adapter.bridge.connector[0].qkv.weight.grad is not None
    assert adapter.bridge.connector[0].qkv.weight.grad.norm() > 0
    assert adapter.style_k[0].weight.grad is not None
    assert adapter.style_k[0].weight.grad.norm() > 0
    assert adapter.style_v[0].weight.grad is not None
    assert adapter.style_v[0].weight.grad.norm() > 0
    assert cross.output_proj.weight.grad is None


def test_zero_alpha_is_exactly_the_native_text_cross_attention():
    torch.manual_seed(13)
    anima = _Anima().requires_grad_(False)
    adapter = _adapter(connector_layers=0)
    adapter.initialize_from_anima(anima)
    adapter.set_style_tokens(torch.randn(2, 3, 6))
    with torch.no_grad():
        adapter.alpha.zero_()
    cross = anima.blocks[0].cross_attn
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)

    expected = cross(hidden, None, text)
    actual = adapter.merged_cross_attention(0, hidden, text, cross, None)

    torch.testing.assert_close(actual, expected)


def test_no_active_style_preserves_native_cross_attention_path():
    torch.manual_seed(17)
    anima = _Anima().requires_grad_(False)
    adapter = _adapter(connector_layers=0)
    adapter.initialize_from_anima(anima)
    cross = anima.blocks[0].cross_attn
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)

    expected = cross(hidden, None, text)
    actual = adapter.merged_cross_attention(0, hidden, text, cross, None)

    torch.testing.assert_close(actual, expected)
