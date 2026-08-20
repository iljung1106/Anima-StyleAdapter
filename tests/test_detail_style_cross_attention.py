from __future__ import annotations

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from anima_style_data.detail_style_cross_attention import (  # noqa: E402
    DetailPreservingTypedSlotReader,
    FreshKVStyleCrossAttention,
    SharedBaseKVStyleCrossAttention,
)
from anima_style_data.detail_style_training import (  # noqa: E402
    _StyleAttenuationRecorder,
    _audit_student_prompts,
    _compose_separate_text_style_guidance,
    _delayed_learning_rate_multiplier,
    _effect_stage_metrics,
    _minimal_native_teacher_objective,
)


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
    assert model.reference_identity_projection.weight.grad is not None
    assert model.pool_type_embeddings.grad is not None
    assert model.input_projections[0].weight.grad is not None
    assert model.reconstruction_output.weight.grad is not None
    assert len(model.mixers) == 2
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
    adapter.begin_alpha_calibration(
        timestep_bin_edges=(0.0, 0.5, 1.000001),
        reset_alpha=False,
    )
    adapter.set_alpha_calibration_timestep(0.75)

    for block_index, block in enumerate(anima.blocks):
        hidden = torch.randn(3, 5, 8)
        text = torch.randn(3, 4, 6)
        expected_clean_path = block.cross_attn(hidden, None, text)
        block.cross_attn.qkv_calls = 0
        actual_clean_path = adapter.merged_cross_attention(
            block_index,
            hidden,
            text,
            block.cross_attn,
            None,
        )
        torch.testing.assert_close(actual_clean_path, expected_clean_path)
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
    assert all(0 < value <= 2.0 for value in calibration["alpha"])
    assert calibration["native_artist_residual_rms_by_timestep_bin"][0] == [None, None]
    assert all(
        value is not None
        for value in calibration["raw_style_attention_rms_by_timestep_bin"][1]
    )
    assert calibration["measured_alpha"] == pytest.approx([0.2, 0.2])
    profile = calibration["block_timestep_profiles"][1]["blocks"]
    for block in profile:
        effective = block["effective_style_residual_rms"]["median"]
        raw = block["raw_style_residual_rms"]["median"]
        assert effective / raw == pytest.approx(0.2, rel=1e-5)
    assert len(calibration["block_timestep_profiles"]) == 2
    assert calibration["block_timestep_profiles"][1]["blocks"][0]["samples"] == 3


def test_attenuation_metrics_remove_common_output_and_pair_stage_captures():
    teacher = torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]])
    common = torch.tensor([[[0.0, 3.0]]])
    student = 2.0 * teacher + common
    metrics = _effect_stage_metrics(student, teacher)

    assert metrics["teacher_projection"] == pytest.approx(2.0)
    assert metrics["teacher_direction_cosine"] == pytest.approx(1.0)
    assert metrics["student_to_teacher_rms"] == pytest.approx(2.0)
    assert metrics["common_output_ratio"] == pytest.approx(0.8320503)

    recorder = _StyleAttenuationRecorder()
    recorder(0, "post_cross_hidden", torch.zeros_like(student))
    recorder(0, "post_mlp_hidden", torch.zeros_like(student))
    recorder.mode = "style"
    recorder(0, "pre_o_style", student)
    recorder(0, "pre_o_teacher", teacher)
    recorder(0, "post_o_style", student)
    recorder(0, "post_o_teacher", teacher)
    recorder(0, "post_gate_style", student)
    recorder(0, "post_gate_teacher", teacher)
    recorder(0, "post_cross_hidden", student)
    recorder(0, "post_mlp_hidden", student)
    captured = recorder.finish()[0]

    assert set(captured) == {
        "pre_o", "post_o", "post_gate", "post_cross_hidden", "post_mlp_hidden"
    }
    assert captured["post_gate"]["teacher_projection"] == pytest.approx(2.0)


def test_timestep_strength_profile_interpolates_per_block_and_keeps_bounds():
    adapter = FreshKVStyleCrossAttention(
        context_dim=6, blocks=2, initial_alpha=0.2
    )
    adapter.configure_timestep_strength(
        timestep_bin_edges=(0.0, 0.25, 0.75, 1.0),
        alpha_by_timestep=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]),
        native_lower_by_timestep=torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.8]]
        ),
        native_upper_by_timestep=torch.tensor(
            [[1.1, 1.2], [1.3, 1.4], [1.5, 1.8]]
        ),
    )
    adapter.set_timesteps(torch.tensor([0.125, 0.3125, 0.875]))

    alpha = adapter._effective_alpha(
        0, 3, device=torch.device("cpu"), dtype=torch.float32
    ).flatten()
    lower, upper = adapter._native_strength_bounds(1, 3)

    torch.testing.assert_close(alpha, torch.tensor([1.0, 2.0, 5.0]))
    torch.testing.assert_close(lower, torch.tensor([0.2, 0.3, 0.8]))
    torch.testing.assert_close(upper, torch.tensor([1.2, 1.3, 1.8]))


def test_fixed_output_strength_matches_post_native_gate_p75_and_preserves_disabled_rows():
    torch.manual_seed(111)
    anima = _Anima(blocks=1).requires_grad_(False)
    adapter = FreshKVStyleCrossAttention(context_dim=6, blocks=1)
    adapter.initialize_from_anima(anima)
    adapter.configure_fixed_output_strength(
        timestep_bin_edges=(0.0, 1.000001),
        native_fixed_output_by_timestep=torch.tensor([[0.35]]),
    )
    adapter.set_timesteps(torch.tensor([0.5, 0.5]))
    adapter.set_style_context(
        torch.randn(2, 3, 6), enabled=torch.tensor([True, False])
    )
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)
    gate = torch.tensor([0.5, 1.5]).reshape(2, 1, 1, 1, 1).expand(-1, -1, -1, -1, 8)
    cross = anima.blocks[0].cross_attn
    clean = cross(hidden, None, text)
    adapter.set_block_gate_context(0, gate, (1, 1, 5))
    actual = adapter.merged_cross_attention(0, hidden, text, cross, None)

    gated_delta = (actual - clean) * gate.reshape(2, 1, 8)
    rms = gated_delta.float().square().mean(dim=(1, 2)).sqrt()
    assert rms[0].item() == pytest.approx(0.35, rel=2e-5)
    assert rms[1].item() == pytest.approx(0.0, abs=1e-7)
    torch.testing.assert_close(actual[1], clean[1], atol=2e-6, rtol=2e-5)
    actual.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in adapter.parameters()
    )


def test_shared_xavier_bases_are_cached_and_block_deltas_train():
    torch.manual_seed(113)
    anima = _Anima().requires_grad_(False)
    adapter = SharedBaseKVStyleCrossAttention(
        context_dim=6,
        blocks=2,
        shared_bases=2,
        medoid_blocks=(0, 1),
        block_to_base=(0, 1),
        delta_rank=3,
        global_gain=1.0,
        relative_block_gain=(0.75, 1.25),
    )
    adapter.initialize_from_anima(anima)

    assert not torch.allclose(
        adapter.base_k[0].weight, anima.blocks[0].cross_attn.k_proj.weight
    )
    assert not torch.allclose(
        adapter.base_v[1].weight, anima.blocks[1].cross_attn.v_proj.weight
    )
    assert not torch.allclose(adapter.base_k[0].weight, adapter.base_k[1].weight)
    assert torch.isfinite(adapter.base_k[0].weight).all()
    assert torch.isfinite(adapter.base_v[1].weight).all()
    assert all(torch.count_nonzero(module.weight) > 0 for module in adapter.delta_k_up)
    assert all(torch.count_nonzero(module.weight) > 0 for module in adapter.delta_v_up)
    assert all(
        module.weight.float().square().mean().sqrt()
        < adapter.base_k[0].weight.float().square().mean().sqrt() * 0.1
        for module in adapter.delta_k_up
    )
    torch.testing.assert_close(adapter.alpha, torch.tensor([0.75, 1.25]))

    calls = [0, 0, 0, 0]
    hooks = []
    for index, module in enumerate([*adapter.base_k, *adapter.base_v]):
        hooks.append(module.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        ))
    style = torch.randn(2, 3, 6)
    adapter.set_style_context(style)
    outputs = []
    for index, block in enumerate(anima.blocks):
        outputs.append(adapter.merged_cross_attention(
            index, torch.randn(2, 5, 8), torch.randn(2, 4, 6),
            block.cross_attn, None,
        ))
    sum(output.square().mean() for output in outputs).backward()
    for hook in hooks:
        hook.remove()

    assert calls == [1, 1, 1, 1]
    assert adapter.base_k[0].weight.grad is not None
    assert adapter.delta_k_down[0].weight.grad is not None
    assert adapter.delta_v_down[1].weight.grad is not None
    assert adapter.delta_k_up[0].weight.grad is not None
    assert adapter.delta_v_up[1].weight.grad is not None
    assert adapter.base_mix_logits.grad is not None
    assert all(block.cross_attn.k_proj.weight.grad is None for block in anima.blocks)


def test_delayed_lr_holds_peak_until_requested_decay_step():
    assert _delayed_learning_rate_multiplier(250, 20_000, 500, 12_000, 0.1) == 0.5
    assert _delayed_learning_rate_multiplier(6_000, 20_000, 500, 12_000, 0.1) == 1.0
    assert _delayed_learning_rate_multiplier(12_000, 20_000, 500, 12_000, 0.1) == 1.0
    assert _delayed_learning_rate_multiplier(20_000, 20_000, 500, 12_000, 0.1) == pytest.approx(0.1)


def test_separate_style_guidance_is_not_scaled_by_text_cfg():
    negative = torch.tensor([1.0, 2.0])
    positive = torch.tensor([3.0, 5.0])
    styled = torch.tensor([4.0, 9.0])

    result = _compose_separate_text_style_guidance(
        negative,
        positive,
        styled,
        text_cfg=4.0,
        style_strength=1.5,
    )

    expected = negative + 4.0 * (positive - negative) + 1.5 * (styled - positive)
    assert torch.equal(result, expected)


def test_minimal_teacher_projection_prevents_zero_output_without_total_rms_fix():
    torch.manual_seed(127)
    teacher = torch.randn(4, 3, 5)
    collapsed = torch.zeros_like(teacher, requires_grad=True)
    config = {
        "residual_weight": 0.20,
        "projection_weight": 0.15,
        "projection_floor_start": 0.25,
        "projection_floor_end": 1.0,
        "projection_floor_start_step": 1,
        "projection_floor_end_step": 1000,
        "orthogonal_weight": 0.05,
        "orthogonal_ratio_maximum": 0.50,
    }
    collapsed_loss, collapsed_metrics = _minimal_native_teacher_objective(
        collapsed, teacher, config, step=1000
    )
    aligned_loss, aligned_metrics = _minimal_native_teacher_objective(
        teacher.clone().requires_grad_(True), teacher, config, step=1000
    )
    collapsed_loss.backward()

    assert collapsed.grad is not None
    assert torch.count_nonzero(collapsed.grad) > 0
    assert float(collapsed_metrics["native_teacher_projection_coefficient"]) == pytest.approx(0.0)
    assert float(collapsed_metrics["native_teacher_projection_floor_loss"]) == pytest.approx(1.0)
    assert float(aligned_metrics["native_teacher_projection_coefficient"]) == pytest.approx(1.0)
    assert float(aligned_loss) < float(collapsed_loss)


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
