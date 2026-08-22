from __future__ import annotations

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from anima_style_data.detail_style_cross_attention import (  # noqa: E402
    DetailPreservingTypedSlotReader,
    FreshKVStyleCrossAttention,
    SeparatedCommonArtistKVStyleCrossAttention,
    SharedBaseKVStyleCrossAttention,
    _leave_one_out_artist_center,
)
from anima_style_data.detail_style_training import (  # noqa: E402
    NativeScaleCommonOutputPenalty,
    _StyleAttenuationRecorder,
    _all_artist_teacher_infonce,
    _audit_student_prompts,
    _backward_adapter_only,
    _centered_native_magnitude_band,
    _common_native_teacher_objective,
    _compose_separate_text_style_guidance,
    _configure_separated_bootstrap_trainability,
    _delayed_learning_rate_multiplier,
    _effect_stage_metrics,
    _initial_performance_curriculum_state,
    _main_flow_total_magnitude_loss,
    _main_flow_projection_floor_loss,
    _minimal_native_teacher_objective,
    _native_effect_scales_for_timesteps,
    _native_bootstrap_status,
    _native_teacher_objective_config,
    _separated_bootstrap_phase,
    _separated_common_transition_status,
    _soft_common_output_objective,
    _teacher_domain_update,
    _teacher_direction_ranking_loss,
    _teacher_infonce_weight,
    _teacher_reference_count_weights,
    _update_performance_curriculum,
    _wrong_flow_ranking_loss,
)
from anima_style_data.query_style_tokenizer import (  # noqa: E402
    _sampling_reference_inputs,
)
from anima_style_data.detail_style_gradient_diagnostics import (  # noqa: E402
    _cosine_matrices,
    _gradient_sample_plan,
    _measure_gradient_sketches,
)


def test_teacher_domains_follow_weighted_schedule_with_local_indices():
    schedule = (0, 0, 0, 1)
    assert [
        _teacher_domain_update(schedule, update)
        for update in range(8)
    ] == [
        (0, 0), (0, 1), (0, 2), (1, 0),
        (0, 3), (0, 4), (0, 5), (1, 1),
    ]


def test_teacher_domain_schedule_rejects_invalid_updates():
    with pytest.raises(ValueError):
        _teacher_domain_update((), 0)
    with pytest.raises(ValueError):
        _teacher_domain_update((0,), -1)


def test_teacher_reference_count_curriculum_interpolates_without_hard_shift():
    training = {
        "teacher_reference_count_curriculum": {
            "start_step": 3001,
            "end_step": 4001,
            "before_weights": [0.0, 0.0, 0.0, 1.0],
            "start_weights": [0.1, 0.15, 0.0, 0.75],
            "end_weights": [0.5, 0.3, 0.0, 0.2],
        }
    }
    assert _teacher_reference_count_weights(training, 3000) == (
        0.0, 0.0, 0.0, 1.0
    )
    assert _teacher_reference_count_weights(training, 3001) == (
        0.1, 0.15, 0.0, 0.75
    )
    assert _teacher_reference_count_weights(training, 3501) == pytest.approx(
        (0.3, 0.225, 0.0, 0.475)
    )
    assert _teacher_reference_count_weights(training, 5000) == pytest.approx((
        0.5, 0.3, 0.0, 0.2
    ))


def test_teacher_infonce_weight_is_stronger_for_single_reference():
    training = {
        "teacher_infonce_weight": 0.2,
        "teacher_infonce_weight_by_reference_count": [0.35, 0.3, 0.25, 0.2],
    }
    assert _teacher_infonce_weight(training, 1) == 0.35
    assert _teacher_infonce_weight(training, 4) == 0.2


def test_exact_self_sampling_uses_only_the_target_reference():
    target = torch.randn(3, 84, 1024, dtype=torch.bfloat16)
    references, mask = _sampling_reference_inputs(
        {"cached_target_tokens": target}, "cpu", "self"
    )

    assert references.shape == (1, 1, 84, 1024)
    assert mask.tolist() == [[True]]
    torch.testing.assert_close(references[0, 0], target[0])

    with pytest.raises(ValueError):
        _sampling_reference_inputs(
            {"cached_target_tokens": target}, "cpu", "unknown"
        )


def test_main_flow_projection_floor_rewards_only_aligned_output():
    base = torch.zeros(2, 1, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
    prediction = torch.zeros_like(target, requires_grad=True)
    training = {
        "main_flow_projection_floor_start_step": 1,
        "main_flow_projection_floor_end_step": 500,
        "main_flow_projection_floor_start": 0.05,
        "main_flow_projection_floor_end": 0.20,
        "main_flow_projection_rms_upper": 2.0,
    }

    loss, metrics = _main_flow_projection_floor_loss(
        prediction, base, target, step=500, training=training
    )
    assert float(metrics["main_flow_projection_coefficient"]) == pytest.approx(0.0)
    assert float(loss) == pytest.approx(0.04)
    loss.backward()
    assert prediction.grad is not None
    assert float(prediction.grad[..., 0].mean()) < 0

    aligned = 0.20 * target
    aligned_loss, aligned_metrics = _main_flow_projection_floor_loss(
        aligned, base, target, step=500, training=training
    )
    assert float(aligned_metrics["main_flow_projection_coefficient"]) == pytest.approx(0.20)
    assert float(aligned_loss) == pytest.approx(0.0)

    orthogonal = torch.tensor([[[[0.0, 0.20]]], [[[0.0, 0.20]]]])
    orthogonal_loss, orthogonal_metrics = _main_flow_projection_floor_loss(
        orthogonal, base, target, step=500, training=training
    )
    assert float(orthogonal_metrics["main_flow_projection_coefficient"]) == pytest.approx(0.0)
    assert float(orthogonal_loss) == pytest.approx(0.04)


def test_main_flow_total_magnitude_accepts_orthogonal_equal_rms():
    base = torch.zeros(2, 1, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
    training = {
        "main_flow_magnitude_target_ratio": 1.0,
        "main_flow_magnitude_huber_beta": 0.10,
    }

    orthogonal = torch.tensor(
        [[[[0.0, 1.0]]], [[[0.0, 1.0]]]], requires_grad=True
    )
    equal_loss, equal_metrics = _main_flow_total_magnitude_loss(
        orthogonal, base, target, training=training
    )
    assert float(equal_metrics["main_flow_magnitude_rms_ratio"]) == pytest.approx(1.0)
    assert float(equal_loss) == pytest.approx(0.0)

    small = (0.1 * orthogonal.detach()).requires_grad_(True)
    small_loss, small_metrics = _main_flow_total_magnitude_loss(
        small, base, target, training=training
    )
    assert float(small_metrics["main_flow_magnitude_rms_ratio"]) == pytest.approx(0.1)
    assert float(small_loss) == pytest.approx(0.85)
    small_loss.backward()
    assert small.grad is not None
    assert float((small.grad * small.detach()).sum()) < 0


def test_centered_native_magnitude_rejects_common_output_shortcut():
    common = torch.ones(4, 1, 2, 2, requires_grad=True)
    common_loss, common_metrics = _centered_native_magnitude_band(
        common, torch.tensor(1.0), lower=0.5, upper=1.25
    )
    assert float(common_metrics["controlled_artist_magnitude_ratio"]) == pytest.approx(0.0)
    assert float(common_loss) == pytest.approx(0.25)

    distinct = torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],
        [[[-1.0, -1.0], [-1.0, -1.0]]],
        [[[1.0, -1.0], [1.0, -1.0]]],
        [[[-1.0, 1.0], [-1.0, 1.0]]],
    ])
    distinct_loss, distinct_metrics = _centered_native_magnitude_band(
        distinct, torch.tensor(1.0), lower=0.5, upper=1.25
    )
    assert float(distinct_metrics["controlled_artist_magnitude_ratio"]) == pytest.approx(1.0)
    assert float(distinct_loss) == pytest.approx(0.0)


def test_native_scale_common_output_penalty_uses_current_controlled_batch():
    penalty = NativeScaleCommonOutputPenalty()
    native = torch.ones(4, 1, 1, 2, 2)
    first = torch.full_like(native, 2.0, requires_grad=True)
    first_loss, first_metrics = penalty.objective(
        first, native, ratio_threshold=0.2
    )
    assert first_metrics[
        "native_teacher_common_output_ratio"
    ] == pytest.approx(2.0)
    assert float(first_loss.detach()) == pytest.approx(3.24)

    # No history is carried between controlled probes.
    second = torch.full_like(native, -2.0, requires_grad=True)
    second_loss, second_metrics = penalty.objective(
        second, native, ratio_threshold=0.2
    )
    assert second_metrics[
        "native_teacher_common_output_batch_ratio"
    ] == pytest.approx(2.0)
    assert second_metrics[
        "native_teacher_common_output_ratio"
    ] == pytest.approx(2.0)
    second_loss.backward()
    assert second.grad is not None
    assert second.grad.abs().sum() > 0

    centered = torch.cat((native[:2], -native[2:]), dim=0)
    centered_loss, centered_metrics = penalty.objective(
        centered, native, ratio_threshold=0.2
    )
    assert float(centered_loss) == pytest.approx(0.0)
    assert centered_metrics["native_teacher_common_output_ratio"] == pytest.approx(0.0)


def test_gradient_diagnostic_reports_known_conflict_and_orthogonality():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameters, plans = _gradient_sample_plan(
        {"reader": [parameter]}, samples_per_group=2, seed=17
    )
    sketches, statistics = _measure_gradient_sketches(
        {
            "aligned": parameter.sum(),
            "orthogonal": parameter[0] - parameter[1],
            "opposed": -parameter.sum(),
        },
        parameters,
        plans,
    )
    matrices = _cosine_matrices(sketches, plans)

    assert matrices["reader"]["aligned"]["orthogonal"] == pytest.approx(0.0)
    assert matrices["reader"]["aligned"]["opposed"] == pytest.approx(-1.0)
    assert matrices["all_trainable"]["aligned"]["opposed"] == pytest.approx(-1.0)
    assert statistics["aligned"]["reader_exact_norm"] == pytest.approx(2**0.5)


def test_joint_common_output_penalty_rejects_zero_artist_energy():
    penalty = NativeScaleCommonOutputPenalty()
    native = torch.tensor([1.0, -1.0, 1.0, -1.0]).reshape(4, 1, 1, 1)
    collapsed = torch.zeros_like(native, requires_grad=True)

    loss, metrics = penalty.objective(
        collapsed,
        native,
        ratio_threshold=0.6,
        artist_energy_floor=0.5,
    )
    loss.backward()

    assert float(metrics["native_teacher_common_output_common_loss"]) == 0.0
    assert float(metrics["native_teacher_artist_energy_loss"]) == pytest.approx(0.25)
    assert collapsed.grad is not None and collapsed.grad.abs().sum() > 0


def test_native_effect_scale_profile_interpolates_frozen_median_rms():
    weighting = {
        "timesteps": torch.tensor([0.0, 0.5, 1.0]),
        "median_rms": torch.tensor([1.0, 2.0, 4.0]),
    }
    result = _native_effect_scales_for_timesteps(
        torch.tensor([0.25, 0.75]), weighting
    )
    assert torch.allclose(result, torch.tensor([1.5, 3.0]))


def test_wrong_flow_ranking_only_pushes_the_wrong_path():
    base = torch.zeros(1, 1, 1, 1)
    target = torch.ones_like(base)
    correct = torch.full_like(base, 0.8, requires_grad=True)
    wrong = torch.full_like(base, 0.9, requires_grad=True)
    loss, metrics = _wrong_flow_ranking_loss(
        correct, wrong, base, target, margin=0.01
    )
    loss.backward()
    assert correct.grad is None
    assert wrong.grad is not None and float(wrong.grad) > 0
    assert metrics["advantage"] < 0


def test_leave_one_out_artist_center_preserves_batch_four_contrast_scale():
    values = torch.arange(1.0, 5.0).reshape(4, 1, 1)
    centered = _leave_one_out_artist_center(values)
    expected = torch.tensor([-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0]).reshape(4, 1, 1)
    torch.testing.assert_close(centered, expected)
    torch.testing.assert_close(centered.mean(dim=0), torch.zeros(1, 1))


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
    assert output.pooled_reconstruction is not None
    assert output.pooled_reconstruction_target is not None
    assert output.pooled_reconstruction.shape == (2, 12, 32)
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


def test_null_common_and_artist_residual_recompose_reference_attention():
    torch.manual_seed(108)
    anima = _Anima(blocks=1).requires_grad_(False)
    adapter = FreshKVStyleCrossAttention(
        context_dim=6, blocks=1, initial_alpha=0.2, null_tokens=3
    )
    adapter.initialize_from_anima(anima)
    common = torch.randn(1, 3, 6)
    with torch.no_grad():
        adapter.null_style_context.copy_(common)
    adapter.set_style_context(common.expand(2, -1, -1))
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)
    cross = anima.blocks[0].cross_attn

    clean = cross(hidden, None, text)
    styled = adapter.merged_cross_attention(0, hidden, text, cross, None)

    # ref == null makes artist residual exactly zero, but the explicit common
    # branch remains active rather than silently disabling one side.
    assert not torch.allclose(styled, clean)


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


def test_post_gate_teacher_distillation_only_records_selected_block():
    torch.manual_seed(110)
    anima = _Anima(blocks=2).requires_grad_(False)
    adapter = FreshKVStyleCrossAttention(
        context_dim=6, blocks=2, initial_alpha=0.2
    )
    adapter.initialize_from_anima(anima)
    style = torch.randn(4, 3, 6, requires_grad=True)
    adapter.set_style_context(style)
    adapter.set_teacher_context(
        torch.randn(4, 4, 6),
        block_indices=(1,),
        post_gate_distillation=True,
    )

    for block_index, block in enumerate(anima.blocks):
        adapter.merged_cross_attention(
            block_index,
            torch.randn(4, 5, 8),
            torch.randn(4, 4, 6),
            block.cross_attn,
            None,
        )
        adapter.record_gated_internal_teacher(
            block_index,
            torch.ones(4, 1, 1, 1, 8),
            (1, 1, 5),
        )

    loss, metrics = adapter.post_gate_teacher_loss(cosine_weight=0.15)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["post_gate_teacher_blocks"] == 1
    assert "post_gate_teacher_block_1_loss" in metrics
    assert "post_gate_teacher_block_0_loss" not in metrics
    assert style.grad is not None and style.grad.norm() > 0
    assert adapter.null_style_context.grad is not None
    assert adapter.null_style_context.grad.norm() > 0


def test_post_gate_direction_and_magnitude_are_independent():
    adapter = FreshKVStyleCrossAttention(
        context_dim=4, blocks=1, initial_alpha=0.2
    )
    teacher = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0, 0.0]],
    ])
    student = 0.25 * teacher
    adapter._record_post_gate_distillation(
        0, student, torch.zeros_like(student), teacher
    )

    direction_only, direction_metrics = adapter.post_gate_teacher_loss(
        direction_weight=1.0,
        magnitude_weight=0.0,
        common_weight=0.0,
        artist_common_leakage_weight=0.0,
    )
    adapter._record_post_gate_distillation(
        0, student, torch.zeros_like(student), teacher
    )
    with_magnitude, magnitude_metrics = adapter.post_gate_teacher_loss(
        direction_weight=1.0,
        magnitude_weight=1.0,
        magnitude_lower=0.5,
        common_weight=0.0,
        artist_common_leakage_weight=0.0,
    )

    assert float(direction_only) == pytest.approx(0.0, abs=1e-7)
    assert float(with_magnitude) > float(direction_only)
    assert float(
        direction_metrics["post_gate_teacher_block_0_projection_coefficient"]
    ) == pytest.approx(0.25)
    assert float(
        magnitude_metrics["post_gate_teacher_block_0_magnitude_lower_loss"]
    ) == pytest.approx(0.0625)


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
    recorder(0, "post_self_hidden", torch.zeros_like(student))
    recorder(0, "post_cross_hidden", torch.zeros_like(student))
    recorder(0, "post_mlp_hidden", torch.zeros_like(student))
    recorder.record_output_stage("final_layer_norm", torch.ones_like(student))
    recorder.mode = "style"
    recorder(0, "pre_o_style", student)
    recorder(0, "pre_o_teacher", teacher)
    recorder(0, "post_o_style", student)
    recorder(0, "post_o_teacher", teacher)
    recorder(0, "post_gate_style", student)
    recorder(0, "post_gate_teacher", teacher)
    recorder(0, "post_self_hidden", student)
    recorder(0, "post_cross_hidden", student)
    recorder(0, "post_mlp_hidden", student)
    recorder.record_output_stage("final_layer_norm", torch.ones_like(student) + student)
    captured = recorder.finish()[0]

    assert set(captured) == {
        "pre_o", "post_o", "post_gate", "post_self_hidden",
        "post_cross_hidden", "post_mlp_hidden"
    }
    assert captured["post_gate"]["teacher_projection"] == pytest.approx(2.0)
    assert recorder.output_metrics["final_layer_norm"]["effect_to_base_rms"] > 0


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

    # Each shared projection is evaluated once for reference tokens and once
    # for the learned null context, then reused across all blocks.
    assert calls == [2, 2, 2, 2]
    assert adapter.base_k[0].weight.grad is not None
    assert adapter.delta_k_down[0].weight.grad is not None
    assert adapter.delta_v_down[1].weight.grad is not None
    assert adapter.delta_k_up[0].weight.grad is not None
    assert adapter.delta_v_up[1].weight.grad is not None
    assert adapter.base_mix_logits.grad is not None
    assert adapter.log_common_gain.grad is not None
    assert adapter.log_artist_gain.grad is not None
    assert adapter.log_common_gain.grad.abs() > 0
    assert adapter.log_artist_gain.grad.abs() > 0
    assert adapter.null_style_context.grad is not None
    # The forward still recomposes the raw reference path at equal gains, but
    # stop-gradient on the artist subtraction prevents the common path from
    # becoming untrainable through exact algebraic cancellation.
    assert adapter.null_style_context.grad.norm() > 0
    assert all(block.cross_attn.k_proj.weight.grad is None for block in anima.blocks)


def test_separated_common_artist_bootstrap_routes_gradients_by_phase():
    torch.manual_seed(117)
    anima = _Anima().requires_grad_(False)
    reader = nn.Linear(6, 6)
    adapter = SeparatedCommonArtistKVStyleCrossAttention(
        context_dim=6,
        blocks=2,
        shared_bases=2,
        medoid_blocks=(0, 1),
        block_to_base=(0, 1),
        delta_rank=3,
        common_tokens=3,
        artist_null_residual=True,
        artist_residual_gain=2.0,
        null_tokens=3,
    )
    adapter.initialize_from_anima(anima)
    style = torch.randn(2, 3, 6)
    hidden = torch.randn(2, 5, 8)
    text = torch.randn(2, 4, 6)

    _configure_separated_bootstrap_trainability(
        reader, adapter, "common_only"
    )
    adapter.set_style_context(style)
    common_output = adapter.merged_cross_attention(
        0, hidden, text, anima.blocks[0].cross_attn, None
    )
    adapter.set_style_context(style.flip(0))
    common_output_other_reference = adapter.merged_cross_attention(
        0, hidden, text, anima.blocks[0].cross_attn, None
    )
    torch.testing.assert_close(common_output, common_output_other_reference)
    common_output.square().mean().backward()
    assert adapter.common_k[0].grad is not None
    assert adapter.common_v[0].grad is not None
    assert all(parameter.grad is None for parameter in adapter.artist_parameters())
    assert all(not parameter.requires_grad for parameter in reader.parameters())

    adapter.zero_grad(set_to_none=True)
    _configure_separated_bootstrap_trainability(reader, adapter, "combined")
    adapter.set_style_context(style)
    combined_output = adapter.merged_cross_attention(
        0, hidden, text, anima.blocks[0].cross_attn, None
    )
    combined_output.square().mean().backward()
    assert all(parameter.grad is None for parameter in adapter.common_parameters())
    assert any(parameter.grad is not None for parameter in adapter.artist_parameters())
    assert adapter.null_style_context.grad is not None
    assert all(parameter.requires_grad for parameter in reader.parameters())

    adapter.set_bootstrap_phase("artist_only")
    with torch.no_grad():
        adapter.null_style_context.zero_()
    adapter.set_style_context(style[:1])
    adapter.artist_residual_gain = 1.0
    artist_output_1x = adapter.merged_cross_attention(
        0, hidden[:1], text[:1], anima.blocks[0].cross_attn, None
    )
    adapter.artist_residual_gain = 2.0
    artist_output_2x = adapter.merged_cross_attention(
        0, hidden[:1], text[:1], anima.blocks[0].cross_attn, None
    )
    native_output = anima.blocks[0].cross_attn(hidden[:1], None, text[:1])
    torch.testing.assert_close(
        artist_output_2x - native_output,
        2.0 * (artist_output_1x - native_output),
    )

    with torch.no_grad():
        adapter.null_style_context.copy_(style[:1])
    adapter.set_style_context(style[:1])
    artist_null_output = adapter.merged_cross_attention(
        0, hidden[:1], text[:1], anima.blocks[0].cross_attn, None
    )
    torch.testing.assert_close(artist_null_output, native_output)
    assert _separated_bootstrap_phase(
        500, {"separated_component_bootstrap": {"enabled": True, "common_steps": 500}}
    ) == "common_only"
    assert _separated_bootstrap_phase(
        501, {"separated_component_bootstrap": {"enabled": True, "common_steps": 500}}
    ) == "combined"


def test_separated_common_transition_requires_stable_metric_windows():
    training = {
        "separated_component_bootstrap": {
            "enabled": True,
            "common_transition": {"enabled": True},
        }
    }
    assert _separated_bootstrap_phase(5_000, training, {}) == "common_only"
    rows = [{
        "native_teacher_common_cosine": 0.75,
        "native_teacher_common_projection_coefficient": 0.60,
        "native_teacher_common_rms_ratio": 1.05,
    }]
    first, consecutive, complete = _separated_common_transition_status(
        rows,
        {
            "minimum_steps": 500,
            "cosine": 0.70,
            "projection": 0.50,
            "rms_lower": 0.70,
            "rms_upper": 1.30,
            "consecutive_validations": 2,
        },
        step=500,
        previous_consecutive=0,
    )
    assert first["separated_common_window_passed"] == 1.0
    assert consecutive == 1 and not complete
    _, consecutive, complete = _separated_common_transition_status(
        rows,
        {
            "minimum_steps": 500,
            "cosine": 0.70,
            "projection": 0.50,
            "rms_lower": 0.70,
            "rms_upper": 1.30,
            "consecutive_validations": 2,
        },
        step=750,
        previous_consecutive=consecutive,
    )
    assert consecutive == 2 and complete
    state = {"separated_common_complete": True}
    assert _separated_bootstrap_phase(751, training, state) == "combined"

    restored = _initial_performance_curriculum_state(
        training,
        {
            "performance_curriculum": {
                "separated_common_complete": True,
                "separated_common_consecutive": 2,
                "separated_common_transition_step": 750,
                "separated_common_metrics": {
                    "native_teacher_common_cosine": 0.75,
                },
            }
        },
    )
    assert restored["separated_common_complete"] is True
    assert restored["separated_common_transition_step"] == 750


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


def test_centered_direction_and_magnitude_prevent_zero_output():
    torch.manual_seed(127)
    teacher = torch.randn(4, 3, 5)
    collapsed = torch.zeros_like(teacher, requires_grad=True)
    config = {
        "residual_weight": 0.20,
        "low_frequency_residual_weight": 0.0,
        "direction_weight": 1.0,
        "magnitude_weight": 0.10,
        "magnitude_floor_start": 1.0,
        "magnitude_floor_end": 1.0,
        "magnitude_floor_start_step": 1,
        "magnitude_floor_end_step": 1000,
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
    assert float(collapsed_metrics["native_teacher_magnitude_lower_loss"]) == pytest.approx(
        1.0, abs=1e-5
    )
    assert float(aligned_metrics["native_teacher_projection_coefficient"]) == pytest.approx(1.0)
    assert float(aligned_loss) < float(collapsed_loss)


def test_nested_teacher_objective_overrides_legacy_top_level_values():
    training = {
        "magnitude_weight": 0.75,
        "teacher_objective": {"magnitude_weight": 0.15},
    }

    resolved = _native_teacher_objective_config(training)

    assert resolved["magnitude_weight"] == pytest.approx(0.15)
    assert training["magnitude_weight"] == pytest.approx(0.75)


def test_centered_teacher_magnitude_has_a_weak_upper_bound():
    torch.manual_seed(128)
    teacher = torch.randn(4, 3, 5)
    config = {
        "residual_weight": 0.0,
        "low_frequency_residual_weight": 0.0,
        "direction_weight": 0.0,
        "magnitude_weight": 1.0,
        "magnitude_floor_start": 0.7,
        "magnitude_floor_end": 0.7,
        "magnitude_upper": 1.5,
        "magnitude_upper_weight": 0.25,
    }

    aligned_loss, aligned_metrics = _minimal_native_teacher_objective(
        teacher.clone(), teacher, config, step=1
    )
    excessive_loss, excessive_metrics = _minimal_native_teacher_objective(
        2.0 * teacher, teacher, config, step=1
    )

    assert float(aligned_loss) == pytest.approx(0.0, abs=1e-7)
    assert float(excessive_metrics["native_teacher_magnitude_upper_loss"]) == (
        pytest.approx(0.25)
    )
    assert float(excessive_loss) > float(aligned_loss)


def test_common_teacher_objective_aligns_direction_and_magnitude():
    torch.manual_seed(129)
    teacher = torch.randn(1, 3, 8, 8)
    config = {
        "common_direction_weight": 1.0,
        "magnitude_weight": 0.1,
        "magnitude_floor_start": 0.7,
        "magnitude_floor_end": 0.7,
        "magnitude_upper": 1.5,
    }

    aligned, aligned_metrics = _common_native_teacher_objective(
        teacher.clone().requires_grad_(True), teacher, config, step=1
    )
    collapsed_input = torch.zeros_like(teacher, requires_grad=True)
    collapsed, collapsed_metrics = _common_native_teacher_objective(
        collapsed_input, teacher, config, step=1
    )
    collapsed.backward()

    assert float(aligned_metrics["native_teacher_common_cosine"]) == pytest.approx(1.0)
    assert float(aligned_metrics["native_teacher_common_rms_ratio"]) == pytest.approx(1.0)
    assert float(collapsed_metrics["native_teacher_common_magnitude_lower_loss"]) > 0
    assert collapsed_input.grad is not None
    assert torch.count_nonzero(collapsed_input.grad) > 0
    assert float(collapsed) > float(aligned)


def test_teacher_direction_ranking_uses_cyclic_frozen_negatives():
    teacher = torch.eye(4).reshape(4, 1, 2, 2)
    student = teacher.clone().requires_grad_(True)

    loss, metrics = _teacher_direction_ranking_loss(
        student, teacher, margin=0.1
    )
    loss.backward()

    assert float(metrics["teacher_direction_accuracy"]) == pytest.approx(1.0)
    assert float(metrics["teacher_direction_advantage"]) > 0.5
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_all_artist_infonce_uses_every_wrong_teacher():
    teacher = torch.eye(4).reshape(4, 1, 2, 2)
    student = teacher[:2].clone().requires_grad_(True)
    labels = torch.tensor([0, 1])

    loss, metrics = _all_artist_teacher_infonce(
        student, teacher[:2], teacher, labels, temperature=0.1
    )
    loss.backward()

    assert student.grad is not None
    assert float(metrics["teacher_infonce_accuracy"]) == pytest.approx(1.0)
    assert float(metrics["teacher_infonce_cosine_gap"]) > 0.9


def test_soft_common_output_keeps_gradient_at_threshold():
    mean = torch.full((1, 2, 2), 0.6, requires_grad=True)
    loss, metrics = _soft_common_output_objective(
        mean, torch.tensor(1.0), ratio_threshold=0.6, softness=0.05
    )
    loss.backward()

    assert float(loss) > 0
    assert mean.grad is not None and torch.count_nonzero(mean.grad) > 0
    assert float(metrics["native_teacher_common_output_ratio"]) == pytest.approx(0.6)


def test_adapter_only_backward_does_not_touch_reader_leaf():
    reader = nn.Parameter(torch.tensor(2.0))
    adapter = nn.Parameter(torch.tensor(3.0))
    shared = reader * adapter
    _backward_adapter_only(shared.square(), [adapter], retain_graph=True)

    assert reader.grad is None
    assert adapter.grad is not None
    shared.backward()
    assert reader.grad is not None


def test_performance_curriculum_requires_consecutive_validation_windows():
    training = {
        "performance_curriculum": {
            "enabled": True,
            "stages": [
                {
                    "name": "bootstrap",
                    "min_references": 1,
                    "max_references": 1,
                    "reference_count_weights": [1.0],
                    "target_probability": 1.0,
                    "advance": {
                        "minimum_step": 100,
                        "final_centered_cosine": 0.2,
                        "native_projection": 0.4,
                        "common_output_ratio": 0.7,
                        "consecutive_validations": 2,
                    },
                },
                {
                    "name": "mixed",
                    "min_references": 1,
                    "max_references": 2,
                    "reference_count_weights": [0.5, 0.5],
                    "target_probability": 0.5,
                },
            ],
        }
    }
    state = {"enabled": True, "stage_index": 0, "consecutive_passes": 0}
    validation = {
        "heldout": {"paired_flow_improvement": 0.02},
        "wrong_artist": {"paired_flow_improvement": 0.0},
        "artist_effect": {
            "functional_artist_common_output_ratio": 0.6,
            "functional_artist_centered_student_rms": 1.0,
            "functional_artist_centered_teacher_rms": 1.0,
        },
    }
    teacher_rows = [{
        "native_teacher_cosine": 0.3,
        "native_teacher_projection_coefficient": 0.5,
        "native_teacher_common_output_ratio": 0.6,
        "teacher_infonce_accuracy": 0.75,
    }]

    first, changed = _update_performance_curriculum(
        training, state, validation, teacher_rows, step=100
    )
    assert not changed and first["consecutive_passes"] == 1
    second, changed = _update_performance_curriculum(
        training, state, validation, teacher_rows, step=200
    )
    assert changed and second["stage_index"] == 1


def test_native_bootstrap_requires_both_common_and_artist_alignment():
    config = {
        "minimum_steps": 500,
        "final_cosine": 0.3,
        "final_projection": 0.25,
        "common_cosine": 0.3,
        "common_projection": 0.25,
        "infonce_gap": 0.02,
        "consecutive_validations": 2,
    }
    rows = [{
        "native_teacher_cosine": 0.4,
        "native_teacher_projection_coefficient": 0.35,
        "native_teacher_common_cosine": 0.4,
        "native_teacher_common_projection_coefficient": 0.3,
        "teacher_infonce_cosine_gap": 0.05,
    }]
    first, consecutive, complete = _native_bootstrap_status(
        rows, config, step=500, previous_consecutive=0
    )
    second, consecutive, complete = _native_bootstrap_status(
        rows, config, step=750, previous_consecutive=consecutive
    )

    assert first["native_bootstrap_window_passed"] == 1.0
    assert second["native_bootstrap_complete"] == 1.0
    assert consecutive == 2 and complete


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
