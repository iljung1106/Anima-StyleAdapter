from __future__ import annotations

import torch
from torch import nn

from anima_style_data.kv_activation_generator import (
    ReferenceConditionedKVActivationGenerator,
    ReferenceConditionedLowRankKVOperator,
    _NativeAttentionProbe,
    _apply_dense_kv_operator,
    _centered_residual_loss,
    _cross_style_queue_diversity,
    _direct_delta_artist_split,
    _direct_delta_flow_due,
    _direct_delta_flow_updates_through,
    _functional_centered_attention_loss,
    _final_effect_constraints,
    _mean_teacher_operator,
    _mixture_target,
    _population_common_occupancy,
    _prediction_population_metrics,
    _same_artist_queue_infonce,
    _same_artist_signature_consistency,
    _whole_model_curriculum,
)
from anima_style_data.kv_activation_modulation import apply_kv_factors
from anima_style_data.kv_activation_sampling import NativeKVActivationInjector
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


def test_direct_delta_generator_can_preserve_reference_strength() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=16,
        context_dim=12,
        output_dim=20,
        blocks=2,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        normalize_style=False,
        normalize_attended=False,
    )
    assert isinstance(model.style_norm, nn.Identity)
    assert isinstance(model.output_norm, nn.Identity)


def test_direct_delta_generator_can_remove_softmax_invariant_block_key() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=16, context_dim=12, output_dim=20, blocks=2,
        hidden_dim=16, heads=4, ff_dim=32, use_block_embedding=False,
    )
    assert model.block_embedding is None


def test_direct_delta_rank32_head_keeps_token_conditioning_and_backpropagates() -> None:
    torch.manual_seed(19)
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=32,
        context_dim=24,
        output_dim=40,
        blocks=2,
        hidden_dim=32,
        heads=4,
        ff_dim=64,
        ff_layers=3,
        output_rank=4,
        output_init_scale=1e-3,
        normalize_style=False,
        normalize_attended=False,
    )
    style = torch.randn(2, 6, 32)
    context = torch.randn(2, 9, 24)
    output = model(style, context, 1)
    assert output.shape == (2, 2, 9, 40)
    assert model.output_head[1].down.out_features == 8
    assert model.output_head[1].up.shape == (2, 4, 40)
    assert not torch.allclose(output[:, :, 0], output[:, :, 1])
    output.square().mean().backward()
    assert model.output_head[1].up.grad is not None
    assert model.output_head[1].down.weight.grad is not None
    assert model.output_head[0].up.grad is None


def test_sparse_expert_generator_routes_kv_and_independent_qo_paths() -> None:
    torch.manual_seed(23)
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=32,
        context_dim=24,
        output_dim=40,
        blocks=2,
        hidden_dim=32,
        heads=4,
        ff_dim=64,
        ff_layers=2,
        output_rank=4,
        output_experts=6,
        output_top_k=2,
        output_init_scale=1e-3,
        enable_qo=True,
        stream_dim=40,
        stream_rank=3,
        stream_experts=4,
        stream_top_k=2,
    )
    style = torch.randn(2, 7, 32)
    context = torch.randn(2, 9, 24)
    stream = torch.randn(2, 11, 40)
    model.reset_routing_records()
    kv = model(style, context, 1)
    codes = model.prepare_stream_codes(style)
    q = model.stream_delta(stream, codes, 1, 0)
    o = model.stream_delta(stream, codes, 1, 1)
    balance, _, metrics = model.routing_auxiliary()
    assert kv.shape == (2, 2, 9, 40)
    assert q.shape == o.shape == stream.shape
    assert not torch.allclose(q, o)
    assert metrics["kv_router_entropy"] > 0
    assert metrics["qo_router_entropy"] > 0
    (kv.square().mean() + q.square().mean() + o.square().mean() + balance).backward()
    assert model.output_expert_up.grad is not None
    assert model.stream_expert_up.grad is not None
    assert model.stream_style_key[0].weight.grad is not None
    assert model.stream_style_key[1].weight.grad is not None


def test_sparse_expert_generator_can_disable_q_and_keep_o() -> None:
    torch.manual_seed(29)
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=32,
        context_dim=24,
        output_dim=40,
        blocks=2,
        hidden_dim=32,
        heads=4,
        ff_dim=64,
        output_rank=4,
        output_experts=8,
        output_top_k=4,
        output_init_scale=1e-3,
        enable_qo=True,
        enable_q=False,
        enable_o=True,
        stream_dim=40,
        stream_rank=3,
        stream_experts=6,
        stream_top_k=3,
    )
    style = torch.randn(2, 7, 32)
    stream = torch.randn(2, 11, 40)
    codes = model.prepare_stream_codes(style)
    q = model.stream_delta(stream, codes, 1, 0)
    o = model.stream_delta(stream, codes, 1, 1)
    assert torch.count_nonzero(q) == 0
    assert torch.count_nonzero(o) > 0


def test_dense_overload_balance_reaches_unselected_router_logits() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=16,
        context_dim=12,
        output_dim=20,
        blocks=1,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        output_rank=2,
        output_experts=8,
        output_top_k=2,
        output_init_scale=1e-3,
        expert_balance_cap=1.5,
    )
    logits = torch.zeros(2, 2, 8, requires_grad=True)
    indices, _, sparse, dense, selected = model._sparse_router(logits, 2)
    model.output_expert_usage.zero_()
    model.output_expert_load.zero_()
    model.output_expert_usage[..., 0] = 0.75
    model.output_expert_load[..., 0] = 1.0
    model.reset_routing_records()
    model._record_routing(
        "kv",
        model.output_expert_usage,
        model.output_expert_load,
        model.output_expert_selection_bias,
        0,
        sparse,
        dense,
        selected,
        2,
        logits,
    )
    balance, _, _ = model.routing_auxiliary()
    balance.backward()
    selected_experts = set(indices.flatten().tolist())
    unselected = [index for index in range(8) if index not in selected_experts]
    assert unselected
    assert torch.count_nonzero(logits.grad[..., unselected]) > 0


def test_selection_bias_changes_topk_but_not_selected_mixture_weights() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=16,
        context_dim=12,
        output_dim=20,
        blocks=1,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        output_rank=2,
        output_experts=4,
        output_top_k=2,
        output_init_scale=1e-3,
    )
    logits = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    bias = torch.tensor([[0.0, 0.0, 4.0, 0.0]])
    indices, weights, _, _, _ = model._sparse_router(logits, 2, bias)
    assert 2 in indices.flatten().tolist()
    expected = logits.gather(-1, indices).softmax(dim=-1)
    assert torch.allclose(weights, expected)


def test_population_loss_free_bias_updates_once_from_recorded_artists() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=16,
        context_dim=12,
        output_dim=20,
        blocks=1,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        output_rank=2,
        output_experts=4,
        output_top_k=2,
        output_init_scale=1e-3,
        expert_usage_decay=0.0,
        expert_bias_update_rate=0.1,
        expert_bias_max=1.0,
        expert_bias_population_update=True,
    )
    logits = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    _, _, sparse, dense, selected = model._sparse_router(logits, 2)
    model.set_routing_recording(False)
    model._record_routing(
        "kv",
        model.output_expert_usage,
        model.output_expert_load,
        model.output_expert_selection_bias,
        0,
        sparse,
        dense,
        selected,
        2,
        logits,
    )
    model.apply_routing_population_update()
    assert torch.count_nonzero(model.output_expert_selection_bias) == 0

    model.set_routing_recording(True)
    for _ in range(2):
        model._record_routing(
            "kv",
            model.output_expert_usage,
            model.output_expert_load,
            model.output_expert_selection_bias,
            0,
            sparse,
            dense,
            selected,
            2,
            logits,
        )
    metrics = model.apply_routing_population_update()
    bias = model.output_expert_selection_bias[0, 0]
    assert torch.all(bias[:2] < 0)
    assert torch.all(bias[2:] > 0)
    assert metrics["kv_population_groups"] == 2
    assert metrics["kv_max_violation"] == 1


def test_router_core_margin_has_gradient_at_uniform_logits() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=16,
        context_dim=12,
        output_dim=20,
        blocks=1,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        output_rank=2,
        output_experts=8,
        output_top_k=4,
        output_init_scale=1e-3,
        output_core_experts=2,
        output_core_margin=0.8,
        expert_specialization_steps=100,
    )
    model.set_routing_step(100)
    logits = torch.zeros(2, 2, 8, requires_grad=True)
    _, _, sparse, dense, selected = model._sparse_router(logits, 4)
    model._record_routing(
        "kv",
        model.output_expert_usage,
        model.output_expert_load,
        model.output_expert_selection_bias,
        0,
        sparse,
        dense,
        selected,
        4,
        logits,
    )
    _, specialization, metrics = model.routing_auxiliary()
    assert specialization > 0
    assert torch.allclose(
        metrics["kv_effective_experts"], torch.tensor(4.0), atol=1e-5
    )
    assert metrics["kv_core_margin_target"] == torch.tensor(0.8)
    specialization.backward()
    assert torch.count_nonzero(logits.grad) > 0


def test_same_artist_consistency_uses_cosine_and_rms_bands() -> None:
    matching, matching_metrics = _same_artist_signature_consistency(
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
        cosine_floor=0.75,
        rms_ratio_tolerance=1.5,
        magnitude_weight=0.25,
    )
    mismatching, mismatching_metrics = _same_artist_signature_consistency(
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        cosine_floor=0.75,
        rms_ratio_tolerance=1.5,
        magnitude_weight=0.25,
    )
    assert matching == 0
    assert matching_metrics["same_artist_signature_cosine"] > 0.99
    assert mismatching > 0
    assert mismatching_metrics["same_artist_signature_cosine"] == 0


def test_same_artist_queue_infonce_prefers_matching_disjoint_view() -> None:
    anchor = torch.tensor([1.0, 0.0])
    matching, matching_metrics = _same_artist_queue_infonce(
        anchor,
        torch.tensor([1.0, 0.0]),
        [torch.tensor([0.0, 1.0])],
        temperature=0.1,
    )
    mismatching, mismatching_metrics = _same_artist_queue_infonce(
        anchor,
        torch.tensor([0.0, 1.0]),
        [torch.tensor([1.0, 0.0])],
        temperature=0.1,
    )

    assert matching < mismatching
    assert (
        matching_metrics["same_artist_contrastive_positive"]
        > matching_metrics["same_artist_contrastive_hardest_negative"]
    )
    assert (
        mismatching_metrics["same_artist_contrastive_positive"]
        < mismatching_metrics["same_artist_contrastive_hardest_negative"]
    )


def test_final_effect_constraints_use_absolute_pairwise_cap_without_centering() -> None:
    teacher = torch.zeros(4, 1, 2, 4)
    for index in range(4):
        teacher[index].flatten()[index] = 1
    collapsed = torch.ones_like(teacher)
    loss, metrics = _final_effect_constraints(
        collapsed, teacher, common_cap=0.25, rms_lower=0.0, rms_upper=10.0
    )
    assert loss > 0.5
    assert metrics["positive_pairwise_cosine"] > 0.99
    assert metrics["common_cap_loss"] > 0.5


def test_final_effect_constraints_only_penalize_rms_outside_band() -> None:
    teacher = torch.ones(2, 1, 2, 4)
    inside, inside_metrics = _final_effect_constraints(
        teacher, teacher, common_cap=1.0, rms_lower=0.75, rms_upper=1.25
    )
    small, small_metrics = _final_effect_constraints(
        teacher * 0.25, teacher, common_cap=1.0,
        rms_lower=0.75, rms_upper=1.25,
    )
    large, large_metrics = _final_effect_constraints(
        teacher * 2.0, teacher, common_cap=1.0,
        rms_lower=0.75, rms_upper=1.25,
    )
    assert inside < 1e-7
    assert inside_metrics["rms_band_loss"] < 1e-7
    assert small_metrics["rms_lower_violation_rate"] == 1
    assert large_metrics["rms_upper_violation_rate"] == 1
    assert small > 0 and large > 0


def test_cross_style_queue_diversity_only_penalizes_cosine_above_cap() -> None:
    signature = torch.tensor([1.0, 0.0])
    collapsed, collapsed_cosine = _cross_style_queue_diversity(
        signature, [torch.tensor([1.0, 0.0])], cosine_cap=0.35
    )
    distinct, distinct_cosine = _cross_style_queue_diversity(
        signature, [torch.tensor([0.0, 1.0])], cosine_cap=0.35
    )
    assert collapsed > 0.4
    assert collapsed_cosine == 1
    assert distinct == 0
    assert distinct_cosine == 0


def test_population_common_occupancy_caps_shared_direction_without_centering() -> None:
    signature = torch.tensor([1.0, 0.0], requires_grad=True)
    collapsed, occupancy = _population_common_occupancy(
        signature,
        [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        occupancy_cap=0.30,
    )
    distinct, distinct_occupancy = _population_common_occupancy(
        torch.tensor([-1.0, 0.0]),
        [torch.tensor([1.0, 0.0])],
        occupancy_cap=0.30,
    )
    assert occupancy == 1
    assert collapsed > 0
    collapsed.backward()
    assert torch.count_nonzero(signature.grad) > 0
    assert distinct == 0
    assert distinct_occupancy == 0


def test_whole_model_curriculum_turns_block_loss_off_at_2000() -> None:
    assert _whole_model_curriculum(500)["block_weight"] == 1
    assert 0 < _whole_model_curriculum(1000)["block_weight"] < 1
    assert _whole_model_curriculum(2000)["block_weight"] == 0
    assert _whole_model_curriculum(2000)["rms_lower"] == 0.75


def test_prediction_population_metrics_detect_common_direction_collapse() -> None:
    common = torch.ones(4, 2, 3, 5)
    diverse = torch.stack(
        [
            torch.ones(2, 3, 5),
            -torch.ones(2, 3, 5),
            torch.cat([torch.ones(1, 3, 5), -torch.ones(1, 3, 5)]),
            torch.cat([-torch.ones(1, 3, 5), torch.ones(1, 3, 5)]),
        ]
    )
    collapsed = _prediction_population_metrics(common)
    separated = _prediction_population_metrics(diverse)
    assert collapsed["common_direction_occupancy"] > 0.99
    assert collapsed["artist_variance_fraction"] < 0.01
    assert separated["common_direction_occupancy"] < 0.01
    assert separated["artist_variance_fraction"] > 0.99


def test_common_direction_loss_penalizes_collapse_beyond_teacher() -> None:
    from anima_style_data.kv_activation_generator import (
        _excess_common_direction_loss,
    )

    collapsed = torch.ones(4, 2, 3, 5)
    separated = torch.zeros(4, 2, 3, 5)
    for index in range(4):
        separated[index].flatten()[index] = 1

    assert _excess_common_direction_loss(collapsed, separated) > 0.99
    assert _excess_common_direction_loss(separated, separated) < 1e-6
    assert _excess_common_direction_loss(collapsed, collapsed) < 1e-6


def test_direct_delta_split_keeps_every_mixture_teacher_in_train() -> None:
    artists = [f"artist-{index}" for index in range(10)]
    rows = [{
        "kind": "signed",
        "style_ids": ["artist-7", "artist-1"],
        "weights": [1.2, -0.2],
    }]
    train, validation, remapped = _direct_delta_artist_split(
        artists, rows, training_artists=6
    )
    assert len(train) == 6
    assert len(validation) == 4
    assert {1, 7}.issubset(train)
    assert remapped[0]["teacher_components"] == [7, 1]


def test_direct_delta_flow_schedule_is_dense_then_every_twenty() -> None:
    config = {
        "enabled": True,
        "warmup_updates": 1000,
        "warmup_every": 10,
        "every": 20,
        "offset": 1,
    }
    assert _direct_delta_flow_due(1, config)
    assert _direct_delta_flow_due(991, config)
    assert not _direct_delta_flow_due(1000, config)
    assert _direct_delta_flow_due(1001, config)
    assert _direct_delta_flow_due(1021, config)
    assert not _direct_delta_flow_due(1020, config)
    assert _direct_delta_flow_updates_through(1000, config) == 100
    assert _direct_delta_flow_updates_through(1020, config) == 101


def test_whole_model_flow_schedule_is_three_of_four_after_5000() -> None:
    config = {
        "enabled": True, "start_step": 5000,
        "cycle": 4, "slots": [0, 1, 2],
    }
    assert not _direct_delta_flow_due(5000, config)
    assert [_direct_delta_flow_due(step, config) for step in range(5001, 5005)] == [
        True, True, True, False
    ]


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


def test_low_rank_operator_reuses_kv_factors_and_modulates_qo() -> None:
    torch.manual_seed(43)
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=20,
        context_dim=12,
        output_dim=16,
        blocks=2,
        hidden_dim=16,
        heads=4,
        ff_dim=32,
        operator_layers=2,
        operator_rank=3,
        normalize_style=False,
        normalize_memory=False,
        enable_qo=True,
        stream_dim=16,
        stream_rank=4,
    )
    style = torch.randn(2, 7, 20)
    context = torch.randn(2, 5, 12)
    down, up = model.prepare_kv_factors(style)
    cached = model.apply_prepared_kv(context, down, up, 1)
    direct = model(style, context, 1)
    codes = model.prepare_stream_codes(style)
    q_delta = model.stream_delta(torch.randn(2, 6, 16), codes, 1, 0)

    assert down.shape == (2, 2, 2, 3, 12)
    assert up.shape == (2, 2, 2, 16, 3)
    assert torch.allclose(cached, direct, atol=2e-5, rtol=2e-4)
    assert codes.shape == (2, 2, 2, 16)
    assert q_delta.shape == (2, 6, 16)
    (cached.square().mean() + q_delta.square().mean()).backward()
    assert model.operator_queries.grad is not None
    assert model.stream_style_queries.grad is not None


class _ScaleNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


class _FakeCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = 2
        self.head_dim = 4
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(6, 8, bias=False)
        self.v_proj = nn.Linear(6, 8, bias=False)
        self.q_norm = _ScaleNorm()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(8, 8, bias=False)


class _FakeAnima(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.cross_attn = _FakeCrossAttention()
        self.blocks = nn.ModuleList([block])


class _ConstantDelta(nn.Module):
    def forward(
        self, style: torch.Tensor, context: torch.Tensor, block: int
    ) -> torch.Tensor:
        assert block == 0
        return context.new_ones(len(context), 2, context.shape[1], 8)


class _ConstantKVOQDelta(_ConstantDelta):
    enable_qo = True
    stream_dim = 8

    def prepare_stream_codes(self, style: torch.Tensor) -> torch.Tensor:
        return style.new_zeros(len(style), 1, 2, 1)

    def stream_delta(
        self,
        values: torch.Tensor,
        codes: torch.Tensor,
        block: int,
        kind: int,
    ) -> torch.Tensor:
        assert codes.shape == (len(values), 1, 2, 1)
        return torch.full_like(values, float(kind + 1))


def test_native_activation_injector_uses_styled_memory_without_base_input() -> None:
    torch.manual_seed(42)
    anima = _FakeAnima()
    injector = NativeKVActivationInjector(anima, _ConstantDelta())
    context = torch.randn(4, 5, 6)
    baseline = anima.blocks[0].cross_attn.k_proj(context)
    injector.set_style(torch.randn(2, 3, 7), strength=1.5)
    styled = anima.blocks[0].cross_attn.k_proj(context)
    torch.testing.assert_close(styled - baseline, torch.full_like(styled, 1.5))
    injector.close()


def test_native_activation_injector_can_drop_one_masked_block_for_a_group() -> None:
    anima = _FakeAnima()
    injector = NativeKVActivationInjector(anima, _ConstantDelta())
    context = torch.randn(3, 5, 6)
    baseline = anima.blocks[0].cross_attn.k_proj(context)
    injector.set_style(
        torch.randn(1, 3, 7), block_mask=torch.tensor([False])
    )

    torch.testing.assert_close(
        anima.blocks[0].cross_attn.k_proj(context), baseline
    )
    injector.close()


def test_native_activation_injector_adds_reference_conditioned_q_and_o() -> None:
    anima = _FakeAnima()
    injector = NativeKVActivationInjector(anima, _ConstantKVOQDelta())
    values = torch.randn(4, 5, 8)
    baseline_q = anima.blocks[0].cross_attn.q_proj(values)
    baseline_o = anima.blocks[0].cross_attn.output_proj(values)
    injector.set_style(torch.randn(2, 3, 7))

    torch.testing.assert_close(
        anima.blocks[0].cross_attn.q_proj(values) - baseline_q,
        torch.ones_like(baseline_q),
    )
    torch.testing.assert_close(
        anima.blocks[0].cross_attn.output_proj(values) - baseline_o,
        torch.full_like(baseline_o, 2.0),
    )
    injector.close()


def test_direct_generator_qo_path_is_style_conditioned_and_differentiable() -> None:
    model = ReferenceConditionedKVActivationGenerator(
        style_dim=12,
        context_dim=10,
        output_dim=8,
        blocks=2,
        hidden_dim=8,
        heads=2,
        ff_dim=16,
        enable_qo=True,
        stream_dim=8,
        stream_rank=3,
        output_init_scale=0.01,
    )
    style = torch.randn(2, 4, 12)
    values = torch.randn(2, 5, 8)
    codes = model.prepare_stream_codes(style)
    delta = model.stream_delta(values, codes, 1, 0)

    assert codes.shape == (2, 2, 2, 8)
    assert delta.shape == values.shape
    delta.square().mean().backward()
    assert model.stream_up[1][0].weight.grad is not None


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
