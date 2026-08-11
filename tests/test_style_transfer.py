from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from anima_style_data.style_transfer import (
    ProductionStyleLoader,
    SharedLowRankStyleAdapter,
    SlotSetAggregator,
    _archive_training_state,
    _episode_resampler_prototype_losses,
    _direction_anneal_multiplier,
    _flow_direction_loss,
    _optimize_frozen_anima,
    _pad_text_conditions,
    _per_sample_condition_comparison,
    _per_sample_cosine,
    _per_sample_flow_residual_metrics,
    _reference_flow_direction_loss,
    _reference_flow_rank_loss,
    _save_training_state,
    _self_reference_curriculum_state,
    _set_adapter_trainable_stage,
    _style_bootstrap_state,
    _soft_interval_loss,
    _summarize_scalar_samples,
    _symmetric_style_contrastive_loss,
    _timestep_interval_bounds,
    _uncached_no_grad_autocast,
    attach_style_adapter,
)


def test_direction_loss_stays_enabled_without_an_explicit_anneal():
    assert _direction_anneal_multiplier(1, {}) == 1.0
    assert _direction_anneal_multiplier(10_000, {}) == 1.0
    config = {
        "style_direction_anneal_start": 10,
        "style_direction_anneal_end": 20,
    }
    assert _direction_anneal_multiplier(10, config) == 1.0
    assert _direction_anneal_multiplier(15, config) == pytest.approx(0.5)
    assert _direction_anneal_multiplier(20, config) == 0.0


def test_flow_residual_diagnostics_separate_magnitude_direction_and_improvement():
    bypass = torch.zeros(2, 1, 1, 2)
    desired = torch.tensor([[[[1.0, -1.0]]], [[[2.0, -2.0]]]])
    target = bypass + desired
    prediction = bypass + 0.5 * desired

    metrics = _per_sample_flow_residual_metrics(prediction, bypass, target)

    torch.testing.assert_close(metrics["direction_cosine"], torch.ones(2))
    torch.testing.assert_close(metrics["desired_projection"], torch.full((2,), 0.5))
    torch.testing.assert_close(metrics["paired_improvement"], torch.full((2,), 0.75))


def test_condition_delta_cosine_and_summary_report_common_or_opposite_outputs():
    first = torch.tensor([[[1.0, -1.0]], [[2.0, 0.0]]])
    second = torch.tensor([[[1.0, -1.0]], [[-2.0, 0.0]]])

    torch.testing.assert_close(_per_sample_cosine(first, second), torch.tensor([1.0, -1.0]))
    summary = _summarize_scalar_samples([1.0, -1.0, 1.0])
    assert summary["mean"] == pytest.approx(1 / 3)
    assert summary["positive_fraction"] == pytest.approx(2 / 3)
    assert summary["samples"] == 3


def test_direct_condition_comparison_reports_advantage_and_repeatability_floor():
    bypass = torch.ones(2, 1, 1, 2)
    desired = torch.tensor([[[[1.0, -1.0]]], [[[1.0, -1.0]]]])
    target = bypass + desired
    first = target
    second = bypass + 0.5 * desired

    metrics = _per_sample_condition_comparison(first, second, bypass, target)
    torch.testing.assert_close(metrics["first_advantage"], torch.full((2,), 0.25))
    torch.testing.assert_close(
        metrics["difference_to_base_ratio"], torch.full((2,), 0.5)
    )
    torch.testing.assert_close(
        metrics["difference_to_desired_ratio"], torch.full((2,), 0.5)
    )

    repeat = _per_sample_condition_comparison(first, first, bypass, target)
    torch.testing.assert_close(repeat["first_advantage"], torch.zeros(2))
    torch.testing.assert_close(repeat["difference_to_base_ratio"], torch.zeros(2))


def test_reference_rank_loss_only_penalizes_insufficient_correct_advantage():
    bypass = torch.ones(2, 1, 1, 2)
    desired = torch.tensor([[[[1.0, -1.0]]], [[[1.0, -1.0]]]])
    target = bypass + desired
    correct = target
    wrong = bypass + 0.5 * desired

    loss, advantage = _reference_flow_rank_loss(
        correct, wrong, bypass, target, margin=0.20
    )
    torch.testing.assert_close(advantage, torch.tensor(0.25))
    torch.testing.assert_close(loss, torch.tensor(0.0))

    loss, advantage = _reference_flow_rank_loss(
        wrong, correct, bypass, target, margin=0.20
    )
    torch.testing.assert_close(advantage, torch.tensor(-0.25))
    torch.testing.assert_close(loss, torch.tensor(0.45))


def test_detached_teacher_does_not_poison_student_autocast_weight_cache():
    layer = nn.Linear(4, 4)
    inputs = torch.randn(2, 4)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with _uncached_no_grad_autocast("cpu"):
            teacher = layer(inputs)
        student = layer(inputs)

    assert not teacher.requires_grad
    assert student.requires_grad
    student.float().sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.norm() > 0


def test_reference_direction_loss_removes_common_residual_and_trains_correct_only():
    common = torch.tensor([[[[2.0, 2.0]]]])
    desired = torch.tensor([[[[1.0, -1.0]]]])
    wrong = common.clone().requires_grad_(True)
    aligned = (common + 0.5 * desired).requires_grad_(True)
    target = common + desired

    aligned_loss = _reference_flow_direction_loss(
        aligned, wrong, target, epsilon=0.01
    )
    assert aligned_loss < 0.001
    aligned_loss.backward()
    assert aligned.grad is not None
    assert wrong.grad is None

    orthogonal = (common + torch.tensor([[[[1.0, 1.0]]]])).requires_grad_(True)
    orthogonal_loss = _reference_flow_direction_loss(
        orthogonal, wrong, target, epsilon=0.01
    )
    assert orthogonal_loss > 0.9


def test_reference_direction_loss_live_wrong_branch_cancels_common_gradient():
    common = torch.tensor([[[[2.0, 2.0]]]])
    desired = torch.tensor([[[[1.0, -1.0]]]])
    correct = (common + torch.tensor([[[[0.2, 0.1]]]])).requires_grad_(True)
    wrong = common.clone().requires_grad_(True)
    target = common + desired

    loss = _reference_flow_direction_loss(
        correct, wrong, target, epsilon=0.05, wrong_has_grad=True
    )
    loss.backward()

    assert correct.grad is not None
    assert wrong.grad is not None
    assert correct.grad.norm() > 0
    torch.testing.assert_close(wrong.grad, -correct.grad)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        return x.float().square().mean(-1, keepdim=True).to(x.dtype)


class _LegacyAttention(nn.Module):
    def __init__(self, *, self_attention: bool):
        super().__init__()
        self.is_selfattn = self_attention
        self.n_heads = 2
        self.head_dim = 4
        self.qkv_format = "bshd"
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()

    def compute_qkv(self, x, context=None, rope_emb=None):
        context = x if context is None else context
        return tuple(
            layer(value).unflatten(-1, (self.n_heads, self.head_dim))
            for layer, value in (
                (self.q_proj, x),
                (self.k_proj, context),
                (self.v_proj, context),
            )
        )


class _FrozenAnimaStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(8)
        self.self_attention = _LegacyAttention(self_attention=True)
        self.cross_attention = _LegacyAttention(self_attention=False)


def test_frozen_anima_optimizations_preserve_projection_outputs_in_activation_dtype():
    model = _FrozenAnimaStub().requires_grad_(False)
    x = torch.randn(2, 5, 8)
    context = torch.randn(2, 3, 8)
    expected_self = model.self_attention.compute_qkv(x)
    expected_cross = model.cross_attention.compute_qkv(x, context)

    counts = _optimize_frozen_anima(
        model, low_precision_rmsnorm=True, fuse_attention_projections=True
    )

    assert counts == {
        "low_precision_rmsnorm": 1,
        "fused_self_attention": 1,
        "fused_cross_attention": 1,
        "final_layer_dtype_guard": 0,
    }
    for actual, expected in zip(
        model.self_attention.compute_qkv(x), expected_self, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        model.cross_attention.compute_qkv(x, context), expected_cross, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    bf16 = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    assert model.norm(bf16).dtype == torch.bfloat16
    assert model.norm(bf16.float()).dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.bfloat16
    assert not hasattr(model.self_attention, "q_proj")
    assert hasattr(model.cross_attention, "q_proj")


def test_self_reference_curriculum_has_explicit_terminal_phase():
    config = {
        "gate_only_steps": 2,
        "self_reference_steps": 5,
        "target_anneal_end": 9,
        "oracle_distill_end": 9,
    }
    first = _self_reference_curriculum_state(1, config)
    assert first["phase"] == "output_bootstrap_self_reference"
    assert first["target_only"] and first["target_probability"] == 1.0
    opened = _self_reference_curriculum_state(3, config)
    assert opened["phase"] == "full_self_reference"
    assert not opened["gate_only"]
    ramp = _self_reference_curriculum_state(7, config)
    assert ramp["phase"] == "oracle_target_anneal"
    assert ramp["target_probability"] == pytest.approx(0.5)
    assert ramp["oracle_required"]
    final = _self_reference_curriculum_state(9, config)
    assert final["phase"] == "target_excluded"
    assert final["target_probability"] == 0.0
    assert not final["oracle_required"]


def test_gate_only_stage_then_opens_entire_adapter():
    adapter = SharedLowRankStyleAdapter(
        style_dim=8, slots=2, hidden_dim=16, heads=4, blocks=28, rank=2,
        aggregator_heads=2, aggregator_layers=1, gate_dim=4,
    )
    _set_adapter_trainable_stage(adapter, gate_only=True)
    assert all(parameter.requires_grad for parameter in adapter.gate.parameters())
    assert adapter.shared_o.weight.requires_grad
    assert all(parameter.requires_grad for parameter in adapter.o_up.parameters())
    assert not adapter.shared_k.weight.requires_grad
    assert not next(adapter.aggregator.parameters()).requires_grad
    _set_adapter_trainable_stage(adapter, gate_only=False)
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_empirical_timestep_interval_only_penalizes_outside_values():
    calibration = {
        "timestep_edges": [0.0, 0.5, 1.0],
        "bins": [
            {"p25": 0.02, "median": 0.03, "p75": 0.04},
            {"p25": 0.05, "median": 0.06, "p75": 0.07},
        ],
    }
    timesteps = torch.tensor([0.1, 0.8])
    lower, upper = _timestep_interval_bounds(timesteps, calibration)
    torch.testing.assert_close(lower, torch.tensor([0.02, 0.05]))
    torch.testing.assert_close(upper, torch.tensor([0.04, 0.07]))
    inside = _soft_interval_loss(torch.tensor([0.03, 0.06]), lower, upper, beta=0.01)
    outside = _soft_interval_loss(torch.tensor([0.0, 0.10]), lower, upper, beta=0.01)
    assert inside == 0
    assert outside > 0


def test_flow_direction_bootstrap_escapes_exact_zero_output():
    delta = torch.zeros(2, 3, 4, requires_grad=True)
    desired = torch.randn_like(delta)
    loss = _flow_direction_loss(
        delta, desired, torch.ones(2), epsilon=0.01
    )
    loss.backward()
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.norm() > 0


def test_style_contrastive_prefers_matching_artist_tokens():
    targets = torch.eye(4).reshape(4, 1, 4)
    matching = _symmetric_style_contrastive_loss(targets, targets, 0.1)
    shuffled = _symmetric_style_contrastive_loss(targets.roll(1, 0), targets, 0.1)
    assert matching < shuffled


def test_style_bootstrap_ramps_then_anneals_to_zero():
    config = {
        "style_output_ratio_floor": 0.05,
        "style_magnitude_start_step": 10,
        "style_magnitude_ramp_steps": 100,
        "target_reference_probability": 1.0,
        "style_aux_anneal_start": 200,
        "style_aux_anneal_end": 300,
    }
    assert _style_bootstrap_state(0, config) == (1.0, 0.0, 1.0)
    assert _style_bootstrap_state(10, config) == (1.0, 0.0, 1.0)
    assert _style_bootstrap_state(110, config) == (1.0, 0.05, 1.0)
    auxiliary, floor, probability = _style_bootstrap_state(250, config)
    assert auxiliary == pytest.approx(0.5)
    assert floor == pytest.approx(0.025)
    assert probability == pytest.approx(0.5)
    assert _style_bootstrap_state(300, config) == (0.0, 0.0, 0.0)


def test_text_conditions_restore_animas_fixed_zero_padding():
    first = torch.randn(3, 8)
    second = torch.randn(5, 8)
    padded = _pad_text_conditions([first, second], 7)
    assert padded.shape == (2, 7, 8)
    torch.testing.assert_close(padded[0, :3], first)
    torch.testing.assert_close(padded[1, :5], second)
    assert torch.count_nonzero(padded[0, 3:]) == 0
    assert torch.count_nonzero(padded[1, 5:]) == 0


def test_slot_set_aggregator_is_reference_order_invariant():
    torch.manual_seed(7)
    model = SlotSetAggregator(slots=3, dim=12, heads=3, layers=2).eval()
    values = torch.randn(2, 4, 3, 12)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(values, mask)
    second = model(values[:, permutation], mask[:, permutation])
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)


def test_cross_slot_mixer_couples_pooled_slots():
    torch.manual_seed(11)
    model = SlotSetAggregator(
        slots=3, dim=12, heads=3, layers=1, slot_mixer_layers=1
    ).eval()
    values = torch.randn(1, 2, 3, 12)
    mask = torch.ones(1, 2, dtype=torch.bool)
    first = model(values, mask)
    changed = values.clone()
    changed[:, :, 1] *= -1.0
    second = model(changed, mask)
    # Changing slot 1 affects slot 0 only through the post-pooling slot mixer.
    assert not torch.allclose(first[:, 0], second[:, 0])


def test_episode_sampler_never_uses_target_as_reference():
    loader = ProductionStyleLoader.__new__(ProductionStyleLoader)
    loader.seed = 41
    loader.batch_size = 1
    loader.min_references = 1
    loader.max_references = 3
    loader.bucket_keys = [(32, 32)]
    loader.bucket_weights = [4]
    loader.buckets = {(32, 32): [10, 11, 12, 13]}
    loader.style_by_id = {
        image_id: {"artist": "artist", "style_id": "style"}
        for image_id in (10, 11, 12, 13)
    }
    loader.by_style = {"style": [10, 11, 12, 13]}
    loader.text_variants = {image_id: [0, 1] for image_id in (10, 11, 12, 13)}
    for step in range(12):
        episode = loader.episodes_for_step(step)[0]
        assert episode.target_id not in episode.reference_ids
        assert 1 <= len(episode.reference_ids) <= 3
        assert len(set(episode.reference_ids)) == len(episode.reference_ids)


class _FakeCrossAttention(nn.Module):
    def __init__(self, hidden: int, context: int | None = None):
        super().__init__()
        context = hidden if context is None else context
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(context, hidden, bias=False)
        self.v_proj = nn.Linear(context, hidden, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.output_proj = nn.Linear(hidden, hidden, bias=False)
        self.output_dropout = nn.Identity()


def test_zero_output_projection_gets_direction_before_opening_kv_path():
    torch.manual_seed(3)
    adapter = SharedLowRankStyleAdapter(
        style_dim=8, slots=2, hidden_dim=16, output_dim=16,
        heads=4, blocks=28, rank=2,
        aggregator_heads=2, aggregator_layers=1, style_dropout=0.0, gate_dim=4,
    )
    cross = _FakeCrossAttention(16)
    cross.requires_grad_(False)
    adapter.set_style_tokens(torch.randn(2, 2, 8))
    query = torch.randn(2, 5, 16)
    timestep = torch.randn(2, 1, 16)

    # The full style residual is exactly neutral, but its output matrix gets an
    # informative gradient immediately instead of blocking the whole K/V path
    # behind a zero scalar gate.
    initial = adapter.attend(0, query, timestep, cross)
    assert torch.count_nonzero(initial) == 0
    initial.sum().backward()
    assert adapter.shared_o.weight.grad is not None
    assert adapter.shared_o.weight.grad.norm() > 0
    assert adapter.shared_k.weight.grad is None or adapter.shared_k.weight.grad.norm() == 0

    adapter.zero_grad(set_to_none=True)
    nn.init.normal_(adapter.shared_o.weight, std=0.01)
    output = adapter.attend(0, query, timestep, cross)
    assert output.shape == query.shape
    output.square().mean().backward()
    assert adapter.shared_k.weight.grad is not None
    assert cross.q_proj.weight.grad is None


def test_pretrained_block_projection_starts_zero_then_trains_style_alignment():
    torch.manual_seed(17)
    adapter = SharedLowRankStyleAdapter(
        style_dim=8, slots=2, hidden_dim=16, output_dim=16,
        heads=4, blocks=28, rank=2,
        aggregator_heads=2, aggregator_layers=1, style_dropout=0.0, gate_dim=4,
        projection_mode="pretrained_block_lora", context_dim=8,
    )
    cross = _FakeCrossAttention(16, context=8).requires_grad_(False)
    adapter.set_style_tokens(torch.randn(2, 2, 8))
    query = torch.randn(2, 5, 16)
    timestep = torch.randn(2, 1, 16)

    initial = adapter.attend(0, query, timestep, cross)
    assert torch.count_nonzero(initial) == 0
    initial.sum().backward()
    assert adapter.gate[-1].weight.grad is not None
    assert adapter.gate[-1].weight.grad.norm() > 0
    assert adapter.style_context_proj.weight.grad is not None
    assert adapter.style_context_proj.weight.grad.norm() == 0

    adapter.zero_grad(set_to_none=True)
    with torch.no_grad():
        adapter.gate[-1].bias.fill_(0.25)
    output = adapter.attend(0, query, timestep, cross)
    assert output.shape == query.shape
    assert torch.count_nonzero(output) > 0
    output.square().mean().backward()
    assert adapter.style_context_proj.weight.grad is not None
    assert adapter.style_context_proj.weight.grad.norm() > 0
    assert adapter.k_up[0].weight.grad is not None
    assert cross.k_proj.weight.grad is None
    assert cross.output_proj.weight.grad is None


def test_attach_patches_all_28_blocks_without_copying_adapter():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()

        def _forward(self):
            return None

    anima = nn.Module()
    anima.blocks = nn.ModuleList([Block() for _ in range(28)])
    adapter = SharedLowRankStyleAdapter(
        style_dim=8, slots=2, hidden_dim=16, heads=4, blocks=28, rank=2,
        aggregator_heads=2, aggregator_layers=1, gate_dim=4,
    )
    attach_style_adapter(anima, adapter)
    assert anima.style_adapter is adapter
    assert [block.__dict__["_style_block_index"] for block in anima.blocks] == list(range(28))
    assert all(isinstance(block._forward, types.MethodType) for block in anima.blocks)


def test_training_state_is_atomic_and_archivable(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    current = tmp_path / "training_state.pt"
    archive = tmp_path / "checkpoints" / "step-0000001.pt"
    archive.parent.mkdir()
    _save_training_state(current, 1, model, optimizer, {"name": "test"})
    _archive_training_state(current, archive)
    state = torch.load(archive, map_location="cpu", weights_only=False)
    assert state["step"] == 1
    assert state["config"] == {"name": "test"}
    assert set(state["adapter"]) == set(model.state_dict())


def test_episode_resampler_prototypes_use_all_references_and_slots():
    references = torch.randn(3, 4, 8, 16, requires_grad=True)
    targets = torch.randn(3, 8, 16, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
    )

    joint, slot = _episode_resampler_prototype_losses(
        references, mask, targets, 0.07
    )
    (joint + slot).backward()

    assert torch.isfinite(joint)
    assert torch.isfinite(slot)
    assert references.grad is not None and bool(torch.isfinite(references.grad).all())
    assert targets.grad is not None and bool(torch.isfinite(targets.grad).all())

    duplicate_joint, duplicate_slot = _episode_resampler_prototype_losses(
        references, mask, targets, 0.07, ["same", "same", "other"]
    )
    assert torch.isfinite(duplicate_joint)
    assert torch.isfinite(duplicate_slot)
