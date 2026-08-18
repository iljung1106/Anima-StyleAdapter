import torch

from anima_style_data.global_query_style_tokenizer import (
    GlobalQueryMemoryStyleTokenizer,
    SlotPreservingGlobalQueryStyleTokenizer,
    attention_map_diversity_loss,
    reference_conditioned_diversity_loss,
)
from anima_style_data.dual_query_style_training import (
    _mean_one_clipped_weights,
    _native_artist_teacher_objective,
    _native_effect_weights_for_timesteps,
    _native_kv_functional_diversity_loss,
    _save_state,
    _scheduled_teacher_every,
)


def _model() -> GlobalQueryMemoryStyleTokenizer:
    return GlobalQueryMemoryStyleTokenizer(
        dim=64,
        spatial_tokens=16,
        global_tokens=4,
        artist_summary_tokens=2,
        output_tokens=8,
        heads=8,
        local_layers=1,
        cross_layers=2,
        ff_dim=128,
        output_rms_init=0.15,
    )


def _slot_preserving_model() -> SlotPreservingGlobalQueryStyleTokenizer:
    return SlotPreservingGlobalQueryStyleTokenizer(
        dim=64,
        spatial_tokens=16,
        global_tokens=4,
        artist_summary_tokens=2,
        output_tokens=8,
        heads=8,
        local_layers=1,
        cross_layers=2,
        ff_dim=128,
        slot_rank=8,
        output_rms_init=0.15,
    )


def test_global_query_tokenizer_is_reference_permutation_invariant():
    torch.manual_seed(7)
    model = _model().eval()
    references = torch.randn(3, 4, 22, 64)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(references, mask).tokens
    second = model(references[:, permutation], mask[:, permutation]).tokens
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_global_query_output_is_initialized_but_not_runtime_normalized():
    torch.manual_seed(11)
    model = _model()
    references = torch.randn(4, 2, 22, 64)
    mask = torch.ones(4, 2, dtype=torch.bool)
    output = model(references, mask)
    rms = output.tokens.float().square().mean().sqrt()
    assert 0.08 < float(rms) < 0.25
    sample_rms = output.tokens.float().square().mean(dim=(1, 2)).sqrt()
    assert float(sample_rms.std()) > 0
    loss = (
        output.tokens.float().square().mean()
        + 0.001 * attention_map_diversity_loss(output.attention_maps)
        + 0.001 * reference_conditioned_diversity_loss(output.tokens)
    )
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_slot_preserving_tokenizer_keeps_slots_and_reference_set_semantics():
    torch.manual_seed(19)
    model = _slot_preserving_model().eval()
    references = torch.randn(3, 4, 22, 64)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(references, mask).tokens
    second = model(references[:, permutation], mask[:, permutation]).tokens
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    centered = first - first.mean(dim=1, keepdim=True)
    slot_energy_ratio = (
        centered.square().mean().sqrt() / first.square().mean().sqrt()
    )
    assert float(slot_energy_ratio) > 0.10

    first.square().mean().backward()
    assert torch.isfinite(model.shared_output.weight.grad).all()
    assert torch.isfinite(model.slot_up.grad).all()


class _FakeCrossAttention(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)


class _FakeBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cross_attn = _FakeCrossAttention(dim)


class _FakeAnima(torch.nn.Module):
    def __init__(self, dim: int, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_FakeBlock(dim) for _ in range(blocks))


def test_functional_value_diversity_observes_native_kv_outputs():
    torch.manual_seed(23)
    anima = _FakeAnima(32)
    collapsed = torch.ones(4, 8, 32, requires_grad=True)
    collapsed_loss, collapsed_metrics = _native_kv_functional_diversity_loss(
        anima,
        collapsed,
        block_indices=[0, 1],
        slot_energy_floor=0.20,
        reference_energy_floor=0.10,
        decorrelation_fraction=0.10,
    )
    assert float(collapsed_metrics["functional_value_slot_energy_ratio"]) < 1e-6
    assert float(collapsed_metrics["functional_value_reference_energy_ratio"]) < 1e-6
    assert float(collapsed_loss) > 0.0

    diverse = torch.randn(4, 8, 32, requires_grad=True)
    diverse_loss, diverse_metrics = _native_kv_functional_diversity_loss(
        anima,
        diverse,
        block_indices=[0, 1],
        slot_energy_floor=0.20,
        reference_energy_floor=0.10,
        decorrelation_fraction=0.10,
    )
    assert float(diverse_metrics["functional_value_slot_energy_ratio"]) > 0.20
    assert float(diverse_metrics["functional_value_reference_energy_ratio"]) > 0.10
    diverse_loss.backward()
    assert torch.isfinite(diverse.grad).all()


def test_artist_teacher_objective_rejects_common_low_energy_shortcut():
    torch.manual_seed(29)
    teacher = torch.randn(4, 3, 8, 8)
    teacher = teacher - teacher.mean(dim=0, keepdim=True)
    training = {
        "center_student_teacher": True,
        "native_teacher_weight": 0.05,
        "native_teacher_ramp_steps": 0,
        "centered_energy_ratio_start": 0.50,
        "centered_energy_ratio_end": 0.50,
        "centered_energy_ramp_end_step": 1,
        "centered_energy_weight": 0.05,
        "centered_energy_weight_ramp_end_step": 1,
        "common_output_weight": 0.02,
        "common_output_start_step": 1,
        "common_output_ramp_end_step": 1,
        "common_output_threshold_start": 0.50,
        "common_output_threshold_end": 0.50,
        "common_output_threshold_end_step": 1,
        "artist_teacher_contrastive_weight": 0.02,
        "artist_teacher_contrastive_start_step": 1,
        "artist_teacher_contrastive_ramp_end_step": 1,
        "artist_teacher_ranking_weight": 0.02,
        "artist_teacher_ranking_start_step": 1,
        "artist_teacher_ranking_ramp_end_step": 1,
    }
    aligned_loss, aligned = _native_artist_teacher_objective(
        teacher.clone().requires_grad_(), teacher, training, step=2
    )
    collapsed_student = torch.ones_like(teacher, requires_grad=True) * 0.01
    collapsed_loss, collapsed = _native_artist_teacher_objective(
        collapsed_student, teacher, training, step=2
    )
    assert float(aligned["native_teacher_centered_student_to_target_rms"]) == 1.0
    assert float(aligned["native_teacher_artist_retrieval_top1"]) == 1.0
    assert float(collapsed["native_teacher_centered_energy_loss"]) > 0.0
    assert float(collapsed["native_teacher_artist_retrieval_top1"]) < 1.0
    assert float(collapsed_loss) > float(aligned_loss)
    collapsed_loss.backward()


def test_dense_teacher_schedule_thins_only_after_artist_alignment_phase():
    training = {
        "dual_domain_teacher_schedule": [
            {"end_step": 1000, "every": 1},
            {"end_step": 3000, "every": 2},
            {"end_step": 8000, "every": 4},
        ]
    }
    assert _scheduled_teacher_every(1, training) == 1
    assert _scheduled_teacher_every(1000, training) == 1
    assert _scheduled_teacher_every(1001, training) == 2
    assert _scheduled_teacher_every(3001, training) == 4


def test_native_effect_timestep_weights_are_bounded_and_interpolated():
    weights = _mean_one_clipped_weights(
        torch.tensor([0.1, 0.8, 1.0, 4.0]), minimum=0.75, maximum=1.33
    )
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert float(weights.min()) >= 0.75
    assert float(weights.max()) <= 1.330001
    interpolated = _native_effect_weights_for_timesteps(
        torch.tensor([0.1, 0.3, 0.5]),
        {
            "timesteps": torch.tensor([0.1, 0.5]),
            "weights": torch.tensor([0.75, 1.25]),
        },
    )
    assert torch.allclose(interpolated, torch.tensor([0.75, 1.0, 1.25]))


def test_checkpoint_preserves_optimizer_and_sparse_teacher_state(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    references = torch.randn(2, 1, 22, 64)
    mask = torch.ones(2, 1, dtype=torch.bool)
    model(references, mask).tokens.square().mean().backward()
    optimizer.step()
    path = tmp_path / "training_state.pt"
    _save_state(
        path,
        step=250,
        tokenizer=model,
        optimizer=optimizer,
        cfg={"model": {"include_artist_summary": True}},
        cache_summary={"slots": 84},
        trainer_state={
            "dual_domain_teacher_update_index": 62,
            "dual_domain_teacher_every": 4,
        },
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["step"] == 250
    assert state["optimizer"]["state"]
    assert state["trainer_state"]["dual_domain_teacher_update_index"] == 62
    assert state["trainer_state"]["dual_domain_teacher_every"] == 4
