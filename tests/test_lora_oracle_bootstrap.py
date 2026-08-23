import torch
import torch.nn.functional as F

from anima_style_data.lora_oracle_bootstrap import (
    _cross_view_functional_objective,
    _functional_effect_fingerprints,
    _FixedOracleCodeReader,
    _artist_centered_oracle_objective,
    _common_artist_oracle_objective,
    _oracle_component_regression_objective,
    _cross_view_artist_objective,
    _interpolate_oracle_visual,
    _oracle_code_alignment_objective,
    _materialize_reader_code_bank,
    _oracle_adapter_initial_state,
    _sample_diverse_functional_batch,
    _timestep_objective_weights,
    _weighted_timestep_index,
    _piecewise_linear_value,
    OracleVisualProjector,
    _ProjectedReader,
)


def test_weighted_timestep_sampling_emphasizes_hard_bins_reproducibly():
    weights = [2.0, 2.0, 1.0, 1.0]
    first = [
        _weighted_timestep_index(step=step, seed=17, count=4, weights=weights)
        for step in range(1, 4001)
    ]
    second = [
        _weighted_timestep_index(step=step, seed=17, count=4, weights=weights)
        for step in range(1, 4001)
    ]

    assert first == second
    assert first.count(0) > 1.7 * first.count(2)
    assert first.count(1) > 1.7 * first.count(3)


def test_timestep_multiplier_changes_direction_not_magnitude():
    base = {
        "low_direction": 2.0,
        "global_direction": 1.5,
        "infonce": 0.75,
        "low_magnitude": 0.10,
        "zero_mean": 1.0,
    }
    result = _timestep_objective_weights(
        base,
        timestep_index=0,
        direction_multipliers=[1.75, 1.0],
        common_multipliers=[1.5, 1.0],
    )

    assert result["low_direction"] == 3.5
    assert result["global_direction"] == 2.625
    assert result["infonce"] == 1.3125
    assert result["low_magnitude"] == 0.10
    assert result["zero_mean"] == 1.5


def test_cross_view_artist_objective_recognizes_matching_artists():
    left = torch.eye(8).reshape(8, 1, 8)
    loss, metrics = _cross_view_artist_objective(
        left, left.clone(), temperature=0.10
    )

    assert torch.isfinite(loss)
    assert float(metrics["accuracy"]) == 1.0
    assert float(metrics["cosine_gap"]) > 0.5


def test_cross_view_functional_objective_rejects_collapsed_effects():
    left = torch.eye(8).reshape(8, 1, 8)
    target_rms = (left - left.mean(dim=0, keepdim=True)).square().mean().sqrt()
    exact_loss, exact_metrics = _cross_view_functional_objective(
        left,
        left.clone(),
        temperature=0.10,
        target_rms=target_rms,
        magnitude_weight=0.10,
    )
    collapsed = (left * 1e-4).requires_grad_()
    collapsed_loss, collapsed_metrics = _cross_view_functional_objective(
        collapsed,
        collapsed.clone(),
        temperature=0.10,
        target_rms=target_rms,
        magnitude_weight=0.10,
    )

    assert float(exact_metrics["accuracy"]) == 1.0
    assert float(exact_metrics["effect_to_lora_teacher_rms"]) > 0.9
    assert float(collapsed_metrics["effect_to_lora_teacher_rms"]) < 1e-3
    assert float(collapsed_loss) > float(exact_loss)
    collapsed_loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_functional_sampler_spreads_a_random_candidate_pool():
    effects = torch.zeros(6, 1, 1, 4, 8, 8)
    effects[:3, ..., 0, :, :] = 1
    effects[3:, ..., 1, :, :] = 1
    fingerprints = _functional_effect_fingerprints(effects)
    similarity = fingerprints @ fingerprints.t()
    selected, mean_similarity = _sample_diverse_functional_batch(
        list(range(6)),
        similarity,
        2,
        rng=__import__("random").Random(7),
        pool_size=6,
    )

    assert (selected[0] < 3) != (selected[1] < 3)
    assert mean_similarity < 0


def test_piecewise_linear_value_supports_a_plateau_then_ramp():
    points = [[0, 0.0], [250, 0.5], [1000, 0.5], [1500, 1.0]]

    assert _piecewise_linear_value(125, points) == 0.25
    assert _piecewise_linear_value(750, points) == 0.5
    assert _piecewise_linear_value(1250, points) == 0.75
    assert _piecewise_linear_value(2000, points) == 1.0


def test_oracle_objective_prefers_distinct_centered_effects():
    teacher = torch.eye(8).reshape(8, 1, 8) + 0.3
    exact = teacher - teacher.mean(dim=0, keepdim=True)
    exact_loss, exact_metrics = _artist_centered_oracle_objective(
        exact, teacher, {"infonce": 0.5}
    )
    collapsed = torch.zeros_like(exact, requires_grad=True)
    collapsed_loss, collapsed_metrics = _artist_centered_oracle_objective(
        collapsed, teacher, {"infonce": 0.5}
    )

    assert float(exact_metrics["centered_cosine"]) > 0.999
    assert float(exact_metrics["functional_infonce_accuracy"]) == 1.0
    assert float(collapsed_loss) > float(exact_loss) + 1.0
    assert float(collapsed_metrics["centered_student_to_teacher_rms"]) < 1e-5
    collapsed_loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_oracle_objective_penalizes_artist_branch_batch_mean():
    teacher = torch.eye(4).reshape(4, 1, 4)
    centered = teacher - teacher.mean(dim=0, keepdim=True)
    shifted = centered + 0.5
    centered_loss, centered_metrics = _artist_centered_oracle_objective(
        centered, teacher, {"infonce": 0.0}
    )
    shifted_loss, shifted_metrics = _artist_centered_oracle_objective(
        shifted, teacher, {"infonce": 0.0}
    )

    assert float(centered_metrics["artist_common_zero_loss"]) < 1e-8
    assert float(shifted_metrics["artist_common_zero_loss"]) > 0.1
    assert float(shifted_loss) > float(centered_loss)


def test_component_regression_preserves_nonzero_teacher_mean():
    teacher = torch.ones(4, 2, 8) * 0.75
    exact_loss, exact_metrics = _oracle_component_regression_objective(
        teacher.clone(), teacher, {}
    )
    zero_loss, zero_metrics = _oracle_component_regression_objective(
        torch.zeros_like(teacher), teacher, {}
    )

    assert float(exact_loss) < 1e-6
    assert float(exact_metrics["cosine"]) > 0.999
    assert float(zero_loss) > float(exact_loss) + 1.0
    assert float(zero_metrics["student_to_teacher_rms"]) < 1e-5


def test_combined_objective_requires_common_and_artist_components():
    common = torch.ones(1, 1, 8) * 0.5
    artist = torch.eye(4, 8).reshape(4, 1, 8)
    teacher = common + artist
    exact_loss, exact_metrics = _common_artist_oracle_objective(
        teacher.clone(), common.clone(), teacher, common, {}
    )
    centered_only = artist - artist.mean(dim=0, keepdim=True)
    incomplete_loss, incomplete_metrics = _common_artist_oracle_objective(
        centered_only, torch.zeros_like(common), teacher, common, {}
    )

    assert float(exact_metrics["full_cosine"]) > 0.999
    assert float(exact_metrics["common_cosine"]) > 0.999
    assert float(incomplete_loss) > float(exact_loss) + 0.1
    assert float(incomplete_metrics["common_student_to_teacher_rms"]) < 1e-5


def test_oracle_objective_prefers_repeatable_low_frequency_artist_effect():
    generator = torch.Generator().manual_seed(9)
    artist = torch.eye(8).reshape(8, 8, 1, 1).expand(-1, -1, 16, 16)
    detail_noise = 2.0 * torch.randn(8, 8, 16, 16, generator=generator)
    detail_noise = detail_noise - F.avg_pool2d(detail_noise, 8, 8).repeat_interleave(
        8, -2
    ).repeat_interleave(8, -1)
    teacher = artist + detail_noise
    correct = artist - artist.mean(dim=0, keepdim=True)
    wrong = correct.roll(1, dims=0)

    correct_loss, correct_metrics = _artist_centered_oracle_objective(
        correct, teacher, {"low_frequency_factors": [8], "infonce": 0.5}
    )
    wrong_loss, wrong_metrics = _artist_centered_oracle_objective(
        wrong, teacher, {"low_frequency_factors": [8], "infonce": 0.5}
    )

    assert float(correct_metrics["low_frequency_cosine"]) > 0.99
    assert float(wrong_metrics["low_frequency_cosine"]) < 0
    assert float(correct_loss) < float(wrong_loss)


def test_fixed_oracle_reader_returns_checkpoint_codes():
    codes = torch.randn(7, 28, 16)
    reader = _FixedOracleCodeReader(codes)
    result = reader(torch.randn(7, 1, 84, 16), torch.ones(7, 1, dtype=torch.bool))

    assert result.tokens.data_ptr() == codes.data_ptr()


def test_fresh_oracle_adapter_keeps_kv_and_copies_only_strength_state():
    fresh = {
        "base_k.0.weight": torch.tensor([1.0]),
        "alpha": torch.tensor([1.0]),
        "strength_timestep_centers": torch.tensor([0.2]),
        "alpha_by_timestep": torch.tensor([[1.0]]),
        "native_lower_by_timestep": torch.tensor([[1.0]]),
        "native_upper_by_timestep": torch.tensor([[2.0]]),
        "native_fixed_output_by_timestep": torch.tensor([[1.5]]),
        "timestep_strength_enabled": torch.tensor(False),
        "fixed_output_strength_enabled": torch.tensor(False),
    }
    checkpoint = {
        key: value + 10 if value.dtype != torch.bool else ~value
        for key, value in fresh.items()
    }
    merged = _oracle_adapter_initial_state(
        fresh, checkpoint, "fresh_kv_checkpoint_strength"
    )

    assert torch.equal(merged["base_k.0.weight"], fresh["base_k.0.weight"])
    assert torch.equal(merged["alpha"], checkpoint["alpha"])
    assert torch.equal(
        merged["native_fixed_output_by_timestep"],
        checkpoint["native_fixed_output_by_timestep"],
    )


def test_oracle_visual_interpolation_reaches_both_endpoints():
    oracle = torch.randn(2, 3, 4)
    visual = torch.randn(2, 3, 4)

    assert torch.equal(_interpolate_oracle_visual(oracle, visual, 0.0), oracle)
    assert torch.equal(_interpolate_oracle_visual(oracle, visual, 1.0), visual)


def test_oracle_visual_projector_preserves_shape_and_initial_scale():
    projector = OracleVisualProjector(
        dim=32, slots=4, heads=4, ff_dim=64, bottleneck_dim=8
    )
    visual = torch.randn(3, 4, 32)
    projected = projector(visual)

    assert projected.shape == visual.shape
    assert float((projected - visual).square().mean().sqrt()) < 0.01


def test_oracle_code_alignment_prefers_oracle_codes():
    oracle = torch.randn(8, 4, 16)
    visual = oracle + 0.8 * torch.randn_like(oracle)
    exact_loss, exact_metrics = _oracle_code_alignment_objective(
        oracle, oracle, visual, {}
    )
    visual_loss, visual_metrics = _oracle_code_alignment_objective(
        visual, oracle, visual, {}
    )

    assert float(exact_loss) < float(visual_loss)
    assert float(exact_metrics["centered_cosine"]) > 0.999
    assert float(visual_metrics["projected_to_oracle_rms"]) > 0.1


class _ToyReferenceLoader:
    def load_styles(self, style_ids, *, references_per_style, seed):
        del seed
        values = torch.arange(
            len(style_ids) * references_per_style * 4 * 8,
            dtype=torch.float32,
        )
        return {"tokens": values.reshape(len(style_ids), references_per_style, 4, 8)}


class _ToyReader(torch.nn.Module):
    def forward(self, references, mask, *, reconstruct=False):
        del mask, reconstruct
        return type("Output", (), {"tokens": references.mean(dim=1)})()


def test_materialized_reader_bank_contains_single_pair_and_quad_views():
    bank, counts = _materialize_reader_code_bank(
        _ToyReader(),
        _ToyReferenceLoader(),
        ["a", "b"],
        reference_images=4,
        seed=1,
        device="cpu",
    )

    assert bank.shape == (2, 7, 4, 8)
    assert counts.tolist() == [1, 1, 1, 1, 2, 2, 4]


def test_materialized_reader_bank_chunks_large_style_sets():
    bank, counts = _materialize_reader_code_bank(
        _ToyReader(),
        _ToyReferenceLoader(),
        ["a", "b", "c", "d", "e"],
        reference_images=4,
        seed=1,
        device="cpu",
        style_chunk_size=2,
    )

    assert bank.shape == (5, 7, 4, 8)
    assert counts.tolist() == [1, 1, 1, 1, 2, 2, 4]


def test_projected_reader_composes_reader_and_projector():
    projector = OracleVisualProjector(
        dim=8, slots=4, heads=2, ff_dim=16, bottleneck_dim=4
    )
    wrapper = _ProjectedReader(_ToyReader(), projector)
    references = torch.randn(2, 3, 4, 8)
    output = wrapper(references, torch.ones(2, 3, dtype=torch.bool))

    expected = projector(references.mean(dim=1))
    assert torch.allclose(output.tokens, expected)
