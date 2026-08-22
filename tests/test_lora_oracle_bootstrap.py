import torch

from anima_style_data.lora_oracle_bootstrap import (
    _FixedOracleCodeReader,
    _artist_centered_oracle_objective,
    _cross_view_artist_objective,
    _interpolate_oracle_visual,
    _oracle_code_alignment_objective,
    _materialize_reader_code_bank,
    _piecewise_linear_value,
    OracleVisualProjector,
    _ProjectedReader,
)


def test_cross_view_artist_objective_recognizes_matching_artists():
    left = torch.eye(8).reshape(8, 1, 8)
    loss, metrics = _cross_view_artist_objective(
        left, left.clone(), temperature=0.10
    )

    assert torch.isfinite(loss)
    assert float(metrics["accuracy"]) == 1.0
    assert float(metrics["cosine_gap"]) > 0.5


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


def test_fixed_oracle_reader_returns_checkpoint_codes():
    codes = torch.randn(7, 28, 16)
    reader = _FixedOracleCodeReader(codes)
    result = reader(torch.randn(7, 1, 84, 16), torch.ones(7, 1, dtype=torch.bool))

    assert result.tokens.data_ptr() == codes.data_ptr()


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
