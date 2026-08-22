import torch

from anima_style_data.artist_effect_losses import (
    centered_functional_artist_loss,
    common_output_and_artist_magnitude_loss,
    episodic_artist_prototype_loss,
)


def test_centered_functional_loss_rewards_repeatable_artist_effects():
    generator = torch.Generator().manual_seed(17)
    first = torch.randn(4, 3, 16, 16, generator=generator)
    common = torch.randn(1, 3, 16, 16, generator=generator) * 5.0
    second = first + common
    style_ids = ["a", "b", "c", "d"]

    good, metrics = centered_functional_artist_loss(
        first, second, style_ids, pool_scales=(2, 4)
    )
    shifted, _ = centered_functional_artist_loss(
        first, second + 11.0, style_ids, pool_scales=(2, 4)
    )
    bad, _ = centered_functional_artist_loss(
        first, second.roll(1, dims=0), style_ids, pool_scales=(2, 4)
    )

    torch.testing.assert_close(good, shifted, atol=1e-5, rtol=1e-5)
    assert good < bad
    assert metrics["functional_artist_retrieval_top1"] == 1
    assert metrics["functional_artist_repeatable_ratio"] > 0.99
    assert metrics["functional_artist_icc"] > 0.99


def test_repeatability_floor_stops_compressing_already_related_views():
    generator = torch.Generator().manual_seed(19)
    first = torch.randn(4, 3, 16, 16, generator=generator)
    related = first + 0.25 * torch.randn(4, 3, 16, 16, generator=generator)
    style_ids = ["a", "b", "c", "d"]

    _, related_metrics = centered_functional_artist_loss(
        first,
        related,
        style_ids,
        repeatability_weight=0.1,
        repeatability_floor=0.3,
    )
    _, unrelated_metrics = centered_functional_artist_loss(
        first,
        related.roll(1, dims=0),
        style_ids,
        repeatability_weight=0.1,
        repeatability_floor=0.3,
    )

    torch.testing.assert_close(
        related_metrics["functional_artist_repeatability_loss"],
        torch.tensor(0.0),
    )
    assert unrelated_metrics["functional_artist_repeatability_loss"] > 0


def test_all_wrong_margin_preserves_reference_variation_after_separation():
    generator = torch.Generator().manual_seed(29)
    first = torch.randn(4, 3, 16, 16, generator=generator)
    # Keep meaningful per-reference variation while preserving the matching
    # artist direction well above the deliberately modest positive floor.
    second = first + 0.35 * torch.randn(
        4, 3, 16, 16, generator=generator
    )
    style_ids = ["a", "b", "c", "d"]

    good, good_metrics = centered_functional_artist_loss(
        first,
        second,
        style_ids,
        contrastive_mode="all_wrong_margin",
        positive_floor=0.25,
        wrong_margin=0.10,
        repeatability_weight=0.02,
    )
    wrong, wrong_metrics = centered_functional_artist_loss(
        first,
        second.roll(1, dims=0),
        style_ids,
        contrastive_mode="all_wrong_margin",
        positive_floor=0.25,
        wrong_margin=0.10,
        repeatability_weight=0.02,
    )

    assert good < wrong
    assert good_metrics["functional_artist_uses_symmetric_nce"] == 0
    assert good_metrics["functional_artist_repeatability_loss"] == 0
    assert good_metrics["functional_artist_all_wrong_violation_fraction"] < (
        wrong_metrics["functional_artist_all_wrong_violation_fraction"]
    )


def test_episodic_prototype_uses_disjoint_artist_view_as_the_class():
    generator = torch.Generator().manual_seed(23)
    first = torch.randn(4, 28, 32, generator=generator)
    second = first + 0.01 * torch.randn(4, 28, 32, generator=generator)
    style_ids = ["a", "b", "c", "d"]

    good, metrics = episodic_artist_prototype_loss(
        first, second, style_ids, slot_type_counts=(16, 8, 4)
    )
    bad, _ = episodic_artist_prototype_loss(
        first, second.roll(1, dims=0), style_ids,
        slot_type_counts=(16, 8, 4),
    )

    assert good < bad
    assert metrics["artist_prototype_retrieval_top1"] == 1
    assert metrics["artist_prototype_cosine_gap"] > 0


def test_common_output_penalty_rejects_shared_residual_without_shrinking_scale():
    generator = torch.Generator().manual_seed(31)
    teacher = torch.randn(4, 3, 16, 16, generator=generator)
    teacher = teacher - teacher.mean(dim=0, keepdim=True)
    common = torch.randn(1, 3, 16, 16, generator=generator) * 4.0
    student = (teacher + common).requires_grad_()

    common_loss, _, metrics = common_output_and_artist_magnitude_loss(
        teacher,
        student,
        common_threshold=0.65,
        magnitude_lower=0.5,
        magnitude_upper=1.25,
    )
    clean_loss, _, clean_metrics = common_output_and_artist_magnitude_loss(
        teacher,
        teacher,
        common_threshold=0.65,
        magnitude_lower=0.5,
        magnitude_upper=1.25,
    )
    common_loss.backward()

    assert common_loss > clean_loss
    assert metrics["functional_artist_student_common_output_ratio"] > 0.9
    assert clean_metrics["functional_artist_student_common_output_ratio"] < 1e-5
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_artist_magnitude_band_uses_teacher_aligned_projection():
    generator = torch.Generator().manual_seed(37)
    teacher = torch.randn(4, 3, 16, 16, generator=generator)
    teacher = teacher - teacher.mean(dim=0, keepdim=True)

    _, good, good_metrics = common_output_and_artist_magnitude_loss(
        teacher,
        0.8 * teacher,
        common_threshold=1.0,
        magnitude_lower=0.6,
        magnitude_upper=1.2,
    )
    _, small, small_metrics = common_output_and_artist_magnitude_loss(
        teacher,
        0.1 * teacher,
        common_threshold=1.0,
        magnitude_lower=0.6,
        magnitude_upper=1.2,
    )
    _, large, _ = common_output_and_artist_magnitude_loss(
        teacher,
        1.6 * teacher,
        common_threshold=1.0,
        magnitude_lower=0.6,
        magnitude_upper=1.2,
    )

    assert good < small
    assert good < large
    assert good_metrics["functional_artist_magnitude_projection"] > 0.79
    assert small_metrics["functional_artist_magnitude_projection"] < 0.11
