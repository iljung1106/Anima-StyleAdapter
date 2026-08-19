import torch

from anima_style_data.artist_effect_losses import (
    centered_functional_artist_loss,
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
