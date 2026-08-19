import torch

from anima_style_data.block_similarity import centered_linear_cka, k_medoids
from anima_style_data.data_mixture import ConstantRatioBatchMixer, auxiliary_step


def test_centered_linear_cka_recovers_rotated_representation():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(64, 8, generator=generator)
    rotation = torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
    unrelated = torch.randn(64, 8, generator=generator)
    result = centered_linear_cka([source, source @ rotation, unrelated], "cpu")
    assert torch.allclose(result.diagonal(), torch.ones(3), atol=1e-5)
    assert result[0, 1] > 0.999
    assert result[0, 2] < 0.5


def test_k_medoids_finds_four_separated_pairs():
    similarity = torch.full((8, 8), 0.05)
    similarity.fill_diagonal_(1.0)
    for start in range(0, 8, 2):
        similarity[start, start + 1] = similarity[start + 1, start] = 0.95
    medoids, clusters = k_medoids(similarity, 4)
    assert len(medoids) == 4
    assert sorted(clusters) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_auxiliary_schedule_is_exactly_fifteen_percent_per_twenty_steps():
    decisions = [auxiliary_step(step, 0.15) for step in range(40)]
    assert sum(decisions[:20]) == 3
    assert sum(decisions[20:]) == 3


def test_constant_ratio_mixer_routes_complete_batches():
    class Loader:
        batch_size = 4

        def __init__(self, name):
            self.name = name

        def load_step(self, step):
            return {"source": self.name, "step": step}

    mixed = ConstantRatioBatchMixer(
        Loader("primary"), Loader("auxiliary"), auxiliary_fraction=0.15
    )
    batches = list(mixed.prefetch(0, 20, workers=2, depth=4))
    assert sum(batch["data_domain"] == "megastyle" for batch in batches) == 3
    assert all(
        (batch["source"] == "auxiliary")
        == (batch["data_domain"] == "megastyle")
        for batch in batches
    )
