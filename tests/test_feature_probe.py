import torch

from anima_style_data.feature_probe import prototype_metrics


def test_prototype_metrics_improve_with_consistent_references():
    # Two identities, three reference candidates and one held-out query each.
    values = torch.tensor(
        [
            [1.0, 0.3], [1.0, -0.2], [1.0, 0.0], [1.0, 0.0],
            [0.3, 1.0], [-0.2, 1.0], [0.0, 1.0], [0.0, 1.0],
        ]
    )
    styles = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ranks = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    metrics = prototype_metrics(values, styles, ranks, [1, 2], max_references=3)

    assert metrics[0]["queries"] == 2
    assert metrics[0]["top1"] == metrics[1]["top1"] == 1.0
    assert metrics[1]["margin"] > 0
