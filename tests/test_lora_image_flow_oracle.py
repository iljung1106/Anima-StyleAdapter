import random

import torch

from anima_style_data.lora_image_flow_oracle import (
    _centered_image_flow_objective,
    _sample_weighted_timestep,
)


def test_centered_image_flow_prefers_correct_artist_effects():
    base = torch.zeros(4, 4, 8, 8)
    effects = torch.zeros_like(base)
    for index in range(4):
        effects[index, index, :, :] = 1.0
    target = base + effects
    weights = {
        "flow_mse": 0.1,
        "common_huber": 0.1,
        "centered_mse": 1.0,
        "centered_direction": 1.0,
        "centered_magnitude": 0.1,
        "infonce": 0.5,
        "descriptor_factors": [4],
    }

    exact_loss, exact = _centered_image_flow_objective(
        target.clone(), base, target, weights
    )
    wrong_loss, wrong = _centered_image_flow_objective(
        target.roll(1, dims=0), base, target, weights
    )
    collapsed_loss, collapsed = _centered_image_flow_objective(
        effects.mean(dim=0, keepdim=True).expand_as(effects), base, target, weights
    )

    assert float(exact_loss) < float(wrong_loss)
    assert float(exact_loss) < float(collapsed_loss)
    assert float(exact["centered_cosine"]) > 0.99
    assert float(exact["infonce_accuracy"]) == 1.0
    assert float(wrong["infonce_cosine_gap"]) < 0.0
    assert float(collapsed["common_output_ratio"]) > 0.99


def test_weighted_timestep_sampler_respects_bins_and_weights():
    rng = random.Random(1234)
    draws = [
        _sample_weighted_timestep(rng, [0.0, 0.2, 0.5, 1.0], [4.0, 1.0, 1.0])
        for _ in range(600)
    ]

    assert all(
        [0.0, 0.2, 0.5][index] <= value <= [0.2, 0.5, 1.0][index]
        for value, index in draws
    )
    assert sum(index == 0 for _, index in draws) > 350
