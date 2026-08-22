from __future__ import annotations

import pytest
import torch

from anima_style_data.artist_lora_teachers import (
    _prompt_probabilities,
    _reset_lora_network,
    select_artist_lora_plans,
)


def _rows(styles: int = 3, images: int = 5):
    latent_rows = []
    text_rows = []
    variants = (
        "full",
        "full_quality",
        "tag_dropout",
        "tag_dropout_quality",
        "short",
        "short_quality",
    )
    image_id = 0
    for style in range(styles):
        for _ in range(images):
            latent_rows.append(
                {
                    "id": image_id,
                    "artist": f"artist_{style}",
                    "style_id": f"style_{style}",
                    "split": "train",
                    "latent_height": 64,
                    "latent_width": 64,
                }
            )
            for index, name in enumerate(variants):
                text_rows.append(
                    {
                        "id": image_id,
                        "split": "train",
                        "variant": index,
                        "variant_name": name,
                        "caption": "safe, 1girl, blue sky",
                    }
                )
            image_id += 1
    return latent_rows, text_rows, variants


def test_artist_plan_is_disjoint_and_deterministic():
    latent_rows, text_rows, variants = _rows()
    cfg = {
        "seed": 7,
        "artist_count": 2,
        "images_per_artist": 5,
        "train_images_per_artist": 3,
        "validation_images_per_artist": 2,
        "split": "train",
        "maximum_bucket_count": 1,
        "variant_names": variants,
    }
    first, summary = select_artist_lora_plans(latent_rows, text_rows, cfg)
    second, _ = select_artist_lora_plans(latent_rows, text_rows, cfg)
    assert first == second
    assert summary["artists"] == 2
    for plan in first:
        assert len(plan.train_ids) == 3
        assert len(plan.validation_ids) == 2
        assert set(plan.train_ids).isdisjoint(plan.validation_ids)


def test_prompt_probabilities_preserve_requested_mass():
    _, _, variants = _rows(styles=1, images=1)
    probabilities = _prompt_probabilities(
        {
            "mode_weights": {
                "full": 0.3,
                "tag_dropout": 0.4,
                "short": 0.2,
                "empty": 0.1,
            },
            "quality_probability": 0.5,
        },
        variants,
    )
    assert abs(sum(probabilities) - 1.0) < 1e-7
    assert probabilities[-1] == pytest.approx(0.1)
    assert probabilities[0] + probabilities[1] == pytest.approx(0.3)


def test_lora_reset_changes_down_and_zeros_up():
    class TinyLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_down = torch.nn.Linear(4, 2, bias=False)
            self.lora_up = torch.nn.Linear(2, 4, bias=False)

    class TinyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.item = TinyLoRA()
            self.unet_loras = [self.item]

    network = TinyNetwork()
    before = network.item.lora_down.weight.detach().clone()
    network.item.lora_up.weight.data.fill_(1)
    _reset_lora_network(network, 11)
    assert not torch.equal(before, network.item.lora_down.weight)
    assert torch.count_nonzero(network.item.lora_up.weight) == 0
