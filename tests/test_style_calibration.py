from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.style_calibration import filter_artist_effects


def test_artist_effect_filter_removes_weak_and_redundant_responses():
    artists = ["strong", "same_but_weaker", "different", "weak"]
    effects = {
        "strong": 0.08,
        "same_but_weaker": 0.06,
        "different": 0.07,
        "weak": 0.001,
    }
    signatures = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    retained, summary = filter_artist_effects(
        effects,
        signatures,
        artists,
        minimum_effect_ratio=0.01,
        minimum_effect_quantile=0.0,
        maximum_similarity=0.9,
    )
    assert "weak" not in [artists[index] for index in retained]
    assert "strong" in [artists[index] for index in retained]
    assert summary["weak_removed"] == 1
