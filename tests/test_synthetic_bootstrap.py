from anima_style_data.synthetic_bootstrap import (
    _bootstrap_eligible,
    assign_bootstrap_splits,
    classify_artist_effects,
)


def test_bootstrap_artist_splits_are_disjoint_and_sized():
    result = assign_bootstrap_splits([f"artist-{index}" for index in range(100)], seed=7)
    assert list(result.values()).count("validation") == 25
    assert list(result.values()).count("meta_test") == 25
    assert list(result.values()).count("train") == 50


def test_artist_effect_filter_removes_only_severe_tail():
    rows = [
        {"artist": f"a{index}", "effect_rms": 1.0, "direction_consistency": 0.8,
         "seed_consistency": 0.8, "content_consistency": 0.8}
        for index in range(100)
    ]
    rows[0].update(effect_rms=0.001, direction_consistency=-0.9, seed_consistency=-0.9)
    labels = classify_artist_effects(rows)
    assert labels["a0"].startswith("excluded")
    assert sum(value.startswith("excluded") for value in labels.values()) <= 2


def test_bootstrap_eligibility_matches_validated_manifest_contract():
    assert _bootstrap_eligible({"kind": "artist", "artist_split": "train"})
    assert _bootstrap_eligible({"kind": "artist", "artist_split": "validation"})
    assert not _bootstrap_eligible({"kind": "artist", "artist_split": "excluded"})
    assert not _bootstrap_eligible({"kind": "content_control", "artist_split": "control"})
