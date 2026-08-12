import torch
from torch.nn.utils.rnn import pad_sequence

from anima_style_data.synthetic_bootstrap import (
    _attention_output,
    _batched_attention_output,
    _bootstrap_eligible,
    assign_bootstrap_splits,
    classify_artist_effects,
)


def test_variable_spatial_tokens_use_batch_local_padding():
    values = [torch.ones(7, 3), torch.ones(5, 3)]
    padded = pad_sequence(values, batch_first=True)
    counts = torch.tensor([value.shape[0] for value in values])
    mask = torch.arange(padded.shape[1])[None] < counts[:, None]

    assert padded.shape == (2, 7, 3)
    assert mask.sum(1).tolist() == [7, 5]


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


def test_batched_attention_matches_individual_attention():
    torch.manual_seed(23)
    batch, heads, queries, context, dim = 3, 2, 5, 7, 4
    q = torch.randn(batch, heads, queries, dim)
    k = torch.randn(batch, heads, context, dim)
    v = torch.randn(batch, heads, context, dim)
    output_weight = torch.randn(heads * dim, heads * dim)
    batched = _batched_attention_output(q, k, v, output_weight)
    individual = torch.stack([
        _attention_output(q[index], k[index], v[index], output_weight)
        for index in range(batch)
    ])
    torch.testing.assert_close(batched, individual)
