import torch

from anima_style_data.tap_resampler import (
    _prototype_loss,
    build_tap_resampler_model,
    select_tap_experiment_rows,
)


def test_selection_is_artist_disjoint_and_fixed_size():
    rows = [
        {"id": artist * 100 + image, "artist": f"a{artist}", "style_id": f"a{artist}", "split": "train"}
        for artist in range(6)
        for image in range(4)
    ]
    config = {
        "source_split": "train",
        "artist_count": 6,
        "images_per_artist": 3,
        "val_artists": 1,
        "test_artists": 1,
        "seed": 7,
    }

    selected = select_tap_experiment_rows(rows, config)

    assert len(selected) == 18
    split_artists = {
        split: {row["style_id"] for row in selected if row["experiment_split"] == split}
        for split in ("meta_train", "meta_val", "meta_test")
    }
    assert len(split_artists["meta_train"]) == 4
    assert len(split_artists["meta_val"]) == len(split_artists["meta_test"]) == 1
    assert not (split_artists["meta_train"] & split_artists["meta_val"])
    assert not (split_artists["meta_train"] & split_artists["meta_test"])


def test_resampler_contract_and_prototype_loss():
    model = build_tap_resampler_model(
        taps=[2, 4],
        reconstruction_taps=[2, 4],
        spatial_dim=12,
        global_kind="native_4",
        global_dim=12,
        model_dim=16,
        latent_tokens=4,
        heads=4,
        resampler_layers=1,
        decoder_layers=1,
        style_dim=8,
    )
    features = {2: torch.randn(4, 6, 12), 4: torch.randn(4, 6, 12)}
    mask = torch.tensor([[1] * 6, [1] * 6, [1] * 4 + [0] * 2, [1] * 4 + [0] * 2]).bool()
    decoded, decoded_mask, embedding = model(
        features,
        mask,
        [(32, 48), (32, 48), (32, 32), (32, 32)],
        torch.randn(4, 12),
    )

    assert decoded[2].shape == decoded[4].shape == (4, 6, 12)
    assert torch.equal(decoded_mask, mask)
    assert embedding.shape == (4, 8)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(_prototype_loss(embedding, 2, 2, 0.07))
