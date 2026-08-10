import torch

from anima_style_data.tap_resampler import (
    _load_feature_batch,
    _prototype_loss,
    _training_rows_for_step,
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


def test_concat_resampler_returns_direct_style_tokens():
    model = build_tap_resampler_model(
        taps=[18, 24],
        reconstruction_taps=[18, 24],
        spatial_dim=12,
        global_kind="native_24",
        global_dim=12,
        model_dim=24,
        latent_tokens=4,
        heads=4,
        resampler_layers=1,
        decoder_layers=1,
        style_dim=24,
        spatial_fusion="concat_mlp",
        direct_style_tokens=True,
    )
    features = {18: torch.randn(4, 6, 12), 24: torch.randn(4, 6, 12)}
    mask = torch.ones(4, 6, dtype=torch.bool)
    decoded, decoded_mask, style_tokens = model(
        features,
        mask,
        [(32, 48)] * 4,
        torch.randn(4, 12),
    )

    assert decoded[18].shape == decoded[24].shape == (4, 6, 12)
    assert torch.equal(decoded_mask, mask)
    assert style_tokens.shape == (4, 4, 24)
    assert torch.isfinite(_prototype_loss(style_tokens, 2, 2, 0.07))


def test_training_episode_is_step_addressable():
    artists = ["a", "b", "c"]
    by_style = {
        artist: [{"id": f"{artist}{index}"} for index in range(4)]
        for artist in artists
    }
    kwargs = {
        "step": 12,
        "seed": 7,
        "artists": artists,
        "train_by_style": by_style,
        "artists_per_batch": 2,
        "images_per_artist": 3,
    }

    first = _training_rows_for_step(**kwargs)
    second = _training_rows_for_step(**kwargs)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 6


def test_feature_loader_keeps_cached_fp16(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "1.layer_18_spatial": torch.randn(3, 12).half(),
            "1.layer_24_spatial": torch.randn(3, 12).half(),
            "1.layer_24_siglip_cls": torch.randn(12).half(),
            "2.layer_18_spatial": torch.randn(2, 12).half(),
            "2.layer_24_spatial": torch.randn(2, 12).half(),
            "2.layer_24_siglip_cls": torch.randn(12).half(),
        },
        tmp_path / "part.safetensors",
    )
    rows = [
        {
            "id": image_id,
            "feature_shard": "part.safetensors",
            "spatial_tokens": tokens,
            "spatial_dim": 12,
            "target_height": 16,
            "target_width": tokens * 16,
        }
        for image_id, tokens in ((1, 3), (2, 2))
    ]

    features, targets, mask, _, global_feature = _load_feature_batch(
        rows, tmp_path, [18, 24], [18, 24], "native_24"
    )

    assert features[18].dtype == features[24].dtype == torch.float16
    assert targets[18].data_ptr() == features[18].data_ptr()
    assert global_feature.dtype == torch.float16
    assert mask.sum().item() == 5


def test_token_bucket_episode_prefers_similar_sizes():
    artists = ["a", "b"]
    by_style = {
        artist: [
            {"id": f"{artist}{index}", "spatial_tokens": 100 if index < 4 else 1000}
            for index in range(6)
        ]
        for artist in artists
    }
    rows = _training_rows_for_step(
        step=3,
        seed=7,
        artists=artists,
        train_by_style=by_style,
        artists_per_batch=2,
        images_per_artist=4,
        token_bucket_centers=[100],
    )

    assert {row["spatial_tokens"] for row in rows} == {100}
