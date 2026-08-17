from io import BytesIO

import numpy as np
from PIL import Image

from anima_style_data.anima_cache import (
    _decode_anima_image,
    build_anima_caption_variants,
    compute_anima_geometry,
    effective_vae_batch_size,
)


def test_caption_variants_are_deterministic_and_preserve_identity_tags():
    row = {
        "id": 17,
        "anima_caption": "safe, 1girl, character, blue hair, outdoors, smile",
        "rating_anima": "safe",
        "count_tags": ["1girl"],
        "character_tags": ["character"],
        "general_tags": ["blue hair", "outdoors", "smile"],
    }
    cfg = {"variants": 2, "general_tag_dropout": 0.5, "variant_seed": 9}
    first = build_anima_caption_variants(row, cfg)
    second = build_anima_caption_variants(row, cfg)
    assert first == second
    assert first[0][2] == row["anima_caption"]
    assert "safe" in first[1][2]
    assert "1girl" in first[1][2]
    assert "character" in first[1][2]


def test_multimode_caption_variants_cover_quality_dropout_and_short():
    row = {
        "id": 17,
        "anima_caption": "safe, 1girl, alice, sky, smile, dress, cloud, flower, cup, blue eyes",
        "rating_anima": "safe",
        "count_tags": ["1girl"],
        "character_tags": ["alice"],
        "general_tags": ["sky", "smile", "dress", "cloud", "flower", "cup", "blue eyes"],
    }
    cfg = {
        "variant_modes": [
            "full", "full_quality", "tag_dropout", "tag_dropout_quality",
            "short", "short_quality",
        ],
        "general_tag_dropout_min": 0.2,
        "general_tag_dropout_max": 0.6,
        "short_general_tags": 2,
        "variant_seed": 9,
    }
    variants = build_anima_caption_variants(row, cfg)
    by_name = {name: caption for _, name, caption in variants}
    assert set(by_name) == set(cfg["variant_modes"])
    assert by_name["full_quality"].startswith("masterpiece, best quality, score_7")
    assert "safe" in by_name["tag_dropout"]
    assert "1girl" in by_name["tag_dropout"]
    assert "alice" in by_name["tag_dropout"]
    assert by_name["short"] == "safe, 1girl, alice, sky, smile"


def test_anima_geometry_limits_area_and_aligns_bucket_without_upscale():
    cfg = {
        "max_side": 1536,
        "max_pixels": 1024 * 1024,
        "min_side": 256,
        "bucket_step": 64,
        "allow_upscale": False,
        "upscale_below_min": True,
    }
    geometry = compute_anima_geometry(2000, 1200, cfg)
    assert geometry.target_height % 64 == 0
    assert geometry.target_width % 64 == 0
    assert geometry.target_height * geometry.target_width <= 1024 * 1024
    assert geometry.resized_height <= 1536
    assert geometry.resized_width <= 1536


def test_anima_decode_produces_normalized_center_crop():
    image = Image.new("RGB", (1200, 800), (255, 64, 0))
    encoded = BytesIO()
    image.save(encoded, format="WEBP", lossless=True)
    array, geometry, _, _ = _decode_anima_image(
        encoded.getvalue(),
        {
            "max_side": 1024,
            "max_pixels": 1024 * 1024,
            "min_side": 256,
            "bucket_step": 64,
            "allow_upscale": False,
            "upscale_below_min": True,
        },
    )
    assert array.shape == (3, geometry.target_height, geometry.target_width)
    assert array.dtype == np.float32
    np.testing.assert_allclose(array[:, 0, 0], [1.0, 64 / 127.5 - 1.0, -1.0])


def test_only_tiny_images_are_upscaled_to_minimum_bucket():
    cfg = {
        "max_side": 1536,
        "max_pixels": 1024 * 1024,
        "min_side": 256,
        "bucket_step": 64,
        "allow_upscale": False,
        "upscale_below_min": True,
    }
    regular = compute_anima_geometry(768, 768, cfg)
    tiny = compute_anima_geometry(200, 300, cfg)
    assert (regular.target_height, regular.target_width) == (768, 768)
    assert min(tiny.target_height, tiny.target_width) >= 256


def test_vae_batch_respects_pixel_budget():
    assert effective_vae_batch_size(512, 512, 8, 4 * 1024 * 1024) == 8
    assert effective_vae_batch_size(768, 704, 8, 4 * 1024 * 1024) == 7
    assert effective_vae_batch_size(1024, 1024, 8, 4 * 1024 * 1024) == 4
    assert effective_vae_batch_size(1024, 1024, 8, None) == 8
