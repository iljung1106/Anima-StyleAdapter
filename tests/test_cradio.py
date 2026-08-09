import numpy as np
import torch
from PIL import Image

from anima_style_data.cradio import (
    build_style_feature_combiner,
    compute_cradio_size,
    content_subtracted_summary,
    preprocess_cradio_image,
)


def test_cradio_size_only_downscales_and_aligns_to_patch():
    assert compute_cradio_size(
        101, 203, max_side=512, max_pixels=512 * 512, step=16
    ) == (101, 203, 96, 192)
    resized_h, resized_w, target_h, target_w = compute_cradio_size(
        1500, 1000, max_side=1024, max_pixels=1024 * 1024, step=16
    )
    assert (resized_h, resized_w) == (1024, 682)
    assert target_h % 16 == target_w % 16 == 0
    assert target_h <= resized_h and target_w <= resized_w


def test_cradio_preprocess_preserves_rgb_range_and_records_crop():
    image = Image.new("RGB", (203, 101), (255, 0, 0))
    array, info = preprocess_cradio_image(
        image,
        {
            "max_side": 512,
            "max_pixels": 512 * 512,
            "patch_size": 16,
            "min_side": 16,
        },
    )
    assert array.shape == (3, 96, 192)
    assert array.dtype == np.float32
    np.testing.assert_allclose(array[:, 0, 0], [1.0, 0.0, 0.0])
    assert info.crop_left == 5 and info.crop_top == 2


def test_content_subtraction_and_trainable_combiner_shapes():
    visual = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    text = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    _, _, residual = content_subtracted_summary(visual, text)
    torch.testing.assert_close(residual.norm(dim=-1), torch.ones(1))

    combiner = build_style_feature_combiner(6, 4, 8)
    combined = combiner(torch.randn(2, 12, 6), torch.randn(2, 4))
    assert combined.shape == (2, 13, 8)
