import numpy as np
import torch
from io import BytesIO
from PIL import Image

from anima_style_data.cradio import (
    _decode_preprocess_bytes,
    _selected_style_tensors,
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


def test_webp_byte_pipeline_preserves_existing_preprocess_pixels():
    source = Image.new("RGBA", (203, 101), (10, 120, 240, 180))
    encoded = BytesIO()
    source.save(encoded, format="WEBP", lossless=True)
    cfg = {
        "max_side": 512,
        "max_pixels": 512 * 512,
        "patch_size": 16,
        "min_side": 16,
    }
    with Image.open(BytesIO(encoded.getvalue())) as image:
        expected, expected_info = preprocess_cradio_image(image, cfg)
    actual, actual_info, decode_s, resize_s = _decode_preprocess_bytes(
        encoded.getvalue(), cfg
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual_info == expected_info
    assert decode_s >= 0 and resize_s >= 0


def test_content_subtraction_and_trainable_combiner_shapes():
    visual = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    text = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    _, _, residual = content_subtracted_summary(visual, text)
    torch.testing.assert_close(residual.norm(dim=-1), torch.ones(1))

    combiner = build_style_feature_combiner(6, 4, 8)
    combined = combiner(torch.randn(2, 12, 6), torch.randn(2, 4))
    assert combined.shape == (2, 13, 8)


def test_batched_selected_style_tensors_match_per_image_reference_bitwise():
    generator = torch.Generator().manual_seed(20260810)
    layer_8 = torch.randn(4, 17, 12, generator=generator)
    layer_20 = torch.randn(4, 17, 12, generator=generator)
    layer_24 = torch.randn(4, 17, 12, generator=generator)
    selected = _selected_style_tensors(
        [layer_8, layer_20, layer_24],
        [8, 20, 24],
        {20, 24},
        {8},
        set(),
        torch.float16,
    )

    assert set(selected[0]) == {
        "layer_08_mean",
        "layer_08_std",
        "layer_20_spatial",
        "layer_24_spatial",
    }
    for item_index, tensors in enumerate(selected):
        expected = {
            "layer_08_mean": layer_8[item_index].float().mean(dim=0).half(),
            "layer_08_std": layer_8[item_index]
            .float()
            .std(dim=0, correction=0)
            .half(),
            "layer_20_spatial": layer_20[item_index].half(),
            "layer_24_spatial": layer_24[item_index].half(),
        }
        for name, value in expected.items():
            assert torch.equal(tensors[name], value), name


def test_selected_style_tensors_extract_siglip_teacher_cls():
    from types import SimpleNamespace

    spatial = torch.randn(2, 7, 12)
    summary = torch.randn(2, 24)
    selected = _selected_style_tensors(
        [SimpleNamespace(features=spatial, summary=summary)],
        [24],
        {24},
        set(),
        {24},
        torch.float16,
    )

    assert torch.equal(selected[0]["layer_24_siglip_cls"], summary[0, :12].half())
