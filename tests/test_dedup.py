import numpy as np
from PIL import Image

from anima_style_data.dedup import hamming, perceptual_hashes


def test_identical_images_have_identical_hashes():
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[10:40, 15:50] = (220, 30, 80)
    first = Image.fromarray(pixels)
    second = first.copy()
    p1, d1 = perceptual_hashes(first)
    p2, d2 = perceptual_hashes(second)
    assert hamming(p1, p2) == 0
    assert hamming(d1, d2) == 0


def test_small_encoding_change_stays_close():
    pixels = np.full((96, 64, 3), 245, dtype=np.uint8)
    pixels[15:80, 10:55] = (20, 80, 180)
    original = Image.fromarray(pixels)
    altered = Image.fromarray(np.clip(pixels.astype(np.int16) + 2, 0, 255).astype(np.uint8))
    p1, d1 = perceptual_hashes(original)
    p2, d2 = perceptual_hashes(altered)
    assert hamming(p1, p2) <= 2
    assert hamming(d1, d2) <= 2
