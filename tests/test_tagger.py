import numpy as np
from PIL import Image

from anima_style_data.tagger import prepare_tagger_image


def test_tagger_preprocessing_matches_space_channel_and_padding_contract():
    image = Image.new("RGB", (2, 1), (255, 0, 0))
    output = prepare_tagger_image(image, input_size=2)
    assert output.shape == (3, 2, 2)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output[:, 0, 0], [-1.0, -1.0, 1.0])
    np.testing.assert_allclose(output[:, 1, 0], [1.0, 1.0, 1.0])
