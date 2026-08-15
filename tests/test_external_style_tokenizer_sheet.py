from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.external_style_tokenizer_sheet import _denoise_batch


class _ContextValueModel:
    def __call__(
        self, x, timestep, *, context, padding_mask, target_input_ids
    ):
        value = context[:, 0, 0].reshape(-1, 1, 1, 1, 1)
        return torch.ones_like(x) * value


def test_style_multiplier_scales_only_styled_minus_text_direction():
    noise = torch.zeros(1, 1, 1, 1, 1, dtype=torch.bfloat16)
    negative = torch.tensor([[[1.0]]])
    text = torch.tensor([[[2.0]]])
    styled = torch.tensor([[[5.0]]])

    outputs = [
        _denoise_batch(
            _ContextValueModel(),
            noise,
            text,
            negative,
            steps=1,
            flow_shift=1.0,
            cfg_scale=4.0,
            style_context=styled,
            style_multiplier=multiplier,
        ).item()
        for multiplier in (1.0, 1.5, 2.0)
    ]

    assert outputs == pytest.approx([-17.0, -23.0, -29.0])
