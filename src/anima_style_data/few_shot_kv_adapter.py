"""Minimal inference boundary for the learned few-shot native-K/V adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .kv_activation_modulation import NativeKVFactorModulator
from .kv_activation_sampling import NativeKVFactorInjector
from .lora_oracle_bootstrap import _oracle_detail_config


def _architecture_from_state(state: dict[str, Any]) -> dict[str, int]:
    if "architecture" in state:
        return {key: int(value) for key, value in state["architecture"].items()}
    model = state["model"]
    return {
        "style_dim": int(model["style_norm.weight"].shape[0]),
        "blocks": int(model["block_embedding.weight"].shape[0]),
        "rank": int(model["factor_query"].shape[1]),
        "context_dim": int(model["down_head.weight"].shape[0]),
        "output_dim": int(model["up_head.weight"].shape[0]),
    }


class FewShotNativeKVStyleAdapter(nn.Module):
    """Convert frozen per-reference tokens into live Anima K/V deltas.

    The input is the existing frozen Dual-query Resampler token tensor.  Raw
    image decoding and C-RADIO/Qwen-VAE feature extraction remain in the
    production preprocessing pipeline; this module owns the learned online
    path from those tokens to Anima.
    """

    def __init__(
        self,
        *,
        reader: DetailPreservingTypedSlotReader,
        modulator: NativeKVFactorModulator,
        anima: nn.Module,
    ) -> None:
        super().__init__()
        self.reader = reader.requires_grad_(False).eval()
        self.modulator = modulator.requires_grad_(False).eval()
        self.injector = NativeKVFactorInjector(anima)
        self.active_strength = 0.0

    @property
    def device(self) -> torch.device:
        return next(self.modulator.parameters()).device

    @torch.no_grad()
    def encode_reference_tokens(
        self,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = reference_tokens.to(
            device=self.device, dtype=next(self.reader.parameters()).dtype
        )
        if reference_mask is None:
            reference_mask = torch.ones(
                tokens.shape[:2], device=self.device, dtype=torch.bool
            )
        else:
            reference_mask = reference_mask.to(
                device=self.device, dtype=torch.bool
            )
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            style_codes = self.reader(tokens, reference_mask).tokens
            down_values = []
            up_values = []
            for block in range(self.modulator.blocks):
                down, up = self.modulator(style_codes, block)
                down_values.append(down)
                up_values.append(up)
        return (
            style_codes,
            torch.stack(down_values, dim=1),
            torch.stack(up_values, dim=1),
        )

    @torch.no_grad()
    def set_references(
        self,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        style_codes, down, up = self.encode_reference_tokens(
            reference_tokens, reference_mask
        )
        self.injector.set_factors(down, up, strength=float(strength))
        self.active_strength = float(strength)
        return style_codes

    def set_strength(self, strength: float) -> None:
        if self.injector.down is None or self.injector.up is None:
            raise RuntimeError("References must be encoded before setting strength")
        self.injector.strength = float(strength)
        self.injector.enabled = True
        self.active_strength = float(strength)

    def disable(self) -> None:
        self.injector.disable()
        self.active_strength = 0.0

    def close(self) -> None:
        self.injector.close()
        self.active_strength = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        config: dict[str, Any],
        destination: Path,
        anima: nn.Module,
        checkpoint_path: Path,
        *,
        device: str = "cuda",
    ) -> "FewShotNativeKVStyleAdapter":
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        cfg = dict(state["config"])
        architecture = _architecture_from_state(state)
        oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
        detail_cfg = _oracle_detail_config(config, oracle_cfg)
        reader_state = torch.load(
            destination / str(cfg["reader_checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"]))
        reader.load_state_dict(reader_state["reader"], strict=True)
        modulator = NativeKVFactorModulator(
            **architecture,
            **dict(cfg["model"]),
        )
        modulator.load_state_dict(state["model"], strict=True)
        reader.to(device=device, dtype=torch.bfloat16)
        modulator.to(device=device, dtype=torch.bfloat16)
        return cls(reader=reader, modulator=modulator, anima=anima)

