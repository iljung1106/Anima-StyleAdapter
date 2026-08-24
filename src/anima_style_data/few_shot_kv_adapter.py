"""Minimal inference boundary for the learned few-shot native-K/V adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_external_samples import encode_dual_query_reference_images
from .kv_activation_modulation import (
    NativeKVFactorModulator,
    load_kv_lora_factor_bank,
)
from .kv_activation_sampling import NativeKVFactorInjector
from .kv_generalizing_modulator import (
    _visual_knn_coefficients,
    concatenate_weighted_lora_factors,
)
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


@torch.no_grad()
def prepare_reference_tokens(
    config: dict[str, Any],
    destination: Path,
    paths: list[Path],
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Tokenize raw references before loading Anima to minimize peak VRAM."""
    encoded = encode_dual_query_reference_images(
        config, destination, paths, device=device
    )
    return encoded["tokens"].unsqueeze(0)


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

    @torch.no_grad()
    def encode_raw_reference_images(
        self,
        config: dict[str, Any],
        destination: Path,
        paths: list[Path],
    ) -> torch.Tensor:
        """Run raw images through the exact frozen production feature path."""
        return prepare_reference_tokens(
            config, destination, paths, device=str(self.device)
        )

    @torch.no_grad()
    def set_raw_references(
        self,
        config: dict[str, Any],
        destination: Path,
        paths: list[Path],
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Encode raw images and immediately activate their native-K/V style."""
        tokens = self.encode_raw_reference_images(config, destination, paths)
        return self.set_references(tokens, strength=strength)

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


class RetrievalFewShotKVStyleAdapter(nn.Module):
    """Retrieve and exactly mix a sparse dictionary of K/V-only LoRAs."""

    def __init__(
        self,
        *,
        reader: DetailPreservingTypedSlotReader,
        anima: nn.Module,
        anchor_codes: torch.Tensor,
        anchor_reference_counts: torch.Tensor,
        artist_ids: list[str],
        teacher_down: torch.Tensor,
        teacher_up: torch.Tensor,
        neighbors: int = 8,
        temperature: float = 0.1,
        dictionary_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if anchor_codes.shape[1] != len(artist_ids):
            raise ValueError("Reader anchor and artist dictionary sizes disagree")
        if teacher_down.shape[0] != len(artist_ids) or teacher_up.shape[0] != len(
            artist_ids
        ):
            raise ValueError("LoRA factor and artist dictionary sizes disagree")
        self.reader = reader.requires_grad_(False).eval()
        self.injector = NativeKVFactorInjector(anima)
        self.register_buffer("anchor_codes", anchor_codes, persistent=False)
        self.register_buffer(
            "anchor_reference_counts",
            anchor_reference_counts.to(torch.int64),
            persistent=False,
        )
        indices = (
            torch.arange(len(artist_ids), dtype=torch.long)
            if dictionary_indices is None
            else dictionary_indices.to(torch.long).cpu()
        )
        self.register_buffer("dictionary_indices", indices, persistent=False)
        self.artist_ids = list(artist_ids)
        # Keep the full dictionary in host RAM. Only selected rows are
        # transferred when references change.
        self.teacher_down = teacher_down.detach().cpu().to(torch.float16)
        self.teacher_up = teacher_up.detach().cpu().to(torch.float16)
        self.neighbors = int(neighbors)
        self.temperature = float(temperature)
        self.active_strength = 0.0
        self.last_retrieval: list[dict[str, Any]] = []

    @property
    def device(self) -> torch.device:
        return next(self.reader.parameters()).device

    def _anchor_position(self, reference_count: int) -> int:
        exact = torch.nonzero(
            self.anchor_reference_counts == int(reference_count)
        ).flatten()
        if exact.numel() == 1:
            return int(exact.item())
        distance = (self.anchor_reference_counts - int(reference_count)).abs()
        return int(distance.argmin().item())

    @torch.no_grad()
    def encode_reference_tokens(
        self,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
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

        selected_rows = []
        selected_weights = []
        retrieval = []
        dictionary = self.dictionary_indices.to(self.anchor_codes.device)
        for row in range(style_codes.shape[0]):
            reference_count = int(reference_mask[row].sum().item())
            position = self._anchor_position(reference_count)
            anchors = self.anchor_codes[position, dictionary].float().flatten(1)
            query = style_codes[row : row + 1].float().flatten(1)
            coefficients = _visual_knn_coefficients(
                anchors,
                query,
                neighbors=self.neighbors,
                temperature=self.temperature,
            )
            weights, local = coefficients.topk(
                min(self.neighbors, coefficients.shape[-1]), dim=-1
            )
            global_indices = dictionary[local[0]].cpu()
            selected_rows.append(global_indices)
            selected_weights.append(weights[0].cpu())
            retrieval.append({
                "reference_count": reference_count,
                "anchor_reference_count": int(
                    self.anchor_reference_counts[position].item()
                ),
                "artist_indices": global_indices.tolist(),
                "artist_ids": [
                    self.artist_ids[int(index)] for index in global_indices
                ],
                "weights": weights[0].cpu().tolist(),
            })
        neighbor_indices = torch.stack(selected_rows)
        weights = torch.stack(selected_weights)
        mixed_down, mixed_up = concatenate_weighted_lora_factors(
            self.teacher_down[neighbor_indices],
            self.teacher_up[neighbor_indices],
            weights,
        )
        return (
            style_codes,
            mixed_down.to(device=self.device, dtype=torch.bfloat16),
            mixed_up.to(device=self.device, dtype=torch.bfloat16),
            retrieval,
        )

    @torch.no_grad()
    def set_references(
        self,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor | None = None,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        style_codes, down, up, retrieval = self.encode_reference_tokens(
            reference_tokens, reference_mask
        )
        self.injector.set_factors(down, up, strength=float(strength))
        self.active_strength = float(strength)
        self.last_retrieval = retrieval
        return style_codes

    @torch.no_grad()
    def set_raw_references(
        self,
        config: dict[str, Any],
        destination: Path,
        paths: list[Path],
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        tokens = prepare_reference_tokens(
            config, destination, paths, device=str(self.device)
        )
        return self.set_references(tokens, strength=strength)

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
    def from_cache(
        cls,
        config: dict[str, Any],
        destination: Path,
        anima: nn.Module,
        *,
        device: str = "cuda",
        neighbors: int = 8,
        temperature: float = 0.1,
        dictionary_indices: torch.Tensor | None = None,
    ) -> "RetrievalFewShotKVStyleAdapter":
        cache_cfg = dict(config["kv_lora_reader_anchor_cache"])
        cache_root = destination / str(cache_cfg["output_directory"])
        anchors = load_file(cache_root / "anchors.safetensors", device="cpu")
        summary = json.loads(
            (cache_root / "summary.json").read_text(encoding="utf-8")
        )
        artist_ids = [str(value) for value in summary["artist_ids"]]
        lora_root = destination / str(cache_cfg["lora_directory"])
        blocks = int(
            config["kv_activation_generalizing_modulator"].get("blocks", 28)
        )
        loaded_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
            lora_root, blocks=blocks, dtype=torch.float16
        )
        if loaded_ids != artist_ids:
            raise RuntimeError("Anchor cache and LoRA factor bank artist order disagree")
        reader_state = torch.load(
            destination / str(cache_cfg["reader_checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        oracle_cfg = dict(config["kv_lora_oracle_bootstrap"])
        detail_cfg = _oracle_detail_config(config, oracle_cfg)
        reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"]))
        reader.load_state_dict(reader_state["reader"], strict=True)
        reader.to(device=device, dtype=torch.bfloat16)
        return cls(
            reader=reader,
            anima=anima,
            anchor_codes=anchors["anchors"].to(device=device),
            anchor_reference_counts=anchors["reference_counts"].to(device=device),
            artist_ids=artist_ids,
            teacher_down=teacher_down,
            teacher_up=teacher_up,
            neighbors=neighbors,
            temperature=temperature,
            dictionary_indices=dictionary_indices,
        )
