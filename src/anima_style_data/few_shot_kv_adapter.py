"""Minimal inference boundary for the learned few-shot native-K/V adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_external_samples import encode_dual_query_reference_images
from .kv_activation_modulation import (
    NativeKVFactorModulator,
    compress_lora_factors,
    load_kv_lora_factor_bank,
)
from .kv_activation_sampling import NativeKVFactorInjector
from .kv_generalizing_modulator import (
    _visual_knn_coefficients,
    concatenate_weighted_lora_factors,
)
from .kv_mixture_analysis import _sparse_ridge_coefficients
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


@torch.no_grad()
def compress_mean_lora_dictionary(
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    *,
    target_rank: int,
    device: str | torch.device,
    oversample: int = 16,
    power_iterations: int = 1,
    seed: int = 20260824,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress the exact arithmetic mean of a LoRA dictionary."""

    if teacher_down.ndim != 5 or teacher_up.ndim != 5:
        raise ValueError("Expected [artist,block,K/V,rank,input/output] factors")
    if teacher_down.shape[:3] != teacher_up.shape[:3]:
        raise ValueError("LoRA dictionary dimensions disagree")
    if teacher_down.shape[-2] != teacher_up.shape[-1]:
        raise ValueError("LoRA dictionary ranks disagree")
    artists, blocks, kinds, teacher_rank, input_dim = teacher_down.shape
    output_dim = int(teacher_up.shape[-2])
    down = teacher_down.to(device=device).permute(1, 2, 0, 3, 4).reshape(
        blocks, kinds, artists * teacher_rank, input_dim
    )
    up = teacher_up.to(device=device).permute(1, 2, 3, 0, 4).reshape(
        blocks, kinds, output_dim, artists * teacher_rank
    ) / float(artists)
    return compress_lora_factors(
        down,
        up,
        target_rank=target_rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


@torch.no_grad()
def cache_count_aware_lora_common(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Cache the full-dictionary affine common used by signed retrieval."""

    cfg = dict(config["kv_lora_count_aware_adapter"])
    anchor_cfg = dict(config["kv_lora_reader_anchor_cache"])
    output = destination / str(anchor_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(cfg.get("ridge_rank", 64))
    common_file = output / str(
        cfg.get("common_file", f"affine-common-rank{rank}.safetensors")
    )
    lora_root = destination / str(anchor_cfg["lora_directory"])
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        lora_root,
        blocks=int(config["kv_activation_generalizing_modulator"].get("blocks", 28)),
        dtype=torch.float16,
    )
    down, up = compress_mean_lora_dictionary(
        teacher_down,
        teacher_up,
        target_rank=rank,
        device=str(cfg.get("device", "cuda")),
        oversample=int(cfg.get("compression_oversample", 16)),
        power_iterations=int(cfg.get("compression_power_iterations", 1)),
        seed=int(cfg.get("compression_seed", 20260824)),
    )
    temporary = common_file.with_suffix(common_file.suffix + ".tmp")
    save_file(
        {
            "down": down.cpu().to(torch.float16).contiguous(),
            "up": up.cpu().to(torch.float16).contiguous(),
        },
        temporary,
    )
    temporary.replace(common_file)
    summary = {
        "artists": len(artist_ids),
        "rank": rank,
        "shape_down": list(down.shape),
        "shape_up": list(up.shape),
        "bytes": common_file.stat().st_size,
        "path": str(common_file),
    }
    from .io import write_json

    write_json(output / "count_aware_common_summary.json", summary)
    return summary


def _safe_sample_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


@torch.no_grad()
def sample_count_aware_raw_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render the seven persistent raw references through the live adapter."""

    from PIL import Image, ImageDraw, ImageOps

    from .dual_query_external_samples import load_dual_query_external_sample
    from .io import write_json
    from .lora_functional_distillation import _preview_pixels
    from .style_transfer import (
        _load_sampling_vae,
        _optimize_frozen_anima,
        _resolve_anima_model,
    )
    from .synthetic_teacher import _sample_anima_batch

    cfg = dict(config["kv_lora_count_aware_raw_sample"])
    device = str(cfg.get("device", "cuda"))
    batch_size = int(cfg.get("batch_size", 4))
    strengths = [float(value) for value in cfg.get("strengths", [1.0])]
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    prepared = load_dual_query_external_sample(config, destination)
    sample_cfg = dict(prepared["cfg"])
    paths = [Path(value) for value in prepared["paths"]]
    names = [path.parent.name for path in paths]
    reference_tokens = prepared["reference_tokens"].unsqueeze(1)
    positive = prepared["positive"].to(device=device, dtype=torch.bfloat16)
    negative = prepared["negative"].to(device=device, dtype=torch.bfloat16)
    if positive.ndim == 2:
        positive = positive[None]
    if negative.ndim == 2:
        negative = negative[None]

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    adapter = CountAwareRetrievalFewShotKVStyleAdapter.from_cache(
        config, destination, anima, device=device
    )
    style_codes, down, up, retrieval = adapter.encode_reference_tokens(
        reference_tokens
    )
    width = int(sample_cfg["width"])
    height = int(sample_cfg["height"])
    seed = int(sample_cfg["seed"])
    steps = int(sample_cfg["steps"])
    shift = float(sample_cfg.get("flow_shift", 3.0))
    text_cfg = float(sample_cfg["cfg"])
    base_noise = torch.randn(
        1,
        16,
        1,
        height // 8,
        width // 8,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)

    def denoise(
        factors: tuple[torch.Tensor, torch.Tensor] | None,
        strength: float,
    ) -> torch.Tensor:
        rows = 1 if factors is None else int(factors[0].shape[0])
        values = []
        for start in range(0, rows, batch_size):
            stop = min(rows, start + batch_size)
            active_rows = stop - start
            if factors is None:
                adapter.injector.disable()
            else:
                adapter.injector.set_factors(
                    factors[0][start:stop],
                    factors[1][start:stop],
                    strength=float(strength),
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values.append(
                    _sample_anima_batch(
                        anima,
                        base_noise.repeat(active_rows, 1, 1, 1, 1),
                        positive.expand(active_rows, -1, -1),
                        negative.expand(active_rows, -1, -1),
                        sigmas,
                        text_cfg=text_cfg,
                        speed=None,
                        generation_seeds=[seed] * active_rows,
                    ).cpu()
                )
        return torch.cat(values)

    baseline = denoise(None, 0.0)
    predictions = {strength: denoise((down, up), strength) for strength in strengths}
    adapter.close()
    del anima, adapter, down, up
    torch.cuda.empty_cache()

    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    vae.requires_grad_(False).eval()
    latent_groups = {
        "Frozen Anima": baseline.expand(len(names), -1, -1, -1, -1)
    }
    latent_groups.update({f"Few-shot {value:g}x": latent for value, latent in predictions.items()})
    images: dict[str, list[Image.Image]] = {}
    for label, latent in latent_groups.items():
        decoded: list[Image.Image] = []
        for start in range(0, len(names), batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded.extend(
                    _preview_pixels(
                        vae.decode_to_pixels(latent[start : start + batch_size].to(device))
                    )
                )
        images[label] = decoded
    del vae
    torch.cuda.empty_cache()

    tile_width = int(cfg.get("panel_tile_width", 320))
    tile_height = round(height * tile_width / width)
    label_height = 30
    labels = ["Raw reference", *latent_groups]
    sheet = Image.new(
        "RGB",
        (tile_width * len(names), (tile_height + label_height) * len(labels)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    raw_images = []
    for path in paths:
        with Image.open(path) as image:
            raw_images.append(ImageOps.pad(
                image.convert("RGB"),
                (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
                color="white",
            ))
    for row, label in enumerate(labels):
        y = row * (tile_height + label_height)
        for column, name in enumerate(names):
            x = column * tile_width
            image = (
                raw_images[column]
                if label == "Raw reference"
                else images[label][column].resize(
                    (tile_width, tile_height), Image.Resampling.LANCZOS
                )
            )
            sheet.paste(image, (x, y + label_height))
            draw.text((x + 6, y + 7), f"{name} | {label}", fill="black")
    panel = output / "raw-reference-few-shot-overview.jpg"
    sheet.save(panel, "JPEG", quality=94, subsampling=0)
    for index, name in enumerate(names):
        row_output = output / f"{index:02d}-{_safe_sample_name(name)}"
        row_output.mkdir(exist_ok=True)
        raw_images[index].save(row_output / "Raw_reference.webp", "WEBP", quality=95)
        for label in latent_groups:
            images[label][index].save(
                row_output / f"{_safe_sample_name(label)}.webp", "WEBP", quality=95
            )

    baseline_rows = baseline.float().expand(len(names), -1, -1, -1, -1)
    metrics: dict[str, Any] = {}
    for strength, latent in predictions.items():
        effect = (latent.float() - baseline_rows).flatten(1)
        common = effect.mean(dim=0, keepdim=True)
        centered = effect - common
        individual_rms = effect.square().mean(dim=1).sqrt().clamp_min(1e-8)
        normalized = torch.nn.functional.normalize(effect, dim=-1)
        pairwise = normalized @ normalized.t()
        mask = ~torch.eye(len(names), dtype=torch.bool)
        metrics[f"{strength:g}x"] = {
            "effect_rms": float(individual_rms.mean()),
            "common_output_ratio": float(
                common.square().mean().sqrt() / individual_rms.mean()
            ),
            "artist_centered_to_effect_ratio": float(
                centered.square().mean(dim=1).sqrt().mean() / individual_rms.mean()
            ),
            "mean_pairwise_effect_cosine": float(pairwise[mask].mean()),
        }
    summary = {
        "references": [str(path) for path in paths],
        "prompt": str(sample_cfg["prompt"]),
        "negative_prompt": str(sample_cfg["negative_prompt"]),
        "seed": seed,
        "steps": steps,
        "text_cfg": text_cfg,
        "strengths": strengths,
        "style_code_shape": list(style_codes.shape),
        "retrieval": retrieval,
        "metrics": metrics,
        "panel": str(panel),
    }
    write_json(output / "summary.json", summary)
    return summary


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


class CountAwareRetrievalFewShotKVStyleAdapter(
    RetrievalFewShotKVStyleAdapter
):
    """Route one reference to convex retrieval and multiple to signed ridge.

    Fixed-heldout evaluation showed that a one-reference Reader code is not
    stable enough for extrapolation: exact kNN LoRA mixtures preserve quality
    better.  With two or more references, the averaged code supports an affine
    signed mixture and a rank-64 compression, which improves the final Anima
    latent effect while halving the injected rank.
    """

    def __init__(
        self,
        *,
        ridge_common_down: torch.Tensor,
        ridge_common_up: torch.Tensor,
        convex_dictionary_size: int = 256,
        ridge_min_references: int = 2,
        ridge_neighbors: int = 32,
        ridge_regularization: float = 0.05,
        ridge_rank: int = 64,
        ridge_gain: float = 1.5,
        compression_oversample: int = 16,
        compression_power_iterations: int = 1,
        compression_seed: int = 20260824,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if ridge_common_down.ndim != 4 or ridge_common_up.ndim != 4:
            raise ValueError("Common LoRA factors must be [block,K/V,rank,dim]")
        if ridge_common_down.shape[:2] != ridge_common_up.shape[:2]:
            raise ValueError("Common LoRA block/K/V dimensions disagree")
        if ridge_common_down.shape[-2] != ridge_common_up.shape[-1]:
            raise ValueError("Common LoRA ranks disagree")
        self.register_buffer(
            "ridge_common_down",
            ridge_common_down.to(device=self.device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "ridge_common_up",
            ridge_common_up.to(device=self.device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.convex_dictionary_size = min(
            int(convex_dictionary_size), len(self.artist_ids)
        )
        self.ridge_min_references = int(ridge_min_references)
        self.ridge_neighbors = min(int(ridge_neighbors), len(self.artist_ids))
        self.ridge_regularization = float(ridge_regularization)
        self.ridge_rank = int(ridge_rank)
        self.ridge_gain = float(ridge_gain)
        self.compression_oversample = int(compression_oversample)
        self.compression_power_iterations = int(compression_power_iterations)
        self.compression_seed = int(compression_seed)

    def _convex_factors(
        self,
        query: torch.Tensor,
        anchor_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        dictionary = self.dictionary_indices[
            : self.convex_dictionary_size
        ].to(self.anchor_codes.device)
        anchors = self.anchor_codes[anchor_position, dictionary].float().flatten(1)
        coefficients = _visual_knn_coefficients(
            anchors,
            query.float().flatten(1),
            neighbors=self.neighbors,
            temperature=self.temperature,
        )
        weights, local = coefficients.topk(
            min(self.neighbors, coefficients.shape[-1]), dim=-1
        )
        global_indices = dictionary[local[0]].cpu()
        down, up = concatenate_weighted_lora_factors(
            self.teacher_down[global_indices][None],
            self.teacher_up[global_indices][None],
            weights.cpu(),
        )
        return down[0], up[0], {
            "route": "convex_knn",
            "artist_indices": global_indices.tolist(),
            "artist_ids": [self.artist_ids[int(index)] for index in global_indices],
            "weights": weights[0].cpu().tolist(),
            "effective_gain": 1.0,
        }

    def _ridge_factors(
        self,
        query: torch.Tensor,
        anchor_position: int,
        *,
        row_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        dictionary = self.dictionary_indices.to(self.anchor_codes.device)
        anchors = self.anchor_codes[anchor_position, dictionary].float().flatten(1)
        coefficients = _sparse_ridge_coefficients(
            anchors,
            query.float().flatten(1),
            neighbors=self.ridge_neighbors,
            ridge=self.ridge_regularization,
        )
        _, local = coefficients.abs().topk(self.ridge_neighbors, dim=-1)
        weights = coefficients.gather(-1, local)[0].cpu()
        global_indices = dictionary[local[0]].cpu()
        selected_down = self.teacher_down[global_indices].to(self.device)
        selected_up = self.teacher_up[global_indices].to(self.device)
        neighbors, blocks, kinds, teacher_rank, input_dim = selected_down.shape
        output_dim = int(selected_up.shape[-2])
        selected_down = selected_down.permute(1, 2, 0, 3, 4).reshape(
            blocks, kinds, neighbors * teacher_rank, input_dim
        )
        selected_up = (
            selected_up * weights[:, None, None, None, None].to(selected_up)
        ).permute(1, 2, 3, 0, 4).reshape(
            blocks, kinds, output_dim, neighbors * teacher_rank
        )
        common_weight = 1.0 - weights.sum()
        combined_down = torch.cat((self.ridge_common_down, selected_down), dim=-2)
        combined_up = torch.cat((
            self.ridge_common_up * common_weight.to(self.ridge_common_up),
            selected_up,
        ), dim=-1)
        down, up = compress_lora_factors(
            combined_down,
            combined_up,
            target_rank=self.ridge_rank,
            oversample=self.compression_oversample,
            power_iterations=self.compression_power_iterations,
            seed=self.compression_seed + int(row_seed) * 1_000_003,
        )
        down = down * self.ridge_gain**0.5
        up = up * self.ridge_gain**0.5
        return down, up, {
            "route": "sparse_signed_ridge",
            "artist_indices": global_indices.tolist(),
            "artist_ids": [self.artist_ids[int(index)] for index in global_indices],
            "weights": weights.tolist(),
            "common_weight": float(common_weight),
            "effective_gain": self.ridge_gain,
        }

    @staticmethod
    def _pad_factor_rank(
        down: torch.Tensor,
        up: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = int(rank) - int(down.shape[-2])
        if missing <= 0:
            return down, up
        down_padding = down.new_zeros(*down.shape[:-2], missing, down.shape[-1])
        up_padding = up.new_zeros(*up.shape[:-1], missing)
        return (
            torch.cat((down, down_padding), dim=-2),
            torch.cat((up, up_padding), dim=-1),
        )

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
            reference_mask = reference_mask.to(device=self.device, dtype=torch.bool)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            style_codes = self.reader(tokens, reference_mask).tokens

        down_rows: list[torch.Tensor] = []
        up_rows: list[torch.Tensor] = []
        retrieval: list[dict[str, Any]] = []
        for row in range(style_codes.shape[0]):
            reference_count = int(reference_mask[row].sum().item())
            anchor_position = self._anchor_position(reference_count)
            query = style_codes[row : row + 1]
            if reference_count < self.ridge_min_references:
                down, up, details = self._convex_factors(query, anchor_position)
            else:
                down, up, details = self._ridge_factors(
                    query, anchor_position, row_seed=row
                )
            down = down.to(device=self.device, dtype=torch.bfloat16)
            up = up.to(device=self.device, dtype=torch.bfloat16)
            details.update({
                "reference_count": reference_count,
                "anchor_reference_count": int(
                    self.anchor_reference_counts[anchor_position].item()
                ),
            })
            down_rows.append(down)
            up_rows.append(up)
            retrieval.append(details)

        max_rank = max(int(value.shape[-2]) for value in down_rows)
        padded = [
            self._pad_factor_rank(down, up, max_rank)
            for down, up in zip(down_rows, up_rows, strict=True)
        ]
        return (
            style_codes,
            torch.stack([value[0] for value in padded]).to(
                device=self.device, dtype=torch.bfloat16
            ),
            torch.stack([value[1] for value in padded]).to(
                device=self.device, dtype=torch.bfloat16
            ),
            retrieval,
        )

    @classmethod
    def from_cache(
        cls,
        config: dict[str, Any],
        destination: Path,
        anima: nn.Module,
        *,
        device: str = "cuda",
    ) -> "CountAwareRetrievalFewShotKVStyleAdapter":
        cfg = dict(config["kv_lora_count_aware_adapter"])
        cache_cfg = dict(config["kv_lora_reader_anchor_cache"])
        cache_root = destination / str(cache_cfg["output_directory"])
        anchors = load_file(cache_root / "anchors.safetensors", device="cpu")
        summary = json.loads(
            (cache_root / "summary.json").read_text(encoding="utf-8")
        )
        artist_ids = [str(value) for value in summary["artist_ids"]]
        common_file = cache_root / str(
            cfg.get(
                "common_file",
                f"affine-common-rank{int(cfg.get('ridge_rank', 64))}.safetensors",
            )
        )
        if not common_file.exists():
            raise FileNotFoundError(
                f"Missing {common_file}; run kv-lora-count-aware-cache first"
            )
        common = load_file(common_file, device="cpu")
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
            neighbors=int(cfg.get("convex_neighbors", 8)),
            temperature=float(cfg.get("convex_temperature", 0.1)),
            ridge_common_down=common["down"],
            ridge_common_up=common["up"],
            convex_dictionary_size=int(cfg.get("convex_dictionary_artists", 256)),
            ridge_min_references=int(cfg.get("ridge_min_references", 2)),
            ridge_neighbors=int(cfg.get("ridge_neighbors", 32)),
            ridge_regularization=float(cfg.get("ridge_regularization", 0.05)),
            ridge_rank=int(cfg.get("ridge_rank", 64)),
            ridge_gain=float(cfg.get("ridge_gain", 1.5)),
            compression_oversample=int(cfg.get("compression_oversample", 16)),
            compression_power_iterations=int(
                cfg.get("compression_power_iterations", 1)
            ),
            compression_seed=int(cfg.get("compression_seed", 20260824)),
        )
