from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .dual_query_external_samples import load_dual_query_external_sample
from .io import write_json
from .kv_activation_modulation import (
    NativeKVFactorModulator,
    apply_kv_factors,
    canonicalize_lora_factor_bank,
    load_kv_lora_factor_bank,
)
from .lora_functional_distillation import _preview_pixels
from .style_transfer import (
    _load_sampling_vae,
    _optimize_frozen_anima,
    _resolve_anima_model,
)
from .synthetic_teacher import _sample_anima_batch


def _repeat_factor_rows(value: torch.Tensor, batch: int) -> torch.Tensor:
    """Match style rows to native, CFG, or repeated native+CFG batches."""

    rows = int(value.shape[0])
    if rows == batch:
        return value
    if rows == 1:
        return value.expand(batch, *value.shape[1:])
    if batch % rows:
        raise ValueError(f"Cannot broadcast {rows} style rows to batch {batch}")
    return value.repeat(batch // rows, *([1] * (value.ndim - 1)))


class NativeKVFactorInjector:
    """Add rank-factorized deltas at Anima's native text K/V projections."""

    def __init__(self, anima: torch.nn.Module) -> None:
        self.down: torch.Tensor | None = None
        self.up: torch.Tensor | None = None
        self.strength = 1.0
        self.enabled = False
        self.handles: list[Any] = []
        for block_index, block in enumerate(anima.blocks):
            cross = block.cross_attn
            if hasattr(cross, "kv_proj"):
                self.handles.append(
                    cross.kv_proj.register_forward_hook(
                        self._fused_hook(block_index)
                    )
                )
            elif hasattr(cross, "k_proj") and hasattr(cross, "v_proj"):
                self.handles.append(
                    cross.k_proj.register_forward_hook(
                        self._split_hook(block_index, 0)
                    )
                )
                self.handles.append(
                    cross.v_proj.register_forward_hook(
                        self._split_hook(block_index, 1)
                    )
                )
            else:
                raise TypeError("Anima cross-attention exposes no text K/V projection")

    def set_factors(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        strength: float = 1.0,
    ) -> None:
        if down.ndim != 5 or up.ndim != 5:
            raise ValueError("Expected [style, block, K/V, ...] factor banks")
        if down.shape[:3] != up.shape[:3] or down.shape[2] != 2:
            raise ValueError("K/V factor-bank leading dimensions disagree")
        self.down = down
        self.up = up
        self.strength = float(strength)
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _block_factors(
        self, block_index: int, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.down is None or self.up is None:
            raise RuntimeError("K/V factors were not set")
        return (
            _repeat_factor_rows(self.down[:, block_index], batch),
            _repeat_factor_rows(self.up[:, block_index], batch),
        )

    def _fused_hook(self, block_index: int):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            context = inputs[0]
            down, up = self._block_factors(block_index, int(context.shape[0]))
            delta = apply_kv_factors(
                context,
                down.to(device=context.device, dtype=context.dtype),
                up.to(device=context.device, dtype=context.dtype),
            )
            fused = torch.cat((delta[:, 0], delta[:, 1]), dim=-1)
            return output + fused.to(output.dtype) * self.strength

        return hook

    def _split_hook(self, block_index: int, kind: int):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            context = inputs[0]
            down, up = self._block_factors(block_index, int(context.shape[0]))
            hidden = torch.einsum(
                "bnc,brc->bnr", context, down[:, kind].to(context.dtype)
            )
            delta = torch.einsum(
                "bnr,bor->bno", hidden, up[:, kind].to(context.dtype)
            )
            return output + delta.to(output.dtype) * self.strength

        return hook


def _load_predicted_and_teacher_factors(
    cfg: dict[str, Any], destination: Path, indices: list[int], device: str
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(
        destination / str(cfg["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    model_cfg = dict(checkpoint["config"]["model"])
    style_codes = checkpoint["style_codes"]
    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(checkpoint["config"]["lora_directory"]),
        blocks=int(checkpoint["config"].get("blocks", 28)),
    )
    if max(indices) >= len(artist_ids):
        raise IndexError("Sample artist index exceeds the K/V-LoRA bank")
    teacher_down = teacher_down[indices].to(device=device)
    teacher_up = teacher_up[indices].to(device=device)
    teacher_down, teacher_up = canonicalize_lora_factor_bank(
        teacher_down,
        teacher_up,
        chunk_size=int(cfg.get("canonicalization_chunk_size", 64)),
    )
    rank = int(teacher_down.shape[-2])
    model = NativeKVFactorModulator(
        style_dim=int(style_codes.shape[-1]),
        blocks=int(teacher_down.shape[1]),
        rank=rank,
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        **model_cfg,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=torch.bfloat16).eval()
    selected_codes = style_codes[indices].to(device=device, dtype=torch.bfloat16)
    predicted_down = []
    predicted_up = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for block in range(model.blocks):
            down, up = model(selected_codes, block)
            predicted_down.append(down)
            predicted_up.append(up)
    predicted_down_tensor = torch.stack(predicted_down, dim=1)
    predicted_up_tensor = torch.stack(predicted_up, dim=1)
    del checkpoint, model, style_codes, selected_codes
    gc.collect()
    torch.cuda.empty_cache()
    return (
        [artist_ids[index] for index in indices],
        teacher_down.to(dtype=torch.bfloat16),
        teacher_up.to(dtype=torch.bfloat16),
        predicted_down_tensor,
        predicted_up_tensor,
    )


def _factor_prompt_metrics(
    context: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    predicted_down: torch.Tensor,
    predicted_up: torch.Tensor,
) -> dict[str, float]:
    rows: dict[str, list[torch.Tensor]] = {
        "cosine": [], "k_cosine": [], "v_cosine": [], "rms_ratio": []
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for block in range(int(teacher_down.shape[1])):
            expanded = context.expand(teacher_down.shape[0], -1, -1)
            teacher = apply_kv_factors(
                expanded, teacher_down[:, block], teacher_up[:, block]
            ).float()
            predicted = apply_kv_factors(
                expanded, predicted_down[:, block], predicted_up[:, block]
            ).float()
            cosine = F.cosine_similarity(
                predicted.flatten(2), teacher.flatten(2), dim=-1
            )
            rows["cosine"].append(cosine.mean())
            rows["k_cosine"].append(cosine[:, 0].mean())
            rows["v_cosine"].append(cosine[:, 1].mean())
            teacher_rms = teacher.square().mean(dim=(2, 3)).sqrt()
            predicted_rms = predicted.square().mean(dim=(2, 3)).sqrt()
            rows["rms_ratio"].append(
                (predicted_rms / teacher_rms.clamp_min(1e-7)).mean()
            )
    return {
        key: float(torch.stack(values).mean()) for key, values in rows.items()
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "artist"


def _save_panel(
    output: Path,
    artist_ids: list[str],
    images: dict[str, list[Image.Image]],
    labels: list[str],
    *,
    tile_width: int,
) -> Path:
    source_width, source_height = images[labels[0]][0].size
    tile_height = round(source_height * tile_width / source_width)
    label_height = 30
    sheet = Image.new(
        "RGB",
        (tile_width * len(labels), (tile_height + label_height) * len(artist_ids)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, artist in enumerate(artist_ids):
        y = row * (tile_height + label_height)
        for column, label in enumerate(labels):
            image = images[label][row].resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
            x = column * tile_width
            sheet.paste(image, (x, y + label_height))
            title = f"{artist} | {label}" if column == 0 else label
            draw.text((x + 6, y + 7), title, fill="black")
    path = output / "teacher-vs-predicted-overview.jpg"
    sheet.save(path, "JPEG", quality=94, subsampling=0)
    return path


@torch.no_grad()
def sample_kv_activation_modulator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["kv_activation_modulator_sample"])
    device = str(cfg.get("device", "cuda"))
    indices = [int(value) for value in cfg.get("artist_indices", range(7))]
    strengths = [float(value) for value in cfg.get("predicted_strengths", [1.0])]
    batch_size = int(cfg.get("batch_size", 4))
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)

    prepared = load_dual_query_external_sample(config, destination)
    sample_cfg = dict(prepared["cfg"])
    positive = prepared["positive"].to(device=device, dtype=torch.bfloat16)
    negative = prepared["negative"].to(device=device, dtype=torch.bfloat16)
    if positive.ndim == 2:
        positive = positive[None]
    if negative.ndim == 2:
        negative = negative[None]
    artist_ids, teacher_down, teacher_up, predicted_down, predicted_up = (
        _load_predicted_and_teacher_factors(cfg, destination, indices, device)
    )
    prompt_metrics = _factor_prompt_metrics(
        positive,
        teacher_down,
        teacher_up,
        predicted_down,
        predicted_up,
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    injector = NativeKVFactorInjector(anima)
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

    def denoise_rows(
        down: torch.Tensor | None,
        up: torch.Tensor | None,
        strength: float,
    ) -> torch.Tensor:
        values = []
        row_count = 1 if down is None else int(down.shape[0])
        for start in range(0, row_count, batch_size):
            stop = min(row_count, start + batch_size)
            rows = stop - start
            if down is None or up is None:
                injector.disable()
            else:
                injector.set_factors(
                    down[start:stop], up[start:stop], strength=strength
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values.append(
                    _sample_anima_batch(
                        anima,
                        base_noise.repeat(rows, 1, 1, 1, 1),
                        positive.expand(rows, -1, -1),
                        negative.expand(rows, -1, -1),
                        sigmas,
                        text_cfg=text_cfg,
                        speed=None,
                        generation_seeds=[seed] * rows,
                    ).cpu()
                )
        return torch.cat(values)

    baseline = denoise_rows(None, None, 0.0)
    teacher = denoise_rows(teacher_down, teacher_up, 1.0)
    predictions = {
        strength: denoise_rows(predicted_down, predicted_up, strength)
        for strength in strengths
    }
    injector.close()
    del anima, injector
    torch.cuda.empty_cache()

    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    vae.requires_grad_(False).eval()
    latent_groups = {"Frozen Anima": baseline.expand(len(artist_ids), -1, -1, -1, -1)}
    latent_groups["Teacher LoRA 1.0x"] = teacher
    for strength, value in predictions.items():
        latent_groups[f"Predicted {strength:g}x"] = value
    images: dict[str, list[Image.Image]] = {}
    for label, latent in latent_groups.items():
        decoded = []
        for start in range(0, latent.shape[0], batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded.extend(
                    _preview_pixels(
                        vae.decode_to_pixels(
                            latent[start : start + batch_size].to(device)
                        )
                    )
                )
        images[label] = decoded
    labels = list(latent_groups)
    for row, artist in enumerate(artist_ids):
        artist_dir = output / f"{row:02d}-{_safe_name(artist)}"
        artist_dir.mkdir(exist_ok=True)
        for label in labels:
            images[label][row].save(
                artist_dir / f"{_safe_name(label)}.webp", "WEBP", quality=95
            )
    panel = _save_panel(
        output,
        artist_ids,
        images,
        labels,
        tile_width=int(cfg.get("panel_tile_width", 416)),
    )

    base = baseline.float().expand(len(artist_ids), -1, -1, -1, -1)
    teacher_effect = (teacher.float() - base).flatten(1)
    latent_metrics = {}
    for strength, value in predictions.items():
        predicted_effect = (value.float() - base).flatten(1)
        difference = (value.float() - teacher.float()).flatten(1)
        teacher_distance = teacher_effect.square().mean(dim=1).sqrt()
        latent_metrics[f"predicted_{strength:g}x"] = {
            "effect_to_teacher_ratio": float(
                (
                    predicted_effect.square().mean(dim=1).sqrt()
                    / teacher_distance.clamp_min(1e-8)
                ).mean()
            ),
            "teacher_direction_cosine": float(
                F.cosine_similarity(predicted_effect, teacher_effect, dim=1).mean()
            ),
            "paired_improvement": float(
                (
                    1
                    - difference.square().mean(dim=1).sqrt()
                    / teacher_distance.clamp_min(1e-8)
                ).mean()
            ),
        }
    summary = {
        "artists": artist_ids,
        "artist_indices": indices,
        "prompt": str(sample_cfg["prompt"]),
        "negative_prompt": str(sample_cfg["negative_prompt"]),
        "seed": seed,
        "steps": steps,
        "text_cfg": text_cfg,
        "width": width,
        "height": height,
        "predicted_strengths": strengths,
        "activation_metrics": prompt_metrics,
        "latent_metrics": latent_metrics,
        "panel": str(panel),
    }
    write_json(output / "summary.json", summary)
    return summary
