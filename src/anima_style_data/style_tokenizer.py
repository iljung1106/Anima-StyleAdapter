from __future__ import annotations

import copy
import gc
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch import nn

from .io import write_json
from .style_transfer import (
    ProductionStyleLoader,
    _learning_rate_multiplier,
    _load_sampling_vae,
    _make_sample_sheet,
    _optimize_frozen_anima,
    _pack_reference_tokens,
    _pad_text_conditions,
    _resolve_anima_model,
    _sample_flow_timesteps,
)


class AnimaStyleTokenizer(nn.Module):
    """Compress frozen Resampler slots into native Anima context tokens.

    The internal Resampler representation remains rich (128 x 1024). A
    learned attention pool creates one style descriptor per reference and a
    second attention pool combines references without depending on their
    order. The StyleTokenizer MLP then emits a compact set of tokens in
    Anima's post-LLM context width.
    """

    def __init__(
        self,
        *,
        source_dim: int = 1024,
        context_dim: int = 1024,
        output_tokens: int = 16,
        bottleneck_dim: int = 512,
        score_hidden_dim: int = 256,
        output_rms_init: float = 0.15,
    ) -> None:
        super().__init__()
        if min(source_dim, context_dim, output_tokens, bottleneck_dim) <= 0:
            raise ValueError("StyleTokenizer dimensions must be positive")
        if output_rms_init <= 0:
            raise ValueError("output_rms_init must be positive")
        self.source_dim = int(source_dim)
        self.context_dim = int(context_dim)
        self.output_tokens = int(output_tokens)
        self.source_norm = nn.LayerNorm(source_dim)
        self.slot_score = nn.Sequential(
            nn.Linear(source_dim, score_hidden_dim),
            nn.SiLU(),
            nn.Linear(score_hidden_dim, 1, bias=False),
        )
        self.reference_norm = nn.LayerNorm(source_dim)
        self.reference_score = nn.Sequential(
            nn.Linear(source_dim, score_hidden_dim),
            nn.SiLU(),
            nn.Linear(score_hidden_dim, 1, bias=False),
        )
        self.tokenizer = nn.Sequential(
            nn.LayerNorm(source_dim),
            nn.Linear(source_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, output_tokens * context_dim),
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(1, output_tokens, context_dim)
        )
        self.output_norm = nn.LayerNorm(context_dim)
        self.log_output_rms = nn.Parameter(
            torch.tensor(math.log(float(output_rms_init)), dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.slot_embedding, std=self.context_dim**-0.5)

    def forward(
        self, references: torch.Tensor, reference_mask: torch.Tensor
    ) -> torch.Tensor:
        if references.ndim != 4:
            raise ValueError(
                "references must have shape [batch, references, slots, dim]"
            )
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference_mask shape does not match references")
        if references.shape[-1] != self.source_dim:
            raise ValueError(
                f"Expected source width {self.source_dim}, got {references.shape[-1]}"
            )
        if not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one reference")

        normalized = self.source_norm(references)
        slot_weights = self.slot_score(normalized).squeeze(-1).softmax(dim=-1)
        per_reference = torch.sum(slot_weights[..., None] * references, dim=2)
        reference_logits = self.reference_score(
            self.reference_norm(per_reference)
        ).squeeze(-1)
        reference_logits = reference_logits.masked_fill(
            ~reference_mask, torch.finfo(reference_logits.dtype).min
        )
        reference_weights = reference_logits.softmax(dim=-1)
        descriptor = torch.sum(
            reference_weights[..., None] * per_reference, dim=1
        )
        tokens = self.tokenizer(descriptor).reshape(
            descriptor.shape[0], self.output_tokens, self.context_dim
        )
        tokens = self.output_norm(tokens + self.slot_embedding)
        # Match the empirical RMS of nonzero post-LLM Anima context tokens.
        # This is a context-coordinate scale, not an output gate.
        return (tokens * self.log_output_rms.exp()).to(dtype=references.dtype)


def insert_style_tokens(
    conditioning: torch.Tensor,
    conditioning_lengths: torch.Tensor,
    style_tokens: torch.Tensor,
) -> torch.Tensor:
    """Replace the first unused post-LLM positions while preserving length.

    A no-style condition is simply `conditioning` itself. Consequently null
    style is exactly the frozen Anima base path and needs neither learned null
    tokens nor style dropout.
    """

    if conditioning.ndim != 3 or style_tokens.ndim != 3:
        raise ValueError("conditioning and style_tokens must be rank-three")
    if conditioning.shape[0] != style_tokens.shape[0]:
        raise ValueError("conditioning and style token batch sizes differ")
    if conditioning.shape[-1] != style_tokens.shape[-1]:
        raise ValueError("conditioning and style token widths differ")
    lengths = conditioning_lengths.to(
        device=conditioning.device, dtype=torch.long
    )
    if lengths.shape != (conditioning.shape[0],):
        raise ValueError("conditioning_lengths must contain one value per sample")
    positions = lengths[:, None] + torch.arange(
        style_tokens.shape[1], device=conditioning.device
    )[None]
    if bool((positions >= conditioning.shape[1]).any()):
        raise ValueError(
            "Cached text context does not have enough unused positions for style tokens"
        )
    indices = positions[..., None].expand(-1, -1, conditioning.shape[-1])
    return conditioning.scatter(1, indices, style_tokens.to(conditioning.dtype))


def _reference_tokens(
    batch: dict[str, Any], device: str, *, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "self":
        values = batch["cached_target_tokens"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )[:, None]
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
        return values, mask
    if mode not in {"heldout", "wrong_artist"}:
        raise ValueError(f"Unknown StyleTokenizer reference mode: {mode}")
    flat = batch["cached_reference_tokens"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    values = _pack_reference_tokens(flat, batch)
    mask = batch["reference_mask"].to(device, non_blocking=True)
    if mode == "wrong_artist":
        if values.shape[0] < 2:
            raise ValueError("wrong_artist evaluation requires batch_size >= 2")
        episodes = batch.get("episodes")
        if episodes:
            style_ids = [str(item.style_id) for item in episodes]
            donors = []
            for index, style_id in enumerate(style_ids):
                donor = next(
                    (
                        (index + offset) % len(style_ids)
                        for offset in range(1, len(style_ids))
                        if style_ids[(index + offset) % len(style_ids)] != style_id
                    ),
                    None,
                )
                if donor is None:
                    raise ValueError(
                        "wrong_artist evaluation batch contains only one artist"
                    )
                donors.append(donor)
            donor_indices = torch.tensor(donors, device=values.device)
            values = values[donor_indices]
            mask = mask[donor_indices]
        else:
            values = values.roll(1, dims=0)
            mask = mask.roll(1, dims=0)
    return values, mask


def _split_reference_views(
    reference_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build two disjoint, non-empty views for rows with at least two refs."""

    if reference_mask.ndim != 2:
        raise ValueError("reference_mask must have shape [batch, references]")
    eligible = reference_mask.sum(dim=1) >= 2
    selected = reference_mask[eligible]
    positions = torch.arange(
        reference_mask.shape[1], device=reference_mask.device
    )[None]
    first = selected & ((positions % 2) == 0)
    second = selected & ((positions % 2) == 1)
    return eligible, first, second


def _style_token_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    style_ids: list[str],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match aligned output slots across two same-artist reference views."""

    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("Style token views must have equal [batch, slots, dim] shape")
    if len(style_ids) != first.shape[0]:
        raise ValueError("style_ids must contain one value per token view")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")

    first = F.normalize(first.float(), dim=-1)
    second = F.normalize(second.float(), dim=-1)
    logits = torch.einsum("bsd,csd->bc", first, second)
    logits = logits / (first.shape[1] * temperature)
    positive = torch.tensor(
        [
            [left == right for right in style_ids]
            for left in style_ids
        ],
        dtype=torch.bool,
        device=logits.device,
    )
    log_prob = logits.log_softmax(dim=1)
    positive_log_prob = torch.logsumexp(
        log_prob.masked_fill(~positive, -torch.inf), dim=1
    )
    reverse_log_prob = logits.T.log_softmax(dim=1)
    reverse_positive_log_prob = torch.logsumexp(
        reverse_log_prob.masked_fill(~positive.T, -torch.inf), dim=1
    )
    loss = -0.5 * (
        positive_log_prob.mean() + reverse_positive_log_prob.mean()
    )
    similarities = logits * temperature
    negative = ~positive
    negative_similarity = (
        similarities[negative].mean()
        if bool(negative.any())
        else similarities.new_zeros(())
    )
    return loss, {
        "token_positive_similarity": similarities[positive].mean(),
        "token_negative_similarity": negative_similarity,
        "token_similarity_margin": (
            similarities[positive].mean() - negative_similarity
        ),
    }


def _artist_direction_loss(
    prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
    centered_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer the correct artist residual without training wrong refs to be harmful."""

    correct_delta = prediction - base_prediction
    wrong_delta = wrong_prediction.detach() - base_prediction
    desired = target - base_prediction
    correct_cosine = F.cosine_similarity(
        correct_delta.flatten(1), desired.flatten(1), dim=1, eps=1e-8
    )
    wrong_cosine = F.cosine_similarity(
        wrong_delta.flatten(1), desired.flatten(1), dim=1, eps=1e-8
    )
    centered_cosine = F.cosine_similarity(
        (correct_delta - wrong_delta).flatten(1),
        desired.flatten(1),
        dim=1,
        eps=1e-8,
    )
    ranking = F.relu(margin - correct_cosine + wrong_cosine).mean()
    centered = (1.0 - centered_cosine).mean()
    loss = ranking + float(centered_weight) * centered
    return loss, {
        "artist_correct_direction_cosine": correct_cosine.mean(),
        "artist_wrong_direction_cosine": wrong_cosine.mean(),
        "artist_centered_direction_cosine": centered_cosine.mean(),
        "artist_direction_ranking_loss": ranking,
    }


def _parameter_gradient_norm(
    loss: torch.Tensor, parameters: list[nn.Parameter]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squared:
        return loss.new_zeros((), dtype=torch.float32)
    return torch.stack(squared).sum().sqrt()


def _flow_metrics(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dimensions = tuple(range(1, prediction.ndim))
    flow_per_sample = (prediction - target).square().mean(dim=dimensions)
    base_per_sample = (base_prediction - target).square().mean(dim=dimensions)
    delta = prediction - base_prediction
    desired = target - base_prediction
    delta_rms = delta.square().mean(dim=dimensions).sqrt()
    desired_rms = desired.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    direction = F.cosine_similarity(
        delta.flatten(1), desired.flatten(1), dim=1, eps=1e-8
    )
    desired_unit = desired / desired_rms.reshape(-1, *([1] * (desired.ndim - 1)))
    projection = (delta * desired_unit).mean(dim=dimensions)
    orthogonal = (
        delta - desired_unit * projection.reshape(
            -1, *([1] * (desired.ndim - 1))
        )
    ).square().mean(dim=dimensions).sqrt()
    return {
        "flow_loss": flow_per_sample.mean(),
        "base_flow_loss": base_per_sample.mean(),
        "paired_flow_improvement": (
            (base_per_sample - flow_per_sample) / base_per_sample.clamp_min(1e-8)
        ).mean(),
        "paired_positive_fraction": (flow_per_sample < base_per_sample).float().mean(),
        "style_output_ratio": (
            delta_rms
            / base_prediction.square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
        ).mean(),
        "style_flow_direction_cosine": direction.mean(),
        "style_flow_delta_to_desired_ratio": (delta_rms / desired_rms).mean(),
        "style_flow_orthogonal_to_desired_ratio": (
            orthogonal / desired_rms
        ).mean(),
    }


def _forward_tokenizer_flow(
    anima: nn.Module,
    tokenizer: AnimaStyleTokenizer,
    batch: dict[str, Any],
    device: str,
    training_cfg: dict[str, Any],
    *,
    generator: torch.Generator,
    reference_mode: str,
    measure_base: bool,
    token_contrastive_weight: float = 0.0,
    artist_direction_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    latents = batch["latents"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    conditioning = batch["conditioning"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    lengths = batch["conditioning_lengths"].to(device, non_blocking=True)
    references, reference_mask = _reference_tokens(
        batch, device, mode=reference_mode
    )
    noise = torch.randn(
        latents.shape, device=device, dtype=latents.dtype, generator=generator
    )
    timesteps = _sample_flow_timesteps(
        latents.shape[0], device, training_cfg, generator
    )
    sigma = timesteps[:, None, None, None].to(latents.dtype)
    noisy = (1 - sigma) * latents + sigma * noise
    padding_mask = torch.zeros(
        latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
        device=device, dtype=latents.dtype,
    )
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        style_tokens = tokenizer(references, reference_mask)
        styled_conditioning = insert_style_tokens(
            conditioning, lengths, style_tokens
        )
        prediction = anima(
            noisy.unsqueeze(2), timesteps.to(latents.dtype),
            context=styled_conditioning, padding_mask=padding_mask,
            target_input_ids=None,
        ).squeeze(2).float()
    target = (noise - latents).float()
    flow_loss = F.mse_loss(prediction, target)
    total_loss = flow_loss
    token_loss = None
    direction_loss = None
    metrics = {
        "total_loss": float(total_loss.detach()),
        "flow_loss": float(flow_loss.detach()),
        "token_rms": float(style_tokens.detach().float().square().mean().sqrt()),
        "token_scale": float(tokenizer.log_output_rms.detach().exp()),
        "references": float(reference_mask.sum(dim=1).float().mean()),
        "timestep_mean": float(timesteps.detach().float().mean()),
    }

    if token_contrastive_weight > 0:
        eligible, first_mask, second_mask = _split_reference_views(reference_mask)
        if bool(eligible.any()) and int(eligible.sum()) >= 2:
            eligible_references = references[eligible]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                first_tokens = tokenizer(eligible_references, first_mask)
                second_tokens = tokenizer(eligible_references, second_mask)
            episodes = batch.get("episodes") or []
            eligible_indices = eligible.nonzero(as_tuple=False).flatten().tolist()
            style_ids = (
                [str(episodes[index].style_id) for index in eligible_indices]
                if episodes
                else [str(index) for index in eligible_indices]
            )
            token_loss, token_metrics = _style_token_contrastive_loss(
                first_tokens,
                second_tokens,
                style_ids,
                float(training_cfg.get("token_contrastive_temperature", 0.10)),
            )
            total_loss = total_loss + token_contrastive_weight * token_loss
            metrics.update({
                "token_contrastive_loss": float(token_loss.detach()),
                "token_contrastive_weight": float(token_contrastive_weight),
                "token_contrastive_eligible": float(eligible.sum()),
                **{
                    key: float(value.detach())
                    for key, value in token_metrics.items()
                },
            })

    base_prediction = None
    if measure_base or artist_direction_weight > 0:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            base_prediction = anima(
                noisy.unsqueeze(2), timesteps.to(latents.dtype),
                context=conditioning, padding_mask=padding_mask,
                target_input_ids=None,
            ).squeeze(2).float()
    if artist_direction_weight > 0:
        direction_batch = min(
            int(training_cfg.get("artist_direction_batch_size", 2)),
            latents.shape[0],
        )
        wrong_references, wrong_mask = _reference_tokens(
            batch, device, mode="wrong_artist"
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            wrong_tokens = tokenizer(
                wrong_references[:direction_batch], wrong_mask[:direction_batch]
            )
            wrong_conditioning = insert_style_tokens(
                conditioning[:direction_batch],
                lengths[:direction_batch],
                wrong_tokens,
            )
            wrong_prediction = anima(
                noisy[:direction_batch].unsqueeze(2),
                timesteps[:direction_batch].to(latents.dtype),
                context=wrong_conditioning,
                padding_mask=padding_mask[:direction_batch],
                target_input_ids=None,
            ).squeeze(2).float()
        direction_loss, direction_metrics = _artist_direction_loss(
            prediction[:direction_batch],
            wrong_prediction,
            base_prediction[:direction_batch],
            target[:direction_batch],
            margin=float(training_cfg.get("artist_direction_margin", 0.02)),
            centered_weight=float(
                training_cfg.get("artist_centered_direction_weight", 0.25)
            ),
        )
        total_loss = total_loss + artist_direction_weight * direction_loss
        metrics.update({
            "artist_direction_loss": float(direction_loss.detach()),
            "artist_direction_weight": float(artist_direction_weight),
            **{
                key: float(value.detach())
                for key, value in direction_metrics.items()
            },
        })
    metrics["total_loss"] = float(total_loss.detach())
    if (
        torch.is_grad_enabled()
        and bool(training_cfg.get("diagnose_auxiliary_gradients", False))
    ):
        trainable = [parameter for parameter in tokenizer.parameters() if parameter.requires_grad]
        metrics["flow_gradient_norm"] = float(
            _parameter_gradient_norm(flow_loss, trainable)
        )
        if token_loss is not None:
            metrics["weighted_token_gradient_norm"] = float(
                _parameter_gradient_norm(
                    token_contrastive_weight * token_loss, trainable
                )
            )
        if direction_loss is not None:
            metrics["weighted_artist_direction_gradient_norm"] = float(
                _parameter_gradient_norm(
                    artist_direction_weight * direction_loss, trainable
                )
            )
    if measure_base:
        assert base_prediction is not None
        metrics.update({
            key: float(value.detach())
            for key, value in _flow_metrics(
                prediction.detach(), base_prediction, target
            ).items()
        })
    return total_loss, metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        if values:
            result[key] = sum(values) / len(values)
    paired = [float(row["paired_flow_improvement"]) for row in rows]
    if len(paired) > 1:
        mean = sum(paired) / len(paired)
        variance = sum((value - mean) ** 2 for value in paired) / (len(paired) - 1)
        result["paired_flow_improvement_ci95"] = 1.96 * math.sqrt(
            variance / len(paired)
        )
    else:
        result["paired_flow_improvement_ci95"] = 0.0
    result["batches"] = float(len(rows))
    return result


def _ramped_auxiliary_weight(
    step: int, *, maximum: float, start_step: int, ramp_steps: int
) -> float:
    if maximum <= 0 or step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return float(maximum)
    progress = min(1.0, (step - start_step + 1) / ramp_steps)
    return float(maximum) * progress


@torch.no_grad()
def _evaluate(
    anima: nn.Module,
    tokenizer: AnimaStyleTokenizer,
    loader: ProductionStyleLoader,
    device: str,
    training_cfg: dict[str, Any],
    *,
    batches: int,
    seed: int,
    reference_mode: str,
) -> dict[str, float]:
    tokenizer.eval()
    rows = []
    started = time.perf_counter()
    for index in range(batches):
        batch = loader.load_step(index)
        _, metrics = _forward_tokenizer_flow(
            anima, tokenizer, batch, device, training_cfg,
            generator=torch.Generator(device=device).manual_seed(seed + index * 97),
            reference_mode=reference_mode, measure_base=True,
        )
        rows.append(metrics)
    result = _mean_metrics(rows)
    result["elapsed_s"] = time.perf_counter() - started
    tokenizer.train()
    return result


def _tokenizer_loader_config(
    config: dict[str, Any], cfg: dict[str, Any], *, split: str
) -> dict[str, Any]:
    base = dict(config["style_transfer"]["loader"])
    base.update(dict(cfg.get("loader", {})))
    base["split"] = split
    base["resampler_token_cache"] = str(
        base.get(
            "resampler_token_cache",
            config["style_transfer"]["loader"]["resampler_token_cache"],
        )
    )
    if split != "train":
        base["self_reference_target_images_per_style"] = 0
        base["reference_curriculum"] = {}
    return base


def _assert_resampler_cache_identity(
    destination: Path, loader_cfg: dict[str, Any], resampler_checkpoint: str
) -> dict[str, Any]:
    summary_path = (
        destination / str(loader_cfg["resampler_token_cache"]) / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded = str(summary.get("resampler_checkpoint", ""))
    if recorded and recorded != str(resampler_checkpoint):
        raise RuntimeError(
            "StyleTokenizer token cache was produced by a different Resampler: "
            f"{recorded!r} != {resampler_checkpoint!r}"
        )
    if tuple((int(summary.get("slots", 0)), int(summary.get("style_dim", 0)))) != (
        128, 1024,
    ):
        raise RuntimeError("StyleTokenizer requires the current 128x1024 Resampler cache")
    return summary


def _take_request(
    loader: ProductionStyleLoader, episode_index: int, mode: str, device: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, int, list[tuple[str, int]]]:
    batch = loader.load_step(episode_index)
    episode = batch["episodes"][0]
    references, mask = _reference_tokens(batch, device, mode=mode)
    references = references[:1]
    mask = mask[:1]
    sources = [("target", int(episode.target_id))]
    if mode == "self":
        sources.append(("exact target", int(episode.target_id)))
    else:
        sources.extend(
            (f"ref {index + 1}", int(image_id))
            for index, image_id in enumerate(episode.reference_ids[:4])
        )
    sheet_batch = {"episodes": [episode]}
    return (
        sheet_batch,
        batch["conditioning"][:1],
        references,
        mask,
        int(batch["conditioning_lengths"][0]),
        sources,
    )


@torch.no_grad()
def _sample_tokenizer(
    anima: nn.Module,
    tokenizer: AnimaStyleTokenizer,
    requests: list[tuple[str, ProductionStyleLoader, int, str]],
    config: dict[str, Any],
    destination: Path,
    output: Path,
    device: str,
    step: int,
    vae: nn.Module | None,
) -> tuple[list[Path], nn.Module]:
    cfg = config["style_tokenizer"]
    sample_cfg = dict(cfg.get("sampling", {}))
    tokenizer.eval()
    loaded = [
        (label, loader, mode, _take_request(loader, episode, mode, device))
        for label, loader, episode, mode in requests
    ]
    positive_text = torch.cat(
        [item[3][1] for item in loaded]
    ).to(device, dtype=torch.bfloat16)
    lengths = torch.tensor(
        [item[3][4] for item in loaded], device=device, dtype=torch.long
    )
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        style_tokens = torch.cat([
            tokenizer(item[3][2], item[3][3]) for item in loaded
        ])
    full_text = insert_style_tokens(positive_text, lengths, style_tokens)
    null_raw = load_file(
        loaded[0][1].text_root / "null_conditioning.safetensors", device="cpu"
    )["empty_prompt"]
    if null_raw.ndim == 3:
        null_raw = null_raw[0]
    null_text = _pad_text_conditions(
        [null_raw] * len(requests), loaded[0][1].text_conditioning_length
    ).to(device, dtype=torch.bfloat16)
    height = int(sample_cfg.get("height", 768))
    width = int(sample_cfg.get("width", 768))
    latent_h, latent_w = height // 8, width // 8
    seed = int(sample_cfg.get("seed", 20260815))
    noises = [
        torch.randn(
            1, 16, 1, latent_h, latent_w,
            generator=torch.Generator(device="cpu").manual_seed(seed + index * 10007),
            dtype=torch.float32,
        )
        for index in range(len(requests))
    ]
    initial_noise = torch.cat(noises).to(device=device, dtype=torch.bfloat16)
    sample_steps = int(sample_cfg.get("steps", 30))
    sigmas = torch.linspace(
        1.0, 0.0, sample_steps + 1, device=device, dtype=torch.bfloat16
    )
    shift = float(sample_cfg.get("flow_shift", 3.0))
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    padding_mask = torch.zeros(
        len(requests), 1, latent_h, latent_w,
        device=device, dtype=torch.bfloat16,
    )
    text_cfg = float(sample_cfg.get("text_cfg", 4.0))
    guidance_mode = str(sample_cfg.get("guidance_mode", "shared"))
    if guidance_mode not in {"shared", "separate"}:
        raise ValueError("StyleTokenizer guidance_mode must be 'shared' or 'separate'")
    style_cfg = float(sample_cfg.get("style_cfg", text_cfg))

    def predict(x: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor):
        return anima(
            x, timestep.expand(len(requests)), context=context,
            padding_mask=padding_mask, target_input_ids=None,
        ).float()

    def denoise(with_style: bool) -> torch.Tensor:
        x = initial_noise.clone()
        for index in range(sample_steps):
            timestep = sigmas[index].to(torch.bfloat16)
            base = predict(x, null_text, timestep)
            if with_style and guidance_mode == "shared":
                full = predict(x, full_text, timestep)
                velocity = base + text_cfg * (full - base)
            else:
                text_only = predict(x, positive_text, timestep)
                velocity = base + text_cfg * (text_only - base)
                if with_style:
                    full = predict(x, full_text, timestep)
                    velocity = velocity + style_cfg * (full - text_only)
            x = (
                x.float()
                + velocity * (sigmas[index + 1] - sigmas[index]).float()
            ).to(torch.bfloat16)
        return x

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        base_x = denoise(False)
        styled_x = denoise(True)
    if vae is None:
        vae = _load_sampling_vae(config, destination)
    vae.to(device=device, dtype=torch.bfloat16)
    decoded = vae.decode_to_pixels(torch.cat((base_x, styled_x), dim=0)).float()

    def to_image(value: torch.Tensor) -> Image.Image:
        if value.ndim == 4:
            value = value[:, 0]
        pixels = (
            (value.clamp(-1, 1) + 1) * 127.5
        ).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)

    sample_dir = output / "samples" / f"step-{step:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for index, (label, loader, mode, item) in enumerate(loaded):
        base_image = to_image(decoded[index])
        styled_image = to_image(decoded[len(requests) + index])
        raw = sample_dir / f"{label}-{mode}.png"
        sheet = sample_dir / f"{label}-{mode}-sheet.png"
        styled_image.save(raw)
        _make_sample_sheet(
            styled_image, loader, item[0], base_generated=base_image,
            generated_label=(
                f"StyleTokenizer shared CFG {text_cfg:g} ({mode})"
                if guidance_mode == "shared"
                else (
                    f"StyleTokenizer text CFG {text_cfg:g} / "
                    f"style CFG {style_cfg:g} ({mode})"
                )
            ),
            sources=item[5],
        ).save(sheet)
        sheets.append(sheet)
    vae.to("cpu")
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    tokenizer.train()
    return sheets, vae


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    tokenizer: AnimaStyleTokenizer,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    resampler_cache: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "tokenizer": tokenizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "resampler_cache": resampler_cache,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def train_style_tokenizer(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_section: str = "style_tokenizer",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[config_section])
    training_cfg = dict(cfg["training"])
    reference_mode = str(training_cfg.get("reference_mode", "self"))
    if reference_mode not in {"self", "heldout"}:
        raise ValueError(
            "StyleTokenizer training.reference_mode must be 'self' or 'heldout'"
        )
    steps = int(steps_override or training_cfg["steps"])
    device = str(training_cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260815))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(training_cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    accumulation = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    train_loader_cfg = _tokenizer_loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    train_loader_cfg["gradient_accumulation_steps"] = accumulation
    if reference_mode == "self":
        train_loader_cfg["reference_curriculum"] = {
            "gate_only_steps": 0,
            "self_reference_steps": steps + 1,
        }
    else:
        # ProductionStyleLoader always excludes the target from its ordinary
        # same-style reference pool. Disabling the old adapter curriculum here
        # therefore gives a genuinely target-excluded multi-reference batch.
        train_loader_cfg["reference_curriculum"] = {}
        train_loader_cfg["self_reference_target_images_per_style"] = 0
    validation_loader_cfg = _tokenizer_loader_config(
        config, cfg, split=str(cfg.get("validation_split", "validation"))
    )
    train_loader = ProductionStyleLoader(destination, train_loader_cfg)
    validation_loader = ProductionStyleLoader(destination, validation_loader_cfg)
    resampler_checkpoint = str(config["style_transfer"]["resampler"]["checkpoint"])
    cache_summary = _assert_resampler_cache_identity(
        destination, train_loader_cfg, resampler_checkpoint
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training_cfg.get("fuse_attention_projections", True)
        ),
    )
    tokenizer = AnimaStyleTokenizer(**dict(cfg["model"])).to(device)
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    state_path = output / "training_state.pt"
    initial_checkpoint: Path | None = None
    initial_value = training_cfg.get("initial_checkpoint")
    if initial_value:
        initial_checkpoint = Path(str(initial_value))
        if not initial_checkpoint.is_absolute():
            initial_checkpoint = destination / initial_checkpoint
        if not (bool(training_cfg.get("resume", True)) and state_path.exists()):
            if not initial_checkpoint.exists():
                raise FileNotFoundError(
                    f"StyleTokenizer initial checkpoint does not exist: {initial_checkpoint}"
                )
            initial_state = torch.load(
                initial_checkpoint, map_location="cpu", weights_only=False
            )
            initial_cache = dict(initial_state.get("resampler_cache", {}))
            expected_sha = str(cache_summary.get("resampler_checkpoint_sha256", ""))
            initial_sha = str(initial_cache.get("resampler_checkpoint_sha256", ""))
            if expected_sha and initial_sha and expected_sha != initial_sha:
                raise RuntimeError(
                    "StyleTokenizer initial checkpoint uses a different Resampler cache"
                )
            tokenizer.load_state_dict(initial_state["tokenizer"], strict=True)
    trainable = sum(parameter.numel() for parameter in tokenizer.parameters())
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
        eps=float(training_cfg.get("adam_eps", 1e-8)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        fused=bool(training_cfg.get("fused_adamw", True) and device.startswith("cuda")),
    )
    start_step = 0
    if bool(training_cfg.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(state["tokenizer"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    warmup_steps = int(training_cfg.get("warmup_steps", 200))
    minimum_ratio = float(training_cfg.get("minimum_lr_ratio", 0.1))
    log_every = int(training_cfg.get("log_every", 10))
    validation_every = int(training_cfg.get("validation_every", 250))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 250))
    sample_every = int(training_cfg.get("sample_every", 500))
    validation_batches = int(training_cfg.get("validation_batches", 8))
    train_validation_batches = int(training_cfg.get("train_validation_batches", 8))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    prefetch_workers = int(training_cfg.get("prefetch_workers", 2))
    prefetch_batches = int(training_cfg.get("prefetch_batches", 4))
    token_contrastive_maximum = float(
        training_cfg.get("token_contrastive_weight", 0.0)
    )
    token_contrastive_start = int(
        training_cfg.get("token_contrastive_start_step", 1)
    )
    token_contrastive_ramp = int(
        training_cfg.get("token_contrastive_ramp_steps", 1000)
    )
    artist_direction_maximum = float(
        training_cfg.get("artist_direction_weight", 0.0)
    )
    artist_direction_start = int(
        training_cfg.get("artist_direction_start_step", 1500)
    )
    artist_direction_ramp = int(
        training_cfg.get("artist_direction_ramp_steps", 1000)
    )
    artist_direction_every = max(
        1, int(training_cfg.get("artist_direction_every", 4))
    )

    wandb_run = None
    wandb_cfg = dict(training_cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "anima-style-tokenizer")),
            id=str(wandb_cfg.get("id", "anima-style-tokenizer-v1")),
            resume="allow" if start_step else "never",
            config={
                "style_tokenizer": cfg,
                "trainable_parameters": trainable,
                "train_reference_mode": reference_mode,
                "initial_checkpoint": (
                    str(initial_checkpoint) if initial_checkpoint is not None else None
                ),
            },
        )

    total_microsteps = max(0, steps - start_step) * accumulation
    micro_start = start_step * accumulation
    prefetched = train_loader.prefetch(
        micro_start, total_microsteps,
        workers=prefetch_workers, depth=prefetch_batches,
    )
    history = []
    history_path = output / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    vae = None
    sample_requests = [
        ("train-0", train_loader, 0, "self"),
        ("train-1", train_loader, 13, "heldout"),
        ("validation-0", validation_loader, 0, "self"),
        ("validation-1", validation_loader, 13, "heldout"),
    ]
    try:
        # A checkpoint is committed before qualitative sampling. If sampling
        # was interrupted, resume the missing panel before advancing so the
        # monitored sequence retains every configured interval.
        resumed_sample_dir = output / "samples" / f"step-{start_step:07d}"
        if (
            start_step > 0
            and sample_every > 0
            and start_step % sample_every == 0
            and not resumed_sample_dir.exists()
        ):
            sheets, vae = _sample_tokenizer(
                anima, tokenizer, sample_requests, config, destination,
                output, device, start_step, vae,
            )
            if wandb_run is not None:
                import wandb
                wandb_run.log({
                    "samples/panel": [wandb.Image(str(path)) for path in sheets]
                }, step=start_step)
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            multiplier = _learning_rate_multiplier(
                step, steps, warmup_steps, minimum_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = base_lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            micro_metrics = []
            token_contrastive_weight = _ramped_auxiliary_weight(
                step,
                maximum=token_contrastive_maximum,
                start_step=token_contrastive_start,
                ramp_steps=token_contrastive_ramp,
            )
            artist_direction_weight = _ramped_auxiliary_weight(
                step,
                maximum=artist_direction_maximum,
                start_step=artist_direction_start,
                ramp_steps=artist_direction_ramp,
            )
            if step % artist_direction_every != 0:
                artist_direction_weight = 0.0
            for micro in range(accumulation):
                batch = next(prefetched)
                should_measure = (
                    step % log_every == 0 and micro == accumulation - 1
                )
                generator = torch.Generator(device=device).manual_seed(
                    seed + step * 100_003 + micro
                )
                loss, metrics = _forward_tokenizer_flow(
                    anima, tokenizer, batch, device, training_cfg,
                    generator=generator, reference_mode=reference_mode,
                    measure_base=should_measure,
                    token_contrastive_weight=token_contrastive_weight,
                    artist_direction_weight=artist_direction_weight,
                )
                (loss / accumulation).backward()
                micro_metrics.append(metrics)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                tokenizer.parameters(), max_grad_norm
            )
            optimizer.step()
            step_s = time.perf_counter() - step_started
            train_metrics = {
                "loss": sum(row["total_loss"] for row in micro_metrics) / len(micro_metrics),
                "flow_loss": sum(row["flow_loss"] for row in micro_metrics) / len(micro_metrics),
                "token_rms": sum(row["token_rms"] for row in micro_metrics) / len(micro_metrics),
                "token_scale": sum(row["token_scale"] for row in micro_metrics) / len(micro_metrics),
                "grad_norm": float(grad_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_s": step_s,
                "images_per_s": train_loader.batch_size * accumulation / max(step_s, 1e-6),
                "token_contrastive_weight": token_contrastive_weight,
                "artist_direction_weight": artist_direction_weight,
            }
            for key in (
                "token_contrastive_loss", "token_contrastive_eligible",
                "token_positive_similarity", "token_negative_similarity",
                "token_similarity_margin", "artist_direction_loss",
                "artist_correct_direction_cosine",
                "artist_wrong_direction_cosine",
                "artist_centered_direction_cosine",
                "artist_direction_ranking_loss",
                "flow_gradient_norm", "weighted_token_gradient_norm",
                "weighted_artist_direction_gradient_norm",
            ):
                values = [row[key] for row in micro_metrics if key in row]
                if values:
                    train_metrics[key] = sum(values) / len(values)
            measured = micro_metrics[-1]
            for key in (
                "base_flow_loss", "paired_flow_improvement",
                "paired_positive_fraction", "style_output_ratio",
                "style_flow_direction_cosine",
                "style_flow_delta_to_desired_ratio",
                "style_flow_orthogonal_to_desired_ratio",
            ):
                if key in measured:
                    train_metrics[key] = measured[key]
            if step % log_every == 0 or step == 1:
                print(
                    f"style-tokenizer step={step}/{steps} "
                    f"loss={train_metrics['loss']:.5f} "
                    f"grad={train_metrics['grad_norm']:.4f} "
                    f"token_rms={train_metrics['token_rms']:.4f} "
                    f"step_s={step_s:.3f} img_s={train_metrics['images_per_s']:.2f}",
                    flush=True,
                )
                if bool(training_cfg.get("diagnose_auxiliary_gradients", False)):
                    print(
                        "style-tokenizer auxiliary-gradients "
                        f"step={step} "
                        f"flow={train_metrics.get('flow_gradient_norm', float('nan')):.6f} "
                        f"token={train_metrics.get('weighted_token_gradient_norm', float('nan')):.6f} "
                        f"direction={train_metrics.get('weighted_artist_direction_gradient_norm', float('nan')):.6f}",
                        flush=True,
                    )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_metrics.items()},
                    step=step,
                )
            if step % validation_every == 0 or step == steps:
                train_self = _evaluate(
                    anima, tokenizer, train_loader, device, training_cfg,
                    batches=train_validation_batches, seed=seed ^ 0xA11CE,
                    reference_mode="self",
                )
                validation_self = _evaluate(
                    anima, tokenizer, validation_loader, device, training_cfg,
                    batches=validation_batches, seed=seed ^ 0xBEEF,
                    reference_mode="self",
                )
                validation_heldout = _evaluate(
                    anima, tokenizer, validation_loader, device, training_cfg,
                    batches=validation_batches, seed=seed ^ 0xC0FFEE,
                    reference_mode="heldout",
                )
                validation_wrong = _evaluate(
                    anima, tokenizer, validation_loader, device, training_cfg,
                    batches=validation_batches, seed=seed ^ 0xC0FFEE,
                    reference_mode="wrong_artist",
                )
                reference_advantage = (
                    validation_heldout["paired_flow_improvement"]
                    - validation_wrong["paired_flow_improvement"]
                )
                row = {
                    "step": step, "train": train_metrics,
                    "train_self": train_self,
                    "validation_self": validation_self,
                    "validation_heldout": validation_heldout,
                    "validation_wrong_artist": validation_wrong,
                    "correct_vs_wrong_paired_advantage": reference_advantage,
                }
                history.append(row)
                write_json(history_path, history)
                print(f"style-tokenizer validation step={step} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({
                        **{f"train_self/{key}": value for key, value in train_self.items()},
                        **{f"validation_self/{key}": value for key, value in validation_self.items()},
                        **{f"validation_heldout/{key}": value for key, value in validation_heldout.items()},
                        **{
                            f"validation_wrong_artist/{key}": value
                            for key, value in validation_wrong.items()
                        },
                        "validation/correct_vs_wrong_paired_advantage": reference_advantage,
                    }, step=step)
            if step % checkpoint_every == 0 or step == steps:
                checkpoint = checkpoint_dir / f"step-{step:07d}.pt"
                _save_checkpoint(
                    checkpoint, step=step, tokenizer=tokenizer,
                    optimizer=optimizer, cfg=cfg, resampler_cache=cache_summary,
                )
                _save_checkpoint(
                    state_path, step=step, tokenizer=tokenizer,
                    optimizer=optimizer, cfg=cfg, resampler_cache=cache_summary,
                )
            if sample_every > 0 and (step % sample_every == 0 or step == steps):
                sheets, vae = _sample_tokenizer(
                    anima, tokenizer, sample_requests, config, destination,
                    output, device, step, vae,
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "samples/panel": [wandb.Image(str(path)) for path in sheets]
                    }, step=step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    result = {
        "steps": steps,
        "start_step": start_step,
        "trainable_parameters": trainable,
        "output_tokens": tokenizer.output_tokens,
        "train_reference_mode": reference_mode,
        "initial_checkpoint": (
            str(initial_checkpoint) if initial_checkpoint is not None else None
        ),
        "style_dropout": 0.0,
        "resampler_cache": cache_summary,
        "elapsed_s": time.perf_counter() - started,
        "final_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", result)
    return result


def train_style_tokenizer_generalization(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train the full-data, target-excluded multi-reference tokenizer."""
    return train_style_tokenizer(
        config, destination, config_section="style_tokenizer_generalization"
    )


def train_style_tokenizer_lr_probe(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Resume the isolated high-LR exact-self branch."""
    return train_style_tokenizer(
        config, destination, config_section="style_tokenizer_lr_probe"
    )


def train_style_tokenizer_lr10x_scratch(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train the fair 10x-LR ablation from the original random seed."""
    return train_style_tokenizer(
        config, destination, config_section="style_tokenizer_lr10x_scratch"
    )


@torch.no_grad()
def select_style_tokenizer_checkpoint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Re-evaluate the strongest checkpoints and materialize one selection."""
    selection_cfg = dict(config["style_tokenizer_selection"])
    source_section = str(selection_cfg.get("source_config_section", "style_tokenizer"))
    source_cfg = copy.deepcopy(config[source_section])
    output = destination / str(source_cfg["output_directory"])
    history_path = output / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    minimum_step = int(selection_cfg.get("minimum_step", 500))
    maximum_step = int(selection_cfg.get("maximum_step", 1 << 30))
    eligible = [
        row for row in history
        if minimum_step <= int(row["step"]) <= maximum_step
    ]
    if not eligible:
        raise RuntimeError("No StyleTokenizer validation checkpoints are eligible")

    def provisional_score(row: dict[str, Any]) -> float:
        heldout = row["validation_heldout"]
        self_metrics = row["validation_self"]
        return float(heldout["paired_flow_improvement"]) + 0.5 * float(
            self_metrics["paired_flow_improvement"]
        )

    top_k = max(1, int(selection_cfg.get("top_k", 5)))
    candidates = sorted(eligible, key=provisional_score, reverse=True)[:top_k]
    latest = max(eligible, key=lambda row: int(row["step"]))
    by_step = {int(row["step"]): row for row in candidates}
    by_step[int(latest["step"])] = latest

    training_cfg = dict(source_cfg["training"])
    device = str(selection_cfg.get("device", training_cfg.get("device", "cuda")))
    validation_loader_cfg = _tokenizer_loader_config(
        config, source_cfg, split=str(source_cfg.get("validation_split", "validation"))
    )
    validation_loader = ProductionStyleLoader(destination, validation_loader_cfg)
    resampler_checkpoint = str(config["style_transfer"]["resampler"]["checkpoint"])
    cache_summary = _assert_resampler_cache_identity(
        destination, validation_loader_cfg, resampler_checkpoint
    )
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training_cfg.get("low_precision_rmsnorm", True)),
        fuse_attention_projections=bool(
            training_cfg.get("fuse_attention_projections", True)
        ),
    )
    tokenizer = AnimaStyleTokenizer(**dict(source_cfg["model"])).to(device).eval()
    batches = int(selection_cfg.get("validation_batches", 32))
    seed = int(selection_cfg.get("seed", 20260817))
    results = []
    for step in sorted(by_step):
        checkpoint = output / "checkpoints" / f"step-{step:07d}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(state["tokenizer"], strict=True)
        self_metrics = _evaluate(
            anima, tokenizer, validation_loader, device, training_cfg,
            batches=batches, seed=seed ^ 0xBEEF, reference_mode="self",
        )
        heldout = _evaluate(
            anima, tokenizer, validation_loader, device, training_cfg,
            batches=batches, seed=seed ^ 0xC0FFEE, reference_mode="heldout",
        )
        wrong = _evaluate(
            anima, tokenizer, validation_loader, device, training_cfg,
            batches=batches, seed=seed ^ 0xC0FFEE, reference_mode="wrong_artist",
        )
        advantage = (
            heldout["paired_flow_improvement"]
            - wrong["paired_flow_improvement"]
        )
        self_lcb = (
            self_metrics["paired_flow_improvement"]
            - self_metrics["paired_flow_improvement_ci95"]
        )
        heldout_lcb = (
            heldout["paired_flow_improvement"]
            - heldout["paired_flow_improvement_ci95"]
        )
        score = heldout_lcb + 0.5 * self_lcb + 0.5 * advantage
        results.append({
            "step": step,
            "checkpoint": str(checkpoint),
            "score": score,
            "correct_vs_wrong_paired_advantage": advantage,
            "validation_self": self_metrics,
            "validation_heldout": heldout,
            "validation_wrong_artist": wrong,
        })
        print(
            f"style-tokenizer selection step={step} score={score:.6f} "
            f"self={self_metrics['paired_flow_improvement']:.6f} "
            f"heldout={heldout['paired_flow_improvement']:.6f} "
            f"advantage={advantage:.6f}",
            flush=True,
        )
    selected = max(results, key=lambda row: float(row["score"]))
    selected_path = output / "checkpoints" / "selected.pt"
    temporary = selected_path.with_suffix(".pt.tmp")
    shutil.copyfile(Path(str(selected["checkpoint"])), temporary)
    temporary.replace(selected_path)
    summary = {
        "source_config_section": source_section,
        "selection_rule": (
            "heldout_lcb95 + 0.5*self_lcb95 + "
            "0.5*(heldout_improvement-wrong_artist_improvement)"
        ),
        "validation_batches": batches,
        "candidate_count": len(results),
        "candidates": results,
        "selected_step": int(selected["step"]),
        "selected_checkpoint": str(selected_path.resolve()),
        "resampler_cache": cache_summary,
    }
    write_json(output / "selection.json", summary)
    return summary


def smoke_test_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    cfg = smoke["style_tokenizer"]
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["validation_every"] = 2
    cfg["training"]["checkpoint_every"] = 2
    cfg["training"]["sample_every"] = 0
    cfg["training"]["wandb"] = {"enabled": False}
    return train_style_tokenizer(smoke, destination, steps_override=2)


def smoke_test_style_tokenizer_generalization(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Exercise both multi-view and wrong-artist auxiliaries on real caches."""

    smoke = copy.deepcopy(config)
    cfg = smoke["style_tokenizer_generalization"]
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    training = cfg["training"]
    training["resume"] = False
    training["validation_every"] = 2
    training["validation_batches"] = 1
    training["train_validation_batches"] = 1
    training["checkpoint_every"] = 2
    training["sample_every"] = 0
    training["token_contrastive_start_step"] = 1
    training["token_contrastive_ramp_steps"] = 0
    training["artist_direction_start_step"] = 1
    training["artist_direction_ramp_steps"] = 0
    training["artist_direction_every"] = 1
    training["diagnose_auxiliary_gradients"] = True
    training["wandb"] = {"enabled": False}
    return train_style_tokenizer(
        smoke,
        destination,
        steps_override=2,
        config_section="style_tokenizer_generalization",
    )
