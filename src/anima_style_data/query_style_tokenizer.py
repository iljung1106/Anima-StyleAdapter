from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .io import write_json
from .style_tokenizer import (
    _assert_resampler_cache_identity,
    _flow_metrics,
    _mean_metrics,
    _reference_tokens,
    _tokenizer_loader_config,
)
from .style_transfer import (
    ProductionStyleLoader,
    _create_and_attach_style_adapter,
    _learning_rate_multiplier,
    _optimize_frozen_anima,
    _replace_reference_with_target,
    _resolve_anima_model,
    _sample_flow_timesteps,
    _self_reference_curriculum_state,
)


@dataclass
class QueryStyleTokenizerOutput:
    tokens: torch.Tensor
    per_reference_tokens: torch.Tensor
    reconstruction: torch.Tensor | None
    reconstruction_target: torch.Tensor | None


class QueryStyleTokenizerV2(nn.Module):
    """Preserve spatial Resampler information in independent style slots.

    Each reference is reduced from 128 frozen Resampler tokens to 32 tokens by
    learned queries, never to a single descriptor.  References are then pooled
    independently for each slot and a small cross-slot transformer produces
    the final native-width Anima context.
    """

    def __init__(
        self,
        *,
        source_dim: int = 1024,
        context_dim: int = 1024,
        source_tokens: int = 128,
        output_tokens: int = 32,
        heads: int = 16,
        per_reference_layers: int = 3,
        per_reference_ff_dim: int = 2048,
        cross_slot_layers: int = 2,
        cross_slot_ff_dim: int = 4096,
        reconstruction_layers: int = 1,
        reconstruction_ff_dim: int = 2048,
        reference_score_dim: int = 256,
        output_rms_init: float = 0.15,
    ) -> None:
        super().__init__()
        if source_dim != context_dim:
            raise ValueError("v2 currently requires equal source and context widths")
        if source_dim % heads:
            raise ValueError("model width must be divisible by the attention heads")
        if min(source_tokens, output_tokens, per_reference_layers, cross_slot_layers) <= 0:
            raise ValueError("token counts and encoder depths must be positive")
        if reconstruction_layers < 0:
            raise ValueError("reconstruction_layers cannot be negative")
        if output_rms_init <= 0:
            raise ValueError("output_rms_init must be positive")

        self.source_dim = int(source_dim)
        self.context_dim = int(context_dim)
        self.source_tokens = int(source_tokens)
        self.output_tokens = int(output_tokens)
        self.input_norm = nn.LayerNorm(source_dim)
        self.reference_queries = nn.Parameter(
            torch.empty(1, output_tokens, context_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=context_dim,
            nhead=heads,
            dim_feedforward=per_reference_ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.per_reference_encoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=per_reference_layers,
            norm=nn.LayerNorm(context_dim),
        )
        self.reference_score = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, reference_score_dim),
            nn.SiLU(),
            nn.Linear(reference_score_dim, 1, bias=False),
        )
        slot_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=heads,
            dim_feedforward=cross_slot_ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_slot_encoder = nn.TransformerEncoder(
            slot_layer,
            num_layers=cross_slot_layers,
            norm=nn.LayerNorm(context_dim),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(context_dim)
        self.log_output_rms = nn.Parameter(
            torch.tensor(math.log(float(output_rms_init)), dtype=torch.float32)
        )

        self.reconstruction_queries: nn.Parameter | None
        self.reconstruction_decoder: nn.Module | None
        if reconstruction_layers:
            self.reconstruction_queries = nn.Parameter(
                torch.empty(1, source_tokens, context_dim)
            )
            reconstruction_layer = nn.TransformerDecoderLayer(
                d_model=context_dim,
                nhead=heads,
                dim_feedforward=reconstruction_ff_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.reconstruction_decoder = nn.TransformerDecoder(
                reconstruction_layer,
                num_layers=reconstruction_layers,
                norm=nn.LayerNorm(context_dim),
            )
        else:
            self.register_parameter("reconstruction_queries", None)
            self.reconstruction_decoder = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.reference_queries, std=self.context_dim**-0.5)
        if self.reconstruction_queries is not None:
            nn.init.normal_(
                self.reconstruction_queries, std=self.context_dim**-0.5
            )

    def _encode_valid_references(self, references: torch.Tensor) -> torch.Tensor:
        memory = self.input_norm(references)
        queries = self.reference_queries.expand(references.shape[0], -1, -1)
        return self.per_reference_encoder(queries, memory)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> QueryStyleTokenizerOutput:
        if references.ndim != 4:
            raise ValueError("references must have shape [batch, references, tokens, dim]")
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference mask shape does not match references")
        if references.shape[2:] != (self.source_tokens, self.source_dim):
            raise ValueError(
                f"Expected reference tail {(self.source_tokens, self.source_dim)}, "
                f"got {tuple(references.shape[2:])}"
            )
        # ProductionStyleLoader guarantees one or more references. Keep the
        # friendly validation for CPU callers without introducing a CUDA
        # scalar synchronization in every training forward.
        if not reference_mask.is_cuda and not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one reference")

        valid_source = references[reference_mask]
        valid_encoded = self._encode_valid_references(valid_source)
        per_reference = valid_encoded.new_zeros(
            *references.shape[:2], self.output_tokens, self.context_dim
        )
        per_reference[reference_mask] = valid_encoded
        logits = self.reference_score(per_reference).squeeze(-1)
        logits = logits.masked_fill(
            ~reference_mask[..., None], torch.finfo(logits.dtype).min
        )
        weights = logits.softmax(dim=1)
        aggregated = torch.sum(weights[..., None] * per_reference, dim=1)
        tokens = self.output_norm(self.cross_slot_encoder(aggregated))
        tokens = tokens * self.log_output_rms.exp().to(tokens.dtype)

        reconstruction = None
        reconstruction_target = None
        if reconstruct:
            if self.reconstruction_decoder is None or self.reconstruction_queries is None:
                raise RuntimeError("The reconstruction decoder is disabled")
            # One randomly rotating reference per sample is sufficient for the
            # weak information-preservation objective and avoids decoding all
            # eight references during the expensive joint Anima run.
            first = reference_mask.to(torch.int64).argmax(dim=1)
            rows = torch.arange(references.shape[0], device=references.device)
            selected_encoded = per_reference[rows, first]
            selected_source = references[rows, first]
            queries = self.reconstruction_queries.expand(
                references.shape[0], -1, -1
            )
            reconstruction = self.reconstruction_decoder(
                queries, selected_encoded
            )
            reconstruction_target = self.input_norm(selected_source).detach()
        return QueryStyleTokenizerOutput(
            tokens=tokens,
            per_reference_tokens=per_reference,
            reconstruction=reconstruction,
            reconstruction_target=reconstruction_target,
        )


def _artist_contrastive_loss(
    targets: torch.Tensor,
    references: torch.Tensor,
    style_ids: list[str],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aligned-slot, multi-positive contrastive classification by artist."""
    if targets.shape != references.shape or targets.ndim != 3:
        raise ValueError("artist views must share [batch, slots, dim] shape")
    if len(style_ids) != targets.shape[0]:
        raise ValueError("style_ids must contain one value per row")
    left = F.normalize(targets.float(), dim=-1)
    right = F.normalize(references.float(), dim=-1)
    logits = torch.einsum("bsd,csd->bc", left, right)
    logits = logits / (left.shape[1] * float(temperature))
    positive = torch.tensor(
        [[a == b for b in style_ids] for a in style_ids],
        device=logits.device,
        dtype=torch.bool,
    )

    def direction_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_probability = values.log_softmax(dim=1)
        return -(
            (log_probability * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        ).mean()

    loss = 0.5 * (
        direction_loss(logits, positive)
        + direction_loss(logits.T, positive.T)
    )
    similarity = torch.einsum("bsd,csd->bc", left, right) / left.shape[1]
    negative = ~positive
    negative_count = negative.sum().clamp_min(1)
    return loss, {
        "artist_positive_similarity": similarity[positive].mean(),
        "artist_negative_similarity": (
            (similarity * negative).sum() / negative_count
        ),
    }


def _slot_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(tokens.float(), dim=-1)
    similarities = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool)
    return similarities[:, ~identity].square().mean()


def _linear_weight(
    step: int, *, start: float, end: float, end_step: int
) -> float:
    if end_step <= 1:
        return float(end)
    progress = min(1.0, max(0.0, (step - 1) / (end_step - 1)))
    return float(start + progress * (end - start))


def _adapter_trainable_state(adapter: nn.Module) -> dict[str, torch.Tensor]:
    prefixes = (
        "style_k_down.", "style_k_up.", "style_v_down.", "style_v_up.", "alpha",
    )
    return {
        key: value.detach().cpu()
        for key, value in adapter.state_dict().items()
        if key.startswith(prefixes)
    }


def _load_adapter_trainable_state(
    adapter: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    incompatible = adapter.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected query-tokenizer adapter keys: {incompatible.unexpected_keys}"
        )


def _save_training_state(
    path: Path,
    *,
    step: int,
    tokenizer: QueryStyleTokenizerV2,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    cache_summary: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "tokenizer": {
                key: value.detach().cpu()
                for key, value in tokenizer.state_dict().items()
            },
            "adapter": _adapter_trainable_state(adapter),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "resampler_cache": cache_summary,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def _query_loader_config(
    config: dict[str, Any], cfg: dict[str, Any], *, split: str
) -> dict[str, Any]:
    result = _tokenizer_loader_config(config, cfg, split=split)
    if split == str(cfg.get("train_split", "train")):
        result["reference_curriculum"] = dict(cfg["training"]["curriculum"])
        result["self_reference_target_images_per_style"] = 0
    return result


def _reference_inputs(
    batch: dict[str, Any], device: str, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "self":
        target = batch["cached_target_tokens"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )
        return target[:, None], torch.ones(
            target.shape[0], 1, dtype=torch.bool, device=device
        )
    return _reference_tokens(batch, device, mode=mode)


def _flow_forward(
    anima: nn.Module,
    adapter: nn.Module,
    tokenizer: QueryStyleTokenizerV2,
    batch: dict[str, Any],
    device: str,
    training_cfg: dict[str, Any],
    *,
    generator: torch.Generator,
    step: int,
    mode: str,
    measure_base: bool,
    train_auxiliaries: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latents = batch["latents"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    conditioning = batch["conditioning"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    target_tokens = batch["cached_target_tokens"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    heldout, heldout_mask = _reference_inputs(batch, device, "heldout")
    curriculum = _self_reference_curriculum_state(
        step, dict(training_cfg.get("curriculum", {}))
    )

    if mode == "self":
        references = target_tokens[:, None]
        reference_mask = torch.ones(
            target_tokens.shape[0], 1, dtype=torch.bool, device=device
        )
        include_target = torch.ones(
            target_tokens.shape[0], dtype=torch.bool, device=device
        )
    elif mode in {"heldout", "wrong_artist"}:
        references, reference_mask = _reference_inputs(batch, device, mode)
        include_target = torch.zeros(
            target_tokens.shape[0], dtype=torch.bool, device=device
        )
    elif mode == "curriculum":
        references, reference_mask = heldout, heldout_mask
        probability = float(curriculum["target_probability"])
        include_target = torch.rand(
            target_tokens.shape[0], device=device, generator=generator
        ) < probability
        references, reference_mask = _replace_reference_with_target(
            references, reference_mask, target_tokens, include_target
        )
    else:
        raise ValueError(f"Unknown query-tokenizer reference mode: {mode}")

    reconstruction_weight = (
        _linear_weight(
            step,
            start=float(training_cfg.get("reconstruction_weight", 0.02)),
            end=float(training_cfg.get("reconstruction_final_weight", 0.005)),
            end_step=int(training_cfg.get("reconstruction_decay_steps", 8000)),
        )
        if train_auxiliaries else 0.0
    )
    contrastive_every = max(1, int(training_cfg.get("artist_contrastive_every", 2)))
    use_contrastive = bool(
        train_auxiliaries
        and float(training_cfg.get("artist_contrastive_weight", 0.0)) > 0
        and step % contrastive_every == 0
        and target_tokens.shape[0] > 1
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

    autocast = torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    )
    with autocast:
        flow_output = tokenizer(
            references, reference_mask, reconstruct=reconstruction_weight > 0
        )
        adapter.set_style_context(flow_output.tokens)
        try:
            prediction = anima(
                noisy.unsqueeze(2), timesteps.to(latents.dtype),
                context=conditioning, padding_mask=padding_mask,
                target_input_ids=None,
            ).squeeze(2).float()
        finally:
            adapter.clear_style_tokens()

    target = (noise - latents).float()
    flow_loss = F.mse_loss(prediction, target)
    total_loss = flow_loss
    reconstruction_loss = flow_loss.new_zeros(())
    if reconstruction_weight > 0:
        assert flow_output.reconstruction is not None
        assert flow_output.reconstruction_target is not None
        reconstruction_loss = F.smooth_l1_loss(
            flow_output.reconstruction.float(),
            flow_output.reconstruction_target.float(),
            beta=float(training_cfg.get("reconstruction_huber_beta", 0.1)),
        )
        total_loss = total_loss + reconstruction_weight * reconstruction_loss

    diversity_weight = (
        float(training_cfg.get("slot_diversity_weight", 0.0))
        if train_auxiliaries else 0.0
    )
    diversity_loss = _slot_diversity_loss(flow_output.tokens)
    total_loss = total_loss + diversity_weight * diversity_loss

    contrastive_loss = flow_loss.new_zeros(())
    positive_similarity = flow_loss.new_zeros(())
    negative_similarity = flow_loss.new_zeros(())
    contrastive_weight = 0.0
    if use_contrastive:
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            target_output = (
                flow_output
                if mode == "self" or (
                    mode == "curriculum" and bool(curriculum["target_only"])
                )
                else tokenizer(
                    target_tokens[:, None],
                    torch.ones(
                        target_tokens.shape[0], 1,
                        dtype=torch.bool, device=device,
                    ),
                )
            )
            heldout_output = (
                flow_output
                if mode == "heldout" or (
                    mode == "curriculum"
                    and float(curriculum["target_probability"]) == 0.0
                )
                else tokenizer(heldout, heldout_mask)
            )
        style_ids = [str(item.style_id) for item in batch["episodes"]]
        contrastive_loss, contrastive_metrics = _artist_contrastive_loss(
            target_output.tokens,
            heldout_output.tokens,
            style_ids,
            float(training_cfg.get("artist_contrastive_temperature", 0.10)),
        )
        positive_similarity = contrastive_metrics["artist_positive_similarity"]
        negative_similarity = contrastive_metrics["artist_negative_similarity"]
        contrastive_weight = float(training_cfg["artist_contrastive_weight"])
        total_loss = total_loss + contrastive_weight * contrastive_loss

    metrics: dict[str, torch.Tensor] = {
        "loss": total_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "reconstruction_loss": reconstruction_loss.detach(),
        "reconstruction_weight": flow_loss.new_tensor(reconstruction_weight),
        "artist_contrastive_loss": contrastive_loss.detach(),
        "artist_contrastive_weight": flow_loss.new_tensor(contrastive_weight),
        "artist_positive_similarity": positive_similarity.detach(),
        "artist_negative_similarity": negative_similarity.detach(),
        "slot_diversity_loss": diversity_loss.detach(),
        "style_token_rms": flow_output.tokens.detach().float().square().mean().sqrt(),
        "references": reference_mask.sum(dim=1).float().mean(),
        "target_inclusion": include_target.float().mean(),
        "target_probability": flow_loss.new_tensor(
            float(curriculum["target_probability"])
        ),
        "timestep_mean": timesteps.detach().mean(),
    }
    if measure_base:
        adapter.clear_style_tokens()
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            base_prediction = anima(
                noisy.unsqueeze(2), timesteps.to(latents.dtype),
                context=conditioning, padding_mask=padding_mask,
                target_input_ids=None,
            ).squeeze(2).float()
        metrics.update({
            key: value.detach()
            for key, value in _flow_metrics(
                prediction.detach(), base_prediction, target
            ).items()
        })
    return total_loss, metrics


@torch.no_grad()
def _evaluate(
    anima: nn.Module,
    adapter: nn.Module,
    tokenizer: QueryStyleTokenizerV2,
    loader: ProductionStyleLoader,
    device: str,
    training_cfg: dict[str, Any],
    *,
    step: int,
    batches: int,
    seed: int,
    mode: str,
) -> dict[str, float]:
    tokenizer.eval()
    adapter.eval()
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for index in range(batches):
        batch = loader.load_step(index)
        _, metrics = _flow_forward(
            anima, adapter, tokenizer, batch, device, training_cfg,
            generator=torch.Generator(device=device).manual_seed(seed + index * 97),
            step=step, mode=mode, measure_base=True, train_auxiliaries=False,
        )
        rows.append({key: float(value) for key, value in metrics.items()})
    result = _mean_metrics(rows)
    result["elapsed_s"] = time.perf_counter() - started
    tokenizer.train()
    adapter.train()
    return result


@torch.no_grad()
def _calibrate_alpha(
    anima: nn.Module,
    adapter: nn.Module,
    tokenizer: QueryStyleTokenizerV2,
    loader: ProductionStyleLoader,
    device: str,
    training_cfg: dict[str, Any],
) -> dict[str, Any]:
    adapter.begin_alpha_calibration()
    batches = int(training_cfg.get("alpha_calibration_batches", 2))
    for index in range(batches):
        batch = loader.load_step(index)
        references, mask = _reference_inputs(batch, device, "self")
        latents = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
        conditioning = batch["conditioning"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )
        generator = torch.Generator(device=device).manual_seed(0xA17A + index)
        noise = torch.randn(
            latents.shape, device=device, dtype=latents.dtype, generator=generator
        )
        timesteps = _sample_flow_timesteps(
            latents.shape[0], device, training_cfg, generator
        )
        sigma = timesteps[:, None, None, None].to(latents.dtype)
        noisy = (1 - sigma) * latents + sigma * noise
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            style = tokenizer(references, mask).tokens
            adapter.set_style_context(style)
            try:
                anima(
                    noisy.unsqueeze(2), timesteps.to(latents.dtype),
                    context=conditioning,
                    padding_mask=torch.zeros(
                        latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
                        device=device, dtype=latents.dtype,
                    ),
                    target_input_ids=None,
                )
            finally:
                adapter.clear_style_tokens()
    return adapter.finish_alpha_calibration(
        float(training_cfg.get("alpha_target_style_to_text_ratio", 0.25)),
        minimum_alpha=float(training_cfg.get("alpha_minimum", 1e-5)),
        maximum_alpha=float(training_cfg.get("alpha_maximum", 1.0)),
    )


def train_query_style_tokenizer(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_section: str = "query_style_tokenizer_v2",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[config_section])
    training_cfg = dict(cfg["training"])
    steps = int(steps_override or training_cfg["steps"])
    device = str(training_cfg.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260821))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(training_cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    accumulation = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    train_loader_cfg = _query_loader_config(
        config, cfg, split=str(cfg.get("train_split", "train"))
    )
    train_loader_cfg["gradient_accumulation_steps"] = accumulation
    validation_loader_cfg = _query_loader_config(
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
    tokenizer = QueryStyleTokenizerV2(**dict(cfg["model"])).to(device)
    adapter_cfg = dict(cfg["adapter"])
    adapter = _create_and_attach_style_adapter(anima, adapter_cfg, device)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.kv_parameters():
        parameter.requires_grad_(True)
    if bool(training_cfg.get("train_alpha", False)):
        for parameter in adapter.alpha_parameters():
            parameter.requires_grad_(True)

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    resume_state = None
    if bool(training_cfg.get("resume", True)) and state_path.exists():
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(resume_state["tokenizer"], strict=True)
        _load_adapter_trainable_state(adapter, resume_state["adapter"])
        start_step = int(resume_state["step"])
    else:
        calibration = _calibrate_alpha(
            anima, adapter, tokenizer, train_loader, device, training_cfg
        )
        write_json(output / "alpha_calibration.json", calibration)
        print(
            "query-style-tokenizer alpha calibration "
            f"target={calibration['target_style_to_text_ratio']:.4f} "
            f"alpha_min={min(calibration['alpha']):.6g} "
            f"alpha_max={max(calibration['alpha']):.6g}",
            flush=True,
        )

    tokenizer_parameters = [
        parameter for parameter in tokenizer.parameters() if parameter.requires_grad
    ]
    adapter_parameters = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": tokenizer_parameters,
                "lr": float(training_cfg.get("learning_rate", 1e-4)),
                "name": "tokenizer",
            },
            {
                "params": adapter_parameters,
                "lr": float(training_cfg.get("kv_learning_rate", 5e-5)),
                "name": "style_kv_delta",
            },
        ],
        betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
        eps=float(training_cfg.get("adam_eps", 1e-8)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        fused=bool(training_cfg.get("fused_adamw", True) and device.startswith("cuda")),
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])

    trainable_tokenizer = sum(parameter.numel() for parameter in tokenizer_parameters)
    trainable_adapter = sum(parameter.numel() for parameter in adapter_parameters)
    total_tokenizer = sum(parameter.numel() for parameter in tokenizer.parameters())
    print(
        "query-style-tokenizer model "
        f"tokenizer={total_tokenizer / 1e6:.2f}M "
        f"trainable_tokenizer={trainable_tokenizer / 1e6:.2f}M "
        f"trainable_adapter={trainable_adapter / 1e6:.2f}M",
        flush=True,
    )

    wandb_run = None
    wandb_cfg = dict(training_cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("name", "query-style-tokenizer-v2")),
            id=str(wandb_cfg.get("id", "query-style-tokenizer-v2")),
            resume="allow" if start_step else "never",
            config={
                "query_style_tokenizer_v2": cfg,
                "tokenizer_parameters": total_tokenizer,
                "trainable_adapter_parameters": trainable_adapter,
            },
        )

    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    kv_lr = float(training_cfg.get("kv_learning_rate", 5e-5))
    warmup = int(training_cfg.get("warmup_steps", 500))
    minimum_ratio = float(training_cfg.get("minimum_lr_ratio", 0.1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    validation_every = int(training_cfg.get("validation_every", 250))
    validation_batches = int(training_cfg.get("validation_batches", 8))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 500))
    prefetch_workers = int(training_cfg.get("prefetch_workers", 2))
    prefetch_batches = int(training_cfg.get("prefetch_batches", 4))
    micro_start = start_step * accumulation
    total_microsteps = max(0, steps - start_step) * accumulation
    prefetched = train_loader.prefetch(
        micro_start, total_microsteps,
        workers=prefetch_workers, depth=prefetch_batches,
    )
    history_path = output / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.exists() else []
    )
    started = time.perf_counter()
    completed_step = start_step
    try:
        for step in range(start_step + 1, steps + 1):
            step_started = time.perf_counter()
            lr_multiplier = _learning_rate_multiplier(
                step, steps, warmup, minimum_ratio
            )
            optimizer.param_groups[0]["lr"] = base_lr * lr_multiplier
            optimizer.param_groups[1]["lr"] = kv_lr * lr_multiplier
            optimizer.zero_grad(set_to_none=True)
            micro_metrics: list[dict[str, torch.Tensor]] = []
            for micro in range(accumulation):
                batch = next(prefetched)
                generator = torch.Generator(device=device).manual_seed(
                    seed + step * 100_003 + micro
                )
                loss, metrics = _flow_forward(
                    anima, adapter, tokenizer, batch, device, training_cfg,
                    generator=generator, step=step, mode="curriculum",
                    measure_base=(step % log_every == 0 and micro == accumulation - 1),
                    train_auxiliaries=True,
                )
                (loss / accumulation).backward()
                micro_metrics.append(metrics)
            parameters = tokenizer_parameters + adapter_parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            should_log = step == 1 or step % log_every == 0
            if should_log:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                averaged = {
                    key: float(torch.stack([row[key] for row in micro_metrics]).mean())
                    for key in micro_metrics[0]
                    if all(key in row for row in micro_metrics)
                }
                averaged.update({
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "kv_learning_rate": float(optimizer.param_groups[1]["lr"]),
                    "step_s": time.perf_counter() - step_started,
                    "images_per_s": (
                        train_loader.batch_size * accumulation
                        / max(time.perf_counter() - step_started, 1e-6)
                    ),
                })
                averaged.update(adapter.runtime_stats())
                print(
                    f"query-style-tokenizer step={step}/{steps} "
                    f"loss={averaged['loss']:.5f} "
                    f"flow={averaged['flow_loss']:.5f} "
                    f"rec={averaged['reconstruction_loss']:.5f}*"
                    f"{averaged['reconstruction_weight']:.4f} "
                    f"artist={averaged['artist_contrastive_loss']:.4f} "
                    f"grad={averaged['grad_norm']:.4f} "
                    f"step_s={averaged['step_s']:.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in averaged.items()},
                        step=step,
                    )

            if step % validation_every == 0 or step == steps:
                validation_self = _evaluate(
                    anima, adapter, tokenizer, validation_loader, device,
                    training_cfg, step=step, batches=validation_batches,
                    seed=seed ^ 0xBEEF, mode="self",
                )
                validation_heldout = _evaluate(
                    anima, adapter, tokenizer, validation_loader, device,
                    training_cfg, step=step, batches=validation_batches,
                    seed=seed ^ 0xC0FFEE, mode="heldout",
                )
                validation_wrong = _evaluate(
                    anima, adapter, tokenizer, validation_loader, device,
                    training_cfg, step=step, batches=validation_batches,
                    seed=seed ^ 0xC0FFEE, mode="wrong_artist",
                )
                row = {
                    "step": step,
                    "validation_self": validation_self,
                    "validation_heldout": validation_heldout,
                    "validation_wrong_artist": validation_wrong,
                    "correct_vs_wrong_paired_advantage": (
                        validation_heldout["paired_flow_improvement"]
                        - validation_wrong["paired_flow_improvement"]
                    ),
                }
                history.append(row)
                write_json(history_path, history)
                print(f"query-style-tokenizer validation step={step} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({
                        **{
                            f"validation_self/{key}": value
                            for key, value in validation_self.items()
                        },
                        **{
                            f"validation_heldout/{key}": value
                            for key, value in validation_heldout.items()
                        },
                        **{
                            f"validation_wrong_artist/{key}": value
                            for key, value in validation_wrong.items()
                        },
                        "validation/correct_vs_wrong_paired_advantage": row[
                            "correct_vs_wrong_paired_advantage"
                        ],
                    }, step=step)

            if step % checkpoint_every == 0 or step == steps:
                checkpoint = checkpoint_dir / f"step-{step:07d}.pt"
                _save_training_state(
                    checkpoint, step=step, tokenizer=tokenizer, adapter=adapter,
                    optimizer=optimizer, cfg=cfg, cache_summary=cache_summary,
                )
                _save_training_state(
                    state_path, step=step, tokenizer=tokenizer, adapter=adapter,
                    optimizer=optimizer, cfg=cfg, cache_summary=cache_summary,
                )
            completed_step = step
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    result = {
        "steps": completed_step,
        "requested_steps": steps,
        "tokenizer_parameters": total_tokenizer,
        "trainable_tokenizer_parameters": trainable_tokenizer,
        "trainable_adapter_parameters": trainable_adapter,
        "reconstruction_weight": float(training_cfg.get("reconstruction_weight", 0.02)),
        "reconstruction_final_weight": float(
            training_cfg.get("reconstruction_final_weight", 0.005)
        ),
        "resampler_cache": cache_summary,
        "elapsed_s": time.perf_counter() - started,
        "final_validation": history[-1] if history else None,
    }
    write_json(output / "summary.json", result)
    return result


def smoke_test_query_style_tokenizer(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    smoke_config = copy.deepcopy(config)
    cfg = smoke_config["query_style_tokenizer_v2"]
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    cfg["loader"]["batch_size"] = 1
    cfg["training"].update({
        "steps": 2,
        "gradient_accumulation_steps": 1,
        "validation_every": 2,
        "validation_batches": 1,
        "checkpoint_every": 2,
        "resume": False,
    })
    cfg["training"].setdefault("wandb", {})["enabled"] = False
    return train_query_style_tokenizer(
        smoke_config, destination, steps_override=2
    )
