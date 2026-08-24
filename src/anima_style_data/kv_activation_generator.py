"""Reference-conditioned native text K/V activation generation.

The student predicts the raw pre-normalization K/V residual used by a K/V-only
LoRA, rather than generating LoRA factors or adding a separate style-attention
branch.  Offline LoRA factors and materialized merged-LoRA images are training
data only; inference receives visual references and the current text context.
"""

from __future__ import annotations

import copy
import gc
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .detail_style_cross_attention import DetailPreservingTypedSlotReader
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json, write_records
from .kv_activation_modulation import apply_kv_factors, load_kv_lora_factor_bank
from .lora_functional_distillation import build_mixture_specs
from .lora_oracle_bootstrap import (
    _materialize_reader_code_bank,
    _oracle_detail_config,
)


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, hidden * 2, bias=False)
        self.output = nn.Linear(hidden, dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, content = self.input(values).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


class ReferenceConditionedKVActivationGenerator(nn.Module):
    """Generate block-local raw delta-K/delta-V activations.

    The 512 text tokens query visual style memory.  Every Anima block owns its
    context projection and final K/V head; only the visual memory projection
    and the small feed-forward trunk are shared.
    """

    def __init__(
        self,
        *,
        style_dim: int = 1024,
        context_dim: int = 1024,
        output_dim: int = 2048,
        blocks: int = 28,
        hidden_dim: int = 256,
        heads: int = 8,
        ff_dim: int = 1024,
        output_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if output_init_scale <= 0:
            raise ValueError("output_init_scale must be positive")
        self.style_dim = int(style_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.blocks = int(blocks)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.style_norm = nn.LayerNorm(style_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.style_key = nn.Linear(style_dim, hidden_dim, bias=False)
        self.style_value = nn.Linear(style_dim, hidden_dim, bias=False)
        self.block_embedding = nn.Embedding(blocks, hidden_dim)
        self.context_query = nn.ModuleList(
            nn.Linear(context_dim, hidden_dim, bias=False) for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = _SwiGLU(hidden_dim, ff_dim)
        self.output_head = nn.ModuleList(
            nn.Linear(hidden_dim, output_dim * 2, bias=False)
            for _ in range(blocks)
        )
        self.log_gain = nn.Parameter(torch.zeros(blocks, 2))
        self.reset_parameters(float(output_init_scale))

    def reset_parameters(self, output_init_scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.block_embedding.weight, std=self.hidden_dim**-0.5)
        for head in self.output_head:
            with torch.no_grad():
                head.weight.mul_(output_init_scale)

    def forward(
        self,
        style_memory: torch.Tensor,
        text_context: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if style_memory.ndim != 3 or style_memory.shape[-1] != self.style_dim:
            raise ValueError("style_memory must be [batch,slots,style_dim]")
        if text_context.ndim != 3 or text_context.shape[-1] != self.context_dim:
            raise ValueError("text_context must be [batch,tokens,context_dim]")
        if style_memory.shape[0] != text_context.shape[0]:
            raise ValueError("style and text batches disagree")
        block = int(block_index)
        if not 0 <= block < self.blocks:
            raise ValueError("block_index is outside the model")
        style = self.style_norm(style_memory)
        block_code = self.block_embedding.weight[block].to(style.dtype)
        key = self.style_key(style) + block_code[None, None]
        value = self.style_value(style)
        query = self.context_query[block](self.context_norm(text_context))
        batch, text_tokens, _ = query.shape
        style_tokens = int(style.shape[1])
        query = query.reshape(
            batch, text_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch, style_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch, style_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(
            batch, text_tokens, self.hidden_dim
        )
        hidden = self.output_norm(attended)
        hidden = hidden + self.ff(self.ff_norm(hidden))
        output = self.output_head[block](hidden).reshape(
            batch, text_tokens, 2, self.output_dim
        ).transpose(1, 2)
        gain = self.log_gain[block].float().clamp(-4.0, 4.0).exp().to(output.dtype)
        return output * gain[None, :, None, None]


class _OperatorCrossBlock(nn.Module):
    """Let operator queries read the complete typed reference memory."""

    def __init__(self, dim: int, heads: int, ff_dim: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("operator hidden dimension must divide heads")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, ff_dim)

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        batch, query_tokens, dim = queries.shape
        memory_tokens = int(memory.shape[1])
        query = self.query(self.query_norm(queries)).reshape(
            batch, query_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        normalized_memory = self.memory_norm(memory)
        key = self.key(normalized_memory).reshape(
            batch, memory_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized_memory).reshape(
            batch, memory_tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, query_tokens, dim)
        queries = queries + self.output(attended)
        return queries + self.ff(self.ff_norm(queries))


class ReferenceConditionedLowRankKVOperator(nn.Module):
    """Generate a fresh low-rank text K/V operator from each reference set.

    The reference does not emit extra style-attention tokens.  Instead it
    generates normalized input/output directions and singular strengths for
    a block-local operator, then applies that operator to all native text
    tokens.  This preserves the exact algebraic form of a K/V-only LoRA while
    never exposing teacher factors, artist IDs, or mixture coefficients.
    """

    def __init__(
        self,
        *,
        style_dim: int = 1024,
        context_dim: int = 1024,
        output_dim: int = 2048,
        blocks: int = 28,
        hidden_dim: int = 256,
        heads: int = 8,
        ff_dim: int = 1024,
        operator_layers: int = 2,
        operator_rank: int = 32,
        initial_sigma: float = 0.01,
        minimum_sigma: float = 1e-6,
        maximum_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if operator_layers <= 0 or operator_rank <= 0:
            raise ValueError("operator layers/rank must be positive")
        if not 0 < minimum_sigma <= initial_sigma <= maximum_sigma:
            raise ValueError("operator sigma bounds/initializer are invalid")
        self.style_dim = int(style_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.blocks = int(blocks)
        self.hidden_dim = int(hidden_dim)
        self.operator_rank = int(operator_rank)
        self.minimum_sigma = float(minimum_sigma)
        self.maximum_sigma = float(maximum_sigma)
        self.style_norm = nn.LayerNorm(style_dim)
        self.style_input = nn.Linear(style_dim, hidden_dim, bias=False)
        # [block, K/V, down/up, rank, hidden].  The identities are explicit;
        # no mean pooling or shared rank token is used.
        self.operator_queries = nn.Parameter(
            torch.empty(blocks, 2, 2, operator_rank, hidden_dim)
        )
        self.reader = nn.ModuleList(
            _OperatorCrossBlock(hidden_dim, heads, ff_dim)
            for _ in range(operator_layers)
        )
        # Dense output directions are block-local.  Sharing only the reader
        # trunk avoids constraining unseen styles to a stored global LoRA basis.
        self.down_output = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, context_dim, bias=False) for _ in range(2)
            )
            for _ in range(blocks)
        )
        self.up_output = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, output_dim, bias=False) for _ in range(2)
            )
            for _ in range(blocks)
        )
        self.log_sigma = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, 1, bias=True) for _ in range(2)
            )
            for _ in range(blocks)
        )
        self.reset_parameters(float(initial_sigma))

    def reset_parameters(self, initial_sigma: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.operator_queries, std=self.hidden_dim**-0.5)
        for block_heads in self.log_sigma:
            for head in block_heads:
                nn.init.normal_(head.weight, std=0.01)
                nn.init.constant_(head.bias, float(torch.tensor(initial_sigma).log()))

    def _operator(
        self, style_memory: torch.Tensor, block: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(style_memory.shape[0])
        memory = self.style_input(self.style_norm(style_memory))
        queries = self.operator_queries[block].reshape(
            4 * self.operator_rank, self.hidden_dim
        )[None].expand(batch, -1, -1)
        for layer in self.reader:
            queries = layer(queries, memory)
        queries = queries.reshape(
            batch, 2, 2, self.operator_rank, self.hidden_dim
        )
        down_values = []
        up_values = []
        sigma_values = []
        for kind in range(2):
            down_hidden = queries[:, kind, 0]
            up_hidden = queries[:, kind, 1]
            down = self.down_output[block][kind](down_hidden)
            up = self.up_output[block][kind](up_hidden)
            # Unit directions plus an explicit singular strength remove the
            # arbitrary up/down scale gauge that otherwise destabilizes a
            # dynamically generated factorization.
            down_values.append(
                F.normalize(down.float(), dim=-1).to(down.dtype)
            )
            up_values.append(F.normalize(up.float(), dim=-1).to(up.dtype))
            log_sigma = self.log_sigma[block][kind](
                0.5 * (down_hidden + up_hidden)
            ).squeeze(-1)
            sigma_values.append(
                log_sigma.float().exp().clamp(
                    self.minimum_sigma, self.maximum_sigma
                ).to(down.dtype)
            )
        return (
            torch.stack(down_values, dim=1),
            torch.stack(up_values, dim=1),
            torch.stack(sigma_values, dim=1),
        )

    def forward(
        self,
        style_memory: torch.Tensor,
        text_context: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if style_memory.ndim != 3 or style_memory.shape[-1] != self.style_dim:
            raise ValueError("style_memory must be [batch,slots,style_dim]")
        if text_context.ndim != 3 or text_context.shape[-1] != self.context_dim:
            raise ValueError("text_context must be [batch,tokens,context_dim]")
        if style_memory.shape[0] != text_context.shape[0]:
            raise ValueError("style and text batches disagree")
        block = int(block_index)
        if not 0 <= block < self.blocks:
            raise ValueError("block_index is outside the model")
        down, up, sigma = self._operator(style_memory, block)
        hidden = torch.einsum("bnc,bkrc->bknr", text_context, down)
        hidden = hidden * sigma[:, :, None]
        return torch.einsum("bknr,bkro->bkno", hidden, up)


class _NativeAttentionProbe(nn.Module):
    """Frozen native K/V normalization and O for functional probe queries."""

    def __init__(self, cross: nn.Module) -> None:
        super().__init__()
        self.n_heads = int(cross.n_heads)
        self.head_dim = int(cross.head_dim)
        self.k_proj = copy.deepcopy(cross.k_proj)
        self.v_proj = copy.deepcopy(cross.v_proj)
        self.q_norm = copy.deepcopy(cross.q_norm)
        self.k_norm = copy.deepcopy(cross.k_norm)
        self.v_norm = copy.deepcopy(cross.v_norm)
        self.output_proj = copy.deepcopy(cross.output_proj)
        self.requires_grad_(False)

    def project_context(
        self, context: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens = context.shape[:2]
        key = self.k_proj(context) + delta[:, 0]
        value = self.v_proj(context) + delta[:, 1]
        key = key.reshape(batch, tokens, self.n_heads, self.head_dim)
        value = value.reshape(batch, tokens, self.n_heads, self.head_dim)
        return self.k_norm(key), self.v_norm(value)

    def attend(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.q_norm(queries)
        attended = F.scaled_dot_product_attention(
            queries.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        attended = attended.transpose(1, 2).reshape(
            attended.shape[0], attended.shape[2], -1
        )
        return self.output_proj(attended)


def _normalized_activation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    direction_weight: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_f = teacher.float()
    student_f = student.float()
    dimensions = tuple(range(2, teacher_f.ndim))
    teacher_rms = teacher_f.square().mean(dim=dimensions).sqrt().clamp_min(1e-5)
    student_rms = student_f.square().mean(dim=dimensions).sqrt()
    scale = teacher_rms[..., None, None]
    huber = F.smooth_l1_loss(student_f / scale, teacher_f / scale, beta=0.1)
    cosine = F.cosine_similarity(
        student_f.flatten(2), teacher_f.flatten(2), dim=-1
    )
    direction = (1.0 - cosine).mean()
    magnitude = (
        student_rms.clamp_min(1e-8).log() - teacher_rms.log()
    ).abs().mean()
    loss = huber + float(direction_weight) * direction + float(
        magnitude_weight
    ) * magnitude
    return loss, {
        "loss": loss.detach(),
        "normalized_huber": huber.detach(),
        "cosine": cosine.mean().detach(),
        "direction_loss": direction.detach(),
        "magnitude_log_error": magnitude.detach(),
        "student_to_teacher_rms": (student_rms / teacher_rms).mean().detach(),
    }


def _mixture_target(
    context: torch.Tensor,
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
    components: torch.Tensor,
    weights: torch.Tensor,
    block: int,
) -> torch.Tensor:
    result = context.new_zeros(
        context.shape[0], 2, context.shape[1], teacher_up.shape[-2]
    )
    for slot in range(components.shape[1]):
        indices = components[:, slot].clamp_min(0)
        active = (components[:, slot] >= 0).to(context.dtype)
        value = apply_kv_factors(
            context,
            teacher_down[indices, block].to(context.dtype),
            teacher_up[indices, block].to(context.dtype),
        )
        result = result + value * (
            weights[:, slot].to(context.dtype) * active
        )[:, None, None, None]
    return result


def prepare_kv_activation_mixture_specs(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Create diverse K/V-only mixture specs and filter unstable functions."""

    cfg = dict(config["kv_activation_mixture_teacher"])
    cache_cfg = dict(cfg["teacher_cache"])
    output = destination / str(cache_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    artist_ids, down, up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    specs = build_mixture_specs(
        len(artist_ids),
        pair_count=int(cache_cfg.get("pair_mixtures", 64)),
        triple_count=int(cache_cfg.get("triple_mixtures", 64)),
        amplified_count=int(cache_cfg.get("amplified_mixtures", 64)),
        signed_count=int(cache_cfg.get("signed_mixtures", 64)),
        amplified_sum_range=tuple(cache_cfg.get("amplified_sum_range", [1.0, 1.5])),
        signed_beta_range=tuple(cache_cfg.get("signed_beta_range", [0.05, 0.5])),
        seed=int(cache_cfg.get("seed", 20260824)),
    )
    device = str(cache_cfg.get("device", "cuda"))
    contexts = load_file(
        destination / str(cache_cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"]
    context_count = min(int(cache_cfg.get("probe_contexts", 8)), len(contexts))
    token_count = min(int(cache_cfg.get("probe_tokens", 64)), contexts.shape[1])
    channel_count = min(int(cache_cfg.get("probe_channels", 256)), up.shape[-2])
    context_ids = torch.linspace(0, len(contexts) - 1, context_count).round().long()
    token_ids = torch.linspace(0, contexts.shape[1] - 1, token_count).round().long()
    channel_ids = torch.linspace(0, up.shape[-2] - 1, channel_count).round().long()
    contexts = contexts[context_ids][:, token_ids].to(device=device, dtype=torch.bfloat16)
    down = down.to(device=device, dtype=torch.bfloat16)
    up = up[:, :, :, channel_ids].to(device=device, dtype=torch.bfloat16)
    blocks = tuple(
        int(value)
        for value in cache_cfg.get("probe_blocks", [0, 4, 8, 12, 16, 20, 24, 27])
    )
    rms_values: dict[int, float] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for spec in specs:
            energies = []
            for block in blocks:
                value = contexts.new_zeros(
                    len(contexts), 2, token_count, channel_count
                )
                for component, weight in zip(
                    spec.components, spec.weights, strict=True
                ):
                    value.add_(
                        apply_kv_factors(
                            contexts,
                            down[component, block][None].expand(len(contexts), -1, -1, -1),
                            up[component, block][None].expand(len(contexts), -1, -1, -1),
                        ),
                        alpha=float(weight),
                    )
                energies.append(value.float().square().mean())
            rms_values[spec.index] = float(torch.stack(energies).mean().sqrt())
    single_median = float(torch.tensor([
        rms_values[index] for index in range(len(artist_ids))
    ]).median())
    minimum, maximum = (
        float(value)
        for value in cache_cfg.get("stable_activation_ratio_range", [0.2, 1.5])
    )
    rows = []
    for spec in specs:
        ratio = rms_values[spec.index] / max(single_median, 1e-8)
        enabled = spec.kind == "single" or minimum <= ratio <= maximum
        rows.append({
            "index": spec.index,
            "kind": spec.kind,
            "components": list(spec.components),
            "weights": list(spec.weights),
            "style_ids": [artist_ids[index] for index in spec.components],
            "mixture_style_id": f"kv-mixture-{spec.index:05d}",
            "activation_rms": rms_values[spec.index],
            "activation_to_single_median_ratio": ratio,
            "enabled": enabled,
        })
    write_records(output / "mixtures.parquet", rows)
    summary = {
        "artists": len(artist_ids),
        "mixtures": len(rows),
        "enabled_mixtures": sum(bool(row["enabled"]) for row in rows),
        "pair": sum(row["kind"] == "pair" for row in rows),
        "triple": sum(row["kind"] == "triple" for row in rows),
        "amplified": sum(row["kind"] == "amplified" for row in rows),
        "signed": sum(row["kind"] == "signed" for row in rows),
        "single_activation_rms_median": single_median,
        "stable_activation_ratio_range": [minimum, maximum],
        "content_contexts": len(contexts),
    }
    write_json(output / "summary.json", summary)
    return summary


def _load_reader(
    config: dict[str, Any], destination: Path, cfg: dict[str, Any], device: str
) -> DetailPreservingTypedSlotReader:
    state = torch.load(
        destination / str(cfg["reader_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    detail = _oracle_detail_config(config, dict(config["kv_lora_oracle_bootstrap"]))
    reader = DetailPreservingTypedSlotReader(**dict(detail["model"]))
    reader.load_state_dict(state["reader"], strict=True)
    return reader.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()


def _materialize_style_codes(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    reader: DetailPreservingTypedSlotReader,
    artist_ids: list[str],
    mixture_rows: list[dict[str, Any]],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    training = dict(cfg["training"])
    chunk = int(training.get("materialization_style_chunk", 16))
    single_references = int(training.get("single_reference_images", 8))
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["synthetic_reference_cache"]),
        split="train", style_ids=artist_ids, batch_size=chunk,
        references=single_references, seed=int(cfg["seed"]) ^ 0x53594E54,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    human_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train", style_ids=artist_ids, batch_size=chunk,
        references=single_references, seed=int(cfg["seed"]) ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    single_codes: dict[str, torch.Tensor] = {}
    counts: torch.Tensor | None = None
    for domain, loader in (("synthetic", synthetic_loader), ("human", human_loader)):
        codes, domain_counts = _materialize_reader_code_bank(
            reader, loader, artist_ids, reference_images=single_references,
            seed=int(cfg["seed"]) ^ (0x11111111 if domain == "synthetic" else 0x22222222),
            device=device, style_chunk_size=chunk,
        )
        single_codes[domain] = codes
        counts = domain_counts if counts is None else counts
        if not torch.equal(counts, domain_counts):
            raise RuntimeError("Single-domain reference-count views disagree")
    mixture_by_kind: dict[str, torch.Tensor] = {}
    mixture_references = int(training.get("mixture_reference_images", 4))
    for kind in ("pair", "triple", "amplified", "signed"):
        rows = [row for row in mixture_rows if str(row["kind"]) == kind]
        style_ids = [str(row["mixture_style_id"]) for row in rows]
        loader = CachedTeacherReferenceLoader(
            destination / str(cfg["mixture_reference_cache"]),
            split="train", style_ids=style_ids, batch_size=chunk,
            references=mixture_references,
            seed=int(cfg["seed"]) ^ (sum(map(ord, kind)) * 1_000_003),
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            strict_style_ids=True,
        )
        codes, _ = _materialize_reader_code_bank(
            reader, loader, style_ids, reference_images=mixture_references,
            seed=int(cfg["seed"]) ^ (sum(map(ord, kind)) * 1_000_003),
            device=device, style_chunk_size=chunk,
        )
        mixture_by_kind[kind] = codes
    assert counts is not None
    return single_codes, mixture_by_kind, counts


def _save_training_state(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": int(step),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, temporary)
    temporary.replace(path)


def _load_native_attention_probes(
    config: dict[str, Any], destination: Path, device: str
) -> nn.ModuleList:
    """Copy only native K/V normalization and O, then release the full DiT."""

    from .style_transfer import _resolve_anima_model

    anima = _resolve_anima_model(config, destination, device)
    anima.requires_grad_(False).eval()
    probes = nn.ModuleList(
        _NativeAttentionProbe(block.cross_attn) for block in anima.blocks
    ).to(device=device, dtype=torch.bfloat16)
    probes.requires_grad_(False).eval()
    del anima
    gc.collect()
    torch.cuda.empty_cache()
    return probes


def train_reference_conditioned_kv_activation_generator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_key: str = "kv_reference_activation_generator",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[config_key])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 3000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)), dtype=torch.float16,
    )
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)
    mixture_rows = [
        row for row in read_records(destination / str(cfg["mixture_manifest"]))
        if str(row["kind"]) != "single" and bool(row.get("enabled", True))
    ]
    rows_by_kind = {
        kind: [row for row in mixture_rows if str(row["kind"]) == kind]
        for kind in ("pair", "triple", "amplified", "signed")
    }
    if any(not rows for rows in rows_by_kind.values()):
        raise RuntimeError("Every configured mixture kind needs enabled rows")
    reader = _load_reader(config, destination, cfg, device)
    single_codes, mixture_codes, reference_counts = _materialize_style_codes(
        config, destination, cfg, reader, artist_ids, mixture_rows, device
    )
    del reader
    torch.cuda.empty_cache()

    contexts = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"].to(device=device, dtype=torch.bfloat16)
    heldout_contexts = int(training.get("heldout_contexts", 32))
    train_context_count = len(contexts) - heldout_contexts
    if train_context_count <= 0:
        raise ValueError("heldout_contexts leaves no training captions")
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "direct_cross_attention"))
    model_type: type[nn.Module]
    if architecture == "direct_cross_attention":
        model_type = ReferenceConditionedKVActivationGenerator
    elif architecture == "bilinear_low_rank_operator":
        model_type = ReferenceConditionedLowRankKVOperator
    else:
        raise ValueError(f"Unsupported K/V activation architecture: {architecture}")
    model = model_type(
        style_dim=int(next(iter(single_codes.values())).shape[-1]),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 2e-4)),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    direction_weight = float(training.get("direction_weight", 0.2))
    magnitude_weight = float(training.get("magnitude_weight", 0.05))
    base_lr = float(training.get("learning_rate", 2e-4))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every", 10))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    validation_every = int(training.get("validation_every", 250))
    validation_tokens = int(training.get("validation_tokens", 64))
    attention_start = int(training.get("attention_start_step", 1000))
    attention_ramp = int(training.get("attention_ramp_steps", 500))
    attention_weight = float(training.get("attention_weight", 0.3))
    attention_queries = int(training.get("attention_queries", 32))
    categories = ("single", "pair", "triple", "amplified", "signed")
    category_weights = tuple(
        float(value)
        for value in training.get("category_weights", [0.25, 0.1875, 0.0625, 0.25, 0.25])
    )
    if len(category_weights) != len(categories):
        raise ValueError("category_weights must cover single/pair/triple/amplified/signed")

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-activation-generator")),
            id=str(wandb_cfg.get("id", "kv-reference-activation-generator")),
            resume="allow" if start_step else "never",
            config={config_key: cfg},
        )
    running: dict[str, list[float]] = defaultdict(list)
    native_probes: nn.ModuleList | None = None
    started = time.perf_counter()
    model.train()
    try:
        for step in range(start_step + 1, steps + 1):
            rng = random.Random(seed + step * 1_000_003)
            category = rng.choices(categories, weights=category_weights, k=1)[0]
            context_indices = torch.tensor(
                [rng.randrange(train_context_count) for _ in range(batch)],
                device=device,
            )
            context = contexts[context_indices]
            if category == "single":
                target_indices = torch.tensor(
                    [rng.randrange(len(artist_ids)) for _ in range(batch)],
                    device=device,
                )
                domain = "human" if rng.random() < 0.5 else "synthetic"
                codes = single_codes[domain]
                view_indices = torch.tensor(
                    [rng.randrange(codes.shape[1]) for _ in range(batch)], device=device
                )
                style = codes[target_indices, view_indices]
                components = target_indices[:, None]
                weights = torch.ones(batch, 1, device=device)
            else:
                source_rows = rows_by_kind[category]
                target_indices = torch.tensor(
                    [rng.randrange(len(source_rows)) for _ in range(batch)], device=device
                )
                codes = mixture_codes[category]
                view_indices = torch.tensor(
                    [rng.randrange(codes.shape[1]) for _ in range(batch)], device=device
                )
                style = codes[target_indices, view_indices]
                selected = [source_rows[index] for index in target_indices.cpu().tolist()]
                max_components = max(len(row["components"]) for row in selected)
                components = torch.full((batch, max_components), -1, device=device, dtype=torch.long)
                weights = torch.zeros(batch, max_components, device=device)
                for index, row in enumerate(selected):
                    count = len(row["components"])
                    components[index, :count] = torch.tensor(row["components"], device=device)
                    weights[index, :count] = torch.tensor(row["weights"], device=device)
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            selected_blocks = [
                (first_block + offset) % model.blocks
                for offset in range(blocks_per_step)
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            metrics_by_block = []
            active_attention_weight = attention_weight * max(
                0.0,
                min(1.0, (step - attention_start) / max(1, attention_ramp)),
            )
            if active_attention_weight > 0 and native_probes is None:
                native_probes = _load_native_attention_probes(
                    config, destination, device
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for block in selected_blocks:
                    student = model(style, context, block)
                    teacher = _mixture_target(
                        context, teacher_down, teacher_up,
                        components, weights, block,
                    )
                    block_loss, block_metrics = _normalized_activation_loss(
                        student, teacher,
                        direction_weight=direction_weight,
                        magnitude_weight=magnitude_weight,
                    )
                    if active_attention_weight > 0:
                        assert native_probes is not None
                        probe = native_probes[block]
                        query_generator = torch.Generator(device=device).manual_seed(
                            seed + step * 1_000_003 + block * 10_007
                        )
                        queries = torch.randn(
                            batch,
                            attention_queries,
                            probe.n_heads,
                            probe.head_dim,
                            device=device,
                            dtype=torch.bfloat16,
                            generator=query_generator,
                        )
                        zero = torch.zeros_like(student)
                        base_key, base_value = probe.project_context(context, zero)
                        student_key, student_value = probe.project_context(
                            context, student
                        )
                        teacher_key, teacher_value = probe.project_context(
                            context, teacher
                        )
                        base_output = probe.attend(queries, base_key, base_value)
                        student_effect = (
                            probe.attend(queries, student_key, student_value)
                            - base_output
                        )
                        teacher_effect = (
                            probe.attend(queries, teacher_key, teacher_value)
                            - base_output
                        )
                        attention_loss, attention_metrics = (
                            _normalized_activation_loss(
                                student_effect[:, None],
                                teacher_effect[:, None],
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                        )
                        block_loss = block_loss + (
                            active_attention_weight * attention_loss
                        )
                        block_metrics.update({
                            f"attention_{key}": value
                            for key, value in attention_metrics.items()
                        })
                    losses.append(block_loss)
                    metrics_by_block.append(block_metrics)
                loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            lr = base_lr * min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            running["loss"].append(float(loss.detach()))
            running["grad_norm"].append(float(grad_norm))
            running["learning_rate"].append(lr)
            running["attention_weight"].append(active_attention_weight)
            running[f"category/{category}"].append(1.0)
            for key in metrics_by_block[0]:
                running[key].append(sum(float(row[key]) for row in metrics_by_block) / len(metrics_by_block))
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["context_unique_fraction"] = len(set(context_indices.tolist())) / batch
                print(f"K/V activation generator step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()
            if validation_every > 0 and step % validation_every == 0:
                model.eval()
                validation_rows: dict[str, list[float]] = defaultdict(list)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    val_artists = torch.linspace(0, len(artist_ids) - 1, min(8, len(artist_ids)), device=device).round().long().unique()
                    val_context_ids = torch.linspace(train_context_count, len(contexts) - 1, min(4, heldout_contexts), device=device).round().long().unique()
                    val_style = single_codes["synthetic"][val_artists, 0]
                    for block in range(model.blocks):
                        for context_index in val_context_ids.tolist():
                            token_ids = torch.linspace(
                                0,
                                contexts.shape[1] - 1,
                                min(validation_tokens, contexts.shape[1]),
                                device=device,
                            ).round().long().unique()
                            val_context = contexts[
                                context_index, token_ids
                            ].expand(len(val_artists), -1, -1)
                            student = model(val_style, val_context, block)
                            teacher = _mixture_target(
                                val_context, teacher_down, teacher_up,
                                val_artists[:, None], torch.ones(len(val_artists), 1, device=device), block,
                            )
                            _, values = _normalized_activation_loss(
                                student, teacher,
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                            for key, value in values.items():
                                validation_rows[key].append(float(value))
                validation = {key: sum(values) / len(values) for key, values in validation_rows.items()}
                print(f"K/V activation generator validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"val/{key}": value for key, value in validation.items()}, step=step)
                model.train()
            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path, step=step, model=model, optimizer=optimizer, cfg=cfg
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "mixtures": len(mixture_rows),
        "contexts": len(contexts),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": architecture,
        "elapsed_seconds": time.perf_counter() - started,
        "end_to_end_flow_training": False,
        "native_artist_teacher": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_reference_conditioned_kv_activation_generator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_activation_generator"]
    cfg["output_directory"] = "kv_reference_activation_generator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    return train_reference_conditioned_kv_activation_generator(
        effective, destination, steps_override=2
    )


def train_reference_conditioned_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_reference_conditioned_kv_activation_generator(
        config,
        destination,
        config_key="kv_reference_bilinear_operator",
    )


def smoke_test_reference_conditioned_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_bilinear_operator"]
    cfg["output_directory"] = "kv_reference_bilinear_operator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    return train_reference_conditioned_kv_activation_generator(
        effective,
        destination,
        steps_override=2,
        config_key="kv_reference_bilinear_operator",
    )
