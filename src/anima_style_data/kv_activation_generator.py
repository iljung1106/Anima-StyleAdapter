"""Reference-conditioned native text K/V activation generation.

The student predicts the raw pre-normalization K/V residual used by a K/V-only
LoRA, rather than generating LoRA factors or adding a separate style-attention
branch.  Offline LoRA factors and materialized merged-LoRA images are training
data only; inference receives visual references and the current text context.
"""

from __future__ import annotations

import copy
import gc
import math
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


class _LowRankKVResidualHead(nn.Module):
    """LoRA-shaped K/V residual head over a conditioned token feature."""

    def __init__(
        self, hidden_dim: int, output_dim: int, rank: int, init_scale: float
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.down = nn.Linear(hidden_dim, 2 * rank, bias=False)
        self.up = nn.Parameter(torch.empty(2, rank, output_dim))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up)
        with torch.no_grad():
            self.up.mul_(float(init_scale))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        low_rank = self.down(hidden).reshape(*hidden.shape[:-1], 2, self.rank)
        return torch.einsum("...kr,kro->...ko", low_rank, self.up)


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
        ff_layers: int = 1,
        output_rank: int = 0,
        output_experts: int = 0,
        output_top_k: int = 0,
        output_init_scale: float = 0.02,
        normalize_style: bool = True,
        normalize_attended: bool = True,
        use_block_embedding: bool = True,
        enable_qo: bool = False,
        enable_q: bool = True,
        enable_o: bool = True,
        stream_dim: int = 2048,
        stream_rank: int = 32,
        stream_experts: int = 0,
        stream_top_k: int = 0,
        expert_usage_decay: float = 0.99,
        expert_balance_cap: float = 1.5,
        router_init_scale: float = 1.0,
        router_jitter: float = 0.0,
        router_jitter_steps: int = 0,
        expert_bias_update_rate: float = 0.0,
        expert_bias_decay: float = 0.999,
        expert_bias_max: float = 0.5,
        expert_bias_update_steps: int = 0,
        expert_bias_decay_end_step: int = 0,
        expert_bias_population_update: bool = False,
        expert_bias_deadband: float = 0.0,
        output_entropy_target: float = -1.0,
        stream_entropy_target: float = -1.0,
        expert_specialization_steps: int = 0,
        output_core_experts: int = 0,
        stream_core_experts: int = 0,
        output_core_margin: float = 0.0,
        stream_core_margin: float = 0.0,
        expert_specialization_start_step: int = 0,
        router_temperature_end: float = 1.0,
        router_temperature_steps: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if ff_layers <= 0 or output_rank < 0:
            raise ValueError("ff_layers must be positive and output_rank non-negative")
        if output_experts < 0 or stream_experts < 0:
            raise ValueError("expert counts must be non-negative")
        if output_experts and not 0 < output_top_k <= output_experts:
            raise ValueError("output_top_k must select existing experts")
        if stream_experts and not 0 < stream_top_k <= stream_experts:
            raise ValueError("stream_top_k must select existing experts")
        if not 0.0 <= expert_usage_decay < 1.0 or expert_balance_cap < 1.0:
            raise ValueError("expert usage controls are invalid")
        if output_init_scale < 0:
            raise ValueError("output_init_scale must be non-negative")
        self.style_dim = int(style_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.blocks = int(blocks)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.output_rank = int(output_rank)
        self.output_experts = int(output_experts)
        self.output_top_k = int(output_top_k)
        self.enable_qo = bool(enable_qo)
        self.enable_q = bool(enable_q) and self.enable_qo
        self.enable_o = bool(enable_o) and self.enable_qo
        if self.enable_qo and not (self.enable_q or self.enable_o):
            raise ValueError("Q/O modulation requires at least one enabled path")
        self.stream_dim = int(stream_dim)
        self.stream_rank = int(stream_rank)
        self.stream_experts = int(stream_experts)
        self.stream_top_k = int(stream_top_k)
        self.expert_usage_decay = float(expert_usage_decay)
        self.expert_balance_cap = float(expert_balance_cap)
        self.router_init_scale = float(router_init_scale)
        self.router_jitter = float(router_jitter)
        self.router_jitter_steps = int(router_jitter_steps)
        self.expert_bias_update_rate = float(expert_bias_update_rate)
        self.expert_bias_decay = float(expert_bias_decay)
        self.expert_bias_max = float(expert_bias_max)
        self.expert_bias_update_steps = int(expert_bias_update_steps)
        self.expert_bias_decay_end_step = int(expert_bias_decay_end_step)
        self.expert_bias_population_update = bool(expert_bias_population_update)
        self.expert_bias_deadband = float(expert_bias_deadband)
        self.output_entropy_target = float(output_entropy_target)
        self.stream_entropy_target = float(stream_entropy_target)
        self.expert_specialization_steps = int(expert_specialization_steps)
        self.output_core_experts = int(output_core_experts)
        self.stream_core_experts = int(stream_core_experts)
        self.output_core_margin = float(output_core_margin)
        self.stream_core_margin = float(stream_core_margin)
        self.expert_specialization_start_step = int(
            expert_specialization_start_step
        )
        self.router_temperature_end = float(router_temperature_end)
        self.router_temperature_steps = int(router_temperature_steps)
        self._routing_step = 0
        if (
            self.router_init_scale < 0
            or self.router_jitter < 0
            or self.router_jitter_steps < 0
            or self.expert_bias_update_rate < 0
            or not 0.0 <= self.expert_bias_decay <= 1.0
            or self.expert_bias_max < 0
            or self.expert_bias_update_steps < 0
            or self.expert_bias_decay_end_step < 0
            or not 0.0 <= self.expert_bias_deadband < 1.0
            or self.expert_specialization_steps < 0
            or self.output_core_margin < 0
            or self.stream_core_margin < 0
            or self.expert_specialization_start_step < 0
            or not 0 < self.router_temperature_end <= 1.0
            or self.router_temperature_steps < 0
        ):
            raise ValueError("router exploration controls are invalid")
        if (
            self.expert_bias_decay_end_step
            and self.expert_bias_decay_end_step < self.expert_bias_update_steps
        ):
            raise ValueError("expert bias decay must end after bias updates")
        if self.output_experts and self.output_entropy_target > math.log(
            self.output_top_k
        ):
            raise ValueError("output entropy target exceeds top-k entropy")
        if self.stream_experts and self.stream_entropy_target > math.log(
            self.stream_top_k
        ):
            raise ValueError("stream entropy target exceeds top-k entropy")
        if self.output_core_experts and not (
            0 < self.output_core_experts < self.output_top_k
        ):
            raise ValueError("output core experts must be inside output top-k")
        if self.stream_core_experts and not (
            0 < self.stream_core_experts < self.stream_top_k
        ):
            raise ValueError("stream core experts must be inside stream top-k")
        self._routing_balance_terms: list[torch.Tensor] = []
        self._routing_specialization_terms: list[torch.Tensor] = []
        self._routing_entropies: dict[str, list[torch.Tensor]] = {
            "kv": [], "qo": [],
        }
        self._routing_margins: dict[str, list[torch.Tensor]] = {
            "kv": [], "qo": [],
        }
        self._routing_record_enabled = True
        self._routing_population_records: list[
            tuple[str, int, int | None, torch.Tensor, torch.Tensor]
        ] = []
        if self.enable_qo and self.stream_rank <= 0:
            raise ValueError("stream_rank must be positive when Q/O is enabled")
        self.style_norm = (
            nn.LayerNorm(style_dim) if bool(normalize_style) else nn.Identity()
        )
        self.context_norm = nn.LayerNorm(context_dim)
        self.style_key = nn.Linear(style_dim, hidden_dim, bias=False)
        self.style_value = nn.Linear(style_dim, hidden_dim, bias=False)
        # Adding one block vector to every key is exactly cancelled by the
        # attention softmax.  Keep the option only for loading older models.
        self.block_embedding = (
            nn.Embedding(blocks, hidden_dim) if bool(use_block_embedding) else None
        )
        self.context_query = nn.ModuleList(
            nn.Linear(context_dim, hidden_dim, bias=False) for _ in range(blocks)
        )
        self.output_norm = (
            nn.LayerNorm(hidden_dim) if bool(normalize_attended) else nn.Identity()
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = _SwiGLU(hidden_dim, ff_dim)
        self.extra_ff_norm = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(int(ff_layers) - 1)
        )
        self.extra_ff = nn.ModuleList(
            _SwiGLU(hidden_dim, ff_dim) for _ in range(int(ff_layers) - 1)
        )
        if self.output_experts:
            if not self.output_rank:
                raise ValueError("output experts require a positive expert rank")
            self.output_router_queries = nn.Parameter(
                torch.empty(blocks, 2, hidden_dim)
            )
            self.output_router_norm = nn.LayerNorm(hidden_dim)
            self.output_routers = nn.ModuleList(
                nn.ModuleList(
                    nn.Linear(hidden_dim, self.output_experts, bias=True)
                    for _ in range(2)
                )
                for _ in range(blocks)
            )
            self.output_expert_down = nn.Parameter(torch.empty(
                blocks, 2, self.output_experts, self.output_rank, hidden_dim
            ))
            self.output_expert_up = nn.Parameter(torch.empty(
                blocks, 2, self.output_experts, self.output_rank, output_dim
            ))
            self.register_buffer(
                "output_expert_usage",
                torch.full(
                    (blocks, 2, self.output_experts),
                    1.0 / self.output_experts,
                ),
            )
            self.register_buffer(
                "output_expert_load",
                torch.full(
                    (blocks, 2, self.output_experts),
                    self.output_top_k / self.output_experts,
                ),
            )
            self.register_buffer(
                "output_expert_selection_bias",
                torch.zeros(blocks, 2, self.output_experts),
            )
            self.output_head = nn.ModuleList()
        elif self.output_rank:
            self.output_head = nn.ModuleList(
                _LowRankKVResidualHead(
                    hidden_dim, output_dim, self.output_rank, output_init_scale
                )
                for _ in range(blocks)
            )
        else:
            self.output_head = nn.ModuleList(
                nn.Linear(hidden_dim, output_dim * 2, bias=False)
                for _ in range(blocks)
            )
        self.log_gain = nn.Parameter(torch.zeros(blocks, 2))
        if self.enable_qo:
            # Two learned queries per block read the full visual style memory:
            # one conditions Q and the other conditions O. The resulting code
            # gates a block-local low-rank projection of the native projection
            # input, preserving token dependence without training the base DiT.
            self.stream_style_queries = nn.Parameter(
                torch.empty(blocks, 2, hidden_dim)
            )
            self.stream_code_norm = nn.LayerNorm(hidden_dim)
            if self.stream_experts:
                self.stream_style_key = nn.ModuleList(
                    nn.Linear(style_dim, hidden_dim, bias=False) for _ in range(2)
                )
                self.stream_style_value = nn.ModuleList(
                    nn.Linear(style_dim, hidden_dim, bias=False) for _ in range(2)
                )
                self.stream_routers = nn.ModuleList(
                    nn.ModuleList(
                        nn.Linear(hidden_dim, self.stream_experts, bias=True)
                        for _ in range(2)
                    )
                    for _ in range(blocks)
                )
                self.stream_channel_gates = nn.ModuleList(
                    nn.ModuleList(
                        nn.Linear(
                            hidden_dim,
                            self.stream_experts * self.stream_rank,
                            bias=True,
                        )
                        for _ in range(2)
                    )
                    for _ in range(blocks)
                )
                self.stream_gain = nn.ModuleList(
                    nn.ModuleList(
                        nn.Linear(hidden_dim, 1, bias=True) for _ in range(2)
                    )
                    for _ in range(blocks)
                )
                self.stream_expert_down = nn.Parameter(torch.empty(
                    blocks, 2, self.stream_experts, self.stream_rank, stream_dim
                ))
                self.stream_expert_up = nn.Parameter(torch.empty(
                    blocks, 2, self.stream_experts, self.stream_rank, stream_dim
                ))
                self.register_buffer(
                    "stream_expert_usage",
                    torch.full(
                        (blocks, 2, self.stream_experts),
                        1.0 / self.stream_experts,
                    ),
                )
                self.register_buffer(
                    "stream_expert_load",
                    torch.full(
                        (blocks, 2, self.stream_experts),
                        self.stream_top_k / self.stream_experts,
                    ),
                )
                self.register_buffer(
                    "stream_expert_selection_bias",
                    torch.zeros(blocks, 2, self.stream_experts),
                )
            else:
                self.stream_gates = nn.ModuleList(
                    nn.ModuleList(
                        nn.Linear(hidden_dim, stream_rank, bias=True)
                        for _ in range(2)
                    )
                    for _ in range(blocks)
                )
                self.stream_down = nn.ModuleList(
                    nn.ModuleList(
                        nn.Linear(stream_dim, stream_rank, bias=False)
                        for _ in range(2)
                    )
                    for _ in range(blocks)
                )
                self.stream_up = nn.ModuleList(
                    nn.ModuleList(
                        nn.Linear(stream_rank, stream_dim, bias=False)
                        for _ in range(2)
                    )
                    for _ in range(blocks)
                )
        self.reset_parameters(float(output_init_scale))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Expert checkpoints created before population load and selection bias
        # existed remain usable for diagnostics. Fresh balanced runs still save
        # and restore these buffers normally.
        for name in (
            "output_expert_load",
            "output_expert_selection_bias",
            "stream_expert_load",
            "stream_expert_selection_bias",
        ):
            if hasattr(self, name) and prefix + name not in state_dict:
                state_dict[prefix + name] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_parameters(self, output_init_scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        if self.block_embedding is not None:
            nn.init.normal_(self.block_embedding.weight, std=self.hidden_dim**-0.5)
        if self.enable_qo:
            nn.init.normal_(
                self.stream_style_queries, std=self.hidden_dim**-0.5
            )
            if self.stream_experts:
                nn.init.normal_(
                    self.stream_expert_down, std=self.stream_dim**-0.5
                )
                nn.init.normal_(
                    self.stream_expert_up,
                    std=self.stream_rank**-0.5 * output_init_scale,
                )
                q_bias = math.log(0.25 / (2.0 - 0.25))
                for block in range(self.blocks):
                    for kind in range(2):
                        nn.init.zeros_(self.stream_routers[block][kind].bias)
                        with torch.no_grad():
                            self.stream_routers[block][kind].weight.mul_(
                                self.router_init_scale
                            )
                        nn.init.zeros_(self.stream_channel_gates[block][kind].bias)
                        nn.init.zeros_(self.stream_gain[block][kind].weight)
                        nn.init.constant_(
                            self.stream_gain[block][kind].bias,
                            q_bias if kind == 0 else 0.0,
                        )
            else:
                for block_gates in self.stream_gates:
                    for gate in block_gates:
                        nn.init.zeros_(gate.bias)
                for block_heads in self.stream_up:
                    for head in block_heads:
                        with torch.no_grad():
                            head.weight.mul_(output_init_scale)
        if self.output_experts:
            nn.init.normal_(self.output_router_queries, std=self.hidden_dim**-0.5)
            nn.init.normal_(
                self.output_expert_down, std=self.hidden_dim**-0.5
            )
            nn.init.normal_(
                self.output_expert_up,
                std=self.output_rank**-0.5 * output_init_scale,
            )
            for block_routers in self.output_routers:
                for router in block_routers:
                    nn.init.zeros_(router.bias)
                    with torch.no_grad():
                        router.weight.mul_(self.router_init_scale)
        elif not self.output_rank:
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
        key = self.style_key(style)
        if self.block_embedding is not None:
            block_code = self.block_embedding.weight[block].to(style.dtype)
            key = key + block_code[None, None]
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
        for norm, ff in zip(self.extra_ff_norm, self.extra_ff, strict=True):
            hidden = hidden + ff(norm(hidden))
        if self.output_experts:
            router_query = self.output_router_queries[block].to(style.dtype)
            router_query = router_query[None].expand(batch, -1, -1)
            router_query = router_query.reshape(
                batch, 2, self.heads, self.head_dim
            ).transpose(1, 2)
            router_code = F.scaled_dot_product_attention(
                router_query, key, value
            ).transpose(1, 2).reshape(batch, 2, self.hidden_dim)
            router_code = self.output_router_norm(router_code)
            logits = torch.stack([
                self.output_routers[block][kind](router_code[:, kind])
                for kind in range(2)
            ], dim=1)
            indices, weights, sparse, dense, selected = self._sparse_router(
                logits,
                self.output_top_k,
                self.output_expert_selection_bias[block],
            )
            self._record_routing(
                "kv",
                self.output_expert_usage,
                self.output_expert_load,
                self.output_expert_selection_bias,
                block,
                sparse,
                dense,
                selected,
                self.output_top_k,
                logits,
            )
            down_bank = self.output_expert_down[block]
            up_bank = self.output_expert_up[block]
            kind_rows = torch.arange(2, device=hidden.device)[None, :, None]
            selected_down = down_bank[kind_rows, indices]
            selected_up = up_bank[kind_rows, indices]
            low_rank = torch.einsum(
                "bnh,bekrh->beknr", hidden, selected_down
            )
            output = torch.einsum(
                "beknr,bekro,bek->bneo", low_rank, selected_up, weights
            )
        else:
            output = self.output_head[block](hidden)
        if not self.output_rank and not self.output_experts:
            output = output.reshape(batch, text_tokens, 2, self.output_dim)
        output = output.transpose(1, 2)
        gain = self.log_gain[block].float().clamp(-4.0, 4.0).exp().to(output.dtype)
        return output * gain[None, :, None, None]

    def prepare_stream_codes(self, style_memory: torch.Tensor) -> torch.Tensor:
        """Read reference memory once for all block-local Q/O adapters."""
        if not self.enable_qo:
            raise RuntimeError("Q/O stream modulation is disabled")
        style = self.style_norm(style_memory)
        batch = int(style.shape[0])
        if self.stream_experts:
            rows = []
            for kind in range(2):
                if not self.stream_kind_enabled(kind):
                    rows.append(
                        style.new_zeros(
                            batch, self.blocks, self.heads, self.head_dim
                        )
                    )
                    continue
                queries = self.stream_style_queries[:, kind][None].expand(
                    batch, -1, -1
                ).reshape(
                    batch, self.blocks, self.heads, self.head_dim
                ).transpose(1, 2)
                key = self.stream_style_key[kind](style).reshape(
                    batch, style.shape[1], self.heads, self.head_dim
                ).transpose(1, 2)
                value = self.stream_style_value[kind](style).reshape(
                    batch, style.shape[1], self.heads, self.head_dim
                ).transpose(1, 2)
                code = F.scaled_dot_product_attention(queries, key, value)
                rows.append(code.transpose(1, 2))
            codes = torch.stack(rows, dim=2).reshape(
                batch, self.blocks, 2, self.hidden_dim
            )
        else:
            key = self.style_key(style)
            value = self.style_value(style)
            queries = self.stream_style_queries.reshape(
                self.blocks * 2, self.hidden_dim
            )[None].expand(batch, -1, -1)
            queries = queries.reshape(
                batch, self.blocks * 2, self.heads, self.head_dim
            ).transpose(1, 2)
            key = key.reshape(
                batch, style.shape[1], self.heads, self.head_dim
            ).transpose(1, 2)
            value = value.reshape(
                batch, style.shape[1], self.heads, self.head_dim
            ).transpose(1, 2)
            codes = F.scaled_dot_product_attention(queries, key, value)
            codes = codes.transpose(1, 2).reshape(
                batch, self.blocks, 2, self.hidden_dim
            )
        return self.stream_code_norm(codes)

    def stream_kind_enabled(self, kind: int) -> bool:
        if int(kind) == 0:
            return self.enable_q
        if int(kind) == 1:
            return self.enable_o
        raise ValueError("stream kind must be Q=0 or O=1")

    def stream_delta(
        self,
        stream_input: torch.Tensor,
        stream_codes: torch.Tensor,
        block_index: int,
        kind: int,
    ) -> torch.Tensor:
        """Apply a reference-gated low-rank delta to Q (0) or O (1)."""
        if not self.enable_qo:
            raise RuntimeError("Q/O stream modulation is disabled")
        if not self.stream_kind_enabled(kind):
            return torch.zeros_like(stream_input)
        if stream_input.shape[-1] != self.stream_dim:
            raise ValueError(
                f"Expected stream width {self.stream_dim}, "
                f"got {stream_input.shape[-1]}"
            )
        code = stream_codes[:, block_index, kind]
        if self.stream_experts:
            logits = self.stream_routers[block_index][kind](code)
            indices, weights, sparse, dense, selected = self._sparse_router(
                logits[:, None],
                self.stream_top_k,
                self.stream_expert_selection_bias[
                    block_index, kind : kind + 1
                ],
            )
            indices = indices[:, 0]
            weights = weights[:, 0]
            self._record_routing(
                "qo",
                self.stream_expert_usage,
                self.stream_expert_load,
                self.stream_expert_selection_bias,
                block_index,
                sparse,
                dense,
                selected,
                self.stream_top_k,
                logits[:, None],
                usage_kind=kind,
            )
            channels = torch.tanh(
                self.stream_channel_gates[block_index][kind](code).reshape(
                    len(code), self.stream_experts, self.stream_rank
                )
            )
            batch_rows = torch.arange(len(code), device=code.device)[:, None]
            selected_channels = channels[batch_rows, indices]
            selected_down = self.stream_expert_down[
                block_index, kind
            ][indices]
            selected_up = self.stream_expert_up[
                block_index, kind
            ][indices]
            hidden = torch.einsum(
                "bnc,bkrc->bknr", stream_input, selected_down
            ) * selected_channels[:, :, None]
            delta = torch.einsum(
                "bknr,bkro,bk->bno", hidden, selected_up, weights
            )
            gain = 2.0 * torch.sigmoid(
                self.stream_gain[block_index][kind](code).float()
            ).to(delta.dtype)
            return delta * gain[:, None]
        gate = torch.tanh(self.stream_gates[block_index][kind](code))
        hidden = self.stream_down[block_index][kind](stream_input)
        return self.stream_up[block_index][kind](hidden * gate[:, None, :])

    def _sparse_router(
        self,
        logits: torch.Tensor,
        top_k: int,
        selection_bias: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        temperature = self._router_temperature()
        tempered_logits = logits.float() / temperature
        dense = tempered_logits.softmax(dim=-1).to(logits.dtype)
        selection_logits = tempered_logits
        if selection_bias is not None:
            selection_logits = selection_logits + (
                self._selection_bias_scale() * selection_bias.float()
            )
        if self.training and self.router_jitter and self.router_jitter_steps:
            progress = min(1.0, self._routing_step / self.router_jitter_steps)
            jitter = self.router_jitter * (1.0 - progress)
            if jitter:
                selection_logits = selection_logits + torch.randn_like(
                    selection_logits
                ) * jitter
        indices = selection_logits.topk(int(top_k), dim=-1).indices
        selected_logits = tempered_logits.gather(-1, indices)
        weights = selected_logits.float().softmax(dim=-1).to(logits.dtype)
        sparse = torch.zeros_like(logits).scatter(-1, indices, weights)
        selected = torch.zeros_like(logits).scatter(
            -1, indices, torch.ones_like(weights)
        )
        return indices, weights, sparse, dense, selected

    def set_routing_step(self, step: int) -> None:
        self._routing_step = max(0, int(step))

    def set_routing_recording(self, enabled: bool) -> None:
        """Include only genuine routed training examples in population stats."""

        self._routing_record_enabled = bool(enabled)

    def _selection_bias_scale(self) -> float:
        if not self.expert_bias_decay_end_step:
            return 1.0
        if self._routing_step <= self.expert_bias_update_steps:
            return 1.0
        if self._routing_step >= self.expert_bias_decay_end_step:
            return 0.0
        width = self.expert_bias_decay_end_step - self.expert_bias_update_steps
        return 1.0 - (
            self._routing_step - self.expert_bias_update_steps
        ) / max(1, width)

    def _router_temperature(self) -> float:
        if not self.router_temperature_steps:
            return self.router_temperature_end
        progress = min(1.0, self._routing_step / self.router_temperature_steps)
        return 1.0 + progress * (self.router_temperature_end - 1.0)

    def _core_margin_target(self, name: str) -> float:
        target = (
            self.output_core_margin if name == "kv" else self.stream_core_margin
        )
        elapsed = max(0, self._routing_step - self.expert_specialization_start_step)
        progress = min(1.0, elapsed / max(1, self.expert_specialization_steps))
        return target * progress

    def _entropy_cap(self, name: str, top_k: int) -> float:
        target = (
            self.output_entropy_target
            if name == "kv"
            else self.stream_entropy_target
        )
        maximum = math.log(max(1, int(top_k)))
        if target < 0:
            return maximum
        progress = min(
            1.0,
            self._routing_step / max(1, self.expert_specialization_steps),
        )
        return maximum + progress * (target - maximum)

    def reset_routing_records(self) -> None:
        self._routing_balance_terms.clear()
        self._routing_specialization_terms.clear()
        for values in self._routing_entropies.values():
            values.clear()
        for values in self._routing_margins.values():
            values.clear()

    def _record_routing(
        self,
        name: str,
        usage: torch.Tensor,
        load: torch.Tensor,
        selection_bias: torch.Tensor,
        block: int,
        probabilities: torch.Tensor,
        dense_probabilities: torch.Tensor,
        selections: torch.Tensor,
        top_k: int,
        router_logits: torch.Tensor,
        *,
        usage_kind: int | None = None,
    ) -> None:
        if not self.training or not self._routing_record_enabled:
            return
        current = probabilities.float().mean(dim=0)
        current_dense = dense_probabilities.float().mean(dim=0)
        current_load = selections.float().mean(dim=0)
        usage_rows = usage[block]
        load_rows = load[block]
        bias_rows = selection_bias[block]
        if usage_kind is not None:
            usage_rows = usage_rows[usage_kind : usage_kind + 1]
            load_rows = load_rows[usage_kind : usage_kind + 1]
            bias_rows = bias_rows[usage_kind : usage_kind + 1]
        experts = probabilities.shape[-1]
        importance_threshold = self.expert_balance_cap / experts
        load_threshold = self.expert_balance_cap * float(top_k) / experts
        importance_overload = F.relu(
            usage_rows.detach().float() - importance_threshold
        )
        load_overload = F.relu(
            load_rows.detach().float() - load_threshold
        )
        # The dispatched output stays hard top-k, while every router logit gets
        # gradient through the pre-top-k dense probability distribution.
        overload = importance_overload + load_overload / max(1, int(top_k))
        self._routing_balance_terms.append(
            (current_dense * overload).sum(dim=-1).mean()
        )
        entropy = -(
            probabilities.float().clamp_min(1e-8).log()
            * probabilities.float()
        ).sum(dim=-1).mean()
        self._routing_entropies[name].append(entropy)
        core_experts = (
            self.output_core_experts if name == "kv" else self.stream_core_experts
        )
        if core_experts:
            ordered = router_logits.float().sort(dim=-1, descending=True).values
            margin = (
                ordered[..., core_experts - 1] - ordered[..., core_experts]
            ).mean()
            margin_target = self._core_margin_target(name)
            self._routing_margins[name].append(margin)
            self._routing_specialization_terms.append(
                F.relu(margin_target - margin).square()
            )
        else:
            entropy_cap = self._entropy_cap(name, top_k)
            self._routing_specialization_terms.append(
                F.relu(entropy - entropy_cap).square()
            )
        if self.expert_bias_population_update:
            self._routing_population_records.append(
                (
                    name,
                    int(block),
                    usage_kind,
                    current.detach(),
                    current_load.detach(),
                )
            )
            return
        with torch.no_grad():
            usage_rows.mul_(self.expert_usage_decay).add_(
                current, alpha=1.0 - self.expert_usage_decay
            )
            load_rows.mul_(self.expert_usage_decay).add_(
                current_load, alpha=1.0 - self.expert_usage_decay
            )
            if (
                self.expert_bias_update_rate
                and (
                    not self.expert_bias_update_steps
                    or self._routing_step <= self.expert_bias_update_steps
                )
            ):
                excess = F.relu(load_rows.float() - load_threshold)
                bias_rows.mul_(self.expert_bias_decay).add_(
                    excess, alpha=-self.expert_bias_update_rate
                )
                bias_rows.sub_(bias_rows.mean(dim=-1, keepdim=True))
                bias_rows.clamp_(-self.expert_bias_max, self.expert_bias_max)

    @torch.no_grad()
    def apply_routing_population_update(self) -> dict[str, torch.Tensor]:
        """Update selection-only biases once from an optimizer's artist set."""

        zero = next(self.parameters()).new_zeros((), dtype=torch.float32)
        if not self.expert_bias_population_update:
            return {}
        grouped: dict[
            tuple[str, int, int | None],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = defaultdict(list)
        for name, block, kind, usage, load in self._routing_population_records:
            grouped[(name, block, kind)].append((usage, load))
        population_sizes: dict[str, list[int]] = defaultdict(list)
        for (name, block, kind), values in grouped.items():
            current_usage = torch.stack([value[0] for value in values]).mean(0)
            current_load = torch.stack([value[1] for value in values]).mean(0)
            if name == "kv":
                usage_rows = self.output_expert_usage[block]
                load_rows = self.output_expert_load[block]
                bias_rows = self.output_expert_selection_bias[block]
                top_k = self.output_top_k
                experts = self.output_experts
            else:
                assert kind is not None
                usage_rows = self.stream_expert_usage[block, kind : kind + 1]
                load_rows = self.stream_expert_load[block, kind : kind + 1]
                bias_rows = self.stream_expert_selection_bias[
                    block, kind : kind + 1
                ]
                top_k = self.stream_top_k
                experts = self.stream_experts
            usage_rows.mul_(self.expert_usage_decay).add_(
                current_usage, alpha=1.0 - self.expert_usage_decay
            )
            load_rows.mul_(self.expert_usage_decay).add_(
                current_load, alpha=1.0 - self.expert_usage_decay
            )
            if self.expert_bias_update_rate:
                target = float(top_k) / experts
                error = target - load_rows.float()
                direction = error.sign()
                if self.expert_bias_deadband:
                    direction = direction * (
                        error.abs() > self.expert_bias_deadband * target
                    )
                bias_rows.add_(direction, alpha=self.expert_bias_update_rate)
                bias_rows.sub_(bias_rows.mean(dim=-1, keepdim=True))
                bias_rows.clamp_(-self.expert_bias_max, self.expert_bias_max)
            population_sizes[name].append(len(values))
        self._routing_population_records.clear()

        metrics: dict[str, torch.Tensor] = {}
        for name, load, top_k, experts in (
            ("kv", getattr(self, "output_expert_load", None), self.output_top_k, self.output_experts),
            ("qo", getattr(self, "stream_expert_load", None), self.stream_top_k, self.stream_experts),
        ):
            if load is None or not experts:
                continue
            if name == "qo" and self.enable_qo and not self.enable_q:
                load = load[:, 1:2]
            target = float(top_k) / experts
            router_max = load.float().amax(dim=-1).flatten()
            metrics[f"{name}_population_groups"] = zero.new_tensor(
                sum(population_sizes[name]) / max(1, len(population_sizes[name]))
            )
            metrics[f"{name}_ema_max_load"] = router_max.max()
            metrics[f"{name}_ema_p95_max_load"] = torch.quantile(
                router_max, 0.95
            )
            metrics[f"{name}_max_violation"] = router_max.max() / target - 1.0
            metrics[f"{name}_overload_router_fraction"] = (
                router_max > self.expert_balance_cap * target
            ).float().mean()
        return metrics

    def routing_auxiliary(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        zero = next(self.parameters()).new_zeros((), dtype=torch.float32)
        balance = (
            torch.stack(self._routing_balance_terms).mean()
            if self._routing_balance_terms else zero
        )
        specialization = (
            torch.stack(self._routing_specialization_terms).mean()
            if self._routing_specialization_terms else zero
        )
        metrics = {
            "expert_balance_loss": balance.detach(),
            "expert_specialization_loss": specialization.detach(),
        }
        for name, values in self._routing_entropies.items():
            entropy = torch.stack(values).mean() if values else zero
            metrics[f"{name}_router_entropy"] = entropy.detach()
            metrics[f"{name}_effective_experts"] = entropy.detach().exp()
            top_k = self.output_top_k if name == "kv" else self.stream_top_k
            metrics[f"{name}_entropy_cap"] = zero.new_tensor(
                self._entropy_cap(name, top_k)
            )
            margins = self._routing_margins[name]
            margin = torch.stack(margins).mean() if margins else zero
            metrics[f"{name}_core_logit_margin"] = margin.detach()
            metrics[f"{name}_core_margin_target"] = zero.new_tensor(
                self._core_margin_target(name)
            )
        metrics["router_temperature"] = zero.new_tensor(
            self._router_temperature()
        )
        if self.output_experts:
            metrics["kv_expert_max_usage"] = self.output_expert_usage.max().detach()
            metrics["kv_expert_max_load"] = self.output_expert_load.max().detach()
            metrics["kv_expert_bias_span"] = (
                self.output_expert_selection_bias.amax(dim=-1)
                - self.output_expert_selection_bias.amin(dim=-1)
            ).max().detach() * self._selection_bias_scale()
        if self.enable_qo and self.stream_experts:
            metrics["qo_expert_max_usage"] = self.stream_expert_usage.max().detach()
            metrics["qo_expert_max_load"] = self.stream_expert_load.max().detach()
            metrics["qo_expert_bias_span"] = (
                self.stream_expert_selection_bias.amax(dim=-1)
                - self.stream_expert_selection_bias.amin(dim=-1)
            ).max().detach() * self._selection_bias_scale()
        self.reset_routing_records()
        return balance, specialization, metrics


class _OperatorCrossBlock(nn.Module):
    """Let operator queries read the complete typed reference memory."""

    def __init__(
        self, dim: int, heads: int, ff_dim: int, *, normalize_memory: bool = True
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("operator hidden dimension must divide heads")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim) if normalize_memory else nn.Identity()
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
        normalize_style: bool = True,
        normalize_memory: bool = True,
        enable_qo: bool = False,
        stream_dim: int = 2048,
        stream_rank: int = 32,
        stream_init_scale: float = 1e-6,
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
        self.heads = int(heads)
        self.head_dim = int(hidden_dim // heads)
        self.operator_rank = int(operator_rank)
        self.minimum_sigma = float(minimum_sigma)
        self.maximum_sigma = float(maximum_sigma)
        self.enable_qo = bool(enable_qo)
        self.stream_dim = int(stream_dim)
        self.stream_rank = int(stream_rank)
        if self.enable_qo and self.stream_rank <= 0:
            raise ValueError("stream_rank must be positive when Q/O is enabled")
        if stream_init_scale < 0:
            raise ValueError("stream_init_scale must be non-negative")
        self.stream_init_scale = float(stream_init_scale)
        self.style_norm = (
            nn.LayerNorm(style_dim) if bool(normalize_style) else nn.Identity()
        )
        self.style_input = nn.Linear(style_dim, hidden_dim, bias=False)
        # [block, K/V, down/up, rank, hidden].  The identities are explicit;
        # no mean pooling or shared rank token is used.
        self.operator_queries = nn.Parameter(
            torch.empty(blocks, 2, 2, operator_rank, hidden_dim)
        )
        self.reader = nn.ModuleList(
            _OperatorCrossBlock(
                hidden_dim, heads, ff_dim, normalize_memory=normalize_memory
            )
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
        if self.enable_qo:
            self.stream_style_queries = nn.Parameter(
                torch.empty(blocks, 2, hidden_dim)
            )
            self.stream_key = nn.Linear(style_dim, hidden_dim, bias=False)
            self.stream_value = nn.Linear(style_dim, hidden_dim, bias=False)
            self.stream_code_norm = nn.LayerNorm(hidden_dim)
            self.stream_gates = nn.ModuleList(
                nn.ModuleList(
                    nn.Linear(hidden_dim, stream_rank, bias=True)
                    for _ in range(2)
                )
                for _ in range(blocks)
            )
            self.stream_down = nn.ModuleList(
                nn.ModuleList(
                    nn.Linear(stream_dim, stream_rank, bias=False)
                    for _ in range(2)
                )
                for _ in range(blocks)
            )
            self.stream_up = nn.ModuleList(
                nn.ModuleList(
                    nn.Linear(stream_rank, stream_dim, bias=False)
                    for _ in range(2)
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
        if self.enable_qo:
            nn.init.normal_(
                self.stream_style_queries, std=self.hidden_dim**-0.5
            )
            for block_gates in self.stream_gates:
                for gate in block_gates:
                    nn.init.zeros_(gate.bias)
            for block_heads in self.stream_up:
                for head in block_heads:
                    with torch.no_grad():
                        head.weight.mul_(self.stream_init_scale)
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

    def prepare_kv_factors(
        self, style_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate every block's K/V LoRA once for reuse across text tokens."""
        down_rows = []
        up_rows = []
        for block in range(self.blocks):
            down, up, sigma = self._operator(style_memory, block)
            down_rows.append(down)
            up_rows.append(up.transpose(-1, -2) * sigma[:, :, None, :])
        return torch.stack(down_rows, dim=1), torch.stack(up_rows, dim=1)

    def apply_prepared_kv(
        self,
        text_context: torch.Tensor,
        down: torch.Tensor,
        up: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        return apply_kv_factors(
            text_context, down[:, block_index], up[:, block_index]
        )

    def prepare_stream_codes(self, style_memory: torch.Tensor) -> torch.Tensor:
        if not self.enable_qo:
            raise RuntimeError("Q/O stream modulation is disabled")
        style = self.style_norm(style_memory)
        key = self.stream_key(style)
        value = self.stream_value(style)
        batch = int(style.shape[0])
        queries = self.stream_style_queries.reshape(
            self.blocks * 2, self.hidden_dim
        )[None].expand(batch, -1, -1)
        queries = queries.reshape(
            batch, self.blocks * 2, self.heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch, style.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch, style.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)
        codes = F.scaled_dot_product_attention(queries, key, value)
        codes = codes.transpose(1, 2).reshape(
            batch, self.blocks, 2, self.hidden_dim
        )
        return self.stream_code_norm(codes)

    def stream_delta(
        self,
        stream_input: torch.Tensor,
        stream_codes: torch.Tensor,
        block_index: int,
        kind: int,
    ) -> torch.Tensor:
        if not self.enable_qo:
            raise RuntimeError("Q/O stream modulation is disabled")
        code = stream_codes[:, block_index, kind]
        gate = torch.tanh(self.stream_gates[block_index][kind](code))
        hidden = self.stream_down[block_index][kind](stream_input)
        return self.stream_up[block_index][kind](hidden * gate[:, None, :])

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


def _build_direct_delta_generator(
    model_cfg: dict[str, Any],
    *,
    style_dim: int,
    context_dim: int,
    output_dim: int,
    blocks: int,
) -> nn.Module:
    effective = dict(model_cfg)
    architecture = str(
        effective.pop("architecture", "direct_cross_attention")
    )
    model_type: type[nn.Module]
    if architecture == "direct_cross_attention":
        model_type = ReferenceConditionedKVActivationGenerator
    elif architecture == "low_rank_kvoq_operator":
        model_type = ReferenceConditionedLowRankKVOperator
    else:
        raise ValueError(
            f"Unsupported direct-delta architecture: {architecture}"
        )
    return model_type(
        style_dim=style_dim,
        context_dim=context_dim,
        output_dim=output_dim,
        blocks=blocks,
        **effective,
    )


def _resolved_experiment_config(
    config: dict[str, Any], config_key: str
) -> dict[str, Any]:
    """Resolve a small experiment override without duplicating its data contract."""
    override = copy.deepcopy(config[config_key])
    base_key = override.pop("extends_config_key", None)
    if base_key is None:
        return override
    base = _resolved_experiment_config(config, str(base_key))
    replace_sections = [
        str(value) for value in override.pop("replace_sections", [])
    ]
    for section in replace_sections:
        if section in override:
            base[section] = override.pop(section)

    def merge(target: dict[str, Any], values: dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(base, override)
    return base


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
        *,
        queries_normalized: bool = False,
    ) -> torch.Tensor:
        if not queries_normalized:
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


def _prediction_population_metrics(
    prediction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure whether artist predictions occupy distinct directions."""

    flat = prediction.float().flatten(1)
    population_mean = flat.mean(dim=0, keepdim=True)
    total_energy = flat.square().mean().clamp_min(1e-12)
    common_energy = population_mean.square().mean()
    centered_energy = (flat - population_mean).square().mean()
    normalized = F.normalize(flat, dim=-1)
    similarities = normalized @ normalized.T
    artists = len(flat)
    if artists > 1:
        pairwise = (similarities.sum() - similarities.diagonal().sum()) / (
            artists * (artists - 1)
        )
    else:
        pairwise = similarities.new_ones(())
    return {
        "artist_variance_fraction": centered_energy / total_energy,
        "common_direction_occupancy": common_energy / total_energy,
        "artist_pairwise_cosine": pairwise,
    }


def _excess_common_direction_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Penalize pairwise collapse beyond the teacher's shared geometry."""

    student_flat = F.normalize(student.float().flatten(1), dim=-1)
    teacher_flat = F.normalize(teacher.float().flatten(1), dim=-1)
    artists = len(student_flat)
    if artists < 2:
        return student_flat.new_zeros(())
    student_similarity = student_flat @ student_flat.T
    teacher_similarity = teacher_flat @ teacher_flat.T
    off_diagonal = ~torch.eye(
        artists, device=student.device, dtype=torch.bool
    )
    return F.relu(
        student_similarity[off_diagonal]
        - teacher_similarity[off_diagonal].detach()
    ).mean()


def _final_effect_retrieval_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Make each predicted final effect identify its matching teacher effect.

    Every row is evaluated under the same content, noise, and timestep. The
    off-diagonal rows are therefore genuine wrong-style negatives rather than
    content negatives. Neither tensor is centered and no batch mean is used.
    """

    batch = int(student.shape[0])
    zero = student.float().new_zeros(())
    if batch < 2:
        return zero, {
            "retrieval_loss": zero.detach(),
            "retrieval_accuracy": zero.detach(),
            "correct_cosine": zero.detach(),
            "hardest_wrong_cosine": zero.detach(),
            "correct_minus_hardest_wrong_cosine": zero.detach(),
        }
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student_unit = F.normalize(student.float().flatten(1), dim=-1)
    teacher_unit = F.normalize(teacher.detach().float().flatten(1), dim=-1)
    cosine = student_unit @ teacher_unit.T
    labels = torch.arange(batch, device=student.device)
    logits = cosine / float(temperature)
    loss = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )
    correct = cosine.diagonal()
    wrong = cosine.masked_fill(
        torch.eye(batch, device=student.device, dtype=torch.bool),
        -torch.inf,
    )
    hardest_wrong = wrong.max(dim=1).values
    accuracy = (cosine.argmax(dim=1) == labels).float().mean()
    return loss, {
        "retrieval_loss": loss.detach(),
        "retrieval_accuracy": accuracy.detach(),
        "correct_cosine": correct.mean().detach(),
        "hardest_wrong_cosine": hardest_wrong.mean().detach(),
        "correct_minus_hardest_wrong_cosine": (
            correct - hardest_wrong
        ).mean().detach(),
    }


def _final_effect_constraints(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    common_cap: float,
    rms_lower: float,
    rms_upper: float,
    common_cap_weight: float = 1.0,
    rms_band_weight: float = 1.0,
    rms_floor: float = 1e-4,
    pair_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Absolute common-direction cap and an output-RMS hinge band.

    This deliberately never centers either tensor.  In particular, no batch
    mean participates in the loss or changes the direction being learned.
    """

    student_f = student.float()
    teacher_f = teacher.detach().float()
    batch = int(student_f.shape[0])
    zero = student_f.new_zeros(())
    positive_pairwise = zero
    relative_common = zero
    if batch > 1:
        student_unit = F.normalize(student_f.flatten(1), dim=-1)
        teacher_unit = F.normalize(teacher_f.flatten(1), dim=-1)
        off_diagonal = ~torch.eye(batch, device=student.device, dtype=torch.bool)
        if pair_mask is not None:
            if pair_mask.shape != (batch, batch):
                raise ValueError("pair_mask must have shape [batch, batch]")
            off_diagonal &= pair_mask.to(device=student.device, dtype=torch.bool)
        student_cosine = (student_unit @ student_unit.T)[off_diagonal]
        teacher_cosine = (teacher_unit @ teacher_unit.T)[off_diagonal]
        if student_cosine.numel() > 0:
            positive_pairwise = F.relu(student_cosine).mean()
            relative_common = F.relu(
                student_cosine - teacher_cosine.detach()
            ).mean()
    common_cap_loss = F.relu(positive_pairwise - float(common_cap)).square()

    dimensions = tuple(range(1, student_f.ndim))
    student_rms = student_f.square().mean(dim=dimensions).sqrt()
    teacher_rms = teacher_f.square().mean(dim=dimensions).sqrt()
    denominator = teacher_rms.clamp_min(float(rms_floor))
    ratio = student_rms / denominator
    lower_rows = teacher_rms >= float(rms_floor)
    lower_violation = F.relu(float(rms_lower) - ratio)
    if bool(lower_rows.any()):
        lower_loss = lower_violation[lower_rows].square().mean()
        lower_rate = (lower_violation[lower_rows] > 0).float().mean()
    else:
        lower_loss = zero
        lower_rate = zero
    upper_violation = F.relu(ratio - float(rms_upper))
    upper_loss = upper_violation.square().mean()
    band_loss = lower_loss + upper_loss
    weighted_common_cap = float(common_cap_weight) * common_cap_loss
    weighted_rms_band = float(rms_band_weight) * band_loss
    return weighted_common_cap + weighted_rms_band, {
        "common_cap_loss": common_cap_loss.detach(),
        "common_cap_weighted_loss": weighted_common_cap.detach(),
        "relative_common_loss": relative_common.detach(),
        "positive_pairwise_cosine": positive_pairwise.detach(),
        "rms_band_loss": band_loss.detach(),
        "rms_band_weighted_loss": weighted_rms_band.detach(),
        "rms_ratio": ratio.mean().detach(),
        "rms_lower_violation_rate": lower_rate.detach(),
        "rms_upper_violation_rate": (upper_violation > 0).float().mean().detach(),
    }


def _adapter_probe_signature(
    generator: ReferenceConditionedKVActivationGenerator,
    style_memory: torch.Tensor,
    text_context: torch.Tensor,
    stream_input: torch.Tensor,
    blocks: list[int],
    signature_width: int = 64,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Evaluate each artist adapter on the same compact K/V/Q/O probe."""

    def compact(values: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(
            values.float().flatten(1), int(signature_width)
        ).squeeze(0)

    style = style_memory[:1]
    parts = [compact(generator(style, text_context, block)) for block in blocks]
    if generator.enable_qo:
        stream_codes = generator.prepare_stream_codes(style)
        for block in blocks:
            parts.extend(
                compact(generator.stream_delta(stream_input, stream_codes, block, kind))
                for kind in range(2)
                if generator.stream_kind_enabled(kind)
            )
    signature = torch.cat(parts)
    return F.normalize(signature, dim=0) if normalize else signature


def _same_artist_signature_consistency(
    student: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_floor: float,
    rms_ratio_tolerance: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match disjoint reference views without forcing exact equality."""

    student_f = student.float().flatten()
    target_f = target.detach().float().flatten()
    cosine = F.cosine_similarity(student_f, target_f, dim=0)
    direction = F.relu(float(cosine_floor) - cosine).square()
    student_rms = student_f.square().mean().sqrt().clamp_min(1e-8)
    target_rms = target_f.square().mean().sqrt().clamp_min(1e-8)
    log_ratio = (student_rms.log() - target_rms.log()).abs()
    tolerance = math.log(float(rms_ratio_tolerance))
    magnitude = F.relu(log_ratio - tolerance).square()
    loss = direction + float(magnitude_weight) * magnitude
    return loss, {
        "same_artist_consistency_loss": loss.detach(),
        "same_artist_signature_cosine": cosine.detach(),
        "same_artist_log_rms_error": log_ratio.detach(),
    }


def _same_artist_memory_consistency(
    student: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_floor: float,
    rms_ratio_tolerance: float,
    magnitude_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match Reader memories from disjoint images of each artist."""

    student_f = student.float().flatten(1)
    target_f = target.detach().float().flatten(1)
    cosine = F.cosine_similarity(student_f, target_f, dim=1)
    direction = F.relu(float(cosine_floor) - cosine).square().mean()
    student_rms = student_f.square().mean(dim=1).sqrt().clamp_min(1e-8)
    target_rms = target_f.square().mean(dim=1).sqrt().clamp_min(1e-8)
    log_ratio = (student_rms.log() - target_rms.log()).abs()
    tolerance = math.log(float(rms_ratio_tolerance))
    magnitude = F.relu(log_ratio - tolerance).square().mean()
    loss = direction + float(magnitude_weight) * magnitude
    return loss, {
        "reader_consistency_loss": loss.detach(),
        "reader_consistency_cosine": cosine.mean().detach(),
        "reader_consistency_log_rms_error": log_ratio.mean().detach(),
    }


def _same_artist_queue_infonce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: list[torch.Tensor],
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Identify an artist across disjoint references against recent artists."""

    if not negatives:
        zero = anchor.new_zeros(())
        return zero, {
            "same_artist_contrastive_positive": zero,
            "same_artist_contrastive_hardest_negative": zero,
        }
    anchor_f = F.normalize(anchor.float().flatten(), dim=0)
    positive_f = F.normalize(positive.detach().float().flatten(), dim=0)
    negative_f = torch.stack(negatives).detach().float()
    positive_logit = (anchor_f * positive_f).sum()
    negative_logits = negative_f @ anchor_f
    logits = torch.cat([positive_logit[None], negative_logits]) / float(
        temperature
    )
    loss = F.cross_entropy(logits[None], logits.new_zeros(1, dtype=torch.long))
    return loss, {
        "same_artist_contrastive_positive": positive_logit.detach(),
        "same_artist_contrastive_hardest_negative": (
            negative_logits.max().detach()
        ),
    }


def _cross_style_queue_diversity(
    signature: torch.Tensor,
    references: list[torch.Tensor],
    *,
    cosine_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap similarity to detached signatures belonging to other artists."""

    if not references:
        zero = signature.new_zeros(())
        return zero, zero
    similarities = torch.stack(references) @ signature
    positive_cosine = F.relu(similarities).mean()
    loss = F.relu(similarities - float(cosine_cap)).square().mean()
    return loss, positive_cosine


def _population_common_occupancy(
    signature: torch.Tensor,
    references: list[torch.Tensor],
    *,
    occupancy_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Limit a detached artist population's shared direction without centering."""

    if not references:
        zero = signature.new_zeros(())
        return zero, zero
    population = torch.stack([*references, signature])
    mean = population.mean(dim=0)
    # Correct the 1/N queue dilution for the only live (current-artist) row.
    mean = mean.detach() + len(population) * (mean - mean.detach())
    occupancy = mean.float().square().sum()
    loss = F.relu(occupancy - float(occupancy_cap)).square()
    return loss, occupancy


def _whole_model_curriculum(relative_step: int) -> dict[str, float]:
    """Weights and RMS bounds for the fixed 10k functional curriculum."""

    step = max(0, int(relative_step))
    if step <= 500:
        block_weight = 1.0
        whole_weight = 0.10
        rms_lower, rms_upper = 0.40, 1.60
    elif step < 2000:
        progress = (step - 500) / 1500
        block_weight = 1.0 - progress
        whole_weight = 0.10 + 0.90 * progress
        rms_lower = 0.40 + 0.20 * progress
        rms_upper = 1.60 - 0.20 * progress
    elif step < 5000:
        block_weight = 0.0
        whole_weight = 1.0
        rms_lower, rms_upper = 0.75, 1.25
    else:
        block_weight = 0.0
        whole_weight = 1.0
        rms_lower, rms_upper = 0.75, 1.25
    return {
        "block_weight": block_weight,
        "whole_weight": whole_weight,
        "rms_lower": rms_lower,
        "rms_upper": rms_upper,
    }


def _clip_outlier_grad_norm(
    parameters: list[nn.Parameter] | Any,
    history: list[float],
    *,
    fallback: float,
    config: dict[str, Any],
) -> tuple[torch.Tensor, float]:
    """Clip only above a trailing historical high quantile."""

    values = list(parameters)
    minimum_history = int(config.get("minimum_history", 100))
    threshold = float(fallback)
    if len(history) >= minimum_history:
        window = int(config.get("window", 1000))
        quantile = float(config.get("quantile", 0.99))
        multiplier = float(config.get("multiplier", 1.25))
        floor = float(config.get("floor", 0.0))
        observed = torch.tensor(history[-window:], dtype=torch.float32)
        threshold = max(
            floor, float(observed.quantile(quantile)) * multiplier
        )
    norm = torch.nn.utils.clip_grad_norm_(
        values, threshold, foreach=True
    )
    return norm, threshold


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


def _mean_teacher_operator(
    teacher_down: torch.Tensor,
    teacher_up: torch.Tensor,
) -> torch.Tensor:
    """Compose and average LoRA functions, never their gauge-dependent factors."""

    artists, blocks, kinds, rank, context_dim = teacher_down.shape
    if teacher_up.shape[:3] != (artists, blocks, kinds):
        raise ValueError("Teacher factor-bank leading dimensions disagree")
    if teacher_up.shape[-1] != rank:
        raise ValueError("Teacher factor-bank rank dimensions disagree")
    output_dim = int(teacher_up.shape[-2])
    values: list[torch.Tensor] = []
    for block in range(blocks):
        block_values: list[torch.Tensor] = []
        for kind in range(kinds):
            up = teacher_up[:, block, kind].float().permute(1, 0, 2).reshape(
                output_dim, artists * rank
            )
            down = teacher_down[:, block, kind].float().reshape(
                artists * rank, context_dim
            )
            block_values.append((up @ down).div_(artists).to(torch.bfloat16))
        values.append(torch.stack(block_values))
    return torch.stack(values)


def _apply_dense_kv_operator(
    context: torch.Tensor,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Apply one block's [K/V, output, context] operator."""

    return torch.einsum(
        "bnc,koc->bkno", context, operator.to(context.dtype)
    )


def _centered_residual_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    direction_weight: float,
    magnitude_weight: float,
    relation_weight: float,
    common_weight: float,
    magnitude_floor: float,
    magnitude_ceiling: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit artist-centered effects without rewarding a second common output."""

    student_f = student.float()
    teacher_f = teacher.detach().float()
    reduce_dims = tuple(range(2, teacher_f.ndim))
    teacher_rms = teacher_f.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-5)
    student_rms = student_f.square().mean(dim=reduce_dims).sqrt()
    scale = teacher_rms[..., None, None]
    reconstruction = F.smooth_l1_loss(
        student_f / scale, teacher_f / scale, beta=0.1
    )
    cosine_rows = F.cosine_similarity(
        student_f.flatten(2), teacher_f.flatten(2), dim=-1
    )
    direction = (1.0 - cosine_rows).mean()
    ratio = student_rms / teacher_rms
    magnitude = (
        F.softplus((float(magnitude_floor) - ratio) / 0.1)
        + F.softplus((ratio - float(magnitude_ceiling)) / 0.1)
    ).mean() * 0.1

    # Match every reference to its own centered teacher effect and use every
    # other artist in the controlled batch as a negative.  K and V are kept
    # separate because their useful geometries are not interchangeable.
    relation_terms: list[torch.Tensor] = []
    relation_accuracy: list[torch.Tensor] = []
    if student_f.shape[0] > 1:
        labels = torch.arange(student_f.shape[0], device=student_f.device)
        for kind in range(student_f.shape[1]):
            student_descriptor = F.normalize(
                student_f[:, kind].flatten(1), dim=1
            )
            teacher_descriptor = F.normalize(
                teacher_f[:, kind].flatten(1), dim=1
            )
            logits = student_descriptor @ teacher_descriptor.t()
            logits = logits / float(temperature)
            relation_terms.append(F.cross_entropy(logits, labels))
            relation_accuracy.append((logits.argmax(dim=1) == labels).float().mean())
    relation = (
        torch.stack(relation_terms).mean()
        if relation_terms
        else reconstruction.new_zeros(())
    )
    accuracy = (
        torch.stack(relation_accuracy).mean()
        if relation_accuracy
        else reconstruction.new_ones(())
    )

    # The exact dataset-level centered target has zero mean.  Matching the
    # stochastic teacher-batch mean gives an unbiased anti-common gradient
    # without forcing every small batch to an artificial zero tensor.
    batch_scale = teacher_rms.mean().clamp_min(1e-5)
    common = F.smooth_l1_loss(
        student_f.mean(dim=0) / batch_scale,
        teacher_f.mean(dim=0) / batch_scale,
        beta=0.1,
    )
    loss = (
        reconstruction
        + float(direction_weight) * direction
        + float(magnitude_weight) * magnitude
        + float(relation_weight) * relation
        + float(common_weight) * common
    )
    return loss, {
        "loss": loss.detach(),
        "centered_huber": reconstruction.detach(),
        "cosine": cosine_rows.mean().detach(),
        "direction_loss": direction.detach(),
        "magnitude_band_loss": magnitude.detach(),
        "student_to_teacher_rms": ratio.mean().detach(),
        "relation_loss": relation.detach(),
        "relation_accuracy": accuracy.detach(),
        "residual_common_loss": common.detach(),
        "residual_common_ratio": (
            student_f.mean(dim=0).square().mean().sqrt()
            / student_f.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
    }


def _functional_centered_attention_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    centered_huber_weight: float,
    direction_weight: float,
    magnitude_weight: float,
    relation_weight: float,
    raw_huber_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only the K/V effect visible through native attention.

    Every row in the controlled batch uses the same context and queries.  The
    batch mean therefore represents the shared functional effect, while the
    centered rows expose reference-specific differences.  This intentionally
    avoids regressing pre-normalization K/V components that native
    normalization, softmax, or O projection subsequently discard.
    """

    student_f = student.float()
    teacher_f = teacher.detach().float()
    if student_f.shape != teacher_f.shape or student_f.shape[0] < 2:
        raise ValueError(
            "Functional centered attention loss needs matching controlled rows"
        )
    student_common = student_f.mean(dim=0, keepdim=True)
    teacher_common = teacher_f.mean(dim=0, keepdim=True)
    student_centered = student_f - student_common
    teacher_centered = teacher_f - teacher_common
    reduce_dims = tuple(range(1, teacher_f.ndim))
    row_shape = (-1,) + (1,) * (teacher_f.ndim - 1)
    teacher_rms = (
        teacher_centered.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-5)
    )
    student_rms = (
        student_centered.square().mean(dim=reduce_dims) + 1e-12
    ).sqrt()
    scale = teacher_rms.reshape(row_shape)
    centered_huber = F.smooth_l1_loss(
        student_centered / scale,
        teacher_centered / scale,
        beta=0.1,
    )
    cosine_rows = F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=1
    )
    direction = (1.0 - cosine_rows).mean()
    magnitude = F.smooth_l1_loss(
        (student_rms / teacher_rms).clamp_min(1e-4).log().clamp(-4, 4),
        torch.zeros_like(student_rms),
        beta=0.1,
    )
    if temperature <= 0:
        raise ValueError("Functional relation temperature must be positive")
    student_unit = F.normalize(student_centered.flatten(1), dim=1)
    teacher_unit = F.normalize(teacher_centered.flatten(1), dim=1)
    logits = student_unit @ teacher_unit.t() / float(temperature)
    labels = torch.arange(len(student_f), device=student_f.device)
    relation = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )
    teacher_total_rms = teacher_f.square().mean().sqrt().clamp_min(1e-5)
    raw_huber = F.smooth_l1_loss(
        student_f / teacher_total_rms,
        teacher_f / teacher_total_rms,
        beta=0.1,
    )
    loss = (
        float(centered_huber_weight) * centered_huber
        + float(direction_weight) * direction
        + float(magnitude_weight) * magnitude
        + float(relation_weight) * relation
        + float(raw_huber_weight) * raw_huber
    )
    positive = (logits.diagonal() * float(temperature)).mean()
    wrong = logits.masked_fill(
        torch.eye(len(student_f), device=student_f.device, dtype=torch.bool),
        torch.finfo(logits.dtype).min,
    ).max(dim=1).values.mul(float(temperature)).mean()
    return loss, {
        "functional_loss": loss.detach(),
        "functional_centered_huber": centered_huber.detach(),
        "functional_centered_cosine": cosine_rows.mean().detach(),
        "functional_centered_magnitude": magnitude.detach(),
        "functional_student_to_teacher_rms": (
            student_rms / teacher_rms
        ).mean().detach(),
        "functional_relation_loss": relation.detach(),
        "functional_relation_accuracy": (
            logits.argmax(dim=1) == labels
        ).float().mean().detach(),
        "functional_relation_cosine_gap": (positive - wrong).detach(),
        "functional_raw_huber": raw_huber.detach(),
        "functional_student_common_ratio": (
            student_common.square().mean().sqrt()
            / student_f.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
        "functional_teacher_common_ratio": (
            teacher_common.square().mean().sqrt()
            / teacher_f.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
    }


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
    reader: nn.Module | None = None,
    ema_model: dict[str, torch.Tensor] | None = None,
    ema_reader: dict[str, torch.Tensor] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "step": int(step),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    if reader is not None:
        state["reader"] = {
            key: value.detach().cpu() for key, value in reader.state_dict().items()
        }
    if ema_model is not None:
        state["ema_model"] = {
            key: value.detach().cpu() for key, value in ema_model.items()
        }
    if ema_reader is not None:
        state["ema_reader"] = {
            key: value.detach().cpu() for key, value in ema_reader.items()
        }
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def _update_parameter_ema(
    shadow: dict[str, torch.Tensor], module: nn.Module, decay: float
) -> None:
    for name, parameter in module.named_parameters():
        if name not in shadow:
            continue
        value = parameter.detach()
        if value.dtype != shadow[name].dtype:
            value = value.to(dtype=shadow[name].dtype)
        shadow[name].mul_(decay).add_(value, alpha=1.0 - decay)


def _ema_checkpoint_state(
    module: nn.Module, shadow: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = dict(module.state_dict())
    state.update(shadow)
    return state


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
    cfg = _resolved_experiment_config(config, config_key)
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
    centered_teacher = str(cfg.get("teacher_decomposition", "full")) == "centered"
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    common_operator: torch.Tensor | None = None
    if centered_teacher:
        common_path = output / "frozen_common_operator.pt"
        if common_path.exists():
            common_operator = torch.load(
                common_path, map_location="cpu", weights_only=True
            )["common_operator"].to(device=device, dtype=torch.bfloat16)
        else:
            common_operator = _mean_teacher_operator(teacher_down, teacher_up)
            torch.save(
                {"common_operator": common_operator.cpu()}, common_path
            )
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
    relation_weight = float(training.get("relation_weight", 0.0))
    common_weight = float(training.get("residual_common_weight", 0.0))
    magnitude_floor = float(training.get("magnitude_floor", 0.7))
    magnitude_ceiling = float(training.get("magnitude_ceiling", 1.3))
    relation_temperature = float(training.get("relation_temperature", 0.1))
    relation_start = int(training.get("relation_start_step", 250))
    relation_ramp = int(training.get("relation_ramp_steps", 250))
    single_only_steps = int(training.get("single_only_steps", 0))
    mixture_ramp_steps = int(training.get("mixture_ramp_steps", 1))
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
            mixture_progress = max(
                0.0,
                min(
                    1.0,
                    (step - single_only_steps) / max(1, mixture_ramp_steps),
                ),
            )
            active_category_weights = (
                (
                    1.0 - mixture_progress
                    + mixture_progress * category_weights[0],
                    *(mixture_progress * value for value in category_weights[1:]),
                )
                if centered_teacher
                else category_weights
            )
            category = rng.choices(
                categories, weights=active_category_weights, k=1
            )[0]
            if centered_teacher:
                context_indices = torch.full(
                    (batch,), rng.randrange(train_context_count),
                    device=device, dtype=torch.long,
                )
            else:
                context_indices = torch.tensor(
                    [rng.randrange(train_context_count) for _ in range(batch)],
                    device=device,
                )
            context = contexts[context_indices]
            if category == "single":
                selected_indices = (
                    rng.sample(range(len(artist_ids)), batch)
                    if centered_teacher and batch <= len(artist_ids)
                    else [rng.randrange(len(artist_ids)) for _ in range(batch)]
                )
                target_indices = torch.tensor(selected_indices, device=device)
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
                selected_rows = (
                    rng.sample(range(len(source_rows)), batch)
                    if centered_teacher and batch <= len(source_rows)
                    else [rng.randrange(len(source_rows)) for _ in range(batch)]
                )
                target_indices = torch.tensor(selected_rows, device=device)
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
            active_relation_weight = relation_weight * max(
                0.0,
                min(1.0, (step - relation_start) / max(1, relation_ramp)),
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
                    common_delta = None
                    if centered_teacher:
                        assert common_operator is not None
                        common_delta = _apply_dense_kv_operator(
                            context, common_operator[block]
                        ) * weights.sum(dim=1)[:, None, None, None]
                        teacher = teacher - common_delta
                        block_loss, block_metrics = _centered_residual_loss(
                            student,
                            teacher,
                            direction_weight=direction_weight,
                            magnitude_weight=magnitude_weight,
                            relation_weight=active_relation_weight,
                            common_weight=common_weight,
                            magnitude_floor=magnitude_floor,
                            magnitude_ceiling=magnitude_ceiling,
                            temperature=relation_temperature,
                        )
                    else:
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
                        attention_base = (
                            common_delta if common_delta is not None else zero
                        )
                        base_key, base_value = probe.project_context(
                            context, attention_base
                        )
                        student_key, student_value = probe.project_context(
                            context, attention_base + student
                        )
                        teacher_key, teacher_value = probe.project_context(
                            context, attention_base + teacher
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
            running["relation_weight"].append(active_relation_weight)
            running["mixture_progress"].append(mixture_progress)
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
                            if centered_teacher:
                                assert common_operator is not None
                                teacher = teacher - _apply_dense_kv_operator(
                                    val_context, common_operator[block]
                                )
                                _, values = _centered_residual_loss(
                                    student,
                                    teacher,
                                    direction_weight=direction_weight,
                                    magnitude_weight=magnitude_weight,
                                    relation_weight=relation_weight,
                                    common_weight=common_weight,
                                    magnitude_floor=magnitude_floor,
                                    magnitude_ceiling=magnitude_ceiling,
                                    temperature=relation_temperature,
                                )
                            else:
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
        "teacher_decomposition": "centered" if centered_teacher else "full",
        "elapsed_seconds": time.perf_counter() - started,
        "end_to_end_flow_training": False,
        "native_artist_teacher": False,
    }
    write_json(output / "summary.json", summary)
    return summary


_FUNCTIONAL_READER_PREFIXES = (
    "set_query",
    "set_norm",
    "reference_identity_norm",
    "reference_identity_projection",
    "pool_type_embeddings",
    "pool_type_preference",
    "set_attention",
    "set_ff_norm",
    "set_ff",
    "mixers",
)


def _open_functional_reader_pooling(
    reader: DetailPreservingTypedSlotReader,
) -> list[nn.Parameter]:
    selected: list[nn.Parameter] = []
    for name, parameter in reader.named_parameters():
        trainable = any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _FUNCTIONAL_READER_PREFIXES
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("No Reader pooling parameters were selected")
    reader.train()
    return selected


@torch.no_grad()
def _materialize_reference_token_bank(
    loader: CachedTeacherReferenceLoader,
    style_ids: list[str],
    *,
    references: int,
    seed: int,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """Keep cached frozen-Resampler tokens resident on the training GPU."""

    chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    for offset in range(0, len(style_ids), chunk_size):
        ids = style_ids[offset : offset + chunk_size]
        loaded = loader.load_styles(
            ids,
            references_per_style=references,
            seed=seed + offset * 1_000_003,
        )
        chunks.append(
            loaded["tokens"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
        )
        if offset + len(ids) == len(style_ids) or (offset // chunk_size + 1) % 5 == 0:
            print(
                "materialized raw reference tokens "
                f"{offset + len(ids)}/{len(style_ids)} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    if not chunks:
        raise ValueError("Cannot materialize an empty reference bank")
    return torch.cat(chunks, dim=0)


def _select_reference_tokens(
    bank: torch.Tensor,
    artist_indices: list[int],
    *,
    reference_counts: list[int],
    reference_start: int,
    reference_stop: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(reference_counts)
    rows = []
    mask = torch.zeros(
        len(artist_indices), maximum, device=bank.device, dtype=torch.bool
    )
    for row, (artist, count) in enumerate(
        zip(artist_indices, reference_counts, strict=True)
    ):
        choices = rng.sample(range(reference_start, reference_stop), count)
        values = bank[artist, choices]
        if count < maximum:
            values = torch.cat(
                [values, values.new_zeros(maximum - count, *values.shape[1:])]
            )
        rows.append(values)
        mask[row, :count] = True
    return torch.stack(rows), mask


def _direct_delta_artist_split(
    artist_ids: list[str],
    mixture_rows: list[dict[str, Any]],
    *,
    training_artists: int,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Keep every mixture component in train and hold out whole artists."""

    index_by_style = {style_id: index for index, style_id in enumerate(artist_ids)}
    remapped = []
    required: set[int] = set()
    for row in mixture_rows:
        style_ids = [str(value) for value in row.get("style_ids", [])]
        if not style_ids:
            raise RuntimeError("Mixture rows must retain component style_ids")
        missing = [style_id for style_id in style_ids if style_id not in index_by_style]
        if missing:
            raise RuntimeError(f"Mixture styles are absent from the 320 bank: {missing[:4]}")
        components = [index_by_style[style_id] for style_id in style_ids]
        if len(components) != len(row["weights"]):
            raise RuntimeError("Mixture component styles and weights disagree")
        required.update(components)
        remapped.append({**row, "teacher_components": components})
    if len(required) > int(training_artists):
        raise ValueError("training_artists cannot contain every mixture component")
    remaining = [index for index in range(len(artist_ids)) if index not in required]
    train = sorted(required) + remaining[: int(training_artists) - len(required)]
    train_set = set(train)
    validation = [index for index in range(len(artist_ids)) if index not in train_set]
    return train, validation, remapped


def _open_direct_delta_reader(
    reader: DetailPreservingTypedSlotReader,
) -> list[nn.Parameter]:
    """Fine-tune every Reader path used to produce style memory."""

    parameters = []
    for name, parameter in reader.named_parameters():
        trainable = not name.startswith("reconstruction_")
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("No direct-delta Reader parameters were selected")
    reader.train()
    return parameters


def _direct_delta_flow_due(relative_step: int, config: dict[str, Any]) -> bool:
    """Select rare image-flow updates without landing on checkpoint boundaries."""

    if relative_step <= 0 or not bool(config.get("enabled", False)):
        return False
    if "start_step" in config:
        start = int(config["start_step"])
        if relative_step <= start:
            return False
        cycle = int(config.get("cycle", 4))
        if cycle <= 0:
            raise ValueError("human_flow cycle must be positive")
        slots = {int(value) % cycle for value in config.get("slots", [0, 1, 2])}
        return (relative_step - start - 1) % cycle in slots
    warmup_updates = int(config.get("warmup_updates", 0))
    interval = int(
        config.get("warmup_every", 10)
        if relative_step <= warmup_updates
        else config.get("every", 20)
    )
    if interval <= 0:
        raise ValueError("human_flow update intervals must be positive")
    offset = int(config.get("offset", 1)) % interval
    return relative_step % interval == offset


def _direct_delta_flow_updates_through(
    relative_step: int, config: dict[str, Any]
) -> int:
    return sum(
        _direct_delta_flow_due(step, config)
        for step in range(1, max(0, int(relative_step)) + 1)
    )


def train_direct_reference_kv_delta_320(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_key: str = "kv_reference_direct_delta_320",
) -> dict[str, Any]:
    """Train styled-reference-only, text-conditioned full native K/V deltas."""

    from .detail_style_training import _loader_config
    from .global_query_style_tokenizer import MultiPromptDualQueryCachedStyleLoader
    from .kv_activation_sampling import NativeKVActivationInjector
    from .lora_functional_distillation import FunctionalLoRATeacherBank
    from .kv_real_query_distillation import _RealQueryBank
    from .query_style_tokenizer import _sampling_reference_inputs
    from .style_transfer import (
        _optimize_frozen_anima,
        _resolve_anima_model,
        _sample_flow_timesteps,
    )

    cfg = _resolved_experiment_config(config, config_key)
    training = dict(cfg["training"])
    flow_only = bool(training.get("flow_only", False))
    steps = int(steps_override or training.get("steps", 4000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260826))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    whole_requested = bool(
        dict(training.get("whole_model_functional", {})).get("enabled", False)
    )
    raw_mixture_rows = [
        row
        for row in read_records(destination / str(cfg["mixture_manifest"]))
        if str(row["kind"]) in {"pair", "triple", "amplified", "signed"}
        and (whole_requested or bool(row.get("enabled", True)))
    ]
    if whole_requested:
        reference_manifest = (
            destination / str(cfg["mixture_reference_cache"]) / "manifest.parquet"
        )
        available_reference_styles = {
            str(row["style_id"])
            for row in read_records(reference_manifest)
            if str(row.get("split", "train")) == "train"
        }
        missing_reference_rows = [
            row for row in raw_mixture_rows
            if str(row["mixture_style_id"]) not in available_reference_styles
        ]
        if missing_reference_rows and bool(
            dict(training.get("whole_model_functional", {})).get(
                "require_all_mixture_references", True
            )
        ):
            missing_by_kind = {
                kind: sum(str(row["kind"]) == kind for row in missing_reference_rows)
                for kind in ("pair", "triple", "amplified", "signed")
            }
            raise RuntimeError(
                "Whole-model functional training requires visual tokens for every "
                f"mixture; missing={len(missing_reference_rows)} "
                f"by_kind={missing_by_kind} first_ids="
                f"{[row['mixture_style_id'] for row in missing_reference_rows[:8]]}"
            )
        raw_mixture_rows = [
            row for row in raw_mixture_rows
            if str(row["mixture_style_id"]) in available_reference_styles
        ]
    train_artists, validation_artists, mixture_rows = _direct_delta_artist_split(
        artist_ids,
        raw_mixture_rows,
        training_artists=int(training.get("training_artists", 256)),
    )
    if not validation_artists:
        raise RuntimeError("Direct-delta training requires held-out artists")
    if whole_requested:
        # The functional bank was deliberately built from all 320 teachers.
        # Keep the former held-out set as a fixed diagnostic cohort, but do
        # not omit those valid teachers from optimization.
        train_artists = list(range(len(artist_ids)))
    rows_by_kind = {
        kind: [row for row in mixture_rows if str(row["kind"]) == kind]
        for kind in ("pair", "triple", "amplified", "signed")
    }
    if any(not rows for rows in rows_by_kind.values()):
        raise RuntimeError("Every direct-delta mixture category needs rows")
    if not flow_only:
        teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
        teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    reader = _load_reader(config, destination, cfg, device)
    reader_parameters = _open_direct_delta_reader(reader)
    chunk = int(training.get("materialization_style_chunk", 16))
    token_lru = int(training.get("token_lru_shards", 8))
    single_images = int(training.get("single_reference_images", 8))
    mixture_images = int(training.get("mixture_reference_images", 4))
    single_bank = None
    mixture_banks: dict[str, torch.Tensor] = {}
    contexts = None
    train_context_count = 0
    query_bank = None
    train_query_count = 0
    probes = []
    if not flow_only:
        single_loader = CachedTeacherReferenceLoader(
            destination / str(cfg["synthetic_reference_cache"]),
            split="train", style_ids=artist_ids, batch_size=chunk,
            references=single_images, seed=seed ^ 0x53594E54,
            token_lru_shards=token_lru, strict_style_ids=True,
        )
        single_bank = _materialize_reference_token_bank(
            single_loader, artist_ids, references=single_images,
            seed=seed ^ 0x53594E54, chunk_size=chunk, device=device,
        )
        for kind, rows in rows_by_kind.items():
            style_ids = [str(row["mixture_style_id"]) for row in rows]
            loader = CachedTeacherReferenceLoader(
                destination / str(cfg["mixture_reference_cache"]),
                split="train", style_ids=style_ids, batch_size=chunk,
                references=mixture_images,
                seed=seed ^ (sum(map(ord, kind)) * 1_000_003),
                token_lru_shards=token_lru, strict_style_ids=True,
            )
            mixture_banks[kind] = _materialize_reference_token_bank(
                loader, style_ids, references=mixture_images,
                seed=seed ^ (sum(map(ord, kind)) * 1_000_003),
                chunk_size=chunk, device=device,
            )
        contexts = load_file(
            destination / str(cfg["text_context_cache"]) / "base.safetensors",
            device="cpu",
        )["base_context"].to(device=device, dtype=torch.bfloat16)
        heldout_contexts = int(training.get("heldout_contexts", 32))
        train_context_count = len(contexts) - heldout_contexts
        if train_context_count <= 0:
            raise ValueError("heldout_contexts leaves no direct-delta training context")
        query_cfg = dict(config["kv_real_query_bank"])
        query_bank = _RealQueryBank(
            destination / str(cfg["query_cache"]),
            destination / str(query_cfg["source_cache"]),
            device=device,
            gpu_resident=bool(training.get("gpu_resident_queries", True)),
        )
        query_heldout = int(training.get("heldout_query_contents", 8))
        train_query_count = len(query_bank.rows) - query_heldout
        if train_query_count <= 0:
            raise ValueError("heldout_query_contents leaves no real-Q training rows")
        probes = _load_native_attention_probes(config, destination, device)

    whole_model = dict(training.get("whole_model_functional", {}))
    whole_enabled = bool(whole_model.get("enabled", False))
    functional_bank = None
    functional_index_by_style: dict[str, int] = {}
    if whole_enabled:
        functional_bank = FunctionalLoRATeacherBank(
            destination / str(cfg["functional_teacher_cache"]),
            effect_slice_lru_entries=int(
                whole_model.get("effect_slice_lru_entries", 16)
            ),
            load_population_mean=False,
        )
        if bool(whole_model.get("gpu_resident_effects", False)):
            if functional_bank.effects is None:
                raise ValueError(
                    "gpu_resident_effects requires effect_slice_lru_entries=0"
                )
            functional_bank.effects = functional_bank.effects.to(device=device)
            print(
                "moved full LoRA functional teacher effects to "
                f"{functional_bank.effects.device}",
                flush=True,
            )
        if len(functional_bank.effect_indices) != len(functional_bank.mixtures):
            raise RuntimeError(
                "Functional teacher cache is incomplete: "
                f"{len(functional_bank.effect_indices)}/"
                f"{len(functional_bank.mixtures)} effects"
            )
        functional_index_by_style = {
            str(row["mixture_style_id"]): int(row["index"])
            for row in functional_bank.mixtures
        }

    model_cfg = dict(cfg["model"])
    model = _build_direct_delta_generator(
        model_cfg,
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
    ).to(device=device, dtype=torch.bfloat16)
    if flow_only:
        del teacher_down, teacher_up
    generator_lr = float(training.get("generator_learning_rate", 2e-4))
    reader_lr = float(training.get("reader_learning_rate", generator_lr * 0.2))
    optimizer = torch.optim.AdamW(
        [
            {"name": "generator", "params": list(model.parameters()), "lr": generator_lr},
            {"name": "reader", "params": reader_parameters, "lr": reader_lr},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    resumed = False
    state: dict[str, Any] | None = None
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        resumed = True
    elif cfg.get("initial_checkpoint"):
        initial = torch.load(
            destination / str(cfg["initial_checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(initial["model"], strict=True)
        reader.load_state_dict(initial["reader"], strict=True)
        start_step = int(cfg.get("initial_step", initial.get("step", 0)))
        if start_step != int(initial.get("step", start_step)):
            raise ValueError("initial_step must match the warm-start checkpoint")
    if (
        hasattr(model, "set_routing_recording")
        and bool(getattr(model, "expert_bias_population_update", False))
    ):
        model.set_routing_recording(False)

    ema_cfg = dict(training.get("ema", {}))
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    if ema_enabled and not 0.0 < ema_decay < 1.0:
        raise ValueError("EMA decay must be between zero and one")
    ema_model: dict[str, torch.Tensor] | None = None
    ema_reader: dict[str, torch.Tensor] | None = None
    if ema_enabled:
        saved_model_ema = state.get("ema_model") if state is not None else None
        saved_reader_ema = state.get("ema_reader") if state is not None else None
        ema_model = {
            name: (
                saved_model_ema[name].to(device=device, dtype=torch.float32)
                if saved_model_ema is not None
                else parameter.detach().float().clone()
            )
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        ema_reader = {
            name: (
                saved_reader_ema[name].to(device=device, dtype=torch.float32)
                if saved_reader_ema is not None
                else parameter.detach().float().clone()
            )
            for name, parameter in reader.named_parameters()
            if parameter.requires_grad
        }

    human_flow = dict(training.get("human_flow", {}))
    flow_enabled = bool(human_flow.get("enabled", False))
    if flow_only:
        if whole_enabled or not flow_enabled:
            raise ValueError(
                "flow_only requires human_flow.enabled=true and "
                "whole_model_functional.enabled=false"
            )
        if any(
            not _direct_delta_flow_due(step, human_flow)
            for step in range(1, steps - int(cfg.get("initial_step", 0)) + 1)
        ):
            raise ValueError(
                "flow_only requires the human-flow schedule to cover every step"
            )
    flow_injector = None
    flow_prefetched = None
    flow_validation_batches: list[dict[str, Any]] = []
    flow_update_index = 0
    anima = None
    if flow_enabled or whole_enabled:
        anima = _resolve_anima_model(
            config, destination, device
        ).requires_grad_(False).eval()
        _optimize_frozen_anima(
            anima, low_precision_rmsnorm=True, fuse_attention_projections=True
        )
        flow_injector = NativeKVActivationInjector(anima, model)
    if flow_enabled:
        detail_cfg = dict(config["detail_preserving_style_cross_attention"])
        flow_loader_cfg = _loader_config(
            config,
            detail_cfg,
            split=str(detail_cfg.get("train_split", "train")),
        )
        flow_loader_cfg.update({
            "batch_size": int(human_flow.get("batch_size", 4)),
            "same_style_target_min": int(
                human_flow.get("same_style_target_min", 1)
            ),
            "same_style_target_max": int(
                human_flow.get("same_style_target_max", 1)
            ),
            "min_references": int(human_flow.get("min_references", 1)),
            "max_references": int(human_flow.get("max_references", 4)),
            "self_reference_target_images_per_style": 0,
            "ram_resident_tokens": False,
            "reference_curriculum": {},
            "pilot_reference_schedule": [],
            "gradient_accumulation_steps": int(
                training.get("gradient_accumulation_steps", 1)
            ),
            "distinct_style_groups_per_optimizer_update": bool(
                human_flow.get(
                    "distinct_style_groups_per_optimizer_update", False
                )
            ),
        })
        if int(human_flow.get("same_style_target_min", 1)) > 1:
            flow_loader_cfg["prompt_modes"] = dict(
                human_flow.get(
                    "prompt_modes",
                    {
                        "full": 0.25,
                        "tag_dropout": 0.45,
                        "short": 0.30,
                        "empty": 0.0,
                    },
                )
            )
        flow_loader = MultiPromptDualQueryCachedStyleLoader(
            destination, flow_loader_cfg
        )
        flow_validation_every = int(
            human_flow.get("fixed_validation_every", 0)
        )
        if flow_validation_every > 0:
            validation_cfg = dict(flow_loader_cfg)
            validation_cfg.update({
                "split": str(detail_cfg.get("validation_split", "validation")),
                "prompt_modes": {
                    "full": 1.0,
                    "tag_dropout": 0.0,
                    "short": 0.0,
                    "empty": 0.0,
                },
                "quality_probability": 0.0,
            })
            validation_loader = MultiPromptDualQueryCachedStyleLoader(
                destination, validation_cfg
            )
            overlapping_artists = set(flow_loader.by_style) & set(
                validation_loader.by_style
            )
            if overlapping_artists:
                raise RuntimeError(
                    "Fixed flow validation is not artist-disjoint; overlap: "
                    + ", ".join(sorted(overlapping_artists)[:8])
                )
            flow_validation_batches = [
                validation_loader.load_step(index)
                for index in range(
                    int(human_flow.get("fixed_validation_batches", 4))
                )
            ]
            print(
                "Prepared fixed artist-disjoint flow validation: "
                f"train_artists={len(flow_loader.by_style)} "
                f"validation_artists={len(validation_loader.by_style)} "
                f"batches={len(flow_validation_batches)}",
                flush=True,
            )
        completed_relative = max(0, start_step - int(cfg.get("initial_step", 0)))
        accumulation = int(training.get("gradient_accumulation_steps", 1))
        flow_update_index = accumulation * _direct_delta_flow_updates_through(
            completed_relative, human_flow
        )
        remaining_flow_updates = (
            accumulation * _direct_delta_flow_updates_through(
                max(0, steps - int(cfg.get("initial_step", 0))), human_flow
            )
            - flow_update_index
        )
        flow_prefetched = flow_loader.prefetch(
            flow_update_index,
            remaining_flow_updates,
            workers=int(human_flow.get("prefetch_workers", 2)),
            depth=int(human_flow.get("prefetch_batches", 4)),
        )

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    categories = ("single", "pair", "triple", "amplified", "signed")
    category_weights = tuple(
        float(value)
        for value in training.get("category_weights", [0.5, 0.125, 0.125, 0.125, 0.125])
    )
    single_counts = [int(value) for value in training.get("single_reference_counts", [1, 2, 4, 8])]
    single_count_weights = [float(value) for value in training.get("single_reference_count_weights", [0.30, 0.25, 0.25, 0.20])]
    mixture_counts = [int(value) for value in training.get("mixture_reference_counts", [1, 2, 4])]
    mixture_count_weights = [float(value) for value in training.get("mixture_reference_count_weights", [0.40, 0.35, 0.25])]
    direction_weight = float(training.get("direction_weight", 0.3))
    magnitude_weight = float(training.get("magnitude_weight", 0.15))
    consistency_weight = float(training.get("reference_consistency_weight", 0.1))
    common_direction_weight = float(training.get("common_direction_weight", 0.0))
    attention_weight = float(training.get("attention_weight", 0.2))
    attention_ramp = int(training.get("attention_ramp_steps", 500))
    warmup = int(training.get("warmup_steps", 100))
    generator_clip = float(training.get("max_grad_norm", 10.0))
    reader_clip = float(training.get("reader_max_grad_norm", 5.0))
    generator_lr_schedule = dict(training.get("generator_lr_schedule", {}))
    reader_lr_schedule = dict(training.get("reader_lr_schedule", {}))

    def flow_generator_lr(relative_step: int) -> float:
        if not generator_lr_schedule:
            return generator_lr * min(1.0, relative_step / max(1, warmup))
        peak = float(generator_lr_schedule.get("peak_lr", generator_lr))
        final = float(generator_lr_schedule.get("final_lr", generator_lr))
        generator_warmup = int(
            generator_lr_schedule.get("warmup_steps", warmup)
        )
        decay_start = int(
            generator_lr_schedule.get("decay_start_step", generator_warmup)
        )
        decay_end = int(
            generator_lr_schedule.get("decay_end_step", decay_start + 1)
        )
        tail_final = float(generator_lr_schedule.get("tail_final_lr", final))
        tail_end = int(generator_lr_schedule.get("tail_decay_end_step", decay_end))
        if peak <= 0 or final < 0 or decay_end <= decay_start:
            raise ValueError("Generator LR schedule is invalid")
        if tail_final < 0 or tail_end < decay_end:
            raise ValueError("Generator LR tail schedule is invalid")
        if relative_step <= generator_warmup:
            return peak * relative_step / max(1, generator_warmup)
        if relative_step <= decay_start:
            return peak
        if relative_step < decay_end:
            progress = (relative_step - decay_start) / (decay_end - decay_start)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final + (peak - final) * cosine
        if relative_step >= tail_end or tail_end == decay_end:
            return tail_final
        progress = (relative_step - decay_end) / (tail_end - decay_end)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return tail_final + (final - tail_final) * cosine

    def flow_reader_lr(relative_step: int) -> float:
        if not reader_lr_schedule:
            return reader_lr * min(1.0, relative_step / max(1, warmup))
        peak = float(reader_lr_schedule.get("peak_lr", reader_lr))
        final = float(reader_lr_schedule.get("final_lr", reader_lr))
        reader_warmup = int(reader_lr_schedule.get("warmup_steps", warmup))
        decay_start = int(
            reader_lr_schedule.get("decay_start_step", reader_warmup)
        )
        decay_end = int(
            reader_lr_schedule.get("decay_end_step", decay_start + 1)
        )
        if peak <= 0 or final <= 0 or decay_end <= decay_start:
            raise ValueError("Reader LR schedule is invalid")
        if relative_step <= reader_warmup:
            return peak * relative_step / max(1, reader_warmup)
        if relative_step <= decay_start:
            return peak
        if relative_step >= decay_end:
            return final
        progress = (relative_step - decay_start) / max(
            1, decay_end - decay_start
        )
        return peak * (final / peak) ** progress
    adaptive_clip = dict(training.get("adaptive_clip", {}))
    generator_adaptive_clip = dict(adaptive_clip.get("generator", {}))
    reader_adaptive_clip = dict(adaptive_clip.get("reader", {}))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 500))
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    validation_tokens = int(training.get("validation_tokens", 64))
    validation_blocks = [
        int(value)
        for value in training.get("validation_blocks", [0, 4, 8, 12, 16, 20, 24, 27])
    ]

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-direct-delta-320")),
            id=str(wandb_cfg.get("id", "kv-reference-direct-delta-320")),
            resume="allow" if resumed else "never",
            config={config_key: cfg},
        )

    @torch.no_grad()
    def fixed_flow_validation() -> dict[str, float]:
        if not flow_validation_batches:
            return {}
        assert anima is not None and flow_injector is not None
        model.eval()
        reader.eval()
        base_values: list[torch.Tensor] = []
        adapted_values: list[torch.Tensor] = []
        timestep_values: list[torch.Tensor] = []
        validation_rows = sum(
            len(batch["latents"]) for batch in flow_validation_batches
        )
        validation_quantile_order = torch.randperm(
            validation_rows,
            generator=torch.Generator().manual_seed(seed ^ 0x5155414E),
        )
        validation_offset = 0
        try:
            for index, validation_batch in enumerate(flow_validation_batches):
                latents = validation_batch["latents"].to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                context = validation_batch["conditioning"].to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                references, reference_mask = _sampling_reference_inputs(
                    validation_batch, device, "heldout"
                )
                validation_rng = torch.Generator(device=device).manual_seed(
                    seed ^ 0x56414C46 ^ (index * 1_000_003)
                )
                noise = torch.randn(
                    latents.shape,
                    device=device,
                    dtype=latents.dtype,
                    generator=validation_rng,
                )
                if (
                    bool(human_flow.get("timestep_stratified_quantiles", False))
                    and str(human_flow.get("timestep_sampling")) in {
                        "sigmoid", "shift"
                    }
                ):
                    positions = validation_quantile_order[
                        validation_offset : validation_offset + len(latents)
                    ].to(device=device, dtype=torch.float32)
                    quantiles = (positions + 0.5) / validation_rows
                    logits = math.sqrt(2.0) * torch.erfinv(
                        2.0 * quantiles - 1.0
                    )
                    logits = (
                        logits * float(human_flow.get("sigmoid_scale", 1.0))
                        + float(human_flow.get("sigmoid_bias", 0.0))
                    )
                    timesteps = logits.sigmoid()
                    if str(human_flow.get("timestep_sampling")) == "shift":
                        shift = float(human_flow.get("discrete_flow_shift", 1.0))
                        timesteps = (timesteps * shift) / (
                            1 + (shift - 1) * timesteps
                        )
                else:
                    timesteps = _sample_flow_timesteps(
                        len(latents), device, human_flow, validation_rng
                    )
                validation_offset += len(latents)
                sigma = timesteps[:, None, None, None].to(latents.dtype)
                noisy = (1 - sigma) * latents + sigma * noise
                target = (noise - latents).float()
                padding = torch.zeros(
                    len(latents), 1, latents.shape[-2], latents.shape[-1],
                    device=device, dtype=latents.dtype,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    flow_injector.disable()
                    base = anima(
                        noisy.unsqueeze(2), timesteps.to(latents.dtype),
                        context=context, padding_mask=padding,
                        target_input_ids=None,
                    ).squeeze(2).float()
                    style = reader(references, reference_mask).tokens
                    flow_injector.set_style(style)
                    adapted = anima(
                        noisy.unsqueeze(2), timesteps.to(latents.dtype),
                        context=context, padding_mask=padding,
                        target_input_ids=None,
                    ).squeeze(2).float()
                    flow_injector.disable()
                dimensions = tuple(range(1, adapted.ndim))
                base_values.append((base - target).square().mean(dim=dimensions))
                adapted_values.append(
                    (adapted - target).square().mean(dim=dimensions)
                )
                timestep_values.append(timesteps)
        finally:
            flow_injector.disable()
            model.train()
            reader.train()
        base_rows = torch.cat(base_values)
        adapted_rows = torch.cat(adapted_values)
        timesteps = torch.cat(timestep_values)
        metrics = {
            "flow_mse": float(adapted_rows.mean()),
            "base_mse": float(base_rows.mean()),
            "correct_gain": float(
                ((base_rows - adapted_rows) / base_rows.clamp_min(1e-6)).mean()
            ),
        }
        for name, mask in {
            "low": timesteps < 0.4,
            "mid": (timesteps >= 0.4) & (timesteps < 0.75),
            "high": timesteps >= 0.75,
        }.items():
            if bool(mask.any()):
                metrics[f"correct_gain_{name}"] = float(
                    (
                        (base_rows[mask] - adapted_rows[mask])
                        / base_rows[mask].clamp_min(1e-6)
                    ).mean()
                )
        return metrics

    running: dict[str, list[float]] = defaultdict(list)
    generator_grad_history: list[float] = []
    reader_grad_history: list[float] = []
    # A small detached population memory makes the diversity constraint valid
    # for grouped batches, which intentionally contain only one artist. It is
    # never used to center or alter the predicted residual.
    cross_style_queue: dict[str, torch.Tensor] = {}
    diversity_probe_context: torch.Tensor | None = None
    diversity_probe_stream: torch.Tensor | None = None
    started = time.perf_counter()
    model.train()
    reader.train()
    try:
        for micro_step in range(
            start_step * accumulation_steps + 1,
            steps * accumulation_steps + 1,
        ):
            step = (micro_step - 1) // accumulation_steps + 1
            accumulation_index = (micro_step - 1) % accumulation_steps
            accumulation_last = accumulation_index == accumulation_steps - 1
            relative_step = step - int(cfg.get("initial_step", 0))
            if hasattr(model, "set_routing_step"):
                model.set_routing_step(relative_step)
            if _direct_delta_flow_due(relative_step, human_flow):
                assert flow_prefetched is not None
                assert flow_injector is not None
                flow_batch = next(flow_prefetched)
                flow_rng = torch.Generator(device=device).manual_seed(
                    seed ^ 0x464C4F57 ^ (micro_step * 100_003)
                )
                latents = flow_batch["latents"].to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                flow_context = flow_batch["conditioning"].to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                if diversity_probe_context is None:
                    diversity_probe_context = flow_context[:1, :16].detach().clone()
                    probe_values = 16 * int(model.stream_dim)
                    diversity_probe_stream = torch.linspace(
                        -1.0, 1.0, probe_values,
                        device=device, dtype=torch.bfloat16,
                    ).reshape(1, 16, int(model.stream_dim))
                flow_references, flow_mask = _sampling_reference_inputs(
                    flow_batch, device, "heldout"
                )
                flow_main_mask = flow_mask
                subset_dropout = float(
                    human_flow.get("reference_subset_dropout", 0.0)
                )
                available_references = int(flow_mask[0].sum())
                if (
                    subset_dropout
                    and available_references >= 2
                    and float(
                        torch.rand((), device=device, generator=flow_rng)
                    ) < subset_dropout
                ):
                    valid = flow_mask[0].nonzero(as_tuple=False).flatten()
                    keep = int(
                        torch.randint(
                            1,
                            available_references,
                            (),
                            device=device,
                            generator=flow_rng,
                        )
                    )
                    chosen = valid[
                        torch.randperm(
                            available_references,
                            device=device,
                            generator=flow_rng,
                        )[:keep]
                    ]
                    flow_main_mask = torch.zeros_like(flow_mask)
                    flow_main_mask[:, chosen] = flow_mask[:, chosen]
                noise = torch.randn(
                    latents.shape,
                    device=device,
                    dtype=latents.dtype,
                    generator=flow_rng,
                )
                timesteps = _sample_flow_timesteps(
                    len(latents), device, human_flow, flow_rng
                )
                sigma = timesteps[:, None, None, None].to(latents.dtype)
                noisy = (1 - sigma) * latents + sigma * noise
                target = (noise - latents).float()
                padding = torch.zeros(
                    len(latents), 1, latents.shape[-2], latents.shape[-1],
                    device=device, dtype=latents.dtype,
                )
                if accumulation_index == 0:
                    optimizer.zero_grad(set_to_none=True)
                assert anima is not None
                assert flow_injector is not None
                flow_injector.disable()
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    base_prediction = anima(
                        noisy.unsqueeze(2),
                        timesteps.to(latents.dtype),
                        context=flow_context,
                        padding_mask=padding,
                        target_input_ids=None,
                    ).squeeze(2).float()
                # Leave inference mode before using the frozen baseline as a
                # constant in differentiable residual losses.
                base_prediction = base_prediction.clone()
                block_dropout = float(
                    human_flow.get("group_shared_block_dropout", 0.0)
                )
                if not 0.0 <= block_dropout < 1.0:
                    raise ValueError("group-shared block dropout must be in [0,1)")
                block_mask = None
                block_strength = 1.0
                if block_dropout:
                    block_rng = random.Random(
                        seed ^ 0x424C4F43 ^ (micro_step * 1_000_003)
                    )
                    block_mask = torch.tensor(
                        [
                            block_rng.random() >= block_dropout
                            for _ in range(model.blocks)
                        ],
                        dtype=torch.bool,
                    )
                    if not bool(block_mask.any()):
                        block_mask[block_rng.randrange(model.blocks)] = True
                    block_strength = 1.0 / (1.0 - block_dropout)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    flow_style = reader(flow_references, flow_main_mask).tokens
                    flow_injector.set_style(
                        flow_style,
                        strength=block_strength,
                        block_mask=block_mask,
                    )
                    if hasattr(model, "set_routing_recording"):
                        model.set_routing_recording(True)
                    prediction = anima(
                        noisy.unsqueeze(2),
                        timesteps.to(latents.dtype),
                        context=flow_context,
                        padding_mask=padding,
                        target_input_ids=None,
                    ).squeeze(2).float()
                    if hasattr(model, "set_routing_recording"):
                        model.set_routing_recording(False)
                    flow_injector.disable()
                    dimensions = tuple(range(1, prediction.ndim))
                    base_rows = (base_prediction - target).square().mean(dim=dimensions)
                    correct_rows = (prediction - target).square().mean(dim=dimensions)
                    desired_effect = target - base_prediction
                    student_effect = prediction - base_prediction
                    effect_scale = desired_effect.square().mean(
                        dim=dimensions, keepdim=True
                    ).sqrt().clamp_min(float(human_flow.get("rms_floor", 1e-4)))
                    residual_huber = F.smooth_l1_loss(
                        student_effect / effect_scale,
                        desired_effect / effect_scale,
                        beta=float(human_flow.get("huber_beta", 0.1)),
                    )
                    residual_cosine = F.cosine_similarity(
                        student_effect.flatten(1), desired_effect.flatten(1), dim=-1
                    ).mean()
                    flow_mse = correct_rows.mean()
                    constraint_progress = min(
                        1.0,
                        relative_step
                        / max(1, int(human_flow.get("constraint_ramp_steps", 1000))),
                    )
                    common_cap_start = float(
                        human_flow.get("common_cap_start", 0.65)
                    )
                    common_cap_end = float(
                        human_flow.get("common_cap_end", 0.30)
                    )
                    flow_common_cap = common_cap_start + constraint_progress * (
                        common_cap_end - common_cap_start
                    )
                    rms_lower_start = float(
                        human_flow.get("rms_lower_start", 0.40)
                    )
                    rms_upper_start = float(
                        human_flow.get("rms_upper_start", 1.80)
                    )
                    flow_rms_lower = rms_lower_start + constraint_progress * (
                        float(human_flow.get("rms_lower", 0.75))
                        - rms_lower_start
                    )
                    flow_rms_upper = rms_upper_start + constraint_progress * (
                        float(human_flow.get("rms_upper", 1.25))
                        - rms_upper_start
                    )
                    flow_style_ids = [
                        str(item.style_id) for item in flow_batch["episodes"]
                    ]
                    cross_style_mask = torch.tensor(
                        [
                            left != right
                            for left in flow_style_ids
                            for right in flow_style_ids
                        ],
                        device=device,
                        dtype=torch.bool,
                    ).reshape(len(flow_style_ids), len(flow_style_ids))
                    residual_band, residual_band_metrics = _final_effect_constraints(
                        student_effect, desired_effect,
                        common_cap=flow_common_cap,
                        rms_lower=flow_rms_lower,
                        rms_upper=flow_rms_upper,
                        common_cap_weight=(
                            constraint_progress
                            * float(human_flow.get("common_cap_weight", 1.0))
                        ),
                        rms_band_weight=float(
                            human_flow.get("rms_band_weight", 1.0)
                        ),
                        rms_floor=float(human_flow.get("rms_floor", 1e-4)),
                        pair_mask=cross_style_mask,
                    )
                    diversity_weight = float(
                        human_flow.get("cross_style_diversity_weight", 0.0)
                    )
                    assert diversity_probe_context is not None
                    assert diversity_probe_stream is not None
                    diversity_signature = _adapter_probe_signature(
                        model,
                        flow_style,
                        diversity_probe_context,
                        diversity_probe_stream,
                        [
                            int(block)
                            for block in human_flow.get(
                                "diversity_probe_blocks", [0, 7, 14, 21, 27]
                            )
                        ],
                        int(human_flow.get("diversity_signature_width", 64)),
                    )
                    same_artist_loss = student_effect.new_zeros(())
                    same_artist_metrics: dict[str, torch.Tensor] | None = None
                    left_signature: torch.Tensor | None = None
                    right_signature: torch.Tensor | None = None
                    same_artist_weight = float(
                        human_flow.get("same_artist_consistency_weight", 0.0)
                    )
                    reference_count = int(flow_mask[0].sum())
                    if same_artist_weight and reference_count >= 2:
                        permutation = torch.randperm(
                            reference_count,
                            device=flow_references.device,
                            generator=flow_rng,
                        )
                        split = (reference_count + 1) // 2
                        left_indices = permutation[:split]
                        right_indices = permutation[split:]
                        left_references = flow_references[:1, left_indices]
                        right_references = flow_references[:1, right_indices]
                        left_mask = torch.ones(
                            1, len(left_indices), device=device, dtype=torch.bool
                        )
                        right_mask = torch.ones(
                            1, len(right_indices), device=device, dtype=torch.bool
                        )
                        # Alternate the optimized view. The other disjoint view
                        # is a stop-gradient target, avoiding a jointly moving
                        # pair while still training every reference position.
                        if micro_step % 2 == 0:
                            left_references, right_references = (
                                right_references,
                                left_references,
                            )
                            left_mask, right_mask = right_mask, left_mask
                        left_style = reader(left_references, left_mask).tokens
                        left_signature = _adapter_probe_signature(
                            model,
                            left_style,
                            diversity_probe_context,
                            diversity_probe_stream,
                            [
                                int(block)
                                for block in human_flow.get(
                                    "diversity_probe_blocks", [0, 7, 14, 21, 27]
                                )
                            ],
                            int(human_flow.get("diversity_signature_width", 64)),
                            normalize=False,
                        )
                        with torch.no_grad():
                            right_style = reader(
                                right_references, right_mask
                            ).tokens
                            right_signature = _adapter_probe_signature(
                                model,
                                right_style,
                                diversity_probe_context,
                                diversity_probe_stream,
                                [
                                    int(block)
                                    for block in human_flow.get(
                                        "diversity_probe_blocks", [0, 7, 14, 21, 27]
                                    )
                                ],
                                int(
                                    human_flow.get(
                                        "diversity_signature_width", 64
                                    )
                                ),
                                normalize=False,
                            )
                        same_artist_loss, same_artist_metrics = (
                            _same_artist_signature_consistency(
                                left_signature,
                                right_signature,
                                cosine_floor=float(
                                    human_flow.get(
                                        "same_artist_cosine_floor", 0.75
                                    )
                                ),
                                rms_ratio_tolerance=float(
                                    human_flow.get(
                                        "same_artist_rms_ratio_tolerance", 1.5
                                    )
                                ),
                                magnitude_weight=float(
                                    human_flow.get(
                                        "same_artist_magnitude_weight", 0.25
                                    )
                                ),
                            )
                        )
                    unique_flow_styles = set(flow_style_ids)
                    if diversity_weight and len(unique_flow_styles) != 1:
                        raise ValueError(
                            "cross-style diversity requires a single-style grouped batch"
                        )
                    flow_style_id = flow_style_ids[0]
                    diversity_references = [
                        value
                        for style_id, value in cross_style_queue.items()
                        if style_id != flow_style_id
                    ]
                    same_artist_contrastive_loss = student_effect.new_zeros(())
                    same_artist_contrastive_metrics: dict[
                        str, torch.Tensor
                    ] | None = None
                    contrastive_weight = float(
                        human_flow.get("same_artist_contrastive_weight", 0.0)
                    )
                    contrastive_min_styles = int(
                        human_flow.get("same_artist_contrastive_min_styles", 16)
                    )
                    if (
                        contrastive_weight
                        and left_signature is not None
                        and right_signature is not None
                        and len(diversity_references) >= contrastive_min_styles
                    ):
                        (
                            same_artist_contrastive_loss,
                            same_artist_contrastive_metrics,
                        ) = _same_artist_queue_infonce(
                            left_signature,
                            right_signature,
                            diversity_references,
                            temperature=float(
                                human_flow.get(
                                    "same_artist_contrastive_temperature", 0.10
                                )
                            ),
                        )
                    diversity_min_styles = int(
                        human_flow.get("diversity_queue_min_styles", 16)
                    )
                    if len(diversity_references) >= diversity_min_styles:
                        diversity_loss, diversity_positive_cosine = (
                            _cross_style_queue_diversity(
                                diversity_signature,
                                diversity_references,
                                cosine_cap=float(
                                    human_flow.get("diversity_cosine_cap", 0.35)
                                ),
                            )
                        )
                    else:
                        diversity_loss = student_effect.new_zeros(())
                        diversity_positive_cosine = student_effect.new_zeros(())
                    diversity_effective_weight = (
                        diversity_weight * constraint_progress
                    )
                    population_common_weight = float(
                        human_flow.get("population_common_weight", 0.0)
                    )
                    population_common_min_styles = int(
                        human_flow.get("population_common_min_styles", 16)
                    )
                    if (
                        population_common_weight
                        and len(diversity_references)
                        >= population_common_min_styles
                    ):
                        (
                            population_common_loss,
                            population_common_occupancy,
                        ) = _population_common_occupancy(
                            diversity_signature,
                            diversity_references,
                            occupancy_cap=float(
                                human_flow.get("population_common_cap", 0.30)
                            ),
                        )
                    else:
                        population_common_loss = student_effect.new_zeros(())
                        population_common_occupancy = student_effect.new_zeros(())
                    population_common_effective_weight = (
                        population_common_weight * constraint_progress
                    )
                    flow_mse_weight = float(
                        human_flow.get(
                            "flow_mse_weight",
                            human_flow.get("absolute_mse_weight", 0.1),
                        )
                    )
                    residual_huber_weight = float(
                        human_flow.get("residual_huber_weight", 1.0)
                    )
                    residual_direction_weight = float(
                        human_flow.get("direction_weight", 1.0)
                    )
                    output_band_weight = float(
                        human_flow.get("output_band_weight", 1.0)
                    )
                    prior_preservation = (
                        (student_effect / effect_scale).square().mean()
                    )
                    prior_preservation_weight = float(
                        human_flow.get("prior_preservation_weight", 0.0)
                    )
                    if hasattr(model, "routing_auxiliary"):
                        (
                            routing_balance,
                            routing_specialization,
                            routing_metrics,
                        ) = model.routing_auxiliary()
                    else:
                        routing_balance = flow_mse.new_zeros(())
                        routing_specialization = flow_mse.new_zeros(())
                        routing_metrics = {}
                    routing_balance_weight = float(
                        human_flow.get("expert_balance_weight", 0.0)
                    )
                    routing_specialization_weight = float(
                        human_flow.get("expert_specialization_weight", 0.0)
                    )
                    consistency_start = int(
                        human_flow.get("same_artist_consistency_start_step", 0)
                    )
                    consistency_ramp = int(
                        human_flow.get("same_artist_consistency_ramp_steps", 1)
                    )
                    consistency_progress = min(
                        1.0,
                        max(0, relative_step - consistency_start)
                        / max(1, consistency_ramp),
                    )
                    same_artist_effective_weight = (
                        same_artist_weight * consistency_progress
                    )
                    contrastive_effective_weight = (
                        contrastive_weight * consistency_progress
                    )
                    flow_loss = flow_mse_weight * flow_mse
                    if residual_huber_weight:
                        flow_loss = (
                            flow_loss + residual_huber_weight * residual_huber
                        )
                    if residual_direction_weight:
                        flow_loss = flow_loss + residual_direction_weight * (
                            1.0 - residual_cosine
                        )
                    if output_band_weight:
                        flow_loss = flow_loss + output_band_weight * residual_band
                    if diversity_effective_weight:
                        flow_loss = flow_loss + (
                            diversity_effective_weight * diversity_loss
                        )
                    if population_common_effective_weight:
                        flow_loss = flow_loss + (
                            population_common_effective_weight
                            * population_common_loss
                        )
                    if prior_preservation_weight:
                        flow_loss = flow_loss + (
                            prior_preservation_weight * prior_preservation
                        )
                    if routing_balance_weight:
                        flow_loss = flow_loss + (
                            routing_balance_weight * routing_balance
                        )
                    if routing_specialization_weight:
                        flow_loss = flow_loss + (
                            routing_specialization_weight
                            * routing_specialization
                        )
                    if same_artist_effective_weight:
                        flow_loss = flow_loss + (
                            same_artist_effective_weight * same_artist_loss
                        )
                    if contrastive_effective_weight:
                        flow_loss = flow_loss + (
                            contrastive_effective_weight
                            * same_artist_contrastive_loss
                        )
                    if not bool(torch.isfinite(flow_loss)):
                        raise RuntimeError(
                            f"Non-finite human-flow loss at step {step}"
                        )
                    weighted_flow = (
                        float(human_flow.get("flow_loss_weight", 1.0))
                        * flow_loss
                    )
                (weighted_flow / accumulation_steps).backward()
                if diversity_weight or population_common_weight:
                    # Updating an existing artist also refreshes its insertion
                    # order, so the bounded queue tracks the recent population.
                    cross_style_queue.pop(flow_style_id, None)
                    cross_style_queue[flow_style_id] = diversity_signature.detach()
                    diversity_queue_size = int(
                        human_flow.get("diversity_queue_size", 128)
                    )
                    while len(cross_style_queue) > diversity_queue_size:
                        cross_style_queue.pop(next(iter(cross_style_queue)))
                del prediction, flow_style
                denominator = base_rows.detach().clamp_min(1e-6)
                correct_gain = (base_rows - correct_rows) / denominator
                wrong_weight = float(
                    human_flow.get("wrong_reference_weight", 0.1)
                )
                ranking = weighted_flow.new_zeros(())
                wrong_gain = None
                wrong_rows = None
                weighted_ranking = weighted_flow.new_zeros(())
                if wrong_weight > 0:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        wrong_style = reader(
                            flow_references.roll(1, dims=0),
                            flow_main_mask.roll(1, dims=0),
                        ).tokens
                        flow_injector.set_style(wrong_style)
                        wrong_prediction = anima(
                            noisy.unsqueeze(2),
                            timesteps.to(latents.dtype),
                            context=flow_context,
                            padding_mask=padding,
                            target_input_ids=None,
                        ).squeeze(2).float()
                        flow_injector.disable()
                        wrong_rows = (
                            wrong_prediction - target
                        ).square().mean(dim=dimensions)
                        wrong_gain = (base_rows - wrong_rows) / denominator
                        ranking = F.relu(
                            float(human_flow.get("wrong_reference_margin", 0.02))
                            + wrong_gain
                            - correct_gain.detach()
                        ).mean()
                        weighted_ranking = wrong_weight * ranking
                    (weighted_ranking / accumulation_steps).backward()
                loss = weighted_flow.detach() + weighted_ranking.detach()
                if accumulation_last:
                    generator_grad, generator_clip_used = _clip_outlier_grad_norm(
                        model.parameters(), generator_grad_history,
                        fallback=generator_clip, config=generator_adaptive_clip,
                    )
                    reader_grad, reader_clip_used = _clip_outlier_grad_norm(
                        reader_parameters, reader_grad_history,
                        fallback=reader_clip, config=reader_adaptive_clip,
                    )
                    generator_grad_history.append(float(generator_grad))
                    reader_grad_history.append(float(reader_grad))
                    optimizer.param_groups[0]["lr"] = flow_generator_lr(
                        relative_step
                    )
                    optimizer.param_groups[1]["lr"] = flow_reader_lr(
                        relative_step
                    )
                    if hasattr(model, "apply_routing_population_update"):
                        routing_metrics.update(
                            model.apply_routing_population_update()
                        )
                    optimizer.step()
                    if ema_model is not None and ema_reader is not None:
                        _update_parameter_ema(ema_model, model, ema_decay)
                        _update_parameter_ema(ema_reader, reader, ema_decay)
                else:
                    generator_grad = loss.new_zeros(())
                    reader_grad = loss.new_zeros(())
                    generator_clip_used = generator_clip
                    reader_clip_used = reader_clip
                flow_update_index += 1
                running["loss"].append(float(loss.detach()))
                running["human_flow/loss"].append(float(flow_loss.detach()))
                running["human_flow/flow_mse"].append(float(flow_mse.detach()))
                running["human_flow/base_mse"].append(
                    float(base_rows.mean().detach())
                )
                running["human_flow/residual_huber"].append(
                    float(residual_huber.detach())
                )
                running["human_flow/correct_gain"].append(
                    float(correct_gain.mean().detach())
                )
                if wrong_rows is not None and wrong_gain is not None:
                    running["human_flow/wrong_reference_ranking"].append(
                        float(ranking.detach())
                    )
                    running["human_flow/correct_minus_wrong_mse"].append(
                        float((correct_rows - wrong_rows).mean().detach())
                    )
                    running["human_flow/wrong_gain"].append(
                        float(wrong_gain.mean().detach())
                    )
                running["human_flow/residual_cosine"].append(
                    float(residual_cosine.detach())
                )
                running["human_flow/prior_preservation"].append(
                    float(prior_preservation.detach())
                )
                running["human_flow/prior_preservation_weighted"].append(
                    float(
                        (prior_preservation_weight * prior_preservation).detach()
                    )
                )
                running["human_flow/expert_balance_loss"].append(
                    float(routing_balance.detach())
                )
                running["human_flow/expert_balance_weighted"].append(
                    float((routing_balance_weight * routing_balance).detach())
                )
                running["human_flow/expert_specialization_weighted"].append(
                    float(
                        (
                            routing_specialization_weight
                            * routing_specialization
                        ).detach()
                    )
                )
                if same_artist_metrics is not None:
                    for name, value in same_artist_metrics.items():
                        running[f"human_flow/{name}"].append(float(value))
                running["human_flow/same_artist_pair_available"].append(
                    float(same_artist_metrics is not None)
                )
                running["human_flow/same_artist_consistency_weighted"].append(
                    float(
                        (
                            same_artist_effective_weight * same_artist_loss
                        ).detach()
                    )
                )
                if same_artist_contrastive_metrics is not None:
                    for name, value in same_artist_contrastive_metrics.items():
                        running[f"human_flow/{name}"].append(float(value))
                running["human_flow/same_artist_contrastive_weighted"].append(
                    float(
                        (
                            contrastive_effective_weight
                            * same_artist_contrastive_loss
                        ).detach()
                    )
                )
                for name, value in routing_metrics.items():
                    running[f"human_flow/{name}"].append(float(value))
                running["human_flow/rms_band_outer_weighted"].append(
                    float((output_band_weight * residual_band).detach())
                )
                running["human_flow/cross_style_diversity_loss"].append(
                    float(diversity_loss.detach())
                )
                running["human_flow/cross_style_diversity_weighted"].append(
                    float(
                        (diversity_effective_weight * diversity_loss).detach()
                    )
                )
                running["human_flow/cross_style_positive_cosine"].append(
                    float(diversity_positive_cosine.detach())
                )
                running["human_flow/diversity_queue_styles"].append(
                    float(len(cross_style_queue))
                )
                running["human_flow/population_common_occupancy"].append(
                    float(population_common_occupancy.detach())
                )
                running["human_flow/population_common_loss"].append(
                    float(population_common_loss.detach())
                )
                running["human_flow/population_common_weighted"].append(
                    float(
                        (
                            population_common_effective_weight
                            * population_common_loss
                        ).detach()
                    )
                )
                running["human_flow/block_keep_rate"].append(
                    float(block_mask.float().mean())
                    if block_mask is not None else 1.0
                )
                for key, value in residual_band_metrics.items():
                    if key.startswith("rms_") or key in {
                        "common_cap_loss",
                        "common_cap_weighted_loss",
                        "positive_pairwise_cosine",
                    }:
                        running[f"human_flow/{key}"].append(float(value))
                running["human_flow/common_cap"].append(flow_common_cap)
                running["human_flow/constraint_progress"].append(
                    constraint_progress
                )
                running["human_flow/timestep"].append(float(timesteps.mean()))
                running["human_flow/reference_count"].append(
                    float(flow_mask.sum(dim=1).float().mean())
                )
                running["human_flow/main_reference_count"].append(
                    float(flow_main_mask.sum(dim=1).float().mean())
                )
                running["human_flow/style_group_size"].append(
                    float(len(flow_style_ids))
                )
                running["human_flow/unique_styles"].append(
                    float(len(set(flow_style_ids)))
                )
                running["generator_grad_norm_unclipped"].append(
                    float(generator_grad)
                )
                running["reader_grad_norm_unclipped"].append(float(reader_grad))
                running["generator_grad_clip_threshold"].append(generator_clip_used)
                running["reader_grad_clip_threshold"].append(reader_clip_used)
                running["update/human_flow"].append(1.0)
                if accumulation_last and step % log_every == 0:
                    row = {
                        key: sum(values) / len(values)
                        for key, values in running.items()
                    }
                    row["generator_lr"] = optimizer.param_groups[0]["lr"]
                    row["reader_lr"] = optimizer.param_groups[1]["lr"]
                    print(
                        f"Direct reference flow-only step={step}/{steps} {row}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"train/{key}": value for key, value in row.items()},
                            step=step,
                        )
                    running.clear()
                if (
                    accumulation_last
                    and flow_validation_batches
                    and step % flow_validation_every == 0
                ):
                    validation_metrics = fixed_flow_validation()
                    print(
                        f"Fixed artist-disjoint flow validation step={step}: "
                        f"{validation_metrics}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                f"validation_flow/{key}": value
                                for key, value in validation_metrics.items()
                            },
                            step=step,
                        )
                if accumulation_last and (
                    step % checkpoint_every == 0 or step == steps
                ):
                    for path in (
                        state_path,
                        checkpoints / f"step-{step:07d}.pt",
                    ):
                        _save_training_state(
                            path,
                            step=step,
                            model=model,
                            reader=reader,
                            optimizer=optimizer,
                            cfg=cfg,
                            ema_model=(
                                _ema_checkpoint_state(model, ema_model)
                                if ema_model is not None else None
                            ),
                            ema_reader=(
                                _ema_checkpoint_state(reader, ema_reader)
                                if ema_reader is not None else None
                            ),
                        )
                continue

            rng = random.Random(seed + micro_step * 1_000_003)
            category = rng.choices(categories, weights=category_weights, k=1)[0]
            alternative_references = None
            alternative_mask = None
            consistency_left = None
            consistency_left_mask = None
            consistency_right = None
            consistency_right_mask = None
            if category == "single":
                artists = rng.sample(train_artists, batch)
                functional_effect_indices = list(artists)
                counts = rng.choices(single_counts, weights=single_count_weights, k=batch)
                references, reference_mask = _select_reference_tokens(
                    single_bank, artists, reference_counts=counts,
                    reference_start=0, reference_stop=single_images, rng=rng,
                )
                alternative_references, alternative_mask = _select_reference_tokens(
                    single_bank, artists, reference_counts=counts,
                    reference_start=0, reference_stop=single_images,
                    rng=random.Random(seed ^ (step * 97_409)),
                )
                if float(whole_model.get("reader_consistency_weight", 0.0)) > 0:
                    split = max(1, single_images // 2)
                    consistency_counts = [min(count, split) for count in counts]
                    consistency_left, consistency_left_mask = _select_reference_tokens(
                        single_bank, artists,
                        reference_counts=consistency_counts,
                        reference_start=0, reference_stop=split,
                        rng=random.Random(seed ^ (step * 193_939)),
                    )
                    consistency_right, consistency_right_mask = _select_reference_tokens(
                        single_bank, artists,
                        reference_counts=consistency_counts,
                        reference_start=split, reference_stop=single_images,
                        rng=random.Random(seed ^ (step * 389_357)),
                    )
                components = torch.tensor(artists, device=device, dtype=torch.long)[:, None]
                weights = torch.ones(batch, 1, device=device)
            else:
                source_rows = rows_by_kind[category]
                selected_rows = rng.sample(range(len(source_rows)), batch)
                counts = rng.choices(mixture_counts, weights=mixture_count_weights, k=batch)
                references, reference_mask = _select_reference_tokens(
                    mixture_banks[category], selected_rows,
                    reference_counts=counts, reference_start=0,
                    reference_stop=mixture_images, rng=rng,
                )
                selected = [source_rows[index] for index in selected_rows]
                functional_effect_indices = [
                    functional_index_by_style[str(row["mixture_style_id"])]
                    for row in selected
                ] if whole_enabled else []
                maximum = max(len(row["teacher_components"]) for row in selected)
                components = torch.full(
                    (batch, maximum), -1, device=device, dtype=torch.long
                )
                weights = torch.zeros(batch, maximum, device=device)
                for row_index, row in enumerate(selected):
                    count = len(row["teacher_components"])
                    components[row_index, :count] = torch.tensor(
                        row["teacher_components"], device=device
                    )
                    weights[row_index, :count] = torch.tensor(
                        row["weights"], device=device
                    )
            context_indices = torch.tensor(
                [rng.randrange(train_context_count) for _ in range(batch)],
                device=device, dtype=torch.long,
            )
            context = contexts[context_indices]
            query_content = rng.randrange(train_query_count)
            query_timestep = rng.randrange(len(query_bank.timesteps))
            functional_context = query_bank.context(query_content)[None].expand(
                batch, -1, -1
            )
            curriculum = _whole_model_curriculum(relative_step)
            curriculum["block_weight"] *= float(
                whole_model.get("block_weight_scale", 1.0)
            )
            if "whole_weight_override" in whole_model:
                curriculum["whole_weight"] = float(
                    whole_model["whole_weight_override"]
                )
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            selected_blocks = [
                (first_block + offset) % model.blocks for offset in range(blocks_per_step)
            ] if not whole_enabled or curriculum["block_weight"] > 0 else []
            active_attention = attention_weight * min(
                1.0, relative_step / max(1, attention_ramp)
            )
            if accumulation_index == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = reader(references, reference_mask).tokens
                alternative_style = (
                    reader(alternative_references, alternative_mask).tokens
                    if alternative_references is not None and alternative_mask is not None
                    else None
                )
                reader_consistency = style.new_zeros((), dtype=torch.float32)
                reader_consistency_metrics: dict[str, torch.Tensor] = {}
                if (
                    consistency_left is not None
                    and consistency_left_mask is not None
                    and consistency_right is not None
                    and consistency_right_mask is not None
                ):
                    if step % 2 == 0:
                        consistency_left, consistency_right = (
                            consistency_right, consistency_left
                        )
                        consistency_left_mask, consistency_right_mask = (
                            consistency_right_mask, consistency_left_mask
                        )
                    left_memory = reader(
                        consistency_left, consistency_left_mask
                    ).tokens
                    with torch.no_grad():
                        right_memory = reader(
                            consistency_right, consistency_right_mask
                        ).tokens
                    reader_consistency, reader_consistency_metrics = (
                        _same_artist_memory_consistency(
                            left_memory,
                            right_memory,
                            cosine_floor=float(
                                whole_model.get(
                                    "reader_consistency_cosine_floor", 0.85
                                )
                            ),
                            rms_ratio_tolerance=float(
                                whole_model.get(
                                    "reader_consistency_rms_tolerance", 1.5
                                )
                            ),
                            magnitude_weight=float(
                                whole_model.get(
                                    "reader_consistency_magnitude_weight", 0.25
                                )
                            ),
                        )
                    )
                whole_loss = style.new_zeros((), dtype=torch.float32)
                whole_metrics: dict[str, torch.Tensor] = {}
                if whole_enabled:
                    assert functional_bank is not None
                    assert flow_injector is not None
                    assert anima is not None
                    functional_content = rng.randrange(
                        int(functional_bank.base["noisy_inputs"].shape[0])
                    )
                    functional_timestep = rng.randrange(
                        int(functional_bank.base["noisy_inputs"].shape[1])
                    )
                    final_context = functional_bank.base["base_context"][
                        functional_content : functional_content + 1
                    ].to(device=device, dtype=torch.bfloat16).expand(batch, -1, -1)
                    final_noisy = functional_bank.base["noisy_inputs"][
                        functional_content, functional_timestep
                    ].to(device=device, dtype=torch.bfloat16)[None].expand(batch, -1, -1, -1)
                    final_base = functional_bank.base["base_predictions"][
                        functional_content, functional_timestep
                    ].to(device=device, dtype=torch.float32)[None].expand(batch, -1, -1, -1)
                    final_teacher = functional_bank.effect_rows(
                        functional_effect_indices,
                        functional_content,
                        functional_timestep,
                    ).to(device=device, dtype=torch.float32)
                    final_t = functional_bank.base["timesteps"][
                        functional_timestep
                    ].to(device=device, dtype=torch.bfloat16).expand(batch)
                    final_padding = torch.zeros(
                        batch, 1, final_noisy.shape[-2], final_noisy.shape[-1],
                        device=device, dtype=torch.bfloat16,
                    )
                    flow_injector.set_style(style)
                    if hasattr(model, "set_routing_recording"):
                        model.set_routing_recording(True)
                    final_prediction = anima(
                        final_noisy.unsqueeze(2), final_t,
                        context=final_context, padding_mask=final_padding,
                        target_input_ids=None,
                    ).squeeze(2).float()
                    if hasattr(model, "set_routing_recording"):
                        model.set_routing_recording(False)
                    flow_injector.disable()
                    final_student = final_prediction - final_base
                    final_scale = final_teacher.square().mean(
                        dim=tuple(range(1, final_teacher.ndim)), keepdim=True
                    ).sqrt().clamp_min(float(whole_model.get("rms_floor", 1e-4)))
                    final_huber = F.smooth_l1_loss(
                        final_student / final_scale,
                        final_teacher / final_scale,
                        beta=float(whole_model.get("huber_beta", 0.1)),
                    )
                    final_cosine = F.cosine_similarity(
                        final_student.flatten(1), final_teacher.flatten(1), dim=-1
                    ).mean()
                    constraint_loss, constraint_metrics = _final_effect_constraints(
                        final_student, final_teacher,
                        common_cap=float(
                            dict(whole_model.get("common_cap_by_kind", {})).get(
                                category, whole_model.get("common_cap", 0.55)
                            )
                        ),
                        rms_lower=curriculum["rms_lower"],
                        rms_upper=curriculum["rms_upper"],
                        common_cap_weight=float(
                            whole_model.get("common_cap_weight", 1.0)
                        ),
                        rms_band_weight=float(
                            whole_model.get("rms_band_weight", 1.0)
                        ),
                        rms_floor=float(whole_model.get("rms_floor", 1e-4)),
                    )
                    relative_common = _excess_common_direction_loss(
                        final_student, final_teacher
                    )
                    relative_common_weighted = float(
                        whole_model.get("relative_common_weight", 0.5)
                    ) * relative_common
                    retrieval_loss, retrieval_metrics = (
                        _final_effect_retrieval_loss(
                            final_student,
                            final_teacher,
                            temperature=float(
                                whole_model.get("retrieval_temperature", 0.07)
                            ),
                        )
                    )
                    retrieval_weighted = float(
                        whole_model.get("retrieval_weight", 0.0)
                    ) * retrieval_loss
                    whole_loss = (
                        final_huber
                        + float(whole_model.get("direction_weight", 1.0))
                        * (1.0 - final_cosine)
                        + float(whole_model.get("constraint_weight", 1.0))
                        * constraint_loss
                        + relative_common_weighted
                        + retrieval_weighted
                        + float(whole_model.get("reader_consistency_weight", 0.0))
                        * reader_consistency
                    )
                    whole_metrics = {
                        "whole/loss": whole_loss.detach(),
                        "whole/normalized_huber": final_huber.detach(),
                        "whole/cosine": final_cosine.detach(),
                        "whole/retrieval_weighted_loss": (
                            retrieval_weighted.detach()
                        ),
                        "whole/relative_common_weighted_loss": (
                            relative_common_weighted.detach()
                        ),
                        **{
                            f"whole/{key}": value
                            for key, value in retrieval_metrics.items()
                        },
                        **{
                            f"whole/{key}": value
                            for key, value in reader_consistency_metrics.items()
                        },
                        **{f"whole/{key}": value for key, value in constraint_metrics.items()},
                    }
                block_losses = []
                block_metrics = []
                common_direction_losses = []
                first_student = None
                first_teacher = None
                for block in selected_blocks:
                    student = model(style, context, block)
                    teacher = _mixture_target(
                        context, teacher_down, teacher_up,
                        components, weights, block,
                    )
                    common_direction_losses.append(
                        _excess_common_direction_loss(student, teacher)
                    )
                    raw_loss, metrics = _normalized_activation_loss(
                        student, teacher,
                        direction_weight=direction_weight,
                        magnitude_weight=magnitude_weight,
                    )
                    if first_student is None:
                        first_student, first_teacher = student, teacher
                    if active_attention > 0:
                        functional_student = model(style, functional_context, block)
                        functional_teacher = _mixture_target(
                            functional_context, teacher_down, teacher_up,
                            components, weights, block,
                        )
                        probe = probes[block]
                        queries = query_bank.query(
                            query_content, query_timestep, block
                        )[None].expand(batch, -1, -1, -1)
                        zero = torch.zeros_like(functional_student)
                        base_key, base_value = probe.project_context(
                            functional_context, zero
                        )
                        student_key, student_value = probe.project_context(
                            functional_context, functional_student
                        )
                        teacher_key, teacher_value = probe.project_context(
                            functional_context, functional_teacher
                        )
                        base_output = probe.attend(
                            queries, base_key, base_value, queries_normalized=True
                        )
                        student_effect = probe.attend(
                            queries, student_key, student_value,
                            queries_normalized=True,
                        ) - base_output
                        teacher_effect = probe.attend(
                            queries, teacher_key, teacher_value,
                            queries_normalized=True,
                        ) - base_output
                        attention_loss, attention_metrics = _normalized_activation_loss(
                            student_effect[:, None], teacher_effect[:, None],
                            direction_weight=direction_weight,
                            magnitude_weight=magnitude_weight,
                        )
                        raw_loss = raw_loss + active_attention * attention_loss
                        metrics.update({
                            f"attention_{key}": value
                            for key, value in attention_metrics.items()
                        })
                    block_losses.append(raw_loss)
                    block_metrics.append(metrics)
                if block_losses:
                    block_loss = torch.stack(block_losses).mean()
                    common_direction = torch.stack(common_direction_losses).mean()
                else:
                    block_loss = whole_loss.new_zeros(())
                    common_direction = whole_loss.new_zeros(())
                if whole_enabled:
                    loss = (
                        curriculum["whole_weight"] * whole_loss
                        + curriculum["block_weight"]
                        * (block_loss + common_direction_weight * common_direction)
                    )
                else:
                    loss = block_loss + common_direction_weight * common_direction
                routing_metrics: dict[str, torch.Tensor] = {}
                routing_balance = loss.new_zeros(())
                if whole_enabled and hasattr(model, "routing_auxiliary"):
                    routing_balance, _, routing_metrics = (
                        model.routing_auxiliary()
                    )
                    loss = loss + float(
                        whole_model.get("expert_balance_weight", 0.0)
                    ) * routing_balance
                consistency = loss.new_zeros(())
                if alternative_style is not None and selected_blocks:
                    assert first_student is not None and first_teacher is not None
                    alternate = model(alternative_style, context, selected_blocks[0])
                    scale = first_teacher.float().square().mean(
                        dim=(2, 3), keepdim=True
                    ).sqrt().clamp_min(1e-5)
                    consistency = F.smooth_l1_loss(
                        alternate.float() / scale,
                        first_student.float() / scale,
                        beta=0.1,
                    )
                    loss = loss + consistency_weight * consistency * (
                        curriculum["block_weight"] if whole_enabled else 1.0
                    )
            (loss / accumulation_steps).backward()
            if accumulation_last:
                generator_grad, generator_clip_used = _clip_outlier_grad_norm(
                    model.parameters(), generator_grad_history,
                    fallback=generator_clip, config=generator_adaptive_clip,
                )
                reader_grad, reader_clip_used = _clip_outlier_grad_norm(
                    reader_parameters, reader_grad_history,
                    fallback=reader_clip, config=reader_adaptive_clip,
                )
                generator_grad_history.append(float(generator_grad))
                reader_grad_history.append(float(reader_grad))
                lr_scale = min(1.0, relative_step / max(1, warmup))
                distill_lr = float(
                    human_flow.get("distillation_lr_multiplier", 1.0)
                )
                optimizer.param_groups[0]["lr"] = (
                    flow_generator_lr(relative_step) * distill_lr
                )
                optimizer.param_groups[1]["lr"] = (
                    flow_reader_lr(relative_step) * distill_lr
                )
                if hasattr(model, "apply_routing_population_update"):
                    routing_metrics.update(
                        model.apply_routing_population_update()
                    )
                optimizer.step()
                if ema_model is not None and ema_reader is not None:
                    _update_parameter_ema(ema_model, model, ema_decay)
                    _update_parameter_ema(ema_reader, reader, ema_decay)
            else:
                generator_grad = loss.new_zeros(())
                reader_grad = loss.new_zeros(())
                generator_clip_used = generator_clip
                reader_clip_used = reader_clip

            running["loss"].append(float(loss.detach()))
            running["generator_grad_norm_unclipped"].append(float(generator_grad))
            running["reader_grad_norm_unclipped"].append(float(reader_grad))
            running["generator_grad_clip_threshold"].append(generator_clip_used)
            running["reader_grad_clip_threshold"].append(reader_clip_used)
            running["reference_count"].append(sum(counts) / len(counts))
            running["reference_consistency"].append(float(consistency.detach()))
            running["common_direction_loss"].append(
                float(common_direction.detach())
            )
            running["attention_weight"].append(active_attention)
            running["curriculum/block_weight"].append(curriculum["block_weight"])
            running["curriculum/whole_weight"].append(curriculum["whole_weight"])
            running["update/distillation"].append(1.0)
            running[f"category/{category}"].append(1.0)
            if block_metrics:
                for key in block_metrics[0]:
                    running[key].append(
                        sum(float(values[key]) for values in block_metrics)
                        / len(block_metrics)
                    )
            for key, value in whole_metrics.items():
                running[key].append(float(value))
                running[f"{key}_by_kind/{category}"].append(float(value))
            running["routing/balance_loss"].append(
                float(routing_balance.detach())
            )
            running["routing/balance_weighted"].append(
                float(
                    (
                        float(whole_model.get("expert_balance_weight", 0.0))
                        * routing_balance
                    ).detach()
                )
            )
            for key, value in routing_metrics.items():
                running[f"routing/{key}"].append(float(value))
            if accumulation_last and step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["generator_lr"] = optimizer.param_groups[0]["lr"]
                row["reader_lr"] = optimizer.param_groups[1]["lr"]
                row["context_unique_fraction"] = len(set(context_indices.tolist())) / batch
                print(f"Direct reference K/V delta step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train/{key}": value for key, value in row.items()},
                        step=step,
                    )
                running.clear()

            if accumulation_last and validation_every > 0 and step % validation_every == 0:
                model.eval()
                reader.eval()
                validation_rows: dict[str, list[float]] = defaultdict(list)
                val_artists = validation_artists[: min(8, len(validation_artists))]
                token_ids = torch.linspace(
                    0, contexts.shape[1] - 1,
                    min(validation_tokens, contexts.shape[1]), device=device,
                ).round().long().unique()
                val_context = contexts[train_context_count, token_ids][None].expand(
                    len(val_artists), -1, -1
                )
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for reference_count in single_counts:
                        val_refs, val_mask = _select_reference_tokens(
                            single_bank, val_artists,
                            reference_counts=[reference_count] * len(val_artists),
                            reference_start=0, reference_stop=single_images,
                            rng=random.Random(seed ^ step ^ reference_count),
                        )
                        val_style = reader(val_refs, val_mask).tokens
                        artist_tensor = torch.tensor(
                            val_artists, device=device, dtype=torch.long
                        )
                        for block in validation_blocks:
                            student = model(val_style, val_context, block)
                            teacher = _mixture_target(
                                val_context, teacher_down, teacher_up,
                                artist_tensor[:, None],
                                torch.ones(len(val_artists), 1, device=device), block,
                            )
                            _, values = _normalized_activation_loss(
                                student, teacher,
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                            correct = F.cosine_similarity(
                                student.float().flatten(2),
                                teacher.float().flatten(2), dim=-1,
                            ).mean()
                            wrong = F.cosine_similarity(
                                student.float().flatten(2),
                                teacher.roll(1, dims=0).float().flatten(2), dim=-1,
                            ).mean()
                            values["correct_minus_wrong_cosine"] = correct - wrong
                            values.update(_prediction_population_metrics(student))
                            for key, value in values.items():
                                validation_rows[f"single_r{reference_count}/{key}"].append(
                                    float(value)
                                )
                    if whole_enabled:
                        assert functional_bank is not None
                        assert flow_injector is not None
                        assert anima is not None
                        final_content = int(
                            functional_bank.base["noisy_inputs"].shape[0]
                        ) - 1
                        final_timestep = int(
                            functional_bank.base["noisy_inputs"].shape[1]
                        ) // 2
                        final_count = min(4, single_images)
                        final_refs, final_mask = _select_reference_tokens(
                            single_bank, val_artists,
                            reference_counts=[final_count] * len(val_artists),
                            reference_start=0, reference_stop=single_images,
                            rng=random.Random(seed ^ 0x56414C),
                        )
                        final_style = reader(final_refs, final_mask).tokens
                        final_context = functional_bank.base["base_context"][
                            final_content : final_content + 1
                        ].to(device=device, dtype=torch.bfloat16).expand(
                            len(val_artists), -1, -1
                        )
                        final_noisy = functional_bank.base["noisy_inputs"][
                            final_content, final_timestep
                        ].to(device=device, dtype=torch.bfloat16)[None].expand(
                            len(val_artists), -1, -1, -1
                        )
                        final_base = functional_bank.base["base_predictions"][
                            final_content, final_timestep
                        ].to(device=device, dtype=torch.float32)[None].expand(
                            len(val_artists), -1, -1, -1
                        )
                        final_teacher = functional_bank.effect_rows(
                            val_artists, final_content, final_timestep
                        ).to(device=device, dtype=torch.float32)
                        final_times = functional_bank.base["timesteps"][
                            final_timestep
                        ].to(device=device, dtype=torch.bfloat16).expand(
                            len(val_artists)
                        )
                        final_padding = torch.zeros(
                            len(val_artists), 1, final_noisy.shape[-2],
                            final_noisy.shape[-1], device=device,
                            dtype=torch.bfloat16,
                        )
                        flow_injector.set_style(final_style)
                        final_prediction = anima(
                            final_noisy.unsqueeze(2), final_times,
                            context=final_context, padding_mask=final_padding,
                            target_input_ids=None,
                        ).squeeze(2).float()
                        flow_injector.disable()
                        final_student = final_prediction - final_base
                        final_cosine = F.cosine_similarity(
                            final_student.flatten(1), final_teacher.flatten(1), dim=-1
                        ).mean()
                        _, final_retrieval_values = _final_effect_retrieval_loss(
                            final_student,
                            final_teacher,
                            temperature=float(
                                whole_model.get("retrieval_temperature", 0.07)
                            ),
                        )
                        _, final_values = _final_effect_constraints(
                            final_student, final_teacher,
                            common_cap=float(
                                dict(whole_model.get("common_cap_by_kind", {})).get(
                                    "single", whole_model.get("common_cap", 0.55)
                                )
                            ),
                            rms_lower=curriculum["rms_lower"],
                            rms_upper=curriculum["rms_upper"],
                            rms_floor=float(whole_model.get("rms_floor", 1e-4)),
                        )
                        validation_rows["whole/cosine"].append(float(final_cosine))
                        for key, value in final_retrieval_values.items():
                            validation_rows[f"whole/{key}"].append(float(value))
                        for key, value in final_values.items():
                            validation_rows[f"whole/{key}"].append(float(value))
                    for kind, source_rows in rows_by_kind.items():
                        selected_rows = list(range(min(4, len(source_rows))))
                        mix_refs, mix_mask = _select_reference_tokens(
                            mixture_banks[kind], selected_rows,
                            reference_counts=[mixture_images] * len(selected_rows),
                            reference_start=0, reference_stop=mixture_images,
                            rng=random.Random(seed ^ step ^ sum(map(ord, kind))),
                        )
                        mix_style = reader(mix_refs, mix_mask).tokens
                        selected = [source_rows[index] for index in selected_rows]
                        maximum = max(len(row["teacher_components"]) for row in selected)
                        mix_components = torch.full(
                            (len(selected), maximum), -1,
                            device=device, dtype=torch.long,
                        )
                        mix_weights = torch.zeros(
                            len(selected), maximum, device=device
                        )
                        for row_index, source in enumerate(selected):
                            count = len(source["teacher_components"])
                            mix_components[row_index, :count] = torch.tensor(
                                source["teacher_components"], device=device
                            )
                            mix_weights[row_index, :count] = torch.tensor(
                                source["weights"], device=device
                            )
                        mix_context = val_context[:1].expand(len(selected), -1, -1)
                        for block in validation_blocks:
                            student = model(mix_style, mix_context, block)
                            teacher = _mixture_target(
                                mix_context, teacher_down, teacher_up,
                                mix_components, mix_weights, block,
                            )
                            _, values = _normalized_activation_loss(
                                student, teacher,
                                direction_weight=direction_weight,
                                magnitude_weight=magnitude_weight,
                            )
                            for key, value in values.items():
                                validation_rows[f"mixture_{kind}/{key}"].append(
                                    float(value)
                                )
                validation = {
                    key: sum(values) / len(values)
                    for key, values in validation_rows.items()
                }
                print(f"Direct reference K/V delta validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/{key}": value for key, value in validation.items()},
                        step=step,
                    )
                model.train()
                reader.train()

            if accumulation_last and (step % checkpoint_every == 0 or step == steps):
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path, step=step, model=model, reader=reader,
                        optimizer=optimizer, cfg=cfg,
                        ema_model=(
                            _ema_checkpoint_state(model, ema_model)
                            if ema_model is not None else None
                        ),
                        ema_reader=(
                            _ema_checkpoint_state(reader, ema_reader)
                            if ema_reader is not None else None
                        ),
                    )
    finally:
        if flow_injector is not None:
            flow_injector.close()
        if wandb_run is not None:
            wandb_run.finish()

    def grad_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"median": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "median": float(tensor.median()),
            "p95": float(tensor.quantile(0.95)),
            "p99": float(tensor.quantile(0.99)),
            "maximum": float(tensor.max()),
        }

    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "training_artists": len(train_artists),
        "validation_artists": len(validation_artists),
        "mixtures": len(mixture_rows),
        "contexts": 0 if contexts is None else len(contexts),
        "reference_input": "styled_only",
        "teacher_decomposition": "none" if flow_only else "full",
        "primary_objective": "image_flow_only" if flow_only else "hybrid_distillation",
        "common_branch": False,
        "reader_end_to_end": True,
        "human_flow_enabled": flow_enabled,
        "human_flow_updates": flow_update_index,
        "distillation_lr_multiplier": float(
            human_flow.get("distillation_lr_multiplier", 1.0)
        ),
        "generator_grad_norm": grad_summary(generator_grad_history),
        "reader_grad_norm": grad_summary(reader_grad_history),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_direct_reference_kv_delta_320(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_direct_delta_320"]
    cfg["output_directory"] = "kv_reference_direct_delta_320_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 50
    cfg["training"]["checkpoint_every"] = 100
    # Measure the true distribution first; the production config is tightened
    # to its p99 after this calibration run.
    cfg["training"]["max_grad_norm"] = 1.0e9
    cfg["training"]["reader_max_grad_norm"] = 1.0e9
    return train_direct_reference_kv_delta_320(
        effective,
        destination,
        steps_override=int(cfg.get("initial_step", 0)) + 100,
    )


@torch.no_grad()
def sample_direct_reference_kv_delta_320(
    config: dict[str, Any], destination: Path, *,
    sample_config_key: str = "kv_reference_direct_delta_320_sample",
) -> dict[str, Any]:
    """Render the historical fixed references and deterministic 4+4 panel."""

    from PIL import Image

    from .detail_style_training import _loader_config
    from .global_query_style_tokenizer import MultiPromptDualQueryCachedStyleLoader
    from .kv_activation_sampling import NativeKVActivationInjector, _save_panel
    from .lora_functional_distillation import _preview_pixels
    from .query_style_tokenizer import _sampling_reference_inputs, _select_sample_episodes
    from .style_transfer import (
        _load_sampling_vae,
        _make_sample_sheet,
        _optimize_frozen_anima,
        _pad_text_conditions,
        _resolve_anima_model,
    )
    from .synthetic_teacher import _sample_anima_batch
    from .dual_query_external_samples import load_dual_query_external_sample

    sample_cfg = dict(config[sample_config_key])
    device = str(sample_cfg.get("device", "cuda"))
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["config"])
    use_ema = bool(
        sample_cfg.get(
            "use_ema",
            dict(cfg.get("training", {})).get("ema", {}).get("sample", True),
        )
    )
    reader = _load_reader(config, destination, cfg, device)
    reader_state = (
        checkpoint["ema_reader"]
        if use_ema and "ema_reader" in checkpoint
        else checkpoint["reader"]
    )
    reader.load_state_dict(reader_state, strict=True)
    reader.requires_grad_(False).eval()
    model_cfg = dict(cfg["model"])
    model_state = (
        checkpoint["ema_model"]
        if use_ema and "ema_model" in checkpoint
        else checkpoint["model"]
    )
    architecture = str(model_cfg.get("architecture", "direct_cross_attention"))
    if architecture == "direct_cross_attention":
        context_dim = int(model_state["context_query.0.weight"].shape[1])
        if int(model_cfg.get("output_experts", 0)):
            output_dim = int(model_state["output_expert_up"].shape[-1])
        elif int(model_cfg.get("output_rank", 0)):
            output_dim = int(model_state["output_head.0.up"].shape[-1])
        else:
            output_dim = int(model_state["output_head.0.weight"].shape[0] // 2)
    elif architecture == "low_rank_kvoq_operator":
        context_dim = int(model_state["down_output.0.0.weight"].shape[0])
        output_dim = int(model_state["up_output.0.0.weight"].shape[0])
    else:
        raise RuntimeError(
            f"Direct-delta sample received architecture {architecture!r}"
        )
    model = _build_direct_delta_generator(
        model_cfg,
        style_dim=int(reader.dim),
        context_dim=context_dim,
        output_dim=output_dim,
        blocks=int(cfg.get("blocks", 28)),
    ).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(model_state, strict=True)
    model.requires_grad_(False).eval()
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    activation_injector = NativeKVActivationInjector(anima, model)
    batch_size = int(sample_cfg.get("batch_size", 4))
    strengths = [float(value) for value in sample_cfg.get("strengths", [1.0, 2.0])]
    stream_strength_scale = float(sample_cfg.get("stream_strength_scale", 1.0))

    def denoise(
        style_memory: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        seeds: list[int],
        *,
        text_cfg: float,
        strength: float | None,
    ) -> torch.Tensor:
        values = []
        for start in range(0, len(seeds), batch_size):
            stop = min(len(seeds), start + batch_size)
            if strength is None:
                activation_injector.disable()
            else:
                activation_injector.set_style(
                    style_memory[start:stop],
                    strength=strength,
                    stream_strength=strength * stream_strength_scale,
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values.append(_sample_anima_batch(
                    anima,
                    noise[start:stop],
                    positive[start:stop],
                    negative[start:stop],
                    sigmas,
                    text_cfg=text_cfg,
                    speed=None,
                    generation_seeds=seeds[start:stop],
                ).cpu())
        return torch.cat(values)

    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(False).eval()

    def decode(latents: torch.Tensor) -> list[Image.Image]:
        decoded = []
        for start in range(0, len(latents), batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded.extend(_preview_pixels(
                    vae.decode_to_pixels(latents[start : start + batch_size].to(device))
                ))
        return decoded

    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)

    prepared = load_dual_query_external_sample(config, destination)
    generation = dict(prepared["cfg"])
    fixed_tokens = prepared["reference_tokens"][:, None].to(
        device=device, dtype=torch.bfloat16
    )
    fixed_mask = torch.ones(fixed_tokens.shape[:2], device=device, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fixed_memory = reader(fixed_tokens, fixed_mask).tokens
    fixed_rows = int(fixed_tokens.shape[0])
    fixed_positive = prepared["positive"]
    fixed_negative = prepared["negative"]
    if fixed_positive.ndim == 2:
        fixed_positive = fixed_positive[None]
    if fixed_negative.ndim == 2:
        fixed_negative = fixed_negative[None]
    fixed_positive = fixed_positive.to(device=device, dtype=torch.bfloat16).expand(fixed_rows, -1, -1)
    fixed_negative = fixed_negative.to(device=device, dtype=torch.bfloat16).expand(fixed_rows, -1, -1)
    fixed_seed = int(generation["seed"])
    fixed_seeds = [fixed_seed] * fixed_rows
    width, height = int(generation["width"]), int(generation["height"])
    fixed_noise = torch.randn(
        1, 16, 1, height // 8, width // 8,
        generator=torch.Generator(device="cpu").manual_seed(fixed_seed),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16).expand(fixed_rows, -1, -1, -1, -1)
    steps = int(generation["steps"])
    shift = float(generation.get("flow_shift", 3.0))
    fixed_sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    fixed_sigmas = fixed_sigmas * shift / (1 + (shift - 1) * fixed_sigmas)
    fixed_base = denoise(
        fixed_memory, fixed_positive, fixed_negative, fixed_noise, fixed_sigmas,
        fixed_seeds, text_cfg=float(generation["cfg"]), strength=None,
    )
    fixed_predicted = {
        strength: denoise(
            fixed_memory, fixed_positive, fixed_negative, fixed_noise, fixed_sigmas,
            fixed_seeds, text_cfg=float(generation["cfg"]), strength=strength,
        )
        for strength in strengths
    }
    fixed_images: dict[str, list[Image.Image]] = {
        "Fixed reference": [Image.open(path).convert("RGB") for path in prepared["paths"]],
        "Frozen Anima": decode(fixed_base),
        **{f"Predicted {strength:g}x": decode(values) for strength, values in fixed_predicted.items()},
    }
    fixed_output = output / "fixed-reference"
    fixed_output.mkdir(parents=True, exist_ok=True)
    fixed_panel = _save_panel(
        fixed_output, [f"TestSample{index + 1}" for index in range(fixed_rows)],
        fixed_images, list(fixed_images),
        tile_width=int(sample_cfg.get("panel_tile_width", 384)),
    )

    detail_cfg = dict(config["detail_preserving_style_cross_attention"])
    panel_loaders = []
    for split in (str(detail_cfg.get("train_split", "train")), str(detail_cfg.get("validation_split", "validation"))):
        loader_cfg = _loader_config(config, detail_cfg, split=split)
        loader_cfg["ram_resident_tokens"] = False
        panel_loaders.append(MultiPromptDualQueryCachedStyleLoader(destination, loader_cfg))
    panel_sampling = dict(detail_cfg.get("sampling", {}))
    panel_seed = int(panel_sampling.get("seed", detail_cfg.get("seed", 0) ^ 0x5A17))
    requests = [
        ("train", panel_loaders[0], episode, panel_seed + index * 10_007)
        for index, episode in enumerate(_select_sample_episodes(panel_loaders[0], 4))
    ] + [
        ("validation", panel_loaders[1], episode, panel_seed + (index + 4) * 10_007)
        for index, episode in enumerate(_select_sample_episodes(panel_loaders[1], 4))
    ]
    batches = [loader.load_step(episode) for _, loader, episode, _ in requests]
    panel_memory = []
    for batch in batches:
        references, mask = _sampling_reference_inputs(batch, device, "heldout")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            panel_memory.append(reader(references, mask).tokens[:1])
    panel_memory_tensor = torch.cat(panel_memory)
    panel_positive = torch.cat([batch["conditioning"][:1] for batch in batches]).to(device=device, dtype=torch.bfloat16)
    null_text = load_file(requests[0][1].text_root / "null_conditioning.safetensors", device="cpu")["empty_prompt"]
    if null_text.ndim == 3:
        null_text = null_text[0]
    panel_negative = _pad_text_conditions(
        [null_text] * len(requests), requests[0][1].text_conditioning_length
    ).to(device=device, dtype=torch.bfloat16)
    panel_width = int(panel_sampling.get("width", 768))
    panel_height = int(panel_sampling.get("height", 768))
    panel_seeds = [seed for _, _, _, seed in requests]
    panel_noise = torch.cat([
        torch.randn(
            1, 16, 1, panel_height // 8, panel_width // 8,
            generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32,
        )
        for seed in panel_seeds
    ]).to(device=device, dtype=torch.bfloat16)
    panel_steps = int(panel_sampling.get("steps", 30))
    panel_shift = float(panel_sampling.get("flow_shift", 3.0))
    panel_sigmas = torch.linspace(1.0, 0.0, panel_steps + 1, device=device)
    panel_sigmas = panel_sigmas * panel_shift / (1 + (panel_shift - 1) * panel_sigmas)
    panel_base = denoise(
        panel_memory_tensor, panel_positive, panel_negative, panel_noise, panel_sigmas,
        panel_seeds, text_cfg=float(panel_sampling.get("text_cfg", 4.0)), strength=None,
    )
    panel_predicted = {
        strength: denoise(
            panel_memory_tensor, panel_positive, panel_negative, panel_noise, panel_sigmas,
            panel_seeds, text_cfg=float(panel_sampling.get("text_cfg", 4.0)), strength=strength,
        )
        for strength in strengths
    }
    panel_base_images = decode(panel_base)
    panel_predicted_images = {strength: decode(values) for strength, values in panel_predicted.items()}
    panel_output = output / "panel"
    panel_output.mkdir(parents=True, exist_ok=True)
    panel_sheets: list[str] = []
    for index, ((split, loader, _, _), batch) in enumerate(zip(requests, batches, strict=True)):
        for strength in strengths:
            sheet = _make_sample_sheet(
                panel_predicted_images[strength][index], loader, {"episodes": [batch["episodes"][0]]},
                base_generated=panel_base_images[index],
                generated_label=f"Direct reference delta {strength:g}x (heldout) — {batch['episodes'][0].style_id}",
            )
            path = panel_output / f"{split}-{index % 4}-strength-{strength:g}x-sheet.png"
            sheet.save(path)
            panel_sheets.append(str(path))
    overview_images: dict[str, list[Image.Image]] = {
        "Target": [
            Image.open(loader.style_by_id[int(batch["episodes"][0].target_id)]["local_path"]).convert("RGB")
            for (_, loader, _, _), batch in zip(requests, batches, strict=True)
        ],
        "Reference 1": [
            Image.open(loader.style_by_id[int(batch["episodes"][0].reference_ids[0])]["local_path"]).convert("RGB")
            for (_, loader, _, _), batch in zip(requests, batches, strict=True)
        ],
        "Frozen Anima": panel_base_images,
        **{f"Predicted {strength:g}x": images for strength, images in panel_predicted_images.items()},
    }
    panel_overview = _save_panel(
        panel_output,
        [f"{split}-{index % 4}: {batch['episodes'][0].style_id}" for index, ((split, _, _, _), batch) in enumerate(zip(requests, batches, strict=True))],
        overview_images, list(overview_images),
        tile_width=int(sample_cfg.get("panel_tile_width", 384)),
    )
    activation_injector.close()
    torch.cuda.empty_cache()
    summary = {
        "sampling_contract": "fixed_test_sample_1_7_and_episodic_4_plus_4_v1",
        "checkpoint": str(checkpoint_path),
        "weights": "ema" if use_ema and "ema_model" in checkpoint else "raw",
        "fixed_reference_panel": str(fixed_panel),
        "panel_overview": str(panel_overview),
        "panel_sheets": panel_sheets,
        "panel_episode_indices": [episode for _, _, episode, _ in requests],
        "panel_style_ids": [batch["episodes"][0].style_id for batch in batches],
        "strengths": strengths,
        "prompt": str(generation["prompt"]),
        "seed": fixed_seed,
    }
    write_json(output / "summary.json", summary)
    return summary


def train_scheduled_direct_reference_kv_delta_320(
    config: dict[str, Any], destination: Path, *,
    config_key: str = "kv_reference_direct_delta_320",
    sample_config_key: str = "kv_reference_direct_delta_320_sample",
) -> dict[str, Any]:
    """Train in checkpoint-sized segments and render each scheduled checkpoint."""

    cfg = _resolved_experiment_config(config, config_key)
    training = dict(cfg["training"])
    steps = int(training.get("steps", 100_000))
    sample_every = int(training.get("sample_every", 0))
    configured_targets = sorted({
        int(value) for value in training.get("sample_steps", [])
        if 0 < int(value) <= steps
    })
    if sample_every <= 0 and not configured_targets:
        return train_direct_reference_kv_delta_320(config, destination)

    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    completed = max(
        (
            int(path.stem.removeprefix("step-"))
            for path in checkpoints.glob("step-*.pt")
        ),
        default=int(cfg.get("initial_step", 0)),
    )
    if configured_targets:
        targets = [target for target in configured_targets if target > completed]
    else:
        targets = list(
            range(((completed // sample_every) + 1) * sample_every, steps + 1, sample_every)
        )
    if completed < steps and (not targets or targets[-1] != steps):
        targets.append(steps)

    segments = []
    samples = []
    sample_root = str(
        training.get(
            "sample_output_root",
            "diagnostics/kv-reference-direct-delta-r16-320",
        )
    )
    wandb_cfg = dict(training.get("wandb", {}))

    def upload_sample(summary: dict[str, Any], target: int) -> None:
        if not bool(wandb_cfg.get("enabled", True)):
            return
        import wandb

        run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-direct-delta-320")),
            id=str(wandb_cfg.get("id", "kv-reference-direct-delta-320")),
            resume="allow",
        )
        run.log(
            {
                "sample_step": target,
                "samples/fixed_reference": wandb.Image(
                    summary["fixed_reference_panel"],
                    caption=f"fixed reference step {target}",
                ),
                "samples/panel": wandb.Image(
                    summary["panel_overview"],
                    caption=f"legacy panel step {target}",
                ),
            },
        )
        run.finish()
    sampling_contract = "fixed_test_sample_1_7_and_episodic_4_plus_4_v1"
    previous_targets = (
        [target for target in configured_targets if target <= completed]
        if configured_targets else range(sample_every, completed + 1, sample_every)
    )
    for target in previous_targets:
        checkpoint = checkpoints / f"step-{target:07d}.pt"
        summary_path = destination / f"{sample_root}-step{target}" / "summary.json"
        current_contract = None
        if summary_path.exists():
            import json
            current_contract = json.loads(summary_path.read_text(encoding="utf-8")).get("sampling_contract")
        if checkpoint.exists() and current_contract != sampling_contract:
            sample_config = copy.deepcopy(config)
            sample_cfg = sample_config[sample_config_key]
            sample_cfg["checkpoint"] = str(checkpoint.relative_to(destination))
            sample_cfg["output_directory"] = f"{sample_root}-step{target}"
            sample_summary = sample_direct_reference_kv_delta_320(
                sample_config, destination, sample_config_key=sample_config_key
            )
            samples.append(sample_summary)
            upload_sample(sample_summary, target)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for target in targets:
        segments.append(
            train_direct_reference_kv_delta_320(
                config, destination, steps_override=target, config_key=config_key
            )
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if configured_targets or target % sample_every == 0:
            sample_config = copy.deepcopy(config)
            sample_cfg = sample_config[sample_config_key]
            sample_cfg["checkpoint"] = (
                f"{cfg['output_directory']}/checkpoints/step-{target:07d}.pt"
            )
            sample_cfg["output_directory"] = f"{sample_root}-step{target}"
            sample_summary = sample_direct_reference_kv_delta_320(
                sample_config, destination, sample_config_key=sample_config_key
            )
            samples.append(sample_summary)
            upload_sample(sample_summary, target)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "steps": steps,
        "initial_step": completed,
        "segment_targets": targets,
        "segments": segments,
        "samples": samples,
    }


def train_scheduled_low_rank_kvoq_flow_50k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_low_rank_kvoq_flow_50k",
        sample_config_key="kv_reference_low_rank_kvoq_flow_50k_sample",
    )


def sample_low_rank_kvoq_flow_50k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_low_rank_kvoq_flow_50k_sample",
    )


def train_scheduled_direct_rank32_kvoq_flow_50k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_direct_rank32_kvoq_flow_50k",
        sample_config_key="kv_reference_direct_rank32_kvoq_flow_50k_sample",
    )


def sample_direct_rank32_kvoq_flow_50k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_direct_rank32_kvoq_flow_50k_sample",
    )


def train_scheduled_expert_kvoq_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvoq_flow_10k",
        sample_config_key="kv_reference_expert_kvoq_flow_10k_sample",
    )


def sample_expert_kvoq_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvoq_flow_10k_sample",
    )


def train_scheduled_expert_kvo_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_flow_10k",
        sample_config_key="kv_reference_expert_kvo_flow_10k_sample",
    )


def sample_expert_kvo_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_flow_10k_sample",
    )


def train_scheduled_expert_kvo_balanced_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_balanced_flow_10k",
        sample_config_key="kv_reference_expert_kvo_balanced_flow_10k_sample",
    )


def sample_expert_kvo_balanced_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_balanced_flow_10k_sample",
    )


def train_scheduled_expert_kvo_specialized_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_specialized_flow_10k",
        sample_config_key="kv_reference_expert_kvo_specialized_flow_10k_sample",
    )


def sample_expert_kvo_specialized_flow_10k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_specialized_flow_10k_sample",
    )


def train_scheduled_expert_kvo_margin_flow_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_margin_flow_5k",
        sample_config_key="kv_reference_expert_kvo_margin_flow_5k_sample",
    )


def sample_expert_kvo_margin_flow_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_margin_flow_5k_sample",
    )


def train_scheduled_expert_kvo_lossfree_flow_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_lossfree_flow_5k",
        sample_config_key="kv_reference_expert_kvo_lossfree_flow_5k_sample",
    )


def sample_expert_kvo_lossfree_flow_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_lossfree_flow_5k_sample",
    )


def train_scheduled_expert_kvo_artist_invariant_flow_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_artist_invariant_flow_5k",
        sample_config_key=(
            "kv_reference_expert_kvo_artist_invariant_flow_5k_sample"
        ),
    )


def sample_expert_kvo_artist_invariant_flow_5k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key=(
            "kv_reference_expert_kvo_artist_invariant_flow_5k_sample"
        ),
    )


def train_scheduled_expert_kvo_flow_aligned_2k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kvo_flow_aligned_2k",
        sample_config_key="kv_reference_expert_kvo_flow_aligned_2k_sample",
    )


def sample_expert_kvo_flow_aligned_2k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_flow_aligned_2k_sample",
    )


def sample_expert_kvo_flow_aligned_1k_no_o(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kvo_flow_aligned_1k_no_o_sample",
    )


def train_scheduled_expert_kv_teacher_functional_2k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kv_teacher_functional_2k",
        sample_config_key="kv_reference_expert_kv_teacher_functional_2k_sample",
    )


def sample_expert_kv_teacher_functional_2k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kv_teacher_functional_2k_sample",
    )


def train_scheduled_expert_kv_teacher_retrieval_1k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kv_teacher_retrieval_1k",
        sample_config_key="kv_reference_expert_kv_teacher_retrieval_1k_sample",
    )


def sample_expert_kv_teacher_retrieval_1k(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kv_teacher_retrieval_1k_sample",
    )


def sample_expert_kv_teacher_retrieval_250_raw(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kv_teacher_retrieval_250_raw_sample",
    )


def train_scheduled_expert_kv_teacher_single_consistent_500(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_scheduled_direct_reference_kv_delta_320(
        config,
        destination,
        config_key="kv_reference_expert_kv_teacher_single_consistent_500",
        sample_config_key="kv_reference_expert_kv_teacher_single_consistent_500_sample",
    )


def sample_expert_kv_teacher_single_consistent_500(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return sample_direct_reference_kv_delta_320(
        config,
        destination,
        sample_config_key="kv_reference_expert_kv_teacher_single_consistent_500_sample",
    )


def train_functional_reference_kv_operator(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    """Train a visual rank operator in native-attention functional space.

    No dataset-mean K/V operator is subtracted.  The exact rank-16 K/V LoRA
    activation is retained only as a weak auxiliary target; native
    normalization, attention softmax and O define the primary supervised
    effect.  A controlled batch shares context and Q so artist centering is
    meaningful, and Reader pooling remains trainable.
    """

    cfg = copy.deepcopy(config["kv_reference_functional_operator"])
    training = dict(cfg["training"])
    steps = int(steps_override or training.get("steps", 4000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260824))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    validation_artists = int(training.get("validation_artists", 32))
    if not 1 <= validation_artists < len(artist_ids):
        raise ValueError("validation_artists must leave training artists")
    train_artist_count = len(artist_ids) - validation_artists
    teacher_down = teacher_down.to(device=device, dtype=torch.bfloat16)
    teacher_up = teacher_up.to(device=device, dtype=torch.bfloat16)

    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    visual_reader = _load_reader(config, destination, cfg, device)
    reader_parameters = _open_functional_reader_pooling(visual_reader)
    reference_images = int(training.get("materialized_reference_images", 12))
    train_reference_images = int(training.get("train_reference_images", 8))
    if reference_images <= train_reference_images:
        raise ValueError("Reference materialization must reserve validation images")
    chunk = int(training.get("materialization_style_chunk", 16))
    reference_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=artist_ids,
        batch_size=chunk,
        references=reference_images,
        seed=seed ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=True,
    )
    reference_bank = _materialize_reference_token_bank(
        reference_loader,
        artist_ids,
        references=reference_images,
        seed=seed ^ 0x48554D41,
        chunk_size=chunk,
        device=device,
    )

    contexts = load_file(
        destination / str(cfg["text_context_cache"]) / "base.safetensors",
        device="cpu",
    )["base_context"].to(device=device, dtype=torch.bfloat16)
    heldout_contexts = int(training.get("heldout_contexts", 32))
    train_context_count = len(contexts) - heldout_contexts
    if train_context_count <= 0:
        raise ValueError("heldout_contexts leaves no training context")
    probes = _load_native_attention_probes(config, destination, device)

    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", "bilinear_low_rank_operator"))
    if architecture != "bilinear_low_rank_operator":
        raise ValueError("Functional operator experiment requires bilinear operator")
    model = ReferenceConditionedLowRankKVOperator(
        style_dim=int(visual_reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)

    operator_lr = float(training.get("operator_learning_rate", 2e-4))
    reader_lr = float(training.get("reader_learning_rate", 2e-5))
    optimizer = torch.optim.AdamW(
        [
            {"name": "operator", "params": list(model.parameters()), "lr": operator_lr},
            {"name": "reader", "params": reader_parameters, "lr": reader_lr},
        ],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        visual_reader.load_state_dict(state["reader"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    batch = int(training.get("batch_size", 8))
    blocks_per_step = int(training.get("blocks_per_step", 4))
    reference_values = [int(value) for value in training.get("reference_counts", [1, 2, 4])]
    reference_weights = [float(value) for value in training.get("reference_count_weights", [0.5, 0.3, 0.2])]
    attention_queries = int(training.get("attention_queries", 64))
    kv_aux_weight = float(training.get("kv_auxiliary_weight", 0.1))
    functional_weight = float(training.get("functional_weight", 1.0))
    loss_cfg = dict(training.get("functional_loss", {}))
    warmup = int(training.get("warmup_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    reader_max_grad_norm = float(training.get("reader_max_grad_norm", 0.25))
    log_every = int(training.get("log_every", 10))
    validation_every = int(training.get("validation_every", 250))
    checkpoint_every = int(training.get("checkpoint_every", 250))

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "kv-reference-functional-operator")),
            id=str(wandb_cfg.get("id", "kv-reference-functional-operator")),
            resume="allow" if start_step else "never",
            config={"kv_reference_functional_operator": cfg},
        )

    def functional_values(
        style: torch.Tensor,
        context: torch.Tensor,
        indices: torch.Tensor,
        block: int,
        query_seed: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_delta = model(style, context, block)
        teacher_delta = _mixture_target(
            context,
            teacher_down,
            teacher_up,
            indices[:, None],
            torch.ones(len(indices), 1, device=device),
            block,
        )
        probe = probes[block]
        generator = torch.Generator(device=device).manual_seed(query_seed)
        queries = torch.randn(
            len(indices), attention_queries, probe.n_heads, probe.head_dim,
            device=device, dtype=torch.bfloat16, generator=generator,
        )
        zero = torch.zeros_like(student_delta)
        base_key, base_value = probe.project_context(context, zero)
        student_key, student_value = probe.project_context(context, student_delta)
        teacher_key, teacher_value = probe.project_context(context, teacher_delta)
        base_output = probe.attend(queries, base_key, base_value)
        student_effect = probe.attend(queries, student_key, student_value) - base_output
        teacher_effect = probe.attend(queries, teacher_key, teacher_value) - base_output
        functional_loss, values = _functional_centered_attention_loss(
            student_effect,
            teacher_effect,
            centered_huber_weight=float(loss_cfg.get("centered_huber", 1.0)),
            direction_weight=float(loss_cfg.get("direction", 1.0)),
            magnitude_weight=float(loss_cfg.get("magnitude", 0.2)),
            relation_weight=float(loss_cfg.get("relation", 0.5)),
            raw_huber_weight=float(loss_cfg.get("raw_huber", 0.05)),
            temperature=float(loss_cfg.get("temperature", 0.1)),
        )
        kv_loss, kv_values = _normalized_activation_loss(
            student_delta,
            teacher_delta,
            direction_weight=float(training.get("kv_direction_weight", 0.1)),
            magnitude_weight=float(training.get("kv_magnitude_weight", 0.02)),
        )
        values.update({f"kv_{key}": value for key, value in kv_values.items()})
        values["weighted_functional"] = functional_loss.detach() * functional_weight
        values["weighted_kv_auxiliary"] = kv_loss.detach() * kv_aux_weight
        return functional_weight * functional_loss + kv_aux_weight * kv_loss, values

    running: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            rng = random.Random(seed + step * 1_000_003)
            artists = rng.sample(range(train_artist_count), batch)
            counts = rng.choices(reference_values, weights=reference_weights, k=batch)
            references, reference_mask = _select_reference_tokens(
                reference_bank,
                artists,
                reference_counts=counts,
                reference_start=0,
                reference_stop=train_reference_images,
                rng=rng,
            )
            context_index = rng.randrange(train_context_count)
            context = contexts[context_index : context_index + 1].expand(batch, -1, -1)
            artist_tensor = torch.tensor(artists, device=device, dtype=torch.long)
            first_block = ((step - 1) * blocks_per_step) % model.blocks
            selected_blocks = [
                (first_block + offset) % model.blocks
                for offset in range(blocks_per_step)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                style = visual_reader(references, reference_mask).tokens
                rows = [
                    functional_values(
                        style,
                        context,
                        artist_tensor,
                        block,
                        seed + step * 1_000_003 + block * 10_007,
                    )
                    for block in selected_blocks
                ]
                loss = torch.stack([row[0] for row in rows]).mean()
            loss.backward()
            operator_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, foreach=True
            )
            reader_grad = torch.nn.utils.clip_grad_norm_(
                reader_parameters, reader_max_grad_norm, foreach=True
            )
            lr_scale = min(1.0, step / max(1, warmup))
            optimizer.param_groups[0]["lr"] = operator_lr * lr_scale
            optimizer.param_groups[1]["lr"] = reader_lr * lr_scale
            optimizer.step()
            running["loss"].append(float(loss.detach()))
            running["operator_grad_norm"].append(float(operator_grad))
            running["reader_grad_norm"].append(float(reader_grad))
            running["reference_count"].append(sum(counts) / len(counts))
            for key in rows[0][1]:
                running[key].append(
                    sum(float(row[1][key]) for row in rows) / len(rows)
                )
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items()}
                row["operator_lr"] = optimizer.param_groups[0]["lr"]
                row["reader_lr"] = optimizer.param_groups[1]["lr"]
                print(f"Functional K/V operator step={step}/{steps} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()

            if validation_every > 0 and step % validation_every == 0:
                model.eval()
                visual_reader.eval()
                val_rows: dict[str, list[float]] = defaultdict(list)
                val_artists = list(range(train_artist_count, len(artist_ids)))
                val_artists = val_artists[: min(16, len(val_artists))]
                val_counts = [min(4, reference_images - train_reference_images)] * len(val_artists)
                val_rng = random.Random(seed ^ step)
                val_refs, val_mask = _select_reference_tokens(
                    reference_bank,
                    val_artists,
                    reference_counts=val_counts,
                    reference_start=train_reference_images,
                    reference_stop=reference_images,
                    rng=val_rng,
                )
                val_indices = torch.tensor(val_artists, device=device, dtype=torch.long)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    val_style = visual_reader(val_refs, val_mask).tokens
                    for context_index in range(train_context_count, min(len(contexts), train_context_count + 2)):
                        val_context = contexts[context_index : context_index + 1].expand(len(val_artists), -1, -1)
                        for block in range(model.blocks):
                            _, values = functional_values(
                                val_style,
                                val_context,
                                val_indices,
                                block,
                                seed ^ step ^ (context_index * 1009 + block * 9176),
                            )
                            for key, value in values.items():
                                val_rows[key].append(float(value))
                validation = {
                    key: sum(values) / len(values)
                    for key, values in val_rows.items()
                }
                print(f"Functional K/V operator validation step={step} {validation}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"val/{key}": value for key, value in validation.items()}, step=step)
                model.train()
                visual_reader.train()

            if step % checkpoint_every == 0 or step == steps:
                for path in (state_path, checkpoints / f"step-{step:07d}.pt"):
                    _save_training_state(
                        path,
                        step=step,
                        model=model,
                        reader=visual_reader,
                        optimizer=optimizer,
                        cfg=cfg,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "steps": steps,
        "start_step": start_step,
        "artists": len(artist_ids),
        "training_artists": train_artist_count,
        "validation_artists": validation_artists,
        "operator_rank": model.operator_rank,
        "operator_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reader_trainable_parameters": sum(parameter.numel() for parameter in reader_parameters),
        "teacher_decomposition": "none",
        "primary_objective": "native_attention_functional_centered",
        "kv_auxiliary_weight": kv_aux_weight,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_functional_reference_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_functional_operator"]
    cfg["output_directory"] = "kv_reference_functional_operator_smoke"
    cfg["training"]["resume"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["validation_every"] = 0
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["blocks_per_step"] = 1
    cfg["training"]["materialized_reference_images"] = 5
    cfg["training"]["train_reference_images"] = 4
    cfg["training"]["reference_counts"] = [1]
    cfg["training"]["reference_count_weights"] = [1.0]
    return train_functional_reference_kv_operator(
        effective, destination, steps_override=2
    )


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


def train_centered_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_reference_conditioned_kv_activation_generator(
        config,
        destination,
        config_key="kv_reference_centered_bilinear_operator",
    )


def smoke_test_centered_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["kv_reference_centered_bilinear_operator"]
    cfg["output_directory"] = "kv_reference_centered_bilinear_operator_smoke"
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
        config_key="kv_reference_centered_bilinear_operator",
    )


@torch.no_grad()
def sample_reference_conditioned_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render fresh held-out Human references through the learned operator."""

    from .kv_activation_sampling import sample_kv_activation_modulator
    from .kv_generalizing_modulator import _teacher_image_split

    sample_cfg = copy.deepcopy(config["kv_reference_bilinear_sample"])
    device = str(sample_cfg.get("device", "cuda"))
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = dict(checkpoint["config"])
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", ""))
    if architecture != "bilinear_low_rank_operator":
        raise RuntimeError(
            f"Expected a bilinear operator checkpoint, got {architecture!r}"
        )

    artist_ids, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    artist_count = min(int(sample_cfg.get("artists", 7)), len(artist_ids))
    positions = torch.linspace(0, len(artist_ids) - 1, artist_count).round().long()
    selected_indices = [int(value) for value in positions.unique().tolist()]
    selected_ids = [artist_ids[index] for index in selected_indices]
    _, validation_image_ids = _teacher_image_split(
        destination / str(cfg["lora_directory"]), artist_ids
    )

    reader = _load_reader(config, destination, cfg, device)
    if "reader" in checkpoint:
        reader.load_state_dict(checkpoint["reader"], strict=True)
    operator = ReferenceConditionedLowRankKVOperator(
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    operator.load_state_dict(checkpoint["model"], strict=True)
    operator.requires_grad_(False).eval()

    maximum_references = max(
        int(value) for value in sample_cfg.get("reference_counts", [1, 4])
    )
    loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train",
        style_ids=selected_ids,
        batch_size=len(selected_ids),
        references=maximum_references,
        seed=int(sample_cfg.get("seed", 20260824)),
        token_lru_shards=int(sample_cfg.get("token_lru_shards", 8)),
        strict_style_ids=True,
        allowed_image_ids=validation_image_ids,
    )
    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for reference_count_value in sample_cfg.get("reference_counts", [1, 4]):
        reference_count = int(reference_count_value)
        loaded = loader.load_styles(
            selected_ids,
            references_per_style=reference_count,
            seed=int(sample_cfg.get("seed", 20260824))
            + reference_count * 1_000_003,
        )
        references = loaded["tokens"].to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        reference_mask = torch.ones(
            references.shape[:2], device=device, dtype=torch.bool
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            style_memory = reader(references, reference_mask).tokens
            down_rows: list[torch.Tensor] = []
            up_rows: list[torch.Tensor] = []
            for block in range(operator.blocks):
                down, up, sigma = operator._operator(style_memory, block)
                down_rows.append(down)
                up_rows.append(
                    up.transpose(-1, -2) * sigma[:, :, None, :]
                )
        predicted_down = torch.stack(down_rows, dim=1).cpu()
        predicted_up = torch.stack(up_rows, dim=1).cpu()
        compatibility = {
            "predicted_down": predicted_down,
            "predicted_up": predicted_up,
            "predicted_artist_indices": selected_indices,
            "config": {
                "lora_directory": cfg["lora_directory"],
                "blocks": int(cfg.get("blocks", 28)),
            },
        }
        compatibility_path = output / f"reference-{reference_count}-operator.pt"
        torch.save(compatibility, compatibility_path)

        effective = copy.deepcopy(config)
        render_output = output / f"reference-{reference_count}"
        effective["kv_activation_modulator_sample"] = {
            **dict(effective["kv_activation_modulator_sample"]),
            "checkpoint": str(compatibility_path.relative_to(destination)),
            "output_directory": str(render_output.relative_to(destination)),
            "device": device,
            "artist_indices": selected_indices,
            "predicted_strengths": [
                float(value)
                for value in sample_cfg.get("predicted_strengths", [1.0, 1.5])
            ],
            "batch_size": int(sample_cfg.get("batch_size", 4)),
            "panel_tile_width": int(sample_cfg.get("panel_tile_width", 320)),
        }
        rendered = sample_kv_activation_modulator(effective, destination)
        rendered["reference_ids"] = [list(ids) for ids in loaded["ids"]]
        results[f"{reference_count}ref"] = rendered

    del operator, reader, checkpoint, teacher_down, teacher_up
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "checkpoint": str(checkpoint_path),
        "artists": selected_ids,
        "artist_indices": selected_indices,
        "fresh_human_validation_references": True,
        "results": results,
    }
    write_json(output / "summary.json", summary)
    return summary


@torch.no_grad()
def _sample_external_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path, *, config_key: str
) -> dict[str, Any]:
    """Render the established TestSample 1-7 external references."""

    from .dual_query_external_samples import load_dual_query_external_sample
    from .kv_activation_sampling import sample_kv_activation_modulator

    sample_cfg = copy.deepcopy(config[config_key])
    device = str(sample_cfg.get("device", "cuda"))
    checkpoint_path = destination / str(sample_cfg["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = dict(checkpoint["config"])
    model_cfg = dict(cfg["model"])
    architecture = str(model_cfg.pop("architecture", ""))
    if architecture != "bilinear_low_rank_operator":
        raise RuntimeError(
            f"Expected a bilinear operator checkpoint, got {architecture!r}"
        )

    prepared = load_dual_query_external_sample(config, destination)
    references = prepared["reference_tokens"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )[:, None]
    reference_mask = torch.ones(
        references.shape[:2], device=device, dtype=torch.bool
    )
    reader = _load_reader(config, destination, cfg, device)
    if "reader" in checkpoint:
        reader.load_state_dict(checkpoint["reader"], strict=True)
    _, teacher_down, teacher_up = load_kv_lora_factor_bank(
        destination / str(cfg["lora_directory"]),
        blocks=int(cfg.get("blocks", 28)),
        dtype=torch.float16,
    )
    operator = ReferenceConditionedLowRankKVOperator(
        style_dim=int(reader.dim),
        context_dim=int(teacher_down.shape[-1]),
        output_dim=int(teacher_up.shape[-2]),
        blocks=int(teacher_down.shape[1]),
        **model_cfg,
    ).to(device=device, dtype=torch.bfloat16)
    operator.load_state_dict(checkpoint["model"], strict=True)
    operator.requires_grad_(False).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        style_memory = reader(references, reference_mask).tokens
        down_rows: list[torch.Tensor] = []
        up_rows: list[torch.Tensor] = []
        for block in range(operator.blocks):
            down, up, sigma = operator._operator(style_memory, block)
            down_rows.append(down)
            up_rows.append(up.transpose(-1, -2) * sigma[:, :, None, :])
    predicted_down = torch.stack(down_rows, dim=1).cpu()
    predicted_up = torch.stack(up_rows, dim=1).cpu()

    output = destination / str(sample_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    compatibility_path = output / "fixed-reference-operator.pt"
    row_indices = list(range(len(prepared["paths"])))
    compatibility = {
            "predicted_down": predicted_down,
            "predicted_up": predicted_up,
            "predicted_artist_indices": row_indices,
            "config": {
                "lora_directory": cfg["lora_directory"],
                "blocks": int(cfg.get("blocks", 28)),
            },
    }
    if str(cfg.get("teacher_decomposition", "full")) == "centered":
        common_path = checkpoint_path.parent.parent / "frozen_common_operator.pt"
        compatibility["common_operator"] = torch.load(
            common_path, map_location="cpu", weights_only=True
        )["common_operator"]
    torch.save(compatibility, compatibility_path)
    effective = copy.deepcopy(config)
    labels = [f"TestSample {index + 1}" for index in row_indices]
    effective["kv_activation_modulator_sample"] = {
        **dict(effective["kv_activation_modulator_sample"]),
        "checkpoint": str(compatibility_path.relative_to(destination)),
        "output_directory": str(output.relative_to(destination)),
        "device": device,
        "artist_indices": row_indices,
        "artist_labels": labels,
        "predicted_strengths": [
            float(value)
            for value in sample_cfg.get("predicted_strengths", [1.0, 1.5])
        ],
        "batch_size": int(sample_cfg.get("batch_size", 4)),
        "panel_tile_width": int(sample_cfg.get("panel_tile_width", 320)),
        "include_teacher": False,
        "include_reference_images": True,
    }
    rendered = sample_kv_activation_modulator(effective, destination)
    rendered["checkpoint"] = str(checkpoint_path)
    rendered["reference_paths"] = [str(path) for path in prepared["paths"]]
    write_json(output / "summary.json", rendered)
    del operator, reader, checkpoint, teacher_down, teacher_up
    gc.collect()
    torch.cuda.empty_cache()
    return rendered


def sample_external_reference_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_bilinear_fixed_sample",
    )


def sample_external_reference_centered_bilinear_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_centered_bilinear_fixed_sample",
    )


def sample_external_reference_functional_kv_operator(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return _sample_external_reference_bilinear_kv_operator(
        config,
        destination,
        config_key="kv_reference_functional_operator_fixed_sample",
    )
