"""IP-Adapter-like style conditioning for Anima.

The text and style branches share the exact same normalized hidden state and Q
tensor.  Their softmax attentions remain separate, and are combined before the
frozen Anima cross-attention output projection::

    O_native(attn(Q, K_text, V_text) + alpha[block] *
             attn(Q, K_style, V_style))

Style K/V projections are full-rank trainable copies of Anima's native text
K/V projections.  There is deliberately no zero-init output projection, output
LoRA, or second post-text cross-attention call.
"""

from __future__ import annotations

import types
import weakref
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..style_transfer import (
    ConnectorTransformerLayer,
    MinimalSlotSetAggregator,
    SlotSetAggregator,
)


def _copy_linear(
    source: nn.Linear, *, device: torch.device, dtype: torch.dtype,
) -> nn.Linear:
    copied = nn.Linear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        copied.weight.copy_(source.weight.to(device=device, dtype=dtype))
        if source.bias is not None:
            copied.bias.copy_(source.bias.to(device=device, dtype=dtype))
    return copied


def _native_kv_linears(cross_attention: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """Read native K/V even after the frozen-Anima projection fusion pass."""
    if hasattr(cross_attention, "k_proj") and hasattr(cross_attention, "v_proj"):
        return cross_attention.k_proj, cross_attention.v_proj
    if not hasattr(cross_attention, "kv_proj"):
        raise TypeError("Anima cross-attention exposes neither k_proj/v_proj nor kv_proj")

    fused = cross_attention.kv_proj
    if not isinstance(fused, nn.Linear) or fused.out_features % 2:
        raise TypeError("Anima fused kv_proj must be an evenly splittable Linear")
    output_dim = fused.out_features // 2
    key = nn.Linear(
        fused.in_features, output_dim, bias=fused.bias is not None,
        device=fused.weight.device, dtype=fused.weight.dtype,
    )
    value = nn.Linear(
        fused.in_features, output_dim, bias=fused.bias is not None,
        device=fused.weight.device, dtype=fused.weight.dtype,
    )
    with torch.no_grad():
        key.weight.copy_(fused.weight[:output_dim])
        value.weight.copy_(fused.weight[output_dim:])
        if fused.bias is not None:
            key.bias.copy_(fused.bias[:output_dim])
            value.bias.copy_(fused.bias[output_dim:])
    return key, value


def _fallback_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, qkv_format: str,
) -> torch.Tensor:
    if qkv_format != "bshd":
        raise RuntimeError(
            "The native Anima attention backend is required for qkv_format "
            f"{qkv_format!r}"
        )
    attended = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    )
    return attended.transpose(1, 2).flatten(-2)


def _run_attention(
    cross_attention: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_params: Any,
) -> torch.Tensor:
    """Use sd-scripts' selected backend, with SDPA as a test/local fallback."""
    forward = getattr(cross_attention.forward, "__func__", cross_attention.forward)
    backend = getattr(forward, "__globals__", {}).get("attention")
    if backend is not None and hasattr(backend, "attention"):
        return backend.attention([query, key, value], attn_params=attn_params)
    return _fallback_attention(
        query, key, value, str(getattr(cross_attention, "qkv_format", "bshd"))
    )


def _match_native_attention_dtypes(
    query: torch.Tensor,
    text_key: torch.Tensor,
    text_value: torch.Tensor,
    style_key: torch.Tensor,
    style_value: torch.Tensor,
    attn_params: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror Anima Attention.forward's mixed-precision backend contract."""
    if query.dtype == text_value.dtype == style_value.dtype:
        return query, text_key, text_value, style_key, style_value
    requires_low_precision = (
        attn_params is not None
        and (
            not bool(getattr(attn_params, "supports_fp32", True))
            or bool(getattr(attn_params, "requires_same_dtype", False))
        )
        and torch.is_autocast_enabled()
    )
    if not requires_low_precision:
        return query, text_key, text_value, style_key, style_value
    dtype = text_value.dtype
    return (
        query.to(dtype), text_key.to(dtype), text_value,
        style_key.to(dtype), style_value.to(dtype),
    )


class StyleTokenBridge(nn.Module):
    """Full-strength normalized bridge followed by optional residual connector."""

    def __init__(
        self,
        style_dim: int,
        context_dim: int,
        *,
        connector_layers: int,
        connector_heads: int,
        init: str,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(style_dim)
        self.projection = nn.Linear(style_dim, context_dim, bias=False)
        if init == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
        elif init == "orthogonal":
            nn.init.orthogonal_(self.projection.weight)
        else:
            raise ValueError("bridge_init must be 'xavier' or 'orthogonal'")
        self.connector = nn.ModuleList(
            ConnectorTransformerLayer(context_dim, connector_heads)
            for _ in range(connector_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(self.norm(tokens))
        for layer in self.connector:
            tokens = layer(tokens)
        return tokens


class SameQFullRankStyleAdapter(nn.Module):
    """Decoupled text/style attention with shared Q and native full-rank O."""

    def __init__(
        self,
        *,
        style_dim: int = 1024,
        context_dim: int = 1024,
        slots: int = 128,
        heads: int = 16,
        blocks: int = 28,
        alpha_init: float = 0.01,
        style_dropout: float = 0.10,
        aggregator_mode: str = "minimal",
        aggregator_heads: int = 16,
        aggregator_layers: int = 2,
        aggregator_slot_mixer_layers: int = 1,
        aggregator_bottleneck: int = 256,
        connector_layers: int = 2,
        connector_heads: int = 16,
        bridge_init: str = "xavier",
        style_kv_delta_rank: int = 32,
    ) -> None:
        super().__init__()
        if style_dim <= 0 or context_dim <= 0 or slots <= 0 or blocks <= 0:
            raise ValueError("style/context dimensions, slots, and blocks must be positive")
        if context_dim % connector_heads:
            raise ValueError("context_dim must divide connector_heads")
        self.style_dim = int(style_dim)
        self.context_dim = int(context_dim)
        self.slots = int(slots)
        self.heads = int(heads)
        self.blocks = int(blocks)
        self.style_dropout = float(style_dropout)
        self.style_kv_delta_rank = int(style_kv_delta_rank)
        if self.style_kv_delta_rank <= 0:
            raise ValueError("style_kv_delta_rank must be positive")

        if aggregator_mode == "minimal":
            self.aggregator = MinimalSlotSetAggregator(
                slots, style_dim, aggregator_bottleneck
            )
        elif aggregator_mode == "transformer":
            self.aggregator = SlotSetAggregator(
                slots, style_dim, aggregator_heads, aggregator_layers,
                aggregator_slot_mixer_layers,
            )
        else:
            raise ValueError(f"Unknown aggregator_mode: {aggregator_mode}")

        self.null_tokens = nn.Parameter(torch.empty(1, slots, style_dim))
        nn.init.normal_(self.null_tokens, std=0.02)
        self.bridge = StyleTokenBridge(
            style_dim,
            context_dim,
            connector_layers=connector_layers,
            connector_heads=connector_heads,
            init=bridge_init,
        )
        self.alpha = nn.Parameter(torch.full((blocks,), float(alpha_init)))

        # Populated from the real Anima blocks by initialize_from_anima().
        self.style_k = nn.ModuleList()
        self.style_v = nn.ModuleList()
        self.style_k_down = nn.ModuleList()
        self.style_k_up = nn.ModuleList()
        self.style_v_down = nn.ModuleList()
        self.style_v_up = nn.ModuleList()
        self.reference_effect_head = None
        self._initialized = False
        self._style_tokens: torch.Tensor | None = None
        self._style_context: torch.Tensor | None = None
        self._runtime_ratio: dict[int, torch.Tensor] = {}
        self._runtime_raw_ratio: dict[int, torch.Tensor] = {}
        self._alpha_calibration_sums: list[torch.Tensor] | None = None
        self._alpha_calibration_counts: list[int] | None = None

    def initialize_from_anima(self, anima: nn.Module) -> None:
        """Create trainable full-rank K/V copies from all native Anima blocks."""
        if len(anima.blocks) != self.blocks:
            raise ValueError(f"Adapter expects {self.blocks} blocks, Anima has {len(anima.blocks)}")
        if self._initialized:
            return
        keys: list[nn.Linear] = []
        values: list[nn.Linear] = []
        key_down: list[nn.Linear] = []
        key_up: list[nn.Linear] = []
        value_down: list[nn.Linear] = []
        value_up: list[nn.Linear] = []
        parameter = self.bridge.projection.weight
        for index, block in enumerate(anima.blocks):
            cross_attention = block.cross_attn
            if int(cross_attention.n_heads) != self.heads:
                raise ValueError(
                    f"Block {index} has {cross_attention.n_heads} heads, adapter expects {self.heads}"
                )
            native_k, native_v = _native_kv_linears(block.cross_attn)
            expected_output = int(cross_attention.n_heads) * int(cross_attention.head_dim)
            if native_k.out_features != expected_output or native_v.out_features != expected_output:
                raise ValueError(
                    f"Block {index} native K/V width does not match its attention heads"
                )
            if native_k.in_features != self.context_dim or native_v.in_features != self.context_dim:
                raise ValueError(
                    f"Block {index} native context width is {native_k.in_features}, "
                    f"adapter context_dim is {self.context_dim}"
                )
            # Keep the copied trainable weights in the adapter's chosen dtype.
            # The production runner uses FP32 parameters under BF16 autocast so
            # AdamW updates are not rounded away.
            keys.append(
                _copy_linear(native_k, device=parameter.device, dtype=parameter.dtype)
            )
            values.append(
                _copy_linear(native_v, device=parameter.device, dtype=parameter.dtype)
            )
            key_down.append(nn.Linear(
                self.context_dim, self.style_kv_delta_rank, bias=False,
                device=parameter.device, dtype=parameter.dtype,
            ))
            key_up.append(nn.Linear(
                self.style_kv_delta_rank, expected_output, bias=False,
                device=parameter.device, dtype=parameter.dtype,
            ))
            value_down.append(nn.Linear(
                self.context_dim, self.style_kv_delta_rank, bias=False,
                device=parameter.device, dtype=parameter.dtype,
            ))
            value_up.append(nn.Linear(
                self.style_kv_delta_rank, expected_output, bias=False,
                device=parameter.device, dtype=parameter.dtype,
            ))
            nn.init.zeros_(key_up[-1].weight)
            nn.init.zeros_(value_up[-1].weight)
        self.style_k = nn.ModuleList(keys)
        self.style_v = nn.ModuleList(values)
        self.style_k.requires_grad_(False)
        self.style_v.requires_grad_(False)
        self.style_k_down = nn.ModuleList(key_down)
        self.style_k_up = nn.ModuleList(key_up)
        self.style_v_down = nn.ModuleList(value_down)
        self.style_v_up = nn.ModuleList(value_up)
        self._initialized = True

    def aggregate(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        tokens = self.aggregator(references, reference_mask)
        if apply_dropout and self.training and self.style_dropout > 0:
            dropped = torch.rand(tokens.shape[0], device=tokens.device) < self.style_dropout
            tokens = torch.where(
                dropped[:, None, None], self.null_tokens.expand_as(tokens), tokens
            )
        return tokens

    def unconditional(self, batch: int) -> torch.Tensor:
        return self.null_tokens.expand(batch, -1, -1)

    def set_style_tokens(self, tokens: torch.Tensor) -> None:
        parameter = next(self.parameters())
        self._style_tokens = tokens.to(device=parameter.device, dtype=parameter.dtype)
        self._style_context = self.bridge(self._style_tokens)

    def clear_style_tokens(self) -> None:
        self._style_tokens = None
        self._style_context = None

    def bridge_parameters(self) -> list[nn.Parameter]:
        return list(self.bridge.parameters())

    def kv_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.style_k_down.parameters())
            + list(self.style_k_up.parameters())
            + list(self.style_v_down.parameters())
            + list(self.style_v_up.parameters())
        )

    def kv_base_parameters(self) -> list[nn.Parameter]:
        """Frozen native K/V copies that define Anima's context coordinates."""
        return list(self.style_k.parameters()) + list(self.style_v.parameters())

    def alpha_parameters(self) -> list[nn.Parameter]:
        return [self.alpha]

    # These names keep the module easy to wire into the existing training
    # utilities while preserving the new architecture's functional grouping.
    def output_parameters(self) -> list[nn.Parameter]:
        return self.alpha_parameters()

    def gate_parameters(self) -> list[nn.Parameter]:
        return []

    def gate_bootstrap_parameters(self) -> list[nn.Parameter]:
        return self.alpha_parameters()

    def reset_runtime_stats(self) -> None:
        self._runtime_ratio.clear()
        self._runtime_raw_ratio.clear()

    def runtime_stats(self) -> dict[str, float]:
        ratios = (
            torch.stack(list(self._runtime_ratio.values())).float()
            if self._runtime_ratio else torch.zeros(1)
        )
        alpha = self.alpha.detach().float().abs()
        return {
            "style_gate_abs_mean": float(alpha.mean()),
            "style_gate_abs_max": float(alpha.max()),
            "style_block_residual_ratio_mean": float(ratios.mean()),
            "style_block_residual_ratio_max": float(ratios.max()),
            "style_block_raw_ratio_mean": float(
                torch.stack(list(self._runtime_raw_ratio.values())).float().mean()
            ) if self._runtime_raw_ratio else 0.0,
        }

    def begin_alpha_calibration(self) -> None:
        """Collect unscaled style/text RMS ratios without perturbing Anima."""
        device = self.alpha.device
        self._alpha_calibration_sums = [torch.zeros((), device=device) for _ in range(self.blocks)]
        self._alpha_calibration_counts = [0 for _ in range(self.blocks)]
        with torch.no_grad():
            self.alpha.zero_()

    def finish_alpha_calibration(
        self,
        target_ratio: float,
        *,
        minimum_alpha: float = 1e-6,
        maximum_alpha: float = 0.01,
    ) -> dict[str, list[float] | float]:
        if self._alpha_calibration_sums is None or self._alpha_calibration_counts is None:
            raise RuntimeError("Alpha calibration was not started")
        if any(count == 0 for count in self._alpha_calibration_counts):
            raise RuntimeError("Alpha calibration did not observe every Anima block")
        raw = torch.stack([
            total / count
            for total, count in zip(
                self._alpha_calibration_sums,
                self._alpha_calibration_counts,
                strict=True,
            )
        ]).clamp_min(1e-8)
        calibrated = (float(target_ratio) / raw).clamp(
            min=float(minimum_alpha), max=float(maximum_alpha)
        )
        with torch.no_grad():
            self.alpha.copy_(calibrated.to(self.alpha))
        result = {
            "target_style_to_text_ratio": float(target_ratio),
            "raw_style_to_text_ratio": raw.detach().float().cpu().tolist(),
            "alpha": calibrated.detach().float().cpu().tolist(),
        }
        self._alpha_calibration_sums = None
        self._alpha_calibration_counts = None
        return result

    def projected_signature(
        self,
        tokens: torch.Tensor,
        cross_attentions: list[nn.Module] | None = None,
    ) -> torch.Tensor:
        """Compact signature of the exact full-rank style K/V projections."""
        del cross_attentions
        if not self._initialized:
            raise RuntimeError("Adapter must be initialized from Anima before use")
        context = self.bridge(tokens)
        values = []
        for index, (key, value) in enumerate(zip(self.style_k, self.style_v, strict=True)):
            projected_key = key(context) + self.style_k_up[index](
                self.style_k_down[index](context)
            )
            projected_value = value(context) + self.style_v_up[index](
                self.style_v_down[index](context)
            )
            values.extend((projected_key.mean(1), projected_value.mean(1)))
        return torch.cat(values, dim=-1)

    def _style_kv(
        self, block_index: int, cross_attention: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized:
            raise RuntimeError("Adapter must be initialized from Anima before use")
        if self._style_context is None:
            raise RuntimeError("No style tokens are active")
        context = self._style_context
        key = self.style_k[block_index](context) + self.style_k_up[block_index](
            self.style_k_down[block_index](context)
        )
        value = self.style_v[block_index](context) + self.style_v_up[block_index](
            self.style_v_down[block_index](context)
        )
        key = rearrange(
            key, "b ... (h d) -> b ... h d", h=cross_attention.n_heads,
            d=cross_attention.head_dim,
        )
        value = rearrange(
            value, "b ... (h d) -> b ... h d", h=cross_attention.n_heads,
            d=cross_attention.head_dim,
        )
        return cross_attention.k_norm(key), cross_attention.v_norm(value)

    def merged_cross_attention(
        self,
        block_index: int,
        normalized_x: torch.Tensor,
        text_context: torch.Tensor,
        cross_attention: nn.Module,
        attn_params: Any,
    ) -> torch.Tensor:
        """Compute one Q, separate text/style softmaxes, and one native O."""
        if self._style_context is None:
            return cross_attention(normalized_x, attn_params, text_context)

        # compute_qkv is the native Anima path and may have fused frozen K/V.
        # Crucially, Q is projected and normalized exactly once.
        query, text_key, text_value = cross_attention.compute_qkv(
            normalized_x, text_context
        )
        style_key, style_value = self._style_kv(block_index, cross_attention)
        query, text_key, text_value, style_key, style_value = (
            _match_native_attention_dtypes(
                query, text_key, text_value, style_key, style_value, attn_params
            )
        )
        text_attended = _run_attention(
            cross_attention, query, text_key, text_value, attn_params
        )
        style_attended = _run_attention(
            cross_attention, query, style_key, style_value, attn_params
        )
        if self._alpha_calibration_sums is not None:
            # The native O projection can amplify text and visual directions
            # differently even though it is shared. Measure the actual branch
            # outputs during the short calibration pass, not merely attention
            # head coordinates. Runtime uses the cheap pre-O proxy below.
            text_for_ratio = cross_attention.output_proj(text_attended)
            style_for_ratio = cross_attention.output_proj(style_attended)
        else:
            text_for_ratio = text_attended
            style_for_ratio = style_attended
        raw_ratio = (
            style_for_ratio.detach().float().square().mean().sqrt()
            / text_for_ratio.detach().float().square().mean().sqrt().clamp_min(1e-8)
        )
        self._runtime_raw_ratio[block_index] = raw_ratio
        if self._alpha_calibration_sums is not None:
            self._alpha_calibration_sums[block_index].add_(raw_ratio)
            assert self._alpha_calibration_counts is not None
            self._alpha_calibration_counts[block_index] += 1
        style_delta = self.alpha[block_index].to(style_attended.dtype) * style_attended
        merged = text_attended + style_delta
        self._runtime_ratio[block_index] = (
            style_delta.detach().float().square().mean().sqrt()
            / text_attended.detach().float().square().mean().sqrt().clamp_min(1e-8)
        )
        return cross_attention.output_dropout(cross_attention.output_proj(merged))


def _same_q_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    emb: torch.Tensor,
    crossattn_emb: torch.Tensor,
    attn_params: Any,
    use_fp32: bool = False,
    rope_emb_L_1_1_D: torch.Tensor | None = None,
    adaln_lora_B_T_3D: torch.Tensor | None = None,
    extra_per_block_pos_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Anima block forward with same-Q text/style cross-attention."""
    if use_fp32:
        x = x.float()
    if extra_per_block_pos_emb is not None:
        x = x + extra_per_block_pos_emb

    modulation = block.adaln_modulation_self_attn[-1]
    modulation_dtype = modulation.weight.dtype
    emb = emb.to(dtype=modulation_dtype)
    if adaln_lora_B_T_3D is not None:
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype=modulation_dtype)
    with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=use_fp32):
        if block.use_adaln_lora:
            self_mod = block.adaln_modulation_self_attn(emb) + adaln_lora_B_T_3D
            cross_mod = block.adaln_modulation_cross_attn(emb) + adaln_lora_B_T_3D
            mlp_mod = block.adaln_modulation_mlp(emb) + adaln_lora_B_T_3D
        else:
            self_mod = block.adaln_modulation_self_attn(emb)
            cross_mod = block.adaln_modulation_cross_attn(emb)
            mlp_mod = block.adaln_modulation_mlp(emb)
        shift_self, scale_self, gate_self = self_mod.chunk(3, dim=-1)
        shift_cross, scale_cross, gate_cross = cross_mod.chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = mlp_mod.chunk(3, dim=-1)

    expand = lambda value: rearrange(value, "b t d -> b t 1 1 d")
    shift_self, scale_self, gate_self = map(expand, (shift_self, scale_self, gate_self))
    shift_cross, scale_cross, gate_cross = map(expand, (shift_cross, scale_cross, gate_cross))
    shift_mlp, scale_mlp, gate_mlp = map(expand, (shift_mlp, scale_mlp, gate_mlp))
    batch, frames, height, width, _ = x.shape

    normalized = block.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
    result = rearrange(
        block.self_attn(
            rearrange(normalized, "b t h w d -> b (t h w) d"),
            attn_params,
            None,
            rope_emb=rope_emb_L_1_1_D,
        ),
        "b (t h w) d -> b t h w d",
        t=frames,
        h=height,
        w=width,
    )
    x = x + gate_self * result

    normalized = block.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
    adapter = block.__dict__["_style_controller"]()
    result = adapter.merged_cross_attention(
        block.__dict__["_same_q_style_block_index"],
        rearrange(normalized, "b t h w d -> b (t h w) d"),
        crossattn_emb,
        block.cross_attn,
        attn_params,
    )
    result = rearrange(
        result, "b (t h w) d -> b t h w d", t=frames, h=height, w=width
    )
    # Native timestep-conditioned gate_cross controls the complete merged path.
    x = x + gate_cross * result
    normalized = block.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
    return x + gate_mlp * block.mlp(normalized)


def attach_same_q_style_adapter(
    anima: nn.Module, adapter: SameQFullRankStyleAdapter,
) -> None:
    """Copy native K/V and patch every Anima block without modifying old code."""
    adapter.initialize_from_anima(anima)
    anima.style_adapter = adapter
    anima.same_q_style_adapter = adapter
    for index, block in enumerate(anima.blocks):
        if "_same_q_style_original_forward" in block.__dict__:
            raise RuntimeError("A same-Q style adapter is already attached")
        if "_style_original_forward" in block.__dict__:
            raise RuntimeError("Detach the legacy style adapter before attaching same-Q")
        block.__dict__["_same_q_style_original_forward"] = block._forward
        # Reuse the controller slot understood by the existing validation,
        # CFG, and frozen-oracle context managers. The forward implementation
        # remains wholly separate from the legacy adapter.
        block.__dict__["_style_controller"] = weakref.ref(adapter)
        block.__dict__["_same_q_style_block_index"] = index

        def patched(self, *args, **kwargs):
            return _same_q_block_forward(self, *args, **kwargs)

        block._forward = types.MethodType(patched, block)
