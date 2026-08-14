# Same-Q full-rank-O Anima style adapter

This module is deliberately separate from the legacy `SharedLowRankStyleAdapter`.

Data flow per Anima block:

1. Compute the native cross-attention-normalized image state once.
2. Compute the native Anima Q once.
3. Run separate text and style softmax attentions with that exact Q.
4. Combine `text + alpha[block] * style` before Anima's frozen full-rank
   `output_proj`.
5. Apply Anima's native timestep-conditioned `gate_cross` once.

Style K/V matrices start as full-rank copies of native text K/V. Their bases
can be opened per active block at a low learning rate; block-specific low-rank
deltas are available for cheaper corrections. The bridge ends with an
affine-free LayerNorm scaled to the measured nonzero text-token RMS. There is
no terminal `o_down/o_up` bottleneck: Anima's native full-rank O remains the
only output projection.

Anima's post-LLM text context is always 512 positions and cross-attention gets
no padding mask. A 128-slot style representation may therefore append 384
exact zero context vectors *after* the learned bridge. These are attention
padding, not the learned 128-slot null-style condition used for Style CFG.

Attach before constructing the optimizer so that copied K/V parameters are
included:

```python
adapter = SameQFullRankStyleAdapter(
    style_dim=1024, context_dim=1024, slots=128, blocks=28,
    alpha_init=0.01, bridge_init="xavier",
).to(device="cuda", dtype=torch.bfloat16)
attach_same_q_style_adapter(anima, adapter)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
```

Do not use a gate-only warm-up with this architecture. `alpha=0.01` is already
the nonzero safety scale, so bridge, connector, K/V, and alpha should all be in
the optimizer from step one. Keep Anima and its native Q/O frozen.
