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
affine-free LayerNorm scaled to the measured nonzero text-token RMS. Anima's
native full-rank O always remains in the path. An optional block-specific
style-only low-rank `ΔO` is additive beside native O; it never replaces or
bottlenecks the pretrained text/style output.

Anima's post-LLM text context is 512 positions, but the separate style softmax
does not require that length. Production uses the 128 meaningful style slots
directly. The learned 128-slot null-style condition used for Style CFG replaces
those slots during dropout; it is not extra attention padding.

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
