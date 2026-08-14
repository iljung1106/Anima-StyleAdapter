# Full-scale IP-Adapter baseline

## Why this run exists

The A2.7--A2.12 experiments kept the style branch deliberately weak, restricted
it to blocks 7--20, and added normalized residual, wrong-reference, or output
direction objectives. They produced an output/base velocity ratio around 2%
but the held-out flow-direction cosine remained near 0.06. Increasing gradient
magnitude through a centered loss did not make the reference-specific direction
improve.

The official IP-Adapter training recipe is materially simpler:

- copy every pretrained text cross-attention K/V into the image branch;
- use a separate image softmax with the same Q;
- add the image attention at scale 1.0 before the pretrained O projection;
- train the image projection and copied full-rank K/V together with AdamW at
  `1e-4`;
- optimize only the ordinary diffusion denoising MSE;
- drop the image condition on 5% of examples.

This experiment reproduces those choices on Anima as closely as the different
DiT architecture permits. It is a baseline, not the final style-only model.

## Fixed architecture

- Frozen Anima backbone and native full-rank O.
- Shared native Q, separate text/style attention softmaxes.
- All 28 cross-attention blocks active.
- Frozen Stage-R Per-reference Resampler, output `128 x 1024`.
- Minimal Set Aggregator; one-reference episodes use its exact bypass.
- One `LayerNorm -> Linear -> LayerNorm` bridge with no connector Transformer.
- Full-rank blockwise style K/V copied from Anima and trainable from step 1.
- Experimental rank-32 K/V deltas and rank-128 output delta disabled.
- Fixed, measured style scale; no learned gate beyond Anima's native
  `gate_cross(t)`. A smoke run showed that literal `alpha=1` gives Anima a
  style-attention RMS 7.2--7.6 times its text-attention RMS and increases flow
  error by 96--127%. Before the first update, alpha is therefore calibrated per
  block so the effective style/text attention RMS ratio is 1.0 (expected alpha
  around 0.13--0.14). This is the Anima-equivalent of IP-Adapter's unit scale,
  not another hand-picked gentle gate.

## Stage and gate

The first run is exact-self only for at most 8,000 optimizer steps. With batch
4 and accumulation 4 this is 128,000 target exposures. Validation is performed
every 250 steps and fixed samples every 500 steps.

Continue to same-artist/different-image training only if all of the following
are true on held-out exact-self targets:

1. paired flow improvement is positive and statistically stable;
2. flow-direction cosine rises materially above the A2.x plateau (~0.06);
3. output magnitude does not grow while paired improvement falls;
4. samples remain structurally valid and visibly respond to the reference.

If this baseline fails, the next change must target the visual-to-context bridge
or representation, not another residual-strength loss. If it passes, use its
checkpoint to start a non-reconstructive curriculum: retain some exact-self
episodes while ramping same-artist target-excluded references, then add the
multi-reference aggregator and null-style CFG training.
