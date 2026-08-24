# Visual K/V LoRA mixture generalization

## Setup

- K/V-only rank-16 LoRA teachers: 256 artists
- Dictionary/train artists: 224
- Completely held-out artists: 32
- Teacher-train and heldout-reference images are disjoint
- Frozen visual path: Dual-query Resampler Reader, 28×1024 tokens
- Functional comparison: 28 Anima blocks, heldout text contexts, native K/V activation
- Inference mixture: exact factor concatenation, not factor-coordinate averaging

Convex two/three-artist mixtures retain respectively 97.1% and 95.0% of
their functional matrix energy at rank 16 on average.  They are therefore a
valid compositional auxiliary target, but they add no new basis direction
outside the span of the teacher LoRAs.

## Heldout functional results

| Method | 1 reference cosine | 4 reference cosine | Notes |
|---|---:|---:|---|
| Activation oracle ridge | 0.329 | — | Uses heldout teacher activation; upper bound, not deployable |
| Visual dense signed ridge (224) | 0.191 | 0.177 | Expensive dense affine mixture |
| Visual sparse signed ridge (32) | 0.178 | 0.169 | Rank up to 512 plus common base |
| Visual sparse signed ridge (16) | 0.160 | 0.158 | No meaningful gain over convex kNN-8 |
| Visual convex kNN-8 | 0.160 | 0.160 | Rank 128, stable exact mixture |

The learned leave-one-out metric did not beat the raw Reader metric.  Its best
heldout checkpoint remained step 0.  This rejects the direct learned selector,
not the LoRA-mixture path: the raw metric already selects functionally useful
neighbors, while only 224 independent teacher points are available to learn a
new metric.

## End-to-end generation evidence

On seven heldout artists, exact convex kNN-8 mixtures produced the following
final latent metrics against each artist's exact teacher LoRA:

| References | Strength | Effect ratio | Cosine | Paired improvement |
|---:|---:|---:|---:|---:|
| 1 | 1.0 | 0.821 | 0.518 | +0.093 |
| 4 | 1.0 | 0.824 | 0.567 | +0.148 |
| 4 | 1.5 | 0.936 | 0.612 | +0.160 |

The visual panels also show substantially stronger artist-specific changes
than the direct visual-to-factor hypernetwork.  More references improve the
functional and visual match.

## Decision

Use raw Reader cosine with eight train-artist neighbors as the stable coarse
few-shot K/V path.  Concatenate their weighted rank-16 factors to obtain an
exact rank-128 LoRA sum.  Do not use the learned selector or the direct
bilinear factor hypernetwork.  Sparse signed mixtures are retained only as an
analysis baseline: their small gain at 32 neighbors does not justify four
times the rank and greater extrapolation risk.

This mechanism provides interpolation/generalization inside the visual and
functional coverage of the LoRA dictionary.  It cannot by itself teach an
unrestricted unseen style direction.  Improving coverage therefore requires
more diverse teacher LoRAs or a separately validated residual correction on
top of the stable mixture, not merely more synthetic convex mixtures.
