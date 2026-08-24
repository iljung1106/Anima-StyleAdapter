# Visual K/V LoRA mixture generalization

## Setup

- K/V-only rank-16 LoRA teachers: 320 artists (expanded from 256)
- Fixed comparison dictionaries: 224 old / 288 expanded artists
- Completely held-out artists: the same 32 original artists in every comparison
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

Expanding the dictionary from 224 to 288 candidates improved the fixed-heldout
activation signal only modestly:

| Metric | 224 dictionary | 288 dictionary | Relative change |
|---|---:|---:|---:|
| Activation oracle ridge | 0.3293 | 0.3451 | +4.8% |
| Visual dense ridge, 1 ref | 0.1905 | 0.1960 | +2.9% |
| Visual kNN-8, 1 ref | 0.1596 | 0.1624 | +1.7% |
| Visual dense ridge, 4 refs | 0.1774 | 0.1827 | +3.0% |
| Visual kNN-8, 4 refs | 0.1601 | 0.1630 | +1.8% |

This is real additional span coverage, but the new Reader anchors also create
nearest-neighbour hubs.  Consequently, larger raw-kNN dictionaries can improve
the local activation metric while making the final generated image worse.

## End-to-end generation evidence

On seven heldout artists, exact convex kNN-8 mixtures produced the following
final latent metrics against each artist's exact teacher LoRA:

| References | Strength | Effect ratio | Cosine | Paired improvement |
|---:|---:|---:|---:|---:|
| 1 | 1.0 | 0.821 | 0.518 | +0.093 |
| 4 | 1.0 | 0.824 | 0.567 | +0.148 |
| 4 | 1.5 | 0.936 | 0.612 | +0.160 |

The expanded experiment then reused exactly the same seven heldout artists,
reference images, prompt, seed and anchors across all methods:

| Route | Refs | Strength | Effect ratio | Final cosine | Paired improvement |
|---|---:|---:|---:|---:|---:|
| Old-dictionary convex kNN-8 | 1 | 1.0 | 0.849 | **0.526** | **+0.092** |
| Expanded convex kNN-8 | 1 | 1.0 | 0.842 | 0.498 | +0.074 |
| Signed ridge-32, rank 64 | 1 | 1.0 | 0.837 | 0.470 | +0.047 |
| Old-dictionary convex kNN-8 | 2 | 1.0 | 0.881 | 0.527 | +0.087 |
| Signed ridge-32, rank 64 | 2 | 1.5 | 0.930 | **0.584** | **+0.135** |
| Old-dictionary convex kNN-8 | 4 | 1.5 | 1.017 | **0.595** | +0.093 |
| Signed ridge-32, rank 64 | 4 | 1.5 | 0.938 | 0.569 | **+0.110** |

One reference is too noisy for signed extrapolation.  From two references,
averaging stabilizes the Reader code enough that the affine mixture improves
the final latent effect.  The rank-64 compressed route also uses half the live
rank of exact kNN-8 (rank 128).

Seven unrelated raw reference files were then encoded through the exact cached
C-RADIO → Qwen VAE → frozen Resampler → Reader path.  Their retrieved kNN sets
were genuinely different (45 unique LoRAs among 56 selections, mean pairwise
Jaccard 0.043), yet the final effects still shared a large component:

| Common-LoRA scale | Effect RMS | Common-output ratio | Centered/effect | Pairwise effect cosine |
|---:|---:|---:|---:|---:|
| 1.00 | 0.335 | 0.800 | 0.622 | 0.561 |
| 0.50 | 0.291 | 0.765 | 0.645 | 0.518 |
| 0.25 | 0.272 | 0.797 | 0.611 | 0.563 |
| 0.00 | 0.247 | **0.745** | **0.699** | **0.442** |

Removing the compressed dictionary mean improves diversity but does not solve
style matching: individual teacher LoRAs themselves contain correlated effects,
and the current random 320-artist basis does not cover arbitrary raw references
well enough.  The next expansion therefore uses four-reference Reader-code
cosine k-center over all 2,185 eligible artists, while reusing the existing 320
weights exactly.  This tests added *coverage*, rather than adding more redundant
random teachers or more mixtures inside the same span.

## Decision

This retrieval experiment is retained for analysis only and is rejected as a
production inference path. Production must generate per-block Style K/V from
the supplied references without loading, retrieving or combining a LoRA
dictionary. Individual and weighted LoRA effects may still be used as offline
functional teachers for that direct generator.

Multi-artist weighted sums contain useful generalization information in a
precise but limited sense.  They densely sample combinations inside the
functional span of the artist LoRAs and teach/validate coefficient composition;
they do **not** create a direction outside that span.  The visual Reader supplies
the unseen-reference-to-coefficient signal, while the LoRA dictionary supplies
the available style basis.  Improving unrestricted coverage therefore needs
more diverse, visually well-spaced teacher LoRAs or a separately validated
raw-reference residual correction—not merely more synthetic convex mixtures.
