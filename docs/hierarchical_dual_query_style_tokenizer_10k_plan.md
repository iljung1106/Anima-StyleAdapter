# Hierarchical Dual-query Style Tokenizer 10k

## Goal

Replace the flat `84 × references` Set Decoder with a target-excluded,
hierarchical 16-token tokenizer. The frozen Dual-query Resampler and its cached
`64 spatial + 16 global + 4 artist-summary` tokens remain unchanged.

## Model

1. Apply separate normalization, projection, and type embeddings to spatial,
   global, and artist-summary tokens.
2. Read every reference independently with 16 learned slot queries.
3. Aggregate corresponding slots across references with masked slot-wise
   attention, preserving the per-image hierarchy and reference-order
   invariance.
4. Add explicit final slot embeddings and emit `16 × 1024` tokens through a
   shallow projection. Initialize near native Anima context scale, but do not
   normalize or constrain output RMS during training.
5. Insert the tokens after the real text length and use frozen Anima's original
   attention K/V/O and shared text CFG. No style gate, separate attention, or
   extra Anima-side connector is used.
6. A training-only decoder weakly reconstructs the selected reference's cached
   spatial/global/summary tokens from its 16 intermediate slots. It is omitted
   at inference.

## Data and curriculum

- Use the complete cached train split for 10,000 optimizer steps.
- Target images never appear among their references.
- Steps 1–500 use 1–2 references; steps 501–2,000 use 1–4; the remainder uses
  1–8. One- and two-reference episodes retain the largest sampling mass.
- Validation uses both image-heldout and artist-heldout splits, with fixed
  1/2/4/8-reference measurements.

## Objective

Flow MSE remains dominant. Auxiliary terms are normalized residual regression,
weak cached-token reconstruction, aligned-slot artist contrastive loss,
same-artist disjoint-subset consistency, reference-dependent slot diversity,
and delayed correct-vs-wrong cyclic flow ranking. Controlled frozen-Anima
forwards add low-weight same-artist residual consistency, centered artist-effect,
and common-output penalties after step 500.

Hard token-RMS constraints, aligned coefficient floors, bounded-effect floors,
and exact-self training are disabled. Raw and weighted loss terms, gradient
norm, functional effect metrics, and token RMS are logged separately.

## Observation schedule

- validation and checkpoint: every 250 and 500 steps respectively;
- train/validation heldout panels: every 500 steps;
- fixed external-reference sheet and controlled 1/2/4/8-reference evaluation:
  every 1,000 steps;
- W&B run: `hierarchical-dual-query-style-tokenizer-v1-10k`.
