# LoRA functional distillation

## Goal

The completed 64-artist rank-16 LoRA bank is a stronger, image-specific style
teacher than Anima's compressed `@artist` tags. The existing v34
detail-preserving Reader and separated Common/Artist style cross-attention are
kept; this experiment changes the supervision, not the injection architecture.

## Reusable data

1. Generate eight 512x512 content-only images per LoRA (512 images total).
   Artist names are absent from every prompt. Cache the decoded image, Qwen
   latent, C-RADIO features in flight, and the frozen Dual-query Resampler's
   84x1024 tokens.
2. On four fixed contents and four timesteps, cache the actual frozen-Anima
   effect of 64 individual LoRAs, 64 two-LoRA convex mixtures, and 64
   three-LoRA convex mixtures. Two- and three-way targets use one forward with
   all LoRA branches active; they are not approximated by adding final output
   tensors.
3. A mixture's scalar weights are passed to the Reader as reference-attention
   log-bias. Therefore inference can reproduce arbitrary convex mixtures rather
   than inferring weights from duplicated images.

## Training schedule

- Steps 1--500: individual LoRA teacher on every update, alternating human and
  LoRA-generated reference domains.
- After step 500: repeat `@artist teacher -> individual LoRA -> mixed LoRA`, an
  exact 1:1:1 update ratio. Mixed updates use pair, pair, triple in sequence.
- Human and LoRA-generated reference domains continue alternating for both
  individual and mixed LoRA updates.
- The 8,000-step run starts from v34 step 500 model weights with a fresh AdamW
  state. Frozen Anima and the frozen Dual-query Resampler are never optimized.

## Objective

For LoRA updates, use normalized residual Huber, residual direction, a weak
log-RMS match, and centered Huber/direction. A controlled batch shares content,
noise and timestep. Its teacher mean supervises the reusable Common effect;
mean-subtracted effects supervise artist differences and penalize a common
output shortcut. Native `@artist` updates reuse the established final-effect
teacher with a weak all-artist InfoNCE term.

Checkpoint and optimizer state are written every 250 steps. The fixed seven
reference panel is rendered at 1x every 1,000 steps and logged to W&B.
