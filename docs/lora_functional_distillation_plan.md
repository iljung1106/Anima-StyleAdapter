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

## v1 result and v2 correction

The v1 run was stopped after preserving step 2,000. At step 1,000 its student
common-output ratio was 0.927 for individual LoRAs and 0.933 for mixtures,
while the full cached teachers measured only 0.572, 0.646 and 0.677 for
single, pair and triple effects. Centered cosine remained 0.220/0.159 despite
reasonable total RMS. The model therefore learned a sufficiently large but
mostly reference-independent output. This was not an inference-strength bug.

v2 makes the following structural corrections:

- Freeze the v34 Common K/V scaffold for the entire run. LoRA raw regression
  must not update the easiest reference-free path.
- Decompose every controlled batch into its exact teacher common and
  artist-centered residual. Use only a weak common Huber/direction target.
- Put the main weight on per-row normalized centered Huber, centered cosine,
  centered log-RMS, and symmetric all-wrong functional InfoNCE.
- Penalize only the amount by which the student's common/total RMS ratio
  exceeds the same batch's teacher ratio plus a small margin. This preserves
  legitimate shared LoRA behavior without permitting the observed collapse.
- Use eight controlled rows instead of four. Alternate human and LoRA-generated
  references as before.
- Steps 1--1,500 are individual-LoRA only. Steps 1,501--2,000 use two
  individual-LoRA updates per native `@artist` update. Exact 1:1:1
  `@artist : single LoRA : mixed LoRA` training begins only after step 2,000.
  Mixed teachers remain pair, pair, triple.

The v2 run starts again from v34 step 500 with a fresh optimizer. It writes
resume-compatible checkpoints every 250 steps and renders the fixed-reference
panel every 500 steps during the high-risk alignment phase.

## Oracle-code representability bootstrap

The v2 step-500 controlled metrics improved, but fixed-reference 1x and 2x
generation remained visually similar. Doubling Artist strength increased the
baseline change while reducing pairwise output diversity. A direct controlled
comparison also showed that the visual path and the connector were still only
weakly aligned to the same LoRA residual. Jointly learning both sides against
an often unrepresentable full-LoRA final effect is therefore split into two
problems.

First, initialize one 28x1024 code per LoRA artist from the frozen v34 Reader's
average Human/Synthetic output. Freeze the Reader, bypass and freeze Common,
and jointly optimize only these 64 visual-anchored codes plus the Artist K/V
connector. The target is the LoRA effect after removing the controlled
cross-artist mean. Normalized centered Huber/direction/magnitude, symmetric
all-wrong InfoNCE, and a zero-mean Artist penalty define the objective. A weak
code-to-initial-visual-anchor loss prevents arbitrary identity embeddings.

This stage determines the strongest LoRA effect that the actual style
cross-attention can represent before asking an image encoder to predict it.
After its functional cosine and samples are satisfactory, freeze the oracle
connector and distill Human and LoRA-generated references into the learned
codes. Only then jointly fine-tune the visual Reader and connector.
