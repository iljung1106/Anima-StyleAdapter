# Compact Dual-query aligned multi-reference run

## Baseline finding

The completed 8k compact baseline proved that the frozen Dual-query representation
can identify artists, but does not yet turn that information into a sufficiently
artist-specific Anima effect. Its final controlled evaluation reported retrieval
top-1 `1.0`, common-output ratio `0.903`, and only `0.216%` paired-flow improvement
with one held-out reference. Two real references were the strongest measured
condition (`1.212%`), while four and eight references added little.

## Controlled change

Keep the 9.48M compact tokenizer, 16×1024 output, frozen Dual-query Resampler, and
native post-LLM token insertion. Train from a fresh initialization for 10k steps.
References always come from the target artist but exclude the target image.

Reference curriculum:

- steps 1–1,000: one reference only;
- steps 1,001–3,000: 70% one and 30% two references;
- steps 3,001–6,000: one/two references remain 85% of batches;
- steps 6,001–10,000: one/two references remain 75%, with four-reference batches
  more common than uninformative intermediate counts.

Flow MSE remains the primary objective. A weak normalized residual objective is
active from the start. Subset consistency and token contrastive learning begin only
after the single-reference path is established. Correct-vs-wrong flow ranking and
the controlled frozen-Anima functional probe are delayed further. The functional
probe softly rewards disjoint same-artist reference views, discourages a shared
artist-independent output, and preserves centered artist-specific effects. It runs
once per eight optimizer steps to control cost.

No token-RMS floor, aligned-output floor, exact-self reconstruction, or target image
inclusion is used. This avoids making a large but incorrectly directed output look
successful.

## Selection

Every 250 steps record validation and a resumable checkpoint; every 500 steps render
the normal qualitative panel; every 1,000 steps render fixed external references and
run 1/2/4/8-reference plus controlled artist evaluation. Select on all of:

- held-out paired-flow improvement and its confidence interval;
- correct-vs-wrong paired advantage;
- one- and two-reference performance;
- direction cosine and delta/desired RMS;
- common-output ratio and same-artist reference-view consistency;
- fixed-reference images, rejecting checkpoints that merely increase visible but
  reference-independent change.

Only after this pure-token run demonstrates a useful aligned direction should its
tokens initialize a separate copied Anima K/V style-attention stage.
