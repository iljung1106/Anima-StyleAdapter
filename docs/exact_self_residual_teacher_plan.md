# Exact-self residual teacher and heldout distillation

## Stage T: exact-self teacher

- Train the summary-enabled Style Tokenizer from a fresh initialization for 3,000 steps.
- Keep Anima and the cached Dual-query Resampler representation frozen.
- Use the target image itself as the only reference for every training row.
- Optimize flow MSE, normalized target-residual alignment, an absolute aligned
  coefficient floor, and a bounded orthogonal residual. Disable wrong-reference,
  artist-contrastive, subset-consistency, and functional losses in this stage.
- Regress the aligned projection coefficient to 1.0 with a Smooth L1 objective.
  This supervises the full useful residual magnitude and cannot be satisfied by
  increasing an orthogonal output norm. Ramp the auxiliary coefficient floor to
  0.50 as a second, one-sided safeguard against a weak-output shortcut.
- After the v3 500-step diagnostic showed nearly collinear slots and mostly
  orthogonal velocity growth, v4 doubles normalized residual and projection
  supervision, doubles the floor weight, increases slot diversity tenfold, and
  applies the bounded orthogonal check every step with a 0.03 tolerance.
- The aligned projection coefficient is allowed up to 1.25. Do not restore the
  historical 0.25 upper bound in either the teacher or the heldout student: the
  student must be able to reproduce the full detached teacher residual magnitude.
- Validate self, heldout, and wrong-artist references every 250 steps. Render four
  train and four validation targets in both self and heldout modes every 500 steps.

The teacher is usable only if its exact-self residual is target-specific, improves
frozen Anima flow on validation data, has a non-trivial aligned magnitude, and does
not collapse to a shared output. A visually strong but unstable or common residual
is not accepted as a teacher.

## Stage S: heldout student, conditional on Stage T

- Start target exclusion at step 500 and anneal target inclusion from 1 to 0 over
  steps 500--1,000.
- Run the accepted exact-self checkpoint on the same noisy latent, timestep, text
  condition, and target reference. Detach its actual Anima velocity residual
  `teacher_styled_velocity - frozen_base_velocity`.
- Train heldout same-artist references to reproduce both the direction and absolute
  magnitude of that residual. The primary objective is direct tensor regression or
  magnitude-aware projection, not cosine-only similarity or a scale-free ratio.
- Express same-artist consistency, centered artist separation, and common-output
  suppression in residual units normalized only by a detached teacher scale floor.
  Log their raw residual RMS, aligned projection coefficient, orthogonal RMS, and
  weighted contribution separately.
- Keep the teacher, Anima, and Dual-query Resampler frozen. Do not continue if the
  Stage T acceptance gate fails.
