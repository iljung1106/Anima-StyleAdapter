# Direct reference-conditioned K/V distillation

## Inference contract

The production adapter must generate Anima Style K/V from the supplied visual
references. Inference is strictly

`reference images -> cached visual features -> frozen Resampler -> typed Reader -> per-block Style K/V`.

It does not load an artist LoRA bank, artist identifier, nearest-neighbour
index, mixture coefficient, or retrieved LoRA factor. The separate style
attention reuses frozen Anima Q/O, while style and text have separate softmax.

## Training supervision

LoRAs are allowed only as offline teachers. For matched `x_t`, timestep and
content, training observes the final functional residual made by:

- one artist LoRA;
- a weighted two- or three-LoRA teacher;
- the native Anima `@artist` path.

The weighted teachers densify the functional style space; they are never
reconstructed by selecting or combining LoRAs at inference. The student must
produce the effect through its own reference-conditioned K/V.

## Fresh joint-teacher run

- do not warm-start v34 or any prior Style Adapter;
- load only the independently reconstruction-pretrained typed Reader;
- initialize the fresh adapter's 28-block x 4-timestep alpha table from the
  previously measured native/raw residual ratio, without loading any adapter
  tensor; alpha=1 made the step-250 output 17.7x too large and pure noise;
- calibrate the structurally separate fresh Common and Artist components
  independently: the first calibrated fresh forward placed Artist at 3-6x
  native scale while Common remained at 0.2-0.7x, so a single shared scalar
  could not correct both;
- render both 1x and Artist-only 2x fixed-reference panels every 250 steps;
  Common stays at 1x so the sweep distinguishes weak Artist signal from a
  reference-independent Common collapse;
- freshly initialize and train Common K/V, shared Style K/V, block-local K/V
  deltas, block mixing and the artist-null residual;
- give the isolated native Common objective a nonzero full weight; disabling
  it leaves the normalized random Common K/V output unable to calibrate itself;
- alternate individual LoRA and native Anima artist teachers 1:1 for the first
  1,500 updates;
- then cycle native artist, individual LoRA and weighted LoRA teachers 1:1:1;
- alternate human and LoRA-rendered references;
- sample 1/2/4 references with probabilities 0.50/0.30/0.20;
- compute weighted-LoRA targets with actual frozen-Anima forwards and never
  expose mixture coefficients to the student;
- save optimizer state and fixed-reference samples every 250 steps.

Acceptance is based on heldout raw-reference samples showing distinct style
effects under identical generation controls. Teacher regression alone is not
sufficient.
