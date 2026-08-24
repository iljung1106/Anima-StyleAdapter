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

## Initial run

- warm-start the visually diverse v34 step-500 direct K/V checkpoint;
- freeze the pretrained Resampler/Reader and reference-independent Common path;
- train shared Style K/V, block-local K/V deltas and block mixing;
- use only individual LoRA teachers for 500 steps;
- then cycle native artist, individual LoRA and weighted LoRA teachers 1:1:1;
- alternate human and LoRA-rendered references;
- sample 1/2/4 references with probabilities 0.50/0.30/0.20;
- save optimizer state every 250 steps and fixed-reference samples every 500.

Acceptance is based on heldout raw-reference samples showing distinct style
effects under identical generation controls. Teacher regression alone is not
sufficient.
