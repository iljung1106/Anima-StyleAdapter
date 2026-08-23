# Independent Anima LoRA teacher bank

## Purpose

Native `@artist` conditioning is consistent but too weak to define the final
style-adapter ceiling. Train 256 independent rank-16 Anima LoRAs as stronger
functional teachers. These weights are not a shared linear basis. The student
will later observe the final nonlinear velocity effect of each complete LoRA,
and optional weighted LoRA combinations, under controlled noise, timestep and
content conditions.

## Data

- deterministic 256-artist subset of the human train split
- 30 cached images per artist: 24 train and 6 held out
- only the 32 most frequent exact latent shapes are eligible
- post-LLM prompts must not contain the artist name
- prompt mixture: Full 30%, tag dropout 40%, short 20%, empty 10%
- quality prefix is present for half of the non-empty prompts

## Teacher training

- Anima, Qwen VAE, Qwen3 and the LLM adapter are frozen
- all target latents and final 512-token conditions are loaded from cache
- standard `sd-scripts` `networks.lora_anima`, rank/alpha 16/16
- default DiT Block LoRA targets: self-attention, cross-attention and MLP;
  modulation, norms, embedders and final layer remain excluded
- rectified-flow MSE, shifted sigmoid timestep distribution with shift 3.0
- 500 optimizer updates per artist, batch 2, fused AdamW, BF16 autocast
- 1,000 image exposures per artist, `1e-4` peak LR and 50-step warmup
- every completed LoRA logs a four-column W&B panel: VAE-decoded train cache,
  held-out cache, matched Frozen Anima generation and matched LoRA generation

## Throughput design

Anima and one fixed-shape LoRA module graph remain on the H100 for all 256
artists. Artist transitions reinitialize LoRA tensors in place and clear Adam
state, preserving compiled graphs. One artist's complete latent and multimode
text tensors are staged on GPU while the next artist is prefetched to host RAM.
There is no VAE, text encoder, image decode or NFS read inside the 250-step
loop. Gradient checkpointing, block swapping and CPU offload are disabled.
The VAE is loaded lazily only for qualitative panels. The first artist and
then every eighth artist get a panel, limiting inference-graph and VAE cost.
It returns to CPU immediately after decoding so its weights do not reduce the
headroom of later high-resolution training buckets.
Per-block `torch.compile` is enabled. A real batch-4/250-step probe reached
only 11–13 images/s while accumulating 73 GB of compiled shape workspaces;
the batch-2 path sustained 12–14 images/s around 32–36 GB, so batch 4 was
rejected rather than assuming larger batches improve this DiT workload.

The trainer writes a single resumable active state at step 250 and removes it
after the corresponding final safetensors and metric record are committed.
Finished artists are skipped on restart.

## Later functional distillation

For identical `x_t`, timestep and content prompt, compute

`T_a = v(Anima + LoRA_a) - v(Anima)`.

Distill this observable effect rather than raw LoRA A/B factors. A mixed
teacher remains valid as

`DeltaW_mix = sum_i lambda_i * (B_i @ A_i)`.

Mixtures are secondary augmentation after the student reproduces individual
teachers; they do not assume that all styles lie in one shared weight basis.

The first 4-content x 4-timestep functional cache made teacher effects almost
orthogonal across unseen content and produced an oracle that changed content
more readily than rendering style. The replacement bank therefore uses 12
distinct cached training conditions and 6 timesteps. It selects only the
well-populated Qwen latent bucket `(H, W) = (128, 96)` and preserves every
latent exactly. Cropping or interpolation is forbidden here because it breaks
the cached caption/latent alignment and injects a second, artificial source of
teacher residual noise.
