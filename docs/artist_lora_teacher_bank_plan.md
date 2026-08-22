# Independent Anima LoRA teacher bank

## Purpose

Native `@artist` conditioning is consistent but too weak to define the final
style-adapter ceiling. Train 64 independent rank-16 Anima LoRAs as stronger
functional teachers. These weights are not a shared linear basis. The student
will later observe the final nonlinear velocity effect of each complete LoRA,
and optional weighted LoRA combinations, under controlled noise, timestep and
content conditions.

## Data

- deterministic 64-artist subset of the human train split
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
- 250 optimizer updates per artist, batch 4, fused AdamW, BF16 autocast
- the total remains 1,000 image exposures per artist; LR is linearly scaled to
  `2e-4` and warmup is shortened to 25 updates
- every completed LoRA logs a four-column W&B panel: VAE-decoded train cache,
  held-out cache, matched Frozen Anima generation and matched LoRA generation

## Throughput design

Anima and one fixed-shape LoRA module graph remain on the H100 for all 64
artists. Artist transitions reinitialize LoRA tensors in place and clear Adam
state, preserving compiled graphs. One artist's complete latent and multimode
text tensors are staged on GPU while the next artist is prefetched to host RAM.
There is no VAE, text encoder, image decode or NFS read inside the 250-step
loop. Gradient checkpointing, block swapping and CPU offload are disabled.
The VAE is loaded lazily only for qualitative panels. The first artist and
then every eighth artist get a panel, limiting inference-graph and VAE cost.
Per-block `torch.compile` remains enabled for the batch-4 production probe;
the run is retained only if its measured image throughput matches or exceeds
the earlier batch-2 compiled path.

The trainer writes a single resumable active state at step 125 and removes it
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
