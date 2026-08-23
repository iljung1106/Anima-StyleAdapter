from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import read_records, write_json


def _serialize_lora_patterns(value: Any) -> str | None:
    """Match sd-scripts' string-valued network-argument contract."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return repr([str(pattern) for pattern in value])


def _selected_lora_modules(network, training: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(
        str(lora.original_name)
        for lora in network.unet_loras
        if hasattr(lora, "original_name")
    )
    expected_count = training.get("expected_module_count")
    if expected_count is not None and len(names) != int(expected_count):
        raise RuntimeError(
            f"Selected {len(names)} LoRA modules, expected {int(expected_count)}"
        )
    allowed = tuple(
        re.compile(str(pattern))
        for pattern in training.get("required_module_patterns", ())
    )
    invalid = [
        name
        for name in names
        if allowed and not any(pattern.fullmatch(name) for pattern in allowed)
    ]
    if invalid:
        raise RuntimeError(f"Unexpected LoRA modules selected: {invalid[:8]}")
    return names


@dataclass(frozen=True)
class ArtistLoRAPlan:
    index: int
    style_id: str
    artist: str
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ArtistLoRAPlan":
        return cls(
            index=int(row["index"]),
            style_id=str(row["style_id"]),
            artist=str(row["artist"]),
            train_ids=tuple(int(value) for value in row["train_ids"]),
            validation_ids=tuple(int(value) for value in row["validation_ids"]),
        )


@dataclass
class _CpuBucket:
    image_ids: tuple[int, ...]
    latents: Any
    conditions: Any


@dataclass
class _CpuArtistPack:
    plan: ArtistLoRAPlan
    train: dict[tuple[int, int], _CpuBucket]
    validation: dict[tuple[int, int], _CpuBucket]
    read_seconds: float
    bytes_loaded: int


@dataclass
class _GpuBucket:
    image_ids: tuple[int, ...]
    latents: Any
    conditions: Any


def _selection_signature(cfg: dict[str, Any]) -> str:
    fields = {
        "seed": int(cfg.get("seed", 20260823)),
        "artist_count": int(cfg.get("artist_count", 64)),
        "images_per_artist": int(cfg.get("images_per_artist", 30)),
        "train_images_per_artist": int(cfg.get("train_images_per_artist", 24)),
        "validation_images_per_artist": int(
            cfg.get("validation_images_per_artist", 6)
        ),
        "split": str(cfg.get("split", "train")),
        "maximum_bucket_count": int(cfg.get("maximum_bucket_count", 32)),
        "variant_names": list(cfg.get("variant_names", [])),
        "latent_cache": str(cfg["latent_cache"]),
        "text_cache": str(cfg["text_cache"]),
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _artist_in_caption(artist: str, caption: str) -> bool:
    """Reject cached prompts that accidentally contain the artist identity."""
    normalized_caption = re.sub(r"[\s_]+", " ", caption.casefold()).strip()
    normalized_artist = re.sub(r"[\s_]+", " ", artist.casefold()).strip()
    candidates = {normalized_artist}
    # Danbooru disambiguators are part of the identity, but the base name is
    # also unsafe when it appears as an explicit tag in the cached prompt.
    base = re.sub(r"\s*\([^)]*\)\s*$", "", normalized_artist).strip()
    if len(base) >= 4:
        candidates.add(base)
    return any(
        candidate
        and re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", normalized_caption)
        for candidate in candidates
    )


def select_artist_lora_plans(
    latent_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[list[ArtistLoRAPlan], dict[str, Any]]:
    """Select deterministic, leakage-free artists from the cache intersection."""
    split = str(cfg.get("split", "train"))
    artist_count = int(cfg.get("artist_count", 64))
    image_count = int(cfg.get("images_per_artist", 30))
    train_count = int(cfg.get("train_images_per_artist", 24))
    validation_count = int(cfg.get("validation_images_per_artist", 6))
    if train_count + validation_count > image_count:
        raise ValueError("train + validation images exceed images_per_artist")
    required_variants = tuple(str(value) for value in cfg["variant_names"])

    text_by_id: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in text_rows:
        if str(row.get("split", "train")) != split:
            continue
        text_by_id[int(row["id"])][str(row["variant_name"])] = row

    eligible_latents = [
        row for row in latent_rows if str(row.get("split", "train")) == split
    ]
    shape_counts = Counter(
        (int(row["latent_height"]), int(row["latent_width"]))
        for row in eligible_latents
    )
    maximum_bucket_count = max(1, int(cfg.get("maximum_bucket_count", 32)))
    allowed_shapes = {
        shape for shape, _ in shape_counts.most_common(maximum_bucket_count)
    }

    by_style: dict[str, list[int]] = defaultdict(list)
    artist_by_style: dict[str, str] = {}
    leakage_rows = 0
    for row in eligible_latents:
        shape = (int(row["latent_height"]), int(row["latent_width"]))
        if shape not in allowed_shapes:
            continue
        image_id = int(row["id"])
        variants = text_by_id.get(image_id, {})
        if any(name not in variants for name in required_variants):
            continue
        artist = str(row["artist"])
        if any(
            _artist_in_caption(artist, str(variants[name].get("caption", "")))
            for name in required_variants
        ):
            leakage_rows += 1
            continue
        style_id = str(row.get("style_id", artist))
        by_style[style_id].append(image_id)
        artist_by_style[style_id] = artist

    eligible_styles = sorted(
        style_id for style_id, ids in by_style.items() if len(ids) >= image_count
    )
    rng = random.Random(int(cfg.get("seed", 20260823)))
    rng.shuffle(eligible_styles)
    if len(eligible_styles) < artist_count:
        raise RuntimeError(
            f"Only {len(eligible_styles)} artists have {image_count} leakage-free images "
            f"within the top {maximum_bucket_count} latent buckets; need {artist_count}"
        )

    plans = []
    for index, style_id in enumerate(eligible_styles[:artist_count]):
        digest = hashlib.blake2b(
            f"{cfg.get('seed', 20260823)}:{style_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        style_rng = random.Random(int.from_bytes(digest, "little"))
        image_ids = sorted(by_style[style_id])
        style_rng.shuffle(image_ids)
        selected = image_ids[:image_count]
        plans.append(
            ArtistLoRAPlan(
                index=index,
                style_id=style_id,
                artist=artist_by_style[style_id],
                train_ids=tuple(selected[:train_count]),
                validation_ids=tuple(
                    selected[train_count : train_count + validation_count]
                ),
            )
        )
    summary = {
        "artists": len(plans),
        "eligible_artists": len(eligible_styles),
        "images_per_artist": image_count,
        "train_images_per_artist": train_count,
        "validation_images_per_artist": validation_count,
        "latent_buckets": len(allowed_shapes),
        "allowed_shapes": [list(shape) for shape in sorted(allowed_shapes)],
        "leakage_rows_excluded": leakage_rows,
    }
    return plans, summary


class _ArtistCacheIndex:
    def __init__(self, destination: Path, cfg: dict[str, Any]):
        self.latent_root = destination / str(cfg["latent_cache"])
        self.text_root = destination / str(cfg["text_cache"])
        self.latent_rows = read_records(self.latent_root / "manifest.parquet")
        self.text_rows = read_records(self.text_root / "manifest.parquet")
        self.latent_by_id = {int(row["id"]): row for row in self.latent_rows}
        self.text_by_key = {
            (int(row["id"]), str(row["variant_name"])): row
            for row in self.text_rows
        }
        self.variant_names = tuple(str(value) for value in cfg["variant_names"])
        self.conditioning_length = int(cfg.get("text_conditioning_length", 512))
        self.pin_memory = bool(cfg.get("pin_memory", True))

        from safetensors.torch import load_file

        null = load_file(
            self.text_root / "null_conditioning.safetensors", device="cpu"
        )["empty_prompt"]
        if null.ndim != 2:
            raise RuntimeError(f"Unexpected null conditioning shape: {tuple(null.shape)}")
        self.condition_dim = int(null.shape[-1])
        self.null_condition = self._pad_condition(null)

    def _pad_condition(self, value):
        import torch

        if value.shape[0] > self.conditioning_length:
            raise RuntimeError(
                f"Condition length {value.shape[0]} exceeds {self.conditioning_length}"
            )
        output = torch.zeros(
            self.conditioning_length, self.condition_dim, dtype=torch.float16
        )
        output[: value.shape[0]].copy_(value.to(dtype=torch.float16))
        return output

    def load_artist(self, plan: ArtistLoRAPlan) -> _CpuArtistPack:
        import torch
        from safetensors import safe_open

        started = time.perf_counter()
        image_ids = tuple(dict.fromkeys(plan.train_ids + plan.validation_ids))
        latents: dict[int, Any] = {}
        latent_groups: dict[str, list[int]] = defaultdict(list)
        for image_id in image_ids:
            latent_groups[str(self.latent_by_id[image_id]["cache_shard"])].append(
                image_id
            )
        for shard_name, shard_ids in latent_groups.items():
            with safe_open(
                self.latent_root / shard_name, framework="pt", device="cpu"
            ) as handle:
                values = handle.get_slice("latents")
                for image_id in shard_ids:
                    row = self.latent_by_id[image_id]
                    latents[image_id] = values[int(row["row_index"])].clone()

        conditions: dict[int, dict[str, Any]] = defaultdict(dict)
        text_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for image_id in image_ids:
            for variant_name in self.variant_names:
                row = self.text_by_key[(image_id, variant_name)]
                text_groups[str(row["cache_shard"])].append((image_id, variant_name))
        for shard_name, keys in text_groups.items():
            with safe_open(
                self.text_root / shard_name, framework="pt", device="cpu"
            ) as handle:
                values = handle.get_slice("conditioning")
                for image_id, variant_name in keys:
                    row = self.text_by_key[(image_id, variant_name)]
                    start = int(row["token_offset"])
                    length = int(row["token_length"])
                    conditions[image_id][variant_name] = self._pad_condition(
                        values[start : start + length]
                    )

        def make_buckets(ids: tuple[int, ...]) -> dict[tuple[int, int], _CpuBucket]:
            grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
            for image_id in ids:
                value = latents[image_id]
                grouped[(int(value.shape[-2]), int(value.shape[-1]))].append(
                    image_id
                )
            result = {}
            for shape, bucket_ids in grouped.items():
                latent_tensor = torch.stack([latents[value] for value in bucket_ids])
                condition_tensor = torch.stack(
                    [
                        torch.stack(
                            [conditions[value][name] for name in self.variant_names]
                        )
                        for value in bucket_ids
                    ]
                )
                if self.pin_memory and torch.cuda.is_available():
                    latent_tensor = latent_tensor.pin_memory()
                    condition_tensor = condition_tensor.pin_memory()
                result[shape] = _CpuBucket(
                    image_ids=tuple(bucket_ids),
                    latents=latent_tensor,
                    conditions=condition_tensor,
                )
            return result

        train = make_buckets(plan.train_ids)
        validation = make_buckets(plan.validation_ids)
        bytes_loaded = sum(
            bucket.latents.numel() * bucket.latents.element_size()
            + bucket.conditions.numel() * bucket.conditions.element_size()
            for buckets in (train, validation)
            for bucket in buckets.values()
        )
        return _CpuArtistPack(
            plan=plan,
            train=train,
            validation=validation,
            read_seconds=time.perf_counter() - started,
            bytes_loaded=bytes_loaded,
        )


def _to_gpu_buckets(cpu_buckets: dict[tuple[int, int], _CpuBucket], device: str):
    import torch

    return {
        shape: _GpuBucket(
            image_ids=bucket.image_ids,
            latents=bucket.latents.to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            ),
            conditions=bucket.conditions.to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            ),
        )
        for shape, bucket in cpu_buckets.items()
    }


def _reset_lora_network(network, seed: int) -> None:
    import torch

    device = next(network.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        for lora in network.unet_loras:
            torch.nn.init.kaiming_uniform_(
                lora.lora_down.weight, a=math.sqrt(5), generator=generator
            )
            lora.lora_up.weight.zero_()
        for parameter in network.parameters():
            parameter.grad = None


def _compile_anima_blocks(anima, cfg: dict[str, Any]) -> None:
    import torch

    if not bool(cfg.get("enabled", False)):
        return
    cache_limit = int(cfg.get("cache_size_limit", 32))
    torch._dynamo.config.cache_size_limit = cache_limit
    if hasattr(torch._dynamo.config, "force_nn_module_property_static_shapes"):
        torch._dynamo.config.force_nn_module_property_static_shapes = False
    dynamic_value = cfg.get("dynamic", None)
    if dynamic_value in {None, "auto"}:
        dynamic = None
    elif isinstance(dynamic_value, str):
        dynamic = dynamic_value.casefold() in {"1", "true", "yes", "on"}
    else:
        dynamic = bool(dynamic_value)
    for index, block in enumerate(anima.blocks):
        anima.blocks[index] = torch.compile(
            block,
            backend=str(cfg.get("backend", "inductor")),
            mode=str(cfg.get("mode", "default")),
            dynamic=dynamic,
            fullgraph=bool(cfg.get("fullgraph", False)),
        )


def _prompt_probabilities(cfg: dict[str, Any], variant_names: tuple[str, ...]):
    quality = float(cfg.get("quality_probability", 0.5))
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality_probability must be between zero and one")
    mode_weights = dict(cfg["mode_weights"])
    weights = {name: 0.0 for name in variant_names}
    for mode in ("full", "tag_dropout", "short"):
        value = float(mode_weights.get(mode, 0.0))
        weights[mode] += value * (1.0 - quality)
        weights[f"{mode}_quality"] += value * quality
    ordered = [weights[name] for name in variant_names]
    ordered.append(float(mode_weights.get("empty", 0.0)))
    total = sum(ordered)
    if total <= 0:
        raise ValueError("Prompt mode weights sum to zero")
    return [value / total for value in ordered]


def _sample_batch(
    buckets: dict[tuple[int, int], _GpuBucket],
    batch_size: int,
    shape_rng: random.Random,
    generator,
    prompt_probabilities,
    null_condition,
):
    import torch

    shapes = list(buckets)
    shape = shape_rng.choices(
        shapes, weights=[len(buckets[value].image_ids) for value in shapes], k=1
    )[0]
    bucket = buckets[shape]
    indices = torch.randint(
        0,
        bucket.latents.shape[0],
        (batch_size,),
        device=bucket.latents.device,
        generator=generator,
    )
    choices = torch.multinomial(
        prompt_probabilities,
        batch_size,
        replacement=True,
        generator=generator,
    )
    empty_index = bucket.conditions.shape[1]
    condition_indices = choices.clamp_max(empty_index - 1)
    conditions = bucket.conditions[indices, condition_indices]
    # Always select with a tensor expression. Branching on empty.any() would
    # synchronize CUDA once per optimizer step merely to read one scalar.
    empty = choices == empty_index
    conditions = torch.where(
        empty[:, None, None],
        null_condition[None].expand_as(conditions),
        conditions,
    )
    return bucket.latents[indices], conditions


def _flow_loss(anima, latents, conditions, cfg, generator):
    import torch
    import torch.nn.functional as F

    from .style_transfer import _sample_flow_timesteps

    noise = torch.randn(
        latents.shape,
        device=latents.device,
        dtype=latents.dtype,
        generator=generator,
    )
    timesteps = _sample_flow_timesteps(
        latents.shape[0], latents.device, cfg, generator
    )
    sigma = timesteps[:, None, None, None].to(dtype=latents.dtype)
    noisy = (1.0 - sigma) * latents + sigma * noise
    padding_mask = torch.zeros(
        latents.shape[0],
        1,
        latents.shape[-2],
        latents.shape[-1],
        device=latents.device,
        dtype=latents.dtype,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = anima(
            noisy.unsqueeze(2),
            timesteps.to(dtype=latents.dtype),
            context=conditions,
            padding_mask=padding_mask,
            target_input_ids=None,
        ).squeeze(2)
    target = noise - latents
    return F.mse_loss(prediction.float(), target.float())


def _learning_rate_scale(step: int, steps: int, warmup: int, minimum: float) -> float:
    if warmup > 0 and step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return minimum + (1.0 - minimum) * cosine


def _evaluate(
    anima,
    network,
    buckets,
    cfg,
    null_condition,
    prompt_probabilities,
    seed: int,
) -> float:
    import torch

    if not buckets:
        return float("nan")
    network.eval()
    anima.eval()
    generator = torch.Generator(device=next(network.parameters()).device).manual_seed(
        seed
    )
    shape_rng = random.Random(seed)
    losses = []
    with torch.no_grad():
        for _ in range(int(cfg.get("validation_batches", 4))):
            latents, conditions = _sample_batch(
                buckets,
                int(cfg.get("batch_size", 2)),
                shape_rng,
                generator,
                prompt_probabilities,
                null_condition,
            )
            losses.append(
                _flow_loss(anima, latents, conditions, cfg, generator).detach()
            )
    network.train()
    anima.train()
    return float(torch.stack(losses).mean())


def _preview_image(value):
    from PIL import Image

    value = value.detach().float().cpu()
    if value.ndim == 5:
        value = value[0]
    if value.ndim == 4:
        value = value[:, 0]
    pixels = (
        ((value.clamp(-1, 1) + 1.0) * 127.5)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(pixels)


def _preview_panel(images: list[tuple[str, Any]], artist: str, tile_size: int):
    from PIL import Image, ImageDraw, ImageOps

    label_height = 28
    panel = Image.new(
        "RGB",
        (tile_size * len(images), tile_size + label_height + 28),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(images):
        tile = Image.new("RGB", (tile_size, tile_size), (238, 238, 238))
        contained = ImageOps.contain(
            image.convert("RGB"),
            (tile_size, tile_size),
            method=Image.Resampling.LANCZOS,
        )
        tile.paste(
            contained,
            ((tile_size - contained.width) // 2, (tile_size - contained.height) // 2),
        )
        x = index * tile_size
        panel.paste(tile, (x, label_height))
        draw.text((x + 6, 7), label, fill="black")
    draw.text((6, tile_size + label_height + 7), artist, fill="black")
    return panel


def _render_artist_lora_preview(
    anima,
    network,
    train_buckets,
    validation_buckets,
    cache_index: _ArtistCacheIndex,
    null_condition,
    root_config: dict[str, Any],
    artist_config: dict[str, Any],
    destination: Path,
    output: Path,
    plan: ArtistLoRAPlan,
    vae,
):
    """Render a train/held-out/base/LoRA panel with matched prompt and noise."""
    import torch

    from .style_transfer import _load_sampling_vae

    preview_cfg = dict(artist_config.get("preview", {}))
    device = next(network.parameters()).device
    train_bucket = max(train_buckets.values(), key=lambda value: len(value.image_ids))
    validation_bucket = max(
        validation_buckets.values(), key=lambda value: len(value.image_ids)
    )
    variant_name = str(preview_cfg.get("prompt_variant", "full_quality"))
    try:
        variant_index = cache_index.variant_names.index(variant_name)
    except ValueError as error:
        raise ValueError(f"Unknown preview prompt variant: {variant_name}") from error

    positive = validation_bucket.conditions[0, variant_index][None]
    negative = null_condition[None]
    target = validation_bucket.latents[0:1].unsqueeze(2)
    train_target = train_bucket.latents[0:1].unsqueeze(2)
    latent_height, latent_width = target.shape[-2:]
    seed = int(preview_cfg.get("seed", 20260823)) + plan.index * 1009
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_noise = torch.randn(
        target.shape, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    steps = int(preview_cfg.get("steps", 20))
    sigmas = torch.linspace(
        1.0, 0.0, steps + 1, device=device, dtype=torch.float32
    )
    shift = float(preview_cfg.get("flow_shift", 3.0))
    sigmas = (sigmas * shift) / (1.0 + (shift - 1.0) * sigmas)
    text_cfg = float(preview_cfg.get("text_cfg", 4.0))
    padding = torch.zeros(
        2, 1, latent_height, latent_width, device=device, dtype=torch.bfloat16
    )
    context = torch.cat((negative, positive), dim=0)

    def denoise(multiplier: float):
        network.set_multiplier(multiplier)
        value = initial_noise.clone()
        for index in range(steps):
            timestep = sigmas[index].to(torch.bfloat16).expand(2)
            model_input = torch.cat((value, value), dim=0)
            velocity = anima(
                model_input,
                timestep,
                context=context,
                padding_mask=padding,
                target_input_ids=None,
            ).float()
            unconditioned, conditioned = velocity.chunk(2)
            guided = unconditioned + text_cfg * (conditioned - unconditioned)
            value = (
                value.float()
                + guided * (sigmas[index + 1] - sigmas[index])
            ).to(torch.bfloat16)
        return value

    anima.eval()
    network.eval()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            base_latent = denoise(0.0)
            lora_latent = denoise(1.0)
            if vae is None:
                vae = _load_sampling_vae(root_config, destination)
            vae.to(device=device, dtype=torch.bfloat16)
            generated = vae.decode_to_pixels(
                torch.cat((base_latent, lora_latent), dim=0)
            )
            heldout_image = vae.decode_to_pixels(target)
            train_image = vae.decode_to_pixels(train_target)
    finally:
        network.set_multiplier(1.0)
        network.train()
        anima.train()
        if vae is not None:
            vae.to("cpu")
        torch.cuda.empty_cache()

    panel = _preview_panel(
        [
            ("train cache", _preview_image(train_image)),
            ("held-out cache", _preview_image(heldout_image)),
            ("Frozen Anima", _preview_image(generated[0:1])),
            ("rank-16 LoRA", _preview_image(generated[1:2])),
        ],
        plan.artist,
        int(preview_cfg.get("tile_size", 320)),
    )
    preview_dir = output / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"artist-{plan.index:03d}.png"
    temporary = preview_path.with_suffix(".tmp.png")
    panel.save(temporary)
    temporary.replace(preview_path)
    difference = (lora_latent.float() - base_latent.float()).square().mean().sqrt()
    base_rms = base_latent.float().square().mean().sqrt()
    metrics = {
        "preview_latent_delta_rms": float(difference),
        "preview_latent_delta_to_base_ratio": float(
            difference / base_rms.clamp_min(1e-8)
        ),
        "preview_seed": seed,
        "preview_steps": steps,
        "preview_prompt_variant": variant_name,
        "preview_path": str(preview_path),
    }
    return preview_path, vae, metrics


def _snapshot_training_state(
    path: Path,
    network,
    optimizer,
    plan,
    step: int,
    generator,
    shape_rng: random.Random,
) -> None:
    import torch

    state = {
        "style_id": plan.style_id,
        "artist_index": plan.index,
        "step": step,
        "network": {
            key: value.detach().cpu() for key, value in network.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "generator_state": generator.get_state().cpu(),
        "shape_rng_state": shape_rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def prepare_artist_lora_teachers(
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str = "artist_lora_teachers",
) -> dict[str, Any]:
    cfg = config[config_key]
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    signature = _selection_signature(cfg)
    plan_path = output / "plan.json"
    if plan_path.exists():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        if payload.get("signature") == signature:
            return {
                **payload["summary"],
                "output": str(output),
                "plan": str(plan_path),
                "reused": True,
            }
        if list(output.glob("artist-*.safetensors")) and not bool(
            cfg.get("force_replan", False)
        ):
            raise RuntimeError(
                "Artist LoRA selection changed after weights were written; "
                "choose a new output_directory"
            )
    latent_rows = read_records(destination / str(cfg["latent_cache"]) / "manifest.parquet")
    text_rows = read_records(destination / str(cfg["text_cache"]) / "manifest.parquet")
    plans, summary = select_artist_lora_plans(latent_rows, text_rows, cfg)
    payload = {
        "signature": signature,
        "summary": summary,
        "artists": [asdict(plan) for plan in plans],
    }
    write_json(plan_path, payload)
    return {**summary, "output": str(output), "plan": str(plan_path), "reused": False}


def _load_plan(output: Path) -> tuple[str, list[ArtistLoRAPlan]]:
    payload = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    return str(payload["signature"]), [
        ArtistLoRAPlan.from_dict(row) for row in payload["artists"]
    ]


def _safe_artist_filename(plan: ArtistLoRAPlan) -> str:
    digest = hashlib.blake2b(plan.style_id.encode("utf-8"), digest_size=5).hexdigest()
    return f"artist-{plan.index:03d}-{digest}.safetensors"


def train_artist_lora_teachers(
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str = "artist_lora_teachers",
) -> dict[str, Any]:
    import torch

    cfg = config[config_key]
    training = dict(cfg["training"])
    device = str(training.get("device", "cuda"))
    if not device.startswith("cuda"):
        raise RuntimeError("The persistent Anima LoRA trainer requires CUDA")
    prepare_artist_lora_teachers(config, destination, config_key=config_key)
    output = destination / str(cfg["output_directory"])
    plan_signature, plans = _load_plan(output)
    weights_dir = output / "weights"
    metrics_dir = output / "metrics"
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))
    torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
    torch.set_float32_matmul_precision("high")

    from .style_transfer import _optimize_frozen_anima, _resolve_anima_model

    anima = _resolve_anima_model(
        config,
        destination,
        device,
        attn_mode=str(training.get("attention_mode", "torch")),
    ).requires_grad_(False)
    optimization_counts = _optimize_frozen_anima(
        anima,
        low_precision_rmsnorm=bool(training.get("low_precision_rmsnorm", True)),
        # Fusing Q/K/V would change the standard sd-scripts LoRA key contract.
        fuse_attention_projections=False,
    )

    sd_root = Path(str(config["anima_cache"]["sd_scripts_path"])).resolve()
    import sys

    if str(sd_root) not in sys.path:
        sys.path.insert(0, str(sd_root))
    from networks import lora_anima

    rank = int(training.get("rank", 16))
    alpha = float(training.get("alpha", rank))
    network = lora_anima.create_network(
        multiplier=1.0,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=anima,
        neuron_dropout=None,
        train_llm_adapter="false",
        include_patterns=_serialize_lora_patterns(training.get("include_patterns")),
        exclude_patterns=_serialize_lora_patterns(training.get("exclude_patterns")),
    )
    selected_modules = _selected_lora_modules(network, training)
    print(
        f"artist-lora selected_modules={len(selected_modules)} "
        f"first={selected_modules[:4]}",
        flush=True,
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    network.to(device=device)
    network.train()
    anima.train()
    _compile_anima_blocks(anima, dict(training.get("compile", {})))

    parameters = [value for value in network.parameters() if value.requires_grad]
    parameter_count = sum(value.numel() for value in parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.get("learning_rate", 1e-4)),
        betas=tuple(training.get("betas", [0.9, 0.99])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", True)),
    )

    cache_index = _ArtistCacheIndex(destination, cfg)
    null_condition = cache_index.null_condition.to(
        device=device, dtype=torch.bfloat16
    )
    prompt_probabilities = torch.tensor(
        _prompt_probabilities(cfg["prompt"], cache_index.variant_names),
        dtype=torch.float32,
        device=device,
    )
    steps = int(training.get("steps", 500))
    batch_size = int(training.get("batch_size", 2))
    warmup = int(training.get("warmup_steps", 50))
    minimum_lr_ratio = float(training.get("minimum_lr_ratio", 0.1))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every", 25))
    validation_every = int(training.get("validation_every", 250))
    state_every = int(training.get("state_every", 250))
    base_lr = float(training.get("learning_rate", 1e-4))
    active_state_path = output / "active_training_state.pt"

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "artist-lora-teachers-r16-64")),
            id=str(wandb_cfg.get("id", "artist-lora-teachers-r16-64-v1")),
            resume="allow",
            config={
                config_key: cfg,
                "parameter_count": parameter_count,
                "plan_signature": plan_signature,
                "selected_modules": selected_modules,
            },
        )

    summaries = []
    preview_cfg = dict(cfg.get("preview", {}))
    preview_vae = None
    prefetch = ThreadPoolExecutor(max_workers=1)
    pending: Future | None = None
    started_all = time.perf_counter()

    def completed(plan: ArtistLoRAPlan) -> bool:
        return (
            weights_dir / _safe_artist_filename(plan)
        ).exists() and (metrics_dir / f"artist-{plan.index:03d}.json").exists()

    remaining = [plan for plan in plans if not completed(plan)]
    if remaining:
        pending = prefetch.submit(cache_index.load_artist, remaining[0])
    try:
        for position, plan in enumerate(remaining):
            assert pending is not None
            cpu_pack = pending.result()
            pending = (
                prefetch.submit(cache_index.load_artist, remaining[position + 1])
                if position + 1 < len(remaining)
                else None
            )
            transfer_started = time.perf_counter()
            train_buckets = _to_gpu_buckets(cpu_pack.train, device)
            validation_buckets = _to_gpu_buckets(cpu_pack.validation, device)
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_started

            reset_seed = int(cfg.get("seed", 20260823)) + plan.index * 1009
            _reset_lora_network(network, reset_seed)
            optimizer.state.clear()
            for group in optimizer.param_groups:
                group["lr"] = base_lr
            start_step = 0
            if active_state_path.exists():
                resume = torch.load(
                    active_state_path, map_location=device, weights_only=False
                )
                if str(resume.get("style_id")) == plan.style_id:
                    network.load_state_dict(resume["network"])
                    optimizer.load_state_dict(resume["optimizer"])
                    start_step = int(resume["step"])
                    torch.set_rng_state(resume["torch_rng"].cpu())
                    torch.cuda.set_rng_state_all(
                        [state.cpu() for state in resume["cuda_rng"]]
                    )

            generator = torch.Generator(device=device).manual_seed(
                reset_seed + start_step * 100_003
            )
            shape_rng = random.Random(reset_seed + start_step * 97)
            if start_step and "generator_state" in resume:
                generator.set_state(resume["generator_state"].cpu())
            if start_step and "shape_rng_state" in resume:
                shape_rng.setstate(resume["shape_rng_state"])
            running_loss = torch.zeros((), device=device, dtype=torch.float32)
            interval_step_count = 0
            interval_started = time.perf_counter()
            artist_started = time.perf_counter()
            last_validation = float("nan")
            for step in range(start_step + 1, steps + 1):
                lr_scale = _learning_rate_scale(
                    step, steps, warmup, minimum_lr_ratio
                )
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * lr_scale
                optimizer.zero_grad(set_to_none=True)
                latents, conditions = _sample_batch(
                    train_buckets,
                    batch_size,
                    shape_rng,
                    generator,
                    prompt_probabilities,
                    null_condition,
                )
                loss = _flow_loss(anima, latents, conditions, training, generator)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, foreach=True
                )
                optimizer.step()
                running_loss += loss.detach()
                interval_step_count += 1

                if step % log_every == 0 or step == steps:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - interval_started
                    mean_loss = float(running_loss / max(1, interval_step_count))
                    step_seconds = elapsed / max(1, interval_step_count)
                    global_step = plan.index * steps + step
                    print(
                        f"artist-lora artist={plan.index + 1}/{len(plans)} "
                        f"step={step}/{steps} loss={mean_loss:.6f} "
                        f"grad={float(grad_norm):.4f} step_s={step_seconds:.3f} "
                        f"img_s={batch_size / max(step_seconds, 1e-6):.2f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": mean_loss,
                                "train/grad_norm": float(grad_norm),
                                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "system/step_seconds": step_seconds,
                                "system/images_per_second": batch_size
                                / max(step_seconds, 1e-6),
                                "progress/artist_index": plan.index,
                                "progress/artist_step": step,
                            },
                            step=global_step,
                        )
                    running_loss.zero_()
                    interval_step_count = 0
                    interval_started = time.perf_counter()

                if step % validation_every == 0 or step == steps:
                    last_validation = _evaluate(
                        anima,
                        network,
                        validation_buckets,
                        training,
                        null_condition,
                        prompt_probabilities,
                        reset_seed ^ (step * 7919),
                    )
                    print(
                        f"artist-lora validation artist={plan.index + 1}/{len(plans)} "
                        f"step={step} loss={last_validation:.6f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "validation/flow_loss": last_validation,
                                "progress/artist_index": plan.index,
                            },
                            step=plan.index * steps + step,
                        )

                if step % state_every == 0 and step < steps:
                    _snapshot_training_state(
                        active_state_path,
                        network,
                        optimizer,
                        plan,
                        step,
                        generator,
                        shape_rng,
                    )

            weight_path = weights_dir / _safe_artist_filename(plan)
            metadata = {
                "ss_network_module": "networks.lora_anima",
                "ss_network_dim": str(rank),
                "ss_network_alpha": str(alpha),
                "ss_base_model_version": "anima-base-v1.0",
                "anima_style_artist": plan.artist,
                "anima_style_id": plan.style_id,
                "anima_style_steps": str(steps),
                "anima_style_plan_signature": plan_signature,
                "anima_style_lora_config": config_key,
                "anima_style_lora_modules": str(len(selected_modules)),
            }
            temporary_weight_path = weight_path.with_name(
                weight_path.stem + ".tmp.safetensors"
            )
            network.save_weights(str(temporary_weight_path), torch.float16, metadata)
            temporary_weight_path.replace(weight_path)
            preview_metrics: dict[str, Any] = {}
            preview_path = None
            preview_every = max(1, int(preview_cfg.get("every_artists", 1)))
            if bool(preview_cfg.get("enabled", False)) and (
                (plan.index + 1) % preview_every == 0 or plan.index == 0
            ):
                preview_path, preview_vae, preview_metrics = _render_artist_lora_preview(
                    anima,
                    network,
                    train_buckets,
                    validation_buckets,
                    cache_index,
                    null_condition,
                    config,
                    cfg,
                    destination,
                    output,
                    plan,
                    preview_vae,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "qualitative/artist_panel": wandb.Image(
                                str(preview_path), caption=plan.artist
                            ),
                            "qualitative/latent_delta_rms": preview_metrics[
                                "preview_latent_delta_rms"
                            ],
                            "qualitative/latent_delta_to_base_ratio": preview_metrics[
                                "preview_latent_delta_to_base_ratio"
                            ],
                            "progress/artist_index": plan.index,
                        },
                        step=plan.index * steps + steps,
                    )
            artist_summary = {
                "index": plan.index,
                "style_id": plan.style_id,
                "artist": plan.artist,
                "train_ids": list(plan.train_ids),
                "validation_ids": list(plan.validation_ids),
                "steps": steps,
                "batch_size": batch_size,
                "image_exposures": steps * batch_size,
                "validation_flow_loss": last_validation,
                "elapsed_seconds": time.perf_counter() - artist_started,
                "cache_read_seconds": cpu_pack.read_seconds,
                "cache_transfer_seconds": transfer_seconds,
                "cache_bytes": cpu_pack.bytes_loaded,
                "weight_path": str(weight_path),
                **preview_metrics,
            }
            write_json(metrics_dir / f"artist-{plan.index:03d}.json", artist_summary)
            summaries.append(artist_summary)
            if active_state_path.exists():
                active_state_path.unlink()
            del train_buckets, validation_buckets, cpu_pack
            torch.cuda.empty_cache()
    finally:
        prefetch.shutdown(wait=True, cancel_futures=True)
        if wandb_run is not None:
            wandb_run.finish()

    all_metrics = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(metrics_dir.glob("artist-*.json"))
    ]
    result = {
        "artists_completed": len(all_metrics),
        "artists_requested": len(plans),
        "rank": rank,
        "alpha": alpha,
        "parameters": parameter_count,
        "selected_module_count": len(selected_modules),
        "selected_modules": list(selected_modules),
        "optimizer_steps": len(all_metrics) * steps,
        "image_exposures": len(all_metrics) * steps * batch_size,
        "elapsed_seconds_this_run": time.perf_counter() - started_all,
        "optimization_counts": optimization_counts,
        "output": str(output),
    }
    write_json(output / "summary.json", result)
    return result


def smoke_test_artist_lora_teachers(
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str = "artist_lora_teachers",
) -> dict[str, Any]:
    import copy

    effective = copy.deepcopy(config)
    cfg = effective[config_key]
    cfg["output_directory"] = str(cfg["output_directory"]) + "_smoke"
    cfg["artist_count"] = 1
    cfg["images_per_artist"] = min(4, int(cfg.get("images_per_artist", 30)))
    cfg["train_images_per_artist"] = 3
    cfg["validation_images_per_artist"] = 1
    cfg["training"]["steps"] = 2
    cfg["training"]["batch_size"] = 1
    cfg["training"]["warmup_steps"] = 1
    cfg["training"]["log_every"] = 1
    cfg["training"]["validation_every"] = 2
    cfg["training"]["state_every"] = 2
    cfg["training"]["compile"]["enabled"] = False
    cfg["training"]["wandb"]["enabled"] = False
    cfg["preview"]["enabled"] = True
    cfg["preview"]["steps"] = 2
    cfg["force_replan"] = True
    return train_artist_lora_teachers(
        effective, destination, config_key=config_key
    )


def prepare_artist_kv_lora_teachers(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return prepare_artist_lora_teachers(
        config, destination, config_key="artist_kv_lora_teachers"
    )


def train_artist_kv_lora_teachers(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_artist_lora_teachers(
        config, destination, config_key="artist_kv_lora_teachers"
    )


def smoke_test_artist_kv_lora_teachers(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return smoke_test_artist_lora_teachers(
        config, destination, config_key="artist_kv_lora_teachers"
    )
