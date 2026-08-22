from __future__ import annotations

import copy
import gc
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file

from .artist_lora_teachers import ArtistLoRAPlan
from .detail_style_cross_attention import (
    DetailPreservingTypedSlotReader,
    SeparatedCommonArtistKVStyleCrossAttention,
)
from .detail_style_teacher_context import NativeArtistContextCache
from .detail_style_training import (
    _build_style_adapter,
    _controlled_teacher_forward,
    _generate_fixed_reference_sample,
    _save_state,
    _teacher_step,
)
from .dual_query_external_samples import load_dual_query_external_sample
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .io import read_records, write_json, write_records
from .native_centered_teacher import NativeCenteredTeacherBank
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_transfer import _optimize_frozen_anima, _resolve_anima_model
from .synthetic_teacher import (
    _load_sampling_vae,
    _sample_anima_batch,
)


@dataclass(frozen=True)
class MixtureSpec:
    index: int
    kind: str
    components: tuple[int, ...]
    weights: tuple[float, ...]


def teacher_category(step: int, *, single_only_steps: int) -> str:
    """Dense individual-LoRA alignment, followed by an exact 1:1:1 cycle."""

    if step <= 0:
        raise ValueError("step must be positive")
    if step <= int(single_only_steps):
        return "lora_single"
    return ("artist_tag", "lora_single", "lora_mixture")[
        (step - int(single_only_steps) - 1) % 3
    ]


def teacher_category_v2(
    step: int,
    *,
    single_only_steps: int,
    artist_intro_steps: int,
) -> str:
    """Center individual LoRA effects before introducing mixed teachers.

    The middle stage keeps two individual-LoRA updates for every native artist
    update.  Convex LoRA mixtures have a larger cross-artist common component,
    so they are deliberately withheld until the artist residual has had enough
    direct supervision.
    """

    if step <= 0:
        raise ValueError("step must be positive")
    single_only_steps = int(single_only_steps)
    artist_intro_steps = int(artist_intro_steps)
    if step <= single_only_steps:
        return "lora_single"
    if step <= single_only_steps + artist_intro_steps:
        return ("lora_single", "artist_tag", "lora_single")[
            (step - single_only_steps - 1) % 3
        ]
    return ("artist_tag", "lora_single", "lora_mixture")[
        (step - single_only_steps - artist_intro_steps - 1) % 3
    ]


def build_mixture_specs(
    artists: int,
    *,
    pair_count: int,
    triple_count: int,
    seed: int,
) -> list[MixtureSpec]:
    if artists < 3:
        raise ValueError("At least three LoRA artists are required")
    specs = [
        MixtureSpec(index, "single", (index,), (1.0,))
        for index in range(artists)
    ]
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()

    def sample_components(count: int) -> tuple[int, ...]:
        for _ in range(10_000):
            values = tuple(sorted(rng.sample(range(artists), count)))
            if values not in seen:
                seen.add(values)
                return values
        raise RuntimeError("Could not construct enough unique LoRA mixtures")

    for _ in range(pair_count):
        components = sample_components(2)
        left = rng.uniform(0.30, 0.70)
        specs.append(
            MixtureSpec(len(specs), "pair", components, (left, 1.0 - left))
        )
    for _ in range(triple_count):
        components = sample_components(3)
        raw = [rng.uniform(0.25, 1.0) for _ in range(3)]
        total = sum(raw)
        specs.append(
            MixtureSpec(
                len(specs),
                "triple",
                components,
                tuple(value / total for value in raw),
            )
        )
    return specs


def decompose_teacher_effects(
    effects: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if effects.ndim < 2 or effects.shape[0] < 2:
        raise ValueError("Teacher decomposition needs at least two rows")
    common = effects.mean(dim=0, keepdim=True)
    return common, effects - common


def _load_lora_plan(root: Path) -> list[ArtistLoRAPlan]:
    payload = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    return [ArtistLoRAPlan.from_dict(row) for row in payload["artists"]]


def _weight_paths(root: Path, plans: list[ArtistLoRAPlan]) -> list[Path]:
    paths = []
    for plan in plans:
        matches = sorted((root / "weights").glob(f"artist-{plan.index:03d}-*.safetensors"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one LoRA for artist {plan.index}, found {len(matches)}")
        paths.append(matches[0])
    return paths


def _create_lora_networks(
    config: dict[str, Any], anima: torch.nn.Module, count: int, device: str
) -> list[torch.nn.Module]:
    sd_root = Path(str(config["anima_cache"]["sd_scripts_path"])).resolve()
    if str(sd_root) not in sys.path:
        sys.path.insert(0, str(sd_root))
    from networks import lora_anima

    networks = []
    for _ in range(count):
        network = lora_anima.create_network(
            multiplier=0.0,
            network_dim=16,
            network_alpha=16.0,
            vae=None,
            text_encoders=[],
            unet=anima,
            neuron_dropout=None,
            train_llm_adapter="false",
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        network.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
        networks.append(network)
    return networks


def _load_content_conditions(root: Path) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    text_root = root / "text"
    rows = [
        row for row in read_records(text_root / "manifest.parquet")
        if str(row["kind"]) == "content"
    ]
    rows.sort(key=lambda row: int(row["condition_id"]))
    shards: dict[str, torch.Tensor] = {}
    values = []
    for row in rows:
        name = str(row["cache_shard"])
        if name not in shards:
            shards[name] = load_file(text_root / name, device="cpu")["conditioning"]
        values.append(shards[name][int(row["row_index"])])
    negative = load_file(text_root / "negative.safetensors", device="cpu")["conditioning"]
    return rows, torch.stack(values), negative


def _preview_pixels(values: torch.Tensor) -> list[Image.Image]:
    if values.ndim == 5:
        values = values[:, :, 0]
    pixels = ((values.float().clamp(-1, 1) + 1) * 127.5).byte()
    pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(value, mode="RGB") for value in pixels]


def generate_lora_teacher_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["lora_teacher_references"])
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "latents").mkdir(exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    lora_root = destination / str(cfg["lora_directory"])
    plans = _load_lora_plan(lora_root)
    weights = _weight_paths(lora_root, plans)
    source = destination / str(cfg["content_source_directory"])
    content_rows, conditions, negative = _load_content_conditions(source)
    images_per_artist = int(cfg.get("images_per_artist", 8))
    if len(content_rows) < images_per_artist:
        raise RuntimeError("The source text bank has too few content prompts")
    content_rows = content_rows[:images_per_artist]
    conditions = conditions[:images_per_artist]
    width, height = int(cfg.get("width", 512)), int(cfg.get("height", 512))
    device = str(cfg.get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=False)
    network = _create_lora_networks(config, anima, 1, device)[0]
    vae = _load_sampling_vae(config, destination).to(device=device, dtype=torch.bfloat16)
    vae.requires_grad_(False).eval()
    steps = int(cfg.get("steps", 20))
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    text_cfg = float(cfg.get("text_cfg", 4.0))
    positive = conditions.to(device=device, dtype=torch.bfloat16)
    negative = negative.to(device=device, dtype=torch.bfloat16).expand(images_per_artist, -1, -1)
    completed_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for artist_index, (plan, weight_path) in enumerate(zip(plans, weights, strict=True)):
        part_path = output / "manifests" / f"part-{artist_index:05d}.parquet"
        if part_path.exists():
            completed_rows.extend(read_records(part_path))
            continue
        info = network.load_weights(str(weight_path))
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(f"LoRA key mismatch for {weight_path}: {info}")
        network.set_multiplier(1.0)
        seeds = [
            int(cfg.get("seed", 20260823)) + artist_index * 100_003 + index * 1009
            for index in range(images_per_artist)
        ]
        noise = torch.stack([
            torch.randn(
                16, 1, height // 8, width // 8,
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device, dtype=torch.bfloat16,
            )
            for seed in seeds
        ])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latents = _sample_anima_batch(
                anima, noise, positive, negative, sigmas,
                text_cfg=text_cfg, speed=None, generation_seeds=seeds,
            )
            decoded = vae.decode_to_pixels(latents)
        images = _preview_pixels(decoded)
        latent_values = latents[:, :, 0].to("cpu", dtype=torch.float16).contiguous()
        latent_name = f"part-{artist_index:05d}.safetensors"
        save_file({"latents": latent_values}, output / "latents" / latent_name)
        artist_dir = output / "images" / f"artist-{artist_index:03d}"
        artist_dir.mkdir(exist_ok=True)
        rows = []
        for content_index, (image, source_row, seed) in enumerate(
            zip(images, content_rows, seeds, strict=True)
        ):
            image_path = artist_dir / f"content-{content_index:02d}.webp"
            image.save(image_path, format="WEBP", quality=int(cfg.get("webp_quality", 95)))
            rows.append({
                "id": int(cfg.get("image_id_base", 30_000_000_000)) + artist_index * images_per_artist + content_index,
                "kind": "artist",
                "artist_index": artist_index,
                "artist": plan.artist,
                "style_id": plan.style_id,
                "artist_split": "train",
                "split": "train",
                "content_index": content_index,
                "generation_seed": seed,
                "content_prompt": str(source_row["prompt"]),
                "artist_prompt": str(source_row["prompt"]),
                "artist_tag": "",
                "local_path": str(image_path.resolve()),
                "width": width,
                "height": height,
                "latent_height": height // 8,
                "latent_width": width // 8,
                "steps": steps,
                "text_cfg": text_cfg,
                "flow_shift": shift,
                "attention_backend": str(cfg.get("attention_mode", "torch")),
                "latent_shard": latent_name,
                "latent_row": content_index,
            })
        write_records(part_path, rows)
        completed_rows.extend(rows)
        print(
            f"LoRA references artist={artist_index + 1}/{len(plans)} "
            f"images={len(completed_rows)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    completed_rows.sort(key=lambda row: int(row["id"]))
    write_records(output / "manifest.parquet", completed_rows)
    summary = {
        "artists": len(plans),
        "images": len(completed_rows),
        "images_per_artist": images_per_artist,
        "elapsed_s": time.perf_counter() - started,
        "content_artist_tags": False,
    }
    write_json(output / "generation_summary.json", summary)
    del anima, network, vae
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _source_probe_bank(
    source: Path, contents: int
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    manifest = [
        row for row in read_records(source / "manifest.parquet")
        if str(row["kind"]) == "content_control" and int(row["seed_index"]) == 0
    ]
    by_content = {int(row["content_index"]): row for row in manifest}
    selected = [by_content[index] for index in range(contents)]
    shards: dict[str, torch.Tensor] = {}
    values = []
    for row in selected:
        name = str(row["latent_shard"])
        if name not in shards:
            shards[name] = load_file(source / "latents" / name, device="cpu")["latents"]
        values.append(shards[name][int(row["latent_row"])])
    return torch.stack(values), selected


def cache_lora_functional_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["lora_functional_distillation"])
    cache_cfg = dict(cfg["teacher_cache"])
    output = destination / str(cache_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists() and (output / "base.safetensors").exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("complete_mixtures", 0)) == int(summary.get("mixtures", -1)):
            return {**summary, "reused": True}
    lora_root = destination / str(cfg["lora_directory"])
    plans = _load_lora_plan(lora_root)
    weight_paths = _weight_paths(lora_root, plans)
    specs = build_mixture_specs(
        len(plans),
        pair_count=int(cache_cfg.get("pair_mixtures", 64)),
        triple_count=int(cache_cfg.get("triple_mixtures", 64)),
        seed=int(cache_cfg.get("seed", 20260823)),
    )
    write_records(output / "mixtures.parquet", [
        {
            "index": spec.index,
            "kind": spec.kind,
            "components": list(spec.components),
            "weights": list(spec.weights),
            "style_ids": [plans[index].style_id for index in spec.components],
        }
        for spec in specs
    ])
    source = destination / str(cache_cfg["content_source_directory"])
    contents = int(cache_cfg.get("contents", 4))
    timesteps = [float(value) for value in cache_cfg.get("timesteps", [0.2, 0.45, 0.7, 0.9])]
    latents, _ = _source_probe_bank(source, contents)
    _, contexts, _ = _load_content_conditions(source)
    contexts = contexts[:contents]
    device = str(cache_cfg.get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=False)
    networks = _create_lora_networks(config, anima, 3, device)
    latent_device = latents.to(device=device, dtype=torch.bfloat16)
    context_device = contexts.to(device=device, dtype=torch.bfloat16)
    noisy_rows = []
    context_rows = []
    timestep_rows = []
    seed = int(cache_cfg.get("seed", 20260823))
    for content_index in range(contents):
        for timestep_index, timestep in enumerate(timesteps):
            noise = torch.randn(
                latent_device[content_index].shape,
                generator=torch.Generator(device=device).manual_seed(
                    seed + content_index * 100_003 + timestep_index * 1009
                ),
                device=device, dtype=torch.bfloat16,
            )
            noisy_rows.append((1 - timestep) * latent_device[content_index] + timestep * noise)
            context_rows.append(context_device[content_index])
            timestep_rows.append(timestep)
    noisy_flat = torch.stack(noisy_rows)
    context_flat = torch.stack(context_rows)
    timestep_flat = torch.tensor(timestep_rows, device=device, dtype=torch.bfloat16)
    padding = torch.zeros(
        len(noisy_flat), 1, noisy_flat.shape[-2], noisy_flat.shape[-1],
        device=device, dtype=torch.bfloat16,
    )
    for network in networks:
        network.set_multiplier(0.0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = anima(
            noisy_flat.unsqueeze(2), timestep_flat, context=context_flat,
            padding_mask=padding, target_input_ids=None,
        ).squeeze(2).float()
    save_file({
        "base_context": contexts[:contents].to(torch.float16),
        "noisy_inputs": noisy_flat.reshape(contents, len(timesteps), *noisy_flat.shape[1:]).to("cpu", dtype=torch.float16),
        "base_predictions": base.reshape(contents, len(timesteps), *base.shape[1:]).to("cpu", dtype=torch.float16),
        "timesteps": torch.tensor(timesteps, dtype=torch.float32),
    }, output / "base.safetensors")
    shard_rows = int(cache_cfg.get("shard_mixtures", 16))
    completed = 0
    started = time.perf_counter()
    for shard_index, offset in enumerate(range(0, len(specs), shard_rows)):
        shard_path = output / f"effects-{shard_index:05d}.safetensors"
        part = specs[offset : offset + shard_rows]
        if shard_path.exists():
            completed += len(part)
            continue
        effect_rows = []
        for spec in part:
            for slot, network in enumerate(networks):
                if slot < len(spec.components):
                    info = network.load_weights(str(weight_paths[spec.components[slot]]))
                    if info.missing_keys or info.unexpected_keys:
                        raise RuntimeError(f"LoRA teacher key mismatch: {info}")
                    network.set_multiplier(float(spec.weights[slot]))
                else:
                    network.set_multiplier(0.0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = anima(
                    noisy_flat.unsqueeze(2), timestep_flat, context=context_flat,
                    padding_mask=padding, target_input_ids=None,
                ).squeeze(2).float()
            effect_rows.append(
                (prediction - base).reshape(
                    contents, len(timesteps), *base.shape[1:]
                ).to("cpu", dtype=torch.bfloat16)
            )
        save_file({
            "effects": torch.stack(effect_rows),
            "mixture_indices": torch.tensor([spec.index for spec in part], dtype=torch.int64),
        }, shard_path)
        completed += len(part)
        print(
            f"LoRA functional cache {completed}/{len(specs)} mixtures "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    summary = {
        "mixtures": len(specs),
        "individual": len(plans),
        "pairs": sum(spec.kind == "pair" for spec in specs),
        "triples": sum(spec.kind == "triple" for spec in specs),
        "contents": contents,
        "timesteps": timesteps,
        "complete_mixtures": completed,
        "actual_merged_lora_forward": True,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    del anima, networks, base, noisy_flat, context_flat
    gc.collect()
    torch.cuda.empty_cache()
    return {**summary, "reused": False}


class FunctionalLoRATeacherBank:
    def __init__(self, root: Path):
        self.root = root
        self.base = load_file(root / "base.safetensors", device="cpu")
        self.mixtures = read_records(root / "mixtures.parquet")
        parts = [load_file(path, device="cpu") for path in sorted(root.glob("effects-*.safetensors"))]
        effects = torch.cat([part["effects"] for part in parts])
        indices = torch.cat([part["mixture_indices"] for part in parts]).tolist()
        order = torch.tensor(sorted(range(len(indices)), key=indices.__getitem__), dtype=torch.long)
        self.effects = effects.index_select(0, order)
        self.by_kind = {
            kind: [int(row["index"]) for row in self.mixtures if str(row["kind"]) == kind]
            for kind in ("single", "pair", "triple")
        }


def _functional_objective(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student = student.float()
    teacher = teacher.detach().float()
    dims = tuple(range(1, student.ndim))
    teacher_rms = teacher.square().mean(dim=dims).sqrt().clamp_min(1e-4)
    student_rms = student.square().mean(dim=dims).sqrt()
    scale = teacher_rms.reshape(-1, *([1] * (student.ndim - 1)))
    raw_huber = F.smooth_l1_loss(student / scale, teacher / scale, beta=0.10)
    raw_cosine = F.cosine_similarity(student.flatten(1), teacher.flatten(1), dim=1).mean()
    magnitude = F.smooth_l1_loss(
        (student_rms / teacher_rms).log().clamp(-4, 4),
        torch.zeros_like(student_rms),
        beta=0.10,
    )
    _, centered_teacher = decompose_teacher_effects(teacher)
    _, centered_student = decompose_teacher_effects(student)
    centered_scale = centered_teacher.square().mean().sqrt().clamp_min(1e-4)
    centered_huber = F.smooth_l1_loss(
        centered_student / centered_scale,
        centered_teacher / centered_scale,
        beta=0.10,
    )
    centered_cosine = F.cosine_similarity(
        centered_student.flatten(1), centered_teacher.flatten(1), dim=1
    ).mean()
    total = (
        float(weights.get("raw_huber", 0.50)) * raw_huber
        + float(weights.get("raw_direction", 0.25)) * (1 - raw_cosine)
        + float(weights.get("magnitude", 0.10)) * magnitude
        + float(weights.get("centered_huber", 0.50)) * centered_huber
        + float(weights.get("centered_direction", 0.75)) * (1 - centered_cosine)
    )
    return total, {
        "loss": total.detach(),
        "raw_huber": raw_huber.detach(),
        "raw_cosine": raw_cosine.detach(),
        "magnitude_loss": magnitude.detach(),
        "student_to_teacher_rms": (student_rms / teacher_rms).mean().detach(),
        "centered_huber": centered_huber.detach(),
        "centered_cosine": centered_cosine.detach(),
        "common_output_ratio": (
            student.mean(dim=0).square().mean().sqrt()
            / student.square().mean().sqrt().clamp_min(1e-8)
        ).detach(),
    }


def _teacher_decomposed_functional_objective(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match a weak common scaffold and a strong artist-centered residual.

    This is intentionally not a raw-residual objective.  A trainable shared
    path can satisfy raw regression by emitting nearly the same effect for
    every reference.  Decomposing both sides makes that shortcut observable,
    while the all-wrong functional InfoNCE term requires each student residual
    to identify its own LoRA teacher among the other artists in the batch.
    """

    student = student.float()
    teacher = teacher.detach().float()
    if student.shape != teacher.shape or student.shape[0] < 2:
        raise ValueError("Decomposed functional supervision needs matching batch rows")

    student_common, student_centered = decompose_teacher_effects(student)
    teacher_common, teacher_centered = decompose_teacher_effects(teacher)
    reduce_dims = tuple(range(1, student.ndim))
    row_shape = (-1,) + (1,) * (student.ndim - 1)

    teacher_centered_rms = (
        teacher_centered.square().mean(dim=reduce_dims).sqrt().clamp_min(1e-4)
    )
    # sqrt has an infinite derivative at exactly zero.  A collapsed model is
    # precisely the case this objective must recover from, so keep that
    # backward path finite instead of relying on a post-sqrt clamp.
    student_centered_rms = (
        student_centered.square().mean(dim=reduce_dims) + 1e-12
    ).sqrt()
    centered_scale = teacher_centered_rms.reshape(row_shape)
    centered_huber = F.smooth_l1_loss(
        student_centered / centered_scale,
        teacher_centered / centered_scale,
        beta=0.10,
    )
    centered_cosine = F.cosine_similarity(
        student_centered.flatten(1), teacher_centered.flatten(1), dim=1
    ).mean()
    centered_magnitude = F.smooth_l1_loss(
        (student_centered_rms / teacher_centered_rms).clamp_min(1e-4).log().clamp(-4, 4),
        torch.zeros_like(student_centered_rms),
        beta=0.10,
    )

    temperature = float(weights.get("functional_infonce_temperature", 0.10))
    if temperature <= 0:
        raise ValueError("functional_infonce_temperature must be positive")
    student_unit = F.normalize(student_centered.flatten(1), dim=1)
    teacher_unit = F.normalize(teacher_centered.flatten(1), dim=1)
    logits = student_unit @ teacher_unit.t() / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    infonce = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    positive = logits.diagonal() * temperature
    wrong = logits.masked_fill(
        torch.eye(student.shape[0], device=student.device, dtype=torch.bool),
        torch.finfo(logits.dtype).min,
    ).max(dim=1).values * temperature

    teacher_common_rms = teacher_common.square().mean().sqrt().clamp_min(1e-4)
    student_common_rms = student_common.square().mean().sqrt()
    common_huber = F.smooth_l1_loss(
        student_common / teacher_common_rms,
        teacher_common / teacher_common_rms,
        beta=0.10,
    )
    common_cosine = F.cosine_similarity(
        student_common.flatten(), teacher_common.flatten(), dim=0
    )
    student_total_rms = student.square().mean().sqrt().clamp_min(1e-8)
    teacher_total_rms = teacher.square().mean().sqrt().clamp_min(1e-8)
    student_common_ratio = student_common_rms / student_total_rms
    teacher_common_ratio = teacher_common_rms / teacher_total_rms
    common_margin = float(weights.get("common_ratio_margin", 0.03))
    common_excess = F.relu(
        student_common_ratio - teacher_common_ratio - common_margin
    )
    common_excess_loss = common_excess.square()

    total = (
        float(weights.get("centered_huber", 1.0)) * centered_huber
        + float(weights.get("centered_direction", 1.0)) * (1 - centered_cosine)
        + float(weights.get("centered_magnitude", 0.25)) * centered_magnitude
        + float(weights.get("functional_infonce", 0.25)) * infonce
        + float(weights.get("common_huber", 0.10)) * common_huber
        + float(weights.get("common_direction", 0.05)) * (1 - common_cosine)
        + float(weights.get("common_ratio_excess", 1.0)) * common_excess_loss
    )
    return total, {
        "loss": total.detach(),
        "centered_huber": centered_huber.detach(),
        "centered_cosine": centered_cosine.detach(),
        "centered_magnitude_loss": centered_magnitude.detach(),
        "centered_student_to_teacher_rms": (
            student_centered_rms / teacher_centered_rms
        ).mean().detach(),
        "functional_infonce_loss": infonce.detach(),
        "functional_infonce_accuracy": (
            logits.argmax(dim=1) == labels
        ).float().mean().detach(),
        "functional_infonce_positive_cosine": positive.mean().detach(),
        "functional_infonce_hardest_wrong_cosine": wrong.mean().detach(),
        "functional_infonce_cosine_gap": (positive - wrong).mean().detach(),
        "common_huber": common_huber.detach(),
        "common_cosine": common_cosine.detach(),
        "common_output_ratio": student_common_ratio.detach(),
        "teacher_common_output_ratio": teacher_common_ratio.detach(),
        "common_output_excess": common_excess.detach(),
        "common_output_excess_loss": common_excess_loss.detach(),
        "student_to_teacher_rms": (student_total_rms / teacher_total_rms).detach(),
    }


def _pack_mixture_references(
    loader: CachedTeacherReferenceLoader,
    rows: list[dict[str, Any]],
    *,
    references_per_component: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_styles = [style for row in rows for style in row["style_ids"]]
    loaded = loader.load_styles(
        flat_styles,
        references_per_style=references_per_component,
        seed=seed,
    )
    tokens = loaded["tokens"]
    batch = len(rows)
    components = len(rows[0]["style_ids"])
    tokens = tokens.reshape(
        batch, components * references_per_component, *tokens.shape[2:]
    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)
    mask = torch.ones(tokens.shape[:2], device=device, dtype=torch.bool)
    reference_weights = torch.empty(tokens.shape[:2], device=device, dtype=torch.float32)
    for row_index, row in enumerate(rows):
        weights = torch.tensor(row["weights"], device=device, dtype=torch.float32)
        reference_weights[row_index] = weights.repeat_interleave(
            references_per_component
        ) / references_per_component
    return tokens, mask, reference_weights


def _lora_teacher_step(
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    bank: FunctionalLoRATeacherBank,
    loader: CachedTeacherReferenceLoader,
    *,
    kind: str,
    update: int,
    step: int,
    device: str,
    training: dict[str, Any],
) -> dict[str, torch.Tensor]:
    candidates = bank.by_kind[kind]
    batch_rows = int(training.get("batch_rows", 4))
    rng = random.Random(int(training.get("seed", 20260823)) + update * 1_000_003)
    indices = rng.sample(candidates, batch_rows)
    rows = [bank.mixtures[index] for index in indices]
    references, mask, reference_weights = _pack_mixture_references(
        loader,
        rows,
        references_per_component=int(training.get("references_per_component", 1)),
        seed=int(training.get("seed", 20260823)) ^ (update * 7919),
        device=device,
    )
    content_index = update % int(bank.base["noisy_inputs"].shape[0])
    timestep_index = (update // int(bank.base["noisy_inputs"].shape[0])) % int(
        bank.base["noisy_inputs"].shape[1]
    )
    noisy = bank.base["noisy_inputs"][content_index, timestep_index].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    base = bank.base["base_predictions"][content_index, timestep_index].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    context = bank.base["base_context"][content_index : content_index + 1].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    timestep = bank.base["timesteps"][timestep_index].to(
        device=device, dtype=torch.bfloat16
    )
    teacher = bank.effects[indices, content_index, timestep_index].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    adapter.set_bootstrap_phase("combined")
    student = _controlled_teacher_forward(
        anima, reader, adapter, references, mask,
        noisy, base, context, timestep, device,
        reference_weights=reference_weights,
    )
    objective = str(training.get("functional_objective", "legacy_raw"))
    objective_fn = (
        _teacher_decomposed_functional_objective
        if objective == "teacher_decomposed"
        else _functional_objective
    )
    if objective not in {"legacy_raw", "teacher_decomposed"}:
        raise ValueError(f"Unsupported functional objective: {objective}")
    loss, metrics = objective_fn(
        student, teacher, dict(training.get("loss_weights", {}))
    )
    loss.backward()
    metrics.update({
        "reference_count": mask.sum(dim=1).float().mean().detach(),
        "timestep": timestep.float().detach(),
    })
    return metrics


def train_lora_functional_distillation(
    config: dict[str, Any],
    destination: Path,
    *,
    steps_override: int | None = None,
    config_key: str = "lora_functional_distillation",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config[config_key])
    training = dict(cfg["training"])
    if steps_override is not None:
        training["steps"] = int(steps_override)
    steps = int(training.get("steps", 8000))
    device = str(training.get("device", "cuda"))
    seed = int(cfg.get("seed", 20260823))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=True)
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    adapter = _build_style_adapter(detail_cfg).to(device)
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("LoRA distillation requires the separated Common/Artist adapter")
    attach_same_q_style_adapter(anima, adapter)
    initial = torch.load(destination / str(cfg["initial_checkpoint"]), map_location="cpu", weights_only=False)
    reader.load_state_dict(initial["reader"], strict=True)
    adapter.load_state_dict(initial["adapter"], strict=True)
    adapter.restore_timestep_strength_state()
    adapter.set_bootstrap_phase("combined")
    freeze_common = bool(training.get("freeze_common", False))
    for parameter in adapter.common_parameters():
        parameter.requires_grad_(not freeze_common)

    bank = FunctionalLoRATeacherBank(destination / str(cfg["teacher_cache"]["output_directory"]))
    lora_root = destination / str(cfg["lora_directory"])
    lora_plans = _load_lora_plan(lora_root)
    style_ids = [plan.style_id for plan in lora_plans]
    max_refs = int(training.get("references_per_component", 1))
    human_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["human_reference_cache"]),
        split="train", style_ids=style_ids, batch_size=int(training.get("batch_rows", 4)),
        references=max_refs, seed=seed ^ 0x48554D41,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        ram_resident_tokens=bool(training.get("ram_resident_tokens", True)),
        strict_style_ids=True,
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        destination / str(cfg["synthetic_reference_cache"]),
        split="train", style_ids=style_ids, batch_size=int(training.get("batch_rows", 4)),
        references=max_refs, seed=seed ^ 0x53594E54,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        ram_resident_tokens=bool(training.get("ram_resident_tokens", True)),
        strict_style_ids=True,
    )
    native_bank = NativeCenteredTeacherBank.load(
        config, destination, config_key=str(cfg["native_teacher"]["bank_config_key"])
    )
    contexts = NativeArtistContextCache(
        destination / str(cfg["native_teacher"]["context_cache"]),
        capacity=int(cfg["native_teacher"].get("context_lru_shards", 8)),
    )
    native_loader = CachedTeacherReferenceLoader(
        [destination / str(value) for value in cfg["native_teacher"]["reference_caches"]],
        split="train", style_ids=list(native_bank.summary["train_style_ids"]),
        batch_size=int(training.get("native_batch_rows", 8)),
        references=int(training.get("native_references", 4)),
        seed=seed ^ 0x4E415449,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        strict_style_ids=False,
    )

    groups = [
        {"params": list(reader.parameters()), "lr": float(training.get("reader_learning_rate", 2e-5)), "name": "reader"},
        {"params": adapter.shared_parameters(), "lr": float(training.get("shared_learning_rate", 1e-4)), "name": "shared_kv"},
        {"params": adapter.delta_parameters(), "lr": float(training.get("delta_learning_rate", 2e-4)), "name": "block_delta"},
        {"params": adapter.mixing_parameters(), "lr": float(training.get("mix_learning_rate", 4e-5)), "name": "base_mix", "weight_decay": 0.0},
    ]
    if not freeze_common:
        groups.insert(1, {
            "params": adapter.common_parameters(),
            "lr": float(training.get("common_learning_rate", 3e-5)),
            "name": "common",
        })
    if adapter.null_parameters():
        groups.append({"params": adapter.null_parameters(), "lr": float(training.get("null_learning_rate", 2e-4)), "name": "artist_null", "weight_decay": 0.0})
    base_lrs = {str(group["name"]): float(group["lr"]) for group in groups}
    optimizer = torch.optim.AdamW(
        groups, betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=float(training.get("adam_eps", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)), fused=True,
    )
    output = destination / str(cfg["output_directory"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    state_path = output / "training_state.pt"
    start_step = 0
    if bool(training.get("resume", True)) and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        reader.load_state_dict(state["reader"], strict=True)
        adapter.load_state_dict(state["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])

    wandb_run = None
    wandb_cfg = dict(training.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "anima-style-adapter")),
            name=str(wandb_cfg.get("name", "detail-style-lora-functional-distill-v1")),
            id=str(wandb_cfg.get("id", "detail-style-lora-functional-distill-v1")),
            resume="allow" if start_step else "never",
            config={config_key: cfg},
        )
    fixed = load_dual_query_external_sample(config, destination)
    single_only = int(training.get("single_only_steps", 500))
    artist_intro = int(training.get("artist_intro_steps", 0))
    curriculum = str(training.get("curriculum", "legacy"))
    warmup = int(training.get("warmup_steps", 100))
    decay_start = int(training.get("lr_decay_start_step", 6000))
    min_lr = float(training.get("minimum_lr_ratio", 0.1))
    log_every = int(training.get("log_every", 10))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    sample_every = int(training.get("sample_every", 1000))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    running: dict[str, list[float]] = defaultdict(list)
    updates = defaultdict(int)
    native_common_cache: dict[tuple[int, int], torch.Tensor] = {}
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, steps + 1):
            if step <= warmup:
                lr_scale = step / max(1, warmup)
            elif step <= decay_start:
                lr_scale = 1.0
            else:
                progress = (step - decay_start) / max(1, steps - decay_start)
                lr_scale = min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = base_lrs[str(group["name"])] * lr_scale
            optimizer.zero_grad(set_to_none=True)
            category = (
                teacher_category_v2(
                    step,
                    single_only_steps=single_only,
                    artist_intro_steps=artist_intro,
                )
                if curriculum == "artist_centered_v2"
                else teacher_category(step, single_only_steps=single_only)
            )
            updates[category] += 1
            if category == "artist_tag":
                native_training = {
                    **training,
                    "teacher_batch_rows": int(training.get("native_batch_rows", 8)),
                    "teacher_microbatch_rows": int(training.get("native_microbatch_rows", 4)),
                    "teacher_global_weight": float(training.get("native_global_weight", 0.25)),
                    "teacher_infonce_weight": float(training.get("native_infonce_weight", 0.10)),
                    "teacher_infonce_temperature": 0.10,
                    "separated_teacher_gradients": not freeze_common,
                    "train_common_in_combined": not freeze_common,
                    "teacher_objective": dict(training.get("native_loss", {})),
                    "separated_component_bootstrap": {
                        "enabled": False,
                        "artist_mean_weight": (
                            0.0 if freeze_common else float(
                                training.get("native_artist_mean_weight", 0.25)
                            )
                        ),
                    },
                    "post_gate_teacher_distillation": {"enabled": False},
                }
                batch = native_loader.load_step(updates[category])
                metrics = _teacher_step(
                    anima, reader, adapter, native_bank, contexts, batch, device,
                    native_training, None, native_common_cache,
                    step=step, probe_index=updates[category],
                )
            else:
                domain_loader = (
                    human_loader if updates[category] % 2 else synthetic_loader
                )
                kind = "single"
                if category == "lora_mixture":
                    kind = "triple" if updates[category] % 3 == 0 else "pair"
                metrics = _lora_teacher_step(
                    anima, reader, adapter, bank, domain_loader,
                    kind=kind, update=updates[category], step=step,
                    device=device, training={**training, "seed": seed},
                )
                metrics["domain_synthetic"] = torch.tensor(
                    float(domain_loader is synthetic_loader), device=device
                )
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, foreach=True)
            optimizer.step()
            for key, value in metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    running[f"{category}/{key}"].append(float(value.detach()))
            running["optimizer/grad_norm"].append(float(grad_norm))
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items() if values}
                row["optimizer/learning_rate"] = float(optimizer.param_groups[0]["lr"])
                row["progress/category"] = {"artist_tag": 0, "lora_single": 1, "lora_mixture": 2}[category]
                row["progress/common_frozen"] = float(freeze_common)
                print(f"LoRA distill step={step}/{steps} category={category} {row}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in row.items()}, step=step)
                running.clear()
            if step % checkpoint_every == 0 or step == steps:
                _save_state(
                    state_path, step=step, reader=reader, adapter=adapter,
                    optimizer=optimizer, cfg=cfg,
                )
                _save_state(
                    checkpoints / f"step-{step:07d}.pt", step=step,
                    reader=reader, adapter=adapter, optimizer=optimizer, cfg=cfg,
                )
            if sample_every > 0 and step % sample_every == 0:
                sample = _generate_fixed_reference_sample(
                    fixed, config, destination, anima, reader, adapter,
                    output, device, step, strengths_override=[1.0],
                    sample_group="fixed_reference_samples",
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "val/functional/fixed_reference": wandb.Image(
                            sample["sheet"], caption=f"step {step}"
                        )
                    }, step=step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        adapter.clear_style_tokens()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "updates": dict(updates),
        "elapsed_s": time.perf_counter() - started,
        "teacher_ratio_after_bootstrap": "1:1:1",
        "lora_reference_domains": "human:synthetic=1:1",
        "curriculum": curriculum,
        "common_frozen": freeze_common,
    }
    write_json(output / "summary.json", summary)
    return summary


def smoke_test_lora_functional_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_functional_distillation"]
    cfg["output_directory"] = str(Path(cfg["output_directory"]).with_name("lora_functional_distillation_smoke"))
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    cfg["training"]["single_only_steps"] = 0
    return train_lora_functional_distillation(effective, destination, steps_override=3)


def train_lora_functional_distillation_v2(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return train_lora_functional_distillation(
        config,
        destination,
        config_key="lora_functional_distillation_v2",
    )


def smoke_test_lora_functional_distillation_v2(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["lora_functional_distillation_v2"]
    cfg["output_directory"] = str(
        Path(cfg["output_directory"]).with_name(
            "lora_functional_distillation_v2_smoke"
        )
    )
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    cfg["training"]["single_only_steps"] = 0
    cfg["training"]["artist_intro_steps"] = 0
    return train_lora_functional_distillation(
        effective,
        destination,
        steps_override=3,
        config_key="lora_functional_distillation_v2",
    )
