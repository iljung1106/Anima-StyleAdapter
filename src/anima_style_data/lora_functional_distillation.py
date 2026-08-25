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
import pyarrow.parquet as pq
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from safetensors.torch import load_file, save_file

from .artist_lora_teachers import ArtistLoRAPlan
from .artist_lora_teachers import _selected_lora_modules, _serialize_lora_patterns
from .detail_style_cross_attention import (
    DetailPreservingTypedSlotReader,
    SeparatedCommonArtistKVStyleCrossAttention,
)
from .detail_style_teacher_context import NativeArtistContextCache
from .detail_style_training import (
    _audit_student_prompts,
    _build_style_adapter,
    _compose_separate_text_style_guidance,
    _controlled_teacher_forward,
    _decode_latents,
    _flow_step,
    _generate_fixed_reference_sample,
    _loader_config,
    _save_state,
    _teacher_step,
    _training_loader,
)
from .dual_query_external_samples import load_dual_query_external_sample
from .dual_query_style_tokenizer import CachedTeacherReferenceLoader
from .global_query_style_tokenizer import MultiPromptDualQueryCachedStyleLoader
from .io import read_records, write_json, write_records
from .native_centered_teacher import NativeCenteredTeacherBank
from .query_style_tokenizer import (
    _sample_query_style_tokenizer,
    _select_sample_episodes,
)
from .same_q_style_adapter import attach_same_q_style_adapter
from .style_calibration import _encode_prompts
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


def _fewshot_prompt_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    cases = tuple(dict(value) for value in cfg.get("prompt_cases", ()))
    if not cases:
        raise ValueError("fewshot validation needs prompt_cases")
    names = [str(value["name"]) for value in cases]
    if len(set(names)) != len(names):
        raise ValueError("fewshot prompt case names must be unique")
    return {
        "version": 1,
        "prompt_cases": [
            {
                "name": str(value["name"]),
                "prompt": str(value["prompt"]),
                "seed": int(value["seed"]),
            }
            for value in cases
        ],
        "negative_prompt": str(cfg["negative_prompt"]),
    }


def _load_or_create_fewshot_prompt_cache(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Encode a small, immutable prompt/seed suite before loading frozen Anima."""

    signature = _fewshot_prompt_signature(cfg)
    output = destination / str(cfg["cache_directory"])
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "prompt_conditions.pt"
    metadata_path = output / "prompt_conditions.json"
    if cache_path.exists() and metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if recorded == signature and tuple(cached["positive"].shape[:1]) == (
            len(signature["prompt_cases"]),
        ):
            return {**signature, **cached}

    prompts = [value["prompt"] for value in signature["prompt_cases"]]
    encoded = _encode_prompts(
        config,
        destination,
        prompts + [signature["negative_prompt"]],
        device,
        batch_size=len(prompts) + 1,
    ).to("cpu", dtype=torch.float16)
    payload = {
        "positive": encoded[:-1].contiguous(),
        "negative": encoded[-1:].contiguous(),
    }
    torch.save(payload, cache_path)
    metadata_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**signature, **payload}


def _select_fewshot_validation_styles(
    manifest_rows: list[dict[str, Any]],
    *,
    split: str,
    artists: int,
    references: int,
    seed: int,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        if str(row.get("split", "train")) == str(split):
            counts[str(row["style_id"])] += 1
    eligible = sorted(
        style_id for style_id, count in counts.items() if count >= int(references)
    )
    if len(eligible) < int(artists):
        raise RuntimeError(
            f"fewshot validation needs {artists} artists with {references} images; "
            f"only {len(eligible)} are eligible"
        )
    random.Random(int(seed)).shuffle(eligible)
    return eligible[: int(artists)]


def _resolve_reference_paths(
    destination: Path,
    image_ids: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Path, ...], ...]:
    by_id = {
        int(row["id"]): Path(str(row["local_path"]))
        for row in read_records(destination / "final_manifest.parquet")
    }
    resolved: list[tuple[Path, ...]] = []
    for values in image_ids:
        paths = []
        for image_id in values:
            path = by_id[int(image_id)]
            if not path.is_absolute():
                path = destination / path
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
        resolved.append(tuple(paths))
    return tuple(resolved)


def _prepare_fewshot_validation(
    destination: Path,
    token_root: Path,
    cfg: dict[str, Any],
    prompt_cache: dict[str, Any],
) -> dict[str, Any]:
    artist_count = int(cfg.get("artist_count", 8))
    max_references = int(cfg.get("max_references", 8))
    manifest_rows = read_records(token_root / "manifest.parquet")
    style_ids = _select_fewshot_validation_styles(
        manifest_rows,
        split=str(cfg.get("split", "validation")),
        artists=artist_count,
        references=max_references,
        seed=int(cfg.get("selection_seed", 20260824)),
    )
    loader = CachedTeacherReferenceLoader(
        token_root,
        split=str(cfg.get("split", "validation")),
        style_ids=style_ids,
        batch_size=artist_count,
        references=max_references,
        seed=int(cfg.get("selection_seed", 20260824)),
        token_lru_shards=int(cfg.get("token_lru_shards", 8)),
        ram_resident_tokens=bool(cfg.get("ram_resident_tokens", True)),
        strict_style_ids=True,
    )
    selected = loader.load_styles(
        style_ids,
        references_per_style=max_references,
        seed=int(cfg.get("reference_seed", 2026082401)),
    )
    return {
        "style_ids": tuple(style_ids),
        "tokens": selected["tokens"],
        "paths": _resolve_reference_paths(destination, selected["ids"]),
        "prompt_cache": prompt_cache,
        "cfg": cfg,
    }


def _fewshot_reference_collage(
    paths: tuple[Path, ...], size: tuple[int, int]
) -> Image.Image:
    width, height = size
    columns = 4
    rows = 2
    tile_size = (width // columns, height // rows)
    canvas = Image.new("RGB", size, "white")
    for index, path in enumerate(paths[: columns * rows]):
        with Image.open(path) as image:
            tile = ImageOps.pad(
                image.convert("RGB"), tile_size, method=Image.Resampling.LANCZOS,
                color="white",
            )
        canvas.paste(tile, ((index % columns) * tile_size[0], (index // columns) * tile_size[1]))
    return canvas


def _fewshot_label_cell(
    size: tuple[int, int], lines: list[str]
) -> Image.Image:
    cell = Image.new("RGB", size, (24, 24, 28))
    draw = ImageDraw.Draw(cell)
    y = 24
    for line in lines:
        draw.text((24, y), str(line), fill="white")
        y += 24
    return cell


def _annotate_fewshot_cell(image: Image.Image, lines: list[str]) -> Image.Image:
    value = image.convert("RGB").copy()
    draw = ImageDraw.Draw(value, "RGBA")
    height = 22 * len(lines) + 18
    draw.rectangle((0, 0, value.width, height), fill=(0, 0, 0, 176))
    y = 8
    for line in lines:
        draw.text((10, y), str(line), fill=(255, 255, 255, 255))
        y += 22
    return value


@torch.no_grad()
def _fewshot_denoise(
    anima: torch.nn.Module,
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    initial_noise: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    style: torch.Tensor | None,
    cfg: dict[str, Any],
    device: str,
) -> torch.Tensor:
    x = initial_noise.clone()
    batch = int(x.shape[0])
    negative_batch = negative.expand(batch, -1, -1)
    sigmas = torch.linspace(
        1.0,
        0.0,
        int(cfg["steps"]) + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    padding_mask = torch.zeros(
        batch,
        1,
        x.shape[-2],
        x.shape[-1],
        device=device,
        dtype=torch.bfloat16,
    )
    text_cfg = float(cfg.get("cfg", 4.0))
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        for index in range(len(sigmas) - 1):
            timestep = sigmas[index].expand(batch)
            adapter.clear_style_tokens()
            negative_null = anima(
                x,
                timestep,
                context=negative_batch,
                padding_mask=padding_mask,
                target_input_ids=None,
            ).float()
            positive_null = anima(
                x,
                timestep,
                context=positive,
                padding_mask=padding_mask,
                target_input_ids=None,
            ).float()
            if style is None:
                velocity = negative_null + text_cfg * (
                    positive_null - negative_null
                )
            else:
                adapter.set_style_context(style, strength=1.0)
                adapter.set_timesteps(timestep)
                positive_style = anima(
                    x,
                    timestep,
                    context=positive,
                    padding_mask=padding_mask,
                    target_input_ids=None,
                ).float()
                velocity = _compose_separate_text_style_guidance(
                    negative_null,
                    positive_null,
                    positive_style,
                    text_cfg=text_cfg,
                    style_strength=1.0,
                )
            x = (
                x.float()
                + velocity * (sigmas[index + 1] - sigmas[index]).float()
            ).to(torch.bfloat16)
    adapter.clear_style_tokens()
    return x.to("cpu")


def _fewshot_effect_metrics(
    baseline: list[Image.Image], styled: list[Image.Image]
) -> dict[str, float]:
    base = np.stack(
        [np.asarray(image, dtype=np.float32) / 255.0 for image in baseline]
    )
    current = np.stack(
        [np.asarray(image, dtype=np.float32) / 255.0 for image in styled]
    )
    effects = current - base
    effect_rms = np.sqrt(np.mean(np.square(effects), axis=(1, 2, 3)))
    total_rms = float(np.sqrt(np.mean(np.square(effects))))
    common_rms = float(np.sqrt(np.mean(np.square(effects.mean(axis=0)))))
    pairwise = []
    for left in range(len(effects)):
        for right in range(left + 1, len(effects)):
            pairwise.append(
                float(np.sqrt(np.mean(np.square(effects[left] - effects[right]))))
            )
    return {
        "mean_effect_pixel_rms": float(effect_rms.mean()),
        "minimum_effect_pixel_rms": float(effect_rms.min()),
        "maximum_effect_pixel_rms": float(effect_rms.max()),
        "common_effect_ratio": common_rms / max(total_rms, 1e-8),
        "mean_pairwise_centered_effect_rms": float(np.mean(pairwise)),
    }


@torch.no_grad()
def _generate_fewshot_reference_sweep(
    prepared: dict[str, Any],
    config: dict[str, Any],
    destination: Path,
    anima: torch.nn.Module,
    reader: DetailPreservingTypedSlotReader,
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    output: Path,
    device: str,
    step: int,
    *,
    mode: str,
    reference_counts: tuple[int, ...],
    artist_limit: int | None = None,
) -> dict[str, Any]:
    """Render nested 1/2/4/8-reference validation with one shared baseline pass."""

    if mode not in {"controlled", "diverse"}:
        raise ValueError("fewshot validation mode must be controlled or diverse")
    cfg = dict(prepared["cfg"])
    generation = dict(cfg["generation"])
    tokens = prepared["tokens"]
    style_ids = prepared["style_ids"]
    paths = prepared["paths"]
    artist_count = len(style_ids) if artist_limit is None else min(
        len(style_ids), int(artist_limit)
    )
    tokens = tokens[:artist_count]
    style_ids = style_ids[:artist_count]
    paths = paths[:artist_count]
    maximum = int(tokens.shape[1])
    counts = tuple(int(value) for value in reference_counts)
    if not counts or any(value <= 0 or value > maximum for value in counts):
        raise ValueError(f"reference counts must be within 1..{maximum}")

    prompt_cache = prepared["prompt_cache"]
    cases = prompt_cache["prompt_cases"]
    positives = prompt_cache["positive"]
    negative = prompt_cache["negative"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )
    if mode == "controlled":
        case_indices = [0] * artist_count
    else:
        case_indices = [index % len(cases) for index in range(artist_count)]
    positive = torch.stack(
        [positives[index] for index in case_indices]
    ).to(device, dtype=torch.bfloat16, non_blocking=True)
    seeds = [int(cases[index]["seed"]) for index in case_indices]
    width = int(generation["width"])
    height = int(generation["height"])
    noise = torch.cat(
        [
            torch.randn(
                1,
                16,
                1,
                height // 8,
                width // 8,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                dtype=torch.float32,
            )
            for seed in seeds
        ]
    ).to(device, dtype=torch.bfloat16)
    batch_size = max(1, int(generation.get("batch_size", 4)))

    reader_was_training = reader.training
    adapter_was_training = adapter.training
    reader.eval()
    adapter.eval()
    anima.eval()
    latent_groups: dict[str, torch.Tensor] = {}
    try:
        base_parts = []
        for offset in range(0, artist_count, batch_size):
            end = min(artist_count, offset + batch_size)
            base_parts.append(
                _fewshot_denoise(
                    anima,
                    adapter,
                    noise[offset:end],
                    positive[offset:end],
                    negative,
                    style=None,
                    cfg=generation,
                    device=device,
                )
            )
        latent_groups["base"] = torch.cat(base_parts)
        for count in counts:
            references = tokens[:, :count].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            mask = torch.ones(
                artist_count, count, device=device, dtype=torch.bool
            )
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ):
                style_tokens = reader(references, mask).tokens
            styled_parts = []
            for offset in range(0, artist_count, batch_size):
                end = min(artist_count, offset + batch_size)
                styled_parts.append(
                    _fewshot_denoise(
                        anima,
                        adapter,
                        noise[offset:end],
                        positive[offset:end],
                        negative,
                        style=style_tokens[offset:end],
                        cfg=generation,
                        device=device,
                    )
                )
            latent_groups[f"r{count}"] = torch.cat(styled_parts)
            del references, mask, style_tokens
    finally:
        adapter.clear_style_tokens()
        if reader_was_training:
            reader.train()
        if adapter_was_training:
            adapter.train()

    decoded = _decode_latents(
        config,
        destination,
        latent_groups,
        device,
        int(generation.get("vae_batch_size", 4)),
    )
    size = (width, height)
    rows = 2 + len(counts)
    sheet = Image.new(
        "RGB", ((artist_count + 1) * width, rows * height), "white"
    )
    sheet.paste(
        _fewshot_label_cell(
            size,
            [
                f"{mode.upper()} FEW-SHOT",
                f"step {step}",
                f"references {list(counts)}",
            ],
        ),
        (0, 0),
    )
    sheet.paste(_fewshot_label_cell(size, ["FROZEN ANIMA", "NO STYLE"]), (0, height))
    for row_index, count in enumerate(counts, start=2):
        sheet.paste(
            _fewshot_label_cell(size, ["STYLE ADAPTER", f"{count} reference(s)"]),
            (0, row_index * height),
        )
    for artist_index, (style_id, artist_paths) in enumerate(
        zip(style_ids, paths, strict=True), start=1
    ):
        case = cases[case_indices[artist_index - 1]]
        reference = _annotate_fewshot_cell(
            _fewshot_reference_collage(artist_paths, size),
            [str(style_id), "8 fixed validation references"],
        )
        sheet.paste(reference, (artist_index * width, 0))
        baseline = _annotate_fewshot_cell(
            decoded["base"][artist_index - 1],
            [str(case["name"]), f"seed {case['seed']}"],
        )
        sheet.paste(baseline, (artist_index * width, height))
        for row_index, count in enumerate(counts, start=2):
            current = _annotate_fewshot_cell(
                decoded[f"r{count}"][artist_index - 1],
                [str(style_id), f"references {count}"],
            )
            sheet.paste(current, (artist_index * width, row_index * height))

    sample_dir = output / "fewshot_validation" / f"step-{step:07d}-{mode}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sample_dir / f"fewshot-{mode}.png"
    sheet.save(sheet_path, compress_level=4)
    metrics = {
        f"r{count}": _fewshot_effect_metrics(
            decoded["base"], decoded[f"r{count}"]
        )
        for count in counts
    }
    summary = {
        "step": int(step),
        "mode": mode,
        "sheet": str(sheet_path),
        "style_ids": list(style_ids),
        "reference_counts": list(counts),
        "prompt_cases": [cases[index] for index in case_indices],
        "metrics": metrics,
    }
    write_json(sample_dir / "summary.json", summary)
    del latent_groups, decoded, noise, positive, negative
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


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


_TEACHER_CATEGORIES = frozenset(("artist_tag", "lora_single", "lora_mixture"))
_REFERENCE_DOMAINS = frozenset(("human", "synthetic"))


def scheduled_teacher_category(
    step: int,
    *,
    single_only_steps: int,
    schedule: tuple[str, ...] | list[str],
    bootstrap_schedule: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Follow explicit bootstrap and post-bootstrap teacher schedules.

    Weighted LoRA mixtures remain offline functional teachers. They are
    delayed until the direct visual mapping has first seen both individual
    LoRA effects and native Anima artist effects. Neither schedule is part of
    inference; the trained adapter only receives visual reference tokens.
    """

    normalized = tuple(str(value) for value in schedule)
    bootstrap = tuple(
        str(value)
        for value in (
            ("lora_single",) if bootstrap_schedule is None else bootstrap_schedule
        )
    )
    if not normalized or not bootstrap:
        raise ValueError("teacher schedules must contain at least one category")
    invalid = (set(normalized) | set(bootstrap)) - _TEACHER_CATEGORIES
    if invalid:
        raise ValueError(f"Unsupported teacher categories: {sorted(invalid)}")
    if step <= int(single_only_steps):
        return bootstrap[(step - 1) % len(bootstrap)]
    return normalized[(step - int(single_only_steps) - 1) % len(normalized)]


def scheduled_reference_domain(
    update: int, schedule: tuple[str, ...] | list[str]
) -> str:
    """Select the visual-reference domain without changing the teacher target.

    Human and LoRA-generated references supervise the same offline LoRA effect,
    but deployment starts from human artwork. An explicit deterministic cycle
    emphasizes that domain without dropping the cleaner synthetic pairing.
    """

    normalized = tuple(str(value) for value in schedule)
    if update <= 0:
        raise ValueError("update must be positive")
    if not normalized:
        raise ValueError("reference domain schedule must not be empty")
    invalid = set(normalized) - _REFERENCE_DOMAINS
    if invalid:
        raise ValueError(f"Unsupported reference domains: {sorted(invalid)}")
    return normalized[(update - 1) % len(normalized)]


_POOLING_READER_PREFIXES = (
    "set_query",
    "set_norm",
    "reference_identity_norm",
    "reference_identity_projection",
    "pool_type_embeddings",
    "pool_type_preference",
    "set_attention",
    "set_ff_norm",
    "set_ff",
    "mixers",
)


def _configure_reader_trainable_scope(
    reader: DetailPreservingTypedSlotReader, scope: str
) -> list[torch.nn.Parameter]:
    """Open reference aggregation/output layers while preserving visual reads."""

    normalized = str(scope).strip().lower()
    if normalized not in {"none", "pooling", "all"}:
        raise ValueError(
            "reader_trainable_scope must be one of: none, pooling, all"
        )
    selected: list[torch.nn.Parameter] = []
    for name, parameter in reader.named_parameters():
        trainable = normalized == "all" or (
            normalized == "pooling"
            and any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _POOLING_READER_PREFIXES
            )
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    reader.train(normalized != "none")
    return selected


def build_mixture_specs(
    artists: int,
    *,
    pair_count: int,
    triple_count: int,
    amplified_count: int = 0,
    signed_count: int = 0,
    amplified_sum_range: tuple[float, float] = (1.05, 1.35),
    signed_beta_range: tuple[float, float] = (0.05, 0.25),
    amplified_triple_probability: float = 0.5,
    signed_triple_probability: float = 0.0,
    signed_l1_maximum: float = 1.5,
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
    amplified_min, amplified_max = map(float, amplified_sum_range)
    if not (1.0 <= amplified_min <= amplified_max):
        raise ValueError("amplified_sum_range must be ordered and at least one")
    if not 0.0 <= amplified_triple_probability <= 1.0:
        raise ValueError("amplified_triple_probability must lie within [0, 1]")
    for _ in range(amplified_count):
        component_count = 3 if rng.random() < amplified_triple_probability else 2
        components = sample_components(component_count)
        raw = [rng.uniform(0.25, 1.0) for _ in range(component_count)]
        target_sum = rng.uniform(amplified_min, amplified_max)
        total = sum(raw)
        specs.append(
            MixtureSpec(
                len(specs),
                "amplified",
                components,
                tuple(target_sum * value / total for value in raw),
            )
        )
    signed_min, signed_max = map(float, signed_beta_range)
    if not (0.0 <= signed_min <= signed_max <= 0.5):
        raise ValueError("signed_beta_range must lie within [0, 0.5]")
    if not 0.0 <= signed_triple_probability <= 1.0:
        raise ValueError("signed_triple_probability must lie within [0, 1]")
    if signed_l1_maximum < 1.0:
        raise ValueError("signed_l1_maximum must be at least one")
    if 1.0 + 2.0 * signed_max > signed_l1_maximum + 1e-7:
        raise ValueError(
            "signed_beta_range exceeds the requested signed L1 maximum"
        )
    for _ in range(signed_count):
        component_count = 3 if rng.random() < signed_triple_probability else 2
        components = list(sample_components(component_count))
        rng.shuffle(components)
        beta = rng.uniform(signed_min, signed_max)
        positive_raw = [rng.uniform(0.25, 1.0) for _ in range(component_count - 1)]
        positive_total = sum(positive_raw)
        positive_weights = [
            (1.0 + beta) * value / positive_total for value in positive_raw
        ]
        specs.append(
            MixtureSpec(
                len(specs), "signed", tuple(components),
                tuple(positive_weights + [-beta]),
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


def _initialize_fresh_adapter_strength(
    adapter: SeparatedCommonArtistKVStyleCrossAttention,
    cfg: dict[str, Any],
    destination: Path,
) -> str:
    """Apply measured Anima block/timestep scale without loading model weights."""

    profile_path = cfg.get("initial_strength_profile")
    if not profile_path:
        return "constructor_default"
    payload = json.loads(
        (destination / str(profile_path)).read_text(encoding="utf-8")
    )
    alpha = torch.tensor(
        payload["teacher_to_centered_raw_ratio_by_timestep_bin"],
        dtype=torch.float32,
        device=adapter.alpha.device,
    )
    alpha.mul_(float(cfg.get("initial_strength_multiplier", 1.0)))
    alpha.clamp_(
        min=float(cfg.get("initial_strength_minimum", 1e-4)),
        max=float(cfg.get("initial_strength_maximum", 1.0)),
    )
    adapter.configure_timestep_strength(
        timestep_bin_edges=payload["timestep_bin_edges"],
        alpha_by_timestep=alpha,
        native_lower_by_timestep=torch.zeros_like(alpha),
        native_upper_by_timestep=torch.full_like(alpha, float("inf")),
    )
    return (
        f"profile:{profile_path}:median={float(alpha.median()):.6g}:"
        f"min={float(alpha.min()):.6g}:max={float(alpha.max()):.6g}"
    )


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
    config: dict[str, Any],
    anima: torch.nn.Module,
    count: int,
    device: str,
    network_selection: dict[str, Any] | None = None,
) -> list[torch.nn.Module]:
    sd_root = Path(str(config["anima_cache"]["sd_scripts_path"])).resolve()
    if str(sd_root) not in sys.path:
        sys.path.insert(0, str(sd_root))
    from networks import lora_anima

    selection = dict(network_selection or {})
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
            include_patterns=_serialize_lora_patterns(
                selection.get("include_patterns")
            ),
            exclude_patterns=_serialize_lora_patterns(
                selection.get("exclude_patterns")
            ),
        )
        _selected_lora_modules(network, selection)
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
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str = "lora_teacher_references",
) -> dict[str, Any]:
    cfg = dict(config[config_key])
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "latents").mkdir(exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    lora_root = destination / str(cfg["lora_directory"])
    plans = _load_lora_plan(lora_root)
    weights = _weight_paths(lora_root, plans)
    artist_start = int(cfg.get("artist_start_index", 0))
    artist_stop = int(cfg.get("artist_stop_index", len(plans)))
    if not (0 <= artist_start < artist_stop <= len(plans)):
        raise ValueError(
            f"Invalid artist range [{artist_start}, {artist_stop}) for {len(plans)} plans"
        )
    selected = list(zip(
        plans[artist_start:artist_stop],
        weights[artist_start:artist_stop],
        strict=True,
    ))
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
    network = _create_lora_networks(
        config, anima, 1, device, dict(cfg.get("network_selection", {}))
    )[0]
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
    for selected_index, (plan, weight_path) in enumerate(selected):
        artist_index = int(plan.index)
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
            f"LoRA references artist={selected_index + 1}/{len(selected)} "
            f"plan_index={artist_index} "
            f"images={len(completed_rows)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    completed_rows.sort(key=lambda row: int(row["id"]))
    write_records(output / "manifest.parquet", completed_rows)
    summary = {
        "artists": len(selected),
        "artist_start_index": artist_start,
        "artist_stop_index": artist_stop,
        "teacher_bank_artists": len(plans),
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


def generate_lora_mixture_references(
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str = "lora_mixture_references_512",
) -> dict[str, Any]:
    """Materialize merged-LoRA styles so the student never sees mixture weights."""

    cfg = dict(config[config_key])
    functional_cfg = dict(config[str(cfg["functional_config_key"])])
    teacher_root = destination / str(
        functional_cfg["teacher_cache"]["output_directory"]
    )
    mixture_rows = [
        row
        for row in read_records(teacher_root / "mixtures.parquet")
        if str(row["kind"]) in set(
            str(value)
            for value in cfg.get(
                "kinds", ["pair", "triple", "amplified", "signed"]
            )
        )
        and bool(row.get("enabled", True))
    ]
    if not mixture_rows:
        raise RuntimeError("No materialized mixture specifications were selected")
    output = destination / str(cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "latents").mkdir(exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    lora_root = destination / str(functional_cfg["lora_directory"])
    plans = _load_lora_plan(lora_root)
    weight_paths = _weight_paths(lora_root, plans)
    context_cache = cfg.get("content_context_cache")
    if context_cache:
        context_root = destination / str(context_cache)
        content_rows = read_records(context_root / "content_manifest.parquet")
        conditions = load_file(
            context_root / "base.safetensors", device="cpu"
        )["base_context"]
        if len(content_rows) != len(conditions):
            raise RuntimeError(
                "Mixture caption manifest and cached text contexts disagree"
            )
        negative_path = destination / str(cfg["negative_conditioning_file"])
        negative = load_file(negative_path, device="cpu")["conditioning"]
    else:
        source = destination / str(cfg["content_source_directory"])
        content_rows, conditions, negative = _load_content_conditions(source)
    images_per_mixture = int(cfg.get("images_per_mixture", 4))
    if len(content_rows) < images_per_mixture:
        raise RuntimeError("The source text bank has too few mixture prompts")
    random_content = bool(cfg.get("random_content_per_mixture", False))
    if not random_content:
        content_rows = content_rows[:images_per_mixture]
        conditions = conditions[:images_per_mixture]
    width, height = int(cfg.get("width", 512)), int(cfg.get("height", 512))
    device = str(cfg.get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    network_count = max(len(row["components"]) for row in mixture_rows)
    networks = _create_lora_networks(
        config,
        anima,
        network_count,
        device,
        dict(functional_cfg.get("network_selection", {})),
    )
    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    vae.requires_grad_(False).eval()
    steps = int(cfg.get("steps", 20))
    shift = float(cfg.get("flow_shift", 3.0))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)
    text_cfg = float(cfg.get("text_cfg", 4.0))
    negative = negative.to(device=device, dtype=torch.bfloat16).expand(
        images_per_mixture, -1, -1
    )
    completed_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, row in enumerate(mixture_rows):
        mixture_index = int(row["index"])
        part_path = output / "manifests" / f"part-{mixture_index:05d}.parquet"
        if part_path.exists():
            completed_rows.extend(read_records(part_path))
            continue
        for slot, network in enumerate(networks):
            if slot < len(row["components"]):
                component = int(row["components"][slot])
                info = network.load_weights(str(weight_paths[component]))
                if info.missing_keys or info.unexpected_keys:
                    raise RuntimeError(f"LoRA mixture key mismatch: {info}")
                network.set_multiplier(float(row["weights"][slot]))
            else:
                network.set_multiplier(0.0)
        if random_content:
            content_rng = random.Random(
                int(cfg.get("content_seed", cfg.get("seed", 20260824)))
                + mixture_index * 1_000_003
            )
            selected_content_indices = content_rng.sample(
                range(len(content_rows)), images_per_mixture
            )
            selected_content_rows = [
                content_rows[index] for index in selected_content_indices
            ]
            positive = conditions[selected_content_indices].to(
                device=device, dtype=torch.bfloat16
            )
        else:
            selected_content_indices = list(range(images_per_mixture))
            selected_content_rows = content_rows
            positive = conditions.to(device=device, dtype=torch.bfloat16)
        seeds = [
            int(cfg.get("seed", 20260824))
            + mixture_index * 100_003
            + index * 1009
            for index in range(images_per_mixture)
        ]
        noise = torch.stack([
            torch.randn(
                16,
                1,
                height // 8,
                width // 8,
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
                dtype=torch.bfloat16,
            )
            for seed in seeds
        ])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latents = _sample_anima_batch(
                anima,
                noise,
                positive,
                negative,
                sigmas,
                text_cfg=text_cfg,
                speed=None,
                generation_seeds=seeds,
            )
            decoded = vae.decode_to_pixels(latents)
        if not torch.isfinite(latents).all():
            raise RuntimeError(f"Non-finite mixture latent at index {mixture_index}")
        images = _preview_pixels(decoded)
        latent_values = latents[:, :, 0].to(
            "cpu", dtype=torch.float16
        ).contiguous()
        latent_name = f"part-{mixture_index:05d}.safetensors"
        save_file({"latents": latent_values}, output / "latents" / latent_name)
        style_id = str(row["mixture_style_id"])
        mixture_dir = output / "images" / style_id
        mixture_dir.mkdir(exist_ok=True)
        rows = []
        for content_index, (image, source_index, source_row, seed) in enumerate(
            zip(
                images,
                selected_content_indices,
                selected_content_rows,
                seeds,
                strict=True,
            )
        ):
            image_path = mixture_dir / f"content-{content_index:02d}.webp"
            image.save(
                image_path, format="WEBP", quality=int(cfg.get("webp_quality", 95))
            )
            rows.append({
                "id": int(cfg.get("image_id_base", 33_000_000_000))
                + mixture_index * images_per_mixture
                + content_index,
                "kind": "lora_mixture",
                "mixture_kind": str(row["kind"]),
                "mixture_index": mixture_index,
                "artist_index": mixture_index,
                "artist": style_id,
                "style_id": style_id,
                "artist_split": "train",
                "split": "train",
                "content_index": content_index,
                "generation_seed": seed,
                "source_content_index": int(source_index),
                "content_prompt": str(
                    source_row.get("prompt", source_row.get("caption", ""))
                ),
                "artist_prompt": str(
                    source_row.get("prompt", source_row.get("caption", ""))
                ),
                "artist_tag": "",
                "components": list(row["components"]),
                "weights": list(row["weights"]),
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
            f"LoRA mixture references {position + 1}/{len(mixture_rows)} "
            f"kind={row['kind']} images={len(completed_rows)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    completed_rows.sort(key=lambda value: int(value["id"]))
    write_records(output / "manifest.parquet", completed_rows)
    summary = {
        "mixtures": len(mixture_rows),
        "images": len(completed_rows),
        "images_per_mixture": images_per_mixture,
        "kinds": {
            kind: sum(str(row["kind"]) == kind for row in mixture_rows)
            for kind in sorted({str(row["kind"]) for row in mixture_rows})
        },
        "functional_teacher_cache": str(teacher_root),
        "content_pool_size": len(content_rows),
        "random_content_per_mixture": random_content,
        "inference_coefficients_exposed_to_student": False,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "generation_summary.json", summary)
    del anima, networks, vae
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


def _cached_training_probe_bank(
    destination: Path,
    cfg: dict[str, Any],
    contents: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Load diverse cached image/text pairs without running VAE or Qwen again."""

    latent_root = destination / str(cfg["latent_cache_directory"])
    text_root = destination / str(cfg["text_cache_directory"])
    latent_rows = [
        row
        for row in pq.read_table(
            latent_root / "manifest.parquet",
            columns=[
                "id",
                "artist",
                "style_id",
                "split",
                "latent_height",
                "latent_width",
                "row_index",
                "cache_shard",
            ],
        ).to_pylist()
        if str(row.get("split", "train")) == "train"
    ]
    requested_shape = tuple(int(value) for value in cfg["probe_latent_shape"])
    if len(requested_shape) != 2:
        raise ValueError("probe_latent_shape must contain [height, width]")
    rng = random.Random(int(cfg.get("seed", 20260823)) ^ 0x434F4E54)
    rng.shuffle(latent_rows)
    candidate_rows: list[dict[str, Any]] = []
    candidate_styles: set[str] = set()
    candidate_limit = max(contents * 16, 256)
    for row in latent_rows:
        height = int(row["latent_height"])
        width = int(row["latent_width"])
        if (height, width) != requested_shape:
            continue
        style_id = str(row.get("style_id", row.get("artist", "")))
        if style_id in candidate_styles:
            continue
        candidate_rows.append(row)
        candidate_styles.add(style_id)
        if len(candidate_rows) >= candidate_limit:
            break
    candidate_ids = {int(row["id"]) for row in candidate_rows}
    variants = [
        str(value)
        for value in cfg.get(
            "content_variants",
            [
                "full",
                "full_quality",
                "tag_dropout",
                "tag_dropout_quality",
                "short",
                "short_quality",
            ],
        )
    ]
    text_by_key = {
        (int(row["id"]), str(row["variant_name"])): row
        for row in pq.read_table(
            text_root / "manifest.parquet",
            columns=[
                "id",
                "artist",
                "style_id",
                "variant_name",
                "caption",
                "token_offset",
                "token_length",
                "cache_shard",
            ],
            filters=[("id", "in", sorted(candidate_ids))],
        ).to_pylist()
        if int(row["id"]) in candidate_ids
    }
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for latent_row in candidate_rows:
        variant = variants[len(selected) % len(variants)]
        text_row = text_by_key.get((int(latent_row["id"]), variant))
        if text_row is None:
            continue
        selected.append((latent_row, text_row))
        if len(selected) == contents:
            break
    if len(selected) != contents:
        raise RuntimeError(
            f"Cached training probes provide {len(selected)}/{contents} rows"
        )

    latent_values: list[torch.Tensor | None] = [None] * contents
    latent_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, (latent_row, _) in enumerate(selected):
        latent_groups[str(latent_row["cache_shard"])].append((index, latent_row))
    for shard_name, rows in latent_groups.items():
        shard = load_file(latent_root / shard_name, device="cpu")["latents"]
        for index, row in rows:
            value = shard[int(row["row_index"])]
            if tuple(value.shape[-2:]) != requested_shape:
                raise RuntimeError(
                    f"Latent shape mismatch for id={row['id']}: "
                    f"manifest={requested_shape}, tensor={tuple(value.shape[-2:])}"
                )
            latent_values[index] = value.to(torch.float16)
        del shard

    text_values: list[torch.Tensor | None] = [None] * contents
    text_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, (_, text_row) in enumerate(selected):
        text_groups[str(text_row["cache_shard"])].append((index, text_row))
    conditioning_length = int(cfg.get("text_conditioning_length", 512))
    for shard_name, rows in text_groups.items():
        shard = load_file(text_root / shard_name, device="cpu")["conditioning"]
        for index, row in rows:
            start = int(row["token_offset"])
            length = min(int(row["token_length"]), conditioning_length)
            value = torch.zeros(
                conditioning_length, shard.shape[-1], dtype=shard.dtype
            )
            value[:length].copy_(shard[start : start + length])
            text_values[index] = value
        del shard

    if any(value is None for value in latent_values + text_values):
        raise RuntimeError("Failed to materialize cached training probe tensors")
    records = [
        {
            "content_index": index,
            "id": int(latent_row["id"]),
            "style_id": str(latent_row.get("style_id", "")),
            "artist": str(latent_row.get("artist", "")),
            "variant_name": str(text_row["variant_name"]),
            "caption": str(text_row.get("caption", "")),
            "latent_height": int(latent_row["latent_height"]),
            "latent_width": int(latent_row["latent_width"]),
            "latent_transform": "none",
        }
        for index, (latent_row, text_row) in enumerate(selected)
    ]
    return (
        torch.stack([value for value in latent_values if value is not None]),
        torch.stack([value for value in text_values if value is not None]),
        records,
    )


def _predict_frozen_anima_in_chunks(
    anima: torch.nn.Module,
    noisy: torch.Tensor,
    contexts: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    batch_rows: int,
) -> torch.Tensor:
    values = []
    for offset in range(0, len(noisy), batch_rows):
        part_noisy = noisy[offset : offset + batch_rows]
        part_context = contexts[offset : offset + batch_rows]
        part_timestep = timesteps[offset : offset + batch_rows]
        padding = torch.zeros(
            len(part_noisy),
            1,
            part_noisy.shape[-2],
            part_noisy.shape[-1],
            device=part_noisy.device,
            dtype=part_noisy.dtype,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = anima(
                part_noisy.unsqueeze(2),
                part_timestep,
                context=part_context,
                padding_mask=padding,
                target_input_ids=None,
            ).squeeze(2)
        values.append(prediction.to("cpu", dtype=torch.float32))
    return torch.cat(values)


def cache_lora_functional_teacher(
    config: dict[str, Any],
    destination: Path,
    *,
    config_key: str = "lora_functional_distillation",
) -> dict[str, Any]:
    cfg = dict(config[config_key])
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
        amplified_count=int(cache_cfg.get("amplified_mixtures", 0)),
        signed_count=int(cache_cfg.get("signed_mixtures", 0)),
        amplified_sum_range=tuple(
            float(value)
            for value in cache_cfg.get("amplified_sum_range", [1.05, 1.35])
        ),
        signed_beta_range=tuple(
            float(value)
            for value in cache_cfg.get("signed_beta_range", [0.05, 0.25])
        ),
        amplified_triple_probability=float(
            cache_cfg.get("amplified_triple_probability", 0.5)
        ),
        signed_triple_probability=float(
            cache_cfg.get("signed_triple_probability", 0.0)
        ),
        signed_l1_maximum=float(cache_cfg.get("signed_l1_maximum", 1.5)),
        seed=int(cache_cfg.get("seed", 20260823)),
    )
    mixture_records = [
        {
            "index": spec.index,
            "kind": spec.kind,
            "components": list(spec.components),
            "weights": list(spec.weights),
            "component_count": len(spec.components),
            "coefficient_sum": sum(spec.weights),
            "coefficient_l1": sum(abs(value) for value in spec.weights),
            "style_ids": [plans[index].style_id for index in spec.components],
            "mixture_style_id": f"lora-mixture-{spec.index:05d}",
        }
        for spec in specs
    ]
    write_records(output / "mixtures.parquet", mixture_records)
    contents = int(cache_cfg.get("contents", 4))
    timesteps = [float(value) for value in cache_cfg.get("timesteps", [0.2, 0.45, 0.7, 0.9])]
    source_mode = str(cache_cfg.get("content_source_mode", "synthetic_teacher"))
    if source_mode == "cached_training":
        latents, contexts, content_rows = _cached_training_probe_bank(
            destination, cache_cfg, contents
        )
    elif source_mode == "synthetic_teacher":
        source = destination / str(cache_cfg["content_source_directory"])
        latents, content_rows = _source_probe_bank(source, contents)
        _, contexts, _ = _load_content_conditions(source)
        contexts = contexts[:contents]
    else:
        raise ValueError(f"Unsupported functional content source: {source_mode}")
    write_records(output / "content_manifest.parquet", content_rows)
    device = str(cache_cfg.get("device", "cuda"))
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=False)
    # Do not instantiate inactive LoRA branches.  Single-only capacity banks
    # need one network, while pair/triple banks retain the previous maximum of
    # two/three.  A zero multiplier still leaves the LoRA module in every
    # linear forward, so allocating all three wastes substantial GPU work.
    active_networks = max(len(spec.components) for spec in specs)
    networks = _create_lora_networks(
        config,
        anima,
        active_networks,
        device,
        dict(cfg.get("network_selection", {})),
    )
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
    for network in networks:
        network.set_multiplier(0.0)
    condition_batch_rows = int(cache_cfg.get("condition_batch_rows", 64))
    base = _predict_frozen_anima_in_chunks(
        anima,
        noisy_flat,
        context_flat,
        timestep_flat,
        batch_rows=condition_batch_rows,
    )
    save_file({
        "base_context": contexts[:contents].to(torch.float16),
        "noisy_inputs": noisy_flat.reshape(contents, len(timesteps), *noisy_flat.shape[1:]).to("cpu", dtype=torch.float16),
        "base_predictions": base.reshape(contents, len(timesteps), *base.shape[1:]).to(dtype=torch.float16),
        "timesteps": torch.tensor(timesteps, dtype=torch.float32),
    }, output / "base.safetensors")
    shard_rows = int(cache_cfg.get("shard_mixtures", 16))
    completed = 0
    effect_rms_by_index: dict[int, float] = {}
    started = time.perf_counter()
    for shard_index, offset in enumerate(range(0, len(specs), shard_rows)):
        shard_path = output / f"effects-{shard_index:05d}.safetensors"
        part = specs[offset : offset + shard_rows]
        if shard_path.exists():
            existing = load_file(shard_path, device="cpu")
            existing_rms = existing["effects"].float().square().mean(
                dim=tuple(range(1, existing["effects"].ndim))
            ).sqrt()
            for mixture_index, rms in zip(
                existing["mixture_indices"].tolist(),
                existing_rms.tolist(),
                strict=True,
            ):
                effect_rms_by_index[int(mixture_index)] = float(rms)
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
            prediction = _predict_frozen_anima_in_chunks(
                anima,
                noisy_flat,
                context_flat,
                timestep_flat,
                batch_rows=condition_batch_rows,
            )
            effect_rows.append(
                (prediction - base).reshape(
                    contents, len(timesteps), *base.shape[1:]
                ).to(dtype=torch.bfloat16)
            )
        stacked_effects = torch.stack(effect_rows)
        save_file({
            "effects": stacked_effects,
            "mixture_indices": torch.tensor([spec.index for spec in part], dtype=torch.int64),
        }, shard_path)
        effect_rms = stacked_effects.float().square().mean(
            dim=tuple(range(1, stacked_effects.ndim))
        ).sqrt()
        for spec, rms in zip(part, effect_rms.tolist(), strict=True):
            effect_rms_by_index[spec.index] = float(rms)
        completed += len(part)
        print(
            f"LoRA functional cache {completed}/{len(specs)} mixtures "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    single_rms = torch.tensor(
        [effect_rms_by_index[index] for index in range(len(plans))],
        dtype=torch.float32,
    )
    single_median = float(single_rms.median())
    ratio_min, ratio_max = (
        float(value)
        for value in cache_cfg.get("stable_effect_ratio_range", [0.5, 2.0])
    )
    effect_stats = []
    for record in mixture_records:
        rms = effect_rms_by_index[int(record["index"])]
        ratio = rms / max(single_median, 1e-8)
        enabled = str(record["kind"]) == "single" or (
            ratio_min <= ratio <= ratio_max
        )
        record["effect_rms"] = rms
        record["effect_to_single_median_ratio"] = ratio
        record["enabled"] = enabled
        effect_stats.append({
            "index": int(record["index"]),
            "kind": str(record["kind"]),
            "effect_rms": rms,
            "effect_to_single_median_ratio": ratio,
            "enabled": enabled,
        })
    write_records(output / "mixtures.parquet", mixture_records)
    write_records(output / "mixture_effect_stats.parquet", effect_stats)
    summary = {
        "mixtures": len(specs),
        "individual": len(plans),
        "pairs": sum(spec.kind == "pair" for spec in specs),
        "triples": sum(spec.kind == "triple" for spec in specs),
        "amplified": sum(spec.kind == "amplified" for spec in specs),
        "amplified_pairs": sum(
            spec.kind == "amplified" and len(spec.components) == 2
            for spec in specs
        ),
        "amplified_triples": sum(
            spec.kind == "amplified" and len(spec.components) == 3
            for spec in specs
        ),
        "signed": sum(spec.kind == "signed" for spec in specs),
        "signed_pairs": sum(
            spec.kind == "signed" and len(spec.components) == 2
            for spec in specs
        ),
        "signed_triples": sum(
            spec.kind == "signed" and len(spec.components) == 3
            for spec in specs
        ),
        "contents": contents,
        "timesteps": timesteps,
        "content_source_mode": source_mode,
        "condition_batch_rows": condition_batch_rows,
        "complete_mixtures": completed,
        "actual_merged_lora_forward": True,
        "single_effect_rms_median": single_median,
        "stable_effect_ratio_range": [ratio_min, ratio_max],
        "disabled_unstable_mixtures": sum(
            not bool(row["enabled"]) for row in effect_stats
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    del anima, networks, base, noisy_flat, context_flat
    gc.collect()
    torch.cuda.empty_cache()
    return {**summary, "reused": False}


def generate_kv_lora_teacher_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return generate_lora_teacher_references(
        config, destination, config_key="kv_lora_teacher_references"
    )


def generate_kv_lora_teacher_references_320(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return generate_lora_teacher_references(
        config, destination, config_key="kv_lora_teacher_references_320"
    )


def generate_kv_activation_mixture_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return generate_lora_mixture_references(
        config, destination, config_key="kv_activation_mixture_references"
    )


def generate_v2d_diverse_mixture_references(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return generate_lora_mixture_references(
        config, destination,
        config_key="lora_mixture_references_v2d_diverse",
    )


def cache_kv_lora_functional_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return cache_lora_functional_teacher(
        config, destination, config_key="kv_lora_functional_teacher"
    )


def cache_v2d_diverse_functional_teacher(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return cache_lora_functional_teacher(
        config, destination,
        config_key="lora_functional_teacher_v2d_diverse",
    )


@torch.no_grad()
def compare_kv_lora_fixed_prompt(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render K/V-LoRA teachers on the exact fixed prompt, seed and trajectory."""

    from .dual_query_external_samples import load_dual_query_external_sample

    cfg = dict(config["kv_lora_teacher_references"])
    prepared = load_dual_query_external_sample(config, destination)
    sample_cfg = dict(prepared["cfg"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"]) / "fixed_teacher_compare"
    output.mkdir(parents=True, exist_ok=True)
    lora_root = destination / str(cfg["lora_directory"])
    plans = _load_lora_plan(lora_root)[:7]
    weights = _weight_paths(lora_root, plans)

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=False
    )
    network = _create_lora_networks(
        config, anima, 1, device, dict(cfg.get("network_selection", {}))
    )[0]
    vae = _load_sampling_vae(config, destination).to(
        device=device, dtype=torch.bfloat16
    )
    vae.requires_grad_(False).eval()

    width = int(sample_cfg["width"])
    height = int(sample_cfg["height"])
    seed = int(sample_cfg["seed"])
    steps = int(sample_cfg["steps"])
    shift = float(sample_cfg.get("flow_shift", 3.0))
    text_cfg = float(sample_cfg["cfg"])
    positive = prepared["positive"].to(device=device, dtype=torch.bfloat16)
    negative = prepared["negative"].to(device=device, dtype=torch.bfloat16)
    if positive.ndim == 2:
        positive = positive[None]
    if negative.ndim == 2:
        negative = negative[None]
    noise = torch.randn(
        1,
        16,
        1,
        height // 8,
        width // 8,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = sigmas * shift / (1 + (shift - 1) * sigmas)

    latents = []
    labels = ["Frozen Anima"]
    network.set_multiplier(0.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents.append(
            _sample_anima_batch(
                anima,
                noise,
                positive,
                negative,
                sigmas,
                text_cfg=text_cfg,
                speed=None,
                generation_seeds=[seed],
            )
        )
    for plan, weight_path in zip(plans, weights, strict=True):
        info = network.load_weights(str(weight_path))
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(f"K/V LoRA key mismatch: {info}")
        network.set_multiplier(1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents.append(
                _sample_anima_batch(
                    anima,
                    noise,
                    positive,
                    negative,
                    sigmas,
                    text_cfg=text_cfg,
                    speed=None,
                    generation_seeds=[seed],
                )
            )
        labels.append(plan.artist)

    latent_batch = torch.cat(latents)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        images = _preview_pixels(vae.decode_to_pixels(latent_batch))
    label_height = 36
    sheet = Image.new(
        "RGB", (width * len(images), height + label_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        x = index * width
        sheet.paste(image, (x, label_height))
        draw.text((x + 8, 10), label, fill="black")
        image.save(output / f"{index:02d}-{label.replace('/', '_')}.webp", "WEBP", quality=95)
    sheet_path = output / "same-prompt-seed-kv-lora-teachers.png"
    sheet.save(sheet_path)
    summary = {
        "artists": [plan.artist for plan in plans],
        "prompt": str(sample_cfg["prompt"]),
        "negative_prompt": str(sample_cfg["negative_prompt"]),
        "seed": seed,
        "steps": steps,
        "text_cfg": text_cfg,
        "width": width,
        "height": height,
        "sheet": str(sheet_path),
    }
    write_json(output / "summary.json", summary)
    return summary


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
        kinds = sorted({str(row["kind"]) for row in self.mixtures})
        self.by_kind = {
            kind: [
                int(row["index"])
                for row in self.mixtures
                if str(row["kind"]) == kind and bool(row.get("enabled", True))
            ]
            for kind in kinds
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

    weighted = {
        "centered_huber": float(weights.get("centered_huber", 1.0)) * centered_huber,
        "centered_direction": float(weights.get("centered_direction", 1.0))
        * (1 - centered_cosine),
        "centered_magnitude": float(weights.get("centered_magnitude", 0.25))
        * centered_magnitude,
        "functional_infonce": float(weights.get("functional_infonce", 0.25))
        * infonce,
        "common_huber": float(weights.get("common_huber", 0.10)) * common_huber,
        "common_direction": float(weights.get("common_direction", 0.05))
        * (1 - common_cosine),
        "common_ratio_excess": float(weights.get("common_ratio_excess", 1.0))
        * common_excess_loss,
    }
    total = sum(weighted.values())
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
        **{
            f"weighted_{name}": value.detach()
            for name, value in weighted.items()
        },
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


def _pack_materialized_mixture_references(
    loader: CachedTeacherReferenceLoader,
    rows: list[dict[str, Any]],
    *,
    references_per_mixture: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read actual merged-LoRA images; coefficients never enter the Reader."""

    style_ids = [str(row["mixture_style_id"]) for row in rows]
    loaded = loader.load_styles(
        style_ids,
        references_per_style=references_per_mixture,
        seed=seed,
    )
    tokens = loaded["tokens"].to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    mask = torch.ones(tokens.shape[:2], device=device, dtype=torch.bool)
    reference_weights = torch.full(
        tokens.shape[:2],
        1.0 / references_per_mixture,
        device=device,
        dtype=torch.float32,
    )
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
    materialized_mixture: bool = False,
    backward_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    candidates = bank.by_kind[kind]
    batch_rows = int(training.get("batch_rows", 4))
    rng = random.Random(int(training.get("seed", 20260823)) + update * 1_000_003)
    indices = rng.sample(candidates, batch_rows)
    rows = [bank.mixtures[index] for index in indices]
    reference_counts = tuple(
        int(value)
        for value in training.get(
            "reference_counts", [training.get("references_per_component", 1)]
        )
    )
    if not reference_counts or any(value <= 0 for value in reference_counts):
        raise ValueError("reference_counts must contain positive integers")
    count_weights = tuple(
        float(value)
        for value in training.get(
            "reference_count_weights", [1.0] * len(reference_counts)
        )
    )
    if (
        len(count_weights) != len(reference_counts)
        or any(value < 0 for value in count_weights)
        or sum(count_weights) <= 0
    ):
        raise ValueError("reference_count_weights must match reference_counts")
    references_per_component = rng.choices(
        reference_counts, weights=count_weights, k=1
    )[0]
    pack_seed = int(training.get("seed", 20260823)) ^ (update * 7919)
    if materialized_mixture:
        references, mask, reference_weights = (
            _pack_materialized_mixture_references(
                loader,
                rows,
                references_per_mixture=references_per_component,
                seed=pack_seed,
                device=device,
            )
        )
    else:
        references, mask, reference_weights = _pack_mixture_references(
            loader,
            rows,
            references_per_component=references_per_component,
            seed=pack_seed,
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
    if backward_scale < 0:
        raise ValueError("backward_scale must be non-negative")
    (float(backward_scale) * loss).backward()
    metrics.update({
        "backward_scale": loss.new_tensor(float(backward_scale)),
        "weighted_loss": (float(backward_scale) * loss).detach(),
        "reference_count": mask.sum(dim=1).float().mean().detach(),
        "timestep": timestep.float().detach(),
        "materialized_mixture_reference": torch.tensor(
            float(materialized_mixture), device=device
        ),
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

    fewshot_cfg = dict(training.get("fewshot_validation", {}))
    fewshot_prompt_cache = None
    if bool(fewshot_cfg.get("enabled", False)):
        fewshot_prompt_cache = _load_or_create_fewshot_prompt_cache(
            config, destination, fewshot_cfg, device
        )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(anima, low_precision_rmsnorm=True, fuse_attention_projections=True)
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    detail_cfg["adapter"].update(dict(cfg.get("adapter_overrides", {})))

    # Optional low-target human flow runs beside the cached functional LoRA
    # teacher.  This keeps the production image objective in the same
    # optimizer step without changing the established teacher-only runners.
    human_flow = dict(cfg.get("human_flow", {}))
    human_flow_enabled = bool(human_flow.get("enabled", False))
    flow_training: dict[str, Any] = {}
    flow_loader = None
    flow_sample_train_loader = None
    flow_validation_loader = None
    flow_accumulation = 0
    if human_flow_enabled:
        flow_training = copy.deepcopy(dict(human_flow["training"]))
        flow_training["steps"] = steps
        flow_detail_cfg = copy.deepcopy(detail_cfg)
        flow_detail_cfg["training"] = flow_training
        flow_detail_cfg["data_mixture"] = {"enabled": False}
        flow_detail_cfg["loader"].update(dict(human_flow.get("loader", {})))
        flow_accumulation = int(human_flow.get("gradient_accumulation_steps", 1))
        if flow_accumulation <= 0:
            raise ValueError("human_flow gradient_accumulation_steps must be positive")
        flow_loader_cfg = _loader_config(
            config,
            flow_detail_cfg,
            split=str(flow_detail_cfg.get("train_split", "train")),
        )
        flow_loader_cfg["gradient_accumulation_steps"] = flow_accumulation
        flow_sample_train_loader, flow_loader = _training_loader(
            destination, flow_detail_cfg, flow_loader_cfg
        )
        for loader in getattr(flow_loader, "loaders", (flow_loader,)):
            _audit_student_prompts(loader)
        flow_validation_cfg = _loader_config(
            config,
            flow_detail_cfg,
            split=str(flow_detail_cfg.get("validation_split", "validation")),
        )
        flow_validation_loader = MultiPromptDualQueryCachedStyleLoader(
            destination, flow_validation_cfg
        )
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device)
    adapter = _build_style_adapter(detail_cfg).to(device)
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("LoRA distillation requires the separated Common/Artist adapter")
    attach_same_q_style_adapter(anima, adapter)
    initial_checkpoint = cfg.get("initial_checkpoint")
    reader_checkpoint = cfg.get("reader_checkpoint")
    initialization = "fresh_reader_and_adapter"
    strength_initialization = "checkpoint_or_constructor_default"
    if initial_checkpoint:
        initial = torch.load(
            destination / str(initial_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        reader.load_state_dict(initial["reader"], strict=True)
        adapter.load_state_dict(initial["adapter"], strict=True)
        adapter.restore_timestep_strength_state()
        initialization = f"full_checkpoint:{initial_checkpoint}"
    elif reader_checkpoint:
        reader_state = torch.load(
            destination / str(reader_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        reader.load_state_dict(reader_state.get("reader", reader_state), strict=True)
        initialization = f"pretrained_reader_fresh_adapter:{reader_checkpoint}"
        strength_initialization = _initialize_fresh_adapter_strength(
            adapter, cfg, destination
        )
    else:
        strength_initialization = _initialize_fresh_adapter_strength(
            adapter, cfg, destination
        )
    adapter.set_bootstrap_phase("combined")
    teacher_schedule = tuple(
        str(value) for value in training.get("teacher_schedule", ())
    )
    bootstrap_teacher_schedule = tuple(
        str(value)
        for value in training.get("bootstrap_teacher_schedule", ("lora_single",))
    )
    freeze_common = bool(training.get("freeze_common", False))
    configured_reader_scope = training.get("reader_trainable_scope")
    reader_scope = str(
        configured_reader_scope
        if configured_reader_scope is not None
        else ("none" if bool(training.get("freeze_reader", False)) else "all")
    )
    reader_parameters = _configure_reader_trainable_scope(reader, reader_scope)
    freeze_reader = not reader_parameters
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
    synthetic_cache_value = cfg["synthetic_reference_cache"]
    synthetic_cache_roots = (
        [destination / str(value) for value in synthetic_cache_value]
        if isinstance(synthetic_cache_value, list)
        else destination / str(synthetic_cache_value)
    )
    synthetic_loader = CachedTeacherReferenceLoader(
        synthetic_cache_roots,
        split="train", style_ids=style_ids, batch_size=int(training.get("batch_rows", 4)),
        references=max_refs, seed=seed ^ 0x53594E54,
        token_lru_shards=int(training.get("token_lru_shards", 8)),
        ram_resident_tokens=bool(training.get("ram_resident_tokens", True)),
        strict_style_ids=True,
    )
    mixture_loader = None
    if cfg.get("mixture_reference_cache"):
        mixture_style_ids = [
            str(row["mixture_style_id"])
            for row in bank.mixtures
            if str(row["kind"]) != "single" and bool(row.get("enabled", True))
        ]
        mixture_loader = CachedTeacherReferenceLoader(
            destination / str(cfg["mixture_reference_cache"]),
            split="train",
            style_ids=mixture_style_ids,
            batch_size=int(training.get("batch_rows", 4)),
            references=max_refs,
            seed=seed ^ 0x4D495854,
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            ram_resident_tokens=bool(training.get("ram_resident_tokens", True)),
            strict_style_ids=True,
        )
    fewshot_validation = None
    if fewshot_prompt_cache is not None:
        fewshot_validation = _prepare_fewshot_validation(
            destination,
            destination / str(cfg["human_reference_cache"]),
            fewshot_cfg,
            fewshot_prompt_cache,
        )
    scheduled_categories = set(teacher_schedule) | set(bootstrap_teacher_schedule)
    native_bank = None
    contexts = None
    native_loader = None
    if "artist_tag" in scheduled_categories:
        native_bank = NativeCenteredTeacherBank.load(
            config,
            destination,
            config_key=str(cfg["native_teacher"]["bank_config_key"]),
        )
        contexts = NativeArtistContextCache(
            destination / str(cfg["native_teacher"]["context_cache"]),
            capacity=int(cfg["native_teacher"].get("context_lru_shards", 8)),
        )
        native_loader = CachedTeacherReferenceLoader(
            [
                destination / str(value)
                for value in cfg["native_teacher"]["reference_caches"]
            ],
            split="train",
            style_ids=list(native_bank.summary["train_style_ids"]),
            batch_size=int(training.get("native_batch_rows", 8)),
            references=int(training.get("native_references", 4)),
            seed=seed ^ 0x4E415449,
            token_lru_shards=int(training.get("token_lru_shards", 8)),
            strict_style_ids=False,
        )

    groups = [
        {"params": adapter.shared_parameters(), "lr": float(training.get("shared_learning_rate", 1e-4)), "name": "shared_kv"},
        {"params": adapter.delta_parameters(), "lr": float(training.get("delta_learning_rate", 2e-4)), "name": "block_delta"},
        {"params": adapter.mixing_parameters(), "lr": float(training.get("mix_learning_rate", 4e-5)), "name": "base_mix", "weight_decay": 0.0},
    ]
    if reader_parameters:
        groups.insert(0, {
            "params": reader_parameters,
            "lr": float(training.get("reader_learning_rate", 2e-5)),
            "name": "reader",
        })
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
    panel_sample_requests = []
    if flow_sample_train_loader is not None and flow_validation_loader is not None:
        sample_seed = int(
            detail_cfg.get("sampling", {}).get("seed", seed ^ 0x5A17)
        )
        panel_sample_requests = [
            (
                "train",
                flow_sample_train_loader,
                episode,
                sample_seed + index * 10_007,
            )
            for index, episode in enumerate(
                _select_sample_episodes(flow_sample_train_loader, 4)
            )
        ] + [
            (
                "validation",
                flow_validation_loader,
                episode,
                sample_seed + (index + 4) * 10_007,
            )
            for index, episode in enumerate(
                _select_sample_episodes(flow_validation_loader, 4)
            )
        ]
    single_only = int(training.get("single_only_steps", 500))
    artist_intro = int(training.get("artist_intro_steps", 0))
    curriculum = str(training.get("curriculum", "legacy"))
    warmup = int(training.get("warmup_steps", 100))
    decay_start = int(training.get("lr_decay_start_step", 6000))
    min_lr = float(training.get("minimum_lr_ratio", 0.1))
    log_every = int(training.get("log_every", 10))
    checkpoint_every = int(training.get("checkpoint_every", 250))
    sample_every = int(training.get("sample_every", 1000))
    panel_sample_every = int(training.get("panel_sample_every", sample_every))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    reader_max_grad_norm = float(
        training.get("reader_max_grad_norm", max_grad_norm)
    )
    running: dict[str, list[float]] = defaultdict(list)
    updates = defaultdict(int)
    native_common_cache: dict[tuple[int, int], torch.Tensor] = {}
    started = time.perf_counter()
    flow_prefetched = None
    panel_vae = None
    if flow_loader is not None:
        flow_prefetched = flow_loader.prefetch(
            start_step * flow_accumulation,
            (steps - start_step) * flow_accumulation,
            workers=int(human_flow.get("prefetch_workers", 2)),
            depth=int(human_flow.get("prefetch_batches", 6)),
        )
    resumed_panel_summary = (
        output / "samples" / f"step-{start_step:07d}" / "summary.json"
    )
    try:
        if (
            start_step > 0
            and panel_sample_every > 0
            and panel_sample_requests
            and start_step % panel_sample_every == 0
            and not resumed_panel_summary.exists()
        ):
            sample_records, panel_vae = _sample_query_style_tokenizer(
                anima,
                adapter,
                reader,
                panel_sample_requests,
                config,
                destination,
                output,
                device,
                start_step,
                panel_vae,
                config_section="detail_preserving_style_cross_attention",
            )
            print(
                f"LoRA distill resumed functional panel step={start_step} "
                f"samples={len(sample_records)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb

                wandb_run.log({
                    "val/functional/panel": [
                        wandb.Image(str(path), caption=label)
                        for label, path in sample_records
                    ]
                }, step=start_step)
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
            common_freeze_step = int(training.get("common_freeze_step", 0))
            common_frozen_now = bool(
                freeze_common
                or (common_freeze_step > 0 and step > common_freeze_step)
            )
            for parameter in adapter.common_parameters():
                parameter.requires_grad_(not common_frozen_now)
            for group in optimizer.param_groups:
                if str(group["name"]) == "common" and common_frozen_now:
                    group["lr"] = 0.0
            gain_start = float(training.get("global_gain_start", 1.0))
            gain_end = float(training.get("global_gain_end", gain_start))
            gain_steps = max(1, int(training.get("global_gain_ramp_steps", 1)))
            gain_progress = min(1.0, max(0.0, (step - 1) / gain_steps))
            adapter.global_gain = gain_start + gain_progress * (gain_end - gain_start)
            optimizer.zero_grad(set_to_none=True)

            if flow_prefetched is not None:
                flow_start = float(human_flow.get("weight_start", 0.25))
                flow_end = float(human_flow.get("weight_end", 1.0))
                flow_ramp_steps = max(1, int(human_flow.get("weight_ramp_steps", 500)))
                flow_progress = min(1.0, max(0.0, (step - 1) / flow_ramp_steps))
                flow_weight = flow_start + flow_progress * (flow_end - flow_start)
                flow_rows: list[dict[str, torch.Tensor]] = []
                for micro in range(flow_accumulation):
                    flow_batch = next(flow_prefetched)
                    generator = torch.Generator(device=device).manual_seed(
                        seed ^ 0x464C4F57 ^ (step * 100_003 + micro)
                    )
                    flow_loss, flow_metrics, _ = _flow_step(
                        anima,
                        reader,
                        adapter,
                        flow_batch,
                        device,
                        flow_training,
                        None,
                        generator=generator,
                        step=step,
                        mode="curriculum",
                        train_auxiliaries=False,
                        measure_base=(step % int(training.get("log_every", 10)) == 0),
                    )
                    (flow_weight * flow_loss / flow_accumulation).backward()
                    flow_rows.append(flow_metrics)
                for key in set().union(*(row.keys() for row in flow_rows)):
                    values = [row[key] for row in flow_rows if key in row]
                    if values and all(value.numel() == 1 for value in values):
                        running[f"human_flow/{key}"].append(
                            float(torch.stack(values).mean().detach())
                        )
                running["human_flow/weight"].append(flow_weight)
            if teacher_schedule:
                category = scheduled_teacher_category(
                    step,
                    single_only_steps=single_only,
                    schedule=teacher_schedule,
                    bootstrap_schedule=bootstrap_teacher_schedule,
                )
            else:
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
            mixture_kind = None
            if category == "artist_tag":
                assert native_bank is not None
                assert contexts is not None
                assert native_loader is not None
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
                                training.get(
                                    "native_common_weight",
                                    training.get("native_artist_mean_weight", 0.25),
                                )
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
                kind = "single"
                materialized_mixture = False
                if category == "lora_mixture":
                    configured_kinds = [
                        str(value)
                        for value in training.get(
                            "mixture_kind_schedule", ["pair", "triple"]
                        )
                    ]
                    if step < int(training.get("extrapolation_start_step", 0)):
                        configured_kinds = [
                            value
                            for value in configured_kinds
                            if value not in {"amplified", "signed"}
                        ]
                    available_kinds = [
                        value
                        for value in configured_kinds
                        if value in bank.by_kind and bank.by_kind[value]
                    ]
                    if not available_kinds:
                        raise RuntimeError("No configured LoRA mixture kind is available")
                    kind = available_kinds[
                        (updates[category] - 1) % len(available_kinds)
                    ]
                    mixture_kind = kind
                    materialized_mixture = mixture_loader is not None
                if materialized_mixture:
                    domain = "materialized_mixture"
                    domain_loader = mixture_loader
                else:
                    domain_schedule = tuple(
                        str(value)
                        for value in training.get(
                            "lora_reference_domain_schedule", ("human", "synthetic")
                        )
                    )
                    domain = scheduled_reference_domain(
                        updates[category], domain_schedule
                    )
                    domain_loader = (
                        human_loader if domain == "human" else synthetic_loader
                    )
                metrics = _lora_teacher_step(
                    anima, reader, adapter, bank, domain_loader,
                    kind=kind, update=updates[category], step=step,
                    device=device, training={**training, "seed": seed},
                    materialized_mixture=materialized_mixture,
                    backward_scale=float(training.get("teacher_backward_scale", 1.0)),
                )
                metrics["domain_synthetic"] = torch.tensor(
                    float(domain != "human"), device=device
                )
            adapter_parameters = [
                parameter
                for group in optimizer.param_groups
                if str(group["name"]) != "reader"
                for parameter in group["params"]
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter_parameters, max_grad_norm, foreach=True
            )
            reader_grad_norm = None
            if reader_parameters:
                reader_grad_norm = torch.nn.utils.clip_grad_norm_(
                    reader_parameters, reader_max_grad_norm, foreach=True
                )
            optimizer.step()
            domain_name = None
            if category != "artist_tag":
                domain_name = str(domain)
            for key, value in metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    running[f"{category}/{key}"].append(float(value.detach()))
                    if domain_name is not None:
                        running[f"{category}/{domain_name}/{key}"].append(
                            float(value.detach())
                        )
                    if mixture_kind is not None:
                        running[f"{category}/kind/{mixture_kind}/{key}"].append(
                            float(value.detach())
                        )
            running["optimizer/grad_norm"].append(float(grad_norm))
            if reader_grad_norm is not None:
                running["optimizer/reader_grad_norm"].append(
                    float(reader_grad_norm)
                )
            if step % log_every == 0:
                row = {key: sum(values) / len(values) for key, values in running.items() if values}
                for group in optimizer.param_groups:
                    row[f"optimizer/lr/{group['name']}"] = float(group["lr"])
                row["progress/category"] = {"artist_tag": 0, "lora_single": 1, "lora_mixture": 2}[category]
                row["progress/common_frozen"] = float(common_frozen_now)
                row["progress/global_gain"] = float(adapter.global_gain)
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
                    output, device, step,
                    strengths_override=[
                        float(value)
                        for value in training.get(
                            "fixed_sample_strengths", [1.0]
                        )
                    ],
                    sample_group="fixed_reference_samples",
                )
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "val/functional/fixed_reference": wandb.Image(
                            sample["sheet"], caption=f"step {step}"
                        )
                    }, step=step)
            if (
                panel_sample_every > 0
                and panel_sample_requests
                and step % panel_sample_every == 0
            ):
                sample_records, panel_vae = _sample_query_style_tokenizer(
                    anima,
                    adapter,
                    reader,
                    panel_sample_requests,
                    config,
                    destination,
                    output,
                    device,
                    step,
                    panel_vae,
                    config_section="detail_preserving_style_cross_attention",
                )
                print(
                    f"LoRA distill functional panel step={step} "
                    f"samples={len(sample_records)}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb

                    wandb_run.log({
                        "val/functional/panel": [
                            wandb.Image(str(path), caption=label)
                            for label, path in sample_records
                        ]
                    }, step=step)
            if fewshot_validation is not None:
                quick_every = int(fewshot_cfg.get("quick_every", 500))
                full_every = int(fewshot_cfg.get("full_every", 1000))
                full_due = full_every > 0 and step % full_every == 0
                quick_due = quick_every > 0 and step % quick_every == 0
                sweeps: list[tuple[str, tuple[int, ...], int | None]] = []
                if full_due:
                    counts = tuple(
                        int(value)
                        for value in fewshot_cfg.get(
                            "full_reference_counts", (1, 2, 4, 8)
                        )
                    )
                    full_modes = tuple(
                        str(value)
                        for value in fewshot_cfg.get(
                            "full_modes", ("controlled", "diverse")
                        )
                    )
                    unknown_modes = set(full_modes) - {"controlled", "diverse"}
                    if unknown_modes:
                        raise ValueError(
                            f"Unknown full few-shot modes: {sorted(unknown_modes)}"
                        )
                    sweeps.extend((mode, counts, None) for mode in full_modes)
                elif quick_due:
                    sweeps.append((
                        "diverse",
                        tuple(
                            int(value)
                            for value in fewshot_cfg.get(
                                "quick_reference_counts", (1, 4)
                            )
                        ),
                        int(fewshot_cfg.get("quick_artist_count", 4)),
                    ))
                for mode, counts, artist_limit in sweeps:
                    result = _generate_fewshot_reference_sweep(
                        fewshot_validation,
                        config,
                        destination,
                        anima,
                        reader,
                        adapter,
                        output,
                        device,
                        step,
                        mode=mode,
                        reference_counts=counts,
                        artist_limit=artist_limit,
                    )
                    if wandb_run is not None:
                        import wandb

                        payload: dict[str, Any] = {
                            f"val/fewshot/{mode}/sheet": wandb.Image(
                                result["sheet"], caption=f"step {step} {mode}"
                            )
                        }
                        for count, values in result["metrics"].items():
                            for name, value in values.items():
                                payload[f"val/fewshot/{mode}/{count}/{name}"] = value
                        wandb_run.log(payload, step=step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        adapter.clear_style_tokens()
    summary = {
        "steps": steps,
        "start_step": start_step,
        "updates": dict(updates),
        "elapsed_s": time.perf_counter() - started,
        "teacher_schedule_after_bootstrap": list(teacher_schedule) or "legacy",
        "teacher_schedule_during_bootstrap": list(bootstrap_teacher_schedule),
        "lora_reference_domain_schedule": list(
            training.get("lora_reference_domain_schedule", ("human", "synthetic"))
        ),
        "mixture_kind_schedule": list(
            training.get("mixture_kind_schedule", ("pair", "triple"))
        ),
        "curriculum": curriculum,
        "common_frozen": freeze_common,
        "common_freeze_step": int(training.get("common_freeze_step", 0)),
        "human_flow_enabled": human_flow_enabled,
        "functional_panel_enabled": bool(panel_sample_requests),
        "panel_sample_every": panel_sample_every,
        "reader_frozen": freeze_reader,
        "reader_trainable_scope": reader_scope,
        "initialization": initialization,
        "strength_initialization": strength_initialization,
        "inference_contract": (
            "reference_tokens -> Reader -> per-block style K/V; "
            "no LoRA dictionary, retrieval, artist ID, or runtime LoRA mixture"
        ),
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


def train_fresh_v34_low_target_kv_lora_joint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train fresh v34 topology from low-target flow and K/V-only LoRA teachers."""

    return train_lora_functional_distillation(
        config,
        destination,
        config_key="fresh_v34_low_target_kv_lora_joint",
    )


def smoke_test_fresh_v34_low_target_kv_lora_joint(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["fresh_v34_low_target_kv_lora_joint"]
    cfg["output_directory"] = str(
        Path(cfg["output_directory"]).with_name(
            "fresh_v34_low_target_kv_lora_joint_smoke"
        )
    )
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    cfg["training"]["fewshot_validation"]["enabled"] = False
    cfg["human_flow"]["prefetch_batches"] = 2
    return train_lora_functional_distillation(
        effective,
        destination,
        steps_override=2,
        config_key="fresh_v34_low_target_kv_lora_joint",
    )


def train_direct_reference_kv_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Train a reference-only direct K/V adapter from functional teachers."""

    return train_lora_functional_distillation(
        config,
        destination,
        config_key="direct_reference_kv_distillation",
    )


def smoke_test_direct_reference_kv_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["direct_reference_kv_distillation"]
    cfg["output_directory"] = str(
        Path(cfg["output_directory"]).with_name(
            "direct_reference_kv_distillation_smoke"
        )
    )
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    cfg["training"]["single_only_steps"] = 1
    return train_lora_functional_distillation(
        effective,
        destination,
        steps_override=2,
        config_key="direct_reference_kv_distillation",
    )


def train_v2d_diverse_mixture_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Run the preserved v2d model with materialized diverse mixtures."""

    return train_lora_functional_distillation(
        config,
        destination,
        config_key="direct_reference_kv_distillation_v2d_diverse",
    )


def evaluate_v2d_diverse_fewshot(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render artist-disjoint 1/2/4/8-reference sweeps from finished v2d checkpoints."""

    cfg = copy.deepcopy(config["direct_reference_kv_distillation_v2d_diverse"])
    evaluation = dict(cfg["fewshot_evaluation"])
    source_key = str(
        evaluation.pop("source_config_key", "direct_reference_kv_distillation")
    )
    fewshot_cfg = copy.deepcopy(
        config[source_key]["training"]["fewshot_validation"]
    )
    checkpoint_steps = tuple(
        int(value) for value in evaluation.pop("checkpoint_steps", (8000,))
    )
    modes = tuple(str(value) for value in evaluation.pop("modes", ("controlled",)))
    reference_counts = tuple(
        int(value)
        for value in evaluation.pop("reference_counts", (1, 2, 4, 8))
    )
    prompt_seed_base = evaluation.pop("prompt_seed_base", None)
    fewshot_cfg.update(evaluation)
    if prompt_seed_base is not None:
        for index, case in enumerate(fewshot_cfg["prompt_cases"]):
            case["seed"] = int(prompt_seed_base) + index

    device = str(cfg["training"].get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    checkpoint_root = output / "checkpoints"
    prompt_cache = _load_or_create_fewshot_prompt_cache(
        config, destination, fewshot_cfg, device
    )
    prepared = _prepare_fewshot_validation(
        destination,
        destination / str(cfg["human_reference_cache"]),
        fewshot_cfg,
        prompt_cache,
    )

    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    detail_cfg = copy.deepcopy(config["detail_preserving_style_cross_attention"])
    detail_cfg["adapter"].update(dict(cfg.get("adapter_overrides", {})))
    reader = DetailPreservingTypedSlotReader(**dict(detail_cfg["model"])).to(device).eval()
    adapter = _build_style_adapter(detail_cfg).to(device).eval()
    if not isinstance(adapter, SeparatedCommonArtistKVStyleCrossAttention):
        raise TypeError("v2d few-shot evaluation requires the separated adapter")
    attach_same_q_style_adapter(anima, adapter)

    results = []
    try:
        for step in checkpoint_steps:
            checkpoint = checkpoint_root / f"step-{step:07d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            reader.load_state_dict(state["reader"], strict=True)
            adapter.load_state_dict(state["adapter"], strict=True)
            adapter.restore_timestep_strength_state()
            del state
            for mode in modes:
                result = _generate_fewshot_reference_sweep(
                    prepared,
                    config,
                    destination,
                    anima,
                    reader,
                    adapter,
                    output,
                    device,
                    step,
                    mode=mode,
                    reference_counts=reference_counts,
                )
                results.append(result)
                print(
                    f"v2d few-shot evaluation step={step} mode={mode} "
                    f"sheet={result['sheet']}",
                    flush=True,
                )
    finally:
        adapter.clear_style_tokens()

    summary = {
        "checkpoint_steps": list(checkpoint_steps),
        "modes": list(modes),
        "reference_counts": list(reference_counts),
        "split": str(fewshot_cfg.get("split", "validation")),
        "selection_seed": int(fewshot_cfg["selection_seed"]),
        "reference_seed": int(fewshot_cfg["reference_seed"]),
        "prompt_seed_base": (
            None if prompt_seed_base is None else int(prompt_seed_base)
        ),
        "artist_ids": list(prepared["style_ids"]),
        "results": results,
    }
    write_json(output / "fewshot_validation" / "evaluation_summary.json", summary)
    return summary


def smoke_test_v2d_diverse_mixture_distillation(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    cfg = effective["direct_reference_kv_distillation_v2d_diverse"]
    cfg["output_directory"] = str(
        Path(cfg["output_directory"]).with_name(
            "direct_reference_kv_distillation_v2d_diverse_smoke"
        )
    )
    cfg["training"]["wandb"]["enabled"] = False
    cfg["training"]["resume"] = False
    cfg["training"]["checkpoint_every"] = 1
    cfg["training"]["sample_every"] = 0
    cfg["training"]["single_only_steps"] = 1
    return train_lora_functional_distillation(
        effective,
        destination,
        steps_override=2,
        config_key="direct_reference_kv_distillation_v2d_diverse",
    )
