from __future__ import annotations

import argparse
import copy
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from .compare_resampler_representations import _loader_config
from .compare_typed_attention_to_small_samples import _load_typed_tokenizer
from .config import load_config, output_dir
from .dual_query_external_samples import load_dual_query_external_sample
from .external_style_tokenizer_sheet import (
    _decode_latents,
    _denoise_batch,
    _fit_with_padding,
)
from .query_style_tokenizer import _select_sample_episodes
from .style_tokenizer import _reference_tokens, insert_style_tokens
from .style_transfer import (
    ProductionStyleLoader,
    _optimize_frozen_anima,
    _pad_text_conditions,
    _resolve_anima_model,
)


@dataclass
class ArtistCase:
    index: int
    style_id: str
    target_id: int
    reference_id: int
    reference_path: Path
    style_tokens: torch.Tensor


def _slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-_")
    return value[:80] or "artist"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _label(image: Image.Image, lines: list[str]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(24)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    padding = 12
    spacing = 5
    width = max(box[2] - box[0] for box in boxes) + padding * 2
    height = (
        sum(box[3] - box[1] for box in boxes)
        + spacing * (len(lines) - 1)
        + padding * 2
    )
    draw.rounded_rectangle(
        (8, 8, 8 + width, 8 + height), radius=8, fill=(0, 0, 0, 185)
    )
    y = 8 + padding
    for line, box in zip(lines, boxes, strict=True):
        draw.text((8 + padding, y), line, font=font, fill="white")
        y += box[3] - box[1] + spacing


def _title_tile(size: tuple[int, int], title: str, subtitle: str) -> Image.Image:
    tile = Image.new("RGB", size, (242, 242, 242))
    draw = ImageDraw.Draw(tile)
    draw.text((28, 36), title, font=_font(34), fill="black")
    draw.multiline_text(
        (28, 94), subtitle, font=_font(23), fill=(50, 50, 50), spacing=9
    )
    return tile


def _prepare_cases(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    device: str,
) -> tuple[list[ArtistCase], dict[str, Any]]:
    source_cfg = copy.deepcopy(config[str(cfg["source_config_section"])])
    source_cfg["seed"] = int(cfg["selection_seed"])
    loader_cfg = dict(source_cfg.get("loader", {}))
    loader_cfg.update(dict(cfg.get("loader", {})))
    source_cfg["loader"] = loader_cfg
    loader = ProductionStyleLoader(
        destination,
        _loader_config(
            config,
            source_cfg,
            token_cache=str(cfg["typed_token_cache"]),
            references=1,
            split="validation",
        ),
    )
    tokenizer, metadata = _load_typed_tokenizer(
        destination / str(cfg["typed_checkpoint"]), device
    )
    indices = _select_sample_episodes(loader, int(cfg["artists"]))
    cases: list[ArtistCase] = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        for artist_index, episode_index in enumerate(indices, start=1):
            batch = loader.load_step(episode_index)
            references, reference_mask = _reference_tokens(
                batch, device, mode="heldout"
            )
            tokens = tokenizer(references, reference_mask)[:1]
            episode = batch["episodes"][0]
            if len(episode.reference_ids) != 1:
                raise RuntimeError("Validation sweep requires exactly one reference")
            reference_id = int(episode.reference_ids[0])
            reference_path = Path(
                str(loader.style_by_id[reference_id]["local_path"])
            )
            cases.append(
                ArtistCase(
                    index=artist_index,
                    style_id=str(episode.style_id),
                    target_id=int(episode.target_id),
                    reference_id=reference_id,
                    reference_path=reference_path,
                    style_tokens=tokens.to("cpu", dtype=torch.bfloat16).contiguous(),
                )
            )
            print(
                f"prepared validation artist {artist_index}/{len(indices)} "
                f"style={episode.style_id} reference={reference_id}",
                flush=True,
            )
    del tokenizer, loader
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return cases, metadata


def _prepare_fixed_cases(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    device: str,
) -> tuple[list[ArtistCase], dict[str, Any]]:
    sample = load_dual_query_external_sample(config, destination)
    tokenizer, metadata = _load_typed_tokenizer(
        destination / str(cfg["typed_checkpoint"]), device
    )
    references = sample["reference_tokens"].to(
        device, dtype=torch.bfloat16, non_blocking=True
    )[:, None]
    reference_mask = torch.ones(
        references.shape[:2], device=device, dtype=torch.bool
    )
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        tokens = tokenizer(references, reference_mask).to(
            "cpu", dtype=torch.bfloat16
        )
    cases = [
        ArtistCase(
            index=index,
            style_id=path.parent.name,
            target_id=-1,
            reference_id=index,
            reference_path=path,
            style_tokens=tokens[index - 1 : index].contiguous(),
        )
        for index, path in enumerate(sample["paths"], start=1)
    ]
    for case in cases:
        print(
            f"prepared fixed reference {case.index}/{len(cases)} "
            f"name={case.style_id}",
            flush=True,
        )
    del tokenizer, references, reference_mask, tokens
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return cases, metadata


def _fixed_conditions(
    config: dict[str, Any], destination: Path, cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
    sample = load_dual_query_external_sample(config, destination)
    fixed_cfg = dict(config[str(cfg["source_config_section"])]["fixed_reference_sampling"])
    expected_prompt = str(fixed_cfg["prompt"])
    expected_negative = str(fixed_cfg["negative_prompt"])
    if (
        str(sample["cfg"]["prompt"]) != expected_prompt
        or str(sample["cfg"]["negative_prompt"]) != expected_negative
    ):
        raise RuntimeError("Cached fixed prompt does not match the typed model contract")
    positive = _pad_text_conditions([sample["positive"][0]], 512)
    negative = _pad_text_conditions([sample["negative"][0]], 512)
    return positive, negative, int(sample["length"]), {
        "prompt": expected_prompt,
        "negative_prompt": expected_negative,
    }


def _noise(generation: dict[str, Any]) -> torch.Tensor:
    return torch.randn(
        1,
        16,
        1,
        int(generation["height"]) // 8,
        int(generation["width"]) // 8,
        generator=torch.Generator(device="cpu").manual_seed(
            int(generation["seed"])
        ),
        dtype=torch.float32,
    ).to(torch.bfloat16)


def _generate(
    config: dict[str, Any],
    destination: Path,
    cfg: dict[str, Any],
    cases: list[ArtistCase],
    positive: torch.Tensor,
    negative: torch.Tensor,
    conditioning_length: int,
    device: str,
) -> dict[str, list[Image.Image]]:
    generation = dict(cfg["generation"])
    multipliers = [float(value) for value in cfg["style_multipliers"]]
    batch_size = int(generation.get("batch_size", 4))
    initial_noise = _noise(generation).to(device)
    positive = positive.to(device, dtype=torch.bfloat16)
    negative = negative.to(device, dtype=torch.bfloat16)
    anima = _resolve_anima_model(config, destination, device).requires_grad_(False).eval()
    _optimize_frozen_anima(
        anima, low_precision_rmsnorm=True, fuse_attention_projections=True
    )
    denoise = {
        "steps": int(generation["steps"]),
        "flow_shift": float(generation.get("flow_shift", 3.0)),
        "cfg_scale": float(generation["cfg"]),
    }
    groups: dict[str, torch.Tensor] = {
        "base": _denoise_batch(
            anima, initial_noise, positive, negative, **denoise
        ).to("cpu")
    }
    total_jobs = len(cases) * len(multipliers)
    completed = 0
    for multiplier in multipliers:
        for offset in range(0, len(cases), batch_size):
            chunk = cases[offset : offset + batch_size]
            tokens = torch.cat([case.style_tokens for case in chunk]).to(device)
            lengths = torch.full(
                (len(chunk),), conditioning_length, device=device, dtype=torch.long
            )
            text = positive.expand(len(chunk), -1, -1)
            style_context = insert_style_tokens(text.clone(), lengths, tokens)
            latents = _denoise_batch(
                anima,
                initial_noise,
                text,
                negative,
                style_context=style_context,
                style_multiplier=multiplier,
                **denoise,
            ).to("cpu")
            for case, latent in zip(chunk, latents, strict=True):
                groups[f"artist::{case.index}::{multiplier:g}"] = latent[None]
            completed += len(chunk)
            print(f"generated styled {completed}/{total_jobs}", flush=True)
    del anima, positive, negative, initial_noise
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _decode_latents(
        config,
        destination,
        groups,
        device,
        int(generation.get("vae_batch_size", 4)),
    )


def _make_overview(
    cases: list[ArtistCase],
    images: dict[str, list[Image.Image]],
    multipliers: list[float],
    prompt: str,
    seed: int,
    *,
    title: str = "VALIDATION / ONE REFERENCE",
    label_prefix: str = "VAL",
) -> Image.Image:
    base = images["base"][0]
    size = base.size
    sheet = Image.new("RGB", ((len(cases) + 1) * size[0], 3 * size[1]), "white")
    sheet.paste(
        _title_tile(
            size,
            title,
            f"same prompt + same seed={seed}\nrow 1: reference (padded)\nrow 2: 1x\nrow 3: 2x\n\n{prompt}",
        ),
        (0, 0),
    )
    for row, multiplier in enumerate(multipliers, start=1):
        baseline = base.copy()
        _label(baseline, ["NO STYLE", f"comparison row {multiplier:g}x"])
        sheet.paste(baseline, (0, row * size[1]))
    for column, case in enumerate(cases, start=1):
        with Image.open(case.reference_path) as source:
            reference = _fit_with_padding(source, size)
        _label(
            reference,
            [f"{label_prefix} {case.index:02d}", case.style_id, "REFERENCE 1/1"],
        )
        sheet.paste(reference, (column * size[0], 0))
        for row, multiplier in enumerate(multipliers, start=1):
            image = images[f"artist::{case.index}::{multiplier:g}"][0].copy()
            _label(
                image,
                [
                    f"{label_prefix} {case.index:02d}",
                    case.style_id,
                    f"STYLE {multiplier:g}x",
                ],
            )
            sheet.paste(image, (column * size[0], row * size[1]))
    return sheet


def generate_validation_artist_sweep(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["validation_artist_strength_sweep"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    raw_root = output / "raw"
    artist_root = output / "artists"
    raw_root.mkdir(parents=True, exist_ok=True)
    artist_root.mkdir(parents=True, exist_ok=True)
    cases, checkpoint = _prepare_cases(config, destination, cfg, device)
    positive, negative, length, prompt_info = _fixed_conditions(
        config, destination, cfg
    )
    images = _generate(
        config, destination, cfg, cases, positive, negative, length, device
    )
    generation = dict(cfg["generation"])
    multipliers = [float(value) for value in cfg["style_multipliers"]]
    if multipliers != [1.0, 2.0]:
        raise ValueError("This comparison contract requires strengths [1.0, 2.0]")
    base = images["base"][0]
    base.save(raw_root / "no-style.png")
    records = []
    for case in cases:
        stem = f"val-{case.index:02d}-{_slug(case.style_id)}"
        with Image.open(case.reference_path) as source:
            reference = _fit_with_padding(source, base.size)
        reference.save(raw_root / f"{stem}-reference-padded.png")
        styled = []
        for multiplier in multipliers:
            image = images[f"artist::{case.index}::{multiplier:g}"][0]
            image.save(raw_root / f"{stem}-style-{multiplier:g}x.png")
            styled.append(image)
        per_artist = Image.new("RGB", (4 * base.width, base.height), "white")
        cells = [reference, base] + styled
        names = ["REFERENCE 1/1", "NO STYLE", "STYLE 1x", "STYLE 2x"]
        for column, (cell, name) in enumerate(zip(cells, names, strict=True)):
            value = cell.copy()
            _label(value, [f"VAL {case.index:02d} | {case.style_id}", name])
            per_artist.paste(value, (column * base.width, 0))
        artist_sheet = artist_root / f"{stem}.png"
        per_artist.save(artist_sheet, compress_level=4)
        records.append(
            {
                "index": case.index,
                "style_id": case.style_id,
                "target_id_used_only_for_episode_selection": case.target_id,
                "reference_id": case.reference_id,
                "reference_path": str(case.reference_path),
                "artist_sheet": str(artist_sheet),
            }
        )
    overview = output / "validation-artists-one-reference-1x-2x.png"
    _make_overview(
        cases,
        images,
        multipliers,
        prompt_info["prompt"],
        int(generation["seed"]),
    ).save(overview, compress_level=4)
    summary = {
        "contract": {
            "split": "validation",
            "artists": len(cases),
            "references_per_artist": 1,
            "same_prompt_for_all_artists": True,
            "same_seed_for_all_artists": True,
            "validation_artist_id_in_prompt": False,
            "reference_images_are_held_out_from_target": True,
            "strengths": multipliers,
            "strength_formula": "uncond + cfg*text_delta + cfg*strength*style_delta",
        },
        "checkpoint": checkpoint,
        "generation": generation,
        **prompt_info,
        "overview": str(overview),
        "artists": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def generate_fixed_reference_sweep(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = dict(config["fixed_reference_strength_sweep"])
    device = str(cfg.get("device", "cuda"))
    output = destination / str(cfg["output_directory"])
    raw_root = output / "raw"
    artist_root = output / "references"
    raw_root.mkdir(parents=True, exist_ok=True)
    artist_root.mkdir(parents=True, exist_ok=True)
    cases, checkpoint = _prepare_fixed_cases(config, destination, cfg, device)
    positive, negative, length, prompt_info = _fixed_conditions(
        config, destination, cfg
    )
    images = _generate(
        config, destination, cfg, cases, positive, negative, length, device
    )
    generation = dict(cfg["generation"])
    multipliers = [float(value) for value in cfg["style_multipliers"]]
    if multipliers != [1.0, 2.0]:
        raise ValueError("This comparison contract requires strengths [1.0, 2.0]")
    base = images["base"][0]
    base.save(raw_root / "no-style.png")
    records = []
    for case in cases:
        stem = f"fixed-{case.index:02d}-{_slug(case.style_id)}"
        with Image.open(case.reference_path) as source:
            reference = _fit_with_padding(source, base.size)
        reference.save(raw_root / f"{stem}-reference-padded.png")
        styled = []
        for multiplier in multipliers:
            image = images[f"artist::{case.index}::{multiplier:g}"][0]
            image.save(raw_root / f"{stem}-style-{multiplier:g}x.png")
            styled.append(image)
        sheet = Image.new("RGB", (4 * base.width, base.height), "white")
        cells = [reference, base] + styled
        names = ["REFERENCE 1/1", "NO STYLE", "STYLE 1x", "STYLE 2x"]
        for column, (cell, name) in enumerate(zip(cells, names, strict=True)):
            value = cell.copy()
            _label(value, [f"FIXED {case.index:02d} | {case.style_id}", name])
            sheet.paste(value, (column * base.width, 0))
        sheet_path = artist_root / f"{stem}.png"
        sheet.save(sheet_path, compress_level=4)
        records.append(
            {
                "index": case.index,
                "name": case.style_id,
                "reference_path": str(case.reference_path),
                "sheet": str(sheet_path),
            }
        )
    overview = output / "fixed-references-one-reference-1x-2x.png"
    _make_overview(
        cases,
        images,
        multipliers,
        prompt_info["prompt"],
        int(generation["seed"]),
        title="FIXED SAMPLES / ONE REFERENCE",
        label_prefix="FIXED",
    ).save(overview, compress_level=4)
    summary = {
        "contract": {
            "source": "fixed_reference_sampling/TestSample1-7",
            "references": len(cases),
            "references_per_result": 1,
            "same_prompt_for_all_references": True,
            "same_seed_for_all_references": True,
            "strengths": multipliers,
            "strength_formula": "uncond + cfg*text_delta + cfg*strength*style_delta",
        },
        "checkpoint": checkpoint,
        "generation": generation,
        **prompt_info,
        "overview": str(overview),
        "references": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.fixed:
        if args.smoke:
            raise ValueError("--smoke is only supported by the validation sweep")
        generate_fixed_reference_sweep(config, output_dir(config))
        return
    if args.smoke:
        config = copy.deepcopy(config)
        cfg = config["validation_artist_strength_sweep"]
        cfg["artists"] = 1
        cfg["output_directory"] = "diagnostics/validation-artist-strength-smoke"
    generate_validation_artist_sweep(config, output_dir(config))


if __name__ == "__main__":
    main()
