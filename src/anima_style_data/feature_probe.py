from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .cradio import (
    _load_cradio,
    _storage_dtype,
    _summary_features,
    content_subtracted_summary,
    preprocess_cradio_image,
)
from .io import read_records, write_json, write_records


def _caption_index(destination: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for path in sorted((destination / "captions").glob("part-*.parquet")):
        for row in read_records(path):
            result[int(row["id"])] = row["content_caption"]
    if not result:
        raise FileNotFoundError("C-RADIO probe requires completed caption shards")
    return result


def _select_rows(config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    cfg = config["cradio_probe"]
    captions = _caption_index(destination)
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_records(destination / "final_manifest.parquet"):
        if row.get("split", "train") != cfg.get("split", "train"):
            continue
        if int(row["id"]) in captions:
            by_style[row.get("style_id", row["artist"])].append(row)

    images_per_style = int(cfg["images_per_style"])
    eligible = sorted(key for key, rows in by_style.items() if len(rows) >= images_per_style)
    style_count = int(cfg["style_count"])
    if len(eligible) < style_count:
        raise RuntimeError(f"Need {style_count} eligible styles, found {len(eligible)}")
    rng = random.Random(int(cfg["seed"]))
    selected_styles = rng.sample(eligible, style_count)
    selected: list[dict[str, Any]] = []
    for style_index, style_id in enumerate(selected_styles):
        rows = list(by_style[style_id])
        rng.shuffle(rows)
        for sample_rank, row in enumerate(rows[:images_per_style]):
            selected.append(
                {
                    **row,
                    "style_index": style_index,
                    "probe_sample_rank": sample_rank,
                    "content_caption": captions[int(row["id"])],
                }
            )
    return selected


def extract_probe_features(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    cfg = config["cradio_probe"]
    radio_cfg = {**config["cradio"], **cfg.get("preprocess", {})}
    rows = _select_rows(config, destination)
    model, device = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    adaptor = model.adaptors[radio_cfg["adaptor_name"]]
    layers = [int(value) for value in cfg["layers"]]
    if min(layers) < 0 or max(layers) >= len(model.model.blocks):
        raise ValueError(f"Probe layers must be between 0 and {len(model.model.blocks) - 1}")

    storage_dtype = _storage_dtype(cfg.get("storage_dtype", "float16"))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[radio_cfg["amp_dtype"]]
    batch_size = int(cfg.get("batch_size", radio_cfg["batch_size"]))
    max_open_buckets = int(cfg.get("max_open_buckets", 16))
    features: dict[str, torch.Tensor] = {}
    buckets: dict[tuple[int, int], list[tuple[int, dict[str, Any], torch.Tensor]]] = defaultdict(list)

    def put(name: str, indices: list[int], value: torch.Tensor) -> None:
        value = value.detach().to("cpu", dtype=storage_dtype)
        if name not in features:
            features[name] = torch.empty((len(rows), value.shape[-1]), dtype=storage_dtype)
        features[name][indices] = value

    def run_batch(items: list[tuple[int, dict[str, Any], torch.Tensor]]) -> None:
        indices = [item[0] for item in items]
        images = torch.stack([item[2] for item in items]).to(device, non_blocking=True)
        captions = [item[1]["content_caption"] for item in items]
        tokenized = adaptor.tokenizer(captions).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")
        ):
            final, intermediate = model.forward_intermediates(
                images,
                indices=layers,
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NLC",
                aggregation="sparse",
            )
            text_summary = adaptor.encode_text(tokenized, normalize=True)

        for layer, output in zip(layers, intermediate):
            spatial = output.features.float()
            put(f"layer_{layer:02d}_summary", indices, output.summary.float())
            put(f"layer_{layer:02d}_spatial_mean", indices, spatial.mean(dim=1))
            put(
                f"layer_{layer:02d}_spatial_stats",
                indices,
                torch.cat((spatial.mean(dim=1), spatial.std(dim=1)), dim=-1),
            )

        backbone_summary, backbone_spatial = _summary_features(final["backbone"])
        siglip_summary, _ = _summary_features(final[radio_cfg["adaptor_name"]])
        spatial = backbone_spatial.float()
        put("final_backbone_summary", indices, backbone_summary.float())
        put(
            "final_backbone_spatial_stats",
            indices,
            torch.cat((spatial.mean(dim=1), spatial.std(dim=1)), dim=-1),
        )
        visual = F.normalize(siglip_summary.float(), dim=-1)
        text = F.normalize(text_summary.float(), dim=-1)
        put("siglip_visual", indices, visual)
        put("siglip_text", indices, text)
        for scale in cfg.get("content_scales", [1.0]):
            _, _, residual = content_subtracted_summary(visual, text, scale=float(scale))
            put(f"siglip_residual_s{float(scale):g}", indices, residual)

    def flush(key: tuple[int, int]) -> None:
        items = buckets.pop(key)
        for offset in range(0, len(items), batch_size):
            run_batch(items[offset : offset + batch_size])

    for index, row in enumerate(rows):
        with Image.open(row["local_path"]) as image:
            array, info = preprocess_cradio_image(image, radio_cfg)
        key = (info.target_height, info.target_width)
        buckets[key].append((index, row, torch.from_numpy(array)))
        if len(buckets[key]) >= batch_size:
            flush(key)
        elif len(buckets) > max_open_buckets:
            flush(max(buckets, key=lambda bucket: len(buckets[bucket])))
        if (index + 1) % 100 == 0:
            print(f"prepared probe images {index + 1}/{len(rows)}", flush=True)
    for key in list(buckets):
        flush(key)

    probe_dir = destination / "cradio_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    feature_path = probe_dir / "features.safetensors"
    save_file({name: value.contiguous() for name, value in features.items()}, feature_path)
    manifest_rows = [
        {
            "id": int(row["id"]),
            "artist": row["artist"],
            "style_id": row.get("style_id", row["artist"]),
            "style_index": int(row["style_index"]),
            "probe_sample_rank": int(row["probe_sample_rank"]),
            "local_path": row["local_path"],
        }
        for row in rows
    ]
    write_records(probe_dir / "manifest.parquet", manifest_rows)
    summary = {
        "styles": len({row["style_id"] for row in rows}),
        "images": len(rows),
        "layers": layers,
        "representations": {name: list(value.shape) for name, value in features.items()},
        "feature_path": str(feature_path.resolve()),
    }
    write_json(probe_dir / "extract_summary.json", summary)
    return summary


def prototype_metrics(
    values, style_indices, sample_ranks, reference_counts: list[int], max_references: int
) -> list[dict[str, float | int]]:
    import torch
    import torch.nn.functional as F

    values = F.normalize(values.float(), dim=-1)
    style_indices = style_indices.long()
    sample_ranks = sample_ranks.long()
    styles = torch.unique(style_indices, sorted=True)
    query_mask = sample_ranks >= int(max_references)
    query_values = values[query_mask]
    query_labels = style_indices[query_mask]
    results = []
    for references in reference_counts:
        prototypes = []
        for style in styles:
            mask = (style_indices == style) & (sample_ranks < int(references))
            prototypes.append(F.normalize(values[mask].mean(dim=0), dim=0))
        prototypes = torch.stack(prototypes)
        scores = query_values @ prototypes.T
        positive = scores.gather(1, query_labels[:, None]).squeeze(1)
        predicted = scores.argmax(dim=1)
        ranks = 1 + (scores > positive[:, None]).sum(dim=1)
        negative = scores.clone()
        negative[torch.arange(len(query_labels)), query_labels] = -float("inf")
        results.append(
            {
                "references": int(references),
                "queries": int(len(query_labels)),
                "top1": float((predicted == query_labels).float().mean()),
                "mrr": float((1.0 / ranks.float()).mean()),
                "positive_similarity": float(positive.mean()),
                "hardest_negative_similarity": float(negative.max(dim=1).values.mean()),
                "margin": float((positive - negative.max(dim=1).values).mean()),
            }
        )
    return results


def evaluate_probe_features(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    cfg = config["cradio_probe"]
    probe_dir = destination / "cradio_probe"
    rows = read_records(probe_dir / "manifest.parquet")
    tensors = load_file(probe_dir / "features.safetensors")
    style_indices = torch.tensor([int(row["style_index"]) for row in rows])
    sample_ranks = torch.tensor([int(row["probe_sample_rank"]) for row in rows])
    reference_counts = [int(value) for value in cfg["reference_counts"]]
    max_references = max(reference_counts)
    results = []
    for name, values in tensors.items():
        metrics = prototype_metrics(
            values, style_indices, sample_ranks, reference_counts, max_references
        )
        results.append(
            {
                "representation": name,
                "dimension": int(values.shape[-1]),
                "metrics": metrics,
                "mean_top1": sum(item["top1"] for item in metrics) / len(metrics),
                "mean_mrr": sum(item["mrr"] for item in metrics) / len(metrics),
            }
        )
    results.sort(key=lambda row: (row["mean_top1"], row["mean_mrr"]), reverse=True)
    summary = {
        "styles": len(set(int(row["style_index"]) for row in rows)),
        "images": len(rows),
        "reference_counts": reference_counts,
        "ranking": results,
    }
    write_json(probe_dir / "evaluation.json", summary)
    return summary


def run_feature_probe(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    extract_probe_features(config, destination)
    return evaluate_probe_features(config, destination)
