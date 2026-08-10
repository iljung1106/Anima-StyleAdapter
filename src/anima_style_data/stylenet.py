from __future__ import annotations

import hashlib
import io
import re
import tarfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .cradio import (
    _load_cradio,
    _storage_dtype,
    compute_cradio_size,
    preprocess_cradio_image,
)
from .io import read_records, write_json, write_records


_MEMBER_RE = re.compile(
    r"^images/Group_(?P<group>\d+)_(?P<subject>.*)/"
    r"(?P<rank>\d+)_(?P<kind>Original|DiffArtist)_(?P<artist>.*)"
    r"\.(?P<extension>jpe?g|png|webp)$",
    re.IGNORECASE,
)


def parse_stylenet_member(name: str) -> dict[str, Any] | None:
    match = _MEMBER_RE.match(name.replace("\\", "/"))
    if match is None:
        return None
    values = match.groupdict()
    return {
        "group_index": int(values["group"]),
        "subject": values["subject"],
        "candidate_rank": int(values["rank"]),
        "candidate_artist": values["artist"],
        "is_original": values["kind"].lower() == "original",
        "extension": values["extension"].lower(),
    }


def _benchmark_dir(config: dict[str, Any], destination: Path) -> Path:
    name = config["stylenet_benchmark"].get("directory", "stylenet_benchmark")
    return destination / name


def prepare_stylenet(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    cfg = config["stylenet_benchmark"]
    root = _benchmark_dir(config, destination)
    source_dir = root / "source"
    image_dir = root / "images"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="dataset",
        revision=cfg.get("revision"),
        allow_patterns=["dataset_*.tar"],
        local_dir=source_dir,
        max_workers=int(cfg.get("download_workers", 16)),
    )

    min_std = float(cfg.get("min_image_std", 2.0))
    min_side = int(cfg.get("min_side", 64))
    rows: list[dict[str, Any]] = []
    invalid_images = 0
    for tar_path in sorted(source_dir.glob("dataset_*.tar")):
        shard = tar_path.stem.removeprefix("dataset_")
        with tarfile.open(tar_path, "r") as archive:
            for member in archive:
                parsed = parse_stylenet_member(member.name)
                if parsed is None or not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load()
                        rgb = image.convert("RGB")
                        width, height = rgb.size
                        image_std = ImageStat.Stat(rgb.convert("L")).stddev[0]
                except (OSError, ValueError):
                    invalid_images += 1
                    continue
                if min(width, height) < min_side or image_std < min_std:
                    invalid_images += 1
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                group_key = f"{shard}:{parsed['group_index']:05d}"
                output = image_dir / shard / f"{parsed['group_index']:05d}"
                output.mkdir(parents=True, exist_ok=True)
                local_path = output / (
                    f"{parsed['candidate_rank']:02d}_{digest[:16]}.{parsed['extension']}"
                )
                if not local_path.exists():
                    local_path.write_bytes(payload)
                rows.append(
                    {
                        "id": len(rows),
                        "shard_artist": shard,
                        "group_key": group_key,
                        "group_index": parsed["group_index"],
                        "subject": parsed["subject"],
                        "candidate_rank": parsed["candidate_rank"],
                        "candidate_artist": parsed["candidate_artist"],
                        "is_original": parsed["is_original"],
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "image_luma_std": image_std,
                        "local_path": str(local_path.resolve()),
                        "source_tar": str(tar_path.resolve()),
                        "source_member": member.name,
                    }
                )
        print(f"indexed StyleNet shard {tar_path.name}: {len(rows)} valid images", flush=True)

    hash_counts = Counter(row["sha256"] for row in rows)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["global_exact_duplicate"] = hash_counts[row["sha256"]] > 1
        by_group[row["group_key"]].append(row)
    valid_groups = {
        key
        for key, group in by_group.items()
        if len(group) == 4
        and sum(bool(row["is_original"]) for row in group) == 1
        and len({row["sha256"] for row in group}) == 4
    }
    for row in rows:
        row["controlled_group_valid"] = row["group_key"] in valid_groups

    if not rows:
        raise RuntimeError("StyleNet preparation found no valid images")
    write_records(root / "manifest.parquet", rows)
    summary = {
        "repo_id": cfg["repo_id"],
        "revision": cfg.get("revision"),
        "tar_shards": len(list(source_dir.glob("dataset_*.tar"))),
        "valid_images": len(rows),
        "invalid_images": invalid_images,
        "controlled_groups": len(valid_groups),
        "shard_artists": len({row["shard_artist"] for row in rows}),
        "global_exact_duplicate_images": sum(
            bool(row["global_exact_duplicate"]) for row in rows
        ),
    }
    write_json(root / "prepare_summary.json", summary)
    return summary


def extract_stylenet_layer_features(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    cfg = config["stylenet_benchmark"]
    root = _benchmark_dir(config, destination)
    rows = [
        row for row in read_records(root / "manifest.parquet")
        if row["controlled_group_valid"]
    ]
    radio_cfg = {**config["cradio"], **cfg.get("preprocess", {})}
    rows.sort(
        key=lambda row: (
            compute_cradio_size(
                int(row["height"]),
                int(row["width"]),
                max_side=int(radio_cfg["max_side"]),
                max_pixels=int(radio_cfg["max_pixels"]),
                step=int(radio_cfg["patch_size"]),
                min_side=int(radio_cfg["min_side"]),
            )[2:],
            int(row["id"]),
        )
    )
    requested_layers = [int(layer) for layer in cfg["layers"]]
    keep_spatial_stats = bool(cfg.get("extract_spatial_stats", True))
    required_kinds = ["summary", "spatial_mean"]
    if keep_spatial_stats:
        required_kinds.append("spatial_stats")
    feature_dir = root / "pooled_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "features.safetensors"
    manifest_path = feature_dir / "manifest.parquet"
    features: dict[str, torch.Tensor] = (
        dict(load_file(feature_path)) if feature_path.exists() else {}
    )
    if manifest_path.exists():
        existing_rows = read_records(manifest_path)
        expected_ids = [int(row["id"]) for row in rows]
        existing_ids = [int(row["id"]) for row in existing_rows]
        if existing_ids != expected_ids:
            raise RuntimeError("Existing StyleNet feature order does not match the manifest")
    layers = [
        layer
        for layer in requested_layers
        if any(f"layer_{layer:02d}_{kind}" not in features for kind in required_kinds)
    ]
    if not layers:
        summary = {
            "images": len(rows),
            "layers": requested_layers,
            "added_layers": [],
            "representations": {
                name: list(value.shape) for name, value in features.items()
            },
        }
        write_json(feature_dir / "extract_summary.json", summary)
        return summary

    model, device = _load_cradio(radio_cfg, destination / "cradio_model_cache")
    if min(layers) < 0 or max(layers) >= len(model.model.blocks):
        raise ValueError(f"StyleNet layers must be in [0, {len(model.model.blocks) - 1}]")
    batch_size = int(cfg.get("batch_size", 32))
    max_open_buckets = int(cfg.get("max_open_buckets", 32))
    decoder_workers = int(cfg.get("decoder_workers", 16))
    decode_prefetch = int(cfg.get("decode_prefetch", 512))
    storage_dtype = _storage_dtype(cfg.get("storage_dtype", "float16"))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp_dtype", radio_cfg["amp_dtype"])
    ]
    buckets: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    started = time.monotonic()
    encoded = 0
    batches = 0

    def put(name: str, indices: list[int], value: torch.Tensor) -> None:
        value = value.detach().to("cpu", dtype=storage_dtype)
        if name not in features:
            features[name] = torch.empty((len(rows), value.shape[-1]), dtype=storage_dtype)
        features[name][indices] = value

    def run_batch(items: list[tuple[int, torch.Tensor]]) -> None:
        nonlocal encoded, batches
        indices = [item[0] for item in items]
        tensors = [item[1] for item in items]
        if device.startswith("cuda"):
            host_images = torch.empty(
                (len(tensors), *tensors[0].shape),
                dtype=tensors[0].dtype,
                pin_memory=True,
            )
            torch.stack(tensors, out=host_images)
        else:
            host_images = torch.stack(tensors)
        images = host_images.to(device, non_blocking=device.startswith("cuda"))
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")
        ):
            intermediate = model.forward_intermediates(
                images,
                indices=layers,
                return_prefix_tokens=True,
                norm=True,
                stop_early=True,
                output_fmt="NLC",
                intermediates_only=True,
                aggregation="sparse",
            )
        for layer, output in zip(layers, intermediate):
            spatial = output.features.float()
            put(f"layer_{layer:02d}_summary", indices, output.summary.float())
            put(f"layer_{layer:02d}_spatial_mean", indices, spatial.mean(dim=1))
            if keep_spatial_stats:
                put(
                    f"layer_{layer:02d}_spatial_stats",
                    indices,
                    torch.cat((spatial.mean(dim=1), spatial.std(dim=1)), dim=-1),
                )
        encoded += len(items)
        batches += 1
        if encoded // 1000 != (encoded - len(items)) // 1000:
            elapsed = time.monotonic() - started
            print(
                f"encoded StyleNet features {encoded}/{len(rows)} "
                f"({encoded / elapsed:.2f} images/s, "
                f"mean_batch={encoded / batches:.1f})",
                flush=True,
            )

    def flush(key: tuple[int, int]) -> None:
        items = buckets.pop(key)
        for offset in range(0, len(items), batch_size):
            run_batch(items[offset : offset + batch_size])

    def decode(index: int, row: dict[str, Any]):
        with Image.open(row["local_path"]) as image:
            image.load()
            array, info = preprocess_cradio_image(image, radio_cfg)
        return index, torch.from_numpy(array), info

    def decoded_rows():
        pending: deque[Future] = deque()
        iterator = iter(enumerate(rows))
        with ThreadPoolExecutor(max_workers=decoder_workers) as pool:
            for _ in range(min(decode_prefetch, len(rows))):
                index, row = next(iterator)
                pending.append(pool.submit(decode, index, row))
            while pending:
                result = pending.popleft().result()
                try:
                    index, row = next(iterator)
                except StopIteration:
                    pass
                else:
                    pending.append(pool.submit(decode, index, row))
                yield result

    for index, tensor, info in decoded_rows():
        key = (info.target_height, info.target_width)
        buckets[key].append((index, tensor))
        if len(buckets[key]) >= batch_size:
            flush(key)
        elif len(buckets) > max_open_buckets:
            flush(max(buckets, key=lambda item: len(buckets[item])))
    for key in list(buckets):
        flush(key)

    temporary = feature_dir / "features.next.safetensors"
    save_file(
        {name: value.contiguous() for name, value in features.items()},
        temporary,
    )
    temporary.replace(feature_path)
    feature_rows = [{**row, "feature_index": index} for index, row in enumerate(rows)]
    write_records(manifest_path, feature_rows)
    summary = {
        "images": len(rows),
        "layers": requested_layers,
        "added_layers": layers,
        "elapsed_s": time.monotonic() - started,
        "decoder_workers": decoder_workers,
        "decode_prefetch": decode_prefetch,
        "batches": batches,
        "mean_batch_size": len(rows) / batches,
        "representations": {name: list(value.shape) for name, value in features.items()},
    }
    write_json(feature_dir / "extract_summary.json", summary)
    return summary


def controlled_style_ranking(
    values,
    rows: list[dict[str, Any]],
    reference_count: int,
    *,
    normalized: bool = False,
    return_correct: bool = False,
):
    import torch
    import torch.nn.functional as F

    values = values.float() if normalized else F.normalize(values.float(), dim=-1)
    originals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)
        if row["is_original"] and not row["global_exact_duplicate"]:
            originals[row["shard_artist"]].append(row)
    for artist_rows in originals.values():
        artist_rows.sort(key=lambda row: (int(row["group_index"]), int(row["id"])))

    reference_indices = []
    candidate_indices = []
    positive_indices = []
    query_artist_indices = []
    for artist, artist_rows in originals.items():
        if len(artist_rows) <= reference_count:
            continue
        references = artist_rows[:reference_count]
        artist_index = len(reference_indices)
        reference_indices.append(
            [int(row["feature_index"]) for row in references]
        )
        reference_groups = {row["group_key"] for row in references}
        for positive in artist_rows[reference_count:]:
            if positive["group_key"] in reference_groups:
                continue
            candidates = groups[positive["group_key"]]
            if len(candidates) != 4:
                continue
            candidate_indices.append(
                [int(row["feature_index"]) for row in candidates]
            )
            positive_indices.append(
                next(index for index, row in enumerate(candidates) if row["is_original"])
            )
            query_artist_indices.append(artist_index)
    if not candidate_indices:
        raise RuntimeError(f"No controlled StyleNet queries for {reference_count} references")

    device = values.device
    reference_index = torch.tensor(reference_indices, dtype=torch.long, device=device)
    prototypes = F.normalize(values[reference_index].mean(dim=1), dim=-1)
    correct = 0
    reciprocal_rank_sum = 0.0
    margin_sum = 0.0
    correct_chunks = []
    chunk_size = 4096
    for offset in range(0, len(candidate_indices), chunk_size):
        stop = min(offset + chunk_size, len(candidate_indices))
        candidate_index = torch.tensor(
            candidate_indices[offset:stop], dtype=torch.long, device=device
        )
        positive_index = torch.tensor(
            positive_indices[offset:stop], dtype=torch.long, device=device
        )
        query_artist_index = torch.tensor(
            query_artist_indices[offset:stop], dtype=torch.long, device=device
        )
        scores = (
            values[candidate_index] * prototypes[query_artist_index, None, :]
        ).sum(dim=-1)
        row_index = torch.arange(scores.shape[0], device=device)
        positive_scores = scores[row_index, positive_index]
        ranks = 1 + (scores > positive_scores[:, None]).sum(dim=1)
        scores[row_index, positive_index] = -torch.inf
        margins = positive_scores - scores.max(dim=1).values
        chunk_correct = ranks == 1
        correct += int(chunk_correct.sum())
        if return_correct:
            correct_chunks.append(chunk_correct.to("cpu"))
        reciprocal_rank_sum += float((1.0 / ranks.float()).sum())
        margin_sum += float(margins.sum())
    query_count = len(candidate_indices)
    metrics = {
        "references": int(reference_count),
        "queries": query_count,
        "top1": correct / query_count,
        "mrr": reciprocal_rank_sum / query_count,
        "margin": margin_sum / query_count,
    }
    if return_correct:
        return metrics, torch.cat(correct_chunks)
    return metrics


def evaluate_stylenet_layer_features(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    cfg = config["stylenet_benchmark"]
    root = _benchmark_dir(config, destination) / "pooled_features"
    rows = read_records(root / "manifest.parquet")
    tensors = load_file(root / "features.safetensors")
    representations = dict(tensors)
    for name, values in tensors.items():
        if not name.endswith("_summary"):
            continue
        layer_name = name.removesuffix("_summary")
        spatial_dim = int(tensors[f"{layer_name}_spatial_mean"].shape[-1])
        if int(values.shape[-1]) != 2 * spatial_dim:
            raise ValueError(
                f"Expected two teacher CLS slots for {name}, got {values.shape[-1]}"
            )
        representations[f"{layer_name}_siglip_cls"] = values[:, :spatial_dim]
        representations[f"{layer_name}_dino_cls"] = values[:, spatial_dim:]

    def normalized_concat(names: list[str]):
        return torch.cat(
            [F.normalize(representations[name].float(), dim=-1) for name in names],
            dim=-1,
        )

    pairwise_layers = [int(layer) for layer in cfg.get("pairwise_layers", [])]
    pair_names = []
    for first_index, first in enumerate(pairwise_layers):
        for second in pairwise_layers[first_index + 1 :]:
            spatial_name = f"pair_l{first:02d}_l{second:02d}_spatial_mean"
            summary_name = f"pair_l{first:02d}_l{second:02d}_summary"
            representations[spatial_name] = normalized_concat(
                [f"layer_{first:02d}_spatial_mean", f"layer_{second:02d}_spatial_mean"]
            )
            representations[summary_name] = normalized_concat(
                [f"layer_{first:02d}_summary", f"layer_{second:02d}_summary"]
            )
            pair_names.append((first, second, spatial_name, summary_name))

    for combo in cfg.get("combinations", []):
        names = [str(name) for name in combo["representations"]]
        representations[str(combo["name"])] = normalized_concat(names)

    reference_counts = [int(value) for value in cfg["reference_counts"]]
    device = str(cfg.get("evaluation_device", config["cradio"]["device"]))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for StyleNet evaluation but is unavailable")
    ranking = []
    decision_names = {
        f"layer_{layer:02d}_{kind}"
        for layer in pairwise_layers
        for kind in ("spatial_mean", "summary")
    }
    decisions = {}
    for name, values in representations.items():
        values = F.normalize(values.to(device=device, dtype=torch.float32), dim=-1)
        metrics = []
        for count in reference_counts:
            result = controlled_style_ranking(
                values,
                rows,
                count,
                normalized=True,
                return_correct=name in decision_names,
            )
            if name in decision_names:
                metric, correct = result
                decisions[(name, count)] = correct
            else:
                metric = result
            metrics.append(metric)
        ranking.append(
            {
                "representation": name,
                "dimension": int(values.shape[-1]),
                "metrics": metrics,
                "mean_top1": sum(item["top1"] for item in metrics) / len(metrics),
                "mean_mrr": sum(item["mrr"] for item in metrics) / len(metrics),
                "mean_margin": sum(item["margin"] for item in metrics) / len(metrics),
            }
        )
    ranking.sort(key=lambda row: (row["mean_top1"], row["mean_mrr"]), reverse=True)
    ranking_by_name = {row["representation"]: row for row in ranking}

    centered_spatial = {}
    spatial_self_hsic = {}
    for layer in pairwise_layers:
        values = tensors[f"layer_{layer:02d}_spatial_mean"].to(
            device=device, dtype=torch.float32
        )
        values = values - values.mean(dim=0, keepdim=True)
        gram = values.T @ values
        centered_spatial[layer] = values
        spatial_self_hsic[layer] = gram.square().sum()

    pair_analysis = []
    for first, second, spatial_name, summary_name in pair_names:
        cross = centered_spatial[first].T @ centered_spatial[second]
        cka = cross.square().sum() / torch.sqrt(
            spatial_self_hsic[first] * spatial_self_hsic[second]
        )
        oracle_values = []
        error_jaccard_values = []
        disagreement_values = []
        for count in reference_counts:
            first_correct = decisions[(f"layer_{first:02d}_spatial_mean", count)]
            second_correct = decisions[(f"layer_{second:02d}_spatial_mean", count)]
            first_error = ~first_correct
            second_error = ~second_correct
            oracle_values.append(float((first_correct | second_correct).float().mean()))
            error_union = first_error | second_error
            error_jaccard_values.append(
                float((first_error & second_error).sum() / error_union.sum())
            )
            disagreement_values.append(float((first_correct != second_correct).float().mean()))
        spatial_pair = ranking_by_name[spatial_name]
        summary_pair = ranking_by_name[summary_name]
        best_spatial_single = max(
            ranking_by_name[f"layer_{first:02d}_spatial_mean"]["mean_top1"],
            ranking_by_name[f"layer_{second:02d}_spatial_mean"]["mean_top1"],
        )
        best_summary_single = max(
            ranking_by_name[f"layer_{first:02d}_summary"]["mean_top1"],
            ranking_by_name[f"layer_{second:02d}_summary"]["mean_top1"],
        )
        pair_analysis.append(
            {
                "layers": [first, second],
                "spatial_mean_cka": float(cka),
                "spatial_pair_mean_top1": spatial_pair["mean_top1"],
                "spatial_gain_over_best_single": spatial_pair["mean_top1"]
                - best_spatial_single,
                "summary_pair_mean_top1": summary_pair["mean_top1"],
                "summary_gain_over_best_single": summary_pair["mean_top1"]
                - best_summary_single,
                "spatial_oracle_mean_top1": sum(oracle_values) / len(oracle_values),
                "spatial_error_jaccard": sum(error_jaccard_values)
                / len(error_jaccard_values),
                "spatial_correctness_disagreement": sum(disagreement_values)
                / len(disagreement_values),
            }
        )
    pair_analysis.sort(
        key=lambda row: row["spatial_pair_mean_top1"], reverse=True
    )
    summary = {
        "protocol": "Within each character-controlled group, rank the target artist original against three different-artist images using references from other groups.",
        "images": len(rows),
        "groups": len({row["group_key"] for row in rows}),
        "reference_counts": reference_counts,
        "ranking": ranking,
        "pair_analysis": pair_analysis,
    }
    write_json(root / "evaluation.json", summary)
    return summary


def run_stylenet_layer_benchmark(config: dict[str, Any], destination: Path):
    prepare_stylenet(config, destination)
    extract_stylenet_layer_features(config, destination)
    return evaluate_stylenet_layer_features(config, destination)
