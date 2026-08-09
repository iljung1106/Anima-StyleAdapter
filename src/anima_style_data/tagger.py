from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageOps

from .io import read_records, write_json, write_records


KAOMOJIS = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
    "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u",
    "x_x", "|_|", "||_||",
}


def prepare_tagger_image(image: Image.Image, input_size: int = 448) -> np.ndarray:
    """Match the canary Space preprocessing exactly, with EXIF correction added."""
    image = ImageOps.exif_transpose(image).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    rgb = canvas.convert("RGB")
    max_dim = max(rgb.size)
    left = (max_dim - rgb.width) // 2
    top = (max_dim - rgb.height) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(rgb, (left, top))
    if max_dim != input_size:
        padded = padded.resize((input_size, input_size), Image.Resampling.BICUBIC)
    array = np.asarray(padded, dtype=np.float32)
    array = array[:, :, ::-1] / 127.5 - 1.0  # RGB -> BGR, then [-1, 1]
    return np.ascontiguousarray(array.transpose(2, 0, 1))


@dataclass(frozen=True)
class Labels:
    names: list[str]
    ratings: np.ndarray
    general: np.ndarray
    characters: np.ndarray


def load_labels(path: str | Path) -> Labels:
    names: list[str] = []
    categories: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["name"]
            names.append(name if name in KAOMOJIS else name.replace("_", " "))
            categories.append(int(row["category"]))
    category_array = np.asarray(categories)
    return Labels(
        names=names,
        ratings=np.flatnonzero(category_array == 9),
        general=np.flatnonzero(category_array == 0),
        characters=np.flatnonzero(category_array == 4),
    )


class _ImageDataset:
    def __init__(self, rows: list[dict[str, Any]], input_size: int):
        self.rows = rows
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        with Image.open(row["local_path"]) as image:
            array = prepare_tagger_image(image, self.input_size)
        return int(row["id"]), torch.from_numpy(array)


def _sparse_scores(
    probabilities: np.ndarray, labels: Labels, threshold: float, top_k: int
) -> tuple[list[str], list[float]]:
    indices = np.flatnonzero(probabilities >= threshold)
    if len(indices) > top_k:
        local = np.argpartition(probabilities[indices], -top_k)[-top_k:]
        indices = indices[local]
    indices = indices[np.argsort(probabilities[indices])[::-1]]
    return [labels.names[index] for index in indices], [float(probabilities[index]) for index in indices]


def _selected_scores(
    probabilities: np.ndarray, indices: np.ndarray, labels: Labels, threshold: float
) -> tuple[list[str], list[float]]:
    selected = indices[probabilities[indices] >= threshold]
    selected = selected[np.argsort(probabilities[selected])[::-1]]
    return [labels.names[index] for index in selected], [float(probabilities[index]) for index in selected]


def _load_runtime(cfg: dict[str, Any], cache_dir: Path):
    import timm
    import torch
    from safetensors.torch import load_file

    repo_id = cfg["repo_id"]
    revision = cfg.get("revision") or "main"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id, "model.safetensors", revision=revision, cache_dir=cache_dir
    )
    labels_path = hf_hub_download(
        repo_id, "selected_tags.csv", revision=revision, cache_dir=cache_dir
    )
    labels = load_labels(labels_path)
    model = timm.create_model(
        cfg["architecture"], pretrained=False, num_classes=len(labels.names)
    )
    model.load_state_dict(load_file(model_path), strict=True)
    device = str(cfg["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for tagging but is not available")
    model.eval().to(device)
    resolved_revision = HfApi().model_info(repo_id, revision=revision).sha
    return model, labels, device, resolved_revision


def _existing_tag_ids(tags_dir: Path) -> tuple[set[int], int]:
    completed: set[int] = set()
    shard_numbers = []
    for path in sorted(tags_dir.glob("part-*.parquet")):
        shard_numbers.append(int(path.stem.split("-")[-1]))
        completed.update(int(row["id"]) for row in read_records(path))
    return completed, (max(shard_numbers) + 1 if shard_numbers else 0)


def tag_images(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    cfg = config["tagger"]
    rows = read_records(destination / "final_manifest.parquet")
    tags_dir = destination / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)
    completed, shard_index = _existing_tag_ids(tags_dir)
    pending = [row for row in rows if int(row["id"]) not in completed]
    if not pending:
        summary = {
            "total": len(rows),
            "tagged": len(completed),
            "newly_tagged": 0,
            "tagger_revision": cfg.get("revision") or "main",
            "threshold": float(cfg["threshold"]),
        }
        write_json(destination / "tagging_summary.json", summary)
        return summary

    model, labels, device, resolved_revision = _load_runtime(
        cfg, destination / "model_cache"
    )
    dataset = _ImageDataset(pending, int(cfg["input_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.startswith("cuda"),
    )
    row_by_id = {int(row["id"]): row for row in pending}
    threshold = float(cfg["threshold"])
    store_threshold = float(cfg["score_store_threshold"])
    top_k = int(cfg["sparse_top_k"])
    exclude = {str(tag).replace("_", " ") for tag in cfg.get("content_exclude_tags", [])}
    shard_rows = int(cfg["shard_rows"])
    buffer: list[dict[str, Any]] = []
    newly_tagged = 0

    for ids, images in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")
            ):
                logits = model(images)
            probabilities = torch.sigmoid(logits).float().cpu().numpy()

        for image_id, probs in zip(ids.tolist(), probabilities):
            source = row_by_id[int(image_id)]
            rating_local = int(np.argmax(probs[labels.ratings]))
            rating_index = int(labels.ratings[rating_local])
            general_tags, general_scores = _selected_scores(
                probs, labels.general, labels, threshold
            )
            character_tags, character_scores = _selected_scores(
                probs, labels.characters, labels, threshold
            )
            sparse_tags, sparse_scores = _sparse_scores(
                probs, labels, store_threshold, top_k
            )
            content_tags = [tag for tag in general_tags if tag not in exclude]
            buffer.append(
                {
                    "id": int(image_id),
                    "artist": source["artist"],
                    "style_id": source.get("style_id", source["artist"]),
                    "split": source.get("split", "train"),
                    "local_path": source["local_path"],
                    "rating": labels.names[rating_index],
                    "rating_score": float(probs[rating_index]),
                    "general_tags": general_tags,
                    "general_scores": general_scores,
                    "character_tags": character_tags,
                    "character_scores": character_scores,
                    "sparse_score_tags": sparse_tags,
                    "sparse_score_values": sparse_scores,
                    "content_caption": ", ".join(content_tags),
                    "tagger_repo": cfg["repo_id"],
                    "tagger_revision": resolved_revision,
                    "tagger_threshold": threshold,
                    "score_store_threshold": store_threshold,
                }
            )
            newly_tagged += 1

        if len(buffer) >= shard_rows:
            write_records(tags_dir / f"part-{shard_index:05d}.parquet", buffer)
            print(f"wrote tag shard {shard_index} ({len(buffer)} rows)", flush=True)
            shard_index += 1
            buffer = []

    if buffer:
        write_records(tags_dir / f"part-{shard_index:05d}.parquet", buffer)
        print(f"wrote tag shard {shard_index} ({len(buffer)} rows)", flush=True)

    summary = {
        "total": len(rows),
        "previously_tagged": len(completed),
        "newly_tagged": newly_tagged,
        "tagger_revision": resolved_revision,
        "threshold": threshold,
    }
    write_json(destination / "tagging_summary.json", summary)
    return summary
