from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .dual_query_training import (
    CacheEpisode,
    _intersect_cache_rows,
    _load_cache_episode,
    _model_from_config,
)
from .io import read_records, write_json, write_records
from .query_style_tokenizer import QueryStyleTokenizerOutput
from .style_transfer import ProductionStyleLoader, StyleEpisode, _pad_text_conditions
from .synthetic_teacher import synthetic_artist_split_map


_RESIDENT_TOKEN_BANKS: dict[str, dict[str, torch.Tensor]] = {}
_RESIDENT_TOKEN_BANKS_LOCK = threading.Lock()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DualQuerySetStyleTokenizer(nn.Module):
    """Aggregate cached per-reference tokens into native Anima context tokens."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        query_tokens: int = 80,
        artist_summary_tokens: int = 4,
        include_artist_summary: bool = True,
        output_tokens: int = 32,
        heads: int = 16,
        cross_layers: int = 1,
        cross_slot_layers: int = 2,
        ff_dim: int = 4096,
        output_rms_init: float = 0.15,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Style width must be divisible by attention heads")
        if min(query_tokens, output_tokens, cross_layers, cross_slot_layers) <= 0:
            raise ValueError("Token counts and transformer depths must be positive")
        if artist_summary_tokens < 0:
            raise ValueError("artist_summary_tokens cannot be negative")
        if output_rms_init <= 0:
            raise ValueError("output_rms_init must be positive")
        self.dim = int(dim)
        self.query_tokens = int(query_tokens)
        self.artist_summary_tokens = int(artist_summary_tokens)
        self.include_artist_summary = bool(include_artist_summary)
        self.output_tokens = int(output_tokens)
        self.input_norm = nn.LayerNorm(dim)
        self.output_queries = nn.Parameter(torch.empty(1, output_tokens, dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cross_layers,
            norm=nn.LayerNorm(dim),
        )
        slot_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_slot = nn.TransformerEncoder(
            slot_layer,
            num_layers=cross_slot_layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.log_output_rms = nn.Parameter(
            torch.tensor(math.log(float(output_rms_init)), dtype=torch.float32)
        )
        self.reset_parameters()

    @property
    def cached_tokens(self) -> int:
        return self.query_tokens + self.artist_summary_tokens

    def reset_parameters(self) -> None:
        nn.init.normal_(self.output_queries, std=self.dim**-0.5)

    def forward(
        self,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reconstruct: bool = False,
    ) -> QueryStyleTokenizerOutput:
        if reconstruct:
            raise ValueError(
                "Dual-query Style Tokenizer reconstruction belongs to the frozen Resampler"
            )
        if references.ndim != 4:
            raise ValueError("references must be [batch, references, tokens, dim]")
        if reference_mask.shape != references.shape[:2]:
            raise ValueError("reference mask does not match reference batch")
        if references.shape[-1] != self.dim:
            raise ValueError(f"Expected reference width {self.dim}")
        if references.shape[2] != self.cached_tokens:
            raise ValueError(
                f"Expected {self.cached_tokens} cached tokens, got {references.shape[2]}"
            )
        if not reference_mask.is_cuda and not bool(reference_mask.any(dim=1).all()):
            raise ValueError("Every sample needs at least one reference")

        tokens_per_reference = (
            self.cached_tokens if self.include_artist_summary else self.query_tokens
        )
        selected = references[:, :, :tokens_per_reference]
        batch, count, tokens, dim = selected.shape
        memory = self.input_norm(selected).reshape(batch, count * tokens, dim)
        memory_mask = reference_mask[:, :, None].expand(-1, -1, tokens).reshape(
            batch, count * tokens
        )
        queries = self.output_queries.expand(batch, -1, -1)
        values = self.set_decoder(
            queries,
            memory,
            memory_key_padding_mask=~memory_mask,
        )
        values = self.output_norm(self.cross_slot(values))
        values = values * self.log_output_rms.exp().to(values.dtype)
        return QueryStyleTokenizerOutput(
            tokens=values,
            per_reference_tokens=selected,
            reconstruction=None,
            reconstruction_target=None,
        )


def _load_resampler(
    config: dict[str, Any],
    destination: Path,
    checkpoint: Path,
    semantic_dim: int,
    vae_channels: int,
    device: str,
):
    resampler_cfg = config["dual_query_resampler"]
    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError as error:
        if "mmap" not in str(error):
            raise
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    effective_cfg = copy.deepcopy(resampler_cfg)
    checkpoint_model_cfg = state.get("model_config")
    if isinstance(checkpoint_model_cfg, dict):
        effective_cfg["model"] = copy.deepcopy(checkpoint_model_cfg)
    model_state = state["model"]
    semantic_layers = tuple(
        int(value)
        for value in effective_cfg["model"].get("semantic_layers", [18, 24])
    )
    checkpoint_semantic_dims = {
        int(model_state[f"semantic_norms.{layer}.weight"].numel())
        for layer in semantic_layers
    }
    if len(checkpoint_semantic_dims) != 1:
        raise RuntimeError(
            "Resampler checkpoint has inconsistent semantic layer widths: "
            f"{sorted(checkpoint_semantic_dims)}"
        )
    checkpoint_semantic_dim = checkpoint_semantic_dims.pop()
    checkpoint_vae_channels = int(model_state["vae_stem.0.weight"].shape[1])
    if (
        checkpoint_semantic_dim != int(semantic_dim)
        or checkpoint_vae_channels != int(vae_channels)
    ):
        print(
            "resampler checkpoint input dimensions override cache hints "
            f"semantic={semantic_dim}->{checkpoint_semantic_dim} "
            f"vae={vae_channels}->{checkpoint_vae_channels}",
            flush=True,
        )
    model = _model_from_config(
        effective_cfg,
        checkpoint_semantic_dim,
        checkpoint_vae_channels,
    )
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, int(state["step"])


def _write_token_shard(
    output: Path,
    index: int,
    tokens: torch.Tensor,
    descriptors: torch.Tensor,
    rows: list[dict[str, Any]],
    signature: str,
) -> tuple[list[dict[str, Any]], int]:
    name = f"part-{index:05d}.safetensors"
    path = output / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {
            "tokens": tokens.contiguous(),
            "descriptors": descriptors.contiguous(),
        },
        temporary,
    )
    temporary.replace(path)
    records = [
        {
            "id": int(row["id"]),
            "artist": str(row["artist"]),
            "style_id": str(row["style_id"]),
            "split": str(row["split"]),
            "token_shard": name,
            "token_row": offset,
            "slots": int(tokens.shape[1]),
            "style_dim": int(tokens.shape[2]),
            "query_slots": 80,
            "artist_summary_slots": 4,
            "cache_signature": signature,
        }
        for offset, row in enumerate(rows)
    ]
    write_records(output / f"part-{index:05d}.parquet", records)
    return records, path.stat().st_size


def cache_dual_query_style_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    cfg = config["dual_query_style_tokenizer"]
    cache_cfg = dict(cfg["cache"])
    device = str(cache_cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured token cache")
    resampler_cfg = config["dual_query_resampler"]
    feature_root, latent_root, rows = _intersect_cache_rows(destination, resampler_cfg)
    checkpoint = destination / str(cfg["resampler_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = _file_sha256(checkpoint)
    signature_payload = {
        "kind": "dual-query-reference-tokens-v1",
        "checkpoint_sha256": checkpoint_sha256,
        "query_tokens": 80,
        "artist_summary_tokens": 4,
        "dim": 1024,
        "dtype": "bfloat16",
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    output = destination / str(cache_cfg["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        if recorded.get("cache_signature") != signature:
            raise RuntimeError("Existing dual-query token cache has another signature")

    rows = sorted(
        rows,
        key=lambda row: (
            str(row["feature_shard"]),
            str(row["latent_shard"]),
            int(row["semantic_height"]),
            int(row["semantic_width"]),
            int(row["latent_height"]),
            int(row["latent_width"]),
            int(row["id"]),
        ),
    )
    semantic_layers = tuple(
        int(value) for value in resampler_cfg["model"].get("semantic_layers", [18, 24])
    )
    semantic_dim = int(rows[0]["spatial_dim"])
    first_latent = rows[0]
    from safetensors import safe_open

    with safe_open(
        latent_root / first_latent["latent_shard"], framework="pt", device="cpu"
    ) as handle:
        vae_channels = int(handle.get_slice("latents").get_shape()[1])
    model, checkpoint_step = _load_resampler(
        config, destination, checkpoint, semantic_dim, vae_channels, device
    )

    batch_size = int(cache_cfg.get("batch_size", 64))
    shard_rows = int(cache_cfg.get("shard_rows", 512))
    reader_workers = max(1, int(cache_cfg.get("reader_workers", 4)))
    prefetch_batches = max(1, int(cache_cfg.get("prefetch_batches", 8)))
    writer_workers = max(1, int(cache_cfg.get("writer_workers", 2)))
    pending_writes = max(1, int(cache_cfg.get("pending_writes", 2)))
    shard_groups = [rows[offset : offset + shard_rows] for offset in range(0, len(rows), shard_rows)]
    records: list[dict[str, Any]] = []
    written_bytes = 0
    started = time.perf_counter()
    writer = ThreadPoolExecutor(max_workers=writer_workers)
    write_futures: list[Future[tuple[list[dict[str, Any]], int]]] = []

    def consume_write(future: Future[tuple[list[dict[str, Any]], int]]) -> None:
        nonlocal written_bytes
        shard_records, size = future.result()
        records.extend(shard_records)
        written_bytes += size

    try:
        with ThreadPoolExecutor(max_workers=reader_workers) as reader:
            for shard_index, shard_items in enumerate(shard_groups):
                token_path = output / f"part-{shard_index:05d}.safetensors"
                row_path = output / f"part-{shard_index:05d}.parquet"
                if token_path.exists() and row_path.exists():
                    reused = read_records(row_path)
                    if any(row.get("cache_signature") != signature for row in reused):
                        raise RuntimeError(f"Token shard {shard_index} signature mismatch")
                    records.extend(reused)
                    written_bytes += token_path.stat().st_size
                    continue
                batches = [
                    shard_items[offset : offset + batch_size]
                    for offset in range(0, len(shard_items), batch_size)
                ]
                futures: dict[int, Future[CacheEpisode]] = {}
                next_batch = 0

                def fill(anchor: int) -> None:
                    nonlocal next_batch
                    while next_batch < len(batches) and next_batch < anchor + prefetch_batches:
                        episode_rows = [
                            {**row, "episode_label": 0} for row in batches[next_batch]
                        ]
                        futures[next_batch] = reader.submit(
                            _load_cache_episode,
                            episode_rows,
                            feature_root,
                            latent_root,
                            semantic_layers,
                            pin_memory=device.startswith("cuda"),
                        )
                        next_batch += 1

                fill(0)
                token_parts = []
                descriptor_parts = []
                for batch_index in range(len(batches)):
                    episode = futures.pop(batch_index).result().to(device)
                    fill(batch_index + 1)
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                    ):
                        encoded = model.encode(
                            episode.semantic_features,
                            episode.semantic_mask,
                            episode.semantic_grid_shapes,
                            episode.vae_latents,
                            episode.vae_shapes,
                            episode.image_sizes,
                            reconstruct=False,
                        )
                    combined = torch.cat((encoded.tokens, encoded.artist_summary), dim=1)
                    token_parts.append(combined.to("cpu", dtype=torch.bfloat16))
                    descriptor_parts.append(encoded.descriptor.to("cpu", dtype=torch.bfloat16))
                tokens = torch.cat(token_parts).contiguous()
                descriptors = torch.cat(descriptor_parts).contiguous()
                if tuple(tokens.shape[1:]) != (84, 1024):
                    raise RuntimeError(f"Unexpected dual-query cache shape {tuple(tokens.shape)}")
                write_futures.append(
                    writer.submit(
                        _write_token_shard,
                        output,
                        shard_index,
                        tokens,
                        descriptors,
                        shard_items,
                        signature,
                    )
                )
                if len(write_futures) >= pending_writes:
                    consume_write(write_futures.pop(0))
                elapsed = max(time.perf_counter() - started, 1e-6)
                processed = min((shard_index + 1) * shard_rows, len(rows))
                print(
                    f"dual-query token cache {processed}/{len(rows)} "
                    f"({processed / elapsed:.1f} img/s)",
                    flush=True,
                )
        for future in write_futures:
            consume_write(future)
    finally:
        writer.shutdown(wait=True)

    expected_ids = {int(row["id"]) for row in rows}
    if {int(row["id"]) for row in records} != expected_ids:
        raise RuntimeError("Dual-query token cache ID set mismatch")
    write_records(output / "manifest.parquet", sorted(records, key=lambda row: int(row["id"])))
    summary = {
        "images": len(records),
        "shards": len(shard_groups),
        "slots": 84,
        "query_slots": 80,
        "artist_summary_slots": 4,
        "style_dim": 1024,
        "descriptor_dim": 512,
        "dtype": "bfloat16",
        "resampler_checkpoint": str(cfg["resampler_checkpoint"]),
        "resampler_checkpoint_step": checkpoint_step,
        "resampler_checkpoint_sha256": checkpoint_sha256,
        "cache_signature": signature,
        "storage_bytes": written_bytes,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    return summary


def cache_synthetic_dual_query_style_tokens(
    config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Cache the current frozen Dual-query Resampler for synthetic references."""

    cfg = dict(config["synthetic_teacher"])
    root = destination / str(cfg["output_directory"])
    manifest_path = root / "manifest.parquet"
    feature_manifest = root / "style_features" / "manifest.parquet"
    if not manifest_path.exists() or not feature_manifest.exists():
        raise FileNotFoundError(
            "Run synthetic-teacher before caching synthetic Dual-query tokens"
        )
    latent_root = root / "latents"
    latent_manifest = latent_root / "manifest.parquet"
    source_rows = [
        row
        for row in read_records(manifest_path)
        if str(row.get("kind")) == "artist"
    ]
    artist_splits = synthetic_artist_split_map(config, source_rows)
    compatibility_rows = [
        {
            "id": int(row["id"]),
            "artist": str(row["artist"]),
            "style_id": str(row["style_id"]),
            "split": "synthetic_teacher",
            "teacher_split": (
                "test"
                if str(row.get("artist_split") or artist_splits[str(row["artist"])])
                == "meta_test"
                else str(
                    row.get("artist_split") or artist_splits[str(row["artist"])]
                )
            ),
            "cache_shard": str(row["latent_shard"]),
            "row_index": int(row["latent_row"]),
            "latent_height": int(row["latent_height"]),
            "latent_width": int(row["latent_width"]),
            "target_height": int(row["height"]),
            "target_width": int(row["width"]),
        }
        for row in source_rows
    ]
    write_records(latent_manifest, compatibility_rows)

    effective = copy.deepcopy(config)
    effective["dual_query_resampler"]["feature_directory"] = str(
        root / "style_features"
    )
    effective["dual_query_resampler"]["latent_directory"] = str(latent_root)
    tokenizer_cfg = effective["dual_query_style_tokenizer"]
    tokenizer_cfg["cache"] = {
        **dict(tokenizer_cfg["cache"]),
        "output_directory": str(
            Path(cfg["output_directory"])
            / str(cfg["dual_query_token_cache"]["output_directory"])
        ),
        **{
            key: value
            for key, value in dict(cfg["dual_query_token_cache"]).items()
            if key != "output_directory"
        },
    }
    return cache_dual_query_style_tokens(effective, destination)


class _FullTokenShardLRU:
    def __init__(self, root: Path, capacity: int) -> None:
        self.root = root
        self.capacity = max(1, int(capacity))
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, name: str) -> torch.Tensor:
        with self.lock:
            cached = self.values.pop(name, None)
            if cached is None:
                cached = load_file(self.root / name, device="cpu")["tokens"]
            self.values[name] = cached
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
            return cached

    def preload(self, workers: int = 8) -> None:
        """Materialize every token shard in host RAM before random sampling."""
        cache_key = str(self.root.resolve())
        with _RESIDENT_TOKEN_BANKS_LOCK:
            resident = _RESIDENT_TOKEN_BANKS.get(cache_key)
        if resident is not None:
            with self.lock:
                self.capacity = len(resident)
                self.values = OrderedDict(resident)
            print(
                f"reused resident token cache {self.root.name}: "
                f"{len(resident)} shards",
                flush=True,
            )
            return
        names = sorted(path.name for path in self.root.glob("part-*.safetensors"))
        if not names:
            raise RuntimeError(f"No token shards found in {self.root}")

        def load(name: str) -> tuple[str, torch.Tensor]:
            return name, load_file(self.root / name, device="cpu")["tokens"]

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            loaded = list(executor.map(load, names))
        with self.lock:
            self.capacity = len(loaded)
            self.values = OrderedDict(loaded)
        with _RESIDENT_TOKEN_BANKS_LOCK:
            _RESIDENT_TOKEN_BANKS[cache_key] = dict(loaded)
        total = sum(value.numel() * value.element_size() for _, value in loaded)
        print(
            f"resident token cache {self.root.name}: {len(loaded)} shards, "
            f"{total / 2**30:.2f} GiB in {time.perf_counter() - started:.1f}s",
            flush=True,
        )


class CachedTeacherReferenceLoader:
    """Artist-balanced reference-only loader for a cached image domain."""

    def __init__(
        self,
        token_root: Path | list[Path],
        *,
        split: str,
        style_ids: list[str],
        batch_size: int,
        references: int,
        seed: int,
        token_lru_shards: int = 8,
        ram_resident_tokens: bool = False,
        ram_preload_workers: int = 8,
        strict_style_ids: bool = True,
    ) -> None:
        token_roots = (
            [Path(token_root)]
            if isinstance(token_root, (str, Path))
            else [Path(value) for value in token_root]
        )
        if not token_roots:
            raise ValueError("At least one teacher token root is required")
        self.token_root = token_roots[0]
        self.token_roots = token_roots
        # Full paths make data provenance explicit. Several caches deliberately
        # share the same leaf directory name, which previously made a
        # synthetic-only teacher look indistinguishable from a Human cache.
        root_label = ",".join(str(root) for root in token_roots)
        self.batch_size = int(batch_size)
        self.references = int(references)
        self.seed = int(seed)
        if self.batch_size <= 0 or self.references <= 0:
            raise ValueError("Teacher batch and reference counts must be positive")
        allowed = set(style_ids)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for root_index, root in enumerate(token_roots):
            for row in read_records(root / "manifest.parquet"):
                style_id = str(row["style_id"])
                if str(row.get("split", "train")) == split and style_id in allowed:
                    grouped[style_id].append(
                        {**row, "_token_root_index": root_index}
                    )
        self.by_style = {
            style_id: sorted(rows, key=lambda row: int(row["id"]))
            for style_id, rows in grouped.items()
            if len(rows) >= self.references
        }
        missing = sorted(allowed - set(self.by_style))
        if missing and strict_style_ids:
            raise RuntimeError(
                f"Reference cache {root_label} is missing {len(missing)} "
                f"teacher artists for split {split!r}"
            )
        self.styles = sorted(self.by_style)
        if not self.styles:
            raise RuntimeError(
                f"Reference cache {root_label} has no teacher artists for "
                f"split {split!r}"
            )
        if missing:
            print(
                f"teacher reference intersection {root_label} split={split}: "
                f"{len(self.styles)}/{len(allowed)} styles",
                flush=True,
            )
        self.shards = [
            _FullTokenShardLRU(root, token_lru_shards) for root in token_roots
        ]
        if ram_resident_tokens:
            for shards in self.shards:
                shards.preload(ram_preload_workers)

    @staticmethod
    def _pin(value: torch.Tensor) -> torch.Tensor:
        return value.pin_memory() if torch.cuda.is_available() else value

    def load_step(self, step: int) -> dict[str, Any]:
        rng = random.Random(self.seed + int(step) * 1_000_003)
        styles = (
            rng.sample(self.styles, self.batch_size)
            if len(self.styles) >= self.batch_size
            else [rng.choice(self.styles) for _ in range(self.batch_size)]
        )
        selected = [
            rng.sample(self.by_style[style_id], self.references)
            for style_id in styles
        ]
        rows = [row for group in selected for row in group]
        grouped_rows: dict[
            tuple[int, str], list[tuple[int, dict[str, Any]]]
        ] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped_rows[
                (int(row["_token_root_index"]), str(row["token_shard"]))
            ].append((index, row))
        values: list[torch.Tensor | None] = [None] * len(rows)
        for (root_index, shard_name), shard_rows in grouped_rows.items():
            shard = self.shards[root_index].get(shard_name)
            for index, row in shard_rows:
                values[index] = shard[int(row["token_row"])]
        if any(value is None for value in values):
            raise RuntimeError("Teacher reference token load is incomplete")
        tokens = torch.stack([value for value in values if value is not None])
        episodes = [
            StyleEpisode(
                target_id=-1,
                reference_ids=tuple(int(row["id"]) for row in group),
                style_id=style_id,
                latent_shape=(0, 0),
                text_variant=0,
            )
            for style_id, group in zip(styles, selected, strict=True)
        ]
        positions = [
            (batch_index, reference_index)
            for batch_index in range(self.batch_size)
            for reference_index in range(self.references)
        ]
        mask = torch.ones(
            self.batch_size, self.references, dtype=torch.bool
        )
        return {
            "episodes": episodes,
            "cached_reference_tokens": self._pin(tokens),
            "reference_positions": positions,
            "reference_mask": self._pin(mask),
        }

    def prefetch(
        self, start_step: int, steps: int, workers: int = 1, depth: int = 4
    ) -> Iterator[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures: dict[int, Future[dict[str, Any]]] = {}
            next_step = int(start_step)
            stop = int(start_step) + int(steps)
            for step in range(int(start_step), stop):
                while next_step < stop and len(futures) < max(1, depth):
                    futures[next_step] = executor.submit(self.load_step, next_step)
                    next_step += 1
                yield futures.pop(step).result()


class DualQueryCachedStyleLoader(ProductionStyleLoader):
    """Production episode loader with sequential whole-shard token caching."""

    def __init__(self, destination: Path, cfg: dict[str, Any]):
        super().__init__(destination, cfg)
        if self.token_root is None:
            raise ValueError("Dual-query loader requires resampler_token_cache")
        self.full_token_shards = _FullTokenShardLRU(
            self.token_root, int(cfg.get("token_lru_shards", 8))
        )
        if bool(cfg.get("ram_resident_tokens", False)):
            self.full_token_shards.preload(int(cfg.get("ram_preload_workers", 8)))
        self.reference_count_weights_by_phase = dict(
            cfg.get("reference_count_weights_by_phase", {})
        )
        self.pilot_reference_schedule = list(
            cfg.get("pilot_reference_schedule", [])
        )

    @staticmethod
    def _pin(value: torch.Tensor) -> torch.Tensor:
        return value.pin_memory() if torch.cuda.is_available() else value

    def _load_episode_condition(
        self, item: StyleEpisode
    ) -> tuple[torch.Tensor, int]:
        row = self.text_by_key[(item.target_id, item.text_variant)]
        shard = self.text_shards.get(str(row["cache_shard"]))
        start = int(row["token_offset"])
        length = int(row["token_length"])
        return shard["conditioning"][start : start + length], length

    def _episode_prompt_mode(self, item: StyleEpisode) -> str:
        return str(
            self.text_by_key[(item.target_id, item.text_variant)].get(
                "variant_name", item.text_variant
            )
        )

    def episodes_for_step(self, step: int) -> list[StyleEpisode]:
        episodes = super().episodes_for_step(step)
        if self.pilot_reference_schedule:
            optimizer_step = step // self.gradient_accumulation_steps + 1
            stage = next(
                (
                    item
                    for item in self.pilot_reference_schedule
                    if optimizer_step <= int(item["end_step"])
                ),
                self.pilot_reference_schedule[-1],
            )
            maximum = int(stage["max_references"])
            weights = [float(value) for value in stage["reference_count_weights"]]
            if maximum <= 0 or len(weights) != maximum or any(value < 0 for value in weights):
                raise ValueError("Invalid pilot reference-count schedule")
            total = sum(weights)
            if total <= 0:
                raise ValueError("Empty pilot reference-count distribution")
            probabilities = [value / total for value in weights]
            rng = random.Random(self.seed ^ 0xA17E_10C0 ^ (step * 1_000_003))
            result = []
            for episode in episodes:
                pool = [
                    image_id
                    for image_id in self.by_style[episode.style_id]
                    if image_id != episode.target_id
                ]
                upper = min(maximum, len(pool))
                counts = list(range(1, upper + 1))
                selected_weights = probabilities[:upper]
                selected_total = sum(selected_weights)
                selected_weights = [
                    value / selected_total for value in selected_weights
                ]
                count = rng.choices(counts, weights=selected_weights, k=1)[0]
                result.append(
                    StyleEpisode(
                        episode.target_id,
                        tuple(rng.sample(pool, count)),
                        episode.style_id,
                        episode.latent_shape,
                        episode.text_variant,
                    )
                )
            return result
        if not self.reference_curriculum or not self.reference_count_weights_by_phase:
            return episodes
        optimizer_step = step // self.gradient_accumulation_steps + 1
        self_steps = int(self.reference_curriculum.get("self_reference_steps", 0))
        mix_end = int(
            self.reference_curriculum.get("target_mix_end_step", self_steps)
        )
        if optimizer_step <= self_steps:
            phase = "exact_self"
            maximum = 1
        elif optimizer_step <= mix_end:
            phase = "target_mix"
            maximum = int(
                self.reference_curriculum.get("target_mix_max_references", 4)
            )
        else:
            phase = "target_excluded"
            maximum = int(
                self.reference_curriculum.get("target_anneal_max_references", 8)
            )
        weights = self.reference_count_weights_by_phase.get(phase)
        if weights is None:
            return episodes
        probabilities = [float(value) for value in weights[:maximum]]
        if len(probabilities) != maximum or any(value < 0 for value in probabilities):
            raise ValueError(f"Invalid {phase} reference-count distribution")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError(f"Empty {phase} reference-count distribution")
        probabilities = [value / total for value in probabilities]
        rng = random.Random(self.seed ^ 0xD0A1_57A1 ^ (step * 1_000_003))
        result = []
        for episode in episodes:
            pool = [
                image_id
                for image_id in self.by_style[episode.style_id]
                if image_id != episode.target_id
            ]
            upper = min(maximum, len(pool))
            counts = list(range(1, upper + 1))
            selected_weights = probabilities[:upper]
            selected_total = sum(selected_weights)
            selected_weights = [value / selected_total for value in selected_weights]
            count = rng.choices(counts, weights=selected_weights, k=1)[0]
            result.append(
                StyleEpisode(
                    episode.target_id,
                    tuple(rng.sample(pool, count)),
                    episode.style_id,
                    episode.latent_shape,
                    episode.text_variant,
                )
            )
        return result

    def load_step(self, step: int) -> dict[str, Any]:
        episodes = self.episodes_for_step(step)
        latent_batch = torch.stack(
            [
                self.latent_shards.get(str(self.latent_by_id[item.target_id]["cache_shard"]))[
                    "latents"
                ][int(self.latent_by_id[item.target_id]["row_index"])]
                for item in episodes
            ]
        )
        conditions = []
        lengths = []
        for item in episodes:
            condition, length = self._load_episode_condition(item)
            conditions.append(condition)
            lengths.append(length)
        conditioning = _pad_text_conditions(conditions, self.text_conditioning_length)

        reference_ids = [image_id for item in episodes for image_id in item.reference_ids]
        target_ids = [item.target_id for item in episodes]
        values: dict[int, torch.Tensor] = {}
        grouped: dict[str, list[int]] = defaultdict(list)
        for image_id in dict.fromkeys(reference_ids + target_ids):
            grouped[str(self.token_by_id[image_id]["token_shard"])].append(image_id)
        for shard_name, image_ids in grouped.items():
            shard = self.full_token_shards.get(shard_name)
            for image_id in image_ids:
                values[image_id] = shard[int(self.token_by_id[image_id]["token_row"])]

        maximum_references = max(len(item.reference_ids) for item in episodes)
        reference_mask = torch.zeros(len(episodes), maximum_references, dtype=torch.bool)
        positions: list[tuple[int, int]] = []
        for batch_index, item in enumerate(episodes):
            reference_mask[batch_index, : len(item.reference_ids)] = True
            positions.extend((batch_index, index) for index in range(len(item.reference_ids)))
        reference_tokens = torch.stack([values[image_id] for image_id in reference_ids])
        target_tokens = torch.stack([values[image_id] for image_id in target_ids])
        return {
            "episodes": episodes,
            "latents": self._pin(latent_batch),
            "conditioning": self._pin(conditioning),
            "conditioning_lengths": self._pin(torch.tensor(lengths, dtype=torch.long)),
            "cached_reference_tokens": self._pin(reference_tokens),
            "cached_target_tokens": self._pin(target_tokens),
            "reference_positions": positions,
            "reference_mask": self._pin(reference_mask),
            "prompt_modes": [self._episode_prompt_mode(item) for item in episodes],
        }
