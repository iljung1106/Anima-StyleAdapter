from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.dual_query_resampler import (
    DualQueryResampler,
    MultiReferenceSetTransformer,
    episodic_angular_prototype_loss,
    supervised_contrastive_loss,
    token_diversity_loss,
)
from anima_style_data.dual_query_training import train_dual_query_resampler
from anima_style_data.io import write_records
from safetensors.torch import save_file


def _small_resampler() -> DualQueryResampler:
    return DualQueryResampler(
        semantic_layers=(18, 24),
        semantic_dim=12,
        vae_channels=4,
        dim=32,
        spatial_query_grid=4,
        global_queries=4,
        layers=2,
        heads=4,
        ff_dim=64,
        artist_descriptor_dim=16,
        artist_pooling_queries=2,
        artist_summary_tokens=2,
        semantic_dropout=0.0,
        vae_dropout=0.0,
    )


def test_dual_query_resampler_reconstructs_both_modalities_and_backpropagates():
    torch.manual_seed(7)
    model = _small_resampler()
    batch = 4
    semantic = {
        18: torch.randn(batch, 12, 12),
        24: torch.randn(batch, 12, 12),
    }
    semantic_mask = torch.ones(batch, 12, dtype=torch.bool)
    semantic_shapes = torch.tensor([[3, 4]] * batch)
    latents = torch.randn(batch, 4, 8, 10)
    vae_shapes = torch.tensor([[8, 10]] * batch)
    image_sizes = torch.tensor([[64, 80]] * batch)

    output = model.encode(
        semantic,
        semantic_mask,
        semantic_shapes,
        latents,
        vae_shapes,
        image_sizes,
        reconstruct=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    prototype, metrics = episodic_angular_prototype_loss(output.descriptor, labels)
    contrastive = supervised_contrastive_loss(output.descriptor, labels)
    semantic_loss = sum(
        torch.nn.functional.smooth_l1_loss(output.semantic_reconstruction[layer], semantic[layer])
        for layer in (18, 24)
    )
    vae_loss = torch.nn.functional.smooth_l1_loss(output.vae_reconstruction, latents)
    loss = semantic_loss + 0.1 * vae_loss + 0.05 * (prototype + 0.25 * contrastive)
    loss = loss + 0.01 * token_diversity_loss(output.tokens)
    loss.backward()

    assert output.tokens.shape == (batch, 20, 32)
    assert output.artist_summary.shape == (batch, 2, 32)
    assert output.descriptor.shape == (batch, 16)
    assert output.semantic_reconstruction[18].shape == semantic[18].shape
    assert output.vae_reconstruction.shape == latents.shape
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["prototype_positive_cosine"])
    assert model.vae_stem[0].weight.grad is not None
    assert model.semantic_projections["18"].weight.grad is not None
    assert model.blocks[0].semantic_attention.k_proj.weight.grad is not None
    assert model.blocks[0].vae_attention.k_proj.weight.grad is not None


def test_set_transformer_is_reference_order_invariant_and_summary_is_optional():
    torch.manual_seed(11)
    aggregator = MultiReferenceSetTransformer(
        dim=32,
        output_tokens=6,
        heads=4,
        cross_layers=1,
        cross_slot_layers=1,
        ff_dim=64,
    ).eval()
    references = torch.randn(2, 3, 20, 32)
    summaries = torch.randn(2, 3, 2, 32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    permutation = torch.tensor([1, 0, 2])

    with_summary = aggregator(
        references, mask, artist_summary=summaries, include_artist_summary=True
    )
    permuted = aggregator(
        references[:, permutation],
        mask[:, permutation],
        artist_summary=summaries[:, permutation],
        include_artist_summary=True,
    )
    without_summary = aggregator(
        references, mask, artist_summary=summaries, include_artist_summary=False
    )

    assert with_summary.shape == (2, 6, 32)
    assert torch.allclose(with_summary, permuted, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(with_summary, without_summary)


def test_prototype_loss_rejects_batches_without_support_images():
    descriptors = torch.randn(3, 8)
    with pytest.raises(ValueError, match="two images per artist"):
        episodic_angular_prototype_loss(descriptors, torch.tensor([0, 1, 1]))


def test_angular_prototype_has_finite_gradient_for_collapsed_descriptors():
    # A randomly initialized artist head commonly emits almost identical
    # descriptors. This is the numerical boundary the pretraining loss must
    # escape rather than producing an acos backward NaN.
    descriptors = torch.ones(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss, _ = episodic_angular_prototype_loss(descriptors, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert descriptors.grad is not None
    assert torch.isfinite(descriptors.grad).all()


def test_training_proxy_breaks_the_collapsed_descriptor_symmetry():
    model = DualQueryResampler(
        semantic_layers=(18, 24),
        semantic_dim=12,
        vae_channels=4,
        dim=32,
        spatial_query_grid=2,
        global_queries=2,
        layers=1,
        heads=4,
        ff_dim=64,
        artist_descriptor_dim=16,
        artist_pooling_queries=2,
        artist_summary_tokens=2,
        artist_classes=4,
    )
    descriptors = torch.ones(4, 16, requires_grad=True)
    labels = torch.arange(4)

    loss, _ = model.artist_proxy_loss(descriptors, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert descriptors.grad is not None
    assert torch.isfinite(descriptors.grad).all()
    assert float(descriptors.grad.abs().sum()) > 0
    assert model.artist_proxies.grad is not None


def test_real_cache_contract_runs_one_training_and_validation_step(tmp_path):
    feature_root = tmp_path / "features"
    latent_root = tmp_path / "latents"
    feature_root.mkdir()
    latent_root.mkdir()
    feature_tensors = {}
    latent_tensors = []
    feature_rows = []
    latent_rows = []
    assignments = [
        ("train", "a"),
        ("train", "a"),
        ("train", "b"),
        ("train", "b"),
        ("validation", "c"),
        ("validation", "c"),
        ("validation", "d"),
        ("validation", "d"),
    ]
    for image_id, (split, artist) in enumerate(assignments, start=1):
        for layer in (18, 24):
            feature_tensors[f"{image_id}.layer_{layer:02d}_spatial"] = torch.randn(
                12, 12, dtype=torch.float16
            )
        latent_tensors.append(torch.randn(4, 8, 10, dtype=torch.float16))
        feature_rows.append(
            {
                "id": image_id,
                "artist": artist,
                "style_id": artist,
                "split": split,
                "target_height": 48,
                "target_width": 64,
                "spatial_tokens": 12,
                "spatial_dim": 12,
                "feature_shard": "part.safetensors",
            }
        )
        latent_rows.append(
            {
                "id": image_id,
                "artist": artist,
                "style_id": artist,
                "split": split,
                "target_height": 64,
                "target_width": 80,
                "latent_height": 8,
                "latent_width": 10,
                "row_index": image_id - 1,
                "cache_shard": "part.safetensors",
            }
        )
    save_file(feature_tensors, feature_root / "part.safetensors")
    save_file(
        {"latents": torch.stack(latent_tensors)}, latent_root / "part.safetensors"
    )
    write_records(feature_root / "manifest.parquet", feature_rows)
    write_records(latent_root / "manifest.parquet", latent_rows)
    config = {
        "dual_query_resampler": {
            "feature_directory": "features",
            "latent_directory": "latents",
            "output_directory": "run",
            "seed": 5,
            "model": {
                "semantic_layers": [18, 24],
                "dim": 32,
                "spatial_query_grid": 4,
                "global_queries": 4,
                "layers": 1,
                "heads": 4,
                "ff_dim": 64,
                "artist_descriptor_dim": 16,
                "artist_pooling_queries": 2,
                "artist_summary_tokens": 2,
                "semantic_dropout": 0.0,
                "vae_dropout": 0.0,
            },
            "training": {
                "device": "cpu",
                "steps": 1,
                "training_artist_count": 2,
                "validation_artist_count": 2,
                "images_per_artist_limit": 2,
                "artists_per_batch": 2,
                "images_per_artist": 2,
                "learning_rate": 0.001,
                "warmup_steps": 0,
                "fused_adamw": False,
                "semantic_reconstruction_sample_tokens": 12,
                "prefetch_workers": 1,
                "prefetch_batches": 1,
                "log_every": 1,
                "validation_every": 1,
                "validation_batches": 1,
                "checkpoint_every": 1,
                "wandb": {"enabled": False},
            },
        }
    }

    summary = train_dual_query_resampler(config, tmp_path)

    assert summary["steps"] == 1
    assert summary["last_validation"]["loss"] > 0
    assert (tmp_path / "run" / "training_state.pt").is_file()
