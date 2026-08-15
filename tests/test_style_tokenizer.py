from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from anima_style_data.style_tokenizer import (
    AnimaStyleTokenizer,
    _artist_direction_loss,
    _reference_tokens,
    _split_reference_views,
    _style_token_contrastive_loss,
    _validation_selection_score,
    export_style_tokenizer_checkpoint,
    insert_style_tokens,
)


def test_continuation_selection_score_rewards_correct_artist_advantage():
    row = {
        "validation_self": {"paired_flow_improvement": 0.004},
        "validation_heldout": {"paired_flow_improvement": 0.006},
        "validation_wrong_artist": {"paired_flow_improvement": 0.001},
    }

    assert _validation_selection_score(row) == pytest.approx(0.0095)


def test_style_tokenizer_is_reference_order_invariant_and_compact():
    tokenizer = AnimaStyleTokenizer(
        source_dim=16,
        context_dim=12,
        output_tokens=4,
        bottleneck_dim=8,
        score_hidden_dim=4,
        output_rms_init=0.2,
    ).eval()
    references = torch.randn(2, 3, 5, 16)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = tokenizer(references, mask)
    permutation = torch.tensor([1, 0, 2])
    actual = tokenizer(references[:, permutation], mask[:, permutation])

    assert expected.shape == (2, 4, 12)
    assert torch.isfinite(expected).all()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert float(expected.detach().float().square().mean().sqrt()) == pytest.approx(
        0.2, rel=0.05
    )


def test_style_tokenizer_accepts_bfloat16_cache_under_autocast():
    tokenizer = AnimaStyleTokenizer(
        source_dim=16,
        context_dim=16,
        output_tokens=2,
        bottleneck_dim=8,
        score_hidden_dim=4,
    ).eval()
    references = torch.randn(1, 1, 4, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = tokenizer(references, torch.ones(1, 1, dtype=torch.bool))

    assert output.shape == (1, 2, 16)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_style_token_insertion_preserves_text_and_backpropagates():
    conditioning = torch.zeros(2, 8, 4)
    conditioning[0, :3] = 1
    conditioning[1, :5] = 2
    style = torch.randn(2, 2, 4, requires_grad=True)

    result = insert_style_tokens(
        conditioning, torch.tensor([3, 5]), style
    )

    assert torch.equal(result[0, :3], conditioning[0, :3])
    assert torch.equal(result[1, :5], conditioning[1, :5])
    assert torch.equal(result[0, 3:5], style[0])
    assert torch.equal(result[1, 5:7], style[1])
    assert torch.count_nonzero(result[0, 5:]) == 0
    result.sum().backward()
    assert torch.equal(style.grad, torch.ones_like(style))


def test_style_token_insertion_reserves_tail_of_full_context():
    conditioning = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    style = torch.full((1, 2, 8), -1.0, requires_grad=True)

    result = insert_style_tokens(conditioning, torch.tensor([4]), style)

    assert torch.equal(result[:, :2], conditioning[:, :2])
    assert torch.equal(result[:, 2:], style)
    result.sum().backward()
    assert torch.equal(style.grad, torch.ones_like(style))


def test_wrong_artist_references_rotate_complete_batch_entries():
    batch = {
        "cached_reference_tokens": torch.tensor([
            [[1.0, 1.0]],
            [[2.0, 2.0]],
            [[3.0, 3.0]],
        ]),
        "reference_mask": torch.tensor([
            [True, True],
            [True, False],
        ]),
        "reference_positions": [(0, 0), (0, 1), (1, 0)],
    }

    heldout, heldout_mask = _reference_tokens(batch, "cpu", mode="heldout")
    wrong, wrong_mask = _reference_tokens(batch, "cpu", mode="wrong_artist")

    assert torch.equal(wrong, heldout.roll(1, dims=0))
    assert torch.equal(wrong_mask, heldout_mask.roll(1, dims=0))


def test_wrong_artist_references_skip_duplicate_artist_in_batch():
    batch = {
        "cached_reference_tokens": torch.tensor([
            [[1.0]], [[2.0]], [[3.0]],
        ]),
        "reference_mask": torch.ones(3, 1, dtype=torch.bool),
        "reference_positions": [(0, 0), (1, 0), (2, 0)],
        "episodes": [
            SimpleNamespace(style_id="artist-a"),
            SimpleNamespace(style_id="artist-a"),
            SimpleNamespace(style_id="artist-b"),
        ],
    }

    wrong, _ = _reference_tokens(batch, "cpu", mode="wrong_artist")

    assert torch.equal(
        wrong.float().flatten(), torch.tensor([3.0, 3.0, 1.0])
    )


def test_reference_views_are_disjoint_and_skip_single_reference_rows():
    mask = torch.tensor([
        [True, True, True, False],
        [True, False, False, False],
        [True, True, False, False],
    ])

    eligible, first, second = _split_reference_views(mask)

    assert torch.equal(eligible, torch.tensor([True, False, True]))
    assert not bool((first & second).any())
    assert torch.equal(first | second, mask[eligible])
    assert bool(first.any(dim=1).all())
    assert bool(second.any(dim=1).all())


def test_slotwise_contrastive_loss_prefers_matching_artist_views():
    first = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[-1.0, 0.0], [0.0, -1.0]],
    ])
    matching = first.clone()
    swapped = first.flip(0)

    matching_loss, metrics = _style_token_contrastive_loss(
        first, matching, ["a", "b"], temperature=0.1
    )
    swapped_loss, _ = _style_token_contrastive_loss(
        first, swapped, ["a", "b"], temperature=0.1
    )

    assert matching_loss < swapped_loss
    assert metrics["token_similarity_margin"] > 0


def test_artist_direction_loss_prefers_correct_residual_alignment():
    target = torch.tensor([[[[1.0, 0.0]]]])
    base = torch.zeros_like(target)
    correct = target.clone().requires_grad_(True)
    wrong = torch.tensor([[[[0.0, 1.0]]]])

    aligned_loss, metrics = _artist_direction_loss(
        correct, wrong, base, target, margin=0.02, centered_weight=0.25
    )
    reversed_loss, _ = _artist_direction_loss(
        -correct, wrong, base, target, margin=0.02, centered_weight=0.25
    )
    aligned_loss.backward()

    assert aligned_loss < reversed_loss
    assert metrics["artist_correct_direction_cosine"] > metrics[
        "artist_wrong_direction_cosine"
    ]
    assert correct.grad is not None


def test_selected_tokenizer_exports_verified_safetensors_bundle(tmp_path):
    model_cfg = {
        "source_dim": 16,
        "context_dim": 12,
        "output_tokens": 4,
        "bottleneck_dim": 8,
        "score_hidden_dim": 4,
        "output_rms_init": 0.2,
    }
    tokenizer = AnimaStyleTokenizer(**model_cfg)
    output = tmp_path / "tokenizer-run"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    selected = checkpoint_dir / "selected.pt"
    torch.save({"tokenizer": tokenizer.state_dict()}, selected)
    selection = {
        "selected_step": 1500,
        "selection_rule": "test-rule",
        "resampler_cache": {"resampler_checkpoint_sha256": "abc"},
        "reference_count_evaluation": {"1": {"heldout": {}}},
        "candidates": [{"step": 1500, "score": 1.0}],
    }
    (output / "selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    config = {
        "style_tokenizer_selection": {
            "source_config_section": "production_tokenizer"
        },
        "production_tokenizer": {
            "output_directory": "tokenizer-run",
            "model": model_cfg,
        },
        "anima_cache": {
            "models": {"repo_id": "circlestone-labs/Anima", "revision": "test"}
        },
    }

    manifest = export_style_tokenizer_checkpoint(config, tmp_path)

    assert manifest["roundtrip_verified"] is True
    assert manifest["input_contract"]["output_tokens"] == ["batch", 4, 12]
    assert (output / "deploy" / "style_tokenizer.safetensors").is_file()
    assert len(manifest["weights_sha256"]) == 64
