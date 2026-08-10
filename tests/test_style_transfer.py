from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from anima_style_data.style_transfer import (
    ProductionStyleLoader,
    SharedLowRankStyleAdapter,
    SlotSetAggregator,
    _archive_training_state,
    _pad_text_conditions,
    _save_training_state,
    attach_style_adapter,
)


def test_text_conditions_restore_animas_fixed_zero_padding():
    first = torch.randn(3, 8)
    second = torch.randn(5, 8)
    padded = _pad_text_conditions([first, second], 7)
    assert padded.shape == (2, 7, 8)
    torch.testing.assert_close(padded[0, :3], first)
    torch.testing.assert_close(padded[1, :5], second)
    assert torch.count_nonzero(padded[0, 3:]) == 0
    assert torch.count_nonzero(padded[1, 5:]) == 0


def test_slot_set_aggregator_is_reference_order_invariant():
    torch.manual_seed(7)
    model = SlotSetAggregator(slots=3, dim=12, heads=3, layers=2).eval()
    values = torch.randn(2, 4, 3, 12)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(values, mask)
    second = model(values[:, permutation], mask[:, permutation])
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)


def test_cross_slot_mixer_couples_pooled_slots():
    torch.manual_seed(11)
    model = SlotSetAggregator(
        slots=3, dim=12, heads=3, layers=1, slot_mixer_layers=1
    ).eval()
    values = torch.randn(1, 2, 3, 12)
    mask = torch.ones(1, 2, dtype=torch.bool)
    first = model(values, mask)
    changed = values.clone()
    changed[:, :, 1] *= -1.0
    second = model(changed, mask)
    # Changing slot 1 affects slot 0 only through the post-pooling slot mixer.
    assert not torch.allclose(first[:, 0], second[:, 0])


def test_episode_sampler_never_uses_target_as_reference():
    loader = ProductionStyleLoader.__new__(ProductionStyleLoader)
    loader.seed = 41
    loader.batch_size = 1
    loader.min_references = 1
    loader.max_references = 3
    loader.bucket_keys = [(32, 32)]
    loader.bucket_weights = [4]
    loader.buckets = {(32, 32): [10, 11, 12, 13]}
    loader.style_by_id = {
        image_id: {"artist": "artist", "style_id": "style"}
        for image_id in (10, 11, 12, 13)
    }
    loader.by_style = {"style": [10, 11, 12, 13]}
    loader.text_variants = {image_id: [0, 1] for image_id in (10, 11, 12, 13)}
    for step in range(12):
        episode = loader.episodes_for_step(step)[0]
        assert episode.target_id not in episode.reference_ids
        assert 1 <= len(episode.reference_ids) <= 3
        assert len(set(episode.reference_ids)) == len(episode.reference_ids)


class _FakeCrossAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.output_proj = nn.Linear(hidden, hidden, bias=False)
        self.output_dropout = nn.Identity()


def test_style_attention_reuses_frozen_q_and_updates_gate():
    torch.manual_seed(3)
    adapter = SharedLowRankStyleAdapter(
        style_dim=8, slots=2, hidden_dim=16, heads=4, blocks=28, rank=2,
        aggregator_heads=2, aggregator_layers=1, style_dropout=0.0, gate_dim=4,
    )
    cross = _FakeCrossAttention(16)
    cross.requires_grad_(False)
    adapter.set_style_tokens(torch.randn(2, 2, 8))
    query = torch.randn(2, 5, 16)
    timestep = torch.randn(2, 1, 16)

    # Zero-init gate makes attachment exactly neutral before training.
    initial = adapter.attend(0, query, timestep, cross)
    assert torch.count_nonzero(initial) == 0
    adapter.gate[-1].bias.data[0] = 0.1
    output = adapter.attend(0, query, timestep, cross)
    assert output.shape == query.shape
    output.square().mean().backward()
    assert adapter.shared_k.weight.grad is not None
    assert cross.q_proj.weight.grad is None


def test_attach_patches_all_28_blocks_without_copying_adapter():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()

        def _forward(self):
            return None

    anima = nn.Module()
    anima.blocks = nn.ModuleList([Block() for _ in range(28)])
    adapter = SharedLowRankStyleAdapter(
        style_dim=8, slots=2, hidden_dim=16, heads=4, blocks=28, rank=2,
        aggregator_heads=2, aggregator_layers=1, gate_dim=4,
    )
    attach_style_adapter(anima, adapter)
    assert anima.style_adapter is adapter
    assert [block.__dict__["_style_block_index"] for block in anima.blocks] == list(range(28))
    assert all(isinstance(block._forward, types.MethodType) for block in anima.blocks)


def test_training_state_is_atomic_and_archivable(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    current = tmp_path / "training_state.pt"
    archive = tmp_path / "checkpoints" / "step-0000001.pt"
    archive.parent.mkdir()
    _save_training_state(current, 1, model, optimizer, {"name": "test"})
    _archive_training_state(current, archive)
    state = torch.load(archive, map_location="cpu", weights_only=False)
    assert state["step"] == 1
    assert state["config"] == {"name": "test"}
    assert set(state["adapter"]) == set(model.state_dict())
