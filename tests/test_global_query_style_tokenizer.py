import torch

from anima_style_data.global_query_style_tokenizer import (
    GlobalQueryMemoryStyleTokenizer,
    attention_map_diversity_loss,
    reference_conditioned_diversity_loss,
)
from anima_style_data.dual_query_style_training import _save_state


def _model() -> GlobalQueryMemoryStyleTokenizer:
    return GlobalQueryMemoryStyleTokenizer(
        dim=64,
        spatial_tokens=16,
        global_tokens=4,
        artist_summary_tokens=2,
        output_tokens=8,
        heads=8,
        local_layers=1,
        cross_layers=2,
        ff_dim=128,
        output_rms_init=0.15,
    )


def test_global_query_tokenizer_is_reference_permutation_invariant():
    torch.manual_seed(7)
    model = _model().eval()
    references = torch.randn(3, 4, 22, 64)
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [True, False, False, False]]
    )
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(references, mask).tokens
    second = model(references[:, permutation], mask[:, permutation]).tokens
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_global_query_output_is_initialized_but_not_runtime_normalized():
    torch.manual_seed(11)
    model = _model()
    references = torch.randn(4, 2, 22, 64)
    mask = torch.ones(4, 2, dtype=torch.bool)
    output = model(references, mask)
    rms = output.tokens.float().square().mean().sqrt()
    assert 0.08 < float(rms) < 0.25
    sample_rms = output.tokens.float().square().mean(dim=(1, 2)).sqrt()
    assert float(sample_rms.std()) > 0
    loss = (
        output.tokens.float().square().mean()
        + 0.001 * attention_map_diversity_loss(output.attention_maps)
        + 0.001 * reference_conditioned_diversity_loss(output.tokens)
    )
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_checkpoint_preserves_optimizer_and_sparse_teacher_state(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    references = torch.randn(2, 1, 22, 64)
    mask = torch.ones(2, 1, dtype=torch.bool)
    model(references, mask).tokens.square().mean().backward()
    optimizer.step()
    path = tmp_path / "training_state.pt"
    _save_state(
        path,
        step=250,
        tokenizer=model,
        optimizer=optimizer,
        cfg={"model": {"include_artist_summary": True}},
        cache_summary={"slots": 84},
        trainer_state={
            "dual_domain_teacher_update_index": 62,
            "dual_domain_teacher_every": 4,
        },
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["step"] == 250
    assert state["optimizer"]["state"]
    assert state["trainer_state"]["dual_domain_teacher_update_index"] == 62
    assert state["trainer_state"]["dual_domain_teacher_every"] == 4
