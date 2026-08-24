import json

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from anima_style_data.kv_activation_modulation import (
    NativeKVFactorModulator,
    apply_kv_factors,
    canonicalize_lora_factor_bank,
    canonicalize_lora_factors,
    compress_lora_factors,
    kv_activation_objective,
    kv_factor_objective,
    load_kv_lora_factor_bank,
)
from anima_style_data.kv_activation_sampling import NativeKVFactorInjector
from anima_style_data.kv_mixture_analysis import (
    _activation_from_coefficients,
    _fixed_artist_holdout,
    _knn_coefficients,
    _mixture_rank_energy_retention,
    _sparse_ridge_coefficients,
)
from anima_style_data.kv_generalizing_modulator import (
    _average_reader_anchors_by_count,
    _stratified_view_indices,
    build_mixed_activation_batch,
    concatenate_weighted_lora_factors,
)
from anima_style_data.few_shot_kv_adapter import (
    CountAwareRetrievalFewShotKVStyleAdapter,
    FewShotNativeKVStyleAdapter,
    RetrievalFewShotKVStyleAdapter,
    compress_mean_lora_dictionary,
)
from anima_style_data.kv_sparse_mixture_selector import (
    SparseLoRAMixtureSelector,
    sparse_mixture_coefficients,
)


def test_generalizing_validation_samples_every_reference_count():
    counts = torch.tensor([1, 1, 1, 1, 2, 2, 4], dtype=torch.float32)

    selected = _stratified_view_indices(counts, views_per_count=2)

    assert selected.tolist() == [0, 3, 4, 5, 6]
    assert counts[selected].tolist() == [1.0, 1.0, 2.0, 2.0, 4.0]


def test_reader_anchor_cache_averages_views_with_the_same_count():
    codes = torch.tensor([
        [[[1.0]], [[3.0]], [[10.0]], [[20.0]]],
        [[[2.0]], [[4.0]], [[12.0]], [[24.0]]],
    ])
    counts = torch.tensor([1, 1, 2, 4])

    anchors, unique = _average_reader_anchors_by_count(codes, counts)

    assert unique.tolist() == [1, 2, 4]
    torch.testing.assert_close(
        anchors[:, :, 0, 0],
        torch.tensor([[2.0, 3.0], [10.0, 12.0], [20.0, 24.0]]),
    )


def test_sparse_selector_coefficients_respect_topk_and_exclusion():
    similarity = torch.tensor([[0.1, 0.9, 0.8, 0.7], [0.9, 0.8, 0.1, 0.0]])
    excluded = torch.tensor([
        [False, True, False, False],
        [True, False, False, False],
    ])

    coefficients = sparse_mixture_coefficients(
        similarity, excluded=excluded, neighbors=2, temperature=0.1
    )

    torch.testing.assert_close(coefficients.sum(dim=-1), torch.ones(2))
    assert (coefficients > 0).sum(dim=-1).tolist() == [2, 2]
    assert coefficients[0, 1] == 0
    assert coefficients[1, 0] == 0


def test_sparse_selector_learns_on_top_of_raw_reader_metric():
    torch.manual_seed(23)
    model = SparseLoRAMixtureSelector(
        slots=3, style_dim=4, hidden_dim=8, learned_mix_initial=0.15
    )
    anchors = torch.randn(5, 3, 4)
    query = anchors[:2] + 0.01 * torch.randn(2, 3, 4)

    similarity, metrics = model.similarities(query, anchors)
    loss = -similarity[:, :2].diagonal().mean()
    loss.backward()

    assert similarity.shape == (2, 5)
    assert 0.14 < float(metrics["learned_metric_fraction"]) < 0.16
    assert model.projection[0].weight.grad is not None
    assert model.learned_mix_logit.grad is not None


def test_sparse_ridge_refits_only_selected_dictionary_atoms():
    torch.manual_seed(31)
    train = torch.randn(12, 20)
    query = 0.7 * train[2:3] - 0.3 * train[7:8]

    coefficients = _sparse_ridge_coefficients(
        train, query, neighbors=3, ridge=0.01
    )

    assert coefficients.shape == (1, 12)
    assert int((coefficients != 0).sum()) == 3
    common = train.mean(dim=0, keepdim=True)
    reconstructed = common + coefficients @ (train - common)
    assert F.cosine_similarity(reconstructed, query).item() > 0.95


def test_expanded_dictionary_keeps_the_original_artist_holdout():
    original = [f"artist-{index}" for index in range(8)]
    expanded = original + ["artist-8", "artist-9"]

    train, validation = _fixed_artist_holdout(
        expanded, validation_count=3, source_artist_ids=original
    )

    assert [expanded[index] for index in validation] == [
        "artist-0", "artist-4", "artist-7"
    ]
    assert set(train).isdisjoint(validation)
    assert len(train) == 7


def test_apply_kv_factors_matches_explicit_low_rank_linears():
    torch.manual_seed(5)
    context = torch.randn(3, 7, 6)
    down = torch.randn(3, 2, 2, 6)
    up = torch.randn(3, 2, 8, 2)

    actual = apply_kv_factors(context, down, up)
    expected = torch.stack([
        torch.stack([
            (context[row] @ down[row, kind].t()) @ up[row, kind].t()
            for kind in range(2)
        ])
        for row in range(3)
    ])

    torch.testing.assert_close(actual, expected)


def test_concatenated_lora_factors_are_the_exact_weighted_function_sum():
    torch.manual_seed(17)
    batch, neighbors, blocks, kinds, rank = 2, 3, 2, 2, 2
    context_dim, output_dim, tokens = 5, 7, 4
    down = torch.randn(
        batch, neighbors, blocks, kinds, rank, context_dim
    )
    up = torch.randn(
        batch, neighbors, blocks, kinds, output_dim, rank
    )
    weights = torch.rand(batch, neighbors)
    weights /= weights.sum(dim=-1, keepdim=True)
    mixed_down, mixed_up = concatenate_weighted_lora_factors(
        down, up, weights
    )
    context = torch.randn(batch, tokens, context_dim)

    for block in range(blocks):
        actual = apply_kv_factors(
            context, mixed_down[:, block], mixed_up[:, block]
        )
        expected = sum(
            weights[:, neighbor, None, None, None]
            * apply_kv_factors(
                context, down[:, neighbor, block], up[:, neighbor, block]
            )
            for neighbor in range(neighbors)
        )
        torch.testing.assert_close(actual, expected)


def test_factor_bank_consolidates_and_reuses_the_single_file_cache(tmp_path):
    root = tmp_path / "teachers"
    weights = root / "weights"
    weights.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"artists": [{
        "index": 0,
        "style_id": "artist-zero",
        "artist": "artist_zero",
        "train_ids": [1],
        "validation_ids": [2],
    }]}), encoding="utf-8")
    source = weights / "artist-000-test.safetensors"
    tensors = {}
    for kind in ("k", "v"):
        prefix = f"lora_unet_blocks_0_cross_attn_{kind}_proj."
        tensors[prefix + "lora_down.weight"] = torch.randn(2, 3)
        tensors[prefix + "lora_up.weight"] = torch.randn(4, 2)
        tensors[prefix + "alpha"] = torch.tensor(2.0)
    save_file(tensors, source)

    artist_ids, first_down, first_up = load_kv_lora_factor_bank(root, blocks=1)
    cache = root / "factor_bank_1x1_fp16.safetensors"
    assert cache.exists()
    assert artist_ids == ["artist-zero"]

    source.unlink()
    cached_ids, cached_down, cached_up = load_kv_lora_factor_bank(root, blocks=1)
    assert cached_ids == artist_ids
    torch.testing.assert_close(cached_down, first_down, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(cached_up, first_up, atol=1e-3, rtol=1e-3)


def test_mixture_rank_retention_reports_the_irreducible_rank_bottleneck():
    # Three orthogonal rank-one deltas have equal singular energy. A rank-one
    # student can retain exactly one third; rank two retains two thirds.
    down = torch.eye(3).reshape(3, 1, 3)
    up = torch.eye(3).t().reshape(3, 3, 1)
    weights = torch.ones(3) / 3

    rank_one = _mixture_rank_energy_retention(
        down, up, weights, target_rank=1
    )
    rank_two = _mixture_rank_energy_retention(
        down, up, weights, target_rank=2
    )

    assert abs(rank_one - 1 / 3) < 1e-6
    assert abs(rank_two - 2 / 3) < 1e-6


def test_canonical_factors_preserve_the_exact_weight_delta():
    torch.manual_seed(11)
    down = torch.randn(3, 7)
    up = torch.randn(9, 3)

    canonical_down, canonical_up = canonicalize_lora_factors(down, up)

    torch.testing.assert_close(
        canonical_up @ canonical_down,
        up @ down,
        atol=1e-5,
        rtol=1e-5,
    )


def test_batched_canonicalization_preserves_every_weight_delta():
    torch.manual_seed(13)
    down = torch.randn(2, 3, 2, 7)
    up = torch.randn(2, 3, 9, 2)

    canonical_down, canonical_up = canonicalize_lora_factor_bank(
        down, up, chunk_size=4
    )

    torch.testing.assert_close(
        canonical_up @ canonical_down,
        up @ down,
        atol=1e-5,
        rtol=1e-5,
    )


def test_modulator_emits_independent_kv_factors_and_backpropagates():
    model = NativeKVFactorModulator(
        style_dim=12,
        blocks=4,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
    )
    style = torch.randn(3, 5, 12)

    down, up = model(style, 2)
    loss = down.square().mean() + up.square().mean()
    loss.backward()

    assert down.shape == (3, 2, 2, 6)
    assert up.shape == (3, 2, 8, 2)
    assert model.style_projection.weight.grad is not None
    assert model.down_head.weight.grad is not None
    assert model.up_head.weight.grad is not None


def test_modulator_applies_block_rank_scales_without_losing_gradients():
    model = NativeKVFactorModulator(
        style_dim=12,
        blocks=4,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
    )
    down_scale = torch.full((4, 2, 2), 0.02)
    up_scale = torch.full((4, 2, 2), 0.003)
    model.set_factor_scales(down_scale, up_scale)

    down, up = model(torch.randn(3, 5, 12), 1)
    down_rms = down.square().mean(dim=-1).sqrt()
    up_rms = up.transpose(-1, -2).square().mean(dim=-1).sqrt()

    torch.testing.assert_close(
        down_rms, torch.full_like(down_rms, 0.02), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        up_rms, torch.full_like(up_rms, 0.003), atol=1e-5, rtol=1e-5
    )


def test_block_specific_heads_only_update_the_selected_block():
    model = NativeKVFactorModulator(
        style_dim=12,
        blocks=3,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
        block_specific_heads=True,
    )

    down, up = model(torch.randn(2, 5, 12), 1)
    (down.square().mean() + up.square().mean()).backward()

    assert model.down_head[2].weight.grad is not None
    assert model.up_head[3].weight.grad is not None
    assert model.down_head[0].weight.grad is None
    assert model.up_head[5].weight.grad is None


def test_shared_heads_with_block_low_rank_delta_backpropagate_selected_block():
    model = NativeKVFactorModulator(
        style_dim=12,
        blocks=3,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
        block_delta_rank=3,
    )

    down, up = model(torch.randn(2, 5, 12), 1)
    (down.square().mean() + up.square().mean()).backward()

    assert model.down_head.weight.grad is not None
    assert model.up_head.weight.grad is not None
    assert torch.count_nonzero(model.down_delta_b.grad[1]) > 0
    assert torch.count_nonzero(model.up_delta_b.grad[1]) > 0
    assert torch.count_nonzero(model.down_delta_b.grad[0]) == 0
    assert torch.count_nonzero(model.up_delta_b.grad[2]) == 0


def test_activation_objective_prefers_exact_teacher_delta():
    teacher = torch.randn(4, 2, 7, 8)
    exact_loss, exact = kv_activation_objective(
        teacher.clone(), teacher, direction_weight=0.5, magnitude_weight=0.1
    )
    collapsed = torch.zeros_like(teacher, requires_grad=True)
    collapsed_loss, collapsed_metrics = kv_activation_objective(
        collapsed, teacher, direction_weight=0.5, magnitude_weight=0.1
    )

    assert float(exact["cosine"]) > 0.999
    assert float(exact["relative_rms_error"]) < 1e-6
    assert float(collapsed_loss) > float(exact_loss) + 1.0
    collapsed_loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_factor_objective_prefers_canonical_teacher_factors():
    teacher = torch.randn(4, 2, 3, 8)
    exact_loss, exact = kv_factor_objective(
        teacher.clone(), teacher, direction_weight=0.5, magnitude_weight=0.1
    )
    wrong_loss, wrong = kv_factor_objective(
        teacher.roll(1, dims=0),
        teacher,
        direction_weight=0.5,
        magnitude_weight=0.1,
    )

    assert float(exact["cosine"]) > 0.999
    assert float(wrong["cosine"]) < 0.5
    assert float(wrong_loss) > float(exact_loss) + 0.5


class _DummyCrossAttention(torch.nn.Module):
    def __init__(self, context_dim: int, output_dim: int):
        super().__init__()
        self.kv_proj = torch.nn.Linear(context_dim, output_dim * 2, bias=False)


class _DummyBlock(torch.nn.Module):
    def __init__(self, context_dim: int, output_dim: int):
        super().__init__()
        self.cross_attn = _DummyCrossAttention(context_dim, output_dim)


class _DummyAnima(torch.nn.Module):
    def __init__(self, blocks: int, context_dim: int, output_dim: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            _DummyBlock(context_dim, output_dim) for _ in range(blocks)
        )


def test_native_kv_factor_injector_matches_exact_low_rank_delta_with_cfg_rows():
    torch.manual_seed(23)
    anima = _DummyAnima(blocks=2, context_dim=6, output_dim=8)
    injector = NativeKVFactorInjector(anima)
    context = torch.randn(4, 7, 6)
    down = torch.randn(2, 2, 2, 3, 6)
    up = torch.randn(2, 2, 2, 8, 3)
    injector.set_factors(down, up, strength=0.75)

    native = F.linear(context, anima.blocks[1].cross_attn.kv_proj.weight)
    actual = anima.blocks[1].cross_attn.kv_proj(context)
    repeated_down = down[:, 1].repeat(2, 1, 1, 1)
    repeated_up = up[:, 1].repeat(2, 1, 1, 1)
    delta = apply_kv_factors(context, repeated_down, repeated_up)
    expected = native + 0.75 * torch.cat((delta[:, 0], delta[:, 1]), dim=-1)

    torch.testing.assert_close(actual, expected)
    injector.disable()
    torch.testing.assert_close(
        anima.blocks[1].cross_attn.kv_proj(context), native
    )
    injector.close()


def test_convex_activation_mixture_uses_visual_neighbor_weights():
    train_codes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = torch.tensor([[0.9, 0.1]])
    weights = _knn_coefficients(
        train_codes, query, neighbors=2, temperature=0.05
    )
    activations = torch.arange(3 * 2 * 1 * 1).reshape(3, 2, 1, 1).float()
    mixed = _activation_from_coefficients(
        activations, weights, affine_centered=False
    )

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    assert torch.count_nonzero(weights) == 2
    torch.testing.assert_close(
        mixed,
        torch.einsum("va,akno->vkno", weights, activations),
    )


def test_generalizing_batch_builds_exact_weighted_teacher_activation():
    torch.manual_seed(37)
    batch, contexts, groups = 2, 2, 3
    tokens, context_dim, output_dim, rank = 5, 7, 11, 3
    sampled_contexts = torch.randn(contexts, tokens, context_dim)
    predicted_down = torch.randn(batch, 2, rank, context_dim)
    predicted_up = torch.randn(batch, 2, output_dim, rank)
    group_down = torch.randn(batch, groups, 2, rank, context_dim)
    group_up = torch.randn(batch, groups, 2, output_dim, rank)
    weights = torch.rand(batch, groups)
    weights /= weights.sum(dim=-1, keepdim=True)
    output_indices = torch.tensor([1, 4, 8])

    student, target = build_mixed_activation_batch(
        sampled_contexts,
        predicted_down,
        predicted_up,
        group_down,
        group_up,
        weights,
        output_indices,
    )

    assert student.shape == (batch * contexts, 2, tokens, 3)
    expected_rows = []
    for artist in range(batch):
        for context in sampled_contexts:
            group_rows = apply_kv_factors(
                context.expand(groups, -1, -1),
                group_down[artist],
                group_up[artist].index_select(2, output_indices),
            )
            expected_rows.append(
                torch.einsum("g,gkno->kno", weights[artist], group_rows)
            )
    torch.testing.assert_close(target, torch.stack(expected_rows))


class _DummyReader(torch.nn.Module):
    def __init__(self, style_dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(style_dim))

    def forward(self, tokens, mask):
        if tokens.ndim != 4:
            raise ValueError("Expected real [batch,references,tokens,dim] contract")
        weights = mask.to(tokens.dtype)
        pooled = (tokens * weights[:, :, None, None]).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1)[:, None, None]
        return type("ReaderOutput", (), {"tokens": pooled * self.scale})()


def test_few_shot_adapter_activates_and_rescales_native_kv_hook():
    torch.manual_seed(41)
    anima = _DummyAnima(blocks=2, context_dim=6, output_dim=8)
    reader = _DummyReader(style_dim=12)
    modulator = NativeKVFactorModulator(
        style_dim=12,
        blocks=2,
        rank=2,
        context_dim=6,
        output_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ff_dim=32,
    )
    adapter = FewShotNativeKVStyleAdapter(
        reader=reader, modulator=modulator, anima=anima
    )
    context = torch.randn(1, 5, 6)
    baseline = anima.blocks[0].cross_attn.kv_proj(context)

    codes = adapter.set_references(torch.randn(1, 4, 5, 12), strength=0.5)
    styled_half = anima.blocks[0].cross_attn.kv_proj(context)
    adapter.set_strength(1.0)
    styled_full = anima.blocks[0].cross_attn.kv_proj(context)

    assert codes.shape == (1, 5, 12)
    torch.testing.assert_close(styled_full - baseline, 2 * (styled_half - baseline))
    adapter.disable()
    torch.testing.assert_close(anima.blocks[0].cross_attn.kv_proj(context), baseline)
    adapter.close()


def test_retrieval_adapter_uses_an_exact_sparse_lora_mixture():
    torch.manual_seed(47)
    anima = _DummyAnima(blocks=2, context_dim=6, output_dim=8)
    reader = _DummyReader(style_dim=12)
    anchors = torch.randn(1, 5, 5, 12)
    down = torch.randn(5, 2, 2, 2, 6)
    up = torch.randn(5, 2, 2, 8, 2)
    adapter = RetrievalFewShotKVStyleAdapter(
        reader=reader,
        anima=anima,
        anchor_codes=anchors,
        anchor_reference_counts=torch.tensor([4]),
        artist_ids=[f"artist-{index}" for index in range(5)],
        teacher_down=down,
        teacher_up=up,
        neighbors=2,
        temperature=0.05,
    )
    context = torch.randn(1, 7, 6)
    baseline = anima.blocks[1].cross_attn.kv_proj(context)
    references = anchors[0, 2].unsqueeze(0).unsqueeze(0).expand(1, 4, -1, -1)

    adapter.set_references(references, strength=1.0)
    actual = anima.blocks[1].cross_attn.kv_proj(context)
    retrieval = adapter.last_retrieval[0]
    indices = torch.tensor(retrieval["artist_indices"])
    weights = torch.tensor(retrieval["weights"])
    expected_down, expected_up = concatenate_weighted_lora_factors(
        adapter.teacher_down[indices][None],
        adapter.teacher_up[indices][None],
        weights[None],
    )
    delta = apply_kv_factors(
        context,
        expected_down[:, 1].to(torch.bfloat16).float(),
        expected_up[:, 1].to(torch.bfloat16).float(),
    )
    expected = baseline + torch.cat((delta[:, 0], delta[:, 1]), dim=-1)

    torch.testing.assert_close(actual, expected)
    assert len(retrieval["artist_ids"]) == 2
    adapter.close()


def test_count_aware_adapter_routes_single_and_multi_reference_rows():
    torch.manual_seed(49)
    anima = _DummyAnima(blocks=2, context_dim=6, output_dim=8)
    reader = _DummyReader(style_dim=12)
    anchors = torch.randn(2, 6, 5, 12)
    down = torch.randn(6, 2, 2, 2, 6)
    up = torch.randn(6, 2, 2, 8, 2)
    common_down, common_up = compress_mean_lora_dictionary(
        down,
        up,
        target_rank=4,
        device="cpu",
        oversample=4,
        seed=7,
    )
    adapter = CountAwareRetrievalFewShotKVStyleAdapter(
        reader=reader,
        anima=anima,
        anchor_codes=anchors,
        anchor_reference_counts=torch.tensor([1, 2]),
        artist_ids=[f"artist-{index}" for index in range(6)],
        teacher_down=down,
        teacher_up=up,
        neighbors=2,
        temperature=0.05,
        ridge_common_down=common_down,
        ridge_common_up=common_up,
        convex_dictionary_size=4,
        ridge_min_references=2,
        ridge_neighbors=3,
        ridge_rank=3,
        ridge_gain=1.5,
        compression_oversample=3,
        compression_seed=11,
    )
    references = torch.stack((
        torch.stack((anchors[0, 1], torch.zeros_like(anchors[0, 1]))),
        torch.stack((anchors[1, 4], anchors[1, 4])),
    ))
    mask = torch.tensor([[True, False], [True, True]])
    context = torch.randn(4, 7, 6)
    baseline = anima.blocks[0].cross_attn.kv_proj(context)

    adapter.set_references(references, mask, strength=1.0)
    styled = anima.blocks[0].cross_attn.kv_proj(context)

    assert adapter.last_retrieval[0]["route"] == "convex_knn"
    assert adapter.last_retrieval[1]["route"] == "sparse_signed_ridge"
    assert adapter.last_retrieval[1]["effective_gain"] == 1.5
    assert adapter.injector.down.shape[0] == 2
    assert adapter.injector.down.shape[-2] == 4
    assert not torch.equal(styled, baseline)
    adapter.set_common_scale(0.25)
    _, centered_down, _, centered_retrieval = adapter.encode_reference_tokens(
        references[:1], mask[:1]
    )
    assert centered_retrieval[0]["common_scale"] == 0.25
    assert centered_down.shape[-2] <= 6
    adapter.close()


def test_compressed_lora_dictionary_mean_preserves_dense_mean():
    torch.manual_seed(51)
    down = torch.randn(3, 1, 2, 2, 7)
    up = torch.randn(3, 1, 2, 6, 2)
    compressed_down, compressed_up = compress_mean_lora_dictionary(
        down,
        up,
        target_rank=6,
        device="cpu",
        oversample=2,
        seed=17,
    )
    expected = (up @ down).mean(dim=0)
    actual = compressed_up @ compressed_down

    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


def test_randomized_lora_compression_matches_the_best_low_rank_matrix():
    torch.manual_seed(53)
    down = torch.randn(2, 12, 17)
    up = torch.randn(2, 13, 12)
    compressed_down, compressed_up = compress_lora_factors(
        down,
        up,
        target_rank=5,
        oversample=7,
        power_iterations=1,
        seed=91,
    )
    dense = up @ down
    left, singular, right_h = torch.linalg.svd(dense, full_matrices=False)
    optimal = (left[:, :, :5] * singular[:, None, :5]) @ right_h[:, :5]
    actual = compressed_up @ compressed_down

    torch.testing.assert_close(actual, optimal, rtol=2e-4, atol=2e-4)


def test_native_kv_injector_repeats_style_rows_in_cfg_branch_order():
    torch.manual_seed(43)
    anima = _DummyAnima(blocks=1, context_dim=6, output_dim=8)
    injector = NativeKVFactorInjector(anima)
    down = torch.randn(2, 1, 2, 2, 6)
    up = torch.randn(2, 1, 2, 8, 2)
    # _sample_anima_batch concatenates all negative rows followed by all
    # positive rows, so both CFG branches must see the same [A, B] order.
    context = torch.randn(4, 5, 6)
    baseline = anima.blocks[0].cross_attn.kv_proj(context)

    injector.set_factors(down, up)
    actual = anima.blocks[0].cross_attn.kv_proj(context)

    repeated_down = torch.cat((down[:, 0], down[:, 0]), dim=0)
    repeated_up = torch.cat((up[:, 0], up[:, 0]), dim=0)
    delta = apply_kv_factors(context, repeated_down, repeated_up)
    expected = baseline + torch.cat((delta[:, 0], delta[:, 1]), dim=-1)
    torch.testing.assert_close(actual, expected)
    injector.close()
