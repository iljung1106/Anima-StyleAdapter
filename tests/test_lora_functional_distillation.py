import json

import torch
from safetensors.torch import save_file

from anima_style_data.lora_functional_distillation import (
    _configure_reader_trainable_scope,
    _cached_training_probe_bank,
    _initialize_fresh_adapter_strength,
    _teacher_decomposed_functional_objective,
    build_mixture_specs,
    decompose_teacher_effects,
    scheduled_teacher_category,
    scheduled_reference_domain,
    teacher_category,
    teacher_category_v2,
)
from anima_style_data.detail_style_cross_attention import (
    DetailPreservingTypedSlotReader,
    SeparatedCommonArtistKVStyleCrossAttention,
)
from anima_style_data.io import write_records


def test_teacher_schedule_becomes_exact_three_way_cycle():
    assert [teacher_category(step, single_only_steps=2) for step in range(1, 9)] == [
        "lora_single",
        "lora_single",
        "artist_tag",
        "lora_single",
        "lora_mixture",
        "artist_tag",
        "lora_single",
        "lora_mixture",
    ]


def test_mixture_plan_is_normalized_and_unique():
    specs = build_mixture_specs(8, pair_count=8, triple_count=8, seed=17)
    assert len(specs) == 24
    compound = [spec.components for spec in specs if spec.kind != "single"]
    assert len(compound) == len(set(compound))
    assert all(abs(sum(spec.weights) - 1.0) < 1e-7 for spec in specs)
    assert all(all(weight > 0 for weight in spec.weights) for spec in specs)


def test_teacher_decomposition_preserves_effects_and_centers_artist_part():
    values = torch.randn(4, 3, 5)
    common, centered = decompose_teacher_effects(values)
    torch.testing.assert_close(common + centered, values)
    torch.testing.assert_close(centered.mean(dim=0), torch.zeros_like(common[0]))


def test_v2_teacher_schedule_delays_mixtures_until_artist_intro_finishes():
    assert [
        teacher_category_v2(step, single_only_steps=2, artist_intro_steps=3)
        for step in range(1, 9)
    ] == [
        "lora_single",
        "lora_single",
        "lora_single",
        "artist_tag",
        "lora_single",
        "artist_tag",
        "lora_single",
        "lora_mixture",
    ]


def test_direct_kv_teacher_schedule_keeps_mixtures_as_offline_supervision():
    assert [
        scheduled_teacher_category(
            step,
            single_only_steps=2,
            schedule=("artist_tag", "lora_single", "lora_mixture"),
        )
        for step in range(1, 9)
    ] == [
        "lora_single",
        "lora_single",
        "artist_tag",
        "lora_single",
        "lora_mixture",
        "artist_tag",
        "lora_single",
        "lora_mixture",
    ]


def test_fresh_direct_kv_bootstrap_jointly_uses_lora_and_native_teachers():
    assert [
        scheduled_teacher_category(
            step,
            single_only_steps=4,
            bootstrap_schedule=("lora_single", "artist_tag"),
            schedule=("artist_tag", "lora_single", "lora_mixture"),
        )
        for step in range(1, 9)
    ] == [
        "lora_single",
        "artist_tag",
        "lora_single",
        "artist_tag",
        "artist_tag",
        "lora_single",
        "lora_mixture",
        "artist_tag",
    ]


def test_continuation_schedule_emphasizes_lora_and_human_references():
    assert [
        scheduled_teacher_category(
            step,
            single_only_steps=0,
            schedule=(
                "lora_single",
                "lora_single",
                "lora_mixture",
                "artist_tag",
            ),
        )
        for step in range(1, 9)
    ] == [
        "lora_single",
        "lora_single",
        "lora_mixture",
        "artist_tag",
        "lora_single",
        "lora_single",
        "lora_mixture",
        "artist_tag",
    ]
    assert [
        scheduled_reference_domain(step, ("human", "human", "synthetic"))
        for step in range(1, 7)
    ] == ["human", "human", "synthetic", "human", "human", "synthetic"]


def test_pooling_reader_scope_preserves_per_reference_encoder():
    reader = DetailPreservingTypedSlotReader(
        dim=32,
        heads=4,
        reader_ff_dim=64,
        mixer_ff_dim=64,
        strict_v1=True,
    )
    selected = _configure_reader_trainable_scope(reader, "pooling")
    selected_ids = {id(parameter) for parameter in selected}
    names = {
        name
        for name, parameter in reader.named_parameters()
        if id(parameter) in selected_ids
    }
    assert names
    assert any(name.startswith("set_attention.") for name in names)
    assert any(name.startswith("mixers.") for name in names)
    assert not any(name.startswith("reader.") for name in names)
    assert not any(name.startswith("input_projections.") for name in names)


def test_fresh_adapter_uses_measured_block_timestep_strength(tmp_path):
    profile = {
        "timestep_bin_edges": [0.0, 0.5, 1.0],
        "teacher_to_centered_raw_ratio_by_timestep_bin": [
            [0.02, 0.04],
            [0.03, 0.06],
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    adapter = SeparatedCommonArtistKVStyleCrossAttention(
        context_dim=8,
        blocks=2,
        shared_bases=1,
        medoid_blocks=[0],
        block_to_base=[0, 0],
        delta_rank=2,
    )

    summary = _initialize_fresh_adapter_strength(
        adapter,
        {"initial_strength_profile": "profile.json"},
        tmp_path,
    )

    torch.testing.assert_close(
        adapter.alpha_by_timestep,
        torch.tensor([[0.02, 0.04], [0.03, 0.06]]),
    )
    assert adapter._timestep_strength_active
    assert "median=" in summary


def test_decomposed_objective_penalizes_common_output_collapse():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.2
    exact = teacher.clone().requires_grad_(True)
    weights = {"functional_infonce": 0.0, "common_ratio_margin": 0.0}
    exact_loss, exact_metrics = _teacher_decomposed_functional_objective(
        exact, teacher, weights
    )
    collapsed = teacher.mean(dim=0, keepdim=True).expand_as(teacher).clone()
    collapsed.requires_grad_(True)
    collapsed_loss, collapsed_metrics = _teacher_decomposed_functional_objective(
        collapsed, teacher, weights
    )

    assert float(exact_loss) < 1e-6
    assert float(exact_metrics["centered_cosine"]) > 0.999
    assert float(collapsed_loss) > float(exact_loss) + 0.5
    assert float(collapsed_metrics["common_output_excess"]) > 0
    assert float(collapsed_metrics["centered_student_to_teacher_rms"]) < 1e-5
    collapsed_loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_functional_infonce_identifies_matching_centered_teacher():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.2
    _, metrics = _teacher_decomposed_functional_objective(
        teacher, teacher, {"functional_infonce": 1.0}
    )
    assert float(metrics["functional_infonce_accuracy"]) == 1.0
    assert float(metrics["functional_infonce_cosine_gap"]) > 1.0


def test_cached_probe_preserves_exact_latent_shape_and_values(tmp_path):
    destination = tmp_path
    latent_root = destination / "latents"
    text_root = destination / "text"
    latent_root.mkdir()
    text_root.mkdir()
    first = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
    second = first + 100
    save_file({"latents": torch.stack([first, second])}, latent_root / "part.safetensors")
    write_records(
        latent_root / "manifest.parquet",
        [
            {
                "id": index + 1,
                "artist": f"artist_{index}",
                "style_id": f"style_{index}",
                "split": "train",
                "latent_height": 3,
                "latent_width": 4,
                "row_index": index,
                "cache_shard": "part.safetensors",
            }
            for index in range(2)
        ],
    )
    conditioning = torch.arange(40, dtype=torch.float16).reshape(5, 8)
    save_file({"conditioning": conditioning}, text_root / "part.safetensors")
    write_records(
        text_root / "manifest.parquet",
        [
            {
                "id": index + 1,
                "artist": f"artist_{index}",
                "style_id": f"style_{index}",
                "variant_name": "full",
                "caption": f"caption {index}",
                "token_offset": index * 2,
                "token_length": 2,
                "cache_shard": "part.safetensors",
            }
            for index in range(2)
        ],
    )
    latents, contexts, rows = _cached_training_probe_bank(
        destination,
        {
            "latent_cache_directory": "latents",
            "text_cache_directory": "text",
            "probe_latent_shape": [3, 4],
            "content_variants": ["full"],
            "text_conditioning_length": 3,
            "seed": 4,
        },
        2,
    )

    assert latents.shape == (2, 2, 3, 4)
    assert {tuple(value.flatten().tolist()) for value in latents} == {
        tuple(first.flatten().tolist()),
        tuple(second.flatten().tolist()),
    }
    assert contexts.shape == (2, 3, 8)
    assert all(row["latent_transform"] == "none" for row in rows)
