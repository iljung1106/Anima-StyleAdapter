import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from anima_style_data.lora_functional_distillation import (
    FunctionalLoRATeacherBank,
    MixtureSpec,
    _artist_only_fixed_population_objective,
    _configure_reader_trainable_scope,
    _cached_training_probe_bank,
    _initialize_fresh_adapter_strength,
    _pack_materialized_mixture_references,
    _fewshot_prompt_signature,
    _select_fewshot_validation_styles,
    _separated_component_functional_objective,
    _teacher_decomposed_functional_objective,
    build_mixture_specs,
    _functional_teacher_specs,
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
from anima_style_data.detail_style_training import (
    _lora_backward_scale_for_step,
    _lora_teacher_schedule_for_step,
)
from anima_style_data.io import write_records


def test_teacher_bank_filters_effect_kinds_and_reuses_population_mean(tmp_path):
    save_file(
        {
            "noisy_inputs": torch.zeros(1, 1, 1),
            "base_predictions": torch.zeros(1, 1, 1),
            "base_context": torch.zeros(1, 1, 1),
            "timesteps": torch.zeros(1),
        },
        tmp_path / "base.safetensors",
    )
    write_records(
        tmp_path / "mixtures.parquet",
        [
            {"index": 0, "kind": "single", "enabled": True},
            {"index": 1, "kind": "pair", "enabled": True},
            {"index": 2, "kind": "single", "enabled": True},
        ],
    )
    save_file(
        {
            "effects": torch.tensor([1.0, 50.0, 5.0]).reshape(3, 1, 1, 1),
            "mixture_indices": torch.tensor([0, 1, 2]),
        },
        tmp_path / "effects-00000.safetensors",
    )

    bank = FunctionalLoRATeacherBank(tmp_path, load_kinds={"single"})

    assert bank.effect_indices == [0, 2]
    assert bank.by_kind["pair"] == []
    assert torch.equal(bank.effect_rows([2, 0], 0, 0).flatten(), torch.tensor([5.0, 1.0]))
    assert torch.equal(bank.single_population_mean.flatten(), torch.tensor([3.0]))
    assert (tmp_path / "single_population_mean.safetensors").exists()

    save_file(
        {
            "effects": torch.tensor([11.0, 50.0, 15.0]).reshape(3, 1, 1, 1),
            "mixture_indices": torch.tensor([0, 1, 2]),
        },
        tmp_path / "effects-00000.safetensors",
    )
    reused = FunctionalLoRATeacherBank(tmp_path, load_kinds={"single"})
    assert torch.equal(reused.single_population_mean.flatten(), torch.tensor([3.0]))

    lazy = FunctionalLoRATeacherBank(
        tmp_path, load_kinds={"single"}, effect_slice_lru_entries=1
    )
    assert lazy.effects is None
    assert torch.equal(
        lazy.effect_rows([2, 0], 0, 0).flatten(),
        torch.tensor([15.0, 11.0]),
    )
    assert len(lazy._effect_shards) == 1


def test_artist_only_population_objective_anchors_common_offset_and_spread():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.3
    population = teacher.mean(dim=0)
    residual = teacher - population
    weights = {
        "pair_huber": 1.0,
        "pair_direction": 0.75,
        "pair_magnitude": 0.25,
        "absolute_anchor": 0.10,
        "functional_infonce": 0.0,
        "population_common_beta": 0.0,
    }

    exact_loss, exact_metrics = _artist_only_fixed_population_objective(
        residual, teacher, population, weights
    )
    leaked_loss, _ = _artist_only_fixed_population_objective(
        residual + 0.5, teacher, population, weights
    )
    collapsed_loss, collapsed_metrics = _artist_only_fixed_population_objective(
        torch.zeros_like(residual), teacher, population, weights
    )

    assert float(exact_loss) < 1e-6
    assert float(leaked_loss) > float(exact_loss)
    assert float(collapsed_loss) > float(exact_loss) + 0.5
    assert float(exact_metrics["pair_cosine"]) > 0.999
    assert float(collapsed_metrics["pair_student_to_teacher_rms"]) < 1e-5


def test_artist_only_population_objective_scales_common_for_amplification():
    population = torch.full((1, 3), 0.4)
    coefficient_sums = torch.tensor([1.0, 1.2, 1.5]).reshape(3, 1, 1)
    residual = torch.eye(3).reshape(3, 1, 3)
    teacher = residual + coefficient_sums * population
    per_row_population = coefficient_sums * population
    weights = {
        "pair_huber": 1.0,
        "pair_direction": 0.75,
        "pair_magnitude": 0.25,
        "absolute_anchor": 0.10,
        "functional_infonce": 0.0,
    }

    exact_loss, _ = _artist_only_fixed_population_objective(
        residual, teacher, per_row_population, weights
    )
    unscaled_loss, _ = _artist_only_fixed_population_objective(
        residual, teacher, population, weights
    )

    assert float(exact_loss) < 1e-6
    assert float(unscaled_loss) > float(exact_loss)


def test_detail_lora_teacher_curriculum_opens_bounded_mixtures_in_stages():
    training = {
        "single_only_steps": 0,
        "teacher_kind_curriculum": [
            {"end_step": 250, "schedule": ["single", "pair"]},
            {"end_step": 500, "schedule": ["single", "pair", "triple"]},
        ],
    }
    default = ("single", "pair", "triple", "amplified", "signed")

    assert _lora_teacher_schedule_for_step(training, 100, default) == (
        "single", "pair"
    )
    assert _lora_teacher_schedule_for_step(training, 400, default) == (
        "single", "pair", "triple"
    )
    assert _lora_teacher_schedule_for_step(training, 700, default) == default


def test_detail_lora_backward_scale_ramps_without_teacher_staging():
    training = {
        "backward_scale_start": 0.10,
        "backward_scale": 0.50,
        "backward_scale_ramp_steps": 250,
    }

    assert _lora_backward_scale_for_step(training, 1) == 0.10
    assert 0.29 < _lora_backward_scale_for_step(training, 125) < 0.31
    assert _lora_backward_scale_for_step(training, 250) == 0.50
    assert _lora_backward_scale_for_step(training, 1000) == 0.50


def test_fourfold_absolute_anchor_fourfolds_common_offset_penalty():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.3
    population = teacher.mean(dim=0)
    residual = teacher - population
    common_offset_student = residual + 0.5
    weights = {
        "pair_huber": 1.0,
        "pair_direction": 0.75,
        "pair_magnitude": 0.25,
        "absolute_anchor": 0.10,
        "functional_infonce": 0.0,
        "population_common_beta": 0.0,
    }
    previous, _ = _artist_only_fixed_population_objective(
        common_offset_student, teacher, population, weights
    )
    weights["absolute_anchor"] = 0.40
    strengthened, _ = _artist_only_fixed_population_objective(
        common_offset_student, teacher, population, weights
    )

    assert torch.isclose(strengthened, 4.0 * previous)


def test_population_mean_anchor_targets_shared_error_not_sampled_artist_mean():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.3
    population = teacher.mean(dim=0)
    residual = teacher - population
    weights = {
        "pair_huber": 0.0,
        "pair_direction": 0.0,
        "pair_magnitude": 0.0,
        "absolute_anchor": 0.0,
        "population_mean_anchor": 1.0,
        "functional_infonce": 0.0,
    }

    exact, exact_metrics = _artist_only_fixed_population_objective(
        residual, teacher, population, weights
    )
    shifted, shifted_metrics = _artist_only_fixed_population_objective(
        residual + 0.5, teacher, population, weights
    )

    assert float(exact) < 1e-7
    assert float(exact_metrics["population_mean_error_to_teacher_rms"]) < 1e-7
    assert float(shifted) > 0.0
    assert float(shifted_metrics["population_mean_error_to_teacher_rms"]) > 0.0


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


def test_mixture_plan_supports_bounded_amplification_and_signed_extrapolation():
    specs = build_mixture_specs(
        32,
        pair_count=4,
        triple_count=4,
        amplified_count=8,
        signed_count=8,
        seed=29,
    )
    amplified = [value for value in specs if value.kind == "amplified"]
    signed = [value for value in specs if value.kind == "signed"]
    assert len(amplified) == len(signed) == 8
    assert all(1.05 <= sum(value.weights) <= 1.35 for value in amplified)
    assert all(min(value.weights) > 0 for value in amplified)
    assert all(abs(sum(value.weights) - 1.0) < 1e-7 for value in signed)
    assert all(-0.25 <= min(value.weights) <= -0.05 for value in signed)
    assert all(
        sum(abs(weight) for weight in value.weights) <= 1.5
        for value in signed
    )


def test_diverse_mixture_plan_supports_signed_triples_with_l1_cap():
    specs = build_mixture_specs(
        32,
        pair_count=0,
        triple_count=0,
        amplified_count=4,
        signed_count=8,
        amplified_sum_range=(1.0, 1.7),
        amplified_triple_probability=1.0,
        signed_beta_range=(0.05, 0.35),
        signed_triple_probability=1.0,
        signed_l1_maximum=1.7,
        seed=31,
    )
    amplified = [value for value in specs if value.kind == "amplified"]
    signed = [value for value in specs if value.kind == "signed"]
    assert all(len(value.components) == 3 for value in amplified + signed)
    assert all(1.0 <= sum(value.weights) <= 1.7 for value in amplified)
    assert all(abs(sum(value.weights) - 1.0) < 1e-7 for value in signed)
    assert all(sum(weight < 0 for weight in value.weights) == 1 for value in signed)
    assert all(
        sum(abs(weight) for weight in value.weights) <= 1.7 + 1e-7
        for value in signed
    )


def test_materialized_signed_mixture_never_passes_coefficients_to_reader():
    class Loader:
        def load_styles(self, style_ids, *, references_per_style, seed):
            del seed
            return {
                "tokens": torch.randn(
                    len(style_ids), references_per_style, 84, 1024
                )
            }

    rows = [{
        "mixture_style_id": "lora-mixture-00001",
        "weights": [1.2, -0.2],
    }]
    tokens, mask, weights = _pack_materialized_mixture_references(
        Loader(),
        rows,
        references_per_mixture=2,
        seed=7,
        device="cpu",
    )
    assert tokens.shape == (1, 2, 84, 1024)
    assert mask.all()
    torch.testing.assert_close(weights, torch.full((1, 2), 0.5))


def test_fewshot_prompt_signature_preserves_fixed_prompt_seed_contract():
    signature = _fewshot_prompt_signature({
        "negative_prompt": "bad",
        "prompt_cases": [
            {"name": "portrait", "prompt": "1girl", "seed": 11},
            {"name": "action", "prompt": "running", "seed": 12},
        ],
    })
    assert [row["name"] for row in signature["prompt_cases"]] == [
        "portrait",
        "action",
    ]
    assert [row["seed"] for row in signature["prompt_cases"]] == [11, 12]


def test_fewshot_validation_selects_only_artist_disjoint_eligible_styles():
    rows = []
    for style_id, count, split in (
        ("train", 8, "train"),
        ("short", 3, "validation"),
        ("valid_a", 8, "validation"),
        ("valid_b", 9, "validation"),
        ("valid_c", 10, "validation"),
    ):
        rows.extend(
            {"style_id": style_id, "split": split, "id": index}
            for index in range(count)
        )
    selected = _select_fewshot_validation_styles(
        rows, split="validation", artists=2, references=8, seed=7
    )
    assert len(selected) == 2
    assert set(selected) <= {"valid_a", "valid_b", "valid_c"}


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


def test_separated_component_objective_penalizes_artist_common_leakage():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.2
    teacher_common, teacher_centered = decompose_teacher_effects(teacher)
    weights = {
        "centered_huber": 1.0,
        "centered_direction": 0.0,
        "centered_magnitude": 0.0,
        "functional_infonce": 0.0,
        "artist_common_leakage": 1.0,
        "common_huber": 1.0,
        "common_direction": 0.0,
        "common_magnitude": 0.0,
    }
    exact_combined = teacher_common + teacher_centered
    exact_loss, exact_metrics = _separated_component_functional_objective(
        teacher_common, exact_combined, teacher, weights
    )
    leaked = torch.full_like(teacher_centered, 0.5)
    common_student = teacher_common.clone().requires_grad_(True)
    leaked_combined = (
        teacher_common + teacher_centered + leaked
    ).clone().requires_grad_(True)
    leaked_loss, leaked_metrics = _separated_component_functional_objective(
        common_student, leaked_combined, teacher, weights
    )

    assert float(exact_loss) < 1e-6
    assert float(exact_metrics["artist_common_leakage_loss"]) < 1e-6
    assert float(leaked_loss) > float(exact_loss) + 0.1
    assert float(leaked_metrics["artist_common_leakage_loss"]) > 0.1
    # Leakage is isolated from the correctly matched centered Artist effect.
    assert float(leaked_metrics["centered_cosine"]) > 0.999
    leaked_loss.backward()
    torch.testing.assert_close(common_student.grad, torch.zeros_like(common_student))
    assert float(leaked_combined.grad.abs().sum()) > 0


def test_separated_component_objective_rejects_zero_artist_shortcut():
    teacher = torch.eye(4).reshape(4, 1, 4) + 0.2
    teacher_common, _ = decompose_teacher_effects(teacher)
    collapsed = teacher_common.expand_as(teacher).clone().requires_grad_(True)
    loss, metrics = _separated_component_functional_objective(
        teacher_common,
        collapsed,
        teacher,
        {
            "functional_infonce": 0.5,
            "artist_common_leakage": 1.0,
            "artist_magnitude_floor": 1.0,
            "artist_magnitude_floor_ratio": 0.75,
        },
    )

    assert float(metrics["centered_student_to_teacher_rms"]) < 1e-5
    assert float(metrics["artist_magnitude_floor_loss"]) > 0.5
    assert float(metrics["artist_magnitude_floor_violation_fraction"]) == 1.0
    assert float(loss) > 0.5
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()


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
def test_functional_teacher_specs_reuse_exact_external_mixtures(tmp_path):
    plans = [SimpleNamespace(style_id=value) for value in ("a", "b", "c")]
    write_records(
        tmp_path / "mixtures.parquet",
        [
            {
                "index": 7,
                "kind": "single",
                "style_ids": ["a"],
                "weights": [1.0],
            },
            {
                "index": 11,
                "kind": "pair",
                "style_ids": ["c", "a"],
                "weights": [0.25, 0.75],
                "mixture_style_id": "exact-pair",
                "enabled": False,
            },
        ],
    )

    specs, rows = _functional_teacher_specs(
        plans, {"mixture_manifest": "mixtures.parquet"}, tmp_path
    )

    assert len(specs) == 4
    assert specs[-1] == MixtureSpec(3, "pair", (2, 0), (0.25, 0.75))
    assert rows[3]["mixture_style_id"] == "exact-pair"
