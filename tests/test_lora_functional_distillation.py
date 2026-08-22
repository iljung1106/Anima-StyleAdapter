import torch

from anima_style_data.lora_functional_distillation import (
    _teacher_decomposed_functional_objective,
    build_mixture_specs,
    decompose_teacher_effects,
    teacher_category,
    teacher_category_v2,
)


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
