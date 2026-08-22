import torch

from anima_style_data.lora_functional_distillation import (
    build_mixture_specs,
    decompose_teacher_effects,
    teacher_category,
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
