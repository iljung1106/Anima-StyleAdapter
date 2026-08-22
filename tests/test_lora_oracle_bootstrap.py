import torch

from anima_style_data.lora_oracle_bootstrap import (
    _artist_centered_oracle_objective,
)


def test_oracle_objective_prefers_distinct_centered_effects():
    teacher = torch.eye(8).reshape(8, 1, 8) + 0.3
    exact = teacher - teacher.mean(dim=0, keepdim=True)
    exact_loss, exact_metrics = _artist_centered_oracle_objective(
        exact, teacher, {"infonce": 0.5}
    )
    collapsed = torch.zeros_like(exact, requires_grad=True)
    collapsed_loss, collapsed_metrics = _artist_centered_oracle_objective(
        collapsed, teacher, {"infonce": 0.5}
    )

    assert float(exact_metrics["centered_cosine"]) > 0.999
    assert float(exact_metrics["functional_infonce_accuracy"]) == 1.0
    assert float(collapsed_loss) > float(exact_loss) + 1.0
    assert float(collapsed_metrics["centered_student_to_teacher_rms"]) < 1e-5
    collapsed_loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_oracle_objective_penalizes_artist_branch_batch_mean():
    teacher = torch.eye(4).reshape(4, 1, 4)
    centered = teacher - teacher.mean(dim=0, keepdim=True)
    shifted = centered + 0.5
    centered_loss, centered_metrics = _artist_centered_oracle_objective(
        centered, teacher, {"infonce": 0.0}
    )
    shifted_loss, shifted_metrics = _artist_centered_oracle_objective(
        shifted, teacher, {"infonce": 0.0}
    )

    assert float(centered_metrics["artist_common_zero_loss"]) < 1e-8
    assert float(shifted_metrics["artist_common_zero_loss"]) > 0.1
    assert float(shifted_loss) > float(centered_loss)
