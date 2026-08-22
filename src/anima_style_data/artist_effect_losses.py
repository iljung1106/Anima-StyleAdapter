"""Artist-repeatable objectives for style tokens and functional flow effects."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _style_labels(style_ids: Sequence[str], device: torch.device) -> torch.Tensor:
    labels: dict[str, int] = {}
    values = []
    for style_id in style_ids:
        if style_id not in labels:
            labels[style_id] = len(labels)
        values.append(labels[style_id])
    return torch.tensor(values, device=device, dtype=torch.long)


def _symmetric_multi_positive_nce(
    first: torch.Tensor,
    second: torch.Tensor,
    style_ids: Sequence[str],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Contrastive views must share [batch,dim] shape")
    if len(style_ids) != first.shape[0] or first.shape[0] < 2:
        raise ValueError("Contrastive views need at least two labelled rows")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    labels = _style_labels(style_ids, first.device)
    positive = labels[:, None] == labels[None]
    first = F.normalize(first.float(), dim=1, eps=1e-8)
    second = F.normalize(second.float(), dim=1, eps=1e-8)
    logits = first @ second.T / float(temperature)

    def direction(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_probability = values.log_softmax(dim=1)
        return -torch.logsumexp(
            log_probability.masked_fill(~mask, -torch.inf), dim=1
        ).mean()

    loss = 0.5 * (
        direction(logits, positive) + direction(logits.T, positive.T)
    )
    return loss, logits * float(temperature), labels


def typewise_artist_embedding(
    tokens: torch.Tensor,
    slot_type_counts: Sequence[int] = (16, 8, 4),
) -> torch.Tensor:
    """Build a fixed, low-capacity artist summary without a classifier head."""

    if tokens.ndim != 3 or sum(int(value) for value in slot_type_counts) != tokens.shape[1]:
        raise ValueError("slot_type_counts must cover every style token")
    offsets = [0]
    for count in slot_type_counts:
        offsets.append(offsets[-1] + int(count))
    summaries = [
        tokens[:, start:end].float().mean(dim=1)
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    return F.normalize(torch.cat(summaries, dim=1), dim=1, eps=1e-8)


def episodic_artist_prototype_loss(
    first_tokens: torch.Tensor,
    second_tokens: torch.Tensor,
    style_ids: Sequence[str],
    *,
    temperature: float = 0.10,
    slot_type_counts: Sequence[int] = (16, 8, 4),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Classify one artist view using prototypes made only from the other view."""

    first = typewise_artist_embedding(first_tokens, slot_type_counts)
    second = typewise_artist_embedding(second_tokens, slot_type_counts)
    loss, similarity, labels = _symmetric_multi_positive_nce(
        first, second, style_ids, temperature
    )
    positive = labels[:, None] == labels[None]
    negative = ~positive
    positive_similarity = similarity[positive].mean()
    negative_similarity = (
        (similarity * negative).sum() / negative.sum().clamp_min(1)
    )
    predicted = similarity.argmax(dim=1)
    retrieval = (labels[predicted] == labels).float().mean()
    return loss, {
        "artist_prototype_loss": loss.detach(),
        "artist_prototype_positive_cosine": positive_similarity.detach(),
        "artist_prototype_negative_cosine": negative_similarity.detach(),
        "artist_prototype_cosine_gap": (
            positive_similarity - negative_similarity
        ).detach(),
        "artist_prototype_retrieval_top1": retrieval.detach(),
    }


def multiscale_effect_vector(
    effect: torch.Tensor,
    pool_scales: Sequence[int] = (2, 4),
) -> torch.Tensor:
    """Create a fixed low/mid-frequency sketch of a latent flow effect."""

    if effect.ndim != 4:
        raise ValueError("Flow effects must have [batch,channels,height,width] shape")
    vectors = []
    for scale in pool_scales:
        scale = int(scale)
        if scale <= 0:
            raise ValueError("pool scales must be positive")
        if scale > min(effect.shape[-2:]):
            continue
        pooled = F.avg_pool2d(effect.float(), kernel_size=scale, stride=scale)
        flattened = pooled.flatten(1)
        # Equalize scales by average energy rather than number of elements.
        vectors.append(flattened / math.sqrt(flattened.shape[1]))
    if not vectors:
        raise ValueError("No pool scale fits the flow effect")
    return torch.cat(vectors, dim=1)


def centered_functional_artist_loss(
    first_delta: torch.Tensor,
    second_delta: torch.Tensor,
    style_ids: Sequence[str],
    *,
    temperature: float = 0.10,
    pool_scales: Sequence[int] = (2, 4),
    repeatability_weight: float = 0.25,
    repeatability_floor: float | None = None,
    contrastive_mode: str = "symmetric_nce",
    positive_floor: float = 0.25,
    wrong_margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Learn the artist effect shared by two disjoint reference views.

    Both deltas must be evaluated at exactly the same noisy latent, text and
    timestep.  Centering across artists makes the objective invariant to an
    artist-independent common output.  Cross-view agreement rejects details
    that occur in only one reference image.
    """

    if first_delta.shape != second_delta.shape or first_delta.shape[0] < 2:
        raise ValueError("Functional artist views need matching artist batches")
    if repeatability_weight < 0:
        raise ValueError("repeatability_weight cannot be negative")
    if repeatability_floor is not None and not -1.0 <= repeatability_floor <= 1.0:
        raise ValueError("repeatability_floor must be between -1 and 1")
    if contrastive_mode not in {"symmetric_nce", "all_wrong_margin"}:
        raise ValueError(
            "contrastive_mode must be symmetric_nce or all_wrong_margin"
        )
    if not -1.0 <= positive_floor <= 1.0 or wrong_margin < 0:
        raise ValueError("invalid weak-positive floor or all-wrong margin")
    first_centered = first_delta.float() - first_delta.float().mean(
        dim=0, keepdim=True
    )
    second_centered = second_delta.float() - second_delta.float().mean(
        dim=0, keepdim=True
    )
    first_vector = multiscale_effect_vector(first_centered, pool_scales)
    second_vector = multiscale_effect_vector(second_centered, pool_scales)
    labels = _style_labels(style_ids, first_vector.device)
    first_normalized = F.normalize(first_vector.float(), dim=1, eps=1e-8)
    second_normalized = F.normalize(second_vector.float(), dim=1, eps=1e-8)
    similarity = first_normalized @ second_normalized.T
    positive = labels[:, None] == labels[None]
    negative = ~positive
    if contrastive_mode == "symmetric_nce":
        contrastive, similarity, labels = _symmetric_multi_positive_nce(
            first_vector, second_vector, style_ids, temperature
        )
        positive = labels[:, None] == labels[None]
        negative = ~positive
        margin_violation = contrastive.new_zeros(())
    else:
        # Match each disjoint view against every wrong artist, but stop once a
        # finite margin is satisfied. Unlike InfoNCE, this does not keep
        # contracting all works by one artist toward an invariant prototype.
        corresponding = similarity.diagonal()

        def all_wrong_margin(
            values: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            violations = F.relu(
                float(wrong_margin) + values - scores[:, None]
            )
            selected = violations[mask]
            if selected.numel() == 0:
                zero = values.new_zeros(())
                return zero, zero
            return selected.mean(), (selected > 0).float().mean()

        forward, forward_violation = all_wrong_margin(
            similarity, corresponding, negative
        )
        backward, backward_violation = all_wrong_margin(
            similarity.T, corresponding, negative.T
        )
        contrastive = 0.5 * (forward + backward)
        margin_violation = 0.5 * (forward_violation + backward_violation)
    dot = (first_vector * second_vector).sum(dim=1)
    energy = first_vector.square().sum(dim=1) + second_vector.square().sum(dim=1)
    repeatable_ratio = 2.0 * dot / energy.clamp_min(1e-8)
    if contrastive_mode == "all_wrong_margin":
        # Only require a modest corresponding-view similarity. Once reached,
        # retain the remaining image-specific style and content information.
        repeatability = F.relu(
            float(positive_floor) - similarity.diagonal()
        ).square().mean()
    elif repeatability_floor is None:
        repeatability = (1.0 - repeatable_ratio).mean()
    else:
        # Different works by one artist are not interchangeable targets.  Once
        # their coarse centered effects have a modest agreement, leave the
        # remaining reference-specific signal untouched instead of rewarding
        # an increasingly invariant (and eventually common) output.
        repeatability = F.relu(
            float(repeatability_floor) - repeatable_ratio
        ).square().mean()
    total = contrastive + float(repeatability_weight) * repeatability

    positive_similarity = similarity[positive].mean()
    negative_similarity = (
        (similarity * negative).sum() / negative.sum().clamp_min(1)
    )
    predicted = similarity.argmax(dim=1)
    retrieval = (labels[predicted] == labels).float().mean()

    artist_mean = 0.5 * (first_vector + second_vector)
    between_variance = artist_mean.square().mean()
    within_variance = 0.25 * (first_vector - second_vector).square().mean()
    functional_icc = between_variance / (
        between_variance + within_variance
    ).clamp_min(1e-8)
    average_effect = 0.5 * (first_delta.float() + second_delta.float())
    dimensions = tuple(range(1, average_effect.ndim))
    effect_rms = average_effect.square().mean(dim=dimensions).sqrt()
    common_rms = average_effect.mean(dim=0).square().mean().sqrt()
    common_ratio = common_rms / effect_rms.mean().clamp_min(1e-8)
    view_difference = (
        (first_delta.float() - second_delta.float())
        .square().mean(dim=dimensions).sqrt()
        / (
            0.5 * (
                first_delta.float().square().mean(dim=dimensions).sqrt()
                + second_delta.float().square().mean(dim=dimensions).sqrt()
            )
        ).clamp_min(1e-8)
    ).mean()
    return total, {
        "functional_artist_loss": total.detach(),
        "functional_artist_contrastive_loss": contrastive.detach(),
        "functional_artist_all_wrong_margin": contrastive.new_tensor(
            float(wrong_margin)
        ),
        "functional_artist_all_wrong_violation_fraction": (
            margin_violation.detach()
        ),
        "functional_artist_positive_floor": contrastive.new_tensor(
            float(positive_floor)
        ),
        "functional_artist_uses_symmetric_nce": contrastive.new_tensor(
            float(contrastive_mode == "symmetric_nce")
        ),
        "functional_artist_repeatability_loss": repeatability.detach(),
        "functional_artist_repeatability_floor": repeatability.new_tensor(
            -1.0 if repeatability_floor is None else float(repeatability_floor)
        ),
        "functional_artist_repeatable_ratio": repeatable_ratio.detach().mean(),
        "functional_artist_icc": functional_icc.detach(),
        "functional_artist_positive_cosine": positive_similarity.detach(),
        "functional_artist_negative_cosine": negative_similarity.detach(),
        "functional_artist_cosine_gap": (
            positive_similarity - negative_similarity
        ).detach(),
        "functional_artist_retrieval_top1": retrieval.detach(),
        "functional_artist_common_output_ratio": common_ratio.detach(),
        "functional_artist_view_difference_ratio": view_difference.detach(),
        "functional_artist_effect_rms": effect_rms.detach().mean(),
    }


def common_output_and_artist_magnitude_loss(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    *,
    common_threshold: float,
    magnitude_lower: float,
    magnitude_upper: float,
    magnitude_upper_weight: float = 0.25,
    pool_scales: Sequence[int] = (2, 4),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Suppress shared output while preserving aligned artist-effect magnitude.

    The common penalty acts on the raw heldout-reference student residual.  Its
    denominator is detached, so reducing every residual cannot lower the ratio
    merely by changing its normalization.  The magnitude band is measured only
    along the centered exact-target teacher direction; orthogonal texture noise
    therefore cannot satisfy the lower bound.
    """

    if teacher_delta.shape != student_delta.shape or teacher_delta.shape[0] < 2:
        raise ValueError("Artist magnitude views need matching artist batches")
    if not 0.0 <= common_threshold <= 1.0:
        raise ValueError("common_threshold must be between zero and one")
    if magnitude_lower < 0 or magnitude_upper < magnitude_lower:
        raise ValueError("invalid artist magnitude band")
    if magnitude_upper_weight < 0:
        raise ValueError("magnitude_upper_weight cannot be negative")

    teacher = teacher_delta.float()
    student = student_delta.float()
    dimensions = tuple(range(1, student.ndim))
    student_rms = student.square().mean(dim=dimensions).sqrt()
    common_rms = student.mean(dim=0).square().mean().sqrt()
    # Stop-gradient normalization makes this a directional common-component
    # penalty instead of an incentive to inflate unrelated residual energy.
    common_ratio = common_rms / student_rms.mean().detach().clamp_min(1e-8)
    common_loss = F.relu(common_ratio - float(common_threshold)).square()

    teacher_centered = teacher - teacher.mean(dim=0, keepdim=True)
    student_centered = student - student.mean(dim=0, keepdim=True)
    teacher_vector = multiscale_effect_vector(teacher_centered, pool_scales)
    student_vector = multiscale_effect_vector(student_centered, pool_scales)
    projection = (
        (student_vector * teacher_vector).sum(dim=1)
        / teacher_vector.square().sum(dim=1).clamp_min(1e-8)
    )
    lower_loss = F.relu(float(magnitude_lower) - projection).square().mean()
    upper_loss = F.relu(projection - float(magnitude_upper)).square().mean()
    magnitude_loss = lower_loss + float(magnitude_upper_weight) * upper_loss

    centered_student_rms = student_centered.square().mean(
        dim=dimensions
    ).sqrt().mean()
    centered_teacher_rms = teacher_centered.square().mean(
        dim=dimensions
    ).sqrt().mean()
    return common_loss, magnitude_loss, {
        "functional_artist_student_common_output_ratio": common_ratio.detach(),
        "functional_artist_common_output_loss": common_loss.detach(),
        "functional_artist_magnitude_projection": projection.detach().mean(),
        "functional_artist_magnitude_positive_fraction": (
            projection.detach() > 0
        ).float().mean(),
        "functional_artist_magnitude_lower_loss": lower_loss.detach(),
        "functional_artist_magnitude_upper_loss": upper_loss.detach(),
        "functional_artist_magnitude_loss": magnitude_loss.detach(),
        "functional_artist_centered_student_rms": centered_student_rms.detach(),
        "functional_artist_centered_teacher_rms": centered_teacher_rms.detach(),
        "functional_artist_centered_student_to_teacher_rms": (
            centered_student_rms / centered_teacher_rms.clamp_min(1e-8)
        ).detach(),
    }
