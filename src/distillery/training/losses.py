"""Exact full-vocabulary forward-KL and CE mixing (pure Python reference math).

This module has no torch/transformers dependency so the base package imports
without ML extras. Optional torch kernels live in ``training.torch_losses`` and
must not be imported from package ``__init__``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("temperature must be a finite real number > 0")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be a finite real number > 0")
    return value


def _finite_row(values: Sequence[float], *, name: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise ValueError(f"{name} must be a one-dimensional numeric sequence")
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    row: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a finite real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name}[{index}] must be finite")
        row.append(converted)
    return row


def _validated_2d_logits(
    logits: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[list[list[float]], int, int]:
    if isinstance(logits, (str, bytes)) or not hasattr(logits, "__len__"):
        raise ValueError(f"{name} must be a two-dimensional numeric sequence")
    if len(logits) == 0:
        raise ValueError(f"{name} must be non-empty")
    n_pos = len(logits)
    rows: list[list[float]] = []
    vocab: int | None = None
    for index, raw_row in enumerate(logits):
        row = _finite_row(raw_row, name=f"{name}[{index}]")
        if vocab is None:
            vocab = len(row)
        elif len(row) != vocab:
            raise ValueError(f"{name} has ragged vocabulary dimension")
        rows.append(row)
    if vocab is None or vocab < 1:
        raise ValueError(f"{name} vocabulary dimension must be >= 1")
    return rows, n_pos, vocab


def validate_binary_mask(
    position_mask: Sequence[float] | None,
    *,
    expected_length: int,
    name: str = "position_mask",
) -> list[float]:
    """Return a finite binary mask, rejecting weighted or malformed masks."""
    if (
        isinstance(expected_length, bool)
        or not isinstance(expected_length, Integral)
        or expected_length < 1
    ):
        raise ValueError("expected_length must be an integer >= 1")
    if position_mask is None:
        return [1.0] * expected_length
    if isinstance(position_mask, (str, bytes)) or not hasattr(position_mask, "__len__"):
        raise ValueError(f"{name} must be a one-dimensional binary sequence")
    if len(position_mask) != expected_length:
        raise ValueError(f"{name} length must match positions")
    mask: list[float] = []
    for index, value in enumerate(position_mask):
        if isinstance(value, bool):
            converted = float(value)
        elif isinstance(value, Real):
            converted = float(value)
        else:
            raise ValueError(f"{name}[{index}] must be binary (0 or 1)")
        if not math.isfinite(converted) or converted not in {0.0, 1.0}:
            raise ValueError(f"{name}[{index}] must be binary (0 or 1)")
        mask.append(converted)
    return mask


def validate_target_ids(
    target_ids: Sequence[int],
    *,
    expected_length: int,
    vocab_size: int,
    position_mask: Sequence[float] | None = None,
    ignore_index: int = -100,
    require_mask_label_alignment: bool = True,
) -> tuple[list[int], list[float]]:
    """Validate integer labels and completion-mask/ignore-index alignment."""
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise ValueError("ignore_index must be an integer")
    if (
        isinstance(expected_length, bool)
        or not isinstance(expected_length, Integral)
        or expected_length < 1
    ):
        raise ValueError("expected_length must be an integer >= 1")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, Integral)
        or vocab_size < 1
    ):
        raise ValueError("vocab_size must be an integer >= 1")
    if isinstance(target_ids, (str, bytes)) or not hasattr(target_ids, "__len__"):
        raise ValueError("target_ids must be a one-dimensional integer sequence")
    if len(target_ids) != expected_length:
        raise ValueError("target_ids length must match positions")
    mask = validate_binary_mask(position_mask, expected_length=expected_length)
    targets: list[int] = []
    for index, target in enumerate(target_ids):
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise ValueError(f"target_ids[{index}] must be an integer")
        converted = int(target)
        is_ignored = converted == ignore_index
        if require_mask_label_alignment and is_ignored != (mask[index] == 0.0):
            raise ValueError(
                "target/mask alignment requires ignore_index exactly at masked positions"
            )
        if not is_ignored and not 0 <= converted < vocab_size:
            raise ValueError(
                f"target id {converted} out of range for vocab {vocab_size}"
            )
        targets.append(converted)
    return targets, mask


def softmax(values: Sequence[float], *, temperature: float = 1.0) -> list[float]:
    return [math.exp(value) for value in log_softmax(values, temperature=temperature)]


def log_softmax(values: Sequence[float], *, temperature: float = 1.0) -> list[float]:
    """Stable log-softmax with no probability clamp or information loss."""
    checked_temperature = _validate_temperature(temperature)
    row = _finite_row(values, name="values")
    scaled = [value / checked_temperature for value in row]
    log_normalizer = _logsumexp(scaled)
    return [value - log_normalizer for value in scaled]


def forward_kl_distribution(
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    *,
    temperature: float,
) -> float:
    """KL(softmax(teacher/T) || softmax(student/T)) for one position."""
    checked_temperature = _validate_temperature(temperature)
    teacher = _finite_row(teacher_logits, name="teacher_logits")
    student = _finite_row(student_logits, name="student_logits")
    if len(teacher) != len(student):
        raise ValueError("teacher and student vocab sizes must match")
    log_p = log_softmax(teacher, temperature=checked_temperature)
    log_q = log_softmax(student, temperature=checked_temperature)
    # KL(p || q) = sum p * (log p - log q)
    return sum(
        math.exp(lpi) * (lpi - lqi)
        for lpi, lqi in zip(log_p, log_q, strict=True)
    )


def forward_kl_full(
    teacher_logits: Sequence[Sequence[float]],
    student_logits: Sequence[Sequence[float]],
    *,
    temperature: float,
    position_mask: Sequence[float] | None = None,
) -> float:
    """
    Mean forward KL over masked output positions (unscaled by T^2).

    Classical KD often multiplies by T^2 when combining with CE; callers apply
    that factor explicitly in ``mixed_kd_ce_loss``.
    """
    checked_temperature = _validate_temperature(temperature)
    teacher, n_pos, vocab = _validated_2d_logits(
        teacher_logits, name="teacher_logits"
    )
    student, n_pos_s, vocab_s = _validated_2d_logits(
        student_logits, name="student_logits"
    )
    if n_pos != n_pos_s:
        raise ValueError("teacher/student position counts must match")
    if vocab != vocab_s:
        raise ValueError("teacher/student vocabulary sizes must match")
    mask = validate_binary_mask(position_mask, expected_length=n_pos)
    if not any(mask):
        raise ValueError("position_mask must select at least one output position")

    total = 0.0
    weight = 0.0
    for i in range(n_pos):
        w = float(mask[i])
        if w == 0.0:
            continue
        total += w * forward_kl_distribution(
            teacher[i],
            student[i],
            temperature=checked_temperature,
        )
        weight += w
    if weight == 0.0:
        return 0.0
    return total / weight


def _logsumexp(values: Sequence[float]) -> float:
    row = _finite_row(values, name="logsumexp_values")
    max_v = max(row)
    return max_v + math.log(math.fsum(math.exp(v - max_v) for v in row))


def forward_kl_chunked(
    teacher_logits: Sequence[Sequence[float]],
    student_logits: Sequence[Sequence[float]],
    *,
    temperature: float,
    vocab_chunk_size: int,
    position_mask: Sequence[float] | None = None,
) -> float:
    """
    Exact full-vocabulary forward KL computed in vocabulary chunks.

    Streams over vocabulary slices so full teacher and student logit tensors need
    not be materialized simultaneously in a GPU implementation. Numerically
    matches ``forward_kl_full`` for any chunk size that divides or covers vocab.
    """
    if isinstance(vocab_chunk_size, bool) or not isinstance(vocab_chunk_size, Integral):
        raise ValueError("vocab_chunk_size must be an integer >= 1")
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be >= 1")
    checked_temperature = _validate_temperature(temperature)
    teacher, n_pos, vocab = _validated_2d_logits(
        teacher_logits, name="teacher_logits"
    )
    student, n_pos_s, vocab_s = _validated_2d_logits(
        student_logits, name="student_logits"
    )
    if n_pos != n_pos_s or vocab != vocab_s:
        raise ValueError("teacher/student shapes must match")
    mask = validate_binary_mask(position_mask, expected_length=n_pos)
    if not any(mask):
        raise ValueError("position_mask must select at least one output position")

    total = 0.0
    weight = 0.0
    for i in range(n_pos):
        w = float(mask[i])
        if w == 0.0:
            continue
        t_row = [v / checked_temperature for v in teacher[i]]
        s_row = [v / checked_temperature for v in student[i]]

        # Streaming log-normalizers over vocabulary chunks.
        t_lse_chunks: list[float] = []
        s_lse_chunks: list[float] = []
        for start in range(0, vocab, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab)
            t_lse_chunks.append(_logsumexp(t_row[start:end]))
            s_lse_chunks.append(_logsumexp(s_row[start:end]))
        t_log_z = _logsumexp(t_lse_chunks)
        s_log_z = _logsumexp(s_lse_chunks)

        kl = 0.0
        for start in range(0, vocab, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab)
            for j in range(start, end):
                log_p = t_row[j] - t_log_z
                log_q = s_row[j] - s_log_z
                p = math.exp(log_p)
                kl += p * (log_p - log_q)
        total += w * kl
        weight += w

    if weight == 0.0:
        return 0.0
    return total / weight


def hard_cross_entropy(
    student_logits: Sequence[Sequence[float]],
    target_ids: Sequence[int],
    *,
    position_mask: Sequence[float] | None = None,
    ignore_index: int = -100,
) -> float:
    """Mean hard-target CE over masked positions (temperature = 1)."""
    student, n_pos, vocab = _validated_2d_logits(
        student_logits, name="student_logits"
    )
    targets, mask = validate_target_ids(
        target_ids,
        expected_length=n_pos,
        vocab_size=vocab,
        position_mask=position_mask,
        ignore_index=ignore_index,
        require_mask_label_alignment=position_mask is not None,
    )
    if not any(mask):
        raise ValueError("position_mask must select at least one output target")

    total = 0.0
    weight = 0.0
    for i in range(n_pos):
        w = float(mask[i])
        if w == 0.0:
            continue
        tid = targets[i]
        if tid == ignore_index:
            raise ValueError("active target cannot equal ignore_index")
        log_probs = log_softmax(student[i], temperature=1.0)
        total += w * (-log_probs[tid])
        weight += w
    if weight == 0.0:
        return 0.0
    return total / weight


def mixed_kd_ce_loss(
    teacher_logits: Sequence[Sequence[float]],
    student_logits: Sequence[Sequence[float]],
    target_ids: Sequence[int],
    *,
    temperature: float,
    kd_weight: float,
    hard_ce_weight: float,
    position_mask: Sequence[float] | None = None,
    vocab_chunk_size: int | None = None,
    scale_kd_by_temperature_squared: bool = True,
    ignore_index: int = -100,
) -> dict[str, float]:
    """
    Hinton-style mixture: ``kd_weight * T^2 * KL + hard_ce_weight * CE``.

    When ``vocab_chunk_size`` is set, KL uses the chunked exact implementation.
    """
    checked_temperature = _validate_temperature(temperature)
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(float(weight))
        for weight in (kd_weight, hard_ce_weight)
    ):
        raise ValueError("loss weights must be finite real numbers")
    if abs(float(kd_weight) + float(hard_ce_weight) - 1.0) > 1e-9:
        raise ValueError("kd_weight + hard_ce_weight must equal 1.0")
    if kd_weight < 0.0 or hard_ce_weight < 0.0:
        raise ValueError("loss weights must be non-negative")

    if vocab_chunk_size is None:
        kl = forward_kl_full(
            teacher_logits,
            student_logits,
            temperature=checked_temperature,
            position_mask=position_mask,
        )
    else:
        kl = forward_kl_chunked(
            teacher_logits,
            student_logits,
            temperature=checked_temperature,
            vocab_chunk_size=vocab_chunk_size,
            position_mask=position_mask,
        )
    ce = hard_cross_entropy(
        student_logits,
        target_ids,
        position_mask=position_mask,
        ignore_index=ignore_index,
    )
    kd_term = (
        kl * (checked_temperature * checked_temperature)
        if scale_kd_by_temperature_squared
        else kl
    )
    total = kd_weight * kd_term + hard_ce_weight * ce
    return {
        "loss": total,
        "kl": kl,
        "ce": ce,
        "kd_term": kd_term,
        "kd_weight": kd_weight,
        "hard_ce_weight": hard_ce_weight,
        "temperature": checked_temperature,
    }


def apply_output_position_mask(
    values: Sequence[float],
    position_mask: Sequence[float],
) -> list[float]:
    """Elementwise mask application for loss diagnostics."""
    checked_values = _finite_row(values, name="values")
    mask = validate_binary_mask(position_mask, expected_length=len(checked_values))
    if len(checked_values) != len(mask):
        raise ValueError("values and position_mask lengths must match")
    return [
        value * mask_value
        for value, mask_value in zip(checked_values, mask, strict=True)
    ]


def masked_mean(values: Sequence[float], position_mask: Sequence[float]) -> float:
    mask = validate_binary_mask(position_mask, expected_length=len(values))
    masked = apply_output_position_mask(values, mask)
    weight = sum(mask)
    if weight == 0.0:
        raise ValueError("position_mask must select at least one value")
    return sum(masked) / weight
