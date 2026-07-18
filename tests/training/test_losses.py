"""Pure-Python reference tests for exact forward KL and CE mixing."""

from __future__ import annotations

import math

import pytest

from distillery.training.losses import (
    forward_kl_chunked,
    forward_kl_distribution,
    forward_kl_full,
    hard_cross_entropy,
    log_softmax,
    mixed_kd_ce_loss,
    softmax,
)


def _almost_equal(a: float, b: float, *, tol: float = 1e-9) -> None:
    assert abs(a - b) <= tol, f"{a} != {b} (tol={tol})"


def test_softmax_temperature_flattens() -> None:
    logits = [2.0, 1.0, 0.1]
    sharp = softmax(logits, temperature=0.5)
    soft = softmax(logits, temperature=5.0)
    assert max(sharp) > max(soft)
    _almost_equal(sum(sharp), 1.0)
    _almost_equal(sum(soft), 1.0)


def test_forward_kl_zero_when_identical() -> None:
    logits = [[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]]
    kl = forward_kl_full(logits, logits, temperature=2.0)
    _almost_equal(kl, 0.0, tol=1e-12)


def test_forward_kl_positive_when_different() -> None:
    teacher = [[3.0, 0.0, 0.0]]
    student = [[0.0, 0.0, 3.0]]
    kl = forward_kl_full(teacher, student, temperature=1.0)
    assert kl > 0.0


def test_chunked_kl_matches_full_for_many_chunk_sizes() -> None:
    teacher = [
        [0.2, -0.1, 1.5, 0.0, 2.2, -1.0, 0.7, 0.3],
        [1.1, 1.1, -0.4, 0.8, 0.0, 0.5, -0.2, 1.7],
        [-0.3, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]
    student = [
        [0.0, 0.4, 1.2, 0.2, 1.8, -0.5, 0.9, 0.1],
        [0.8, 1.3, -0.1, 0.5, 0.2, 0.4, 0.0, 1.4],
        [0.0, 1.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
    ]
    mask = [1.0, 0.0, 1.0]
    for temperature in (1.0, 2.0, 4.0):
        full = forward_kl_full(
            teacher, student, temperature=temperature, position_mask=mask
        )
        for chunk in (1, 2, 3, 5, 8, 64):
            chunked = forward_kl_chunked(
                teacher,
                student,
                temperature=temperature,
                vocab_chunk_size=chunk,
                position_mask=mask,
            )
            _almost_equal(full, chunked, tol=1e-9)


def test_extreme_gap_kl_is_not_probability_clamped() -> None:
    teacher = [[1_000.0, -1_000.0, -2_000.0]]
    student = [[-1_000.0, 1_000.0, -2_000.0]]
    for temperature in (0.5, 1.0, 2.0, 8.0):
        full = forward_kl_full(teacher, student, temperature=temperature)
        assert full == pytest.approx(2_000.0 / temperature, rel=1e-12)
        for chunk_size in (1, 2, 3, 17):
            chunked = forward_kl_chunked(
                teacher,
                student,
                temperature=temperature,
                vocab_chunk_size=chunk_size,
            )
            assert chunked == pytest.approx(full, rel=1e-12)


def test_stable_log_softmax_preserves_extreme_log_probability() -> None:
    values = log_softmax([1_000.0, -1_000.0])
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(-2_000.0)


def test_output_position_mask_excludes_prompt_positions() -> None:
    teacher = [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    student = [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]]
    # Only middle position contributes.
    masked = forward_kl_full(
        teacher, student, temperature=1.0, position_mask=[0.0, 1.0, 0.0]
    )
    single = forward_kl_distribution(teacher[1], student[1], temperature=1.0)
    _almost_equal(masked, single, tol=1e-12)


def test_hard_ce_and_mixture_endpoints() -> None:
    student = [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]
    teacher = [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]]
    targets = [1, 0]
    ce = hard_cross_entropy(student, targets)
    assert ce < 0.05

    # Pure CE endpoint
    pure_ce = mixed_kd_ce_loss(
        teacher,
        student,
        targets,
        temperature=2.0,
        kd_weight=0.0,
        hard_ce_weight=1.0,
    )
    _almost_equal(pure_ce["loss"], pure_ce["ce"], tol=1e-12)

    # Pure KD endpoint
    pure_kd = mixed_kd_ce_loss(
        teacher,
        student,
        targets,
        temperature=2.0,
        kd_weight=1.0,
        hard_ce_weight=0.0,
        vocab_chunk_size=2,
    )
    expected = pure_kd["kl"] * (2.0**2)
    _almost_equal(pure_kd["loss"], expected, tol=1e-12)


def test_mixture_is_convex_combination() -> None:
    teacher = [[1.5, -0.5, 0.25]]
    student = [[0.5, 0.5, 0.5]]
    targets = [0]
    out = mixed_kd_ce_loss(
        teacher,
        student,
        targets,
        temperature=2.0,
        kd_weight=0.7,
        hard_ce_weight=0.3,
        vocab_chunk_size=1,
    )
    expected = 0.7 * out["kd_term"] + 0.3 * out["ce"]
    _almost_equal(out["loss"], expected, tol=1e-12)


def test_kl_manual_reference_small_vocab() -> None:
    teacher = [2.0, 0.0]
    student = [0.0, 1.0]
    t = 2.0
    p = softmax(teacher, temperature=t)
    q = softmax(student, temperature=t)
    expected = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
    got = forward_kl_distribution(teacher, student, temperature=t)
    _almost_equal(got, expected, tol=1e-12)


def test_invalid_temperature_rejected() -> None:
    for temperature in (0.0, -1.0, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            softmax([1.0, 2.0], temperature=temperature)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_logits_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        forward_kl_full([[0.0, bad]], [[0.0, 1.0]], temperature=1.0)
    with pytest.raises(ValueError, match="finite"):
        forward_kl_chunked(
            [[0.0, 1.0]],
            [[0.0, bad]],
            temperature=1.0,
            vocab_chunk_size=1,
        )


@pytest.mark.parametrize(
    "teacher,student",
    [
        ([], []),
        ([[1.0], [1.0, 2.0]], [[1.0], [1.0]]),
        ([1.0, 2.0], [[1.0, 2.0]]),
        ([[1.0, 2.0]], [[1.0]]),
    ],
)
def test_malformed_logit_shapes_rejected(
    teacher: list,
    student: list,
) -> None:
    with pytest.raises(ValueError):
        forward_kl_full(teacher, student, temperature=1.0)


@pytest.mark.parametrize(
    "mask",
    [
        [1.0, 0.5],
        [1.0, -1.0],
        [1.0, math.nan],
        [1.0, math.inf],
        [1.0],
        [0.0, 0.0],
    ],
)
def test_invalid_output_masks_rejected(mask: list[float]) -> None:
    with pytest.raises(ValueError):
        forward_kl_full(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            temperature=1.0,
            position_mask=mask,
        )


def test_ce_accepts_ignore_index_only_at_masked_positions() -> None:
    logits = [[9.0, 0.0], [0.0, 9.0]]
    ce = hard_cross_entropy(
        logits,
        [-100, 1],
        position_mask=[0, 1],
        ignore_index=-100,
    )
    assert ce < 0.001
    with pytest.raises(ValueError, match="alignment"):
        hard_cross_entropy(
            logits,
            [0, 1],
            position_mask=[0, 1],
            ignore_index=-100,
        )
    with pytest.raises(ValueError, match="alignment"):
        hard_cross_entropy(
            logits,
            [-100, -100],
            position_mask=[0, 1],
            ignore_index=-100,
        )


@pytest.mark.parametrize("targets", [[2], [-1], [0.5], [True]])
def test_invalid_targets_rejected(targets: list) -> None:
    with pytest.raises(ValueError):
        hard_cross_entropy([[1.0, 0.0]], targets)


def test_nonfinite_loss_weights_rejected() -> None:
    for bad_weight in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            mixed_kd_ce_loss(
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [1],
                temperature=2.0,
                kd_weight=bad_weight,
                hard_ce_weight=0.0,
            )
