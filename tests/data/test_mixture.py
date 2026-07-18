"""Exact 45/45/10 task and 30/40/30 difficulty mixture tests."""

from __future__ import annotations

from distillery.contracts.tasks import Difficulty, TaskId
from distillery.data.mixture import (
    DIFFICULTY_MIXTURE,
    TASK_MIXTURE,
    difficulty_counts,
    hamilton_apportion,
    joint_cell_counts,
    mixture_plan,
    task_counts,
)


def test_task_mixture_weights_sum_to_one() -> None:
    assert abs(sum(TASK_MIXTURE.values()) - 1.0) < 1e-12
    assert abs(sum(DIFFICULTY_MIXTURE.values()) - 1.0) < 1e-12


def test_hamilton_sums_exactly() -> None:
    for total in (0, 1, 7, 80, 160, 320, 400, 800, 3200):
        counts = task_counts(total)
        assert sum(counts.values()) == total
        dcounts = difficulty_counts(total)
        assert sum(dcounts.values()) == total


def test_smoke_and_full_split_mixtures() -> None:
    # Smoke
    assert task_counts(320) == {
        TaskId.TRANSACTION_REVIEW: 144,
        TaskId.VARIANCE_ANALYSIS: 144,
        TaskId.CASH_RECONCILIATION: 32,
    }
    assert task_counts(80) == {
        TaskId.TRANSACTION_REVIEW: 36,
        TaskId.VARIANCE_ANALYSIS: 36,
        TaskId.CASH_RECONCILIATION: 8,
    }
    assert task_counts(160) == {
        TaskId.TRANSACTION_REVIEW: 72,
        TaskId.VARIANCE_ANALYSIS: 72,
        TaskId.CASH_RECONCILIATION: 16,
    }
    # Full
    assert task_counts(3200)[TaskId.TRANSACTION_REVIEW] == 1440
    assert task_counts(3200)[TaskId.CASH_RECONCILIATION] == 320
    assert task_counts(800)[TaskId.TRANSACTION_REVIEW] == 360


def test_difficulty_mixture_global_margins() -> None:
    for n in (80, 160, 320, 400, 800, 3200):
        d = difficulty_counts(n)
        assert d[Difficulty.EASY] + d[Difficulty.MEDIUM] + d[Difficulty.HARD] == n
        assert d == {
            Difficulty.EASY: int(n * 0.30),
            Difficulty.MEDIUM: int(n * 0.40),
            Difficulty.HARD: int(n * 0.30),
        }


def test_mixture_plan_length_and_cells() -> None:
    plan = mixture_plan(320)
    assert len(plan) == 320
    cells = joint_cell_counts(320)
    assert sum(cells.values()) == 320
    for task, expected in task_counts(320).items():
        assert (
            sum(count for (cell_task, _difficulty), count in cells.items() if cell_task == task)
            == expected
        )
    for difficulty, expected in difficulty_counts(320).items():
        assert (
            sum(
                count
                for (_task, cell_difficulty), count in cells.items()
                if cell_difficulty == difficulty
            )
            == expected
        )


def test_hamilton_rejects_bad_weights() -> None:
    try:
        hamilton_apportion(10, {"a": 0.5}, ["a"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
