"""Exact task/difficulty mixture apportionment for sealed corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from distillery.contracts.tasks import Difficulty, TaskId

# Locked finance_world.v1 experiment mixture (Primary A/B + cash backup).
TASK_MIXTURE: dict[TaskId, float] = {
    TaskId.TRANSACTION_REVIEW: 0.45,
    TaskId.VARIANCE_ANALYSIS: 0.45,
    TaskId.CASH_RECONCILIATION: 0.10,
}

# Locked finance_world.v2 mixture (Primary A/B/C + cash diagnostic backup).
# Justification: 35/35/20/10 on a 6,240-example full corpus yields 1,248
# merchant_tagging examples (>=1,000) while keeping A/B each at 2,184
# (only ~6.7% below v1's 2,340) and cash diagnostic weight unchanged at 10%.
TASK_MIXTURE_V2: dict[TaskId, float] = {
    TaskId.TRANSACTION_REVIEW: 0.35,
    TaskId.VARIANCE_ANALYSIS: 0.35,
    TaskId.MERCHANT_TAGGING: 0.20,
    TaskId.CASH_RECONCILIATION: 0.10,
}

DIFFICULTY_MIXTURE: dict[Difficulty, float] = {
    Difficulty.EASY: 0.30,
    Difficulty.MEDIUM: 0.40,
    Difficulty.HARD: 0.30,
}

TASK_ORDER: tuple[TaskId, ...] = (
    TaskId.TRANSACTION_REVIEW,
    TaskId.VARIANCE_ANALYSIS,
    TaskId.CASH_RECONCILIATION,
)

TASK_ORDER_V2: tuple[TaskId, ...] = (
    TaskId.TRANSACTION_REVIEW,
    TaskId.VARIANCE_ANALYSIS,
    TaskId.MERCHANT_TAGGING,
    TaskId.CASH_RECONCILIATION,
)

DIFFICULTY_ORDER: tuple[Difficulty, ...] = (
    Difficulty.EASY,
    Difficulty.MEDIUM,
    Difficulty.HARD,
)


def hamilton_apportion(
    total: int,
    weights: Mapping[str, float],
    order: Sequence[str],
) -> dict[str, int]:
    """Largest-remainder (Hamilton) method; sums exactly to ``total``."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if set(weights) != set(order):
        raise ValueError("weights keys must match order exactly")
    weight_sum = sum(weights[k] for k in order)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

    exact = {k: total * weights[k] for k in order}
    floors = {k: int(exact[k]) for k in order}
    assigned = sum(floors.values())
    remainders = sorted(
        ((exact[k] - floors[k], -i, k) for i, k in enumerate(order)),
        reverse=True,
    )
    result = dict(floors)
    for _, _, key in remainders[: total - assigned]:
        result[key] += 1
    if sum(result.values()) != total:
        raise RuntimeError("apportionment failed to sum to total")
    return result


def task_counts(
    total: int,
    *,
    mixture: Mapping[TaskId, float] | None = None,
    order: Sequence[TaskId] | None = None,
) -> dict[TaskId, int]:
    weights = dict(mixture or TASK_MIXTURE)
    task_order = tuple(order or TASK_ORDER)
    raw = hamilton_apportion(
        total,
        {t.value: weights[t] for t in task_order},
        [t.value for t in task_order],
    )
    return {TaskId(k): v for k, v in raw.items()}


def difficulty_counts(task_total: int) -> dict[Difficulty, int]:
    raw = hamilton_apportion(
        task_total,
        {d.value: DIFFICULTY_MIXTURE[d] for d in DIFFICULTY_ORDER},
        [d.value for d in DIFFICULTY_ORDER],
    )
    return {Difficulty(k): v for k, v in raw.items()}


def joint_cell_counts(
    total: int,
    *,
    mixture: Mapping[TaskId, float] | None = None,
    order: Sequence[TaskId] | None = None,
) -> dict[tuple[TaskId, Difficulty], int]:
    """Allocate cells while satisfying exact task and difficulty margins."""
    task_order = tuple(order or TASK_ORDER)
    rows = task_counts(total, mixture=mixture, order=task_order)
    columns = difficulty_counts(total)
    exact = {
        (task, difficulty): rows[task] * DIFFICULTY_MIXTURE[difficulty]
        for task in task_order
        for difficulty in DIFFICULTY_ORDER
    }
    cells = {cell: int(value) for cell, value in exact.items()}
    row_remaining = {
        task: rows[task] - sum(cells[(task, difficulty)] for difficulty in DIFFICULTY_ORDER)
        for task in task_order
    }
    column_remaining = {
        difficulty: columns[difficulty] - sum(cells[(task, difficulty)] for task in task_order)
        for difficulty in DIFFICULTY_ORDER
    }
    while sum(row_remaining.values()):
        candidates = [
            (
                exact[(task, difficulty)] - cells[(task, difficulty)],
                -task_order.index(task),
                -DIFFICULTY_ORDER.index(difficulty),
                task,
                difficulty,
            )
            for task in task_order
            for difficulty in DIFFICULTY_ORDER
            if row_remaining[task] > 0 and column_remaining[difficulty] > 0
        ]
        if not candidates:
            raise RuntimeError("could not satisfy joint mixture margins")
        *_score, task, difficulty = max(candidates)
        cells[(task, difficulty)] += 1
        row_remaining[task] -= 1
        column_remaining[difficulty] -= 1

    if any(column_remaining.values()):
        raise RuntimeError("joint mixture left unmatched difficulty counts")
    return cells


def mixture_plan(
    total: int,
    *,
    mixture: Mapping[TaskId, float] | None = None,
    order: Sequence[TaskId] | None = None,
) -> list[tuple[TaskId, Difficulty]]:
    """Expand a split size into an ordered list of (task, difficulty) slots."""
    task_order = tuple(order or TASK_ORDER)
    slots: list[tuple[TaskId, Difficulty]] = []
    cells = joint_cell_counts(total, mixture=mixture, order=task_order)
    for task in task_order:
        for difficulty in DIFFICULTY_ORDER:
            slots.extend([(task, difficulty)] * cells[(task, difficulty)])
    if len(slots) != total:
        raise RuntimeError(f"mixture_plan size {len(slots)} != {total}")
    return slots


def summarize_mixture(
    examples: Sequence[tuple[TaskId, Difficulty]],
    *,
    order: Sequence[TaskId] | None = None,
) -> dict[str, dict[str, int]]:
    task_order = tuple(order or TASK_ORDER)
    by_task: dict[str, int] = {t.value: 0 for t in task_order}
    by_difficulty: dict[str, int] = {d.value: 0 for d in DIFFICULTY_ORDER}
    by_cell: dict[str, int] = {}
    for task, difficulty in examples:
        by_task[task.value] = by_task.get(task.value, 0) + 1
        by_difficulty[difficulty.value] += 1
        key = f"{task.value}:{difficulty.value}"
        by_cell[key] = by_cell.get(key, 0) + 1
    return {"by_task": by_task, "by_difficulty": by_difficulty, "by_cell": by_cell}
