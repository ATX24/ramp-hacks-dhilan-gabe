"""Version-aware Dataset task-count compatibility and invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.dataset import Dataset, SplitHashes, TaskDifficultyCounts
from distillery.contracts.tasks import Difficulty, TaskId

HEX_A = "a" * 64
HEX_B = "b" * 64
V1_RESOURCE_HASH = "71b8e27253d00207e67d4cb934b165602921ad59ce0a32783c1ef703e0435e09"

V1_BY_TASK = {
    TaskId.TRANSACTION_REVIEW: 5,
    TaskId.VARIANCE_ANALYSIS: 5,
    TaskId.CASH_RECONCILIATION: 2,
}
V2_BY_TASK = {
    TaskId.TRANSACTION_REVIEW: 4,
    TaskId.VARIANCE_ANALYSIS: 4,
    TaskId.CASH_RECONCILIATION: 2,
    TaskId.MERCHANT_TAGGING: 2,
}
BY_DIFFICULTY = {
    Difficulty.EASY: 3,
    Difficulty.MEDIUM: 5,
    Difficulty.HARD: 4,
}


def _counts(version: str) -> TaskDifficultyCounts:
    return TaskDifficultyCounts(
        task_set_version=version,
        by_task=V1_BY_TASK if version == "finance_world.v1" else V2_BY_TASK,
        by_difficulty=BY_DIFFICULTY,
    )


def _dataset(
    counts: TaskDifficultyCounts,
    *,
    example_count: int = 12,
    created_at: object = datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
) -> Dataset:
    return Dataset(
        dataset_id="ds_finance_world_v1",
        content_sha256=HEX_A,
        split_sha256=SplitHashes(train=HEX_A, validation=HEX_B, test=HEX_A),
        uri="s3://bucket/datasets/ds_finance_world_v1/",
        provenance_summary="synthetic finance_world.v1",
        task_difficulty_counts=counts,
        example_count=example_count,
        created_at=created_at,
    )


def test_v1_roundtrip_wire_and_resource_hash_are_unchanged() -> None:
    dataset = _dataset(_counts("finance_world.v1"))
    wire = dataset.model_dump(mode="json")

    assert wire["task_difficulty_counts"] == {
        "by_task": {
            "transaction_review": 5,
            "variance_analysis": 5,
            "cash_reconciliation": 2,
        },
        "by_difficulty": {"easy": 3, "medium": 5, "hard": 4},
    }
    assert "task_set_version" not in wire["task_difficulty_counts"]
    assert Dataset.model_validate(wire) == dataset
    assert dataset.resource_hash() == V1_RESOURCE_HASH


def test_v2_requires_all_four_tasks_and_roundtrips_version() -> None:
    counts = _counts("finance_world.v2")
    wire = counts.model_dump(mode="json")

    assert set(counts.by_task) == set(TaskId)
    assert wire["task_set_version"] == "finance_world.v2"
    assert set(wire["by_task"]) == {task.value for task in TaskId}
    assert TaskDifficultyCounts.model_validate(wire) == counts


def test_unversioned_four_task_wire_is_inferred_as_v2() -> None:
    counts = TaskDifficultyCounts(
        by_task=V2_BY_TASK,
        by_difficulty=BY_DIFFICULTY,
    )

    assert counts.task_set_version == "finance_world.v2"
    assert counts.model_dump(mode="json")["task_set_version"] == "finance_world.v2"


@pytest.mark.parametrize(
    ("version", "by_task", "missing", "extra"),
    [
        (
            "finance_world.v1",
            {TaskId.TRANSACTION_REVIEW: 1, TaskId.VARIANCE_ANALYSIS: 1},
            "cash_reconciliation",
            None,
        ),
        (
            "finance_world.v1",
            {task: 1 for task in TaskId},
            None,
            "merchant_tagging",
        ),
        (
            "finance_world.v2",
            V1_BY_TASK,
            "merchant_tagging",
            None,
        ),
    ],
)
def test_versioned_task_sets_reject_missing_and_extra_tasks(
    version: str,
    by_task: dict[TaskId, int],
    missing: str | None,
    extra: str | None,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TaskDifficultyCounts(
            task_set_version=version,
            by_task=by_task,
            by_difficulty={
                Difficulty.EASY: sum(by_task.values()),
                Difficulty.MEDIUM: 0,
                Difficulty.HARD: 0,
            },
        )

    message = str(exc_info.value)
    assert version in message
    if missing is not None:
        assert f"missing=['{missing}']" in message
    if extra is not None:
        assert f"extra=['{extra}']" in message


@pytest.mark.parametrize("version", ["finance_world.v1", "finance_world.v2"])
def test_each_version_enforces_task_and_difficulty_total_equality(version: str) -> None:
    by_task = V1_BY_TASK if version == "finance_world.v1" else V2_BY_TASK
    with pytest.raises(ValidationError, match="task and difficulty count totals must match"):
        TaskDifficultyCounts(
            task_set_version=version,
            by_task=by_task,
            by_difficulty={
                Difficulty.EASY: 3,
                Difficulty.MEDIUM: 5,
                Difficulty.HARD: 3,
            },
        )


@pytest.mark.parametrize("version", ["finance_world.v1", "finance_world.v2"])
def test_each_version_enforces_dataset_example_count_total(version: str) -> None:
    with pytest.raises(
        ValidationError,
        match="example_count must equal both task and difficulty count totals",
    ):
        _dataset(_counts(version), example_count=13)


def test_invalid_timestamp_error_precedes_dataset_total_validator() -> None:
    payload = _dataset(_counts("finance_world.v1")).model_dump(mode="json")
    payload["created_at"] = 1_721_321_600
    payload["example_count"] = 13

    with pytest.raises(ValidationError) as exc_info:
        Dataset.model_validate(payload)

    message = str(exc_info.value)
    assert "RFC 3339" in message
    assert "example_count must equal" not in message
