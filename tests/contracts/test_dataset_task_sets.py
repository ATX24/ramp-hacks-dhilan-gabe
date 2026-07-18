"""Dataset task-set compatibility and versioned construction invariants."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.dataset import Dataset, SplitHashes, TaskDifficultyCounts
from distillery.contracts.tasks import Difficulty, TaskId

HEX_A = "a" * 64
HEX_B = "b" * 64
V1_RESOURCE_HASH = "71b8e27253d00207e67d4cb934b165602921ad59ce0a32783c1ef703e0435e09"
FROZEN_DATASET_V1_JSON = (
    '{"schema_version":"distillery.dataset.v1",'
    '"dataset_id":"ds_finance_world_v1",'
    '"content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"split_sha256":{'
    '"train":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"validation":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
    '"test":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"iid_test":null,"ood_test":null},'
    '"uri":"s3://bucket/datasets/ds_finance_world_v1/",'
    '"provenance_summary":"synthetic finance_world.v1",'
    '"task_difficulty_counts":{"by_task":{'
    '"transaction_review":5,"variance_analysis":5,"cash_reconciliation":2},'
    '"by_difficulty":{"easy":3,"medium":5,"hard":4}},'
    '"example_count":12,"created_at":"2026-07-18T12:00:00Z","metadata":{}}'
)
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


def _counts(by_task: dict[TaskId, int]) -> TaskDifficultyCounts:
    return TaskDifficultyCounts(
        by_task=by_task,
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


def test_frozen_v1_json_roundtrip_and_resource_hash_are_unchanged() -> None:
    dataset = Dataset.model_validate_json(FROZEN_DATASET_V1_JSON)

    assert dataset.model_dump_json() == FROZEN_DATASET_V1_JSON
    assert dataset.resource_hash() == V1_RESOURCE_HASH
    assert "task_set_version" not in TaskDifficultyCounts.model_fields


def test_v2_complete_task_set_roundtrips_without_a_version_field() -> None:
    counts = _counts(V2_BY_TASK)
    wire = counts.model_dump(mode="json")

    assert set(counts.by_task) == set(TaskId)
    assert set(wire) == {"by_task", "by_difficulty"}
    assert TaskDifficultyCounts.model_validate(wire) == counts
    counts.require_finance_world("finance_world.v2")


@pytest.mark.parametrize(
    "by_task",
    [
        {TaskId.TRANSACTION_REVIEW: 1},
        {
            TaskId.TRANSACTION_REVIEW: 1,
            TaskId.VARIANCE_ANALYSIS: 1,
            TaskId.MERCHANT_TAGGING: 1,
        },
    ],
)
def test_arbitrary_and_mixed_task_subsets_are_rejected(
    by_task: dict[TaskId, int],
) -> None:
    with pytest.raises(ValidationError, match="complete supported task set"):
        TaskDifficultyCounts(
            by_task=by_task,
            by_difficulty={
                Difficulty.EASY: sum(by_task.values()),
                Difficulty.MEDIUM: 0,
                Difficulty.HARD: 0,
            },
        )


def test_unknown_extra_task_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TaskDifficultyCounts.model_validate(
            {
                "by_task": {
                    "transaction_review": 4,
                    "variance_analysis": 4,
                    "merchant_tagging": 2,
                    "cash_reconciliation": 2,
                    "not_a_task": 0,
                },
                "by_difficulty": {"easy": 3, "medium": 5, "hard": 4},
            }
        )

    assert "not_a_task" in str(exc_info.value)


def test_v2_explicit_seam_rejects_missing_merchant() -> None:
    with pytest.raises(ValueError, match=r"missing=\['merchant_tagging'\]"):
        _counts(V1_BY_TASK).require_finance_world("finance_world.v2")


def test_v1_explicit_seam_rejects_extra_merchant() -> None:
    with pytest.raises(ValueError, match=r"extra=\['merchant_tagging'\]"):
        _counts(V2_BY_TASK).require_finance_world("finance_world.v1")


@pytest.mark.parametrize("by_task", [V1_BY_TASK, V2_BY_TASK])
def test_each_supported_task_set_enforces_total_equality(
    by_task: dict[TaskId, int],
) -> None:
    with pytest.raises(ValidationError, match="task and difficulty count totals must match"):
        TaskDifficultyCounts(
            by_task=by_task,
            by_difficulty={
                Difficulty.EASY: 3,
                Difficulty.MEDIUM: 5,
                Difficulty.HARD: 3,
            },
        )


@pytest.mark.parametrize("by_task", [V1_BY_TASK, V2_BY_TASK])
def test_each_supported_task_set_enforces_dataset_example_count_total(
    by_task: dict[TaskId, int],
) -> None:
    with pytest.raises(
        ValidationError,
        match="example_count must equal both task and difficulty count totals",
    ):
        _dataset(_counts(by_task), example_count=13)


def test_legacy_v1_timestamp_error_precedes_dataset_total_validator() -> None:
    payload = json.loads(FROZEN_DATASET_V1_JSON)
    payload["created_at"] = 1_721_321_600
    payload["example_count"] = 13

    with pytest.raises(ValidationError) as exc_info:
        Dataset.model_validate(payload)

    message = str(exc_info.value)
    assert "RFC 3339" in message
    assert "complete supported task set" not in message
    assert "example_count must equal" not in message
