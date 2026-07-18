"""Strict scalar and JSON-value validation at contract boundaries."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.dataset import Dataset, SplitHashes, TaskDifficultyCounts
from distillery.contracts.errors import DistilleryErrorCode, ErrorPayload
from distillery.contracts.hashing import RFC8785_SAFE_INTEGER_MAX
from distillery.contracts.manifest import ManifestTraining
from distillery.contracts.proof import ArmResult
from distillery.contracts.tasks import (
    Difficulty,
    JournalLine,
    TaskId,
    TransactionReviewOutput,
)

HEX64 = "a" * 64


def _counts() -> TaskDifficultyCounts:
    return TaskDifficultyCounts(
        by_task={
            TaskId.TRANSACTION_REVIEW: 1,
            TaskId.VARIANCE_ANALYSIS: 0,
            TaskId.CASH_RECONCILIATION: 0,
        },
        by_difficulty={
            Difficulty.EASY: 1,
            Difficulty.MEDIUM: 0,
            Difficulty.HARD: 0,
        },
    )


@pytest.mark.parametrize("amount", ["100", 100.0, True])
def test_accounting_minor_units_require_strict_integers(amount: object) -> None:
    with pytest.raises(ValidationError):
        JournalLine(account="6100", side="debit", amount_minor=amount)


@pytest.mark.parametrize("confidence", ["0.9", float("nan"), float("inf")])
def test_confidence_requires_finite_numeric_wire_value(confidence: object) -> None:
    with pytest.raises(ValidationError):
        TransactionReviewOutput(
            gl_account="6100",
            journal_entry=(
                {"account": "6100", "side": "debit", "amount_minor": 100},
                {"account": "2100", "side": "credit", "amount_minor": 100},
            ),
            policy_action="approve",
            rule_ids=("POL-1",),
            evidence=(),
            confidence=confidence,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"seed": "17"},
        {"max_steps": "30"},
        {"token_budget": False},
        {"max_length": 512.0},
        {"qlora": {"rank": {8}}},
        {"qlora": {"dropout": float("nan")}},
    ],
)
def test_manifest_training_rejects_coercion_and_non_json(
    updates: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "seed": 17,
        "max_steps": 30,
        "token_budget": 0,
        "max_length": 512,
    }
    payload.update(updates)
    with pytest.raises(ValidationError):
        ManifestTraining.model_validate(payload)


def test_resource_counts_do_not_coerce_strings() -> None:
    with pytest.raises(ValidationError):
        Dataset(
            dataset_id="ds_strict_001",
            content_sha256=HEX64,
            split_sha256=SplitHashes(train=HEX64, validation=HEX64),
            uri="s3://bucket/dataset/",
            provenance_summary="strict",
            task_difficulty_counts=_counts(),
            example_count="1",
            created_at=datetime(2026, 7, 18, tzinfo=UTC),
        )


def test_numeric_timestamps_are_not_coerced() -> None:
    with pytest.raises(ValidationError, match="RFC 3339"):
        Dataset(
            dataset_id="ds_strict_002",
            content_sha256=HEX64,
            split_sha256=SplitHashes(train=HEX64, validation=HEX64),
            uri="s3://bucket/dataset/",
            provenance_summary="strict",
            task_difficulty_counts=_counts(),
            example_count=1,
            created_at=1_721_321_600,
        )


def test_error_retryable_and_proof_metrics_are_strict() -> None:
    with pytest.raises(ValidationError):
        ErrorPayload(
            code=DistilleryErrorCode.INVALID_DATASET,
            message="invalid",
            retryable="false",
        )
    with pytest.raises(ValidationError):
        ArmResult(arm_id="rules", primary_index="0.8")
    with pytest.raises(ValidationError):
        ArmResult(arm_id="rules", metrics={"score": float("nan")})


def test_all_contract_integer_paths_reject_unsafe_json_integers() -> None:
    unsafe = RFC8785_SAFE_INTEGER_MAX + 1
    with pytest.raises(ValidationError, match="less than or equal"):
        JournalLine(account="6100", side="debit", amount_minor=unsafe)
    with pytest.raises(ValidationError, match="safe domain"):
        ErrorPayload(
            code=DistilleryErrorCode.INVALID_DATASET,
            message="unsafe",
            details={"nested": {"count": unsafe}},
            retryable=False,
        )
    with pytest.raises(ValidationError, match="less than or equal"):
        ManifestTraining(
            seed=17,
            max_steps=unsafe,
            token_budget=0,
            max_length=512,
        )


def test_dataset_count_breakdowns_are_complete_and_consistent() -> None:
    with pytest.raises(ValidationError, match="complete supported task set"):
        TaskDifficultyCounts(
            by_task={TaskId.TRANSACTION_REVIEW: 1},
            by_difficulty={
                Difficulty.EASY: 1,
                Difficulty.MEDIUM: 0,
                Difficulty.HARD: 0,
            },
        )
    with pytest.raises(ValidationError, match="totals must match"):
        TaskDifficultyCounts(
            by_task={
                TaskId.TRANSACTION_REVIEW: 1,
                TaskId.VARIANCE_ANALYSIS: 1,
                TaskId.CASH_RECONCILIATION: 0,
                TaskId.MERCHANT_TAGGING: 0,
            },
            by_difficulty={
                Difficulty.EASY: 1,
                Difficulty.MEDIUM: 0,
                Difficulty.HARD: 0,
            },
        )
    with pytest.raises(ValidationError, match="example_count"):
        Dataset(
            dataset_id="ds_counts_001",
            content_sha256=HEX64,
            split_sha256=SplitHashes(train=HEX64, validation=HEX64),
            uri="s3://bucket/dataset/",
            provenance_summary="counts",
            task_difficulty_counts=_counts(),
            example_count=2,
            created_at=datetime(2026, 7, 18, tzinfo=UTC),
        )
