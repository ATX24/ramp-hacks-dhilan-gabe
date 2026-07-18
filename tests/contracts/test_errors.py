"""Typed error payload serialization and retryability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.errors import (
    ERROR_CODES,
    RETRYABLE_ERROR_CODES,
    TERMINAL_ERROR_CODES,
    DistilleryError,
    DistilleryErrorCode,
    ErrorPayload,
)


def test_error_payload_roundtrip() -> None:
    payload = ErrorPayload.from_code(
        DistilleryErrorCode.TOKENIZER_MISMATCH,
        "teacher/student tokenizer fingerprints differ",
        details={"teacher": "aaa", "student": "bbb"},
        run_id="run_err_001",
    )
    data = payload.model_dump(mode="json")
    again = ErrorPayload.model_validate(data)
    assert again == payload
    assert again.retryable is False
    assert again.code is DistilleryErrorCode.TOKENIZER_MISMATCH


def test_retryable_only_for_transport_codes() -> None:
    retryable = {DistilleryErrorCode.AWS_SUBMISSION_FAILED}
    for code in DistilleryErrorCode:
        payload = ErrorPayload.from_code(code, "msg")
        assert payload.retryable is False
    assert RETRYABLE_ERROR_CODES == retryable
    assert ErrorPayload.from_code(
        DistilleryErrorCode.AWS_SUBMISSION_FAILED,
        "transient service failure",
        retryable=True,
    ).retryable


def test_job_failure_and_timeout_are_terminal_non_retryable() -> None:
    assert {
        DistilleryErrorCode.AWS_JOB_FAILED,
        DistilleryErrorCode.RUN_TIMEOUT,
    } <= TERMINAL_ERROR_CODES
    for code in TERMINAL_ERROR_CODES:
        assert not ErrorPayload.from_code(code, "terminal").retryable


def test_retryability_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="cannot be retryable"):
        ErrorPayload.from_code(
            DistilleryErrorCode.AWS_JOB_FAILED,
            "failed",
            retryable=True,
        )
    with pytest.raises(ValidationError, match="cannot be retryable"):
        ErrorPayload(
            code=DistilleryErrorCode.RUN_TIMEOUT,
            message="timed out",
            details={},
            retryable=True,
        )


def test_validation_error_locations_normalize_to_json_arrays() -> None:
    payload = ErrorPayload.from_code(
        DistilleryErrorCode.SCHEMA_MISMATCH,
        "invalid",
        details={
            "validation_errors": [
                {
                    "loc": ("training", "completion_evidence"),
                    "message": "missing",
                }
            ]
        },
    )
    assert payload.model_dump(mode="json")["details"]["validation_errors"][0][
        "loc"
    ] == ["training", "completion_evidence"]


def test_contract_errors_never_retryable() -> None:
    for code in (
        DistilleryErrorCode.INVALID_DATASET,
        DistilleryErrorCode.LICENSE_GATE_UNRESOLVED,
        DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
        DistilleryErrorCode.ESTIMATED_BUDGET_EXCEEDED,
        DistilleryErrorCode.DATA_LEAKAGE_DETECTED,
        DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
    ):
        assert ErrorPayload.from_code(code, "x").retryable is False


def test_distillery_error_exposes_payload() -> None:
    err = DistilleryError(
        ErrorPayload.from_code(
            DistilleryErrorCode.CANCELLED,
            "run cancelled",
            run_id="run_err_002",
        )
    )
    assert err.code is DistilleryErrorCode.CANCELLED
    assert str(err) == "run cancelled"
    assert err.payload.run_id == "run_err_002"


def test_error_codes_frozenset_matches_enum() -> None:
    assert ERROR_CODES == frozenset(c.value for c in DistilleryErrorCode)
