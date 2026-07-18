"""Typed Distillery error codes and payloads."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictStr,
    model_validator,
)

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.ids import RunId


class DistilleryErrorCode(StrEnum):
    INVALID_DATASET = "INVALID_DATASET"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    DATA_LEAKAGE_DETECTED = "DATA_LEAKAGE_DETECTED"
    UNSUPPORTED_LABEL_SOURCE = "UNSUPPORTED_LABEL_SOURCE"
    MODEL_REVISION_UNPINNED = "MODEL_REVISION_UNPINNED"
    TOKENIZER_MISMATCH = "TOKENIZER_MISMATCH"
    CHAT_TEMPLATE_MISMATCH = "CHAT_TEMPLATE_MISMATCH"
    LICENSE_GATE_UNRESOLVED = "LICENSE_GATE_UNRESOLVED"
    OUTPUT_USE_NOT_ALLOWED = "OUTPUT_USE_NOT_ALLOWED"
    RECIPE_NOT_IMPLEMENTED = "RECIPE_NOT_IMPLEMENTED"
    RECIPE_INCOMPATIBLE = "RECIPE_INCOMPATIBLE"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    MEMORY_DRY_RUN_FAILED = "MEMORY_DRY_RUN_FAILED"
    ESTIMATED_BUDGET_EXCEEDED = "ESTIMATED_BUDGET_EXCEEDED"
    AWS_QUOTA_UNAVAILABLE = "AWS_QUOTA_UNAVAILABLE"
    AWS_SUBMISSION_FAILED = "AWS_SUBMISSION_FAILED"
    AWS_JOB_FAILED = "AWS_JOB_FAILED"
    RUN_TIMEOUT = "RUN_TIMEOUT"
    CANCELLED = "CANCELLED"
    ARTIFACT_INTEGRITY_FAILED = "ARTIFACT_INTEGRITY_FAILED"
    EVALUATION_INCOMPLETE = "EVALUATION_INCOMPLETE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    AUTO_RESOLVER_FAILED = "AUTO_RESOLVER_FAILED"
    TECHNIQUE_UNKNOWN = "TECHNIQUE_UNKNOWN"
    TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN = "TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN"
    TECHNIQUE_NONDETERMINISTIC = "TECHNIQUE_NONDETERMINISTIC"


ERROR_CODES: frozenset[str] = frozenset(member.value for member in DistilleryErrorCode)

RETRYABLE_ERROR_CODES: frozenset[DistilleryErrorCode] = frozenset(
    {DistilleryErrorCode.AWS_SUBMISSION_FAILED}
)
TERMINAL_ERROR_CODES: frozenset[DistilleryErrorCode] = frozenset(
    {
        DistilleryErrorCode.AWS_JOB_FAILED,
        DistilleryErrorCode.RUN_TIMEOUT,
        DistilleryErrorCode.CANCELLED,
    }
)


class ErrorPayload(FrozenModel):
    """Machine-readable error returned by API/SDK surfaces."""

    code: DistilleryErrorCode
    message: StrictStr = Field(min_length=1)
    details: FrozenJsonObject = Field(default_factory=dict)
    retryable: StrictBool
    run_id: RunId | None = None

    @model_validator(mode="after")
    def _fixed_retryability(self) -> ErrorPayload:
        if self.retryable and self.code not in RETRYABLE_ERROR_CODES:
            raise ValueError(
                f"{self.code.value} cannot be retryable; only explicit "
                "transport/service submission failures may be retried"
            )
        return self

    @classmethod
    def from_code(
        cls,
        code: DistilleryErrorCode,
        message: StrictStr,
        *,
        details: dict[StrictStr, JsonValue] | None = None,
        run_id: RunId | None = None,
        retryable: bool | None = None,
    ) -> ErrorPayload:
        explicit_retryable = False if retryable is None else retryable
        if explicit_retryable and code not in RETRYABLE_ERROR_CODES:
            raise ValueError(
                f"{code.value} cannot be retryable; only explicit "
                "transport/service submission failures may be retried"
            )
        return cls(
            code=code,
            message=message,
            details=details or {},
            retryable=explicit_retryable,
            run_id=run_id,
        )


class DistilleryError(Exception):
    """Exception carrying a typed ErrorPayload."""

    def __init__(self, payload: ErrorPayload) -> None:
        super().__init__(payload.message)
        self.payload = payload

    @property
    def code(self) -> DistilleryErrorCode:
        return self.payload.code
