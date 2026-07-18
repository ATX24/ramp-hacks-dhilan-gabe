"""Typed failures for the sequence-teacher seam."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NoReturn

from pydantic import Field, JsonValue, StrictBool, StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel


class TeacherErrorCode(StrEnum):
    ACCESS_DENIED = "TEACHER_ACCESS_DENIED"
    USE_CASE_PENDING = "TEACHER_USE_CASE_PENDING"
    OUTPUT_USE_NOT_ALLOWED = "TEACHER_OUTPUT_USE_NOT_ALLOWED"
    PROVIDER_USE_PROHIBITED = "TEACHER_PROVIDER_USE_PROHIBITED"
    AUTHORIZATION_REQUIRED = "TEACHER_AUTHORIZATION_REQUIRED"
    AUTHORIZATION_INVALID = "TEACHER_AUTHORIZATION_INVALID"
    AUTHORIZATION_EXPIRED = "TEACHER_AUTHORIZATION_EXPIRED"
    AUTHORIZATION_SCOPE_MISMATCH = "TEACHER_AUTHORIZATION_SCOPE_MISMATCH"
    SECRET_UNAVAILABLE = "TEACHER_SECRET_UNAVAILABLE"
    MODEL_UNAVAILABLE = "TEACHER_MODEL_UNAVAILABLE"
    LICENSE_GATE_FAILED = "TEACHER_LICENSE_GATE_FAILED"
    THROTTLED = "TEACHER_THROTTLED"
    MALFORMED_JSON = "TEACHER_MALFORMED_JSON"
    SCHEMA_REJECTED = "TEACHER_SCHEMA_REJECTED"
    COST_EXHAUSTED = "TEACHER_COST_EXHAUSTED"
    REQUEST_CAP_EXCEEDED = "TEACHER_REQUEST_CAP_EXCEEDED"
    CACHE_INTEGRITY_FAILED = "TEACHER_CACHE_INTEGRITY_FAILED"
    MODEL_SUBSTITUTION = "TEACHER_MODEL_SUBSTITUTION"
    RECIPE_INCOMPATIBLE = "TEACHER_RECIPE_INCOMPATIBLE"
    TOOL_USE_REJECTED = "TEACHER_TOOL_USE_REJECTED"
    RETRIES_EXHAUSTED = "TEACHER_RETRIES_EXHAUSTED"
    CHAIN_EXHAUSTED = "TEACHER_CHAIN_EXHAUSTED"
    PREMIUM_FALLBACK_FORBIDDEN = "TEACHER_PREMIUM_FALLBACK_FORBIDDEN"
    INVALID_REQUEST = "TEACHER_INVALID_REQUEST"


RETRYABLE_TEACHER_CODES: frozenset[TeacherErrorCode] = frozenset({TeacherErrorCode.THROTTLED})


class TeacherErrorPayload(FrozenModel):
    code: TeacherErrorCode
    message: StrictStr = Field(min_length=1)
    details: FrozenJsonObject = Field(default_factory=dict)
    retryable: StrictBool = False


class TeacherError(Exception):
    """Exception carrying a stable, serializable failure payload."""

    def __init__(self, payload: TeacherErrorPayload) -> None:
        super().__init__(payload.message)
        self.payload = payload

    @property
    def code(self) -> TeacherErrorCode:
        return self.payload.code

    @property
    def retryable(self) -> bool:
        return bool(self.payload.retryable)


def teacher_error(
    code: TeacherErrorCode,
    message: str,
    *,
    details: dict[str, JsonValue] | None = None,
    retryable: bool | None = None,
) -> TeacherError:
    explicit_retryable = code in RETRYABLE_TEACHER_CODES if retryable is None else retryable
    if explicit_retryable and code not in RETRYABLE_TEACHER_CODES:
        raise ValueError(f"{code.value} cannot be retryable")
    return TeacherError(
        TeacherErrorPayload(
            code=code,
            message=message,
            details=details or {},
            retryable=explicit_retryable,
        )
    )


def raise_teacher_error(
    code: TeacherErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool | None = None,
) -> NoReturn:
    raise teacher_error(code, message, details=details, retryable=retryable)


__all__ = [
    "RETRYABLE_TEACHER_CODES",
    "TeacherError",
    "TeacherErrorCode",
    "TeacherErrorPayload",
    "raise_teacher_error",
    "teacher_error",
]
