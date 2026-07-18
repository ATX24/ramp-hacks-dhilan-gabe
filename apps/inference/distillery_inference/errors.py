"""Explicit inference error codes. Never silently substitute another model."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class InferenceErrorCode(StrEnum):
    SERVING_NOT_READY = "SERVING_NOT_READY"
    MODEL_NOT_IN_REGISTRY = "MODEL_NOT_IN_REGISTRY"
    ARTIFACT_NOT_SERVABLE = "ARTIFACT_NOT_SERVABLE"
    ARTIFACT_CHECKSUM_FAILED = "ARTIFACT_CHECKSUM_FAILED"
    ARTIFACT_ID_MISMATCH = "ARTIFACT_ID_MISMATCH"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    TOKEN_LIMIT_EXCEEDED = "TOKEN_LIMIT_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"
    ADAPTER_SWITCH_FAILED = "ADAPTER_SWITCH_FAILED"
    NETWORK_FORBIDDEN = "NETWORK_FORBIDDEN"
    REQUEST_LIMIT_EXCEEDED = "REQUEST_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class InferenceError(Exception):
    """Fail-loud inference error with a stable machine code."""

    def __init__(
        self,
        code: InferenceErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        self.http_status = http_status

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": "error",
            "provenance": "none",
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
