"""Typed errors for the BYODT technique seam."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NoReturn

from pydantic import Field, JsonValue, StrictBool, StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload
from distillery.contracts.ids import RunId


class TechniqueErrorCode(StrEnum):
    """Local technique lifecycle / negotiation failures."""

    TECHNIQUE_UNKNOWN = "TECHNIQUE_UNKNOWN"
    TECHNIQUE_VERSION_COLLISION = "TECHNIQUE_VERSION_COLLISION"
    TECHNIQUE_DESCRIPTOR_INVALID = "TECHNIQUE_DESCRIPTOR_INVALID"
    TECHNIQUE_DESCRIPTOR_MUTABLE = "TECHNIQUE_DESCRIPTOR_MUTABLE"
    TECHNIQUE_SCHEMA_MISMATCH = "TECHNIQUE_SCHEMA_MISMATCH"
    TECHNIQUE_CONFIG_MISMATCH = "TECHNIQUE_CONFIG_MISMATCH"
    TECHNIQUE_CAPABILITY_UNKNOWN = "TECHNIQUE_CAPABILITY_UNKNOWN"
    TECHNIQUE_INCOMPATIBLE = "TECHNIQUE_INCOMPATIBLE"
    TECHNIQUE_DIGEST_INVALID = "TECHNIQUE_DIGEST_INVALID"
    TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH = "TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH"
    TECHNIQUE_NONDETERMINISTIC = "TECHNIQUE_NONDETERMINISTIC"
    TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN = "TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN"
    TECHNIQUE_CHANNEL_INVALID = "TECHNIQUE_CHANNEL_INVALID"
    TECHNIQUE_LIFECYCLE_INVALID = "TECHNIQUE_LIFECYCLE_INVALID"


# Map technique failures onto the frozen Distillery control-plane codes when
# crossing the existing API/SDK error seam. Local codes remain authoritative
# inside the techniques module.
_DISTILLERY_CODE_MAP: dict[TechniqueErrorCode, DistilleryErrorCode] = {
    TechniqueErrorCode.TECHNIQUE_UNKNOWN: DistilleryErrorCode.RECIPE_NOT_IMPLEMENTED,
    TechniqueErrorCode.TECHNIQUE_VERSION_COLLISION: DistilleryErrorCode.SCHEMA_MISMATCH,
    TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID: DistilleryErrorCode.SCHEMA_MISMATCH,
    TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_MUTABLE: DistilleryErrorCode.ARTIFACT_INTEGRITY_FAILED,
    TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH: DistilleryErrorCode.SCHEMA_MISMATCH,
    TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH: DistilleryErrorCode.SCHEMA_MISMATCH,
    TechniqueErrorCode.TECHNIQUE_CAPABILITY_UNKNOWN: DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
    TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE: DistilleryErrorCode.RECIPE_INCOMPATIBLE,
    TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID: DistilleryErrorCode.ARTIFACT_INTEGRITY_FAILED,
    TechniqueErrorCode.TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH: (
        DistilleryErrorCode.ARTIFACT_INTEGRITY_FAILED
    ),
    TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC: DistilleryErrorCode.SCHEMA_MISMATCH,
    TechniqueErrorCode.TECHNIQUE_EXTERNAL_IMPORT_FORBIDDEN: (
        DistilleryErrorCode.CAPABILITY_UNAVAILABLE
    ),
    TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID: DistilleryErrorCode.AWS_SUBMISSION_FAILED,
    TechniqueErrorCode.TECHNIQUE_LIFECYCLE_INVALID: DistilleryErrorCode.INVALID_TRANSITION,
}


class TechniqueErrorPayload(FrozenModel):
    code: TechniqueErrorCode
    message: StrictStr = Field(min_length=1)
    details: FrozenJsonObject = Field(default_factory=dict)
    retryable: StrictBool = False
    run_id: RunId | None = None

    def to_distillery_payload(self) -> ErrorPayload:
        return ErrorPayload.from_code(
            _DISTILLERY_CODE_MAP[self.code],
            self.message,
            details={
                "technique_error_code": self.code.value,
                **dict(self.details),
            },
            run_id=self.run_id,
            retryable=False,
        )


class TechniqueError(Exception):
    """Exception carrying a typed TechniqueErrorPayload."""

    def __init__(self, payload: TechniqueErrorPayload) -> None:
        super().__init__(payload.message)
        self.payload = payload

    @property
    def code(self) -> TechniqueErrorCode:
        return self.payload.code

    def as_distillery_error(self) -> DistilleryError:
        return DistilleryError(self.payload.to_distillery_payload())


def technique_error(
    code: TechniqueErrorCode,
    message: str,
    *,
    details: dict[str, JsonValue] | None = None,
    run_id: str | None = None,
) -> TechniqueError:
    return TechniqueError(
        TechniqueErrorPayload(
            code=code,
            message=message,
            details=details or {},
            retryable=False,
            run_id=run_id,
        )
    )


def raise_technique_error(
    code: TechniqueErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> NoReturn:
    raise technique_error(code, message, details=details, run_id=run_id)


__all__ = [
    "TechniqueError",
    "TechniqueErrorCode",
    "TechniqueErrorPayload",
    "raise_technique_error",
    "technique_error",
]
