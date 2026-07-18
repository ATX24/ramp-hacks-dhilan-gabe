"""Hash-bound evidence primitives shared by every 72B execution gate."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256

Sha256 = str
SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
PREFIXED_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

_PLACEHOLDERS = frozenset(
    {
        "",
        "0" * 40,
        "0" * 64,
        "sha256:" + ("0" * 64),
        "PENDING",
        "PLACEHOLDER",
        "REPLACE_ME",
        "TBD",
        "TODO",
        "UNSET",
    }
)


class VerificationSource(StrEnum):
    LOCAL_BYTES = "local_bytes"
    LIVE_AWS = "live_aws"
    TARGET_DEVICE = "target_device"
    FINANCE_WORLD_V2 = "finance_world.v2"


def reject_placeholders(value: Any, *, path: str = "$") -> None:
    """Reject exact sentinel values recursively without phrase matching."""
    if isinstance(value, str):
        if value in _PLACEHOLDERS or value.upper() in _PLACEHOLDERS:
            raise ValueError(f"{path} is an unset placeholder")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            reject_placeholders(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_placeholders(nested, path=f"{path}[{index}]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HashBoundEvidence(BaseModel):
    """Immutable evidence whose digest covers every field except itself."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @classmethod
    def seal(cls, **values: Any) -> Self:
        reject_placeholders(values)
        provisional = cls.model_construct(evidence_sha256="0" * 64, **values)
        body = provisional.model_dump(mode="json", exclude={"evidence_sha256"})
        reject_placeholders(body)
        return cls(evidence_sha256=content_sha256(body), **values)

    @model_validator(mode="after")
    def _verify_evidence_hash(self) -> Self:
        body = self.model_dump(mode="json", exclude={"evidence_sha256"})
        reject_placeholders(body)
        expected = content_sha256(body)
        if self.evidence_sha256 != expected:
            raise ValueError(
                f"evidence_sha256 mismatch: sealed={self.evidence_sha256} computed={expected}"
            )
        return self


def hash_bound_payload(payload: dict[str, Any]) -> str:
    reject_placeholders(payload)
    return content_sha256(payload)
