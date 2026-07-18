"""Canonical JSON and content-addressed SHA-256 helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Annotated, Any

import rfc8785
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    StrictInt,
    StrictStr,
)

RFC8785_SAFE_INTEGER_MAX = 9_007_199_254_740_991
Sha256Hex = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
PrefixedSha256 = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
GitCommitSha = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")]
SafeInt = Annotated[
    StrictInt,
    Field(ge=-RFC8785_SAFE_INTEGER_MAX, le=RFC8785_SAFE_INTEGER_MAX),
]
NonNegativeSafeInt = Annotated[
    StrictInt,
    Field(ge=0, le=RFC8785_SAFE_INTEGER_MAX),
]
PositiveSafeInt = Annotated[
    StrictInt,
    Field(ge=1, le=RFC8785_SAFE_INTEGER_MAX),
]


def _parse_aware_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string or datetime")
    if "T" not in value:
        raise ValueError("timestamp must include date and time")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be valid RFC 3339") from exc


def _require_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


AwareDatetime = Annotated[
    datetime,
    BeforeValidator(_parse_aware_datetime),
    AfterValidator(_require_aware_datetime),
]


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    """Normalize explicitly supported values into the RFC 8785 JSON data model."""
    if isinstance(value, BaseModel):
        return _normalize_json(value.model_dump(mode="python"), path=path)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{path}: naive datetime is not canonicalizable")
        normalized = value.astimezone(UTC)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if abs(value) > RFC8785_SAFE_INTEGER_MAX:
            raise ValueError(
                f"{path}: integer exceeds RFC 8785 interoperable safe domain"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite floats are not valid JSON")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: JSON object keys must be strings")
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError(f"{path}: duplicate normalized JSON key {normalized_key!r}")
            normalized[normalized_key] = _normalize_json(
                item,
                path=f"{path}.{normalized_key}",
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path}: unsupported canonical JSON type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return RFC 8785 canonical JSON bytes for an explicitly supported value."""
    return rfc8785.dumps(_normalize_json(value))


def sha256_hex(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("sha256_hex requires bytes")
    return hashlib.sha256(data).hexdigest()


def content_sha256(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of ``value``."""
    return sha256_hex(canonical_json_bytes(value))
