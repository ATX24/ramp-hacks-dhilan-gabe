"""Shared fail-closed evidence labels for proof artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceKind(StrEnum):
    """Exhaustive provenance labels for numeric proof evidence."""

    MEASURED = "measured"
    PROJECTED = "projected"
    MISSING = "missing"


def evidence_kind(value: EvidenceKind | str) -> EvidenceKind:
    """Validate and normalize an evidence kind.

    Unknown labels are contract errors. They are never coerced to a favorable
    default.
    """

    return value if isinstance(value, EvidenceKind) else EvidenceKind(value)


@dataclass(frozen=True)
class LabeledValue:
    """A value with explicit measured/projected/missing provenance."""

    value: float | int | str | None
    kind: EvidenceKind | str
    unit: str | None = None
    label: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        kind = evidence_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is EvidenceKind.MISSING and self.value is not None:
            raise ValueError("missing evidence must not carry a value")
        if kind is not EvidenceKind.MISSING and self.value is None:
            raise ValueError(f"{kind.value} evidence requires a value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "kind": self.kind.value,
            "unit": self.unit,
            "label": self.label,
            "reason": self.reason,
        }
