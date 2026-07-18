"""Typed technique lifecycle states for validate → register → plan."""

from __future__ import annotations

from enum import StrEnum

from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error


class TechniqueLifecycle(StrEnum):
    """Deterministic states exposed through the technique seam."""

    DESCRIBED = "described"
    VALIDATED = "validated"
    REGISTERED = "registered"
    COMPATIBLE = "compatible"
    PLANNED = "planned"
    REJECTED = "rejected"


_ALLOWED_TRANSITIONS: dict[TechniqueLifecycle, frozenset[TechniqueLifecycle]] = {
    TechniqueLifecycle.DESCRIBED: frozenset(
        {
            TechniqueLifecycle.VALIDATED,
            TechniqueLifecycle.REJECTED,
        }
    ),
    TechniqueLifecycle.VALIDATED: frozenset(
        {
            TechniqueLifecycle.REGISTERED,
            TechniqueLifecycle.REJECTED,
        }
    ),
    TechniqueLifecycle.REGISTERED: frozenset(
        {
            TechniqueLifecycle.COMPATIBLE,
            TechniqueLifecycle.REJECTED,
        }
    ),
    TechniqueLifecycle.COMPATIBLE: frozenset(
        {
            TechniqueLifecycle.PLANNED,
            TechniqueLifecycle.REJECTED,
        }
    ),
    TechniqueLifecycle.PLANNED: frozenset(),
    TechniqueLifecycle.REJECTED: frozenset(),
}


def advance_lifecycle(
    current: TechniqueLifecycle,
    nxt: TechniqueLifecycle,
) -> TechniqueLifecycle:
    """Advance lifecycle or fail loud; never skip or silently rewind."""
    if nxt not in _ALLOWED_TRANSITIONS[current]:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_LIFECYCLE_INVALID,
            f"invalid technique lifecycle transition {current.value} -> {nxt.value}",
            details={"from": current.value, "to": nxt.value},
        )
    return nxt


__all__ = ["TechniqueLifecycle", "advance_lifecycle"]
