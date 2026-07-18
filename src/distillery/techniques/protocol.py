"""Canonical recomputation and verification for sealed technique plans."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from distillery.contracts.hashing import content_sha256
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error

if TYPE_CHECKING:
    from distillery.techniques.runtime import TechniquePlan

PROTOCOL_SCHEMA_VERSION = "distillery.technique.protocol.v2"


def canonical_protocol_payload(
    plan: TechniquePlan | Mapping[str, Any],
) -> dict[str, Any]:
    if hasattr(plan, "model_dump"):
        payload = plan.model_dump(mode="json")
    else:
        payload = dict(plan)
    payload.pop("protocol_sha256", None)
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "resolved_plan": payload,
    }


def recompute_protocol_hash(
    plan: TechniquePlan | Mapping[str, Any],
) -> str:
    """Recompute the hash from the complete resolved plan identity."""
    return content_sha256(canonical_protocol_payload(plan))


def verify_protocol_hash(plan: TechniquePlan) -> None:
    expected = recompute_protocol_hash(plan)
    if not hmac.compare_digest(plan.protocol_sha256, expected):
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC,
            "protocol_sha256 does not match the complete resolved plan identity",
            details={
                "technique_id": plan.technique_id,
                "version": plan.version,
                "expected": expected,
                "actual": plan.protocol_sha256,
            },
        )


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "canonical_protocol_payload",
    "recompute_protocol_hash",
    "verify_protocol_hash",
]
