"""Deterministic protocol hashing for planned techniques."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.techniques.compatibility import CompatibilityDecision
from distillery.techniques.descriptor import TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error

PROTOCOL_SCHEMA_VERSION = "distillery.technique.protocol.v1"


def protocol_payload(
    *,
    descriptor: TechniqueDescriptor,
    config_sha256: str,
    compatibility: CompatibilityDecision,
    channel_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "technique_id": descriptor.technique_id,
        "version": descriptor.version,
        "descriptor_sha256": descriptor.descriptor_sha256,
        "config_sha256": config_sha256,
        "compatibility": compatibility.model_dump(mode="json"),
        "channel_contract": dict(channel_contract) if channel_contract is not None else None,
    }


def compute_protocol_hash(
    *,
    descriptor: TechniqueDescriptor,
    config_sha256: str,
    compatibility: CompatibilityDecision,
    channel_contract: Mapping[str, Any] | None = None,
) -> str:
    """Content-addressed hash of the sealed planning protocol."""
    return content_sha256(
        protocol_payload(
            descriptor=descriptor,
            config_sha256=config_sha256,
            compatibility=compatibility,
            channel_contract=channel_contract,
        )
    )


def assert_protocol_deterministic(
    *,
    descriptor: TechniqueDescriptor,
    config_sha256: str,
    compatibility: CompatibilityDecision,
    channel_contract: Mapping[str, Any] | None = None,
) -> str:
    """Compute the protocol hash twice; diverge fails loud."""
    first = compute_protocol_hash(
        descriptor=descriptor,
        config_sha256=config_sha256,
        compatibility=compatibility,
        channel_contract=channel_contract,
    )
    second = compute_protocol_hash(
        descriptor=descriptor,
        config_sha256=config_sha256,
        compatibility=compatibility,
        channel_contract=channel_contract,
    )
    if first != second:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC,
            "technique protocol hash is non-deterministic",
            details={
                "technique_id": descriptor.technique_id,
                "version": descriptor.version,
                "first": first,
                "second": second,
            },
        )
    return first


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "assert_protocol_deterministic",
    "compute_protocol_hash",
    "protocol_payload",
]
