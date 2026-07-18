"""Config JSON Schema validation and canonical config hashing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.contracts.base import deep_thaw
from distillery.contracts.hashing import content_sha256
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error


def canonical_config_hash(config: Mapping[str, Any] | dict[str, Any]) -> str:
    """RFC 8785 content hash of technique config (key-order independent)."""
    return content_sha256(deep_thaw(dict(config)))


def validate_config_against_schema(
    config: Mapping[str, Any] | dict[str, Any],
    config_schema: Mapping[str, Any] | dict[str, Any],
    *,
    technique_id: str,
    version: str,
) -> str:
    """
    Validate config against the descriptor's JSON Schema.

    Returns the canonical config hash on success. Never coerces or defaults
    values that alter the canonical hash.
    """
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dev/runtime dependency
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH,
            "jsonschema is required to validate technique configs",
            details={"technique_id": technique_id, "version": version},
        )
        raise exc

    schema = deep_thaw(dict(config_schema))
    instance = deep_thaw(dict(config))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda err: list(err.path))
    if errors:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
            "technique config does not match declared JSON Schema",
            details={
                "technique_id": technique_id,
                "version": version,
                "error_count": len(errors),
                "first_error": errors[0].message,
                "first_path": list(errors[0].path),
            },
        )
    # Reject schemas that silently inject defaults into the instance.
    if instance != deep_thaw(dict(config)):
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC,
            "config validation mutated the instance; defaults are forbidden",
            details={"technique_id": technique_id, "version": version},
        )
    del jsonschema  # imported for availability side effect above
    return canonical_config_hash(instance)


__all__ = ["canonical_config_hash", "validate_config_against_schema"]
