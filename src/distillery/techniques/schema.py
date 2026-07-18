"""Offline-only JSON Schema validation and canonical config sealing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.contracts.base import deep_thaw
from distillery.contracts.hashing import content_sha256
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error

_FORBIDDEN_SCHEMA_KEYWORDS = frozenset(
    {
        "$anchor",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$recursiveAnchor",
        "$recursiveRef",
        "$ref",
        "$schema",
        "default",
    }
)
_SECRET_TOKENS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "authorization",
        "authtoken",
        "apitoken",
        "bearertoken",
        "credential",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "sessionkey",
        "sessiontoken",
        "token",
    }
)


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    return content_sha256(deep_thaw(dict(config)))


def validate_schema_definition(config_schema: Mapping[str, Any]) -> None:
    """Check one self-contained Draft 2020-12 schema without retrieval."""
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH,
            "jsonschema is required to validate technique schemas",
        )

    schema = deep_thaw(dict(config_schema))
    forbidden = _find_forbidden_schema_keywords(schema)
    if forbidden:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH,
            "technique config schemas must be fully inline and contain no "
            "references, retrieval identifiers, or defaults",
            details={"forbidden_keywords": forbidden},
        )
    secret_paths = _find_secret_schema_properties(schema)
    if secret_paths:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH,
            "technique config schemas cannot declare secret-like fields",
            details={"secret_like_properties": secret_paths},
        )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH,
            "technique config_schema is not valid Draft 2020-12 JSON Schema",
            details={"error": exc.message},
        )


def validate_config_against_schema(
    config: Mapping[str, Any],
    config_schema: Mapping[str, Any],
    *,
    technique_id: str,
    version: str,
) -> tuple[dict[str, Any], str]:
    """Return unchanged validated config and its RFC 8785 hash."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_SCHEMA_MISMATCH,
            "jsonschema is required to validate technique configs",
            details={"technique_id": technique_id, "version": version},
        )

    schema = deep_thaw(dict(config_schema))
    instance = deep_thaw(dict(config))
    secret_paths = _find_secret_config_fields(instance)
    if secret_paths:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
            "technique config cannot contain secret-like fields",
            details={
                "technique_id": technique_id,
                "version": version,
                "secret_like_fields": secret_paths,
            },
        )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: list(error.path),
    )
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
    return instance, canonical_config_hash(instance)


def _find_forbidden_schema_keywords(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in _FORBIDDEN_SCHEMA_KEYWORDS:
                found.append(child)
            found.extend(_find_forbidden_schema_keywords(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_schema_keywords(item, f"{path}[{index}]"))
    return found


def _normalized_secret_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _is_secret_like(key: str) -> bool:
    normalized = _normalized_secret_key(key)
    return normalized in _SECRET_TOKENS or any(
        normalized.endswith(token) for token in _SECRET_TOKENS if token != "token"
    )


def _find_secret_schema_properties(schema: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for key in properties:
                if _is_secret_like(str(key)):
                    found.append(f"{path}.properties.{key}")
        for key, item in schema.items():
            found.extend(_find_secret_schema_properties(item, f"{path}.{key}"))
    elif isinstance(schema, (list, tuple)):
        for index, item in enumerate(schema):
            found.extend(_find_secret_schema_properties(item, f"{path}[{index}]"))
    return found


def _find_secret_config_fields(config: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(config, Mapping):
        for key, item in config.items():
            child = f"{path}.{key}"
            if _is_secret_like(str(key)):
                found.append(child)
            found.extend(_find_secret_config_fields(item, child))
    elif isinstance(config, (list, tuple)):
        for index, item in enumerate(config):
            found.extend(_find_secret_config_fields(item, f"{path}[{index}]"))
    return found


__all__ = [
    "canonical_config_hash",
    "validate_config_against_schema",
    "validate_schema_definition",
]
