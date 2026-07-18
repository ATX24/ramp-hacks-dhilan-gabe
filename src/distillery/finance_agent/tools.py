"""Canonical, bounded JSON Schemas for the synthetic Finance Agent tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from distillery.contracts.hashing import RFC8785_SAFE_INTEGER_MAX, content_sha256
from distillery.finance_agent.contracts import ToolDefinition, ToolName

_SAFE_MIN = -RFC8785_SAFE_INTEGER_MAX
_SAFE_MAX = RFC8785_SAFE_INTEGER_MAX

_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.CHART_OF_ACCOUNTS_LOOKUP: (
        "Search the synthetic entity chart of accounts. If account_code is supplied, "
        "both the query and code must match."
    ),
    ToolName.POLICY_LOOKUP: (
        "Return the policy version temporally effective on as_of. Superseded versions "
        "remain valid for dates inside their historical effective window."
    ),
    ToolName.LEDGER_QUERY: (
        "Return synthetic ledger rows filtered by account, period, and optional merchant."
    ),
    ToolName.CALCULATOR: (
        "Perform bounded deterministic integer arithmetic. pct_of returns basis points."
    ),
    ToolName.TRANSACTION_MATCHING: (
        "Compare unique synthetic book and bank IDs and return exact aggregate difference."
    ),
    ToolName.VARIANCE_DRILL_DOWN: (
        "Return ranked variance drivers plus full-period and returned-driver totals."
    ),
}

_TOOL_SCHEMAS: dict[ToolName, dict[str, Any]] = {
    ToolName.CHART_OF_ACCOUNTS_LOOKUP: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 128},
            "account_code": {
                "type": "string",
                "pattern": r"^[0-9]{4,8}$",
                "maxLength": 8,
            },
        },
    },
    ToolName.POLICY_LOOKUP: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["policy_id", "as_of"],
        "properties": {
            "policy_id": {
                "type": "string",
                "pattern": r"^pol_[a-z0-9_]{1,60}$",
                "maxLength": 64,
            },
            "as_of": {
                "type": "string",
                "format": "date",
                "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
            },
            "include_history": {"type": "boolean"},
        },
    },
    ToolName.LEDGER_QUERY: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["account_code", "period"],
        "properties": {
            "account_code": {
                "type": "string",
                "pattern": r"^[0-9]{4,8}$",
                "maxLength": 8,
            },
            "period": {
                "type": "string",
                "pattern": r"^[0-9]{4}-[0-9]{2}$",
            },
            "merchant_id": {
                "type": "string",
                "pattern": r"^mrc_[a-z0-9]{18}$",
                "maxLength": 64,
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
    },
    ToolName.CALCULATOR: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "operands"],
        "properties": {
            "op": {
                "type": "string",
                "enum": ["add", "sub", "mul", "abs_diff", "pct_of"],
            },
            "operands": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                    "type": "integer",
                    "minimum": _SAFE_MIN,
                    "maximum": _SAFE_MAX,
                },
            },
        },
        "allOf": [
            {
                "if": {"properties": {"op": {"const": "abs_diff"}}},
                "then": {"properties": {"operands": {"minItems": 2, "maxItems": 2}}},
            },
            {
                "if": {"properties": {"op": {"const": "pct_of"}}},
                "then": {"properties": {"operands": {"minItems": 2, "maxItems": 2}}},
            },
        ],
    },
    ToolName.TRANSACTION_MATCHING: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["book_ids", "bank_ids", "tolerance_minor"],
        "properties": {
            "book_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "pattern": r"^bok_[a-z0-9]{18}$",
                    "maxLength": 64,
                },
            },
            "bank_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "pattern": r"^bnk_[a-z0-9]{18}$",
                    "maxLength": 64,
                },
            },
            "tolerance_minor": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10_000,
            },
        },
    },
    ToolName.VARIANCE_DRILL_DOWN: {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["period", "account_code", "top_k"],
        "properties": {
            "period": {
                "type": "string",
                "pattern": r"^[0-9]{4}-[0-9]{2}$",
            },
            "account_code": {
                "type": "string",
                "pattern": r"^[0-9]{4,8}$",
                "maxLength": 8,
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    },
}

TOOL_ARGUMENT_SCHEMAS: Mapping[ToolName, Mapping[str, Any]] = MappingProxyType(
    {name: MappingProxyType(schema) for name, schema in _TOOL_SCHEMAS.items()}
)
ALL_TOOLS: tuple[ToolName, ...] = tuple(ToolName)


def tool_schema(tool: ToolName) -> dict[str, Any]:
    """Return a detached JSON copy so callers cannot mutate the canonical schema."""
    return json.loads(json.dumps(_TOOL_SCHEMAS[tool], sort_keys=True))


def tool_definitions(tools: tuple[ToolName, ...]) -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition.seal(
            name=tool,
            description=_DESCRIPTIONS[tool],
            input_schema=tool_schema(tool),
        )
        for tool in tools
    )


def catalog_for_prompt(tools: tuple[ToolName, ...] | None = None) -> list[dict[str, Any]]:
    selected = tools or ALL_TOOLS
    return [definition.model_dump(mode="json") for definition in tool_definitions(selected)]


def tool_catalog_sha256(tools: tuple[ToolName, ...]) -> str:
    return content_sha256(catalog_for_prompt(tools))


__all__ = [
    "ALL_TOOLS",
    "TOOL_ARGUMENT_SCHEMAS",
    "catalog_for_prompt",
    "tool_catalog_sha256",
    "tool_definitions",
    "tool_schema",
]
