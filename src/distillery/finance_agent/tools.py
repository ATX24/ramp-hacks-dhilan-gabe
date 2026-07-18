"""Bounded JSON Schema contracts for Finance Agent sandbox tools."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from distillery.finance_agent.contracts import ToolName

_TOOL_SCHEMAS: dict[ToolName, dict[str, Any]] = {
    ToolName.CHART_OF_ACCOUNTS_LOOKUP: {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 128},
            "account_code": {"type": "string", "minLength": 1, "maxLength": 32},
        },
    },
    ToolName.POLICY_LOOKUP: {
        "type": "object",
        "additionalProperties": False,
        "required": ["policy_id", "as_of"],
        "properties": {
            "policy_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "as_of": {
                "type": "string",
                "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
            },
            "include_superseded": {"type": "boolean"},
        },
    },
    ToolName.LEDGER_QUERY: {
        "type": "object",
        "additionalProperties": False,
        "required": ["account_code"],
        "properties": {
            "account_code": {"type": "string", "minLength": 1, "maxLength": 32},
            "period": {
                "type": "string",
                "pattern": r"^[0-9]{4}-[0-9]{2}$",
            },
            "merchant_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
    },
    ToolName.CALCULATOR: {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "operands_minor"],
        "properties": {
            "op": {
                "type": "string",
                "enum": ["add", "sub", "mul", "abs_diff", "pct_of"],
            },
            "operands_minor": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "integer"},
            },
        },
    },
    ToolName.TRANSACTION_MATCHING: {
        "type": "object",
        "additionalProperties": False,
        "required": ["book_ids", "bank_ids"],
        "properties": {
            "book_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {"type": "string", "minLength": 1, "maxLength": 64},
            },
            "bank_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {"type": "string", "minLength": 1, "maxLength": 64},
            },
            "tolerance_minor": {"type": "integer", "minimum": 0, "maximum": 10_000},
        },
    },
    ToolName.VARIANCE_DRILL_DOWN: {
        "type": "object",
        "additionalProperties": False,
        "required": ["period", "account_code"],
        "properties": {
            "period": {
                "type": "string",
                "pattern": r"^[0-9]{4}-[0-9]{2}$",
            },
            "account_code": {"type": "string", "minLength": 1, "maxLength": 32},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    },
}

TOOL_ARGUMENT_SCHEMAS: Mapping[ToolName, Mapping[str, Any]] = MappingProxyType(
    {name: MappingProxyType(schema) for name, schema in _TOOL_SCHEMAS.items()}
)

ALL_TOOLS: tuple[ToolName, ...] = tuple(ToolName)


def tool_schema(tool: ToolName) -> Mapping[str, Any]:
    return TOOL_ARGUMENT_SCHEMAS[tool]


def catalog_for_prompt(tools: tuple[ToolName, ...] | None = None) -> list[dict[str, Any]]:
    selected = tools or ALL_TOOLS
    return [
        {
            "name": tool.value,
            "arguments_schema": dict(TOOL_ARGUMENT_SCHEMAS[tool]),
        }
        for tool in selected
    ]


__all__ = [
    "ALL_TOOLS",
    "TOOL_ARGUMENT_SCHEMAS",
    "catalog_for_prompt",
    "tool_schema",
]
