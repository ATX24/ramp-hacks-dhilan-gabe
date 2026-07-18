"""Sandboxed synthetic tool executor: deterministic, no shell/network."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

from distillery.finance_agent.contracts import ProvenanceRef, ToolName, ToolResult
from distillery.finance_agent.tools import tool_schema
from distillery.finance_agent.world import AgentWorld

_FORBIDDEN_ARG_KEYS = frozenset(
    {
        "cmd",
        "command",
        "shell",
        "subprocess",
        "url",
        "http",
        "https",
        "endpoint",
        "path",
        "filename",
        "socket",
        "host",
    }
)


class SandboxError(ValueError):
    """Raised when a tool call violates sandbox or schema bounds."""


class FinanceAgentSandbox:
    """In-process tool environment bound to one latent world."""

    def __init__(self, world: AgentWorld, *, allowed_tools: tuple[ToolName, ...] | None = None):
        self._world = world
        self._allowed = frozenset(allowed_tools or tuple(ToolName))
        self._validators = {
            tool: Draft202012Validator(tool_schema(tool)) for tool in ToolName
        }

    @property
    def world(self) -> AgentWorld:
        return self._world

    def execute(self, *, call_id: str, tool: ToolName, arguments: Mapping[str, Any]) -> ToolResult:
        if tool not in self._allowed:
            return ToolResult(
                call_id=call_id,
                tool=tool,
                ok=False,
                result={},
                error_code="TOOL_NOT_ALLOWED",
            )
        args = _jsonable_args(dict(arguments))
        bad = sorted(set(args) & _FORBIDDEN_ARG_KEYS)
        if bad:
            raise SandboxError(f"forbidden argument keys: {bad}")
        # Reject string values that look like shell/network escapes.
        for key, value in args.items():
            if isinstance(value, str) and _looks_like_escape(value):
                raise SandboxError(f"argument {key!r} looks like shell/network escape")
        errors = sorted(
            self._validators[tool].iter_errors(args),
            key=lambda err: list(err.absolute_path),
        )
        if errors:
            return ToolResult(
                call_id=call_id,
                tool=tool,
                ok=False,
                result={"validation_errors": [error.message for error in errors[:5]]},
                error_code="INVALID_ARGUMENTS",
            )
        handler = {
            ToolName.CHART_OF_ACCOUNTS_LOOKUP: self._coa,
            ToolName.POLICY_LOOKUP: self._policy,
            ToolName.LEDGER_QUERY: self._ledger,
            ToolName.CALCULATOR: self._calculator,
            ToolName.TRANSACTION_MATCHING: self._match,
            ToolName.VARIANCE_DRILL_DOWN: self._variance,
        }[tool]
        payload, provenance = handler(args)
        return ToolResult(
            call_id=call_id,
            tool=tool,
            ok=True,
            result=payload,
            provenance=tuple(provenance),
        )

    def _coa(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        query = str(args["query"]).lower()
        code = args.get("account_code")
        matches = []
        for account in self._world.accounts:
            if code and account.code != code:
                continue
            hay = f"{account.code} {account.name} {account.domain}".lower()
            if query in hay or account.code == args.get("account_code"):
                matches.append(
                    {"code": account.code, "name": account.name, "domain": account.domain}
                )
        provenance = [
            ProvenanceRef(source_id=m["code"], field="account", value=m["name"]) for m in matches
        ]
        return {"matches": matches}, provenance

    def _policy(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        policy_id = args["policy_id"]
        as_of = args["as_of"]
        include_superseded = bool(args.get("include_superseded", False))
        versions = [
            policy
            for policy in self._world.policies
            if policy.policy_id == policy_id
            and (include_superseded or not policy.superseded)
            and policy.effective_from <= as_of
            and (policy.effective_to is None or as_of <= policy.effective_to)
        ]
        if not versions and include_superseded:
            versions = [p for p in self._world.policies if p.policy_id == policy_id]
        versions = sorted(versions, key=lambda p: p.effective_from, reverse=True)
        if not versions:
            return {"policy": None}, []
        chosen = versions[0]
        payload = {
            "policy_id": chosen.policy_id,
            "version": chosen.version,
            "rule_text": chosen.rule_text,
            "action": chosen.action,
            "superseded": chosen.superseded,
            "effective_from": chosen.effective_from,
            "effective_to": chosen.effective_to,
        }
        provenance = [
            ProvenanceRef(
                source_id=chosen.policy_id,
                field="version",
                value=chosen.version,
            )
        ]
        return {"policy": payload}, provenance

    def _ledger(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        limit = int(args.get("limit", 20))
        rows = []
        for row in self._world.ledger:
            if row.account_code != args["account_code"]:
                continue
            if args.get("period") and row.period != args["period"]:
                continue
            if args.get("merchant_id") and row.merchant_id != args["merchant_id"]:
                continue
            rows.append(
                {
                    "entry_id": row.entry_id,
                    "account_code": row.account_code,
                    "period": row.period,
                    "merchant_id": row.merchant_id,
                    "amount_minor": row.amount_minor,
                    "memo": row.memo,
                }
            )
            if len(rows) >= limit:
                break
        provenance = [
            ProvenanceRef(
                source_id=row["entry_id"],
                field="amount_minor",
                value=str(row["amount_minor"]),
            )
            for row in rows
        ]
        return {"rows": rows}, provenance

    def _calculator(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        op = args["op"]
        operands = [int(x) for x in args["operands_minor"]]
        if op == "add":
            value = sum(operands)
        elif op == "sub":
            value = operands[0] - sum(operands[1:])
        elif op == "mul":
            value = 1
            for item in operands:
                value *= item
        elif op == "abs_diff":
            if len(operands) != 2:
                raise SandboxError("abs_diff requires exactly 2 operands")
            value = abs(operands[0] - operands[1])
        elif op == "pct_of":
            if len(operands) != 2 or operands[1] == 0:
                raise SandboxError("pct_of requires two operands and non-zero denominator")
            value = (operands[0] * 10_000) // operands[1]
        else:
            raise SandboxError(f"unsupported op {op}")
        return {"result_minor": value, "op": op}, [
            ProvenanceRef(source_id="calculator", field="result_minor", value=str(value))
        ]

    def _match(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        tolerance = int(args.get("tolerance_minor", 0))
        books = {entry.book_id: entry for entry in self._world.book_entries}
        banks = {event.bank_id: event for event in self._world.bank_events}
        missing = [bid for bid in args["book_ids"] if bid not in books] + [
            bid for bid in args["bank_ids"] if bid not in banks
        ]
        if missing:
            return {
                "matched": False,
                "missing_ids": missing,
                "difference_minor": None,
            }, []
        book_sum = sum(books[bid].amount_minor for bid in args["book_ids"])
        bank_sum = sum(banks[bid].amount_minor for bid in args["bank_ids"])
        diff = book_sum - bank_sum
        matched = abs(diff) <= tolerance
        return {
            "matched": matched,
            "book_sum_minor": book_sum,
            "bank_sum_minor": bank_sum,
            "difference_minor": diff,
        }, [
            ProvenanceRef(source_id="match", field="difference_minor", value=str(diff)),
        ]

    def _variance(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        top_k = int(args.get("top_k", 3))
        drivers = [
            driver
            for driver in self._world.variance_drivers
            if driver.account_code == args["account_code"] and driver.period == args["period"]
        ]
        drivers = sorted(drivers, key=lambda d: (-abs(d.impact_minor), d.driver_id))[:top_k]
        payload = {
            "drivers": [
                {
                    "driver_id": driver.driver_id,
                    "impact_minor": driver.impact_minor,
                    "label": driver.label,
                }
                for driver in drivers
            ],
            "total_impact_minor": sum(driver.impact_minor for driver in drivers),
        }
        provenance = [
            ProvenanceRef(
                source_id=driver.driver_id,
                field="impact_minor",
                value=str(driver.impact_minor),
            )
            for driver in drivers
        ]
        return payload, provenance


def _jsonable_args(value: Any) -> Any:
    """Convert frozen tuples/mappings back to JSON-schema-friendly containers."""
    if isinstance(value, Mapping) and not isinstance(value, dict):
        return {key: _jsonable_args(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _jsonable_args(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable_args(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_args(item) for item in value]
    return value


def _looks_like_escape(value: str) -> bool:
    lowered = value.lower()
    needles = (
        "://",
        "`",
        "$(",
        "${",
        "&&",
        ";",
        "|",
        "curl ",
        "wget ",
        "/bin/",
        "subprocess",
    )
    return any(needle in lowered for needle in needles)


def canonical_tool_result_json(result: ToolResult) -> str:
    return json.dumps(result.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


__all__ = [
    "FinanceAgentSandbox",
    "SandboxError",
    "canonical_tool_result_json",
]
