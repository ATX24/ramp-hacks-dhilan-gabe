"""Deterministic in-process tools with no shell, filesystem, or network surface."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from jsonschema import Draft202012Validator

from distillery.contracts.hashing import (
    RFC8785_SAFE_INTEGER_MAX,
    canonical_json_bytes,
)
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


class SandboxSecurityError(ValueError):
    """A call attempted to escape the closed synthetic capability surface."""


class ToolSemanticError(ValueError):
    """A normal, serializable tool failure."""

    def __init__(self, error_code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.result = {"message": message, **details}


class FinanceAgentSandbox:
    """Tool executor bound to one immutable latent world and explicit capability set."""

    def __init__(self, world: AgentWorld, *, allowed_tools: tuple[ToolName, ...]) -> None:
        if not allowed_tools:
            raise ValueError("sandbox requires at least one allowed tool")
        self._world = world
        self._allowed = frozenset(allowed_tools)
        self._validators = {tool: Draft202012Validator(tool_schema(tool)) for tool in ToolName}

    @property
    def world(self) -> AgentWorld:
        return self._world

    def execute(
        self,
        *,
        call_id: str,
        tool: ToolName,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        args = _jsonable(arguments)
        _reject_escape_surface(args)
        if tool not in self._allowed:
            return self._error(
                call_id=call_id,
                tool=tool,
                error_code="TOOL_NOT_ALLOWED",
                message="tool is not present in this episode capability set",
            )
        errors = sorted(
            self._validators[tool].iter_errors(args),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        if errors:
            return self._error(
                call_id=call_id,
                tool=tool,
                error_code="INVALID_ARGUMENTS",
                message="arguments do not satisfy the canonical tool schema",
                validation_errors=[error.message for error in errors[:8]],
            )
        try:
            payload, provenance = {
                ToolName.CHART_OF_ACCOUNTS_LOOKUP: self._coa,
                ToolName.POLICY_LOOKUP: self._policy,
                ToolName.LEDGER_QUERY: self._ledger,
                ToolName.CALCULATOR: self._calculator,
                ToolName.TRANSACTION_MATCHING: self._match,
                ToolName.VARIANCE_DRILL_DOWN: self._variance,
            }[tool](args)
        except ToolSemanticError as exc:
            return ToolResult.seal(
                call_id=call_id,
                tool=tool,
                ok=False,
                result=exc.result,
                error_code=exc.error_code,
            )
        return ToolResult.seal(
            call_id=call_id,
            tool=tool,
            ok=True,
            result=payload,
            provenance=tuple(provenance),
        )

    def _error(
        self,
        *,
        call_id: str,
        tool: ToolName,
        error_code: str,
        message: str,
        **details: Any,
    ) -> ToolResult:
        return ToolResult.seal(
            call_id=call_id,
            tool=tool,
            ok=False,
            result={"message": message, **details},
            error_code=error_code,
        )

    def _coa(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        tokens = tuple(token for token in str(args["query"]).lower().split() if token)
        code = args.get("account_code")
        matches: list[dict[str, Any]] = []
        for account in self._world.accounts:
            if code is not None and account.code != code:
                continue
            haystack = (
                f"{account.code} {account.name} {account.domain} {account.normal_balance}"
            ).lower()
            if not all(token in haystack for token in tokens):
                continue
            matches.append(
                {
                    "code": account.code,
                    "name": account.name,
                    "domain": account.domain,
                    "normal_balance": account.normal_balance,
                }
            )
        if not matches:
            raise ToolSemanticError(
                "NOT_FOUND",
                "no chart-of-accounts row matched both query and account_code",
                query=args["query"],
                account_code=code,
            )
        matches.sort(key=lambda value: value["code"])
        return {"matches": matches}, [
            ProvenanceRef(
                source_id=match["code"],
                field="account_name",
                value=match["name"],
            )
            for match in matches
        ]

    def _policy(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        try:
            requested_date = date.fromisoformat(args["as_of"])
        except ValueError as exc:
            raise ToolSemanticError(
                "INVALID_ARGUMENTS", "as_of must be a real ISO calendar date"
            ) from exc
        candidates = []
        for policy in self._world.policies:
            if policy.policy_id != args["policy_id"]:
                continue
            starts = date.fromisoformat(policy.effective_from)
            ends = (
                date.fromisoformat(policy.effective_to) if policy.effective_to is not None else None
            )
            if starts <= requested_date and (ends is None or requested_date <= ends):
                candidates.append(policy)
        if not candidates:
            raise ToolSemanticError(
                "NOT_FOUND",
                "no policy version is effective on the requested date",
                policy_id=args["policy_id"],
                as_of=args["as_of"],
            )
        candidates.sort(
            key=lambda policy: (policy.effective_from, policy.version),
            reverse=True,
        )
        chosen = candidates[0]

        def render(policy: Any) -> dict[str, Any]:
            return {
                "policy_id": policy.policy_id,
                "version": policy.version,
                "effective_from": policy.effective_from,
                "effective_to": policy.effective_to,
                "threshold_minor": policy.threshold_minor,
                "action_above": policy.action_above,
                "action_at_or_below": policy.action_at_or_below,
                "rule_text": policy.rule_text,
                "superseded": policy.superseded,
            }

        payload: dict[str, Any] = {
            "requested_as_of": args["as_of"],
            "policy": render(chosen),
        }
        if args.get("include_history", False):
            payload["history"] = [
                render(policy)
                for policy in sorted(
                    (
                        policy
                        for policy in self._world.policies
                        if policy.policy_id == args["policy_id"]
                    ),
                    key=lambda policy: (policy.effective_from, policy.version),
                )
            ]
        return payload, [
            ProvenanceRef(
                source_id=f"{chosen.policy_id}:{chosen.version}",
                field="effective_version",
                value=chosen.version,
            ),
            ProvenanceRef(
                source_id=f"{chosen.policy_id}:{chosen.version}",
                field="threshold_minor",
                value=str(chosen.threshold_minor),
            ),
        ]

    def _ledger(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        limit = int(args.get("limit", 20))
        rows = []
        for row in sorted(self._world.ledger, key=lambda value: value.entry_id):
            if row.account_code != args["account_code"] or row.period != args["period"]:
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
            if len(rows) == limit:
                break
        if not rows:
            raise ToolSemanticError(
                "NOT_FOUND",
                "no ledger rows matched all filters",
                account_code=args["account_code"],
                period=args["period"],
                merchant_id=args.get("merchant_id"),
            )
        total = sum(row["amount_minor"] for row in rows)
        provenance = [
            ProvenanceRef(
                source_id=row["entry_id"],
                field="amount_minor",
                value=str(row["amount_minor"]),
            )
            for row in rows
        ]
        provenance.append(
            ProvenanceRef(
                source_id=f"ledger:{args['account_code']}:{args['period']}",
                field="total_amount_minor",
                value=str(total),
            )
        )
        return {"rows": rows, "total_amount_minor": total}, provenance

    def _calculator(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        op = args["op"]
        operands = [int(value) for value in args["operands"]]
        if op == "add":
            value = sum(operands)
            field = "result"
        elif op == "sub":
            value = operands[0] - sum(operands[1:])
            field = "result"
        elif op == "mul":
            value = 1
            for operand in operands:
                value *= operand
            field = "result"
        elif op == "abs_diff":
            value = abs(operands[0] - operands[1])
            field = "result"
        elif op == "pct_of":
            if operands[1] == 0:
                raise ToolSemanticError("DIVIDE_BY_ZERO", "pct_of denominator cannot be zero")
            value = (operands[0] * 10_000) // operands[1]
            field = "result_basis_points"
        else:  # pragma: no cover - schema closes this branch
            raise ToolSemanticError("INVALID_ARGUMENTS", f"unsupported op {op}")
        if abs(value) > RFC8785_SAFE_INTEGER_MAX:
            raise ToolSemanticError(
                "ARITHMETIC_OVERFLOW",
                "calculator result exceeds the canonical JSON safe integer domain",
            )
        payload = {"op": op, "operands": operands, field: value}
        return payload, [ProvenanceRef(source_id="calculator", field=field, value=str(value))]

    def _match(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        books = {entry.book_id: entry for entry in self._world.book_entries}
        banks = {event.bank_id: event for event in self._world.bank_events}
        missing = [identifier for identifier in args["book_ids"] if identifier not in books] + [
            identifier for identifier in args["bank_ids"] if identifier not in banks
        ]
        if missing:
            raise ToolSemanticError(
                "NOT_FOUND",
                "one or more transaction identifiers do not exist",
                missing_ids=sorted(missing),
            )
        book_sum = sum(books[identifier].amount_minor for identifier in args["book_ids"])
        bank_sum = sum(banks[identifier].amount_minor for identifier in args["bank_ids"])
        difference = book_sum - bank_sum
        tolerance = int(args["tolerance_minor"])
        payload = {
            "matched": abs(difference) <= tolerance,
            "book_sum_minor": book_sum,
            "bank_sum_minor": bank_sum,
            "difference_minor": difference,
            "tolerance_minor": tolerance,
        }
        provenance = [
            *(
                ProvenanceRef(
                    source_id=identifier,
                    field="amount_minor",
                    value=str(books[identifier].amount_minor),
                )
                for identifier in args["book_ids"]
            ),
            *(
                ProvenanceRef(
                    source_id=identifier,
                    field="amount_minor",
                    value=str(banks[identifier].amount_minor),
                )
                for identifier in args["bank_ids"]
            ),
            ProvenanceRef(
                source_id="transaction_match",
                field="difference_minor",
                value=str(difference),
            ),
        ]
        return payload, provenance

    def _variance(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[ProvenanceRef]]:
        all_drivers = [
            driver
            for driver in self._world.variance_drivers
            if driver.account_code == args["account_code"] and driver.period == args["period"]
        ]
        if not all_drivers:
            raise ToolSemanticError(
                "NOT_FOUND",
                "no variance drivers match account and period",
                account_code=args["account_code"],
                period=args["period"],
            )
        all_drivers.sort(key=lambda driver: (-abs(driver.impact_minor), driver.driver_id))
        returned = all_drivers[: int(args["top_k"])]
        full_total = sum(driver.impact_minor for driver in all_drivers)
        returned_total = sum(driver.impact_minor for driver in returned)
        payload = {
            "drivers": [
                {
                    "driver_id": driver.driver_id,
                    "impact_minor": driver.impact_minor,
                    "label": driver.label,
                }
                for driver in returned
            ],
            "returned_impact_minor": returned_total,
            "full_period_impact_minor": full_total,
            "omitted_impact_minor": full_total - returned_total,
            "driver_count": len(all_drivers),
        }
        provenance = [
            ProvenanceRef(
                source_id=driver.driver_id,
                field="impact_minor",
                value=str(driver.impact_minor),
            )
            for driver in all_drivers
        ]
        provenance.append(
            ProvenanceRef(
                source_id=f"variance:{args['account_code']}:{args['period']}",
                field="full_period_impact_minor",
                value=str(full_total),
            )
        )
        return payload, provenance


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _reject_escape_surface(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        bad = sorted(key for key in value if str(key).lower() in _FORBIDDEN_ARG_KEYS)
        if bad:
            raise SandboxSecurityError(f"{path}: forbidden argument keys: {bad}")
        for key, item in value.items():
            _reject_escape_surface(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_escape_surface(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
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
        "../",
    )
    if any(needle in lowered for needle in needles):
        raise SandboxSecurityError(f"{path}: value looks like shell/network escape")


def canonical_tool_result_bytes(result: ToolResult) -> bytes:
    return canonical_json_bytes(result.model_dump(mode="json"))


__all__ = [
    "FinanceAgentSandbox",
    "SandboxSecurityError",
    "ToolSemanticError",
    "canonical_tool_result_bytes",
]
