"""Sandbox isolation and deterministic tool execution."""

from __future__ import annotations

import pytest

from distillery.finance_agent.contracts import ToolName
from distillery.finance_agent.sandbox import FinanceAgentSandbox, SandboxError
from distillery.finance_agent.world import build_agent_world


def test_sandbox_rejects_shell_keys() -> None:
    world = build_agent_world(seed=1, index=0)
    sandbox = FinanceAgentSandbox(world)
    with pytest.raises(SandboxError, match="forbidden argument keys"):
        sandbox.execute(
            call_id="c1",
            tool=ToolName.CALCULATOR,
            arguments={"op": "add", "operands_minor": [1], "shell": "rm -rf /"},
        )


def test_sandbox_rejects_url_escape_strings() -> None:
    world = build_agent_world(seed=1, index=0)
    sandbox = FinanceAgentSandbox(world)
    with pytest.raises(SandboxError, match="escape"):
        sandbox.execute(
            call_id="c1",
            tool=ToolName.CHART_OF_ACCOUNTS_LOOKUP,
            arguments={"query": "https://evil.example"},
        )


def test_calculator_deterministic() -> None:
    world = build_agent_world(seed=3, index=1)
    sandbox = FinanceAgentSandbox(world)
    first = sandbox.execute(
        call_id="c1",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands_minor": [10, 20, 30]},
    )
    second = sandbox.execute(
        call_id="c2",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands_minor": [10, 20, 30]},
    )
    assert first.ok and second.ok
    assert first.result["result_minor"] == 60
    assert second.result["result_minor"] == 60


def test_policy_lookup_skips_superseded_by_default() -> None:
    world = build_agent_world(seed=4, index=2, stale_policy=True)
    sandbox = FinanceAgentSandbox(world)
    result = sandbox.execute(
        call_id="c1",
        tool=ToolName.POLICY_LOOKUP,
        arguments={"policy_id": "pol_meal_limit", "as_of": world.as_of},
    )
    assert result.ok
    assert result.result["policy"]["version"] == "v2"
    assert result.result["policy"]["superseded"] is False
