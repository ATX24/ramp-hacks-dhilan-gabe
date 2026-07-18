"""Episode validation helpers for Finance Agent envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distillery.finance_agent.contracts import AgentEpisodeEnvelope, ToolName
from distillery.finance_agent.sandbox import FinanceAgentSandbox, SandboxError
from distillery.finance_agent.world import AgentWorld


def validate_episode(example: AgentEpisodeEnvelope) -> None:
    """Validate envelope invariants beyond Pydantic model checks."""
    AgentEpisodeEnvelope.model_validate(example.model_dump(mode="python"))
    if example.oracle.technique_id != "agent_trajectory.v1":
        raise ValueError("oracle.technique_id must be agent_trajectory.v1")
    if example.oracle.generator_revision != "finance_agent.v1":
        raise ValueError("oracle.generator_revision must be finance_agent.v1")
    calls = [
        turn.tool_call
        for turn in example.trajectory.turns
        if turn.tool_call is not None
    ]
    if len(calls) > example.expected_output.max_tool_calls:
        raise ValueError("trajectory exceeds max_tool_calls")
    for call in calls:
        if call.tool in example.expected_output.forbidden_tools:
            raise ValueError(f"gold trajectory used forbidden tool {call.tool}")


def replay_gold_tools(world: AgentWorld, example: AgentEpisodeEnvelope) -> None:
    """Re-execute gold tool calls; results must match trajectory tool turns."""
    sandbox = FinanceAgentSandbox(world, allowed_tools=example.available_tools)
    result_by_call = {
        turn.tool_result.call_id: turn.tool_result
        for turn in example.trajectory.turns
        if turn.tool_result is not None
    }
    for call in example.expected_output.gold_tool_calls:
        try:
            replayed = sandbox.execute(
                call_id=call.call_id,
                tool=call.tool,
                arguments=call.arguments,
            )
        except SandboxError as exc:
            raise ValueError(f"sandbox rejected gold call {call.call_id}: {exc}") from exc
        expected = result_by_call.get(call.call_id)
        if expected is None:
            raise ValueError(f"missing tool result for gold call {call.call_id}")
        if replayed.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValueError(f"non-deterministic or drifted tool result for {call.call_id}")


def assert_no_network_surface(arguments: Mapping[str, Any]) -> None:
    for key in arguments:
        if key.lower() in {"url", "endpoint", "host", "shell", "command"}:
            raise ValueError(f"network/shell argument key forbidden: {key}")


def tools_used(example: AgentEpisodeEnvelope) -> tuple[ToolName, ...]:
    return tuple(
        turn.tool_call.tool
        for turn in example.trajectory.turns
        if turn.tool_call is not None
    )
