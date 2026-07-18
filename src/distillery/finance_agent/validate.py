"""Full Finance Agent validation, including grounded arguments and world replay."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from distillery.contracts.hashing import canonical_json_bytes
from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    ToolName,
)
from distillery.finance_agent.prompts import build_system_prompt
from distillery.finance_agent.sandbox import (
    FinanceAgentSandbox,
    SandboxSecurityError,
    canonical_tool_result_bytes,
)
from distillery.finance_agent.tools import tool_definitions
from distillery.finance_agent.world import AgentWorld

_GOLD_ONLY_KEYS = frozenset(
    {
        "gold",
        "trajectory",
        "expected_output",
        "oracle",
        "tool_result",
        "final_answer",
        "result_bindings",
        "latent_state_hash",
    }
)


def validate_episode(example: AgentEpisodeEnvelope, *, world: AgentWorld) -> None:
    """Validate all seals and replay every tool call against the bound world."""
    AgentEpisodeEnvelope.model_validate(example.model_dump(mode="python"))
    if example.world_id != world.world_id or example.group_id != world.group_id:
        raise ValueError("episode world/group identity does not match replay world")
    if example.gold.oracle.latent_state_hash != world.latent_state_hash():
        raise ValueError("episode latent_state_hash does not match replay world")
    if canonical_json_bytes(example.model_input.public_world) != canonical_json_bytes(
        world.public_view()
    ):
        raise ValueError("sealed public world view does not match latent world projection")
    available_tools = tuple(definition.name for definition in example.model_input.tools)
    canonical_definitions = tool_definitions(available_tools)
    if canonical_json_bytes(
        [tool.model_dump(mode="json") for tool in example.model_input.tools]
    ) != canonical_json_bytes([tool.model_dump(mode="json") for tool in canonical_definitions]):
        raise ValueError("model-visible tool definitions drifted from canonical schemas")
    expected_prompt = build_system_prompt(
        public_world=world.public_view(),
        tools=canonical_definitions,
    )
    if example.model_input.system_prompt != expected_prompt:
        raise ValueError("system prompt drifted from sealed public world/tool definitions")
    assert_model_record_is_input_only(example.model_record())
    _validate_grounded_and_replayed_calls(example, world=world)
    _validate_result_bindings(example)


def _validate_grounded_and_replayed_calls(
    example: AgentEpisodeEnvelope,
    *,
    world: AgentWorld,
) -> None:
    sandbox = FinanceAgentSandbox(
        world,
        allowed_tools=tuple(definition.name for definition in example.model_input.tools),
    )
    visible_text = json.dumps(
        example.model_input.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
    )
    turns = example.gold.trajectory.turns
    for index, turn in enumerate(turns):
        if turn.tool_call is None:
            if turn.tool_result is not None:
                visible_text += json.dumps(
                    turn.tool_result.model_dump(mode="json"),
                    sort_keys=True,
                    ensure_ascii=False,
                )
            continue
        _assert_arguments_grounded(turn.tool_call.arguments, visible_text=visible_text)
        expected_result = turns[index + 1].tool_result
        if expected_result is None:
            raise ValueError(f"call {turn.tool_call.call_id} has no adjacent result")
        try:
            replayed = sandbox.execute(
                call_id=turn.tool_call.call_id,
                tool=turn.tool_call.tool,
                arguments=turn.tool_call.arguments,
            )
        except SandboxSecurityError as exc:
            raise ValueError(
                f"sandbox security rejection for gold call {turn.tool_call.call_id}: {exc}"
            ) from exc
        if canonical_tool_result_bytes(replayed) != canonical_tool_result_bytes(expected_result):
            raise ValueError(
                "tool-result bytes/provenance mismatch for "
                f"{turn.tool_call.call_id}: expected={expected_result.result_sha256} "
                f"replayed={replayed.result_sha256}"
            )


def _assert_arguments_grounded(arguments: Mapping[str, Any], *, visible_text: str) -> None:
    """Every scalar argument must occur in model input or a prior tool result."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if value is None or isinstance(value, bool):
            return
        serialized = json.dumps(value, ensure_ascii=False)
        grounded = (
            str(value) in visible_text if isinstance(value, str) else serialized in visible_text
        )
        if not grounded:
            raise ValueError(
                f"gold argument {path}={value!r} is absent from model-visible "
                "input and prior tool results"
            )

    visit(arguments, "$.arguments")


def _value_at(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for component in path:
        if isinstance(current, (tuple, list)):
            current = current[int(component)]
        elif isinstance(current, Mapping):
            current = current[component]
        else:
            raise ValueError(f"path component {component!r} cannot index {current!r}")
    return current


def _validate_result_bindings(example: AgentEpisodeEnvelope) -> None:
    results = {result.call_id: result for result in example.gold.trajectory.tool_results()}
    final = example.gold.expected_output.final_answer
    for binding in example.gold.expected_output.result_bindings:
        result = results.get(binding.call_id)
        if result is None:
            raise ValueError(f"result binding references missing call {binding.call_id}")
        answer_value = _value_at(final.structured, binding.answer_path)
        result_value = _value_at(result.result, binding.result_path)
        if canonical_json_bytes(answer_value) != canonical_json_bytes(result_value):
            raise ValueError(
                f"answer binding {binding.answer_path} does not use "
                f"{binding.call_id}:{binding.result_path}"
            )


def assert_model_record_is_input_only(record: Mapping[str, Any]) -> None:
    """Fail if trainer/eval model rows contain evaluator-only gold fields."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in _GOLD_ONLY_KEYS:
                    raise ValueError(f"gold-only key leaked into model record: {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(record, "$")


def replay_gold_tools(world: AgentWorld, example: AgentEpisodeEnvelope) -> None:
    """Compatibility name; performs full validation rather than partial replay."""
    validate_episode(example, world=world)


def assert_no_network_surface(arguments: Mapping[str, Any]) -> None:
    """Public negative check used by tests and descriptor review."""
    forbidden = {"url", "endpoint", "host", "shell", "command", "path", "socket"}
    for key, value in arguments.items():
        if key.lower() in forbidden:
            raise ValueError(f"network/shell argument key forbidden: {key}")
        if isinstance(value, Mapping):
            assert_no_network_surface(value)


def tools_used(example: AgentEpisodeEnvelope) -> tuple[ToolName, ...]:
    return tuple(call.tool for call in example.gold.trajectory.tool_calls())


__all__ = [
    "assert_model_record_is_input_only",
    "assert_no_network_surface",
    "replay_gold_tools",
    "tools_used",
    "validate_episode",
]
