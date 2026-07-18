"""Ordered per-turn metrics for Finance Agent trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field, StrictBool

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import NonNegativeSafeInt, canonical_json_bytes
from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    AgentTrajectory,
    EconomicsObservation,
    ToolCall,
    ToolResult,
)

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ToolTurnScore(FrozenModel):
    position: NonNegativeSafeInt
    expected_call_id: str | None
    predicted_call_id: str | None
    tool_exact: StrictBool
    arguments_exact: StrictBool
    result_exact: StrictBool


class AgentMetrics(FrozenModel):
    schema_version: str = "finance_agent.metrics.v2"
    tool_selection_accuracy: FiniteFloat = Field(ge=0.0, le=1.0)
    action_order_accuracy: FiniteFloat = Field(ge=0.0, le=1.0)
    argument_exactness: FiniteFloat = Field(ge=0.0, le=1.0)
    tool_result_exactness: FiniteFloat = Field(ge=0.0, le=1.0)
    tool_result_use: FiniteFloat = Field(ge=0.0, le=1.0)
    final_answer_correctness: FiniteFloat = Field(ge=0.0, le=1.0)
    unnecessary_calls: NonNegativeSafeInt
    skipped_calls: NonNegativeSafeInt
    latency_ms: NonNegativeSafeInt | None
    cost_usd_micros: NonNegativeSafeInt | None
    economics_measured: StrictBool
    turn_scores: tuple[ToolTurnScore, ...]
    end_to_end_success: StrictBool


def _pairs(trajectory: AgentTrajectory) -> list[tuple[ToolCall, ToolResult]]:
    pairs: list[tuple[ToolCall, ToolResult]] = []
    turns = trajectory.turns
    for index, turn in enumerate(turns):
        if turn.tool_call is None:
            continue
        result = turns[index + 1].tool_result
        if result is None:  # AgentTrajectory already rejects this.
            raise ValueError("tool call is missing adjacent result")
        pairs.append((turn.tool_call, result))
    return pairs


def _ratio(matches: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else matches / denominator


def _value_at(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for component in path:
        if isinstance(current, (tuple, list)):
            current = current[int(component)]
        elif isinstance(current, Mapping):
            current = current[component]
        else:
            raise KeyError(component)
    return current


def score_episode(
    gold: AgentEpisodeEnvelope,
    *,
    predicted: AgentTrajectory,
    economics: EconomicsObservation | None = None,
) -> AgentMetrics:
    """Score ordered calls, exact arguments/results, result use, and final answer."""
    gold_pairs = _pairs(gold.gold.trajectory)
    predicted_pairs = _pairs(predicted)
    denominator = max(len(gold_pairs), len(predicted_pairs))
    tool_matches = 0
    argument_matches = 0
    result_matches = 0
    turn_scores: list[ToolTurnScore] = []
    for position in range(denominator):
        expected = gold_pairs[position] if position < len(gold_pairs) else None
        actual = predicted_pairs[position] if position < len(predicted_pairs) else None
        tool_exact = bool(
            expected is not None and actual is not None and expected[0].tool is actual[0].tool
        )
        arguments_exact = bool(
            tool_exact
            and expected is not None
            and actual is not None
            and canonical_json_bytes(expected[0].arguments)
            == canonical_json_bytes(actual[0].arguments)
        )
        result_exact = bool(
            expected is not None
            and actual is not None
            and canonical_json_bytes(expected[1].model_dump(mode="json"))
            == canonical_json_bytes(actual[1].model_dump(mode="json"))
        )
        tool_matches += int(tool_exact)
        argument_matches += int(arguments_exact)
        result_matches += int(result_exact)
        turn_scores.append(
            ToolTurnScore(
                position=position,
                expected_call_id=expected[0].call_id if expected else None,
                predicted_call_id=actual[0].call_id if actual else None,
                tool_exact=tool_exact,
                arguments_exact=arguments_exact,
                result_exact=result_exact,
            )
        )

    gold_tools = tuple(call.tool for call, _ in gold_pairs)
    predicted_tools = tuple(call.tool for call, _ in predicted_pairs)
    action_order_accuracy = 1.0 if gold_tools == predicted_tools else 0.0

    predicted_results = {result.call_id: result for _, result in predicted_pairs}
    predicted_final = predicted.turns[-1].final_answer
    assert predicted_final is not None
    binding_matches = 0
    bindings = gold.gold.expected_output.result_bindings
    for binding in bindings:
        result = predicted_results.get(binding.call_id)
        if result is None:
            continue
        try:
            answer_value = _value_at(predicted_final.structured, binding.answer_path)
            result_value = _value_at(result.result, binding.result_path)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if canonical_json_bytes(answer_value) == canonical_json_bytes(result_value):
            binding_matches += 1
    tool_result_use = _ratio(binding_matches, len(bindings))

    expected_final = gold.gold.expected_output.final_answer
    final_exact = canonical_json_bytes(
        predicted_final.model_dump(mode="json")
    ) == canonical_json_bytes(expected_final.model_dump(mode="json"))
    unnecessary_calls = max(0, len(predicted_pairs) - len(gold_pairs))
    skipped_calls = max(0, len(gold_pairs) - len(predicted_pairs))
    selection_accuracy = _ratio(tool_matches, denominator)
    argument_exactness = _ratio(argument_matches, denominator)
    result_exactness = _ratio(result_matches, denominator)

    observed = economics or EconomicsObservation()
    end_to_end = (
        action_order_accuracy == 1.0
        and selection_accuracy == 1.0
        and argument_exactness == 1.0
        and result_exactness == 1.0
        and tool_result_use == 1.0
        and final_exact
        and unnecessary_calls == 0
        and skipped_calls == 0
    )
    return AgentMetrics(
        tool_selection_accuracy=selection_accuracy,
        action_order_accuracy=action_order_accuracy,
        argument_exactness=argument_exactness,
        tool_result_exactness=result_exactness,
        tool_result_use=tool_result_use,
        final_answer_correctness=1.0 if final_exact else 0.0,
        unnecessary_calls=unnecessary_calls,
        skipped_calls=skipped_calls,
        latency_ms=observed.latency_ms,
        cost_usd_micros=observed.cost_usd_micros,
        economics_measured=observed.source == "measured",
        turn_scores=tuple(turn_scores),
        end_to_end_success=end_to_end,
    )


def aggregate_metrics(rows: list[AgentMetrics]) -> dict[str, Any]:
    names = (
        "tool_selection_accuracy",
        "action_order_accuracy",
        "argument_exactness",
        "tool_result_exactness",
        "tool_result_use",
        "final_answer_correctness",
    )
    if not rows:
        return {
            "n": 0,
            **{name: 0.0 for name in names},
            "unnecessary_calls_mean": 0.0,
            "skipped_calls_mean": 0.0,
            "latency_ms_mean": None,
            "cost_usd_micros_mean": None,
            "measured_economics_n": 0,
            "end_to_end_success_rate": 0.0,
        }
    measured_latency = [row.latency_ms for row in rows if row.latency_ms is not None]
    measured_cost = [row.cost_usd_micros for row in rows if row.cost_usd_micros is not None]
    n = len(rows)
    return {
        "n": n,
        **{name: sum(float(getattr(row, name)) for row in rows) / n for name in names},
        "unnecessary_calls_mean": sum(row.unnecessary_calls for row in rows) / n,
        "skipped_calls_mean": sum(row.skipped_calls for row in rows) / n,
        "latency_ms_mean": (
            sum(measured_latency) / len(measured_latency) if measured_latency else None
        ),
        "cost_usd_micros_mean": (
            sum(measured_cost) / len(measured_cost) if measured_cost else None
        ),
        "measured_economics_n": sum(row.economics_measured for row in rows),
        "end_to_end_success_rate": (
            sum(1.0 if row.end_to_end_success else 0.0 for row in rows) / n
        ),
    }


__all__ = [
    "AgentMetrics",
    "ToolTurnScore",
    "aggregate_metrics",
    "score_episode",
]
