"""Evaluation metrics for Finance Agent trajectories."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, StrictBool

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import NonNegativeSafeInt
from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    AgentTrajectory,
    ToolCall,
    ToolName,
)

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class AgentMetrics(FrozenModel):
    schema_version: str = "finance_agent.metrics.v1"
    tool_selection_accuracy: FiniteFloat = Field(ge=0.0, le=1.0)
    argument_exactness: FiniteFloat = Field(ge=0.0, le=1.0)
    tool_result_use: FiniteFloat = Field(ge=0.0, le=1.0)
    final_answer_correctness: FiniteFloat = Field(ge=0.0, le=1.0)
    unnecessary_calls: NonNegativeSafeInt
    latency_ms: NonNegativeSafeInt
    cost_usd_micros: NonNegativeSafeInt
    end_to_end_success: StrictBool


def _predicted_calls(trajectory: AgentTrajectory) -> list[ToolCall]:
    return [turn.tool_call for turn in trajectory.turns if turn.tool_call is not None]


def _call_key(call: ToolCall) -> tuple[str, str]:
    import json

    return (
        call.tool.value,
        json.dumps(call.model_dump(mode="json")["arguments"], sort_keys=True, ensure_ascii=False),
    )


def score_episode(
    gold: AgentEpisodeEnvelope,
    *,
    predicted: AgentTrajectory | None = None,
    latency_ms: int | None = None,
    cost_usd_micros: int | None = None,
) -> AgentMetrics:
    """Score a predicted trajectory against a gold episode.

    When ``predicted`` is omitted, scores the gold trajectory against itself
    (oracle sanity / smoke metrics).
    """
    pred = predicted or gold.trajectory
    gold_calls = list(gold.expected_output.gold_tool_calls)
    pred_calls = _predicted_calls(pred)

    required = set(gold.expected_output.required_tools)
    pred_tools = {call.tool for call in pred_calls}
    if not required:
        tool_selection = 1.0 if not pred_tools else 0.0
    else:
        tool_selection = len(required & pred_tools) / len(required)

    if not gold_calls:
        argument_exactness = 1.0 if not pred_calls else 0.0
    else:
        gold_keys = {_call_key(call) for call in gold_calls}
        pred_keys = {_call_key(call) for call in pred_calls}
        argument_exactness = len(gold_keys & pred_keys) / len(gold_keys)

    final = pred.turns[-1].final_answer
    assert final is not None
    gold_final = gold.expected_output.final_answer
    final_ok = final.model_dump(mode="json") == gold_final.model_dump(mode="json")

    provenance_ids = {
        ref.source_id
        for turn in pred.turns
        if turn.tool_result is not None
        for ref in turn.tool_result.provenance
    }
    evidence_ids = {ref.source_id for ref in final.evidence}
    if gold.expected_output.required_tools:
        if not provenance_ids or not evidence_ids:
            tool_result_use = 0.0
        else:
            tool_result_use = len(evidence_ids & provenance_ids) / len(evidence_ids)
    else:
        tool_result_use = 1.0 if not evidence_ids else 0.0

    unnecessary = 0
    gold_tools = {call.tool for call in gold_calls}
    for call in pred_calls:
        if call.tool in gold.expected_output.forbidden_tools:
            unnecessary += 1
        elif call.tool not in gold.expected_output.required_tools and call.tool not in gold_tools:
            unnecessary += 1
    if len(pred_calls) > gold.expected_output.max_tool_calls:
        unnecessary += len(pred_calls) - gold.expected_output.max_tool_calls

    latency = gold.estimated_latency_ms if latency_ms is None else latency_ms
    cost = gold.estimated_cost_usd_micros if cost_usd_micros is None else cost_usd_micros
    end_to_end = (
        tool_selection == 1.0
        and argument_exactness == 1.0
        and final_ok
        and unnecessary == 0
        and tool_result_use == 1.0
    )
    return AgentMetrics(
        tool_selection_accuracy=tool_selection,
        argument_exactness=argument_exactness,
        tool_result_use=tool_result_use,
        final_answer_correctness=1.0 if final_ok else 0.0,
        unnecessary_calls=unnecessary,
        latency_ms=latency,
        cost_usd_micros=cost,
        end_to_end_success=end_to_end,
    )


def aggregate_metrics(rows: list[AgentMetrics]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "tool_selection_accuracy": 0.0,
            "argument_exactness": 0.0,
            "tool_result_use": 0.0,
            "final_answer_correctness": 0.0,
            "unnecessary_calls_mean": 0.0,
            "latency_ms_mean": 0.0,
            "cost_usd_micros_mean": 0.0,
            "end_to_end_success_rate": 0.0,
        }
    n = len(rows)
    return {
        "n": n,
        "tool_selection_accuracy": sum(r.tool_selection_accuracy for r in rows) / n,
        "argument_exactness": sum(r.argument_exactness for r in rows) / n,
        "tool_result_use": sum(r.tool_result_use for r in rows) / n,
        "final_answer_correctness": sum(r.final_answer_correctness for r in rows) / n,
        "unnecessary_calls_mean": sum(r.unnecessary_calls for r in rows) / n,
        "latency_ms_mean": sum(r.latency_ms for r in rows) / n,
        "cost_usd_micros_mean": sum(r.cost_usd_micros for r in rows) / n,
        "end_to_end_success_rate": sum(1.0 if r.end_to_end_success else 0.0 for r in rows) / n,
    }


__all__ = ["AgentMetrics", "aggregate_metrics", "score_episode", "ToolName"]
