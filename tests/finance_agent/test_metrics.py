"""Adversarial ordered-trajectory metric and baseline regressions."""

from __future__ import annotations

from distillery.finance_agent.baselines import (
    always_refuse_baseline,
    first_tool_invalid_args_baseline,
)
from distillery.finance_agent.contracts import (
    AgentTrajectory,
    CaseFamily,
    FinalAnswer,
    ToolCall,
    ToolName,
    ToolResult,
    TrajectoryTurn,
    TurnRole,
)
from distillery.finance_agent.generate import GeneratedAgentCorpus
from distillery.finance_agent.metrics import aggregate_metrics, score_episode


def _reindex(turns: list[TrajectoryTurn]) -> AgentTrajectory:
    return AgentTrajectory(
        turns=tuple(
            turn.model_copy(update={"turn_index": index}) for index, turn in enumerate(turns)
        )
    )


def _happy(corpus: GeneratedAgentCorpus):
    return next(
        example for example in corpus.examples if example.case_family is CaseFamily.HAPPY_PATH
    )


def _multi(corpus: GeneratedAgentCorpus):
    return next(
        example
        for example in corpus.examples
        if example.case_family is CaseFamily.MULTI_STEP_RECONCILIATION
    )


def test_exact_gold_path_scores_perfect_but_economics_remain_unknown(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    gold = _happy(smoke_corpus)
    metrics = score_episode(gold, predicted=gold.gold.trajectory)
    assert metrics.end_to_end_success
    assert metrics.action_order_accuracy == 1.0
    assert metrics.tool_result_exactness == 1.0
    assert metrics.latency_ms is None
    assert metrics.cost_usd_micros is None
    assert metrics.economics_measured is False


def test_wrong_tool_is_caught(smoke_corpus: GeneratedAgentCorpus) -> None:
    gold = _happy(smoke_corpus)
    original = gold.gold.trajectory
    call = ToolCall.seal(
        call_id="wrong",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands": [1, 2]},
    )
    result = ToolResult.seal(
        call_id="wrong",
        tool=ToolName.CALCULATOR,
        ok=True,
        result={"op": "add", "operands": [1, 2], "result": 3},
    )
    predicted = _reindex(
        [
            original.turns[0],
            TrajectoryTurn(turn_index=0, role=TurnRole.ASSISTANT, tool_call=call),
            TrajectoryTurn(turn_index=0, role=TurnRole.TOOL, tool_result=result),
            original.turns[-1],
        ]
    )
    metrics = score_episode(gold, predicted=predicted)
    assert metrics.tool_selection_accuracy == 0.0
    assert not metrics.end_to_end_success


def test_correct_tool_wrong_arguments_is_caught(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    gold = _happy(smoke_corpus)
    original = gold.gold.trajectory
    expected_call = original.tool_calls()[0]
    call = ToolCall.seal(
        call_id=expected_call.call_id,
        tool=expected_call.tool,
        arguments={
            "account_code": expected_call.arguments["account_code"],
            "period": "2099-01",
        },
    )
    result = ToolResult.seal(
        call_id=call.call_id,
        tool=call.tool,
        ok=False,
        result={"message": "no rows"},
        error_code="NOT_FOUND",
    )
    predicted = _reindex(
        [
            original.turns[0],
            TrajectoryTurn(turn_index=0, role=TurnRole.ASSISTANT, tool_call=call),
            TrajectoryTurn(turn_index=0, role=TurnRole.TOOL, tool_result=result),
            original.turns[-1],
        ]
    )
    metrics = score_episode(gold, predicted=predicted)
    assert metrics.tool_selection_accuracy == 1.0
    assert metrics.argument_exactness == 0.0
    assert metrics.tool_result_exactness == 0.0
    assert not metrics.end_to_end_success


def test_wrong_order_is_caught_even_with_same_calls_results_and_final(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    gold = _multi(smoke_corpus)
    turns = gold.gold.trajectory.turns
    predicted = _reindex([turns[0], turns[3], turns[4], turns[1], turns[2], turns[-1]])
    metrics = score_episode(gold, predicted=predicted)
    assert metrics.action_order_accuracy == 0.0
    assert metrics.argument_exactness == 0.0
    assert not metrics.end_to_end_success


def test_skipped_dependency_is_caught(smoke_corpus: GeneratedAgentCorpus) -> None:
    gold = _multi(smoke_corpus)
    turns = gold.gold.trajectory.turns
    predicted = _reindex([turns[0], turns[1], turns[2], turns[-1]])
    metrics = score_episode(gold, predicted=predicted)
    assert metrics.skipped_calls == 1
    assert metrics.tool_result_use < 1.0
    assert not metrics.end_to_end_success


def test_extra_call_is_caught(smoke_corpus: GeneratedAgentCorpus) -> None:
    gold = _happy(smoke_corpus)
    turns = list(gold.gold.trajectory.turns)
    call = ToolCall.seal(
        call_id="extra",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands": [1, 2]},
    )
    result = ToolResult.seal(
        call_id="extra",
        tool=ToolName.CALCULATOR,
        ok=True,
        result={"op": "add", "operands": [1, 2], "result": 3},
    )
    predicted = _reindex(
        [
            *turns[:-1],
            TrajectoryTurn(turn_index=0, role=TurnRole.ASSISTANT, tool_call=call),
            TrajectoryTurn(turn_index=0, role=TurnRole.TOOL, tool_result=result),
            turns[-1],
        ]
    )
    metrics = score_episode(gold, predicted=predicted)
    assert metrics.unnecessary_calls == 1
    assert metrics.action_order_accuracy == 0.0
    assert not metrics.end_to_end_success


def test_wrong_tool_result_bytes_and_result_use_are_caught(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    gold = _happy(smoke_corpus)
    turns = list(gold.gold.trajectory.turns)
    result = turns[2].tool_result
    assert result is not None
    wrong_payload = result.model_dump(mode="json")["result"]
    wrong_payload["total_amount_minor"] += 1
    wrong_result = ToolResult.seal(
        call_id=result.call_id,
        tool=result.tool,
        ok=True,
        result=wrong_payload,
        provenance=result.provenance,
    )
    turns[2] = turns[2].model_copy(update={"tool_result": wrong_result})
    metrics = score_episode(gold, predicted=AgentTrajectory(turns=tuple(turns)))
    assert metrics.tool_result_exactness == 0.0
    assert metrics.tool_result_use == 0.0
    assert not metrics.end_to_end_success


def test_wrong_final_value_is_caught(smoke_corpus: GeneratedAgentCorpus) -> None:
    gold = _happy(smoke_corpus)
    turns = list(gold.gold.trajectory.turns)
    final = turns[-1].final_answer
    assert final is not None
    structured = final.model_dump(mode="json")["structured"]
    structured["amount_minor"] += 1
    wrong_final = FinalAnswer(
        kind=final.kind,
        text="Wrong amount.",
        structured=structured,
        confidence=final.confidence,
        evidence=final.evidence,
    )
    turns[-1] = turns[-1].model_copy(update={"final_answer": wrong_final})
    metrics = score_episode(gold, predicted=AgentTrajectory(turns=tuple(turns)))
    assert metrics.final_answer_correctness == 0.0
    assert metrics.tool_result_use == 0.0
    assert not metrics.end_to_end_success


def test_non_tautological_baselines_do_not_pass(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    refusal_rows = [
        score_episode(example, predicted=always_refuse_baseline(example))
        for example in smoke_corpus.examples
    ]
    first_tool_rows = [
        score_episode(example, predicted=first_tool_invalid_args_baseline(example))
        for example in smoke_corpus.examples
    ]
    assert aggregate_metrics(refusal_rows)["end_to_end_success_rate"] == 0.0
    assert aggregate_metrics(first_tool_rows)["end_to_end_success_rate"] == 0.0
