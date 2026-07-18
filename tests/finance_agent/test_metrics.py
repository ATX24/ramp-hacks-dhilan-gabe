"""Metric behavior for imperfect trajectories."""

from __future__ import annotations

from distillery.finance_agent.contracts import (
    AgentTrajectory,
    CaseFamily,
    FinalAnswer,
    FinalAnswerKind,
    TrajectoryTurn,
    TurnRole,
)
from distillery.finance_agent.generate import generate_agent_corpus
from distillery.finance_agent.metrics import score_episode


def test_wrong_tool_prediction_penalizes_selection() -> None:
    corpus = generate_agent_corpus("smoke")
    gold = next(ex for ex in corpus.examples if ex.case_family is CaseFamily.WRONG_TOOL)
    bad = AgentTrajectory(
        turns=(
            TrajectoryTurn(turn_index=0, role=TurnRole.USER, text=gold.user_goal),
            TrajectoryTurn(
                turn_index=1,
                role=TurnRole.ASSISTANT,
                final_answer=FinalAnswer(
                    kind=FinalAnswerKind.ANSWER,
                    text="guessing without tools",
                    structured={},
                    confidence=0.1,
                ),
            ),
        )
    )
    metrics = score_episode(gold, predicted=bad)
    assert metrics.tool_selection_accuracy == 0.0
    assert metrics.final_answer_correctness == 0.0
    assert metrics.end_to_end_success is False
