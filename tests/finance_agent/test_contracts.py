"""Finance Agent contract invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.tasks import TaskId
from distillery.finance_agent.contracts import (
    SCHEMA_VERSION_FINANCE_AGENT,
    AgentTrajectory,
    CaseFamily,
    FinalAnswer,
    FinalAnswerKind,
    ToolCall,
    ToolName,
    TrajectoryTurn,
    TurnRole,
)
from distillery.finance_agent.generate import generate_agent_corpus


def test_tinyfable_task_ids_unchanged() -> None:
    assert set(TaskId) == {
        TaskId.TRANSACTION_REVIEW,
        TaskId.VARIANCE_ANALYSIS,
        TaskId.CASH_RECONCILIATION,
    }


def test_smoke_envelopes_use_finance_agent_schema() -> None:
    corpus = generate_agent_corpus("smoke", validate=True)
    assert corpus.manifest["envelope_schema_version"] == SCHEMA_VERSION_FINANCE_AGENT
    assert all(ex.schema_version == "finance_agent.v1" for ex in corpus.examples)
    assert all(ex.oracle.technique_id == "agent_trajectory.v1" for ex in corpus.examples)


def test_trajectory_requires_final_answer() -> None:
    with pytest.raises(ValidationError):
        AgentTrajectory(
            turns=(
                TrajectoryTurn(turn_index=0, role=TurnRole.USER, text="hello"),
                TrajectoryTurn(
                    turn_index=1,
                    role=TurnRole.ASSISTANT,
                    tool_call=ToolCall(
                        call_id="c1",
                        tool=ToolName.CALCULATOR,
                        arguments={"op": "add", "operands_minor": [1, 2]},
                    ),
                ),
            )
        )


def test_case_families_cover_hard_modes() -> None:
    required = {
        CaseFamily.WRONG_TOOL,
        CaseFamily.CORRECT_TOOL_WRONG_ARGS,
        CaseFamily.STALE_POLICY,
        CaseFamily.AMBIGUOUS_MERCHANT,
        CaseFamily.MULTI_STEP_RECONCILIATION,
        CaseFamily.ARITHMETIC_TRAP,
        CaseFamily.CONFLICTING_EVIDENCE,
        CaseFamily.REFUSAL_MISSING_DATA,
    }
    corpus = generate_agent_corpus("smoke", validate=True)
    present = {ex.case_family for ex in corpus.examples}
    assert required <= present


def test_refusal_final_answer_kind() -> None:
    corpus = generate_agent_corpus("smoke", validate=True)
    refusals = [ex for ex in corpus.examples if ex.case_family is CaseFamily.REFUSAL_MISSING_DATA]
    assert refusals
    for example in refusals:
        final = example.trajectory.turns[-1].final_answer
        assert final is not None
        assert final.kind is FinalAnswerKind.REFUSAL
        assert isinstance(final, FinalAnswer)
