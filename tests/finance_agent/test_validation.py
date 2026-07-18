"""Adversarial world replay, provenance, and argument-grounding validation."""

from __future__ import annotations

import pytest

from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    AgentGold,
    AgentTrajectory,
    CaseFamily,
    ProvenanceRef,
    ToolCall,
    ToolResult,
)
from distillery.finance_agent.generate import GeneratedAgentCorpus
from distillery.finance_agent.validate import validate_episode
from distillery.finance_agent.world import build_agent_world


def _reseal(
    example: AgentEpisodeEnvelope,
    trajectory: AgentTrajectory,
) -> AgentEpisodeEnvelope:
    gold = AgentGold.seal(
        trajectory=trajectory,
        expected_output=example.gold.expected_output,
        oracle=example.gold.oracle,
    )
    return AgentEpisodeEnvelope.seal(
        example_id=example.example_id,
        world_id=example.world_id,
        group_id=example.group_id,
        difficulty=example.difficulty,
        case_family=example.case_family,
        model_input=example.model_input,
        gold=gold,
        provenance=example.provenance,
        economics=example.economics,
    )


def test_validation_replays_every_call_in_multi_step_trajectory(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = next(
        item
        for item in smoke_corpus.examples
        if item.case_family is CaseFamily.MULTI_STEP_RECONCILIATION
    )
    turns = list(example.gold.trajectory.turns)
    second_result = turns[4].tool_result
    assert second_result is not None
    payload = second_result.model_dump(mode="json")["result"]
    payload["result"] += 1
    turns[4] = turns[4].model_copy(
        update={
            "tool_result": ToolResult.seal(
                call_id=second_result.call_id,
                tool=second_result.tool,
                ok=True,
                result=payload,
                provenance=second_result.provenance,
            )
        }
    )
    mutated = _reseal(example, AgentTrajectory(turns=tuple(turns)))
    with pytest.raises(ValueError, match="tool-result bytes/provenance mismatch"):
        validate_episode(mutated, world=smoke_corpus.worlds[example.world_id])


def test_validation_rejects_provenance_drift_even_when_result_payload_matches(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = next(
        item for item in smoke_corpus.examples if item.case_family is CaseFamily.HAPPY_PATH
    )
    turns = list(example.gold.trajectory.turns)
    result = turns[2].tool_result
    assert result is not None
    turns[2] = turns[2].model_copy(
        update={
            "tool_result": ToolResult.seal(
                call_id=result.call_id,
                tool=result.tool,
                ok=result.ok,
                result=result.model_dump(mode="json")["result"],
                provenance=(
                    ProvenanceRef(
                        source_id="forged",
                        field="amount_minor",
                        value="1",
                    ),
                ),
            )
        }
    )
    mutated = _reseal(example, AgentTrajectory(turns=tuple(turns)))
    with pytest.raises(ValueError, match="tool-result bytes/provenance mismatch"):
        validate_episode(mutated, world=smoke_corpus.worlds[example.world_id])


def test_validation_rejects_latent_only_gold_argument_before_replay(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = next(
        item for item in smoke_corpus.examples if item.case_family is CaseFamily.HAPPY_PATH
    )
    turns = list(example.gold.trajectory.turns)
    original = turns[1].tool_call
    assert original is not None
    ungrounded = ToolCall.seal(
        call_id=original.call_id,
        tool=original.tool,
        arguments={
            "account_code": original.arguments["account_code"],
            "period": "2099-01",
        },
    )
    turns[1] = turns[1].model_copy(update={"tool_call": ungrounded})
    result = turns[2].tool_result
    assert result is not None
    turns[2] = turns[2].model_copy(
        update={
            "tool_result": ToolResult.seal(
                call_id=result.call_id,
                tool=result.tool,
                ok=False,
                result={"message": "no rows"},
                error_code="NOT_FOUND",
            )
        }
    )
    mutated = _reseal(example, AgentTrajectory(turns=tuple(turns)))
    with pytest.raises(ValueError, match="absent from model-visible"):
        validate_episode(mutated, world=smoke_corpus.worlds[example.world_id])


def test_validation_rejects_wrong_world_even_with_same_domain(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = smoke_corpus.examples[0]
    wrong_world = build_agent_world(seed=999, index=0, domain="travel")
    with pytest.raises(ValueError, match="world/group identity"):
        validate_episode(example, world=wrong_world)
