"""Finance Agent envelope, visibility, and hard-case contract invariants."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from distillery.contracts.tasks import LabelSource, TaskId
from distillery.finance_agent.contracts import (
    CaseFamily,
    OracleMeta,
    ToolCall,
    ToolName,
)
from distillery.finance_agent.generate import GeneratedAgentCorpus
from distillery.finance_agent.validate import assert_model_record_is_input_only


def test_tinyfable_task_ids_unchanged() -> None:
    assert set(TaskId) == {
        TaskId.TRANSACTION_REVIEW,
        TaskId.VARIANCE_ANALYSIS,
        TaskId.CASH_RECONCILIATION,
    }


def test_model_records_are_input_only_and_gold_is_separate(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = smoke_corpus.examples[0]
    model_record = example.model_record()
    assert_model_record_is_input_only(model_record)
    serialized = json.dumps(model_record, sort_keys=True)
    for forbidden in (
        '"gold"',
        '"trajectory"',
        '"expected_output"',
        '"final_answer"',
        '"tool_result"',
    ):
        assert forbidden not in serialized
    gold_record = example.gold_record()
    assert gold_record["model_input_sha256"] == example.model_input.model_input_sha256
    assert "gold" in gold_record


def test_model_input_contains_and_hashes_prompt_schemas_and_public_ids(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    for example in smoke_corpus.examples:
        model_input = example.model_input
        assert model_input.system_prompt
        assert model_input.system_prompt_sha256
        assert model_input.public_world_sha256
        assert model_input.tool_schemas_sha256
        assert all(tool.input_schema for tool in model_input.tools)
        public = model_input.public_world
        assert public["as_of"] in model_input.system_prompt
        assert public["periods"][0] in model_input.system_prompt
        assert public["entity"]["entity_id"] in model_input.system_prompt
        assert public["accounts"][0]["code"] in model_input.system_prompt
        recon = public["reconciliation_sets"][0]
        assert recon["book_ids"]
        assert recon["bank_ids"]


def test_all_required_hard_case_families_exist(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    assert {
        CaseFamily.WRONG_TOOL,
        CaseFamily.CORRECT_TOOL_WRONG_ARGS,
        CaseFamily.STALE_POLICY,
        CaseFamily.AMBIGUOUS_MERCHANT,
        CaseFamily.MULTI_STEP_RECONCILIATION,
        CaseFamily.ARITHMETIC_TRAP,
        CaseFamily.CONFLICTING_EVIDENCE,
        CaseFamily.REFUSAL_MISSING_DATA,
    } <= {example.case_family for example in smoke_corpus.examples}


def test_hard_cases_contain_actual_recovery_trajectories(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    wrong_tool = next(
        example for example in smoke_corpus.examples if example.case_family is CaseFamily.WRONG_TOOL
    )
    assert [call.tool for call in wrong_tool.gold.trajectory.tool_calls()] == [
        ToolName.CALCULATOR,
        ToolName.POLICY_LOOKUP,
    ]

    wrong_args = next(
        example
        for example in smoke_corpus.examples
        if example.case_family is CaseFamily.CORRECT_TOOL_WRONG_ARGS
    )
    wrong_results = wrong_args.gold.trajectory.tool_results()
    assert wrong_results[0].ok is False
    assert wrong_results[0].error_code == "NOT_FOUND"
    assert wrong_results[1].ok is True

    stale = next(
        example
        for example in smoke_corpus.examples
        if example.case_family is CaseFamily.STALE_POLICY
    )
    versions = [
        result.result["policy"]["version"] for result in stale.gold.trajectory.tool_results()
    ]
    assert versions == ["v1", "v2"]


def test_ambiguous_case_is_real_multi_turn_user_dialogue(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    example = next(
        item for item in smoke_corpus.examples if item.case_family is CaseFamily.AMBIGUOUS_MERCHANT
    )
    assert [message.turn_index for message in example.model_input.user_messages] == [0, 2]
    assistant_clarification = example.gold.trajectory.turns[1]
    assert assistant_clarification.text is not None
    assert "merchant_id" in assistant_clarification.text


def test_oracle_and_teacher_label_evidence_cannot_be_conflated() -> None:
    with pytest.raises(ValidationError, match="teacher labels require"):
        OracleMeta(
            latent_state_hash=f"sha256:{'a' * 64}",
            label_source=LabelSource.TEACHER,
        )


def test_tool_call_argument_hash_rejects_mutation() -> None:
    call = ToolCall.seal(
        call_id="c1",
        tool=ToolName.CALCULATOR,
        arguments={"op": "add", "operands": [1, 2]},
    )
    payload = call.model_dump(mode="json")
    payload["arguments"]["operands"] = [1, 3]
    with pytest.raises(ValidationError, match="arguments_sha256"):
        ToolCall.model_validate(payload)


def test_generated_economics_are_honestly_unmeasured(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    assert all(example.economics.source == "unmeasured" for example in smoke_corpus.examples)
    assert all(example.economics.latency_ms is None for example in smoke_corpus.examples)
    assert all(example.economics.cost_usd_micros is None for example in smoke_corpus.examples)
