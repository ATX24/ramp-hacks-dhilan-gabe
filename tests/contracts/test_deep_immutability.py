"""Adversarial mutation probes for every nested public JSON boundary."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from distillery.contracts.errors import (
    DistilleryErrorCode,
    ErrorPayload,
)
from distillery.contracts.hashing import content_sha256
from distillery.contracts.recipes import AUTO_SEQUENCE_RESPONSES_REASONS
from distillery.contracts.run import DistillationRun, RunFailure
from distillery.contracts.states import RunState
from distillery.contracts.tasks import FinanceTaskEnvelope

HEX64 = "a" * 64
NOW = datetime(2026, 7, 18, tzinfo=UTC)


def _envelope() -> FinanceTaskEnvelope:
    return FinanceTaskEnvelope(
        example_id="ex_immutability_001",
        world_id="world_immutability_001",
        group_id="grp_immutability_001",
        task="transaction_review",
        difficulty="easy",
        input={"nested": {"items": [{"amount_minor": 100}]}},
        expected_output={
            "schema_version": "transaction_review.v1",
            "task": "transaction_review",
            "gl_account": "6100",
            "journal_entry": [
                {"account": "6100", "side": "debit", "amount_minor": 100},
                {"account": "2100", "side": "credit", "amount_minor": 100},
            ],
            "policy_action": "approve",
            "rule_ids": ["POL-1"],
            "evidence": [],
            "confidence": 0.9,
        },
        oracle={
            "generator_revision": "contracts-v1",
            "latent_state_hash": f"sha256:{HEX64}",
        },
        provenance={
            "split": "train",
            "template_family": "immutable_v1",
            "label_source": "oracle",
        },
    )


def test_task_envelope_nested_input_and_output_are_immutable() -> None:
    envelope = _envelope()
    before = content_sha256(envelope)
    with pytest.raises(TypeError):
        envelope.input["nested"]["items"][0]["amount_minor"] = 200  # type: ignore[index]
    with pytest.raises(TypeError):
        envelope.expected_output["journal_entry"][0]["amount_minor"] = 200  # type: ignore[index]
    assert content_sha256(envelope) == before


def test_run_failure_and_error_details_are_immutable() -> None:
    failure = RunFailure(
        code=DistilleryErrorCode.AWS_JOB_FAILED,
        message="failed",
        details={"provider": {"events": [{"attempt": 1}]}},
    )
    run = DistillationRun(
        run_id="run_immutability_001",
        dataset_id="ds_immutability_001",
        state=RunState.QUEUED,
        manifest_sha256=HEX64,
        requested_recipe="auto",
        resolved_recipe="sequence.v1",
        resolver_reasons=AUTO_SEQUENCE_RESPONSES_REASONS,
        created_at=NOW,
        updated_at=NOW,
    ).transition(RunState.FAILED, at=NOW, failure=failure)
    before = run.resource_hash()
    with pytest.raises(TypeError):
        run.failure.details["provider"]["events"][0]["attempt"] = 2  # type: ignore[index,union-attr]
    assert run.resource_hash() == before

    payload = ErrorPayload(
        code=DistilleryErrorCode.INVALID_DATASET,
        message="invalid",
        details={"records": [{"line": 1}]},
        retryable=False,
    )
    with pytest.raises(TypeError):
        payload.details["records"][0]["line"] = 2  # type: ignore[index]
