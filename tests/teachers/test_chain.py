"""Priority-chain attempt recording and terminal fallback rules."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from distillery.teachers.adapters import (
    BedrockConverseTeacher,
    BedrockError,
    LocalOpenWeightTeacher,
)
from distillery.teachers.budget import RequestGovernor
from distillery.teachers.chain import PriorityTeacherChain, TeacherCandidate
from distillery.teachers.errors import TeacherError, TeacherErrorCode
from distillery.teachers.policy import assert_request_allowed
from distillery.teachers.types import (
    AttemptOutcome,
    IntendedUse,
    TeacherBudget,
    TeacherModelFamily,
    TeacherModelRef,
    TeacherProvider,
    TeacherRequest,
    WrittenAuthorizationEvidence,
)

TODAY = date(2026, 7, 18)


class FakeBedrock:
    def __init__(self, outcome: dict[str, Any] | Exception) -> None:
        self.outcome = outcome
        self.calls = 0

    def converse(self, **_request: Any) -> dict[str, Any]:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeLocal:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = 0

    def generate(self, **_request: Any) -> dict[str, Any]:
        self.calls += 1
        return self.output


def _governor() -> RequestGovernor:
    return RequestGovernor(
        TeacherBudget(
            max_requests=10,
            max_retries=0,
            min_request_interval_seconds=0.0,
            cost_ceiling_usd=1.0,
            input_usd_per_1k_tokens=0.01,
            output_usd_per_1k_tokens=0.02,
            pricing_version="fake-v1",
        )
    )


def _gate(request: TeacherRequest) -> None:
    assert_request_allowed(request, on_date=TODAY)


def _response(model: TeacherModelRef, valid_output: dict[str, object]) -> dict[str, Any]:
    return {
        "modelId": model.invocation_id,
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": json.dumps(valid_output)}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def test_priority_chain_records_every_rejection_before_success(
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    first_model = TeacherModelRef(
        provider=TeacherProvider.BEDROCK,
        family=TeacherModelFamily.OTHER,
        model_id="vendor.model-one-v1",
    )
    second_model = TeacherModelRef(
        provider=TeacherProvider.BEDROCK,
        family=TeacherModelFamily.OTHER,
        model_id="vendor.model-two-v1",
    )
    first_request = request_factory(model=first_model)
    second_request = request_factory(model=second_model)
    first_client = FakeBedrock(BedrockError("AccessDeniedException", "not enabled"))
    second_client = FakeBedrock(_response(second_model, valid_output))
    chain = PriorityTeacherChain(
        (
            TeacherCandidate(
                BedrockConverseTeacher(
                    first_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                first_request,
            ),
            TeacherCandidate(
                BedrockConverseTeacher(
                    second_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                second_request,
            ),
        )
    )
    result = chain.generate()
    assert [attempt.outcome for attempt in result.attempts] == [
        AttemptOutcome.REJECTED,
        AttemptOutcome.SUCCEEDED,
    ]
    assert result.attempts[0].error_code == TeacherErrorCode.ACCESS_DENIED.value
    assert first_client.calls == 1
    assert second_client.calls == 1


def test_fable_access_failure_never_falls_back(
    claude_model: TeacherModelRef,
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    first_client = FakeBedrock(BedrockError("AccessDeniedException", "model unavailable"))
    local_client = FakeLocal({})
    chain = PriorityTeacherChain(
        (
            TeacherCandidate(
                BedrockConverseTeacher(
                    first_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                request_factory(model=claude_model),
            ),
            TeacherCandidate(
                LocalOpenWeightTeacher(
                    local_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                request_factory(model=qwen_model),
            ),
        )
    )
    with pytest.raises(TeacherError) as caught:
        chain.generate()
    assert caught.value.code is TeacherErrorCode.PREMIUM_FALLBACK_FORBIDDEN
    assert len(caught.value.payload.details["attempts"]) == 1
    assert first_client.calls == 1
    assert local_client.calls == 0


def test_prohibited_claude_request_cannot_fallback_to_qwen(
    claude_model: TeacherModelRef,
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    bedrock_client = FakeBedrock({})
    local_client = FakeLocal({})
    chain = PriorityTeacherChain(
        (
            TeacherCandidate(
                BedrockConverseTeacher(
                    bedrock_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                request_factory(
                    model=claude_model,
                    intended_use=IntendedUse.SEQUENCE_KD,
                ),
            ),
            TeacherCandidate(
                LocalOpenWeightTeacher(
                    local_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                request_factory(
                    model=qwen_model,
                    intended_use=IntendedUse.SEQUENCE_KD,
                ),
            ),
        )
    )
    with pytest.raises(TeacherError) as caught:
        chain.generate()
    assert caught.value.code is TeacherErrorCode.PREMIUM_FALLBACK_FORBIDDEN
    attempts = caught.value.payload.details["attempts"]
    assert attempts[0]["error_code"] == TeacherErrorCode.AUTHORIZATION_REQUIRED
    assert bedrock_client.calls == 0
    assert local_client.calls == 0


def test_inadequate_authorization_cannot_trigger_fallback(
    claude_model: TeacherModelRef,
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
) -> None:
    evidence = authorization_factory(student_ids=("Qwen/not-the-student",))
    bedrock_client = FakeBedrock({})
    local_client = FakeLocal({})
    chain = PriorityTeacherChain(
        (
            TeacherCandidate(
                BedrockConverseTeacher(
                    bedrock_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                request_factory(
                    model=claude_model,
                    intended_use=IntendedUse.SEQUENCE_KD,
                    written_authorization=evidence,
                ),
            ),
            TeacherCandidate(
                LocalOpenWeightTeacher(
                    local_client,
                    governor=_governor(),
                    request_gate=_gate,
                ),
                request_factory(
                    model=qwen_model,
                    intended_use=IntendedUse.SEQUENCE_KD,
                ),
            ),
        )
    )
    with pytest.raises(TeacherError) as caught:
        chain.generate()
    assert caught.value.code is TeacherErrorCode.PREMIUM_FALLBACK_FORBIDDEN
    assert bedrock_client.calls == 0
    assert local_client.calls == 0
