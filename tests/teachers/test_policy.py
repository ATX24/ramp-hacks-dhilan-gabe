"""Written-authorization and provider-policy fail-closed tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from distillery.teachers.adapters import BedrockConverseTeacher
from distillery.teachers.budget import RequestGovernor
from distillery.teachers.errors import TeacherError, TeacherErrorCode
from distillery.teachers.policy import (
    AUTHORIZATION_LIMITATION,
    PROJECT_PROVIDER_POLICY_CITATIONS,
    assert_request_allowed,
)
from distillery.teachers.types import (
    IntendedUse,
    OutputStorage,
    RedistributionScope,
    ReviewStatus,
    TeacherBudget,
    TeacherModelRef,
    TeacherRequest,
    WrittenAuthorizationEvidence,
)

TODAY = date(2026, 7, 18)


class CountingClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def converse(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        assert self.response is not None
        return self.response


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


def _response(model: TeacherModelRef, output: dict[str, object]) -> dict[str, Any]:
    return {
        "modelId": model.invocation_id,
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": json.dumps(output)}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


@pytest.mark.parametrize(
    "intended_use",
    [
        IntendedUse.TRAINING,
        IntendedUse.SYNTHETIC_LABELS,
        IntendedUse.SEQUENCE_KD,
        IntendedUse.LOGIT_KD,
        IntendedUse.FINE_TUNING,
        IntendedUse.DERIVED_WEIGHTS,
    ],
)
def test_claude_weight_deriving_uses_need_authorization_and_make_zero_calls(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    intended_use: IntendedUse,
) -> None:
    client = CountingClient()
    request = request_factory(model=claude_model, intended_use=intended_use)
    with pytest.raises(TeacherError):
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert client.calls == []


@pytest.mark.parametrize(
    "intended_use",
    [
        IntendedUse.TRAINING,
        IntendedUse.SYNTHETIC_LABELS,
        IntendedUse.SEQUENCE_KD,
        IntendedUse.LOGIT_KD,
        IntendedUse.FINE_TUNING,
        IntendedUse.DERIVED_WEIGHTS,
    ],
)
def test_nova_cannot_be_priority_converted_into_qwen_training(
    nova_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    intended_use: IntendedUse,
) -> None:
    client = CountingClient()
    request = request_factory(model=nova_model, intended_use=intended_use)
    with pytest.raises(TeacherError):
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert client.calls == []


def test_exact_authorization_enables_claude_sequence_kd_and_is_provenanced(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
    valid_output: dict[str, object],
) -> None:
    evidence = authorization_factory()
    request = request_factory(
        model=claude_model,
        intended_use=IntendedUse.SEQUENCE_KD,
        written_authorization=evidence,
    )
    client = CountingClient(_response(claude_model, valid_output))
    result = BedrockConverseTeacher(
        client,
        governor=_governor(),
        request_gate=_gate,
    ).generate(request)

    assert len(client.calls) == 1
    assert result.provenance.written_authorization_sha256 == (evidence.evidence_sha256)
    assert result.provenance.authorization_limitation == AUTHORIZATION_LIMITATION
    serialized_call = json.dumps(client.calls[0], sort_keys=True)
    assert "provided_out_of_band" not in serialized_call
    assert evidence.evidence_sha256 not in serialized_call


def test_expired_authorization_makes_zero_calls(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
) -> None:
    evidence = authorization_factory(expiration_date=date(2026, 7, 17))
    request = request_factory(
        model=claude_model,
        intended_use=IntendedUse.SEQUENCE_KD,
        written_authorization=evidence,
    )
    client = CountingClient()
    with pytest.raises(TeacherError) as caught:
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.AUTHORIZATION_EXPIRED
    assert client.calls == []


@pytest.mark.parametrize(
    ("authorization_kwargs", "request_kwargs"),
    [
        (
            {"uses": (IntendedUse.BENCHMARK,)},
            {},
        ),
        (
            {"covered_models": ("anthropic.claude-opus-4-8-v1:0",)},
            {},
        ),
        (
            {"student_ids": ("Qwen/Qwen2.5-3B-Instruct",)},
            {},
        ),
        (
            {"storage": (OutputStorage.ENCRYPTED_PROJECT_STORAGE,)},
            {},
        ),
    ],
)
def test_authorization_scope_mismatch_makes_zero_calls(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
    authorization_kwargs: dict[str, object],
    request_kwargs: dict[str, object],
) -> None:
    evidence = authorization_factory(**authorization_kwargs)
    request = request_factory(
        model=claude_model,
        intended_use=IntendedUse.SEQUENCE_KD,
        written_authorization=evidence,
        **request_kwargs,
    )
    client = CountingClient()
    with pytest.raises(TeacherError) as caught:
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.AUTHORIZATION_SCOPE_MISMATCH
    assert client.calls == []


def test_no_redistribution_scope_cannot_create_derived_weights(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
) -> None:
    evidence = authorization_factory(redistribution=RedistributionScope.NONE)
    request = request_factory(
        model=claude_model,
        intended_use=IntendedUse.SEQUENCE_KD,
        written_authorization=evidence,
    )
    client = CountingClient()
    with pytest.raises(TeacherError) as caught:
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.AUTHORIZATION_SCOPE_MISMATCH
    assert client.calls == []


def test_authorization_evidence_tampering_is_rejected(
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
) -> None:
    evidence = authorization_factory()
    payload = evidence.model_dump(mode="python")
    payload["permitted_student_model_ids"] = ("Qwen/tampered",)
    with pytest.raises(ValidationError, match="evidence hash mismatch"):
        WrittenAuthorizationEvidence.model_validate(payload)


def test_wildcard_authorization_scope_is_forbidden(
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
) -> None:
    with pytest.raises(ValidationError, match="wildcard"):
        authorization_factory(covered_models=("*",))


def test_pending_use_case_fails_before_client_call(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    request = request_factory(
        model=claude_model,
        review_status=ReviewStatus.PENDING,
    )
    client = CountingClient()
    with pytest.raises(TeacherError) as caught:
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.USE_CASE_PENDING
    assert client.calls == []


def test_policy_evidence_names_required_terms() -> None:
    assert "AWS Service Terms §50.5" in PROJECT_PROVIDER_POLICY_CITATIONS
    assert "Anthropic Commercial Terms §D.4" in PROJECT_PROVIDER_POLICY_CITATIONS
    assert "Anthropic Acceptable Use Policy" in PROJECT_PROVIDER_POLICY_CITATIONS
