"""Fake-client coverage for local and Bedrock teacher adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from distillery.contracts.tasks import SplitName
from distillery.teachers.adapters import (
    BedrockConverseTeacher,
    BedrockError,
    LocalOpenWeightTeacher,
)
from distillery.teachers.budget import RequestGovernor
from distillery.teachers.cache import ImmutableTeacherCache
from distillery.teachers.errors import TeacherError, TeacherErrorCode
from distillery.teachers.hashing import prompt_sha256, request_sha256
from distillery.teachers.policy import assert_request_allowed
from distillery.teachers.types import (
    IntendedUse,
    LicenseStatus,
    TeacherBudget,
    TeacherModelRef,
    TeacherRequest,
    ToolSpec,
)

TODAY = date(2026, 7, 18)


class FakeBedrockClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def converse(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeLocalClient:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def generate(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        return dict(self.output)


def _governor(
    *,
    max_requests: int = 10,
    max_retries: int = 1,
    cost_ceiling: float = 1.0,
    input_rate: float = 0.01,
    output_rate: float = 0.02,
) -> RequestGovernor:
    return RequestGovernor(
        TeacherBudget(
            max_requests=max_requests,
            max_retries=max_retries,
            min_request_interval_seconds=0.0,
            cost_ceiling_usd=cost_ceiling,
            input_usd_per_1k_tokens=input_rate,
            output_usd_per_1k_tokens=output_rate,
            pricing_version="fake-pricing-v1",
        )
    )


def _gate(request: TeacherRequest) -> None:
    assert_request_allowed(request, on_date=TODAY)


def _response(
    *,
    model_id: str,
    output: dict[str, object] | str,
    input_tokens: int = 12,
    output_tokens: int = 8,
) -> dict[str, Any]:
    text = output if isinstance(output, str) else json.dumps(output)
    return {
        "modelId": model_id,
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
    }


def test_bedrock_uses_exact_profile_settings_and_records_policy(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    request = request_factory(model=claude_model)
    response = _response(model_id=claude_model.invocation_id, output=valid_output)
    client = FakeBedrockClient([response])
    teacher = BedrockConverseTeacher(
        client,
        governor=_governor(),
        request_gate=_gate,
    )

    result = teacher.generate(request)

    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent["modelId"] == claude_model.inference_profile_id
    assert sent["inferenceConfig"] == {
        "maxTokens": 64,
        "temperature": 0.0,
        "topP": 1.0,
        "stopSequences": ["END"],
    }
    assert sent["additionalModelRequestFields"] == {
        "performanceConfig": "standard",
        "seed": 7,
    }
    assert result.provenance.model_id == claude_model.model_id
    assert result.provenance.inference_profile_id == claude_model.inference_profile_id
    assert result.provenance.provider_policy_version == (request.provider_policy.policy_version)
    assert result.provenance.provider_policy_sha256 == (request.provider_policy.policy_sha256)
    assert result.provenance.output_use_policy_version == (request.output_use_policy.record_version)
    assert result.tokens.total_tokens == 20
    assert result.cost.total_usd > 0.0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            BedrockError("AccessDeniedException", "model unavailable"),
            TeacherErrorCode.ACCESS_DENIED,
        ),
        (
            BedrockError("AccessDeniedException", "use case form pending"),
            TeacherErrorCode.USE_CASE_PENDING,
        ),
    ],
)
def test_bedrock_access_failures_are_typed_without_retry(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    error: BedrockError,
    expected: TeacherErrorCode,
) -> None:
    client = FakeBedrockClient([error])
    teacher = BedrockConverseTeacher(
        client,
        governor=_governor(max_retries=3),
        request_gate=_gate,
    )
    with pytest.raises(TeacherError) as caught:
        teacher.generate(request_factory(model=claude_model))
    assert caught.value.code is expected
    assert len(client.calls) == 1


def test_bedrock_throttling_retries_are_bounded(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    client = FakeBedrockClient(
        [
            BedrockError("ThrottlingException", "slow down"),
            _response(model_id=claude_model.invocation_id, output=valid_output),
        ]
    )
    sleeps: list[float] = []
    teacher = BedrockConverseTeacher(
        client,
        governor=_governor(max_retries=1),
        request_gate=_gate,
        retry_sleep=sleeps.append,
    )
    result = teacher.generate(request_factory(model=claude_model))
    assert result.response_text
    assert len(client.calls) == 2
    assert sleeps == [0.25]


def test_bedrock_throttling_exhaustion_is_typed(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeBedrockClient(
        [
            BedrockError("ThrottlingException", "slow"),
            BedrockError("ThrottlingException", "still slow"),
        ]
    )
    teacher = BedrockConverseTeacher(
        client,
        governor=_governor(max_retries=1),
        request_gate=_gate,
        retry_sleep=lambda _delay: None,
    )
    with pytest.raises(TeacherError) as caught:
        teacher.generate(request_factory(model=claude_model))
    assert caught.value.code is TeacherErrorCode.RETRIES_EXHAUSTED
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("response_text", "expected"),
    [
        ("not json", TeacherErrorCode.MALFORMED_JSON),
        (
            json.dumps(
                {
                    "schema_version": "merchant_tagging.v1",
                    "task": "merchant_tagging",
                    "merchant_id": "m",
                    "merchant_name": "x",
                    "spend_category": "not-a-category",
                    "tags": ["recurring"],
                    "confidence": 0.5,
                }
            ),
            TeacherErrorCode.SCHEMA_REJECTED,
        ),
    ],
)
def test_bedrock_rejects_malformed_or_schema_invalid_json(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    response_text: str,
    expected: TeacherErrorCode,
) -> None:
    client = FakeBedrockClient(
        [_response(model_id=claude_model.invocation_id, output=response_text)]
    )
    teacher = BedrockConverseTeacher(
        client,
        governor=_governor(),
        request_gate=_gate,
    )
    with pytest.raises(TeacherError) as caught:
        teacher.generate(request_factory(model=claude_model))
    assert caught.value.code is expected


def test_bedrock_accepts_explicit_allowed_tool_call(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    tool = ToolSpec(
        name="submit_merchant_tagging",
        description="Submit a merchant tagging task response.",
        input_schema={"type": "object"},
    )
    request = request_factory(
        model=claude_model,
        tools=(tool,),
        allow_tool_use=True,
    )
    response = {
        "modelId": claude_model.invocation_id,
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_1",
                            "name": tool.name,
                            "input": valid_output,
                        }
                    }
                ],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }
    client = FakeBedrockClient([response])
    result = BedrockConverseTeacher(
        client,
        governor=_governor(),
        request_gate=_gate,
    ).generate(request)
    assert result.provenance.tool_use is True
    assert client.calls[0]["toolConfig"]["tools"][0]["toolSpec"]["name"] == tool.name


def test_bedrock_rejects_model_substitution(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    client = FakeBedrockClient(
        [_response(model_id="us.anthropic.claude-other", output=valid_output)]
    )
    with pytest.raises(TeacherError) as caught:
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request_factory(model=claude_model))
    assert caught.value.code is TeacherErrorCode.MODEL_SUBSTITUTION


def test_cost_ceiling_blocks_before_client_call(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeBedrockClient([])
    teacher = BedrockConverseTeacher(
        client,
        governor=_governor(cost_ceiling=0.0),
        request_gate=_gate,
    )
    with pytest.raises(TeacherError) as caught:
        teacher.generate(request_factory(model=claude_model))
    assert caught.value.code is TeacherErrorCode.COST_EXHAUSTED
    assert client.calls == []


def test_bedrock_rejects_logit_recipe_before_client_call(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeBedrockClient([])
    request = request_factory(
        model=claude_model,
        intended_use=IntendedUse.LOGIT_KD,
    )
    with pytest.raises(TeacherError) as caught:
        BedrockConverseTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.RECIPE_INCOMPATIBLE
    assert client.calls == []


def test_local_qwen_cache_is_immutable_and_records_hits(
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    request = request_factory(
        model=qwen_model,
        intended_use=IntendedUse.SEQUENCE_KD,
    )
    client = FakeLocalClient(
        {
            "model_id": qwen_model.model_id,
            "revision": qwen_model.revision,
            "response_text": json.dumps(valid_output),
            "input_tokens": 10,
            "output_tokens": 5,
        }
    )
    cache = ImmutableTeacherCache()
    teacher = LocalOpenWeightTeacher(
        client,
        governor=_governor(),
        cache=cache,
        request_gate=_gate,
    )
    first = teacher.generate(request)
    second = teacher.generate(request)
    assert len(client.calls) == 1
    assert first.provenance.cache_hit is False
    assert second.provenance.cache_hit is True
    assert second.cost.total_usd == 0.0
    assert len(cache) == 1

    cache._entry_digests[next(iter(cache._entry_digests))] = "0" * 64
    with pytest.raises(TeacherError) as caught:
        teacher.generate(request)
    assert caught.value.code is TeacherErrorCode.CACHE_INTEGRITY_FAILED


def test_license_gate_fails_closed_before_local_call(
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeLocalClient({})
    request = request_factory(
        model=qwen_model,
        intended_use=IntendedUse.SEQUENCE_KD,
        license_status=LicenseStatus.UNRESOLVED,
    )
    with pytest.raises(TeacherError) as caught:
        LocalOpenWeightTeacher(
            client,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.LICENSE_GATE_FAILED
    assert client.calls == []


def test_prompt_hash_is_deterministic_and_settings_bound(
    claude_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    request = request_factory(model=claude_model)
    clone = TeacherRequest.model_validate(request.model_dump(mode="python"))
    assert prompt_sha256(request) == prompt_sha256(clone)
    assert request_sha256(request) == request_sha256(clone)
    changed = request.model_copy(
        update={"settings": request.settings.model_copy(update={"max_tokens": 65})}
    )
    assert prompt_sha256(request) == prompt_sha256(changed)
    assert request_sha256(request) != request_sha256(changed)


@pytest.mark.parametrize(
    "split",
    [SplitName.TEST, SplitName.IID_TEST, SplitName.OOD_TEST],
)
def test_held_out_splits_are_rejected(
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
    split: SplitName,
) -> None:
    with pytest.raises(ValidationError, match="held-out split"):
        request_factory(
            model=qwen_model,
            intended_use=IntendedUse.SEQUENCE_KD,
            split=split,
        )


def test_label_fields_are_rejected(
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    with pytest.raises(ValidationError, match="forbidden teacher label"):
        request_factory(
            model=qwen_model,
            intended_use=IntendedUse.SEQUENCE_KD,
            input_payload={"memo": "x", "expected_output": {"answer": "secret"}},
        )
