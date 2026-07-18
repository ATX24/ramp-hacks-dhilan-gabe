"""Fake-only direct Anthropic provider and secret-boundary tests."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from distillery.teachers.adapters.anthropic import (
    AnthropicAPIError,
    DirectAnthropicTeacher,
)
from distillery.teachers.budget import RequestGovernor
from distillery.teachers.cache import ImmutableTeacherCache
from distillery.teachers.chain import PriorityTeacherChain, TeacherCandidate
from distillery.teachers.errors import TeacherError, TeacherErrorCode
from distillery.teachers.policy import AUTHORIZATION_LIMITATION, assert_request_allowed
from distillery.teachers.secrets import (
    ANTHROPIC_SECRET_NAME,
    EnvironmentAnthropicSecretResolver,
)
from distillery.teachers.types import (
    IntendedUse,
    TeacherBudget,
    TeacherModelFamily,
    TeacherModelRef,
    TeacherProvider,
    TeacherRecipe,
    TeacherRequest,
    TeacherResult,
    ToolSpec,
    WrittenAuthorizationEvidence,
)

TODAY = date(2026, 7, 18)
DIRECT_MODEL_ID = "claude-fable-exact-id-returned-by-test-probe"


class FakeSecretResolver:
    def __init__(self, secret: SecretStr | None) -> None:
        self.secret = secret
        self.calls = 0

    def resolve(self, name: str) -> SecretStr | None:
        assert name == ANTHROPIC_SECRET_NAME
        self.calls += 1
        return self.secret


class FakeAnthropicClient:
    def __init__(
        self,
        *,
        probe_outcomes: list[Mapping[str, Any] | Exception],
        message_outcomes: list[Mapping[str, Any] | Exception],
    ) -> None:
        self.probe_outcomes = list(probe_outcomes)
        self.message_outcomes = list(message_outcomes)
        self.probe_requests: list[str] = []
        self.message_requests: list[dict[str, Any]] = []

    def probe_model(self, *, model_id: str) -> Mapping[str, Any]:
        self.probe_requests.append(model_id)
        outcome = self.probe_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def create_message(self, **request: Any) -> Mapping[str, Any]:
        self.message_requests.append(request)
        outcome = self.message_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClientFactory:
    def __init__(self, client: FakeAnthropicClient) -> None:
        self.client = client
        self.calls = 0

    def create(self, api_key: SecretStr) -> FakeAnthropicClient:
        assert isinstance(api_key, SecretStr)
        self.calls += 1
        return self.client


def _model(model_id: str = DIRECT_MODEL_ID) -> TeacherModelRef:
    return TeacherModelRef(
        provider=TeacherProvider.ANTHROPIC,
        family=TeacherModelFamily.CLAUDE,
        model_id=model_id,
    )


def _governor(
    *,
    max_requests: int = 10,
    max_retries: int = 1,
    cost_ceiling: float = 1.0,
) -> RequestGovernor:
    return RequestGovernor(
        TeacherBudget(
            max_requests=max_requests,
            max_retries=max_retries,
            min_request_interval_seconds=0.0,
            cost_ceiling_usd=cost_ceiling,
            input_usd_per_1k_tokens=0.01,
            output_usd_per_1k_tokens=0.02,
            pricing_version="anthropic-fake-v1",
        )
    )


def _gate(request: TeacherRequest) -> None:
    assert_request_allowed(request, on_date=TODAY)


def _secret() -> SecretStr:
    return SecretStr(secrets.token_urlsafe(24))


def _response(
    output: dict[str, object],
    *,
    model_id: str = DIRECT_MODEL_ID,
) -> dict[str, Any]:
    return {
        "id": "msg_fake",
        "model": model_id,
        "content": [{"type": "text", "text": json.dumps(output)}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _evaluation_request(
    request_factory: Callable[..., TeacherRequest],
    *,
    model: TeacherModelRef | None = None,
) -> TeacherRequest:
    return request_factory(
        model=model or _model(),
        seed=None,
        additional_model_fields={},
    )


def _authorized_request(
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
    *,
    model: TeacherModelRef | None = None,
    authorization_models: tuple[str, ...] = (DIRECT_MODEL_ID,),
) -> TeacherRequest:
    source = model or _model()
    evidence = authorization_factory(
        covered_models=(),
        covered_anthropic_models=authorization_models,
    )
    return request_factory(
        model=source,
        intended_use=IntendedUse.SEQUENCE_KD,
        written_authorization=evidence,
        seed=None,
        additional_model_fields={},
    )


def test_direct_anthropic_probes_exact_model_and_records_returned_id(
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[_response(valid_output)],
    )
    resolver = FakeSecretResolver(_secret())
    factory = FakeClientFactory(client)
    result = DirectAnthropicTeacher(
        secret_resolver=resolver,
        client_factory=factory,
        governor=_governor(),
        request_gate=_gate,
    ).generate(_evaluation_request(request_factory))

    assert client.probe_requests == [DIRECT_MODEL_ID]
    assert client.message_requests[0]["model"] == DIRECT_MODEL_ID
    assert result.provenance.returned_model_id == DIRECT_MODEL_ID
    assert result.provenance.written_authorization_sha256 is None
    assert factory.calls == 1


def test_direct_anthropic_missing_key_fails_before_client_creation(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeAnthropicClient(probe_outcomes=[], message_outcomes=[])
    resolver = FakeSecretResolver(None)
    factory = FakeClientFactory(client)
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=resolver,
            client_factory=factory,
            governor=_governor(),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    assert caught.value.code is TeacherErrorCode.SECRET_UNAVAILABLE
    assert factory.calls == 0
    assert client.probe_requests == []


def test_environment_resolver_reads_only_named_anthropic_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ANTHROPIC_SECRET_NAME, raising=False)
    resolver = EnvironmentAnthropicSecretResolver()
    assert resolver.resolve(ANTHROPIC_SECRET_NAME) is None
    with pytest.raises(ValueError, match="only ANTHROPIC_API_KEY"):
        resolver.resolve("SOME_OTHER_SECRET")


@pytest.mark.parametrize("status", [401, 403])
def test_direct_anthropic_auth_errors_are_redacted(
    request_factory: Callable[..., TeacherRequest],
    status: int,
) -> None:
    secret = _secret()
    raw_sentinel = secret.get_secret_value()
    client = FakeAnthropicClient(
        probe_outcomes=[
            AnthropicAPIError(
                status,
                f"authorization: Bearer {raw_sentinel}; x-api-key={raw_sentinel}",
            )
        ],
        message_outcomes=[],
    )
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=FakeSecretResolver(secret),
            client_factory=FakeClientFactory(client),
            governor=_governor(),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    serialized = json.dumps(caught.value.payload.model_dump(mode="json"))
    assert caught.value.code is TeacherErrorCode.ACCESS_DENIED
    assert raw_sentinel not in str(caught.value)
    assert raw_sentinel not in repr(caught.value)
    assert raw_sentinel not in serialized


def test_direct_anthropic_model_unavailable_is_typed(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeAnthropicClient(
        probe_outcomes=[AnthropicAPIError(404, "unavailable")],
        message_outcomes=[],
    )
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=FakeSecretResolver(_secret()),
            client_factory=FakeClientFactory(client),
            governor=_governor(),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    assert caught.value.code is TeacherErrorCode.MODEL_UNAVAILABLE
    assert client.message_requests == []


def test_direct_anthropic_rate_limit_retries_are_bounded(
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[
            AnthropicAPIError(429, "rate limit"),
            _response(valid_output),
        ],
    )
    sleeps: list[float] = []
    result = DirectAnthropicTeacher(
        secret_resolver=FakeSecretResolver(_secret()),
        client_factory=FakeClientFactory(client),
        governor=_governor(max_retries=1),
        request_gate=_gate,
        retry_sleep=sleeps.append,
    ).generate(_evaluation_request(request_factory))
    assert result.response_text
    assert len(client.message_requests) == 2
    assert sleeps == [0.25]


def test_direct_anthropic_cost_cap_blocks_message_request(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[],
    )
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=FakeSecretResolver(_secret()),
            client_factory=FakeClientFactory(client),
            governor=_governor(cost_ceiling=0.0),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    assert caught.value.code is TeacherErrorCode.COST_EXHAUSTED
    assert client.probe_requests == [DIRECT_MODEL_ID]
    assert client.message_requests == []


def test_direct_authorization_mismatch_makes_zero_network_calls(
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
) -> None:
    request = _authorized_request(
        request_factory,
        authorization_factory,
        authorization_models=("claude-opus-other-exact-id",),
    )
    client = FakeAnthropicClient(probe_outcomes=[], message_outcomes=[])
    resolver = FakeSecretResolver(_secret())
    factory = FakeClientFactory(client)
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=resolver,
            client_factory=factory,
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.AUTHORIZATION_SCOPE_MISMATCH
    assert resolver.calls == 0
    assert factory.calls == 0
    assert client.probe_requests == []


def test_direct_authorized_sequence_kd_is_hash_bound_and_cached(
    request_factory: Callable[..., TeacherRequest],
    authorization_factory: Callable[..., WrittenAuthorizationEvidence],
    valid_output: dict[str, object],
) -> None:
    request = _authorized_request(request_factory, authorization_factory)
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[_response(valid_output)],
    )
    cache = ImmutableTeacherCache()
    teacher = DirectAnthropicTeacher(
        secret_resolver=FakeSecretResolver(_secret()),
        client_factory=FakeClientFactory(client),
        governor=_governor(),
        cache=cache,
        request_gate=_gate,
    )
    first = teacher.generate(request)
    second = teacher.generate(request)
    assert len(client.probe_requests) == 1
    assert len(client.message_requests) == 1
    assert second.provenance.cache_hit is True
    assert first.provenance.written_authorization_sha256 == (
        request.written_authorization.evidence_sha256
    )
    assert first.provenance.authorization_limitation == AUTHORIZATION_LIMITATION


def test_direct_anthropic_tool_trajectory_is_schema_validated(
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    tool = ToolSpec(
        name="submit_merchant_tagging",
        description="Submit task output.",
        input_schema={"type": "object"},
    )
    request = request_factory(
        model=_model(),
        seed=None,
        additional_model_fields={},
        tools=(tool,),
        allow_tool_use=True,
    )
    response = {
        "model": DIRECT_MODEL_ID,
        "content": [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": tool.name,
                "input": valid_output,
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[response],
    )
    result = DirectAnthropicTeacher(
        secret_resolver=FakeSecretResolver(_secret()),
        client_factory=FakeClientFactory(client),
        governor=_governor(),
        request_gate=_gate,
    ).generate(request)
    assert result.provenance.tool_use is True
    assert client.message_requests[0]["tools"][0]["name"] == tool.name


def test_direct_anthropic_secret_never_serializes(
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    secret = _secret()
    raw_sentinel = secret.get_secret_value()
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[_response(valid_output)],
    )
    request = _evaluation_request(request_factory)
    result = DirectAnthropicTeacher(
        secret_resolver=FakeSecretResolver(secret),
        client_factory=FakeClientFactory(client),
        governor=_governor(),
        request_gate=_gate,
    ).generate(request)
    serialized = json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "wire": client.message_requests,
        },
        sort_keys=True,
    )
    assert raw_sentinel not in serialized


def test_direct_anthropic_rejects_secret_echo_without_leaking_it(
    request_factory: Callable[..., TeacherRequest],
    valid_output: dict[str, object],
) -> None:
    secret = _secret()
    raw_sentinel = secret.get_secret_value()
    echoed = {**valid_output, "merchant_name": raw_sentinel}
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[_response(echoed)],
    )
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=FakeSecretResolver(secret),
            client_factory=FakeClientFactory(client),
            governor=_governor(),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    assert raw_sentinel not in str(caught.value)
    assert raw_sentinel not in json.dumps(caught.value.payload.model_dump(mode="json"))


def test_auth_fields_cannot_enter_generation_config(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    with pytest.raises(ValidationError, match="auth/header"):
        request_factory(
            model=_model(),
            seed=None,
            additional_model_fields={"transport": {"headers": {"authorization": "not-stored"}}},
        )


def test_direct_anthropic_exact_model_probe_rejects_substitution(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": "claude-opus-different-exact-id"}],
        message_outcomes=[],
    )
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=FakeSecretResolver(_secret()),
            client_factory=FakeClientFactory(client),
            governor=_governor(),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    assert caught.value.code is TeacherErrorCode.MODEL_SUBSTITUTION
    assert client.message_requests == []


def test_direct_anthropic_request_cap_includes_model_probe(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    client = FakeAnthropicClient(
        probe_outcomes=[{"id": DIRECT_MODEL_ID}],
        message_outcomes=[],
    )
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=FakeSecretResolver(_secret()),
            client_factory=FakeClientFactory(client),
            governor=_governor(max_requests=1),
            request_gate=_gate,
        ).generate(_evaluation_request(request_factory))
    assert caught.value.code is TeacherErrorCode.REQUEST_CAP_EXCEEDED
    assert client.probe_requests == [DIRECT_MODEL_ID]
    assert client.message_requests == []


def test_direct_anthropic_rejects_logit_before_secret_resolution(
    request_factory: Callable[..., TeacherRequest],
) -> None:
    resolver = FakeSecretResolver(_secret())
    request = request_factory(
        model=_model(),
        intended_use=IntendedUse.LOGIT_KD,
        recipe=TeacherRecipe.LOGIT_V1,
        seed=None,
        additional_model_fields={},
    )
    client = FakeAnthropicClient(probe_outcomes=[], message_outcomes=[])
    with pytest.raises(TeacherError) as caught:
        DirectAnthropicTeacher(
            secret_resolver=resolver,
            client_factory=FakeClientFactory(client),
            governor=_governor(),
            request_gate=_gate,
        ).generate(request)
    assert caught.value.code is TeacherErrorCode.RECIPE_INCOMPATIBLE
    assert resolver.calls == 0
    assert client.probe_requests == []


class _NeverCalledGenerator:
    provider_name = "local"

    def __init__(self) -> None:
        self.calls = 0

    def supports_recipe(self, _recipe: TeacherRecipe | None) -> bool:
        return True

    def generate(self, _request: TeacherRequest) -> TeacherResult:
        self.calls += 1
        raise AssertionError("fallback must not run")


def test_direct_fable_model_failure_never_crosses_provider_boundary(
    qwen_model: TeacherModelRef,
    request_factory: Callable[..., TeacherRequest],
) -> None:
    direct_request = _evaluation_request(request_factory)
    qwen_request = request_factory(
        model=qwen_model,
        seed=None,
        additional_model_fields={},
    )
    direct_client = FakeAnthropicClient(
        probe_outcomes=[AnthropicAPIError(404, "unavailable")],
        message_outcomes=[],
    )
    fallback = _NeverCalledGenerator()
    chain = PriorityTeacherChain(
        (
            TeacherCandidate(
                DirectAnthropicTeacher(
                    secret_resolver=FakeSecretResolver(_secret()),
                    client_factory=FakeClientFactory(direct_client),
                    governor=_governor(),
                    request_gate=_gate,
                ),
                direct_request,
            ),
            TeacherCandidate(fallback, qwen_request),
        )
    )
    with pytest.raises(TeacherError) as caught:
        chain.generate()
    assert caught.value.code is TeacherErrorCode.PREMIUM_FALLBACK_FORBIDDEN
    assert fallback.calls == 0
