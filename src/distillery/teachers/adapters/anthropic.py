"""Direct Anthropic Messages adapter with runtime-only secret resolution.

Model IDs are caller-supplied and verified through the injected model probe.
No Fable or Opus alias is hardcoded. Direct Anthropic responses provide
sequence text/tool trajectories only and never satisfy ``logit.v1``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import SecretStr

from distillery.teachers.budget import RequestGovernor, estimate_token_count
from distillery.teachers.cache import ImmutableTeacherCache
from distillery.teachers.errors import (
    TeacherError,
    TeacherErrorCode,
    raise_teacher_error,
    teacher_error,
)
from distillery.teachers.hashing import cache_key_sha256, canonical_prompt_text
from distillery.teachers.policy import make_request_gate
from distillery.teachers.records import build_teacher_result, cached_result
from distillery.teachers.schema import require_valid_task_response
from distillery.teachers.secrets import (
    AnthropicSecretResolver,
    require_anthropic_secret,
)
from distillery.teachers.types import (
    AttemptOutcome,
    OutputRetention,
    OutputStorage,
    TeacherAttempt,
    TeacherProvider,
    TeacherRecipe,
    TeacherRequest,
    TeacherResult,
    TokenUsage,
)

_RESERVED_REQUEST_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "headers",
        "max_tokens",
        "messages",
        "model",
        "system",
        "tool_choice",
        "tools",
    }
)


class AnthropicModelDiscovery(Protocol):
    def probe_model(self, *, model_id: str) -> Mapping[str, Any]:
        """Return metadata containing the exact canonical ``id``."""


class AnthropicMessagesClient(AnthropicModelDiscovery, Protocol):
    def create_message(self, **request: Any) -> Mapping[str, Any]:
        """Call the official Messages API and return a JSON-like response."""


class AnthropicClientFactory(Protocol):
    def create(self, api_key: SecretStr) -> AnthropicMessagesClient:
        """Construct an authenticated client without retaining key metadata."""


class AnthropicAPIError(Exception):
    """Sanitized fake/wrapper error contract for tests and custom clients."""

    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code


class OfficialAnthropicClientFactory:
    """Optional wrapper around the official ``anthropic`` Python SDK."""

    def create(self, api_key: SecretStr) -> AnthropicMessagesClient:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("official Anthropic SDK is unavailable") from exc
        sdk_client = anthropic.Anthropic(api_key=api_key.get_secret_value())
        return _OfficialAnthropicClient(sdk_client)


class _OfficialAnthropicClient:
    def __init__(self, sdk_client: Any) -> None:
        self._sdk_client = sdk_client

    def probe_model(self, *, model_id: str) -> Mapping[str, Any]:
        response = self._sdk_client.models.retrieve(model_id)
        return _as_mapping(response)

    def create_message(self, **request: Any) -> Mapping[str, Any]:
        response = self._sdk_client.messages.create(**request)
        return _as_mapping(response)


class DirectAnthropicTeacher:
    def __init__(
        self,
        *,
        secret_resolver: AnthropicSecretResolver,
        client_factory: AnthropicClientFactory,
        governor: RequestGovernor,
        cache: ImmutableTeacherCache | None = None,
        request_gate: Callable[[TeacherRequest], None] | None = None,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._secret_resolver = secret_resolver
        self._client_factory = client_factory
        self._governor = governor
        self._cache = cache if cache is not None else ImmutableTeacherCache()
        self._request_gate = request_gate or make_request_gate()
        self._retry_sleep = retry_sleep

    @property
    def provider_name(self) -> str:
        return TeacherProvider.ANTHROPIC.value

    def supports_recipe(self, recipe: TeacherRecipe | None) -> bool:
        return recipe in {None, TeacherRecipe.SEQUENCE_V1}

    def generate(self, request: TeacherRequest) -> TeacherResult:
        if request.model.provider is not TeacherProvider.ANTHROPIC:
            raise_teacher_error(
                TeacherErrorCode.INVALID_REQUEST,
                "DirectAnthropicTeacher requires provider=anthropic",
                details={"provider": request.model.provider.value},
            )
        if not self.supports_recipe(request.recipe):
            raise_teacher_error(
                TeacherErrorCode.RECIPE_INCOMPATIBLE,
                "Anthropic Messages returns text/tool trajectories, not logits",
                details={"recipe": request.recipe.value if request.recipe else None},
            )
        if request.settings.do_sample or request.settings.seed is not None:
            raise_teacher_error(
                TeacherErrorCode.INVALID_REQUEST,
                "direct Anthropic settings cannot include do_sample or seed",
            )

        # Policy/authorization must fail before secret resolution or network.
        self._request_gate(request)
        use_cache = (
            request.output_storage is OutputStorage.IMMUTABLE_CACHE
            and request.output_use_policy.retention is OutputRetention.RETAINED
        )
        key = cache_key_sha256(request)
        if use_cache:
            stored = self._cache.get(key)
            if stored is not None:
                return cached_result(request, stored)

        secret = require_anthropic_secret(self._secret_resolver)
        raw_secret = secret.get_secret_value()
        try:
            client = self._client_factory.create(secret)
        except Exception:
            raise_teacher_error(
                TeacherErrorCode.SECRET_UNAVAILABLE,
                "failed to construct authenticated Anthropic client",
            )

        discovered = self._call_with_retries(
            lambda: client.probe_model(model_id=request.model.model_id),
            estimated_input_tokens=0,
            max_output_tokens=0,
            operation="model_probe",
        )
        _reject_secret_echo(discovered, raw_secret)
        returned_model_id = discovered.get("id")
        if not isinstance(returned_model_id, str) or returned_model_id != request.model.model_id:
            raise_teacher_error(
                TeacherErrorCode.MODEL_SUBSTITUTION,
                "Anthropic model probe did not return the exact requested model ID",
                details={
                    "requested_model_id": request.model.model_id,
                    "returned_model_id": (
                        returned_model_id if isinstance(returned_model_id, str) else None
                    ),
                },
            )

        payload = build_anthropic_request(request)
        _reject_secret_echo(payload, raw_secret)
        prompt = canonical_prompt_text(request)
        response = self._call_with_retries(
            lambda: client.create_message(**payload),
            estimated_input_tokens=estimate_token_count(prompt),
            max_output_tokens=request.settings.max_tokens,
            operation="message",
        )
        _reject_secret_echo(response, raw_secret)
        response_model_id = response.get("model")
        if response_model_id != returned_model_id:
            raise_teacher_error(
                TeacherErrorCode.MODEL_SUBSTITUTION,
                "Anthropic response model differs from the probed exact model",
                details={
                    "expected_model_id": returned_model_id,
                    "returned_model_id": (
                        response_model_id if isinstance(response_model_id, str) else None
                    ),
                },
            )
        response_content, used_tool = _extract_anthropic_content(request, response)
        canonical = require_valid_task_response(
            task=request.task,
            response=response_content,
        )
        usage = _anthropic_usage(response)
        cost = self._governor.record_usage(usage)
        result = build_teacher_result(
            request=request,
            response_text=canonical,
            tokens=usage,
            cost=cost,
            tool_use=used_tool,
            cache_hit=False,
            attempts=(
                TeacherAttempt(
                    model=request.model,
                    outcome=AttemptOutcome.SUCCEEDED,
                ),
            ),
            returned_model_id=returned_model_id,
        )
        return self._cache.put(key, result) if use_cache else result

    def _call_with_retries(
        self,
        call: Callable[[], Mapping[str, Any]],
        *,
        estimated_input_tokens: int,
        max_output_tokens: int,
        operation: str,
    ) -> Mapping[str, Any]:
        max_retries = self._governor.budget.max_retries
        for retry_index in range(max_retries + 1):
            self._governor.before_network_call(
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=max_output_tokens,
            )
            try:
                return call()
            except Exception as exc:
                mapped = _map_anthropic_error(exc)
                if mapped.code is not TeacherErrorCode.THROTTLED:
                    raise mapped from None
                if retry_index >= max_retries:
                    raise_teacher_error(
                        TeacherErrorCode.RETRIES_EXHAUSTED,
                        "Anthropic API rate-limit retries exhausted",
                        details={
                            "operation": operation,
                            "attempts": retry_index + 1,
                        },
                    )
                self._retry_sleep(min(4.0, 0.25 * (2**retry_index)))
        raise AssertionError("bounded retry loop exhausted unexpectedly")


def build_anthropic_request(request: TeacherRequest) -> dict[str, Any]:
    additional = request.settings.model_dump(mode="json")["additional_model_fields"]
    sensitive = sorted(key for key in additional if key.lower() in _RESERVED_REQUEST_KEYS)
    if sensitive:
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "additional Anthropic fields contain reserved or sensitive keys",
            details={"rejected_field_count": len(sensitive)},
        )
    payload: dict[str, Any] = {
        "model": request.model.model_id,
        "max_tokens": request.settings.max_tokens,
        "system": request.settings.system_prompt,
        "messages": [
            {
                "role": "user",
                "content": canonical_prompt_text(request),
            }
        ],
        "temperature": request.settings.temperature,
        "top_p": request.settings.top_p,
        **additional,
    }
    if request.settings.stop_sequences:
        payload["stop_sequences"] = list(request.settings.stop_sequences)
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in request.tools
        ]
        payload["tool_choice"] = {"type": "auto"}
    return payload


def _extract_anthropic_content(
    request: TeacherRequest,
    response: Mapping[str, Any],
) -> tuple[str | dict[str, Any], bool]:
    content = response.get("content")
    if not isinstance(content, list) or not content:
        raise_teacher_error(
            TeacherErrorCode.MALFORMED_JSON,
            "Anthropic response has no content blocks",
        )
    text_parts: list[str] = []
    tool_blocks: list[Mapping[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        if block.get("type") == "tool_use":
            tool_blocks.append(block)
    if tool_blocks:
        if not request.allow_tool_use:
            raise_teacher_error(
                TeacherErrorCode.TOOL_USE_REJECTED,
                "Anthropic returned tool use when the request forbids it",
            )
        if len(tool_blocks) != 1 or any(text.strip() for text in text_parts):
            raise_teacher_error(
                TeacherErrorCode.TOOL_USE_REJECTED,
                "Anthropic response must contain one unambiguous tool call",
            )
        tool = tool_blocks[0]
        allowed_names = {spec.name for spec in request.tools}
        name = tool.get("name")
        tool_input = tool.get("input")
        if name not in allowed_names or not isinstance(tool_input, dict):
            raise_teacher_error(
                TeacherErrorCode.TOOL_USE_REJECTED,
                "Anthropic returned an unknown tool or non-object input",
                details={"tool_name": str(name)},
            )
        return tool_input, True
    text = "".join(text_parts).strip()
    if not text:
        raise_teacher_error(
            TeacherErrorCode.MALFORMED_JSON,
            "Anthropic response contains no text or allowed tool call",
        )
    return text, False


def _anthropic_usage(response: Mapping[str, Any]) -> TokenUsage:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "Anthropic response is missing token usage",
        )
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
    ):
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "Anthropic usage requires integer input/output token counts",
        )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _map_anthropic_error(exc: Exception) -> TeacherError:
    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return teacher_error(
            TeacherErrorCode.ACCESS_DENIED,
            "Anthropic authentication or model access was denied",
            details={"http_status": status},
        )
    if status == 404:
        return teacher_error(
            TeacherErrorCode.MODEL_UNAVAILABLE,
            "the exact requested Anthropic model is unavailable",
            details={"http_status": status},
        )
    if status == 429 or (isinstance(status, int) and status >= 500):
        return teacher_error(
            TeacherErrorCode.THROTTLED,
            "Anthropic API is rate limited or temporarily unavailable",
            details={"http_status": status},
        )
    return teacher_error(
        TeacherErrorCode.INVALID_REQUEST,
        "Anthropic API request failed",
        details={"http_status": status if isinstance(status, int) else None},
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError("official Anthropic SDK returned an unsupported response type")


def _reject_secret_echo(value: Any, raw_secret: str) -> None:
    if raw_secret and _contains_exact_text(value, raw_secret):
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "Anthropic transport attempted to expose secret material",
        )


def _contains_exact_text(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(
            _contains_exact_text(key, needle) or _contains_exact_text(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_exact_text(item, needle) for item in value)
    return False


__all__ = [
    "AnthropicAPIError",
    "AnthropicClientFactory",
    "AnthropicMessagesClient",
    "AnthropicModelDiscovery",
    "DirectAnthropicTeacher",
    "OfficialAnthropicClientFactory",
    "build_anthropic_request",
]
