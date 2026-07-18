"""Bedrock Converse adapter for text/tool sequence responses only.

No AWS client is created here. Callers inject a Converse-compatible client.
Bedrock does not expose logits, so ``logit.v1`` always fails before invocation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from distillery.teachers.budget import RequestGovernor, estimate_token_count
from distillery.teachers.cache import ImmutableTeacherCache
from distillery.teachers.errors import (
    TeacherError,
    TeacherErrorCode,
    raise_teacher_error,
)
from distillery.teachers.hashing import cache_key_sha256, canonical_prompt_text
from distillery.teachers.policy import make_request_gate
from distillery.teachers.records import build_teacher_result, cached_result
from distillery.teachers.schema import require_valid_task_response
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


class BedrockConverseClient(Protocol):
    def converse(self, **request: Any) -> Mapping[str, Any]:
        """Match ``bedrock-runtime.converse`` without importing boto3."""


class BedrockError(Exception):
    """Small fake/client-neutral service error used by tests and wrappers."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BedrockConverseTeacher:
    def __init__(
        self,
        client: BedrockConverseClient,
        *,
        governor: RequestGovernor,
        cache: ImmutableTeacherCache | None = None,
        request_gate: Callable[[TeacherRequest], None] | None = None,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._governor = governor
        self._cache = cache if cache is not None else ImmutableTeacherCache()
        self._request_gate = request_gate or make_request_gate()
        self._retry_sleep = retry_sleep

    @property
    def provider_name(self) -> str:
        return TeacherProvider.BEDROCK.value

    def supports_recipe(self, recipe: TeacherRecipe | None) -> bool:
        return recipe in {None, TeacherRecipe.SEQUENCE_V1}

    def generate(self, request: TeacherRequest) -> TeacherResult:
        if request.model.provider is not TeacherProvider.BEDROCK:
            raise_teacher_error(
                TeacherErrorCode.INVALID_REQUEST,
                "BedrockConverseTeacher requires provider=bedrock",
                details={"provider": request.model.provider.value},
            )
        if not self.supports_recipe(request.recipe):
            raise_teacher_error(
                TeacherErrorCode.RECIPE_INCOMPATIBLE,
                "Bedrock Converse returns text/tool messages, not logits; logit.v1 is unsupported",
                details={"recipe": request.recipe.value if request.recipe else None},
            )
        if request.settings.do_sample:
            raise_teacher_error(
                TeacherErrorCode.INVALID_REQUEST,
                "Bedrock Converse has no provider-neutral do_sample setting",
            )

        # This must precede cache lookup, accounting, and every network path.
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

        payload = build_converse_request(request)
        prompt = canonical_prompt_text(request)
        response = self._converse_with_retries(
            payload,
            prompt_tokens=estimate_token_count(prompt),
            max_output_tokens=request.settings.max_tokens,
        )
        returned_model_id = _assert_response_identity(request, response)
        response_text, used_tool = _extract_response_content(request, response)
        canonical = require_valid_task_response(
            task=request.task,
            response=response_text,
        )
        usage = _bedrock_usage(response)
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

    def _converse_with_retries(
        self,
        payload: dict[str, Any],
        *,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> Mapping[str, Any]:
        max_retries = self._governor.budget.max_retries
        for retry_index in range(max_retries + 1):
            self._governor.before_network_call(
                estimated_input_tokens=prompt_tokens,
                max_output_tokens=max_output_tokens,
            )
            try:
                return self._client.converse(**payload)
            except Exception as exc:
                mapped = _map_client_error(exc)
                if mapped.code is not TeacherErrorCode.THROTTLED:
                    raise mapped from None
                if retry_index >= max_retries:
                    raise_teacher_error(
                        TeacherErrorCode.RETRIES_EXHAUSTED,
                        "Bedrock throttling retries exhausted",
                        details={
                            "attempts": retry_index + 1,
                            "last_error_code": mapped.code.value,
                        },
                    )
                self._retry_sleep(min(4.0, 0.25 * (2**retry_index)))
        raise AssertionError("bounded retry loop exhausted unexpectedly")


def build_converse_request(request: TeacherRequest) -> dict[str, Any]:
    inference: dict[str, Any] = {
        "maxTokens": request.settings.max_tokens,
        "temperature": request.settings.temperature,
        "topP": request.settings.top_p,
    }
    if request.settings.stop_sequences:
        inference["stopSequences"] = list(request.settings.stop_sequences)

    payload: dict[str, Any] = {
        "modelId": request.model.invocation_id,
        "system": [{"text": request.settings.system_prompt}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": canonical_prompt_text(request)}],
            }
        ],
        "inferenceConfig": inference,
    }
    additional = request.settings.model_dump(mode="json")["additional_model_fields"]
    if request.settings.seed is not None:
        if "seed" in additional and additional["seed"] != request.settings.seed:
            raise_teacher_error(
                TeacherErrorCode.INVALID_REQUEST,
                "conflicting seed values in exact generation settings",
            )
        additional = {**additional, "seed": request.settings.seed}
    if additional:
        payload["additionalModelRequestFields"] = additional
    if request.tools:
        payload["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": dict(tool.input_schema)},
                    }
                }
                for tool in request.tools
            ],
            "toolChoice": {"auto": {}},
        }
    return payload


def _extract_response_content(
    request: TeacherRequest,
    response: Mapping[str, Any],
) -> tuple[str | dict[str, Any], bool]:
    output = response.get("output")
    message = output.get("message") if isinstance(output, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, list) or not content:
        raise_teacher_error(
            TeacherErrorCode.MALFORMED_JSON,
            "Bedrock response is missing output.message.content",
        )

    text_parts: list[str] = []
    tool_blocks: list[Mapping[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        text = block.get("text")
        if isinstance(text, str):
            text_parts.append(text)
        tool_use = block.get("toolUse")
        if isinstance(tool_use, Mapping):
            tool_blocks.append(tool_use)

    if tool_blocks:
        if not request.allow_tool_use:
            raise_teacher_error(
                TeacherErrorCode.TOOL_USE_REJECTED,
                "Bedrock returned tool use when the request forbids it",
            )
        if len(tool_blocks) != 1 or any(part.strip() for part in text_parts):
            raise_teacher_error(
                TeacherErrorCode.TOOL_USE_REJECTED,
                "teacher response must contain exactly one unambiguous tool call",
            )
        tool = tool_blocks[0]
        allowed_names = {spec.name for spec in request.tools}
        name = tool.get("name")
        tool_input = tool.get("input")
        if name not in allowed_names or not isinstance(tool_input, dict):
            raise_teacher_error(
                TeacherErrorCode.TOOL_USE_REJECTED,
                "teacher returned an unknown tool or non-object input",
                details={"tool_name": str(name)},
            )
        return tool_input, True

    text_response = "".join(text_parts).strip()
    if not text_response:
        raise_teacher_error(
            TeacherErrorCode.MALFORMED_JSON,
            "Bedrock response contains neither text nor an allowed tool call",
        )
    return text_response, False


def _bedrock_usage(response: Mapping[str, Any]) -> TokenUsage:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "Bedrock response is missing token usage",
        )
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    total_tokens = usage.get("totalTokens")
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
    ):
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "Bedrock token usage must contain integer input/output counts",
        )
    expected_total = input_tokens + output_tokens
    if total_tokens is not None and total_tokens != expected_total:
        raise_teacher_error(
            TeacherErrorCode.INVALID_REQUEST,
            "Bedrock totalTokens does not match inputTokens + outputTokens",
        )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=expected_total,
    )


def _assert_response_identity(
    request: TeacherRequest,
    response: Mapping[str, Any],
) -> str | None:
    returned = response.get("modelId")
    if returned is not None and returned != request.model.invocation_id:
        raise_teacher_error(
            TeacherErrorCode.MODEL_SUBSTITUTION,
            "Bedrock response identified a model/profile other than requested",
            details={
                "requested": request.model.invocation_id,
                "returned": str(returned),
            },
        )
    return str(returned) if returned is not None else None


def _map_client_error(exc: Exception) -> TeacherError:
    code, message = _client_error_details(exc)
    lowered = message.lower()
    if code in {"AccessDeniedException", "AccessDenied"}:
        if "use case" in lowered or "use-case" in lowered or "agreement" in lowered:
            return _error(
                TeacherErrorCode.USE_CASE_PENDING,
                "Bedrock model use-case approval is pending",
                service_code=code,
            )
        return _error(
            TeacherErrorCode.ACCESS_DENIED,
            "Bedrock denied access to the exact requested model/profile",
            service_code=code,
        )
    if code in {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
    }:
        return _error(
            TeacherErrorCode.THROTTLED,
            "Bedrock request was throttled or temporarily unavailable",
            service_code=code,
        )
    return _error(
        TeacherErrorCode.INVALID_REQUEST,
        "Bedrock Converse request failed",
        service_code=code,
    )


def _client_error_details(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, BedrockError):
        return exc.code, exc.message
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, Mapping) else None
    if isinstance(error, Mapping):
        return str(error.get("Code", type(exc).__name__)), str(error.get("Message", str(exc)))
    return type(exc).__name__, str(exc)


def _error(
    code: TeacherErrorCode,
    message: str,
    *,
    service_code: str,
) -> TeacherError:
    from distillery.teachers.errors import teacher_error

    return teacher_error(
        code,
        message,
        details={"service_error_code": service_code},
    )


__all__ = [
    "BedrockConverseClient",
    "BedrockConverseTeacher",
    "BedrockError",
    "build_converse_request",
]
