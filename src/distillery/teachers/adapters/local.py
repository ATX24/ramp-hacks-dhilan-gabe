"""Injectable local/open-weight teacher adapter for the existing Qwen path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from distillery.teachers.budget import RequestGovernor, estimate_token_count
from distillery.teachers.cache import ImmutableTeacherCache
from distillery.teachers.errors import TeacherErrorCode, raise_teacher_error
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


class LocalOpenWeightClient(Protocol):
    def generate(
        self,
        *,
        model_id: str,
        revision: str,
        prompt: str,
        settings: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return exact model/revision, response_text, and optional usage."""


class LocalOpenWeightTeacher:
    """Policy-gated adapter; it never loads model weights itself."""

    def __init__(
        self,
        client: LocalOpenWeightClient,
        *,
        governor: RequestGovernor,
        cache: ImmutableTeacherCache | None = None,
        support_logit: bool = True,
        request_gate: Callable[[TeacherRequest], None] | None = None,
    ) -> None:
        self._client = client
        self._governor = governor
        self._cache = cache if cache is not None else ImmutableTeacherCache()
        self._support_logit = support_logit
        self._request_gate = request_gate or make_request_gate()

    @property
    def provider_name(self) -> str:
        return TeacherProvider.LOCAL.value

    def supports_recipe(self, recipe: TeacherRecipe | None) -> bool:
        if recipe is None or recipe is TeacherRecipe.SEQUENCE_V1:
            return True
        return recipe is TeacherRecipe.LOGIT_V1 and self._support_logit

    def generate(self, request: TeacherRequest) -> TeacherResult:
        if request.model.provider is not TeacherProvider.LOCAL:
            raise_teacher_error(
                TeacherErrorCode.INVALID_REQUEST,
                "LocalOpenWeightTeacher requires provider=local",
                details={"provider": request.model.provider.value},
            )
        if not self.supports_recipe(request.recipe):
            raise_teacher_error(
                TeacherErrorCode.RECIPE_INCOMPATIBLE,
                f"local teacher does not support {request.recipe}",
            )
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

        prompt = canonical_prompt_text(request)
        self._governor.before_network_call(
            estimated_input_tokens=estimate_token_count(prompt),
            max_output_tokens=request.settings.max_tokens,
        )
        raw = self._client.generate(
            model_id=request.model.model_id,
            revision=request.model.revision or "",
            prompt=prompt,
            settings=request.settings.model_dump(mode="json"),
        )
        _assert_local_identity(request, raw)
        response_text = raw.get("response_text")
        if not isinstance(response_text, str) or not response_text:
            raise_teacher_error(
                TeacherErrorCode.MALFORMED_JSON,
                "local teacher returned no response_text",
            )
        canonical = require_valid_task_response(
            task=request.task,
            response=response_text,
        )
        usage = _local_usage(raw, prompt=prompt, response_text=canonical)
        cost = self._governor.record_usage(usage)
        result = build_teacher_result(
            request=request,
            response_text=canonical,
            tokens=usage,
            cost=cost,
            tool_use=False,
            cache_hit=False,
            attempts=(
                TeacherAttempt(
                    model=request.model,
                    outcome=AttemptOutcome.SUCCEEDED,
                ),
            ),
            returned_model_id=request.model.model_id,
        )
        return self._cache.put(key, result) if use_cache else result


def _assert_local_identity(request: TeacherRequest, response: Mapping[str, Any]) -> None:
    returned_model = response.get("model_id")
    returned_revision = response.get("revision")
    if returned_model != request.model.model_id or returned_revision != request.model.revision:
        raise_teacher_error(
            TeacherErrorCode.MODEL_SUBSTITUTION,
            "local client returned a different source identity",
            details={
                "requested_model_id": request.model.model_id,
                "returned_model_id": str(returned_model),
                "requested_revision": request.model.revision,
                "returned_revision": str(returned_revision),
            },
        )


def _local_usage(raw: Mapping[str, Any], *, prompt: str, response_text: str) -> TokenUsage:
    input_tokens = raw.get("input_tokens")
    output_tokens = raw.get("output_tokens")
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    estimated_input = estimate_token_count(prompt)
    estimated_output = estimate_token_count(response_text)
    return TokenUsage(
        input_tokens=estimated_input,
        output_tokens=estimated_output,
        total_tokens=estimated_input + estimated_output,
    )


__all__ = ["LocalOpenWeightClient", "LocalOpenWeightTeacher"]
