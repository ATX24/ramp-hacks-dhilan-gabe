"""Caller-configured teacher priority chains with explicit attempt records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.teachers.errors import (
    TeacherError,
    TeacherErrorCode,
    raise_teacher_error,
)
from distillery.teachers.protocol import TeacherGenerator
from distillery.teachers.records import build_teacher_result
from distillery.teachers.types import (
    AttemptOutcome,
    TeacherAttempt,
    TeacherRequest,
    TeacherResult,
)

_NO_FALLBACK_CODES: frozenset[TeacherErrorCode] = frozenset(
    {
        TeacherErrorCode.USE_CASE_PENDING,
        TeacherErrorCode.OUTPUT_USE_NOT_ALLOWED,
        TeacherErrorCode.PROVIDER_USE_PROHIBITED,
        TeacherErrorCode.AUTHORIZATION_REQUIRED,
        TeacherErrorCode.AUTHORIZATION_INVALID,
        TeacherErrorCode.AUTHORIZATION_EXPIRED,
        TeacherErrorCode.AUTHORIZATION_SCOPE_MISMATCH,
        TeacherErrorCode.SECRET_UNAVAILABLE,
        TeacherErrorCode.MODEL_UNAVAILABLE,
        TeacherErrorCode.LICENSE_GATE_FAILED,
        TeacherErrorCode.RECIPE_INCOMPATIBLE,
        TeacherErrorCode.COST_EXHAUSTED,
        TeacherErrorCode.REQUEST_CAP_EXCEEDED,
        TeacherErrorCode.MODEL_SUBSTITUTION,
    }
)


@dataclass(frozen=True, slots=True)
class TeacherCandidate:
    generator: TeacherGenerator
    request: TeacherRequest


class PriorityTeacherChain:
    """Try candidates in caller order without hiding policy/model failures."""

    def __init__(self, candidates: Sequence[TeacherCandidate]) -> None:
        if not candidates:
            raise ValueError("priority chain requires at least one candidate")
        self._candidates = tuple(candidates)
        expected = _intent_sha256(self._candidates[0].request)
        if any(_intent_sha256(item.request) != expected for item in self._candidates[1:]):
            raise ValueError("priority candidates must preserve the exact request intent")

    def generate(self) -> TeacherResult:
        attempts: list[TeacherAttempt] = []
        for candidate in self._candidates:
            try:
                result = candidate.generator.generate(candidate.request)
            except TeacherError as exc:
                attempts.append(
                    TeacherAttempt(
                        model=candidate.request.model,
                        outcome=AttemptOutcome.REJECTED,
                        rejection_reason=str(exc),
                        error_code=exc.code.value,
                    )
                )
                if candidate.request.model.is_premium_claude:
                    raise_teacher_error(
                        TeacherErrorCode.PREMIUM_FALLBACK_FORBIDDEN,
                        "Claude failure is terminal; cross-provider fallback is forbidden",
                        details={
                            "original_error_code": exc.code.value,
                            "attempts": _attempt_payloads(attempts),
                        },
                    )
                if exc.code in _NO_FALLBACK_CODES:
                    raise_teacher_error(
                        exc.code,
                        str(exc),
                        details={
                            **dict(exc.payload.details),
                            "attempts": _attempt_payloads(attempts),
                            "fallback_blocked": True,
                        },
                    )
                continue

            combined = (*attempts, *result.attempts)
            return build_teacher_result(
                request=candidate.request,
                response_text=result.response_text,
                tokens=result.tokens,
                cost=result.cost,
                tool_use=result.provenance.tool_use,
                cache_hit=result.provenance.cache_hit,
                attempts=combined,
                returned_model_id=result.provenance.returned_model_id,
            )

        raise_teacher_error(
            TeacherErrorCode.CHAIN_EXHAUSTED,
            "all configured teacher candidates were rejected",
            details={"attempts": _attempt_payloads(attempts)},
        )


def _intent_sha256(request: TeacherRequest) -> str:
    return content_sha256(
        {
            "example_id": request.example_id,
            "task": request.task.value,
            "difficulty": request.difficulty.value,
            "split": request.split.value,
            "input": dict(request.input),
            "recipe": request.recipe.value if request.recipe else None,
            "intended_use": request.intended_use.value,
            "target_family": (request.target_family.value if request.target_family else None),
            "student_model_id": request.student_model_id,
            "output_storage": request.output_storage.value,
            "derived_artifact_disposition": (request.derived_artifact_disposition.value),
            "settings": request.settings.model_dump(mode="json"),
            "tools": [item.model_dump(mode="json") for item in request.tools],
            "allow_tool_use": request.allow_tool_use,
        }
    )


def _attempt_payloads(attempts: Sequence[TeacherAttempt]) -> list[dict[str, Any]]:
    return [attempt.model_dump(mode="json") for attempt in attempts]


__all__ = ["PriorityTeacherChain", "TeacherCandidate"]
