"""Helpers for sealing response, provenance, token, and cost records."""

from __future__ import annotations

from collections.abc import Sequence

from distillery.contracts.hashing import content_sha256
from distillery.teachers.hashing import prompt_sha256, request_sha256
from distillery.teachers.policy import authorization_limitation
from distillery.teachers.types import (
    CostRecord,
    TeacherAttempt,
    TeacherRequest,
    TeacherResult,
    TokenUsage,
    seal_provenance,
    seal_teacher_result,
)


def build_teacher_result(
    *,
    request: TeacherRequest,
    response_text: str,
    tokens: TokenUsage,
    cost: CostRecord,
    tool_use: bool,
    cache_hit: bool,
    attempts: Sequence[TeacherAttempt],
    returned_model_id: str | None,
) -> TeacherResult:
    authorization_sha256 = (
        request.written_authorization.evidence_sha256 if request.written_authorization else None
    )
    provenance = seal_provenance(
        provider=request.model.provider,
        family=request.model.family,
        model_id=request.model.model_id,
        returned_model_id=returned_model_id,
        inference_profile_id=request.model.inference_profile_id,
        revision=request.model.revision,
        recipe=request.recipe,
        intended_use=request.intended_use,
        output_retention=request.output_use_policy.retention,
        output_storage=request.output_storage,
        derived_artifact_disposition=request.derived_artifact_disposition,
        prompt_sha256=prompt_sha256(request),
        request_sha256=request_sha256(request),
        settings_sha256=request.settings.settings_sha256(),
        output_use_policy_version=request.output_use_policy.record_version,
        output_use_policy_sha256=request.output_use_policy.policy_sha256,
        provider_policy_version=request.provider_policy.policy_version,
        provider_policy_sha256=request.provider_policy.policy_sha256,
        license_evidence_version=request.license_disposition.evidence_version,
        license_disposition_sha256=(request.license_disposition.disposition_sha256),
        written_authorization_sha256=authorization_sha256,
        authorization_limitation=authorization_limitation(request),
        cache_hit=cache_hit,
        tool_use=tool_use,
        response_sha256=content_sha256(response_text),
    )
    return seal_teacher_result(
        example_id=request.example_id,
        task=request.task,
        response_text=response_text,
        tokens=tokens,
        cost=cost,
        provenance=provenance,
        attempts=tuple(attempts),
    )


def cached_result(request: TeacherRequest, stored: TeacherResult) -> TeacherResult:
    zero_cost = CostRecord(
        input_usd=0.0,
        output_usd=0.0,
        total_usd=0.0,
        pricing_version=stored.cost.pricing_version,
    )
    return build_teacher_result(
        request=request,
        response_text=stored.response_text,
        tokens=stored.tokens,
        cost=zero_cost,
        tool_use=stored.provenance.tool_use,
        cache_hit=True,
        attempts=stored.attempts,
        returned_model_id=stored.provenance.returned_model_id,
    )


__all__ = ["build_teacher_result", "cached_result"]
