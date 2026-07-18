"""Canonical label-free prompt and full request hashing."""

from __future__ import annotations

from typing import Any

from distillery.contracts.hashing import canonical_json_bytes, content_sha256
from distillery.teachers.types import TeacherRequest


def canonical_prompt_text(request: TeacherRequest) -> str:
    return canonical_json_bytes(request.prompt_payload()).decode("utf-8")


def prompt_sha256(request: TeacherRequest) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.teacher.prompt.v1",
            "prompt": request.prompt_payload(),
        }
    )


def request_payload(request: TeacherRequest) -> dict[str, Any]:
    authorization_sha256 = (
        request.written_authorization.evidence_sha256 if request.written_authorization else None
    )
    return {
        "schema_version": "distillery.teacher.request.v1",
        "example_id": request.example_id,
        "split": request.split.value,
        "recipe": request.recipe.value if request.recipe else None,
        "intended_use": request.intended_use.value,
        "target_family": request.target_family.value if request.target_family else None,
        "student_model_id": request.student_model_id,
        "output_storage": request.output_storage.value,
        "derived_artifact_disposition": (request.derived_artifact_disposition.value),
        "model": request.model.identity_payload(),
        "prompt_sha256": prompt_sha256(request),
        "settings": request.settings.model_dump(mode="json"),
        "tools": [tool.model_dump(mode="json") for tool in request.tools],
        "allow_tool_use": request.allow_tool_use,
        "output_use_policy_sha256": request.output_use_policy.policy_sha256,
        "provider_policy_sha256": request.provider_policy.policy_sha256,
        "license_disposition_sha256": (request.license_disposition.disposition_sha256),
        "written_authorization_sha256": authorization_sha256,
    }


def request_sha256(request: TeacherRequest) -> str:
    return content_sha256(request_payload(request))


def cache_key_sha256(request: TeacherRequest) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.teacher.cache_key.v1",
            "request_sha256": request_sha256(request),
        }
    )


__all__ = [
    "cache_key_sha256",
    "canonical_prompt_text",
    "prompt_sha256",
    "request_payload",
    "request_sha256",
]
