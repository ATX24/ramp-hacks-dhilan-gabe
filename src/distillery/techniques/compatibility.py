"""Fail-closed compatibility negotiation over complete environment claims."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictStr

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import GitCommitSha, Sha256Hex, content_sha256
from distillery.techniques.capabilities import EvidenceRequirement
from distillery.techniques.descriptor import (
    ExecutionKind,
    TeacherSignal,
    TechniqueDescriptor,
    TokenizerConstraint,
)
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error


class CompatibilityContext(FrozenModel):
    """Typed dataset/model/backend claims hashed into every technique plan."""

    backend_kind: Literal["local", "sagemaker"]
    student_model_id: StrictStr = Field(min_length=1)
    student_revision: GitCommitSha
    tokenizer_sha256_student: Sha256Hex
    chat_template_sha256_student: Sha256Hex
    teacher_model_id: StrictStr | None = None
    teacher_revision: GitCommitSha | None = None
    tokenizer_sha256_teacher: Sha256Hex | None = None
    chat_template_sha256_teacher: Sha256Hex | None = None
    special_token_map_match: StrictBool | None = None
    full_logits_available: StrictBool | None = None
    local_white_box: StrictBool | None = None
    memory_dry_run_ok: StrictBool | None = None
    usable_responses: StrictBool | None = None
    network_isolation: StrictBool
    instance_type: StrictStr = Field(min_length=1)
    run_id: StrictStr | None = None


class CompatibilityDecision(FrozenModel):
    """Auditable decision containing both evidence and exact claim values."""

    compatible: StrictBool
    satisfied_evidence: tuple[StrictStr, ...]
    rejected_reasons: tuple[StrictStr, ...] = ()
    environment_sha256: Sha256Hex
    negotiated_claims: FrozenJsonObject


def negotiate_compatibility(
    descriptor: TechniqueDescriptor,
    context: CompatibilityContext,
) -> CompatibilityDecision:
    satisfied: list[str] = []
    rejected: list[str] = []
    for requirement in descriptor.evidence_requirements:
        result = _evidence_satisfied(requirement, context)
        if result is True:
            satisfied.append(requirement)
        elif result is False:
            rejected.append(requirement)
        else:
            rejected.append(f"{requirement}:incomplete")

    if descriptor.tokenizer_constraint is TokenizerConstraint.EXACT_MATCH:
        if context.tokenizer_sha256_teacher is None:
            rejected.append("tokenizer_constraint:incomplete")
        elif context.tokenizer_sha256_student != context.tokenizer_sha256_teacher:
            rejected.append("tokenizer_constraint:mismatch")
        else:
            satisfied.append("tokenizer_constraint:exact_match")

    if descriptor.teacher_signal is TeacherSignal.FULL_LOGITS:
        if (
            context.teacher_model_id is None
            or context.teacher_revision is None
            or context.tokenizer_sha256_teacher is None
            or context.chat_template_sha256_teacher is None
        ):
            rejected.append("teacher_signal:full_logits_teacher_identity_incomplete")
        if context.full_logits_available is not True:
            rejected.append("teacher_signal:full_logits_unavailable")
        if context.local_white_box is not True:
            rejected.append("teacher_signal:full_logits_requires_local_white_box")

    if context.instance_type not in descriptor.hardware.approved_instance_types:
        rejected.append("hardware:instance_type_not_approved")
    if descriptor.hardware.requires_network_isolation and context.network_isolation is not True:
        rejected.append("hardware:network_isolation_required")
    if (
        descriptor.execution is ExecutionKind.EXTERNAL_CONTAINER
        and context.network_isolation is not True
    ):
        rejected.append("external:network_isolation_required")

    if rejected:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
            "technique is incompatible with dataset/model/backend claims",
            details={
                "technique_id": descriptor.technique_id,
                "version": descriptor.version,
                "rejected_reasons": rejected,
                "satisfied_evidence": satisfied,
            },
            run_id=context.run_id,
        )

    claims = context.model_dump(mode="json")
    return CompatibilityDecision(
        compatible=True,
        satisfied_evidence=tuple(dict.fromkeys(satisfied)),
        rejected_reasons=(),
        environment_sha256=content_sha256(claims),
        negotiated_claims=claims,
    )


def _evidence_satisfied(
    requirement: str,
    context: CompatibilityContext,
) -> bool | None:
    if requirement == EvidenceRequirement.PINNED_STUDENT_REVISION.value:
        return bool(context.student_revision)
    if requirement == EvidenceRequirement.PINNED_TEACHER_REVISION.value:
        return None if context.teacher_revision is None else True
    if requirement == EvidenceRequirement.TOKENIZER_FINGERPRINT_MATCH.value:
        if context.tokenizer_sha256_teacher is None:
            return None
        return context.tokenizer_sha256_student == context.tokenizer_sha256_teacher
    if requirement == EvidenceRequirement.SPECIAL_TOKEN_MAP_MATCH.value:
        return context.special_token_map_match
    if requirement == EvidenceRequirement.CHAT_TEMPLATE_COMPATIBLE.value:
        if context.chat_template_sha256_teacher is None:
            return None
        return context.chat_template_sha256_student == context.chat_template_sha256_teacher
    if requirement == EvidenceRequirement.MEMORY_DRY_RUN_OK.value:
        return context.memory_dry_run_ok
    if requirement == EvidenceRequirement.FULL_LOGITS_AVAILABLE.value:
        return context.full_logits_available
    if requirement == EvidenceRequirement.LOCAL_WHITE_BOX.value:
        return context.local_white_box
    if requirement == EvidenceRequirement.USABLE_RESPONSES.value:
        return context.usable_responses
    if requirement == EvidenceRequirement.NETWORK_ISOLATION.value:
        return context.network_isolation
    return None


__all__ = [
    "CompatibilityContext",
    "CompatibilityDecision",
    "negotiate_compatibility",
]
