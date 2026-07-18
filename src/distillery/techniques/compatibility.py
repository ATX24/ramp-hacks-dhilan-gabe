"""Compatibility negotiation against dataset / model / backend claims."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictStr

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import GitCommitSha, Sha256Hex
from distillery.techniques.capabilities import EvidenceRequirement
from distillery.techniques.descriptor import (
    ExecutionKind,
    TeacherSignal,
    TechniqueDescriptor,
    TokenizerConstraint,
)
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error


class CompatibilityContext(FrozenModel):
    """Caller-supplied environment claims. Incomplete claims fail closed."""

    backend_kind: Literal["local", "sagemaker"]
    student_model_id: StrictStr = Field(min_length=1)
    student_revision: GitCommitSha
    teacher_model_id: StrictStr | None = None
    teacher_revision: GitCommitSha | None = None
    tokenizer_sha256_student: Sha256Hex | None = None
    tokenizer_sha256_teacher: Sha256Hex | None = None
    chat_template_sha256_student: Sha256Hex | None = None
    chat_template_sha256_teacher: Sha256Hex | None = None
    special_token_map_match: StrictBool | None = None
    local_white_box: StrictBool | None = None
    memory_dry_run_ok: StrictBool | None = None
    usable_responses: StrictBool | None = None
    network_isolation: StrictBool | None = None
    instance_type: StrictStr | None = None
    run_id: StrictStr | None = None


class CompatibilityDecision(FrozenModel):
    """Auditable negotiation result (hashed into the protocol)."""

    compatible: StrictBool
    satisfied_evidence: tuple[StrictStr, ...]
    rejected_reasons: tuple[StrictStr, ...] = ()


def negotiate_compatibility(
    descriptor: TechniqueDescriptor,
    context: CompatibilityContext,
) -> CompatibilityDecision:
    """
    Negotiate technique requirements against environment claims.

    Unknown / incomplete required evidence fails closed. No silent fallback
    to a different technique.
    """
    satisfied: list[str] = []
    rejected: list[str] = []

    for requirement in descriptor.evidence_requirements:
        ok = _evidence_satisfied(requirement, descriptor, context)
        if ok is True:
            satisfied.append(requirement)
        elif ok is False:
            rejected.append(requirement)
        else:
            rejected.append(f"{requirement}:incomplete")

    if descriptor.tokenizer_constraint is TokenizerConstraint.EXACT_MATCH:
        if context.tokenizer_sha256_student is None or context.tokenizer_sha256_teacher is None:
            rejected.append("tokenizer_constraint:incomplete")
        elif context.tokenizer_sha256_student != context.tokenizer_sha256_teacher:
            rejected.append("tokenizer_constraint:mismatch")
        else:
            satisfied.append("tokenizer_constraint:exact_match")

    if descriptor.teacher_signal is TeacherSignal.FULL_LOGITS:
        if context.local_white_box is not True:
            rejected.append("teacher_signal:full_logits_requires_local_white_box")

    if context.instance_type is not None:
        if context.instance_type not in descriptor.hardware.approved_instance_types:
            rejected.append("hardware:instance_type_not_approved")

    if descriptor.execution is ExecutionKind.EXTERNAL_CONTAINER:
        if context.network_isolation is not True:
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

    return CompatibilityDecision(
        compatible=True,
        satisfied_evidence=tuple(dict.fromkeys(satisfied)),
        rejected_reasons=(),
    )


def _evidence_satisfied(
    requirement: str,
    descriptor: TechniqueDescriptor,
    context: CompatibilityContext,
) -> bool | None:
    """Return True/False/None (incomplete)."""
    if requirement == EvidenceRequirement.PINNED_STUDENT_REVISION.value:
        return bool(context.student_revision)
    if requirement == EvidenceRequirement.PINNED_TEACHER_REVISION.value:
        if context.teacher_revision is None:
            return None
        return True
    if requirement == EvidenceRequirement.TOKENIZER_FINGERPRINT_MATCH.value:
        if context.tokenizer_sha256_student is None or context.tokenizer_sha256_teacher is None:
            return None
        return context.tokenizer_sha256_student == context.tokenizer_sha256_teacher
    if requirement == EvidenceRequirement.SPECIAL_TOKEN_MAP_MATCH.value:
        return context.special_token_map_match
    if requirement == EvidenceRequirement.CHAT_TEMPLATE_COMPATIBLE.value:
        if (
            context.chat_template_sha256_student is None
            or context.chat_template_sha256_teacher is None
        ):
            return None
        return context.chat_template_sha256_student == context.chat_template_sha256_teacher
    if requirement == EvidenceRequirement.MEMORY_DRY_RUN_OK.value:
        return context.memory_dry_run_ok
    if requirement == EvidenceRequirement.LOCAL_WHITE_BOX.value:
        return context.local_white_box
    if requirement == EvidenceRequirement.USABLE_RESPONSES.value:
        return context.usable_responses
    if requirement == EvidenceRequirement.PLUGIN_IMAGE_DIGEST.value:
        return descriptor.plugin_image is not None
    if requirement == EvidenceRequirement.REVIEWED_SOURCE_BINDING.value:
        return descriptor.reviewed_source is not None
    if requirement == EvidenceRequirement.NETWORK_ISOLATION.value:
        return context.network_isolation
    return None


__all__ = [
    "CompatibilityContext",
    "CompatibilityDecision",
    "negotiate_compatibility",
]
