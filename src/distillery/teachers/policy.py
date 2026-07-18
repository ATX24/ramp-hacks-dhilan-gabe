"""Fail-closed provider, license, output-use, and authorization gates.

Baseline policy permits Claude and Nova only for non-retained evaluation.
Apache-2.0 open-weight Qwen is the default source for Qwen distillation.
Claude-to-Qwen sequence KD requires narrow, current, out-of-band written
authorization evidence. Permission documents and sensitive terms are never
stored here, sent to models, or emitted in errors/provenance.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from distillery.teachers.errors import TeacherErrorCode, raise_teacher_error
from distillery.teachers.types import (
    AttestationSource,
    AuthorizationProvider,
    DerivedArtifactDisposition,
    IntendedUse,
    LicenseStatus,
    OutputRetention,
    OutputStorage,
    RedistributionScope,
    ReviewStatus,
    StudentModelFamily,
    TeacherModelFamily,
    TeacherProvider,
    TeacherRequest,
    WrittenAuthorizationEvidence,
    seal_provider_policy_evidence,
)

PROJECT_PROVIDER_POLICY_VERSION = "distillery-teacher-output-use-2026-07-18.2"
PROJECT_PROVIDER_POLICY_CITATIONS: tuple[str, ...] = (
    "AWS Service Terms §50.5",
    "Anthropic Commercial Terms §D.4",
    "Anthropic Acceptable Use Policy",
    "Amazon Bedrock Model Distillation supported-model compatibility",
    "Apache License 2.0",
)
PROJECT_PROVIDER_POLICY = seal_provider_policy_evidence(
    policy_version=PROJECT_PROVIDER_POLICY_VERSION,
    citations=PROJECT_PROVIDER_POLICY_CITATIONS,
)
AUTHORIZATION_LIMITATION = (
    "operator_attested_out_of_band; permission document not stored or independently verified"
)


def make_request_gate(
    *, today: Callable[[], date] = date.today
) -> Callable[[TeacherRequest], None]:
    def gate(request: TeacherRequest) -> None:
        assert_request_allowed(request, on_date=today())

    return gate


def assert_request_allowed(request: TeacherRequest, *, on_date: date) -> None:
    """Apply every gate before cache lookup, accounting, or client calls."""
    _assert_current_policy_evidence(request)
    _assert_output_use_review(request)
    _assert_provider_use(request, on_date=on_date)
    _assert_license(request)


def authorization_limitation(request: TeacherRequest) -> str | None:
    return AUTHORIZATION_LIMITATION if request.written_authorization else None


def _assert_current_policy_evidence(request: TeacherRequest) -> None:
    evidence = request.provider_policy
    if (
        evidence.policy_version != PROJECT_PROVIDER_POLICY.policy_version
        or evidence.policy_sha256 != PROJECT_PROVIDER_POLICY.policy_sha256
    ):
        raise_teacher_error(
            TeacherErrorCode.OUTPUT_USE_NOT_ALLOWED,
            "request is not bound to the current reviewed provider policy",
            details={
                "expected_policy_version": PROJECT_PROVIDER_POLICY.policy_version,
                "actual_policy_version": evidence.policy_version,
                "expected_policy_sha256": PROJECT_PROVIDER_POLICY.policy_sha256,
                "actual_policy_sha256": evidence.policy_sha256,
            },
        )


def _assert_output_use_review(request: TeacherRequest) -> None:
    policy = request.output_use_policy
    if policy.status is ReviewStatus.PENDING:
        raise_teacher_error(
            TeacherErrorCode.USE_CASE_PENDING,
            "output-use disposition is pending review; generation refused",
            details={
                "record_id": policy.record_id,
                "record_version": policy.record_version,
                "intended_use": request.intended_use.value,
            },
        )
    if policy.status is not ReviewStatus.ALLOWED:
        raise_teacher_error(
            TeacherErrorCode.OUTPUT_USE_NOT_ALLOWED,
            "output-use disposition does not allow this generation",
            details={
                "record_id": policy.record_id,
                "record_version": policy.record_version,
                "status": policy.status.value,
                "intended_use": request.intended_use.value,
            },
        )


def _assert_provider_use(request: TeacherRequest, *, on_date: date) -> None:
    model = request.model
    use = request.intended_use

    if model.family in {TeacherModelFamily.CLAUDE, TeacherModelFamily.NOVA}:
        if not use.derives_weights:
            _require_non_retained_evaluation(request)
            return
        if model.family is TeacherModelFamily.NOVA:
            raise_teacher_error(
                TeacherErrorCode.PROVIDER_USE_PROHIBITED,
                "Nova output cannot teach Qwen; only supported Nova-to-Nova "
                "Bedrock Model Distillation is permitted",
                details={
                    "model_id": model.model_id,
                    "intended_use": use.value,
                    "target_family": _target_family(request),
                },
            )
        _assert_claude_authorization(request, on_date=on_date)
        return

    if use.derives_weights and request.target_family is StudentModelFamily.QWEN:
        if (
            model.family is not TeacherModelFamily.QWEN
            or model.provider is not TeacherProvider.LOCAL
        ):
            raise_teacher_error(
                TeacherErrorCode.PROVIDER_USE_PROHIBITED,
                "only reviewed open-weight Qwen output may teach a Qwen student",
                details={
                    "source_family": model.family.value,
                    "provider": model.provider.value,
                    "target_family": request.target_family.value,
                    "intended_use": use.value,
                },
            )


def _require_non_retained_evaluation(request: TeacherRequest) -> None:
    if (
        request.output_use_policy.retention is not OutputRetention.NON_RETAINED
        or request.output_storage is not OutputStorage.EPHEMERAL
    ):
        raise_teacher_error(
            TeacherErrorCode.OUTPUT_USE_NOT_ALLOWED,
            "Claude/Nova evaluation output must be non-retained and ephemeral",
            details={
                "family": request.model.family.value,
                "intended_use": request.intended_use.value,
                "retention": request.output_use_policy.retention.value,
                "output_storage": request.output_storage.value,
            },
        )


def _assert_claude_authorization(request: TeacherRequest, *, on_date: date) -> None:
    evidence = request.written_authorization
    if evidence is None:
        raise_teacher_error(
            TeacherErrorCode.AUTHORIZATION_REQUIRED,
            "Claude-to-Qwen distillation requires written authorization evidence",
            details={
                "model_id": request.model.model_id,
                "intended_use": request.intended_use.value,
                "policy_version": PROJECT_PROVIDER_POLICY.policy_version,
            },
        )
    if request.intended_use is not IntendedUse.SEQUENCE_KD:
        _scope_mismatch("authorization override is limited to sequence_kd", evidence)
    if request.target_family is not StudentModelFamily.QWEN:
        _scope_mismatch("authorization override requires a Qwen student", evidence)
    if evidence.provider is not AuthorizationProvider.ANTHROPIC:
        _scope_mismatch("authorization provider does not cover Anthropic", evidence)
    if on_date < evidence.effective_date:
        raise_teacher_error(
            TeacherErrorCode.AUTHORIZATION_INVALID,
            "written authorization is not yet effective",
            details={
                "evidence_sha256": evidence.evidence_sha256,
                "effective_date": evidence.effective_date.isoformat(),
                "checked_date": on_date.isoformat(),
            },
        )
    if on_date > evidence.expiration_date:
        raise_teacher_error(
            TeacherErrorCode.AUTHORIZATION_EXPIRED,
            "written authorization has expired",
            details={
                "evidence_sha256": evidence.evidence_sha256,
                "expiration_date": evidence.expiration_date.isoformat(),
                "checked_date": on_date.isoformat(),
            },
        )
    covered_model_ids = (
        evidence.covered_anthropic_model_ids
        if request.model.provider is TeacherProvider.ANTHROPIC
        else evidence.covered_bedrock_model_ids
    )
    if request.model.model_id not in covered_model_ids:
        _scope_mismatch("selected Claude model is not covered", evidence)
    if request.intended_use not in evidence.permitted_intended_uses:
        _scope_mismatch("selected intended use is not covered", evidence)
    if request.student_model_id not in evidence.permitted_student_model_ids:
        _scope_mismatch("selected student model is not covered", evidence)
    if request.target_family not in evidence.permitted_student_families:
        _scope_mismatch("selected student family is not covered", evidence)
    if request.output_storage not in evidence.permitted_storage:
        _scope_mismatch("selected output storage is not covered", evidence)
    if not _redistribution_covers(
        evidence.derived_weight_redistribution_scope,
        request.derived_artifact_disposition,
    ):
        _scope_mismatch("derived artifact disposition is not covered", evidence)
    if evidence.approver_attestation.source is not AttestationSource.PROVIDED_OUT_OF_BAND:
        _scope_mismatch("approver attestation is not out-of-band", evidence)


def _redistribution_covers(
    scope: RedistributionScope,
    disposition: DerivedArtifactDisposition,
) -> bool:
    if disposition is DerivedArtifactDisposition.INTERNAL_ONLY:
        return scope in {
            RedistributionScope.INTERNAL_ONLY,
            RedistributionScope.EXTERNAL_REDISTRIBUTION,
        }
    if disposition is DerivedArtifactDisposition.EXTERNAL_REDISTRIBUTION:
        return scope is RedistributionScope.EXTERNAL_REDISTRIBUTION
    return False


def _scope_mismatch(reason: str, evidence: WrittenAuthorizationEvidence) -> None:
    raise_teacher_error(
        TeacherErrorCode.AUTHORIZATION_SCOPE_MISMATCH,
        reason,
        details={"evidence_sha256": evidence.evidence_sha256},
    )


def _assert_license(request: TeacherRequest) -> None:
    disposition = request.license_disposition
    if disposition.status is not LicenseStatus.APPROVED:
        raise_teacher_error(
            TeacherErrorCode.LICENSE_GATE_FAILED,
            "license/terms disposition is not approved; generation refused",
            details={
                "model_id": disposition.model_id,
                "status": disposition.status.value,
                "evidence_version": disposition.evidence_version,
            },
        )
    if (
        request.intended_use.derives_weights
        and request.target_family is StudentModelFamily.QWEN
        and request.model.family is TeacherModelFamily.QWEN
        and disposition.license_id != "Apache-2.0"
    ):
        raise_teacher_error(
            TeacherErrorCode.LICENSE_GATE_FAILED,
            "open-weight Qwen distillation requires Apache-2.0 evidence",
            details={
                "license_id": disposition.license_id,
                "evidence_version": disposition.evidence_version,
            },
        )


def _target_family(request: TeacherRequest) -> str | None:
    return request.target_family.value if request.target_family else None


__all__ = [
    "AUTHORIZATION_LIMITATION",
    "PROJECT_PROVIDER_POLICY",
    "PROJECT_PROVIDER_POLICY_CITATIONS",
    "PROJECT_PROVIDER_POLICY_VERSION",
    "assert_request_allowed",
    "authorization_limitation",
    "make_request_gate",
]
