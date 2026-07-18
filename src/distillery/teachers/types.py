"""Immutable contracts for local and Bedrock sequence teachers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    GitCommitSha,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.tasks import Difficulty, SplitName, TaskId

FORBIDDEN_TEACHER_LABEL_KEYS: frozenset[str] = frozenset(
    {
        "answer",
        "expected_output",
        "label",
        "oracle",
        "predicted_output",
        "target",
        "target_output",
    }
)
HELD_OUT_SPLITS: frozenset[SplitName] = frozenset(
    {SplitName.TEST, SplitName.IID_TEST, SplitName.OOD_TEST}
)
SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "anthropic_api_key",
        "api_key",
        "authorization",
        "headers",
        "x-api-key",
        "x_api_key",
    }
)


class TeacherProvider(StrEnum):
    LOCAL = "local"
    BEDROCK = "bedrock"
    ANTHROPIC = "anthropic"


class TeacherModelFamily(StrEnum):
    QWEN = "qwen"
    CLAUDE = "claude"
    NOVA = "nova"
    OTHER = "other"


class StudentModelFamily(StrEnum):
    QWEN = "qwen"
    NOVA = "nova"
    OTHER = "other"


class TeacherRecipe(StrEnum):
    SEQUENCE_V1 = "sequence.v1"
    LOGIT_V1 = "logit.v1"


class IntendedUse(StrEnum):
    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"
    TRAINING = "training"
    SYNTHETIC_LABELS = "synthetic_labels"
    SEQUENCE_KD = "sequence_kd"
    LOGIT_KD = "logit_kd"
    FINE_TUNING = "fine_tuning"
    DERIVED_WEIGHTS = "derived_weights"

    @property
    def derives_weights(self) -> bool:
        return self not in {IntendedUse.EVALUATION, IntendedUse.BENCHMARK}


class OutputRetention(StrEnum):
    RETAINED = "retained"
    NON_RETAINED = "non_retained"


class OutputStorage(StrEnum):
    EPHEMERAL = "ephemeral"
    ENCRYPTED_PROJECT_STORAGE = "encrypted_project_storage"
    IMMUTABLE_CACHE = "immutable_cache"


class DerivedArtifactDisposition(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    INTERNAL_ONLY = "internal_only"
    EXTERNAL_REDISTRIBUTION = "external_redistribution"


class RedistributionScope(StrEnum):
    NONE = "none"
    INTERNAL_ONLY = "internal_only"
    EXTERNAL_REDISTRIBUTION = "external_redistribution"


class ReviewStatus(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    PENDING = "pending"


class LicenseStatus(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"
    UNRESOLVED = "unresolved"


class AttestationSource(StrEnum):
    PROVIDED_OUT_OF_BAND = "provided_out_of_band"


class AuthorizationProvider(StrEnum):
    ANTHROPIC = "anthropic"


class AttemptOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"


class GenerationSettings(FrozenModel):
    """Exact provider-neutral generation settings sealed into every request."""

    temperature: StrictFloat = Field(ge=0.0, le=2.0)
    top_p: StrictFloat = Field(gt=0.0, le=1.0)
    max_tokens: StrictInt = Field(ge=1, le=8192)
    stop_sequences: tuple[StrictStr, ...] = ()
    seed: StrictInt | None = None
    do_sample: StrictBool = False
    system_prompt: StrictStr = Field(min_length=1)
    additional_model_fields: FrozenJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_auth_material(self) -> GenerationSettings:
        sensitive = _find_sensitive_keys(self.additional_model_fields)
        if sensitive:
            raise ValueError("additional model fields cannot contain auth/header fields")
        return self

    def settings_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": "distillery.teacher.generation_settings.v1",
                **self.model_dump(mode="json"),
            }
        )


class ToolSpec(FrozenModel):
    name: StrictStr = Field(min_length=1)
    description: StrictStr = Field(min_length=1)
    input_schema: FrozenJsonObject


class TeacherModelRef(FrozenModel):
    """Exact source identity. Profiles never replace the underlying model ID."""

    provider: TeacherProvider
    family: TeacherModelFamily
    model_id: StrictStr = Field(min_length=1)
    inference_profile_id: StrictStr | None = None
    revision: GitCommitSha | None = None

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> TeacherModelRef:
        identity = f"{self.model_id} {self.inference_profile_id or ''}".lower()
        detected = _detect_family(identity)
        if detected is not None and detected is not self.family:
            raise ValueError(f"model identity indicates {detected.value}, not {self.family.value}")
        if self.provider is TeacherProvider.LOCAL:
            if not self.revision:
                raise ValueError("local/open-weight teachers require a pinned revision")
            if self.inference_profile_id is not None:
                raise ValueError("local teachers cannot use a Bedrock inference profile")
        elif self.revision is not None:
            raise ValueError("hosted models use model/profile IDs, not local revisions")
        if self.provider is TeacherProvider.ANTHROPIC and self.inference_profile_id is not None:
            raise ValueError("direct Anthropic models cannot use Bedrock profiles")
        if (
            self.provider is TeacherProvider.ANTHROPIC
            and self.family is not TeacherModelFamily.CLAUDE
        ):
            raise ValueError("direct Anthropic provider requires family=claude")
        return self

    @property
    def invocation_id(self) -> str:
        return self.inference_profile_id or self.model_id

    @property
    def is_premium_claude(self) -> bool:
        return self.family is TeacherModelFamily.CLAUDE

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _detect_family(identity: str) -> TeacherModelFamily | None:
    if "qwen" in identity:
        return TeacherModelFamily.QWEN
    if "anthropic.claude" in identity or "claude-" in identity:
        return TeacherModelFamily.CLAUDE
    if "amazon.nova" in identity or "nova-" in identity:
        return TeacherModelFamily.NOVA
    return None


class ProviderPolicyEvidence(FrozenModel):
    schema_version: Literal["distillery.teacher.provider_policy.v1"] = (
        "distillery.teacher.provider_policy.v1"
    )
    policy_version: StrictStr = Field(min_length=1)
    citations: tuple[StrictStr, ...] = Field(min_length=1)
    policy_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash_bound(self) -> ProviderPolicyEvidence:
        if self.policy_sha256 != content_sha256(self.canonical_body()):
            raise ValueError("provider policy evidence hash mismatch")
        return self

    def canonical_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"policy_sha256"})


def seal_provider_policy_evidence(
    *, policy_version: str, citations: Sequence[str]
) -> ProviderPolicyEvidence:
    provisional = ProviderPolicyEvidence.model_construct(
        policy_version=policy_version,
        citations=tuple(citations),
        policy_sha256="0" * 64,
    )
    return ProviderPolicyEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "policy_sha256": content_sha256(provisional.canonical_body()),
        }
    )


class OutputUsePolicy(FrozenModel):
    schema_version: Literal["distillery.teacher.output_use_policy.v1"] = (
        "distillery.teacher.output_use_policy.v1"
    )
    record_id: StrictStr = Field(min_length=1)
    record_version: StrictStr = Field(min_length=1)
    status: ReviewStatus
    intended_use: IntendedUse
    retention: OutputRetention
    reviewer: StrictStr = Field(min_length=1)
    notes: StrictStr = ""
    policy_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash_bound(self) -> OutputUsePolicy:
        if self.policy_sha256 != content_sha256(self.canonical_body()):
            raise ValueError("output-use policy hash mismatch")
        return self

    def canonical_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"policy_sha256"})


def seal_output_use_policy(
    *,
    record_id: str,
    record_version: str,
    status: ReviewStatus,
    intended_use: IntendedUse,
    retention: OutputRetention,
    reviewer: str,
    notes: str = "",
) -> OutputUsePolicy:
    provisional = OutputUsePolicy.model_construct(
        record_id=record_id,
        record_version=record_version,
        status=status,
        intended_use=intended_use,
        retention=retention,
        reviewer=reviewer,
        notes=notes,
        policy_sha256="0" * 64,
    )
    return OutputUsePolicy.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "policy_sha256": content_sha256(provisional.canonical_body()),
        }
    )


class LicenseDisposition(FrozenModel):
    schema_version: Literal["distillery.teacher.license_disposition.v1"] = (
        "distillery.teacher.license_disposition.v1"
    )
    model_id: StrictStr = Field(min_length=1)
    status: LicenseStatus
    license_id: StrictStr | None = None
    evidence_version: StrictStr = Field(min_length=1)
    notes: StrictStr = ""
    disposition_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash_bound(self) -> LicenseDisposition:
        if self.disposition_sha256 != content_sha256(self.canonical_body()):
            raise ValueError("license disposition hash mismatch")
        return self

    def canonical_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"disposition_sha256"})


def seal_license_disposition(
    *,
    model_id: str,
    status: LicenseStatus,
    evidence_version: str,
    license_id: str | None = None,
    notes: str = "",
) -> LicenseDisposition:
    provisional = LicenseDisposition.model_construct(
        model_id=model_id,
        status=status,
        evidence_version=evidence_version,
        license_id=license_id,
        notes=notes,
        disposition_sha256="0" * 64,
    )
    return LicenseDisposition.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "disposition_sha256": content_sha256(provisional.canonical_body()),
        }
    )


class HumanApproverAttestation(FrozenModel):
    source: AttestationSource
    approver: StrictStr = Field(min_length=1)
    attested_at: AwareDatetime
    statement: Literal["written_authorization_reviewed_and_scope_transcribed"] = (
        "written_authorization_reviewed_and_scope_transcribed"
    )


class WrittenAuthorizationEvidence(FrozenModel):
    """Opaque, hash-bound scope only. Never contains the permission document."""

    schema_version: Literal["distillery.teacher.written_authorization.v1"] = (
        "distillery.teacher.written_authorization.v1"
    )
    provider: AuthorizationProvider
    covered_bedrock_model_ids: tuple[StrictStr, ...] = ()
    covered_anthropic_model_ids: tuple[StrictStr, ...] = ()
    permitted_intended_uses: tuple[IntendedUse, ...] = Field(min_length=1)
    permitted_student_model_ids: tuple[StrictStr, ...] = Field(min_length=1)
    permitted_student_families: tuple[StudentModelFamily, ...] = Field(min_length=1)
    permitted_storage: tuple[OutputStorage, ...] = Field(min_length=1)
    derived_weight_redistribution_scope: RedistributionScope
    effective_date: date
    expiration_date: date
    external_reference: StrictStr | None = None
    document_sha256: Sha256Hex | None = None
    approver_attestation: HumanApproverAttestation
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_and_bind(self) -> WrittenAuthorizationEvidence:
        if self.expiration_date < self.effective_date:
            raise ValueError("authorization expiration precedes effective date")
        if (self.external_reference is None) == (self.document_sha256 is None):
            raise ValueError(
                "authorization requires exactly one opaque reference or document SHA-256"
            )
        if not self.covered_bedrock_model_ids and not self.covered_anthropic_model_ids:
            raise ValueError("authorization must cover at least one exact model ID")
        for values in (
            self.covered_bedrock_model_ids,
            self.covered_anthropic_model_ids,
            self.permitted_student_model_ids,
        ):
            if any(_is_wildcard(value) for value in values):
                raise ValueError("authorization scope cannot contain wildcard IDs")
        if self.evidence_sha256 != content_sha256(self.canonical_body()):
            raise ValueError("written authorization evidence hash mismatch")
        return self

    def canonical_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"evidence_sha256"})


def seal_written_authorization(
    *,
    provider: AuthorizationProvider,
    covered_bedrock_model_ids: Sequence[str] = (),
    covered_anthropic_model_ids: Sequence[str] = (),
    permitted_intended_uses: Sequence[IntendedUse],
    permitted_student_model_ids: Sequence[str],
    permitted_student_families: Sequence[StudentModelFamily],
    permitted_storage: Sequence[OutputStorage],
    derived_weight_redistribution_scope: RedistributionScope,
    effective_date: date,
    expiration_date: date,
    approver_attestation: HumanApproverAttestation,
    external_reference: str | None = None,
    document_sha256: str | None = None,
) -> WrittenAuthorizationEvidence:
    provisional = WrittenAuthorizationEvidence.model_construct(
        provider=provider,
        covered_bedrock_model_ids=tuple(covered_bedrock_model_ids),
        covered_anthropic_model_ids=tuple(covered_anthropic_model_ids),
        permitted_intended_uses=tuple(permitted_intended_uses),
        permitted_student_model_ids=tuple(permitted_student_model_ids),
        permitted_student_families=tuple(permitted_student_families),
        permitted_storage=tuple(permitted_storage),
        derived_weight_redistribution_scope=derived_weight_redistribution_scope,
        effective_date=effective_date,
        expiration_date=expiration_date,
        external_reference=external_reference,
        document_sha256=document_sha256,
        approver_attestation=approver_attestation,
        evidence_sha256="0" * 64,
    )
    return WrittenAuthorizationEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": content_sha256(provisional.canonical_body()),
        }
    )


def _is_wildcard(value: str) -> bool:
    return value.strip().lower() in {"*", "all", "any"} or "*" in value


class TeacherBudget(FrozenModel):
    max_requests: StrictInt = Field(ge=1, le=1_000_000)
    max_retries: StrictInt = Field(ge=0, le=16)
    min_request_interval_seconds: StrictFloat = Field(ge=0.0, le=3600.0)
    cost_ceiling_usd: StrictFloat = Field(ge=0.0)
    input_usd_per_1k_tokens: StrictFloat = Field(ge=0.0)
    output_usd_per_1k_tokens: StrictFloat = Field(ge=0.0)
    pricing_version: StrictStr = Field(min_length=1)


class TeacherRequest(FrozenModel):
    """One label-free teacher request bound to policy and source identity."""

    example_id: StrictStr = Field(min_length=1)
    task: TaskId
    difficulty: Difficulty
    split: SplitName
    input: FrozenJsonObject
    recipe: TeacherRecipe | None
    intended_use: IntendedUse
    target_family: StudentModelFamily | None
    student_model_id: StrictStr | None
    output_storage: OutputStorage
    derived_artifact_disposition: DerivedArtifactDisposition
    model: TeacherModelRef
    settings: GenerationSettings
    output_use_policy: OutputUsePolicy
    license_disposition: LicenseDisposition
    provider_policy: ProviderPolicyEvidence
    written_authorization: WrittenAuthorizationEvidence | None = None
    tools: tuple[ToolSpec, ...] = ()
    allow_tool_use: StrictBool = False

    @model_validator(mode="after")
    def _safety_invariants(self) -> TeacherRequest:
        if self.split in HELD_OUT_SPLITS:
            raise ValueError(f"teacher requests forbid held-out split {self.split.value}")
        _assert_no_forbidden_keys(self.input)
        if _find_sensitive_keys(self.input):
            raise ValueError("teacher input cannot contain auth/header fields")
        if any(_find_sensitive_keys(tool.input_schema) for tool in self.tools):
            raise ValueError("teacher tool schemas cannot contain auth/header fields")
        if self.tools and not self.allow_tool_use:
            raise ValueError("tools require allow_tool_use=True")
        if self.license_disposition.model_id != self.model.model_id:
            raise ValueError("license disposition model_id must match source model_id")
        if self.output_use_policy.intended_use is not self.intended_use:
            raise ValueError("output-use policy intended_use must match request")
        if self.intended_use in {IntendedUse.EVALUATION, IntendedUse.BENCHMARK}:
            if self.recipe is not None:
                raise ValueError("evaluation/benchmark requests cannot select a KD recipe")
            if self.target_family is not None or self.student_model_id is not None:
                raise ValueError("evaluation/benchmark requests cannot name a student")
            if self.derived_artifact_disposition is not (DerivedArtifactDisposition.NOT_APPLICABLE):
                raise ValueError("evaluation/benchmark has no derived artifact")
        else:
            if self.target_family is None or not self.student_model_id:
                raise ValueError("weight-deriving uses require exact student identity")
            if self.derived_artifact_disposition is (DerivedArtifactDisposition.NOT_APPLICABLE):
                raise ValueError("weight-deriving uses require artifact disposition")
        if self.intended_use is IntendedUse.SEQUENCE_KD:
            if self.recipe is not TeacherRecipe.SEQUENCE_V1:
                raise ValueError("sequence_kd requires recipe=sequence.v1")
        if self.intended_use is IntendedUse.LOGIT_KD:
            if self.recipe is not TeacherRecipe.LOGIT_V1:
                raise ValueError("logit_kd requires recipe=logit.v1")
        return self

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "difficulty": self.difficulty.value,
            "input": dict(self.input),
        }


class TokenUsage(FrozenModel):
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _total_matches(self) -> TokenUsage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class CostRecord(FrozenModel):
    input_usd: StrictFloat = Field(ge=0.0)
    output_usd: StrictFloat = Field(ge=0.0)
    total_usd: StrictFloat = Field(ge=0.0)
    pricing_version: StrictStr = Field(min_length=1)
    currency: Literal["USD"] = "USD"

    @model_validator(mode="after")
    def _total_matches(self) -> CostRecord:
        expected = round(self.input_usd + self.output_usd, 10)
        if abs(self.total_usd - expected) > 1e-9:
            raise ValueError("total_usd must equal input_usd + output_usd")
        return self


class TeacherAttempt(FrozenModel):
    model: TeacherModelRef
    outcome: AttemptOutcome
    rejection_reason: StrictStr | None = None
    error_code: StrictStr | None = None

    @model_validator(mode="after")
    def _consistent(self) -> TeacherAttempt:
        rejected = self.outcome is AttemptOutcome.REJECTED
        if rejected != bool(self.rejection_reason and self.error_code):
            raise ValueError("rejected attempts require reason/code; success forbids them")
        return self


class TeacherProvenance(FrozenModel):
    schema_version: Literal["distillery.teacher.provenance.v1"] = "distillery.teacher.provenance.v1"
    provider: TeacherProvider
    family: TeacherModelFamily
    model_id: StrictStr
    returned_model_id: StrictStr | None
    inference_profile_id: StrictStr | None
    revision: StrictStr | None
    recipe: TeacherRecipe | None
    intended_use: IntendedUse
    output_retention: OutputRetention
    output_storage: OutputStorage
    derived_artifact_disposition: DerivedArtifactDisposition
    prompt_sha256: Sha256Hex
    request_sha256: Sha256Hex
    settings_sha256: Sha256Hex
    output_use_policy_version: StrictStr
    output_use_policy_sha256: Sha256Hex
    provider_policy_version: StrictStr
    provider_policy_sha256: Sha256Hex
    license_evidence_version: StrictStr
    license_disposition_sha256: Sha256Hex
    written_authorization_sha256: Sha256Hex | None
    authorization_limitation: StrictStr | None
    cache_hit: StrictBool
    tool_use: StrictBool
    response_sha256: Sha256Hex
    provenance_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash_bound(self) -> TeacherProvenance:
        if self.provenance_sha256 != content_sha256(self.canonical_body()):
            raise ValueError("teacher provenance hash mismatch")
        return self

    def canonical_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"provenance_sha256"})


def seal_provenance(**fields: Any) -> TeacherProvenance:
    provisional = TeacherProvenance.model_construct(**fields, provenance_sha256="0" * 64)
    return TeacherProvenance.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "provenance_sha256": content_sha256(provisional.canonical_body()),
        }
    )


class TeacherResult(FrozenModel):
    schema_version: Literal["distillery.teacher.result.v1"] = "distillery.teacher.result.v1"
    example_id: StrictStr
    task: TaskId
    response_text: StrictStr = Field(min_length=1)
    tokens: TokenUsage
    cost: CostRecord
    provenance: TeacherProvenance
    attempts: tuple[TeacherAttempt, ...] = Field(min_length=1)
    result_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash_bound(self) -> TeacherResult:
        if self.result_sha256 != content_sha256(self.canonical_body()):
            raise ValueError("teacher result hash mismatch")
        return self

    def canonical_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"result_sha256"})


def seal_teacher_result(**fields: Any) -> TeacherResult:
    provisional = TeacherResult.model_construct(**fields, result_sha256="0" * 64)
    return TeacherResult.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "result_sha256": content_sha256(provisional.canonical_body()),
        }
    )


def _assert_no_forbidden_keys(payload: Mapping[str, Any]) -> None:
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key in FORBIDDEN_TEACHER_LABEL_KEYS:
                    raise ValueError(f"forbidden teacher label field {key!r}")
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def _find_sensitive_keys(payload: Mapping[str, Any]) -> tuple[str, ...]:
    found: set[str] = set()
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key.lower() in SENSITIVE_FIELD_NAMES:
                    found.add(key.lower())
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return tuple(sorted(found))


__all__ = [
    "FORBIDDEN_TEACHER_LABEL_KEYS",
    "HELD_OUT_SPLITS",
    "SENSITIVE_FIELD_NAMES",
    "AttestationSource",
    "AttemptOutcome",
    "AuthorizationProvider",
    "CostRecord",
    "DerivedArtifactDisposition",
    "GenerationSettings",
    "HumanApproverAttestation",
    "IntendedUse",
    "LicenseDisposition",
    "LicenseStatus",
    "OutputRetention",
    "OutputStorage",
    "OutputUsePolicy",
    "ProviderPolicyEvidence",
    "RedistributionScope",
    "ReviewStatus",
    "StudentModelFamily",
    "TeacherAttempt",
    "TeacherBudget",
    "TeacherModelFamily",
    "TeacherModelRef",
    "TeacherProvider",
    "TeacherProvenance",
    "TeacherRecipe",
    "TeacherRequest",
    "TeacherResult",
    "TokenUsage",
    "ToolSpec",
    "WrittenAuthorizationEvidence",
    "seal_license_disposition",
    "seal_output_use_policy",
    "seal_provider_policy_evidence",
    "seal_provenance",
    "seal_teacher_result",
    "seal_written_authorization",
]
