"""Live-verifier readiness engine. No caller-supplied readiness booleans."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.qwen72b_fallback.cost import CostAuthorizationEvidence
from experiments.qwen72b_fallback.evidence import (
    PREFIXED_SHA256_PATTERN,
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
)
from experiments.qwen72b_fallback.finance_world_targets import (
    FinanceWorldCorpusEvidence,
)
from experiments.qwen72b_fallback.license_policy import LicenseComplianceEvidence
from experiments.qwen72b_fallback.memory import Qwen72BMemoryProbeEvidence
from experiments.qwen72b_fallback.pins import Qwen72BIdentityEvidence
from experiments.qwen72b_fallback.profile import Qwen72BTrainingProfile
from experiments.qwen72b_fallback.tokenizer_compat import (
    TokenizerCompatibilityEvidence,
)


class ExecutionAction(StrEnum):
    MATERIALIZE = "materialize"
    MEMORY_PROBE = "memory_probe"
    REHEARSAL = "rehearsal"
    FULL = "full"
    TEACHER_TRAJECTORIES = "teacher_trajectories"


class GateCode(StrEnum):
    LOCAL_IDENTITY = "local_identity"
    LICENSE_ARTIFACTS = "license_artifacts"
    EXECUTION_REVIEWS = "execution_reviews"
    IAM_SCOPE = "iam_scope"
    ACTIVE_CONFLICTS = "active_conflicts"
    COST_EXPOSURE = "cost_exposure"
    S3_BODY_HASHES = "s3_body_hashes"
    TOKENIZER_PAIRS = "tokenizer_pairs"
    ECR_EXACT_IMAGE = "ecr_exact_image"
    MEMORY_PROBE = "memory_probe"
    FINANCE_WORLD_DATA = "finance_world_data"
    TYPED_CONFIRMATION = "typed_confirmation"
    TEACHER_TRAJECTORIES = "teacher_trajectories"


class VerificationFailure(RuntimeError):
    def __init__(self, gate: GateCode, detail: str) -> None:
        super().__init__(detail)
        self.gate = gate
        self.detail = detail


class LocalPolicyEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.local_policy.v1"] = (
        "distillery.qwen72b_fallback.local_policy.v1"
    )
    source: Literal[VerificationSource.LOCAL_BYTES] = VerificationSource.LOCAL_BYTES
    identity: Qwen72BIdentityEvidence
    license: LicenseComplianceEvidence
    execution_bindings_bytes_sha256: str = Field(pattern=SHA256_PATTERN)


class ReviewClearanceEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.review_clearance.v1"] = (
        "distillery.qwen72b_fallback.review_clearance.v1"
    )
    source: Literal[VerificationSource.LOCAL_BYTES] = VerificationSource.LOCAL_BYTES
    review_packet_sha256: tuple[str, str]
    execution_bindings_bytes_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _distinct_packets(self) -> ReviewClearanceEvidence:
        if len(set(self.review_packet_sha256)) != 2:
            raise ValueError("both independent review packet hashes must be distinct")
        return self


class IamScopeEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.iam_scope.v1"] = (
        "distillery.qwen72b_fallback.iam_scope.v1"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    action: ExecutionAction
    caller_account_id: Literal["225989358036"]
    caller_arn: str
    role_arn: str
    role_id: str
    instance_profile_arn: str | None = None
    transfer_ami_id: str | None = Field(default=None, pattern=r"^ami-[0-9a-f]{17}$")
    transfer_subnet_id: str | None = Field(
        default=None,
        pattern=r"^subnet-[0-9a-f]{17}$",
    )
    transfer_security_group_id: str | None = Field(
        default=None,
        pattern=r"^sg-[0-9a-f]{17}$",
    )
    policy_document_sha256: tuple[str, ...] = Field(min_length=1)
    allowed_s3_resources: tuple[str, ...] = Field(min_length=1)
    checked_region: Literal["us-east-1"] = "us-east-1"

    @model_validator(mode="after")
    def _materializer_resources(self) -> IamScopeEvidence:
        transfer_fields = (
            self.instance_profile_arn,
            self.transfer_ami_id,
            self.transfer_subnet_id,
            self.transfer_security_group_id,
        )
        if self.action is ExecutionAction.MATERIALIZE:
            if any(value is None for value in transfer_fields):
                raise ValueError("materialization IAM evidence lacks exact EC2 resources")
        elif any(value is not None for value in transfer_fields):
            raise ValueError("training IAM evidence must not carry transfer resources")
        return self


class ConflictEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.conflicts.v1"] = (
        "distillery.qwen72b_fallback.conflicts.v1"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    action: ExecutionAction
    requested_launch_name: str = Field(min_length=1)
    active_p4de_jobs: tuple[str, ...]
    active_g5_jobs: tuple[str, ...]
    active_14b_or_32b_jobs: tuple[str, ...]
    active_transfer_instance_ids: tuple[str, ...]
    duplicate_launches: tuple[str, ...]
    orphan_resource_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _no_conflicts(self) -> ConflictEvidence:
        groups = (
            self.active_p4de_jobs,
            self.active_g5_jobs,
            self.active_14b_or_32b_jobs,
            self.active_transfer_instance_ids,
            self.duplicate_launches,
            self.orphan_resource_ids,
        )
        if any(groups):
            raise ValueError("active, duplicate, or orphaned AWS resources block execution")
        return self


class S3SnapshotEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.s3_snapshot.v2"] = (
        "distillery.qwen72b_fallback.s3_snapshot.v2"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    bucket: Literal["distillery-225989358036-us-east-1"]
    prefix: Literal["models/Qwen/Qwen2.5-72B-Instruct/495f39366efef23836d0cfae4fbe635880d2be31"]
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    object_body_sha256: dict[str, str]
    object_sizes: dict[str, int]
    snapshot_manifest_body_sha256: str = Field(pattern=SHA256_PATTERN)
    sha256sums_body_sha256: str = Field(pattern=SHA256_PATTERN)
    materialization_manifest_body_sha256: str = Field(pattern=SHA256_PATTERN)


class EcrImageEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.ecr_image.v2"] = (
        "distillery.qwen72b_fallback.ecr_image.v2"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    account_id: Literal["225989358036"]
    region: Literal["us-east-1"]
    repository: Literal["distillery-training"]
    image_digest: str = Field(pattern=PREFIXED_SHA256_PATTERN)
    image_uri: str
    image_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    source_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    qwen72b_trainer_packaged: Literal[True]
    attention_backend: Literal["sdpa_math"]
    flash_attention_2_packaged: Literal[False]


class ExecutionConfirmation(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.confirmation.v1"] = (
        "distillery.qwen72b_fallback.confirmation.v1"
    )
    action: ExecutionAction
    launch_name: str = Field(min_length=1)
    typed_text: str = Field(min_length=1)
    operator_account_id: Literal["225989358036"]

    @model_validator(mode="after")
    def _exact_text(self) -> ExecutionConfirmation:
        expected = required_confirmation(self.action, self.launch_name)
        if self.typed_text != expected:
            raise ValueError(f"typed confirmation must equal: {expected}")
        return self


class VerifiedEvidenceBundle(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.evidence_bundle.v2"] = (
        "distillery.qwen72b_fallback.evidence_bundle.v2"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    action: ExecutionAction
    launch_name: str
    target_profile_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    local_policy: LocalPolicyEvidence
    reviews: ReviewClearanceEvidence
    iam: IamScopeEvidence
    conflicts: ConflictEvidence
    cost: CostAuthorizationEvidence
    confirmation: ExecutionConfirmation
    s3_snapshot: S3SnapshotEvidence | None
    tokenizer_compatibility: TokenizerCompatibilityEvidence | None
    ecr_image: EcrImageEvidence | None
    memory_probe: Qwen72BMemoryProbeEvidence | None
    finance_world_data: FinanceWorldCorpusEvidence | None

    @model_validator(mode="after")
    def _bundle_invariants(self) -> VerifiedEvidenceBundle:
        action_bound = (
            self.iam.action,
            self.conflicts.action,
            self.confirmation.action,
        )
        if any(bound is not self.action for bound in action_bound):
            raise ValueError("evidence component action differs from bundle action")
        if self.cost.action.value != self.action.value:
            raise ValueError("cost evidence action differs from bundle action")
        launch_bound = (
            self.conflicts.requested_launch_name,
            self.confirmation.launch_name,
        )
        if any(bound != self.launch_name for bound in launch_bound):
            raise ValueError("evidence component launch name differs from bundle")
        if self.action is ExecutionAction.MATERIALIZE:
            if self.target_profile_sha256 is not None:
                raise ValueError("materialization must not bind a training profile")
            optional = (
                self.s3_snapshot,
                self.tokenizer_compatibility,
                self.ecr_image,
                self.memory_probe,
                self.finance_world_data,
            )
            if any(item is not None for item in optional):
                raise ValueError("materialization bundle contains training evidence")
            return self
        if self.target_profile_sha256 is None:
            raise ValueError("probe/training bundle lacks the exact target profile hash")
        required = (
            self.s3_snapshot,
            self.tokenizer_compatibility,
            self.ecr_image,
            self.finance_world_data,
        )
        if any(item is None for item in required):
            raise ValueError("probe/training bundle lacks required live evidence")
        if self.action is ExecutionAction.MEMORY_PROBE:
            if self.memory_probe is not None:
                raise ValueError("memory-probe bundle cannot contain prior probe evidence")
        elif self.action in {ExecutionAction.REHEARSAL, ExecutionAction.FULL}:
            if self.memory_probe is None:
                raise ValueError("training bundle lacks measured memory-probe evidence")
            if self.memory_probe.profile_sha256 != self.target_profile_sha256:
                raise ValueError("memory probe differs from target profile")
        else:
            raise ValueError("teacher trajectories cannot issue an execution bundle")
        return self


class ExecutionAuthorization(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.authorization.v2"] = (
        "distillery.qwen72b_fallback.authorization.v2"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    action: ExecutionAction
    launch_name: str
    evidence_bundle: VerifiedEvidenceBundle
    issued_unix_seconds: int = Field(gt=0)
    expires_unix_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def _authorization_invariants(self) -> ExecutionAuthorization:
        if self.action is not self.evidence_bundle.action:
            raise ValueError("authorization action differs from evidence bundle")
        if self.launch_name != self.evidence_bundle.launch_name:
            raise ValueError("authorization launch name differs from evidence bundle")
        if self.expires_unix_seconds <= self.issued_unix_seconds:
            raise ValueError("authorization expiry must follow issuance")
        maximum_lifetime = 900 if self.action is ExecutionAction.MATERIALIZE else 3 * 3600
        if self.expires_unix_seconds - self.issued_unix_seconds > maximum_lifetime:
            raise ValueError("authorization lifetime exceeds the action-specific limit")
        return self

    def require_current(
        self,
        *,
        action: ExecutionAction,
        launch_name: str,
        now_unix_seconds: int | None = None,
    ) -> None:
        now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
        if self.action is not action or self.launch_name != launch_name:
            raise RuntimeError("execution authorization does not match this launch")
        if now >= self.expires_unix_seconds:
            raise RuntimeError("execution authorization has expired")


class BlockedGate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gate: GateCode
    detail: str


class ReadinessState(StrEnum):
    BLOCKED = "blocked"
    AUTHORIZED = "authorized"


class ReadinessReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.qwen72b_fallback.readiness_report.v3"] = (
        "distillery.qwen72b_fallback.readiness_report.v3"
    )
    action: ExecutionAction
    launch_name: str
    state: ReadinessState
    blocked_gates: tuple[BlockedGate, ...]
    evidence_sha256: dict[str, str]
    authorization: ExecutionAuthorization | None

    @model_validator(mode="after")
    def _state_invariant(self) -> ReadinessReport:
        if self.state is ReadinessState.AUTHORIZED:
            if self.blocked_gates or self.authorization is None:
                raise ValueError("authorized report cannot contain blocked gates")
        elif self.authorization is not None:
            raise ValueError("blocked report cannot contain an authorization")
        return self


class LiveVerifier(Protocol):
    def verify_local_policy(self) -> LocalPolicyEvidence: ...

    def verify_reviews(self) -> ReviewClearanceEvidence: ...

    def verify_iam(self, action: ExecutionAction) -> IamScopeEvidence: ...

    def verify_conflicts(
        self,
        action: ExecutionAction,
        launch_name: str,
    ) -> ConflictEvidence: ...

    def verify_cost(
        self,
        action: ExecutionAction,
        profile: Qwen72BTrainingProfile | None,
    ) -> CostAuthorizationEvidence: ...

    def verify_s3_snapshot(self) -> S3SnapshotEvidence: ...

    def verify_tokenizer_compatibility(self) -> TokenizerCompatibilityEvidence: ...

    def verify_ecr_image(self) -> EcrImageEvidence: ...

    def verify_memory_probe(
        self,
        *,
        profile: Qwen72BTrainingProfile,
        image: EcrImageEvidence,
        local_policy: LocalPolicyEvidence,
    ) -> Qwen72BMemoryProbeEvidence: ...

    def verify_finance_world_data(
        self,
        profile: Qwen72BTrainingProfile,
    ) -> FinanceWorldCorpusEvidence: ...


def required_confirmation(action: ExecutionAction, launch_name: str) -> str:
    return f"EXECUTE QWEN72B {action.value.upper()} {launch_name}"


def seal_confirmation(
    *,
    action: ExecutionAction,
    launch_name: str,
    typed_text: str,
) -> ExecutionConfirmation:
    return ExecutionConfirmation.seal(
        action=action,
        launch_name=launch_name,
        typed_text=typed_text,
        operator_account_id="225989358036",
    )


def _capture(
    failures: list[BlockedGate],
    evidence: dict[str, str],
    gate: GateCode,
    operation: object,
) -> object | None:
    try:
        value = operation()
    except VerificationFailure as exc:
        failures.append(BlockedGate(gate=exc.gate, detail=exc.detail))
        return None
    except (OSError, RuntimeError, ValueError) as exc:
        failures.append(BlockedGate(gate=gate, detail=str(exc)))
        return None
    if not isinstance(value, HashBoundEvidence):
        failures.append(BlockedGate(gate=gate, detail="live verifier returned unbound evidence"))
        return None
    evidence[gate.value] = value.evidence_sha256
    return value


def evaluate_readiness(
    verifier: LiveVerifier,
    *,
    action: ExecutionAction,
    launch_name: str,
    profile: Qwen72BTrainingProfile | None,
    typed_confirmation: str | None,
    now_unix_seconds: int | None = None,
) -> ReadinessReport:
    """Probe real resources, then issue a short-lived authorization or block."""
    failures: list[BlockedGate] = []
    hashes: dict[str, str] = {}

    local = _capture(
        failures,
        hashes,
        GateCode.LOCAL_IDENTITY,
        verifier.verify_local_policy,
    )
    reviews = _capture(
        failures,
        hashes,
        GateCode.EXECUTION_REVIEWS,
        verifier.verify_reviews,
    )
    iam = _capture(
        failures,
        hashes,
        GateCode.IAM_SCOPE,
        lambda: verifier.verify_iam(action),
    )
    conflicts = _capture(
        failures,
        hashes,
        GateCode.ACTIVE_CONFLICTS,
        lambda: verifier.verify_conflicts(action, launch_name),
    )
    cost = _capture(
        failures,
        hashes,
        GateCode.COST_EXPOSURE,
        lambda: verifier.verify_cost(action, profile),
    )
    confirmation = _capture(
        failures,
        hashes,
        GateCode.TYPED_CONFIRMATION,
        lambda: seal_confirmation(
            action=action,
            launch_name=launch_name,
            typed_text=typed_confirmation or "",
        ),
    )
    base_objects = (local, reviews, iam, conflicts, cost, confirmation)
    if failures or not all(isinstance(item, HashBoundEvidence) for item in base_objects):
        return ReadinessReport(
            action=action,
            launch_name=launch_name,
            state=ReadinessState.BLOCKED,
            blocked_gates=tuple(failures),
            evidence_sha256=hashes,
            authorization=None,
        )

    s3_snapshot = None
    tokenizer_compatibility = None
    image = None
    probe = None
    finance_data = None
    if action is not ExecutionAction.MATERIALIZE:
        s3_snapshot = _capture(
            failures,
            hashes,
            GateCode.S3_BODY_HASHES,
            verifier.verify_s3_snapshot,
        )
        tokenizer_compatibility = _capture(
            failures,
            hashes,
            GateCode.TOKENIZER_PAIRS,
            verifier.verify_tokenizer_compatibility,
        )
        image = _capture(
            failures,
            hashes,
            GateCode.ECR_EXACT_IMAGE,
            verifier.verify_ecr_image,
        )
        if profile is None:
            failures.append(
                BlockedGate(
                    gate=GateCode.MEMORY_PROBE,
                    detail="training/probe action requires an exact profile",
                )
            )
        else:
            finance_data = _capture(
                failures,
                hashes,
                GateCode.FINANCE_WORLD_DATA,
                lambda: verifier.verify_finance_world_data(profile),
            )
            if (
                action
                in {
                    ExecutionAction.REHEARSAL,
                    ExecutionAction.FULL,
                    ExecutionAction.TEACHER_TRAJECTORIES,
                }
                and isinstance(image, EcrImageEvidence)
                and isinstance(local, LocalPolicyEvidence)
            ):
                probe = _capture(
                    failures,
                    hashes,
                    GateCode.MEMORY_PROBE,
                    lambda: verifier.verify_memory_probe(
                        profile=profile,
                        image=image,
                        local_policy=local,
                    ),
                )
    if action is ExecutionAction.TEACHER_TRAJECTORIES:
        failures.append(
            BlockedGate(
                gate=GateCode.TEACHER_TRAJECTORIES,
                detail=(
                    "teacher trajectory generation remains unavailable until a "
                    "separate non-empty generation protocol is reviewed"
                ),
            )
        )

    if failures:
        return ReadinessReport(
            action=action,
            launch_name=launch_name,
            state=ReadinessState.BLOCKED,
            blocked_gates=tuple(failures),
            evidence_sha256=hashes,
            authorization=None,
        )

    bundle = VerifiedEvidenceBundle.seal(
        action=action,
        launch_name=launch_name,
        target_profile_sha256=profile.profile_sha256 if profile is not None else None,
        local_policy=local,
        reviews=reviews,
        iam=iam,
        conflicts=conflicts,
        cost=cost,
        confirmation=confirmation,
        s3_snapshot=s3_snapshot,
        tokenizer_compatibility=tokenizer_compatibility,
        ecr_image=image,
        memory_probe=probe,
        finance_world_data=finance_data,
    )
    issued = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    authorization = ExecutionAuthorization.seal(
        action=action,
        launch_name=launch_name,
        evidence_bundle=bundle,
        issued_unix_seconds=issued,
        expires_unix_seconds=issued + (900 if action is ExecutionAction.MATERIALIZE else 3 * 3600),
    )
    hashes["evidence_bundle"] = bundle.evidence_sha256
    return ReadinessReport(
        action=action,
        launch_name=launch_name,
        state=ReadinessState.AUTHORIZED,
        blocked_gates=(),
        evidence_sha256=hashes,
        authorization=authorization,
    )
