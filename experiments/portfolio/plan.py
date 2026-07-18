"""Immutable TinyFable portfolio plan, evidence, wave, and cost contracts.

This module plans work only. It cannot launch training or publish a serving
registry. Materialization, campaign staging, selection, and publication are
separate fail-closed interfaces.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictStr, field_validator, model_validator

from distillery.contracts.base import FrozenDict, FrozenModel
from distillery.contracts.budgets import ProofGates, TrainingBudget
from distillery.contracts.hashing import (
    AwareDatetime,
    GitCommitSha,
    NonNegativeSafeInt,
    PositiveSafeInt,
    PrefixedSha256,
    Sha256Hex,
    canonical_json_bytes,
    content_sha256,
    sha256_hex,
)
from distillery.contracts.ids import DatasetId, RunId
from distillery.contracts.tasks import SplitName
from distillery.proof.protocol_v2 import (
    PROOF_PROTOCOL_ID_V2,
    finance_proof_v2_sha256,
)
from experiments.aws_smoke.campaign_index import (
    AcceleratorType,
    CampaignPricingEvidenceReference,
    HardwareInstanceType,
    HardwareProfileId,
    campaign_hardware_profile,
)

DECISION_ID = "decision_tinyfable_tiered_portfolio_v3"
SCREEN_SEED = int(ProofGates().required_seed_screen)
REPLICATION_SEED = int(ProofGates().required_seed_replication)
BOOTSTRAP_RESAMPLES = int(ProofGates().bootstrap_resamples)
PHYSICAL_SLOTS = 16
ACCOUNT_CEILING_MICROUSD = 10_000 * 1_000_000
GIB = 1024**3
A10G_BYTES = 24 * GIB
A100_BYTES = 80 * GIB

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
ModelId = Annotated[StrictStr, Field(pattern=r"^model_[a-z0-9][a-z0-9_-]{1,180}$")]
ArtifactId = Annotated[StrictStr, Field(pattern=r"^art_[a-z0-9][a-z0-9_-]{1,180}$")]


class Task(StrEnum):
    TRANSACTION_REVIEW = "transaction_review"
    VARIANCE_ANALYSIS = "variance_analysis"
    MERCHANT_TAGGING = "merchant_tagging"
    CASH_RECONCILIATION = "cash_reconciliation"


TASKS: tuple[Task, ...] = tuple(Task)


class Tier(StrEnum):
    NANO = "nano"
    CORE = "core"
    PLUS = "plus"


TIERS: tuple[Tier, ...] = tuple(Tier)


class Surface(StrEnum):
    LOGIT = "logit"
    SEQUENCE = "sequence"
    REPLICATION = "replication"


class PortfolioArm(StrEnum):
    ORACLE_SFT = "oracle_sft"
    SEQUENCE_KD = "sequence_kd"
    LOGIT_KD = "logit_kd"
    CE_ABLATION = "ce_ablation"
    SEQUENCE_CE_CONTROL = "sequence_ce_control"


Role = Literal["generalist", "specialist"]
EvidenceRole = Literal["teacher", "student"]
Recipe = Literal["sequence.v1", "logit.v1"]
ComparisonPosition = Literal["treatment", "control"]
ProbeKind = Literal[
    "oracle_sft_train_step",
    "sequence_kd_train_step",
    "logit_kd_joint_train_step",
    "ce_ablation_train_step",
    "optional_merged_export",
]

REQUIRED_PROBES: tuple[ProbeKind, ...] = (
    "oracle_sft_train_step",
    "sequence_kd_train_step",
    "logit_kd_joint_train_step",
    "ce_ablation_train_step",
)


def arm_recipe(arm: PortfolioArm) -> Recipe:
    return {
        PortfolioArm.ORACLE_SFT: "sequence.v1",
        PortfolioArm.SEQUENCE_KD: "sequence.v1",
        PortfolioArm.LOGIT_KD: "logit.v1",
        PortfolioArm.CE_ABLATION: "logit.v1",
        PortfolioArm.SEQUENCE_CE_CONTROL: "sequence.v1",
    }[arm]


def campaign_arm(arm: PortfolioArm) -> str:
    """Map portfolio controls onto the reviewed trainer arm vocabulary."""
    if arm is PortfolioArm.SEQUENCE_CE_CONTROL:
        return PortfolioArm.ORACLE_SFT.value
    return arm.value


def _s3_prefix(value: str, *, field_name: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or not value.endswith("/")
    ):
        raise ValueError(f"{field_name} must be a plain s3:// prefix ending in '/'")
    return value


class ModelRoleEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.role_evidence.v2"] = (
        "distillery.portfolio.role_evidence.v2"
    )
    role: EvidenceRole
    model_id: StrictStr = Field(min_length=1)
    revision: GitCommitSha
    model_config_sha256: Sha256Hex
    tokenizer_sha256: Sha256Hex
    chat_template_sha256: Sha256Hex
    special_tokens_sha256: Sha256Hex
    license_evidence_sha256: Sha256Hex
    output_use_evidence_sha256: Sha256Hex
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _bound(self) -> ModelRoleEvidence:
        if self.evidence_sha256 != _role_hash(self):
            raise ValueError("role evidence hash mismatch")
        return self


def _role_hash(evidence: ModelRoleEvidence) -> str:
    payload = evidence.model_dump(mode="json", exclude={"evidence_sha256"})
    return content_sha256(payload)


def role_evidence(
    *,
    role: EvidenceRole,
    model_id: str,
    revision: str,
    model_config_sha256: str,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    special_tokens_sha256: str,
    license_sha256: str,
    output_use_sha256: str,
) -> ModelRoleEvidence:
    provisional = ModelRoleEvidence.model_construct(
        role=role,
        model_id=model_id,
        revision=revision,
        model_config_sha256=model_config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        special_tokens_sha256=special_tokens_sha256,
        license_evidence_sha256=license_sha256,
        output_use_evidence_sha256=output_use_sha256,
        evidence_sha256="0" * 64,
    )
    return ModelRoleEvidence.model_validate(
        {**provisional.model_dump(mode="python"), "evidence_sha256": _role_hash(provisional)}
    )


class ModelPair(FrozenModel):
    schema_version: Literal["distillery.portfolio.model_pair.v2"] = (
        "distillery.portfolio.model_pair.v2"
    )
    teacher: ModelRoleEvidence
    student: ModelRoleEvidence
    binding_sha256: Sha256Hex

    @model_validator(mode="after")
    def _roles_and_tokenizer(self) -> ModelPair:
        if self.teacher.role != "teacher" or self.student.role != "student":
            raise ValueError("pair requires role-specific teacher and student evidence")
        for field in ("tokenizer_sha256", "chat_template_sha256", "special_tokens_sha256"):
            if getattr(self.teacher, field) != getattr(self.student, field):
                raise ValueError(f"logit KD requires matching {field}")
        if self.binding_sha256 != _pair_hash(self.teacher, self.student):
            raise ValueError("model pair hash mismatch")
        return self


def _pair_hash(teacher: ModelRoleEvidence, student: ModelRoleEvidence) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.portfolio.model_pair_binding.v2",
            "teacher_evidence_sha256": teacher.evidence_sha256,
            "student_evidence_sha256": student.evidence_sha256,
        }
    )


def model_pair(teacher: ModelRoleEvidence, student: ModelRoleEvidence) -> ModelPair:
    return ModelPair(
        teacher=teacher,
        student=student,
        binding_sha256=_pair_hash(teacher, student),
    )


class Candidate(FrozenModel):
    tier: Tier
    name: Literal["TinyFable Nano", "TinyFable Core", "TinyFable Plus"]
    name_is_metadata_not_claim: Literal[True] = True
    priority: Literal["fastest_baseline", "priority_larger", "stretch"]
    parameter_label: Literal["494M", "1.5B", "3B"]
    pair: ModelPair
    larger_is_better_assumption: Literal[False] = False
    throughput_tokens_per_second: None = None
    cost_per_1k_tokens_microusd: None = None
    descriptor_sha256: Sha256Hex

    @model_validator(mode="after")
    def _known_pair(self) -> Candidate:
        expected = _tier_spec(self.tier)
        actual = (
            self.name,
            self.priority,
            self.parameter_label,
            self.pair.teacher.model_id,
            self.pair.student.model_id,
        )
        if actual != expected:
            raise ValueError("tier/model-role mismatch")
        if self.descriptor_sha256 != _candidate_hash(self):
            raise ValueError("candidate hash mismatch")
        return self


def _tier_spec(tier: Tier) -> tuple[str, ...]:
    return {
        Tier.NANO: (
            "TinyFable Nano",
            "fastest_baseline",
            "494M",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ),
        Tier.CORE: (
            "TinyFable Core",
            "priority_larger",
            "1.5B",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        Tier.PLUS: (
            "TinyFable Plus",
            "stretch",
            "3B",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
        ),
    }[tier]


def _candidate_hash(value: Candidate) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"descriptor_sha256"}))


def candidate(tier: Tier, pair: ModelPair) -> Candidate:
    name, priority, label, _, _ = _tier_spec(tier)
    provisional = Candidate.model_construct(
        tier=tier,
        name=name,
        priority=priority,
        parameter_label=label,
        pair=pair,
        descriptor_sha256="0" * 64,
    )
    return Candidate.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "descriptor_sha256": _candidate_hash(provisional),
        }
    )


class RuntimeBinding(FrozenModel):
    tier: Tier
    hardware_profile: HardwareProfileId
    instance_type: HardwareInstanceType
    accelerator: AcceleratorType
    runtime_image_digest: PrefixedSha256
    package_lock_hash: Sha256Hex
    source_revision: GitCommitSha
    runtime_sha256: Sha256Hex

    @model_validator(mode="after")
    def _profile(self) -> RuntimeBinding:
        expected_profile = (
            "g5-48xlarge-8xa10g-independent-v1"
            if self.tier is Tier.NANO
            else "p4de-24xlarge-8xa100-80gb-independent-v1"
        )
        if self.hardware_profile != expected_profile:
            raise ValueError("tier uses the wrong hardware profile")
        profile = campaign_hardware_profile(self.hardware_profile)
        if (self.instance_type, self.accelerator) != (
            profile.instance_type,
            profile.accelerator,
        ):
            raise ValueError("runtime hardware labels do not resolve through campaign profile")
        if self.runtime_sha256 != _runtime_hash(self):
            raise ValueError("runtime binding hash mismatch")
        return self


def _runtime_hash(value: RuntimeBinding) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"runtime_sha256"}))


def runtime_binding(
    *,
    tier: Tier,
    runtime_image_digest: str,
    package_lock_hash: str,
    source_revision: str,
) -> RuntimeBinding:
    profile_id: HardwareProfileId = (
        "g5-48xlarge-8xa10g-independent-v1"
        if tier is Tier.NANO
        else "p4de-24xlarge-8xa100-80gb-independent-v1"
    )
    profile = campaign_hardware_profile(profile_id)
    provisional = RuntimeBinding.model_construct(
        tier=tier,
        hardware_profile=profile.profile_id,
        instance_type=profile.instance_type,
        accelerator=profile.accelerator,
        runtime_image_digest=runtime_image_digest,
        package_lock_hash=package_lock_hash,
        source_revision=source_revision,
        runtime_sha256="0" * 64,
    )
    return RuntimeBinding.model_validate(
        {**provisional.model_dump(mode="python"), "runtime_sha256": _runtime_hash(provisional)}
    )


class TrainingProtocol(FrozenModel):
    schema_version: Literal["distillery.portfolio.training_protocol.v2"] = (
        "distillery.portfolio.training_protocol.v2"
    )
    finance_world: Literal["finance_world.v2"] = "finance_world.v2"
    proof_protocol_id: Literal["finance-proof.v2"] = PROOF_PROTOCOL_ID_V2
    proof_protocol_sha256: Sha256Hex
    screen_seed: Literal[17] = SCREEN_SEED
    replication_seed: Literal[23] = REPLICATION_SEED
    bootstrap_resamples: Literal[10000] = BOOTSTRAP_RESAMPLES
    max_length: PositiveSafeInt
    max_completion: PositiveSafeInt
    max_steps: PositiveSafeInt
    microbatch: PositiveSafeInt
    grad_accumulation: PositiveSafeInt
    lora_rank: PositiveSafeInt
    lora_alpha: PositiveSafeInt
    lora_dropout: FiniteFloat
    logit_temperature: FiniteFloat
    logit_kd_weight: FiniteFloat
    logit_hard_ce_weight: FiniteFloat
    vocab_chunk: PositiveSafeInt
    max_runtime_seconds: PositiveSafeInt
    student_precision: Literal["qlora_nf4"] = "qlora_nf4"
    logit_teacher_precision: Literal["bf16_frozen_no_grad"] = "bf16_frozen_no_grad"
    deterministic_algorithms: Literal[True] = True
    network_isolation: Literal[True] = True
    protocol_sha256: Sha256Hex

    @model_validator(mode="after")
    def _proof_constants(self) -> TrainingProtocol:
        budget = TrainingBudget()
        expected = (
            finance_proof_v2_sha256(),
            int(ProofGates().required_seed_screen),
            int(ProofGates().required_seed_replication),
            int(ProofGates().bootstrap_resamples),
            budget.max_length,
            budget.max_completion,
            budget.max_steps,
            budget.microbatch,
            budget.grad_accumulation,
            budget.lora_rank,
            budget.lora_alpha,
            budget.lora_dropout,
            budget.logit_temperature,
            budget.kd_weight,
            budget.hard_ce_weight,
            budget.vocab_chunk,
            budget.max_runtime_seconds,
        )
        actual = (
            self.proof_protocol_sha256,
            self.screen_seed,
            self.replication_seed,
            self.bootstrap_resamples,
            self.max_length,
            self.max_completion,
            self.max_steps,
            self.microbatch,
            self.grad_accumulation,
            self.lora_rank,
            self.lora_alpha,
            self.lora_dropout,
            self.logit_temperature,
            self.logit_kd_weight,
            self.logit_hard_ce_weight,
            self.vocab_chunk,
            self.max_runtime_seconds,
        )
        if actual != expected:
            raise ValueError("portfolio protocol must use TrainingBudget and ProofGates constants")
        if self.protocol_sha256 != _training_protocol_hash(self):
            raise ValueError("training protocol hash mismatch")
        return self


def _training_protocol_hash(value: TrainingProtocol) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"protocol_sha256"}))


def training_protocol() -> TrainingProtocol:
    budget = TrainingBudget()
    provisional = TrainingProtocol.model_construct(
        proof_protocol_sha256=finance_proof_v2_sha256(),
        max_length=budget.max_length,
        max_completion=budget.max_completion,
        max_steps=budget.max_steps,
        microbatch=budget.microbatch,
        grad_accumulation=budget.grad_accumulation,
        lora_rank=budget.lora_rank,
        lora_alpha=budget.lora_alpha,
        lora_dropout=budget.lora_dropout,
        logit_temperature=budget.logit_temperature,
        logit_kd_weight=budget.kd_weight,
        logit_hard_ce_weight=budget.hard_ce_weight,
        vocab_chunk=budget.vocab_chunk,
        max_runtime_seconds=budget.max_runtime_seconds,
        protocol_sha256="0" * 64,
    )
    return TrainingProtocol.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "protocol_sha256": _training_protocol_hash(provisional),
        }
    )


class DatasetView(FrozenModel):
    view_id: DatasetId
    parent_bundle_sha256: Sha256Hex
    relative_prefix: StrictStr = Field(pattern=r"^views/[a-z0-9][a-z0-9_-]{1,126}/$")
    task_filter: tuple[Task, ...] = Field(min_length=1, max_length=4)
    content_sha256: Sha256Hex
    split_sha256: FrozenDict[SplitName, Sha256Hex]
    filter_protocol_sha256: Sha256Hex
    view_sha256: Sha256Hex

    @model_validator(mode="after")
    def _sealed_filter(self) -> DatasetView:
        if self.task_filter != TASKS and len(self.task_filter) != 1:
            raise ValueError("dataset view must be all four tasks or exactly one task")
        if {SplitName.TRAIN, SplitName.VALIDATION} - set(self.split_sha256):
            raise ValueError("dataset view requires train and validation split hashes")
        expected_filter = content_sha256(
            {
                "schema_version": "distillery.portfolio.task_filter.v1",
                "finance_world": "finance_world.v2",
                "tasks": [task.value for task in self.task_filter],
                "parent_bundle_sha256": self.parent_bundle_sha256,
            }
        )
        if self.filter_protocol_sha256 != expected_filter:
            raise ValueError("task filter protocol hash mismatch")
        if self.view_sha256 != _dataset_view_hash(self):
            raise ValueError("dataset view hash mismatch")
        return self


def _dataset_view_hash(value: DatasetView) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"view_sha256"}))


def dataset_view(
    *,
    view_id: str,
    parent_bundle_sha256: str,
    relative_prefix: str,
    task_filter: tuple[Task, ...],
    content_digest: str,
    split_sha256: dict[SplitName | str, str],
) -> DatasetView:
    filter_sha = content_sha256(
        {
            "schema_version": "distillery.portfolio.task_filter.v1",
            "finance_world": "finance_world.v2",
            "tasks": [task.value for task in task_filter],
            "parent_bundle_sha256": parent_bundle_sha256,
        }
    )
    provisional = DatasetView.model_construct(
        view_id=view_id,
        parent_bundle_sha256=parent_bundle_sha256,
        relative_prefix=relative_prefix,
        task_filter=task_filter,
        content_sha256=content_digest,
        split_sha256=split_sha256,
        filter_protocol_sha256=filter_sha,
        view_sha256="0" * 64,
    )
    return DatasetView.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "view_sha256": _dataset_view_hash(provisional),
        }
    )


class DatasetBundle(FrozenModel):
    schema_version: Literal["distillery.portfolio.dataset_bundle.v1"] = (
        "distillery.portfolio.dataset_bundle.v1"
    )
    bundle_id: DatasetId
    uri: StrictStr
    content_sha256: Sha256Hex
    split_sha256: FrozenDict[SplitName, Sha256Hex]
    views: tuple[DatasetView, ...] = Field(min_length=5, max_length=5)
    binding_sha256: Sha256Hex

    @field_validator("uri")
    @classmethod
    def _uri(cls, value: str) -> str:
        return _s3_prefix(value, field_name="dataset bundle uri")

    @model_validator(mode="after")
    def _views(self) -> DatasetBundle:
        if {SplitName.TRAIN, SplitName.VALIDATION} - set(self.split_sha256):
            raise ValueError("dataset bundle requires train and validation split hashes")
        filters = [view.task_filter for view in self.views]
        expected = [TASKS, *((task,) for task in TASKS)]
        if filters != expected:
            raise ValueError("dataset views must be generalist then one view per task")
        if any(view.parent_bundle_sha256 != self.content_sha256 for view in self.views):
            raise ValueError("dataset view parent hash mismatch")
        if len({view.view_id for view in self.views}) != len(self.views):
            raise ValueError("dataset view IDs must be unique")
        if self.binding_sha256 != _dataset_bundle_hash(self):
            raise ValueError("dataset bundle hash mismatch")
        return self

    def view_for(self, tasks: tuple[Task, ...]) -> DatasetView:
        matches = [view for view in self.views if view.task_filter == tasks]
        if len(matches) != 1:
            raise ValueError(f"dataset bundle has no unique view for {tasks!r}")
        return matches[0]


def _dataset_bundle_hash(value: DatasetBundle) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"binding_sha256"}))


def build_dataset_bundle(
    *,
    bundle_id: str,
    uri: str,
    content_digest: str,
    split_sha256: dict[SplitName | str, str],
    views: tuple[DatasetView, ...],
) -> DatasetBundle:
    provisional = DatasetBundle.model_construct(
        bundle_id=bundle_id,
        uri=uri,
        content_sha256=content_digest,
        split_sha256=split_sha256,
        views=views,
        binding_sha256="0" * 64,
    )
    return DatasetBundle.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "binding_sha256": _dataset_bundle_hash(provisional),
        }
    )


class PricingEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.pricing.v2"] = "distillery.portfolio.pricing.v2"
    source_uri: StrictStr
    region: StrictStr = Field(min_length=1)
    instance_type: HardwareInstanceType
    current_hourly_price_microusd: PositiveSafeInt
    currency: Literal["USD"] = "USD"
    attestor: StrictStr = Field(min_length=1)
    attested_at: AwareDatetime
    effective_at: AwareDatetime
    expires_at: AwareDatetime
    evidence_bytes_sha256: Sha256Hex
    evidence_size_bytes: PositiveSafeInt
    pricing_sha256: Sha256Hex

    @field_validator("source_uri")
    @classmethod
    def _source(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "s3"} or not parsed.netloc:
            raise ValueError("pricing source must be an https:// or s3:// URI")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("pricing source must not contain credentials")
        return value

    @model_validator(mode="after")
    def _bound(self) -> PricingEvidence:
        if not self.effective_at <= self.attested_at < self.expires_at:
            raise ValueError("pricing attestation must fall inside its validity interval")
        if self.pricing_sha256 != _pricing_hash(self):
            raise ValueError("pricing evidence hash mismatch")
        return self

    def verify_evidence_bytes(self, evidence_bytes: bytes) -> None:
        if len(evidence_bytes) != self.evidence_size_bytes:
            raise ValueError("pricing evidence byte length mismatch")
        if sha256_hex(evidence_bytes) != self.evidence_bytes_sha256:
            raise ValueError("pricing evidence bytes hash mismatch")

    def campaign_reference(self) -> CampaignPricingEvidenceReference:
        return CampaignPricingEvidenceReference(
            reference=self.source_uri,
            evidence_sha256=self.pricing_sha256,
            region=self.region,
            instance_type=self.instance_type,
            hourly_price_microusd=self.current_hourly_price_microusd,
            attested_by=self.attestor,
            attested_at=self.attested_at,
        )


def _pricing_hash(value: PricingEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"pricing_sha256"}))


def pricing_evidence(
    *,
    source_uri: str,
    region: str,
    instance_type: HardwareInstanceType,
    current_hourly_price_microusd: int,
    attestor: str,
    attested_at: datetime,
    effective_at: datetime,
    expires_at: datetime,
    evidence_bytes: bytes,
) -> PricingEvidence:
    provisional = PricingEvidence.model_construct(
        source_uri=source_uri,
        region=region,
        instance_type=instance_type,
        current_hourly_price_microusd=current_hourly_price_microusd,
        attestor=attestor,
        attested_at=attested_at,
        effective_at=effective_at,
        expires_at=expires_at,
        evidence_bytes_sha256=sha256_hex(evidence_bytes),
        evidence_size_bytes=len(evidence_bytes),
        pricing_sha256="0" * 64,
    )
    return PricingEvidence.model_validate(
        {**provisional.model_dump(mode="python"), "pricing_sha256": _pricing_hash(provisional)}
    )


class ProtocolGate(FrozenModel):
    tier: Tier
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    runtime_sha256: Sha256Hex
    training_protocol_sha256: Sha256Hex
    instance_type: HardwareInstanceType
    accelerator: AcceleratorType
    capacity_bytes: PositiveSafeInt
    max_peak_basis_points: Literal[8500] = 8500
    min_headroom_bytes: PositiveSafeInt
    required_probes: tuple[ProbeKind, ...] = REQUIRED_PROBES
    optional_merge_requires_probe: Literal[True] = True
    gate_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hardware(self) -> ProtocolGate:
        expected = (
            ("ml.g5.48xlarge", "NVIDIA A10G", A10G_BYTES, 4 * GIB)
            if self.tier is Tier.NANO
            else ("ml.p4de.24xlarge", "NVIDIA A100 80GB", A100_BYTES, 8 * GIB)
        )
        if (
            self.instance_type,
            self.accelerator,
            self.capacity_bytes,
            self.min_headroom_bytes,
        ) != expected:
            raise ValueError("exact hardware memory gate mismatch")
        if self.gate_sha256 != _gate_hash(self):
            raise ValueError("protocol gate hash mismatch")
        return self


def _gate_hash(value: ProtocolGate) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"gate_sha256"}))


def protocol_gate(
    item: Candidate,
    runtime: RuntimeBinding,
    protocol: TrainingProtocol,
) -> ProtocolGate:
    if item.tier is not runtime.tier:
        raise ValueError("candidate/runtime tier mismatch")
    capacity, headroom = (A10G_BYTES, 4 * GIB) if item.tier is Tier.NANO else (A100_BYTES, 8 * GIB)
    provisional = ProtocolGate.model_construct(
        tier=item.tier,
        candidate_sha256=item.descriptor_sha256,
        pair_sha256=item.pair.binding_sha256,
        runtime_sha256=runtime.runtime_sha256,
        training_protocol_sha256=protocol.protocol_sha256,
        instance_type=runtime.instance_type,
        accelerator=runtime.accelerator,
        capacity_bytes=capacity,
        min_headroom_bytes=headroom,
        gate_sha256="0" * 64,
    )
    return ProtocolGate.model_validate(
        {**provisional.model_dump(mode="python"), "gate_sha256": _gate_hash(provisional)}
    )


class MemoryProbe(FrozenModel):
    schema_version: Literal["distillery.portfolio.memory_probe.v2"] = (
        "distillery.portfolio.memory_probe.v2"
    )
    tier: Tier
    kind: ProbeKind
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    gate_sha256: Sha256Hex
    teacher_role_evidence_sha256: Sha256Hex
    student_role_evidence_sha256: Sha256Hex
    teacher_license_evidence_sha256: Sha256Hex
    student_license_evidence_sha256: Sha256Hex
    teacher_output_use_evidence_sha256: Sha256Hex
    student_output_use_evidence_sha256: Sha256Hex
    runtime_sha256: Sha256Hex
    runtime_image_digest: PrefixedSha256
    image_evidence_sha256: Sha256Hex
    instance_type: HardwareInstanceType
    accelerator: AcceleratorType
    peak_bytes: PositiveSafeInt
    capacity_bytes: PositiveSafeInt
    headroom_bytes: NonNegativeSafeInt
    measured_at: AwareDatetime
    attestor: StrictStr = Field(min_length=1)
    raw_evidence_sha256: Sha256Hex
    raw_evidence_size_bytes: PositiveSafeInt
    probe_sha256: Sha256Hex

    @model_validator(mode="after")
    def _measurement(self) -> MemoryProbe:
        if self.peak_bytes >= self.capacity_bytes:
            raise ValueError("measured peak must be below capacity")
        if self.headroom_bytes != self.capacity_bytes - self.peak_bytes:
            raise ValueError("memory headroom arithmetic mismatch")
        if self.probe_sha256 != _memory_probe_hash(self):
            raise ValueError("memory probe hash mismatch")
        return self


def _memory_probe_hash(value: MemoryProbe) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"probe_sha256"}))


def memory_probe(**values: object) -> MemoryProbe:
    provisional = MemoryProbe.model_construct(**values, probe_sha256="0" * 64)
    return MemoryProbe.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "probe_sha256": _memory_probe_hash(provisional),
        }
    )


class ReadinessEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.readiness.v2"] = (
        "distillery.portfolio.readiness.v2"
    )
    tier: Tier
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    gate_sha256: Sha256Hex
    runtime_sha256: Sha256Hex
    runtime_image_digest: PrefixedSha256
    image_evidence_sha256: Sha256Hex
    image_evidence_size_bytes: PositiveSafeInt
    teacher_role_evidence_sha256: Sha256Hex
    student_role_evidence_sha256: Sha256Hex
    teacher_license_evidence_sha256: Sha256Hex
    student_license_evidence_sha256: Sha256Hex
    teacher_output_use_evidence_sha256: Sha256Hex
    student_output_use_evidence_sha256: Sha256Hex
    probes: tuple[MemoryProbe, ...] = Field(min_length=4, max_length=5)
    evidence_manifest_sha256: Sha256Hex
    readiness_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash(self) -> ReadinessEvidence:
        if self.readiness_sha256 != _readiness_hash(self):
            raise ValueError("readiness evidence hash mismatch")
        return self


def _readiness_hash(value: ReadinessEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"readiness_sha256"}))


def readiness_evidence(**values: object) -> ReadinessEvidence:
    provisional = ReadinessEvidence.model_construct(**values, readiness_sha256="0" * 64)
    return ReadinessEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "readiness_sha256": _readiness_hash(provisional),
        }
    )


def validate_readiness(
    item: Candidate,
    runtime: RuntimeBinding,
    gate: ProtocolGate,
    evidence: ReadinessEvidence,
    *,
    merge_requested: bool = False,
) -> None:
    teacher = item.pair.teacher
    student = item.pair.student
    expected = (
        item.tier,
        item.descriptor_sha256,
        item.pair.binding_sha256,
        gate.gate_sha256,
        runtime.runtime_sha256,
        runtime.runtime_image_digest,
        teacher.evidence_sha256,
        student.evidence_sha256,
        teacher.license_evidence_sha256,
        student.license_evidence_sha256,
        teacher.output_use_evidence_sha256,
        student.output_use_evidence_sha256,
    )
    actual = (
        evidence.tier,
        evidence.candidate_sha256,
        evidence.pair_sha256,
        evidence.gate_sha256,
        evidence.runtime_sha256,
        evidence.runtime_image_digest,
        evidence.teacher_role_evidence_sha256,
        evidence.student_role_evidence_sha256,
        evidence.teacher_license_evidence_sha256,
        evidence.student_license_evidence_sha256,
        evidence.teacher_output_use_evidence_sha256,
        evidence.student_output_use_evidence_sha256,
    )
    if actual != expected:
        raise ValueError("readiness model-role/license/output/image binding mismatch")
    required = set(REQUIRED_PROBES)
    if merge_requested:
        required.add("optional_merged_export")
    kinds = [probe.kind for probe in evidence.probes]
    if len(kinds) != len(set(kinds)) or set(kinds) != required:
        raise ValueError("memory probes do not exactly cover required paths")
    max_peak = gate.capacity_bytes * gate.max_peak_basis_points // 10_000
    for probe in evidence.probes:
        probe_expected = (
            evidence.tier,
            evidence.candidate_sha256,
            evidence.pair_sha256,
            evidence.gate_sha256,
            evidence.teacher_role_evidence_sha256,
            evidence.student_role_evidence_sha256,
            evidence.teacher_license_evidence_sha256,
            evidence.student_license_evidence_sha256,
            evidence.teacher_output_use_evidence_sha256,
            evidence.student_output_use_evidence_sha256,
            evidence.runtime_sha256,
            evidence.runtime_image_digest,
            evidence.image_evidence_sha256,
            gate.instance_type,
            gate.accelerator,
            gate.capacity_bytes,
        )
        probe_actual = (
            probe.tier,
            probe.candidate_sha256,
            probe.pair_sha256,
            probe.gate_sha256,
            probe.teacher_role_evidence_sha256,
            probe.student_role_evidence_sha256,
            probe.teacher_license_evidence_sha256,
            probe.student_license_evidence_sha256,
            probe.teacher_output_use_evidence_sha256,
            probe.student_output_use_evidence_sha256,
            probe.runtime_sha256,
            probe.runtime_image_digest,
            probe.image_evidence_sha256,
            probe.instance_type,
            probe.accelerator,
            probe.capacity_bytes,
        )
        if probe_actual != probe_expected:
            raise ValueError("memory probe evidence binding mismatch")
        if probe.peak_bytes > max_peak:
            raise ValueError("measured peak exceeds 85% ceiling")
        if probe.headroom_bytes < gate.min_headroom_bytes:
            raise ValueError("measured headroom below exact minimum")


class PlannedRunSlot(FrozenModel):
    state: Literal["planned"] = "planned"
    slot: NonNegativeSafeInt = Field(le=15)
    node: Literal[0, 1]
    gpu: NonNegativeSafeInt = Field(le=7)
    run_id: RunId
    model_id: ModelId
    artifact_id: ArtifactId
    role: Role
    arm: PortfolioArm
    tasks: tuple[Task, ...] = Field(min_length=1, max_length=4)
    recipe: Recipe
    seed: Literal[17, 23]
    dataset_view_sha256: Sha256Hex
    comparison_id: StrictStr = Field(pattern=r"^cmp_[a-z0-9][a-z0-9_-]{1,126}$")
    comparison_position: ComparisonPosition
    comparison_sha256: Sha256Hex
    protocol_sha256: Sha256Hex
    materialization_binding_sha256: Sha256Hex
    output_prefix: StrictStr
    artifact_prefix: StrictStr
    rationale: StrictStr = Field(min_length=1)

    @field_validator("output_prefix", "artifact_prefix")
    @classmethod
    def _prefix(cls, value: str, info: object) -> str:
        return _s3_prefix(value, field_name=str(getattr(info, "field_name", "prefix")))

    @model_validator(mode="after")
    def _shape(self) -> PlannedRunSlot:
        if (self.node, self.gpu) != (self.slot % 2, self.slot // 2):
            raise ValueError("slot must preserve deterministic node/GPU identity")
        if self.recipe != arm_recipe(self.arm):
            raise ValueError("recipe/arm mismatch")
        if self.role == "generalist" and self.tasks != TASKS:
            raise ValueError("generalists must cover all four tasks")
        if self.role == "specialist" and len(self.tasks) != 1:
            raise ValueError("specialists require exactly one task-filtered dataset")
        if f"s{self.seed}" not in self.run_id or f"s{self.seed}" not in self.model_id:
            raise ValueError("seed must be bound into run and model IDs")
        return self


class NotStartedSlot(FrozenModel):
    state: Literal["not_started"] = "not_started"
    slot: NonNegativeSafeInt = Field(le=15)
    node: Literal[0, 1]
    gpu: NonNegativeSafeInt = Field(le=7)
    reason: StrictStr = Field(min_length=1)
    cost_included: Literal[True] = True

    @model_validator(mode="after")
    def _identity(self) -> NotStartedSlot:
        if (self.node, self.gpu) != (self.slot % 2, self.slot // 2):
            raise ValueError("not-started slot must preserve node/GPU identity")
        return self


PortfolioSlot = PlannedRunSlot | NotStartedSlot


def _comparison_hash(slot: PlannedRunSlot) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.portfolio.matched_comparison.v2",
            "role": slot.role,
            "tasks": [task.value for task in slot.tasks],
            "recipe": slot.recipe,
            "seed": slot.seed,
            "dataset_view_sha256": slot.dataset_view_sha256,
            "comparison_id": slot.comparison_id,
        }
    )


def _slot_protocol_hash(slot: PlannedRunSlot, shared_protocol_sha256: str) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.portfolio.slot_protocol.v2",
            "shared_protocol_sha256": shared_protocol_sha256,
            "slot": slot.slot,
            "run_id": slot.run_id,
            "model_id": slot.model_id,
            "role": slot.role,
            "arm": slot.arm.value,
            "tasks": [task.value for task in slot.tasks],
            "recipe": slot.recipe,
            "seed": slot.seed,
            "dataset_view_sha256": slot.dataset_view_sha256,
            "comparison_id": slot.comparison_id,
            "comparison_position": slot.comparison_position,
            "comparison_sha256": slot.comparison_sha256,
        }
    )


def _materialization_hash(slot: PlannedRunSlot) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.portfolio.materialization_binding.v1",
            "slot_protocol_sha256": slot.protocol_sha256,
            "run_id": slot.run_id,
            "model_id": slot.model_id,
            "artifact_id": slot.artifact_id,
            "output_prefix": slot.output_prefix,
            "artifact_prefix": slot.artifact_prefix,
        }
    )


class Wave(FrozenModel):
    schema_version: Literal["distillery.portfolio.wave.v2"] = "distillery.portfolio.wave.v2"
    wave_id: StrictStr = Field(pattern=r"^wave_[a-z0-9][a-z0-9_-]{1,126}$")
    phase: Literal["screen", "replication"]
    surface: Surface
    tier: Tier
    candidate_sha256: Sha256Hex
    pair_sha256: Sha256Hex
    gate_sha256: Sha256Hex
    runtime_sha256: Sha256Hex
    training_protocol_sha256: Sha256Hex
    dataset_bundle_sha256: Sha256Hex
    instance_type: HardwareInstanceType
    hardware_profile: HardwareProfileId
    accelerator: AcceleratorType
    seed: Literal[17, 23]
    selection_lock_sha256: Sha256Hex | None = None
    shared_protocol_sha256: Sha256Hex
    slots: tuple[PortfolioSlot, ...] = Field(min_length=16, max_length=16)
    matrix_sha256: Sha256Hex

    @model_validator(mode="after")
    def _matrix(self) -> Wave:
        if [slot.slot for slot in self.slots] != list(range(PHYSICAL_SLOTS)):
            raise ValueError("wave slots must remain ordered 0..15")
        if self.phase == "screen" and self.seed != SCREEN_SEED:
            raise ValueError("screen waves require seed 17")
        if self.phase == "replication" and (
            self.seed != REPLICATION_SEED or self.selection_lock_sha256 is None
        ):
            raise ValueError("replication waves require seed 23 and a selection lock")
        if self.surface is Surface.REPLICATION and self.phase != "replication":
            raise ValueError("replication surface requires replication phase")
        if self.shared_protocol_sha256 != _wave_shared_hash(self):
            raise ValueError("wave shared protocol hash mismatch")
        active = [slot for slot in self.slots if isinstance(slot, PlannedRunSlot)]
        idle = [slot for slot in self.slots if isinstance(slot, NotStartedSlot)]
        if any(slot.seed != self.seed for slot in active):
            raise ValueError("every active slot must use the wave seed")
        for field in (
            "run_id",
            "model_id",
            "artifact_id",
            "output_prefix",
            "artifact_prefix",
            "materialization_binding_sha256",
        ):
            values = [getattr(slot, field) for slot in active]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate active-slot {field}")
        for slot in active:
            if slot.comparison_sha256 != _comparison_hash(slot):
                raise ValueError("matched comparison hash mismatch")
            if slot.protocol_sha256 != _slot_protocol_hash(slot, self.shared_protocol_sha256):
                raise ValueError("slot protocol hash mismatch")
            if slot.materialization_binding_sha256 != _materialization_hash(slot):
                raise ValueError("slot materialization binding mismatch")
        groups: dict[str, list[PlannedRunSlot]] = {}
        for slot in active:
            groups.setdefault(slot.comparison_id, []).append(slot)
        for comparison_id, pair in groups.items():
            if len(pair) != 2:
                raise ValueError(f"{comparison_id} must contain one treatment and one control")
            if {slot.comparison_position for slot in pair} != {"treatment", "control"}:
                raise ValueError(f"{comparison_id} lacks treatment/control positions")
            if len({slot.comparison_sha256 for slot in pair}) != 1:
                raise ValueError(f"{comparison_id} differs beyond the declared treatment")
            treatment = next(slot for slot in pair if slot.comparison_position == "treatment")
            control = next(slot for slot in pair if slot.comparison_position == "control")
            expected_control = (
                {PortfolioArm.ORACLE_SFT, PortfolioArm.SEQUENCE_CE_CONTROL}
                if treatment.arm is PortfolioArm.SEQUENCE_KD
                else {PortfolioArm.CE_ABLATION}
            )
            if treatment.arm not in {PortfolioArm.SEQUENCE_KD, PortfolioArm.LOGIT_KD}:
                raise ValueError(f"{comparison_id} treatment is not a KD arm")
            if control.arm not in expected_control:
                raise ValueError(f"{comparison_id} uses a cross-recipe control")
        if self.phase == "screen":
            expected_active = 12 if self.surface is Surface.LOGIT else 8
            if len(active) != expected_active or len(idle) != PHYSICAL_SLOTS - expected_active:
                raise ValueError("screen wave active/not-started slot count mismatch")
        if self.surface is Surface.LOGIT:
            generalists = [slot for slot in active if slot.role == "generalist"]
            if {slot.arm for slot in generalists} != {
                PortfolioArm.ORACLE_SFT,
                PortfolioArm.SEQUENCE_KD,
                PortfolioArm.LOGIT_KD,
                PortfolioArm.CE_ABLATION,
            }:
                raise ValueError("logit screen requires four matched generalist arms")
            for task in TASKS:
                specialist = [
                    slot for slot in active if slot.role == "specialist" and slot.tasks == (task,)
                ]
                if {slot.arm for slot in specialist} != {
                    PortfolioArm.LOGIT_KD,
                    PortfolioArm.CE_ABLATION,
                }:
                    raise ValueError(f"logit specialist pair missing for {task.value}")
        if self.surface is Surface.SEQUENCE:
            if any(slot.role != "specialist" for slot in active):
                raise ValueError("sequence screen is reserved for specialist pairs")
            for task in TASKS:
                specialist = [slot for slot in active if slot.tasks == (task,)]
                if {slot.arm for slot in specialist} != {
                    PortfolioArm.SEQUENCE_KD,
                    PortfolioArm.SEQUENCE_CE_CONTROL,
                }:
                    raise ValueError(f"sequence specialist pair missing for {task.value}")
        if self.matrix_sha256 != _wave_hash(self):
            raise ValueError("wave matrix hash mismatch")
        return self

    @property
    def active_slots(self) -> tuple[PlannedRunSlot, ...]:
        return tuple(slot for slot in self.slots if isinstance(slot, PlannedRunSlot))


def _wave_shared_hash(value: Wave) -> str:
    return content_sha256(
        {
            "schema_version": "distillery.portfolio.wave_shared_protocol.v2",
            "wave_id": value.wave_id,
            "phase": value.phase,
            "surface": value.surface.value,
            "tier": value.tier.value,
            "candidate_sha256": value.candidate_sha256,
            "pair_sha256": value.pair_sha256,
            "gate_sha256": value.gate_sha256,
            "runtime_sha256": value.runtime_sha256,
            "training_protocol_sha256": value.training_protocol_sha256,
            "dataset_bundle_sha256": value.dataset_bundle_sha256,
            "instance_type": value.instance_type,
            "hardware_profile": value.hardware_profile,
            "accelerator": value.accelerator,
            "seed": value.seed,
            "selection_lock_sha256": value.selection_lock_sha256,
        }
    )


def _wave_hash(value: Wave) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"matrix_sha256"}))


def _task_slug(tasks: tuple[Task, ...]) -> str:
    if tasks == TASKS:
        return "all"
    return {
        Task.TRANSACTION_REVIEW: "txn",
        Task.VARIANCE_ANALYSIS: "var",
        Task.MERCHANT_TAGGING: "merchant",
        Task.CASH_RECONCILIATION: "cash",
    }[tasks[0]]


def _arm_slug(arm: PortfolioArm) -> str:
    return {
        PortfolioArm.ORACLE_SFT: "oracle",
        PortfolioArm.SEQUENCE_KD: "seq",
        PortfolioArm.LOGIT_KD: "logit",
        PortfolioArm.CE_ABLATION: "ce",
        PortfolioArm.SEQUENCE_CE_CONTROL: "seqce",
    }[arm]


def _make_slot(
    *,
    index: int,
    item: Candidate,
    dataset: DatasetBundle,
    artifact_root: str,
    wave_id: str,
    surface: Surface,
    seed: Literal[17, 23],
    shared_protocol_sha256: str,
    role: Role,
    arm: PortfolioArm,
    tasks: tuple[Task, ...],
    comparison_id: str,
    comparison_position: ComparisonPosition,
    rationale: str,
) -> PlannedRunSlot:
    task_slug = _task_slug(tasks)
    arm_slug = _arm_slug(arm)
    role_slug = "g" if role == "generalist" else "s"
    stem = (
        f"pf-{item.tier.value}-{surface.value}-s{seed}-{index:02d}-"
        f"{role_slug}-{task_slug}-{arm_slug}"
    )
    prefix = artifact_root.rstrip("/") + f"/{item.tier.value}/{wave_id}/slot-{index:02d}/"
    provisional = PlannedRunSlot.model_construct(
        slot=index,
        node=index % 2,
        gpu=index // 2,
        run_id=f"run_{stem}",
        model_id=f"model_{stem}",
        artifact_id=f"art_{stem}",
        role=role,
        arm=arm,
        tasks=tasks,
        recipe=arm_recipe(arm),
        seed=seed,
        dataset_view_sha256=dataset.view_for(tasks).view_sha256,
        comparison_id=comparison_id,
        comparison_position=comparison_position,
        comparison_sha256="0" * 64,
        protocol_sha256="0" * 64,
        materialization_binding_sha256="0" * 64,
        output_prefix=prefix + "output/",
        artifact_prefix=prefix + "artifacts/",
        rationale=rationale,
    )
    comparison_sha = _comparison_hash(provisional)
    with_comparison = provisional.model_copy(update={"comparison_sha256": comparison_sha})
    protocol_sha = _slot_protocol_hash(with_comparison, shared_protocol_sha256)
    with_protocol = with_comparison.model_copy(update={"protocol_sha256": protocol_sha})
    return with_protocol.model_copy(
        update={"materialization_binding_sha256": _materialization_hash(with_protocol)}
    )


def _screen_assignments(
    surface: Surface,
) -> list[
    tuple[
        Role,
        PortfolioArm,
        tuple[Task, ...],
        str,
        ComparisonPosition,
        str,
    ]
]:
    if surface is Surface.LOGIT:
        assignments = [
            (
                "generalist",
                PortfolioArm.ORACLE_SFT,
                TASKS,
                "cmp_generalist_sequence",
                "control",
                "Four-task hard-CE control for the generalist sequence-KD treatment.",
            ),
            (
                "generalist",
                PortfolioArm.SEQUENCE_KD,
                TASKS,
                "cmp_generalist_sequence",
                "treatment",
                "Four-task sequence-KD treatment with a same-recipe oracle control.",
            ),
            (
                "generalist",
                PortfolioArm.LOGIT_KD,
                TASKS,
                "cmp_generalist_logit",
                "treatment",
                "Four-task logit-KD treatment with a logit-surface CE control.",
            ),
            (
                "generalist",
                PortfolioArm.CE_ABLATION,
                TASKS,
                "cmp_generalist_logit",
                "control",
                "Four-task hard-CE control on the logit trainer surface.",
            ),
        ]
        for task in TASKS:
            slug = _task_slug((task,))
            assignments.extend(
                [
                    (
                        "specialist",
                        PortfolioArm.LOGIT_KD,
                        (task,),
                        f"cmp_{slug}_logit",
                        "treatment",
                        f"{task.value} logit-KD specialist treatment; starts planned.",
                    ),
                    (
                        "specialist",
                        PortfolioArm.CE_ABLATION,
                        (task,),
                        f"cmp_{slug}_logit",
                        "control",
                        f"{task.value} same-recipe hard-CE specialist control.",
                    ),
                ]
            )
        return assignments
    if surface is Surface.SEQUENCE:
        assignments = []
        for task in TASKS:
            slug = _task_slug((task,))
            assignments.extend(
                [
                    (
                        "specialist",
                        PortfolioArm.SEQUENCE_KD,
                        (task,),
                        f"cmp_{slug}_sequence",
                        "treatment",
                        f"{task.value} sequence-KD specialist treatment; starts planned.",
                    ),
                    (
                        "specialist",
                        PortfolioArm.SEQUENCE_CE_CONTROL,
                        (task,),
                        f"cmp_{slug}_sequence",
                        "control",
                        f"{task.value} sequence-surface hard-CE specialist control.",
                    ),
                ]
            )
        return assignments
    raise ValueError("screen assignments only support logit or sequence surfaces")


def build_screen_wave(
    *,
    item: Candidate,
    gate: ProtocolGate,
    runtime: RuntimeBinding,
    protocol: TrainingProtocol,
    dataset: DatasetBundle,
    artifact_root: str,
    surface: Surface,
) -> Wave:
    if surface not in {Surface.LOGIT, Surface.SEQUENCE}:
        raise ValueError("screen surface must be logit or sequence")
    if (
        gate.candidate_sha256,
        gate.runtime_sha256,
        gate.training_protocol_sha256,
    ) != (
        item.descriptor_sha256,
        runtime.runtime_sha256,
        protocol.protocol_sha256,
    ):
        raise ValueError("candidate/gate/runtime/protocol mismatch")
    wave_id = f"wave_pf_{item.tier.value}_{surface.value}_screen_s{SCREEN_SEED}"
    provisional = Wave.model_construct(
        wave_id=wave_id,
        phase="screen",
        surface=surface,
        tier=item.tier,
        candidate_sha256=item.descriptor_sha256,
        pair_sha256=item.pair.binding_sha256,
        gate_sha256=gate.gate_sha256,
        runtime_sha256=runtime.runtime_sha256,
        training_protocol_sha256=protocol.protocol_sha256,
        dataset_bundle_sha256=dataset.binding_sha256,
        instance_type=runtime.instance_type,
        hardware_profile=runtime.hardware_profile,
        accelerator=runtime.accelerator,
        seed=SCREEN_SEED,
        shared_protocol_sha256="0" * 64,
        slots=(),
        matrix_sha256="0" * 64,
    )
    shared_sha = _wave_shared_hash(provisional)
    active = [
        _make_slot(
            index=index,
            item=item,
            dataset=dataset,
            artifact_root=artifact_root,
            wave_id=wave_id,
            surface=surface,
            seed=SCREEN_SEED,
            shared_protocol_sha256=shared_sha,
            role=role,
            arm=arm,
            tasks=tasks,
            comparison_id=comparison_id,
            comparison_position=position,
            rationale=rationale,
        )
        for index, (role, arm, tasks, comparison_id, position, rationale) in enumerate(
            _screen_assignments(surface)
        )
    ]
    slots: list[PortfolioSlot] = list(active)
    reason = (
        "Sequence specialists are preregistered in their own same-recipe control wave."
        if surface is Surface.LOGIT
        else "Unused physical capacity remains not-started and must not remap active slots."
    )
    for index in range(len(active), PHYSICAL_SLOTS):
        slots.append(
            NotStartedSlot(
                slot=index,
                node=index % 2,  # type: ignore[arg-type]
                gpu=index // 2,
                reason=reason,
            )
        )
    with_slots = Wave.model_construct(
        **{
            **provisional.model_dump(mode="python"),
            "shared_protocol_sha256": shared_sha,
            "slots": tuple(slots),
        }
    )
    return Wave.model_validate(
        {
            **with_slots.model_dump(mode="python"),
            "matrix_sha256": _wave_hash(with_slots),
        }
    )


class FinalistPair(FrozenModel):
    treatment_model_id: ModelId
    control_model_id: ModelId


class ReplicationSelectionLock(FrozenModel):
    schema_version: Literal["distillery.portfolio.replication_lock.v1"] = (
        "distillery.portfolio.replication_lock.v1"
    )
    tier: Tier
    source_wave_sha256: tuple[Sha256Hex, ...] = Field(min_length=1, max_length=2)
    validation_split_sha256: Sha256Hex
    selection_protocol_sha256: Sha256Hex
    selected_pairs: tuple[FinalistPair, ...] = Field(min_length=1, max_length=8)
    locked_at: AwareDatetime
    test_dataset_sha256: None = None
    lock_sha256: Sha256Hex

    @model_validator(mode="after")
    def _sealed(self) -> ReplicationSelectionLock:
        model_ids = [
            model_id
            for pair in self.selected_pairs
            for model_id in (pair.treatment_model_id, pair.control_model_id)
        ]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("replication selection cannot reuse a model")
        if self.lock_sha256 != _selection_lock_hash(self):
            raise ValueError("replication selection lock hash mismatch")
        return self


def _selection_lock_hash(value: ReplicationSelectionLock) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"lock_sha256"}))


def replication_selection_lock(**values: object) -> ReplicationSelectionLock:
    provisional = ReplicationSelectionLock.model_construct(**values, lock_sha256="0" * 64)
    return ReplicationSelectionLock.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "lock_sha256": _selection_lock_hash(provisional),
        }
    )


def build_replication_wave(
    *,
    plan: PortfolioPlan,
    selection: ReplicationSelectionLock,
) -> Wave:
    tier_index = TIERS.index(selection.tier)
    item = plan.candidates[tier_index]
    runtime = plan.runtimes[tier_index]
    gate = plan.gates[tier_index]
    source_waves = [
        wave
        for wave in plan.screen_waves
        if wave.tier is selection.tier and wave.matrix_sha256 in selection.source_wave_sha256
    ]
    if {wave.matrix_sha256 for wave in source_waves} != set(selection.source_wave_sha256):
        raise ValueError("replication selection references unknown source waves")
    by_model = {slot.model_id: slot for wave in source_waves for slot in wave.active_slots}
    source_pairs: list[tuple[PlannedRunSlot, PlannedRunSlot]] = []
    for pair in selection.selected_pairs:
        treatment = by_model.get(pair.treatment_model_id)
        control = by_model.get(pair.control_model_id)
        if treatment is None or control is None:
            raise ValueError("replication selection references unknown model")
        if (
            treatment.comparison_position,
            control.comparison_position,
            treatment.comparison_id,
            treatment.comparison_sha256,
        ) != (
            "treatment",
            "control",
            control.comparison_id,
            control.comparison_sha256,
        ):
            raise ValueError("replication finalist pair is not a matched treatment/control")
        source_pairs.append((treatment, control))
    wave_id = (
        f"wave_pf_{selection.tier.value}_replication_s{REPLICATION_SEED}_"
        f"{selection.lock_sha256[:8]}"
    )
    provisional = Wave.model_construct(
        wave_id=wave_id,
        phase="replication",
        surface=Surface.REPLICATION,
        tier=item.tier,
        candidate_sha256=item.descriptor_sha256,
        pair_sha256=item.pair.binding_sha256,
        gate_sha256=gate.gate_sha256,
        runtime_sha256=runtime.runtime_sha256,
        training_protocol_sha256=plan.protocol.protocol_sha256,
        dataset_bundle_sha256=plan.dataset.binding_sha256,
        instance_type=runtime.instance_type,
        hardware_profile=runtime.hardware_profile,
        accelerator=runtime.accelerator,
        seed=REPLICATION_SEED,
        selection_lock_sha256=selection.lock_sha256,
        shared_protocol_sha256="0" * 64,
        slots=(),
        matrix_sha256="0" * 64,
    )
    shared_sha = _wave_shared_hash(provisional)
    active: list[PlannedRunSlot] = []
    for pair_index, (treatment, control) in enumerate(source_pairs):
        for source in (treatment, control):
            index = len(active)
            active.append(
                _make_slot(
                    index=index,
                    item=item,
                    dataset=plan.dataset,
                    artifact_root=plan.artifacts.root,
                    wave_id=wave_id,
                    surface=Surface.REPLICATION,
                    seed=REPLICATION_SEED,
                    shared_protocol_sha256=shared_sha,
                    role=source.role,
                    arm=source.arm,
                    tasks=source.tasks,
                    comparison_id=f"cmp_rep_{pair_index}_{source.comparison_id.removeprefix('cmp_')}",
                    comparison_position=source.comparison_position,
                    rationale=(
                        f"Seed-23 replication of {source.model_id}; selection was locked "
                        "against validation evidence before test access."
                    ),
                )
            )
    slots: list[PortfolioSlot] = list(active)
    for index in range(len(active), PHYSICAL_SLOTS):
        slots.append(
            NotStartedSlot(
                slot=index,
                node=index % 2,  # type: ignore[arg-type]
                gpu=index // 2,
                reason="No additional finalist was selected before the replication lock.",
            )
        )
    with_slots = Wave.model_construct(
        **{
            **provisional.model_dump(mode="python"),
            "shared_protocol_sha256": shared_sha,
            "slots": tuple(slots),
        }
    )
    return Wave.model_validate(
        {
            **with_slots.model_dump(mode="python"),
            "matrix_sha256": _wave_hash(with_slots),
        }
    )


class ModelDescriptor(FrozenModel):
    model_id: ModelId
    run_id: RunId
    artifact_id: ArtifactId
    tier: Tier
    candidate_name: StrictStr
    role: Role
    arm: PortfolioArm
    tasks: tuple[Task, ...]
    teacher_model_id: StrictStr
    teacher_revision: GitCommitSha
    student_model_id: StrictStr
    student_revision: GitCommitSha
    pair_sha256: Sha256Hex
    recipe: Recipe
    seed: Literal[17, 23]
    protocol_sha256: Sha256Hex
    materialization_binding_sha256: Sha256Hex
    state: Literal["planned"] = "planned"
    throughput_tokens_per_second: None = None
    cost_per_1k_tokens_microusd: None = None
    descriptor_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash(self) -> ModelDescriptor:
        if self.role == "generalist" and self.tasks != TASKS:
            raise ValueError("generalist descriptor must cover all four tasks")
        if self.role == "specialist" and len(self.tasks) != 1:
            raise ValueError("specialist descriptor must cover exactly one task")
        if self.descriptor_sha256 != _model_descriptor_hash(self):
            raise ValueError("model descriptor hash mismatch")
        return self


def _model_descriptor_hash(value: ModelDescriptor) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"descriptor_sha256"}))


def descriptors(item: Candidate, matrix: Wave) -> tuple[ModelDescriptor, ...]:
    values: list[ModelDescriptor] = []
    for slot in matrix.active_slots:
        provisional = ModelDescriptor.model_construct(
            model_id=slot.model_id,
            run_id=slot.run_id,
            artifact_id=slot.artifact_id,
            tier=item.tier,
            candidate_name=item.name,
            role=slot.role,
            arm=slot.arm,
            tasks=slot.tasks,
            teacher_model_id=item.pair.teacher.model_id,
            teacher_revision=item.pair.teacher.revision,
            student_model_id=item.pair.student.model_id,
            student_revision=item.pair.student.revision,
            pair_sha256=item.pair.binding_sha256,
            recipe=slot.recipe,
            seed=slot.seed,
            protocol_sha256=slot.protocol_sha256,
            materialization_binding_sha256=slot.materialization_binding_sha256,
            descriptor_sha256="0" * 64,
        )
        values.append(
            ModelDescriptor.model_validate(
                {
                    **provisional.model_dump(mode="python"),
                    "descriptor_sha256": _model_descriptor_hash(provisional),
                }
            )
        )
    return tuple(values)


class ArtifactReservation(FrozenModel):
    artifact_id: ArtifactId
    model_id: ModelId
    run_id: RunId
    planned_prefix: StrictStr
    materialization_binding_sha256: Sha256Hex
    state: Literal["planned"] = "planned"
    adapter_uri: None = None
    merged_uri: None = None
    checksum_sha256: None = None

    @field_validator("planned_prefix")
    @classmethod
    def _planned_prefix(cls, value: str) -> str:
        return _s3_prefix(value, field_name="planned artifact prefix")


class ArtifactPlan(FrozenModel):
    root: StrictStr
    reservations: tuple[ArtifactReservation, ...] = Field(min_length=1)

    @field_validator("root")
    @classmethod
    def _root(cls, value: str) -> str:
        return _s3_prefix(value, field_name="artifact root")

    @model_validator(mode="after")
    def _unique(self) -> ArtifactPlan:
        for field in ("artifact_id", "model_id", "run_id", "planned_prefix"):
            values = [getattr(item, field) for item in self.reservations]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate artifact reservation {field}")
        return self


class PlannedRegistryEntry(FrozenModel):
    model_id: ModelId
    descriptor_sha256: Sha256Hex
    tier: Tier
    role: Role
    arm: PortfolioArm
    tasks: tuple[Task, ...]
    seed: Literal[17, 23]
    state: Literal["planned"] = "planned"


class PlannedRegistry(FrozenModel):
    schema_version: Literal["distillery.portfolio.planned_registry.v1"] = (
        "distillery.portfolio.planned_registry.v1"
    )
    entries: tuple[PlannedRegistryEntry, ...] = Field(min_length=1)
    publishable: Literal[False] = False
    registry_sha256: Sha256Hex

    @model_validator(mode="after")
    def _planned_only(self) -> PlannedRegistry:
        ids = [entry.model_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("planned registry model IDs must be unique")
        if any(entry.role == "specialist" and entry.state != "planned" for entry in self.entries):
            raise ValueError("specialists must start planned")
        if self.registry_sha256 != _planned_registry_hash(self):
            raise ValueError("planned registry hash mismatch")
        return self


def _planned_registry_hash(value: PlannedRegistry) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"registry_sha256"}))


def planned_registry(models: tuple[ModelDescriptor, ...]) -> PlannedRegistry:
    entries = tuple(
        PlannedRegistryEntry(
            model_id=model.model_id,
            descriptor_sha256=model.descriptor_sha256,
            tier=model.tier,
            role=model.role,
            arm=model.arm,
            tasks=model.tasks,
            seed=model.seed,
        )
        for model in models
    )
    provisional = PlannedRegistry.model_construct(entries=entries, registry_sha256="0" * 64)
    return PlannedRegistry.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "registry_sha256": _planned_registry_hash(provisional),
        }
    )


class CostCeilings(FrozenModel):
    per_run_microusd: PositiveSafeInt = int(TrainingBudget().default_max_run_usd * 1_000_000)
    per_wave_microusd: PositiveSafeInt = 250 * 1_000_000
    experiment_microusd: PositiveSafeInt = 1_250 * 1_000_000
    account_microusd: Literal[10000000000] = ACCOUNT_CEILING_MICROUSD

    @model_validator(mode="after")
    def _ordered(self) -> CostCeilings:
        if not (
            self.per_run_microusd
            <= self.per_wave_microusd
            <= self.experiment_microusd
            <= self.account_microusd
        ):
            raise ValueError("cost ceilings must be ordered run <= wave <= experiment <= account")
        return self


class SlotCost(FrozenModel):
    slot: NonNegativeSafeInt = Field(le=15)
    node: Literal[0, 1]
    gpu: NonNegativeSafeInt = Field(le=7)
    run_id: RunId | None
    state: Literal["planned", "not_started", "failed", "succeeded"]
    allocated_ceiling_microusd: NonNegativeSafeInt


class WaveCost(FrozenModel):
    wave_id: StrictStr
    pricing_sha256: Sha256Hex
    parent_ceiling_microusd: PositiveSafeInt
    aggregate_ceiling_microusd: PositiveSafeInt
    slots: tuple[SlotCost, ...] = Field(min_length=16, max_length=16)

    @model_validator(mode="after")
    def _sum(self) -> WaveCost:
        if sum(slot.allocated_ceiling_microusd for slot in self.slots) != (
            self.aggregate_ceiling_microusd
        ):
            raise ValueError("wave cost allocation mismatch")
        for node in (0, 1):
            if (
                sum(slot.allocated_ceiling_microusd for slot in self.slots if slot.node == node)
                != self.parent_ceiling_microusd
            ):
                raise ValueError("each parent campaign cost must be allocated exactly once")
        return self


class ExperimentCost(FrozenModel):
    waves: tuple[WaveCost, ...] = Field(min_length=1)
    aggregate_ceiling_microusd: PositiveSafeInt
    experiment_ceiling_microusd: PositiveSafeInt
    account_ceiling_microusd: Literal[10000000000] = ACCOUNT_CEILING_MICROUSD

    @model_validator(mode="after")
    def _ceiling(self) -> ExperimentCost:
        total = sum(wave.aggregate_ceiling_microusd for wave in self.waves)
        if total != self.aggregate_ceiling_microusd:
            raise ValueError("experiment cost does not sum all waves")
        if total > self.experiment_ceiling_microusd:
            raise ValueError("portfolio experiment exceeds its sealed ceiling")
        if total > self.account_ceiling_microusd:
            raise ValueError("portfolio experiment exceeds the $10k account ceiling")
        return self


def cost(
    matrix: Wave,
    price: PricingEvidence,
    protocol: TrainingProtocol,
    ceilings: CostCeilings,
) -> WaveCost:
    if matrix.instance_type != price.instance_type:
        raise ValueError("pricing hardware mismatch")
    parent = (price.current_hourly_price_microusd * protocol.max_runtime_seconds + 3599) // 3600
    base, remainder = divmod(parent, 8)
    slots = tuple(
        SlotCost(
            slot=item.slot,
            node=item.node,
            gpu=item.gpu,
            run_id=item.run_id if isinstance(item, PlannedRunSlot) else None,
            state=item.state,
            allocated_ceiling_microusd=base + (1 if item.gpu < remainder else 0),
        )
        for item in matrix.slots
    )
    if max(slot.allocated_ceiling_microusd for slot in slots) > ceilings.per_run_microusd:
        raise ValueError("slot cost exceeds the per-run ceiling")
    aggregate = parent * 2
    if aggregate > ceilings.per_wave_microusd:
        raise ValueError("wave cost exceeds the per-wave ceiling")
    if aggregate > ceilings.account_microusd:
        raise ValueError("wave cost exceeds the $10k account ceiling")
    return WaveCost(
        wave_id=matrix.wave_id,
        pricing_sha256=price.pricing_sha256,
        parent_ceiling_microusd=parent,
        aggregate_ceiling_microusd=aggregate,
        slots=slots,
    )


class DefaultPolicy(FrozenModel):
    default_role: Literal["generalist"] = "generalist"
    specialist_state_before_promotion: Literal["planned"] = "planned"
    specialist_switch_after_promotion: Literal["explicit_user_switch_only"] = (
        "explicit_user_switch_only"
    )
    silent_task_routing_forbidden: Literal[True] = True
    silent_tier_routing_forbidden: Literal[True] = True
    larger_is_better_assumption: Literal[False] = False


class PortfolioPlan(FrozenModel):
    schema_version: Literal["distillery.portfolio.plan.v3"] = "distillery.portfolio.plan.v3"
    decision_id: Literal["decision_tinyfable_tiered_portfolio_v3"] = DECISION_ID
    supersedes: Literal["decision_tinyfable_tiered_portfolio_v2"] = (
        "decision_tinyfable_tiered_portfolio_v2"
    )
    mode: Literal["plan_only"] = "plan_only"
    created_at: AwareDatetime
    protocol: TrainingProtocol
    dataset: DatasetBundle
    candidates: tuple[Candidate, Candidate, Candidate]
    runtimes: tuple[RuntimeBinding, RuntimeBinding, RuntimeBinding]
    gates: tuple[ProtocolGate, ProtocolGate, ProtocolGate]
    pricing: tuple[PricingEvidence, PricingEvidence]
    ceilings: CostCeilings
    screen_waves: tuple[Wave, Wave, Wave, Wave, Wave, Wave]
    models: tuple[ModelDescriptor, ...] = Field(min_length=60, max_length=60)
    artifacts: ArtifactPlan
    costs: ExperimentCost
    default_policy: DefaultPolicy
    registry: PlannedRegistry
    plan_sha256: Sha256Hex

    @model_validator(mode="after")
    def _complete(self) -> PortfolioPlan:
        if [item.tier for item in self.candidates] != list(TIERS):
            raise ValueError("candidate order must be Nano, Core, Plus")
        if [item.tier for item in self.runtimes] != list(TIERS):
            raise ValueError("runtime order must be Nano, Core, Plus")
        if [item.tier for item in self.gates] != list(TIERS):
            raise ValueError("gate order must be Nano, Core, Plus")
        expected_wave_order = [
            (tier, surface) for tier in TIERS for surface in (Surface.LOGIT, Surface.SEQUENCE)
        ]
        if [(wave.tier, wave.surface) for wave in self.screen_waves] != expected_wave_order:
            raise ValueError("screen waves must be tier-ordered logit then sequence")
        if any(wave.seed != SCREEN_SEED for wave in self.screen_waves):
            raise ValueError("screen plan must contain only seed-17 waves")
        if len({model.model_id for model in self.models}) != len(self.models):
            raise ValueError("screen model IDs collide")
        if any(
            entry.role == "specialist" and entry.state != "planned"
            for entry in self.registry.entries
        ):
            raise ValueError("specialists cannot enter the plan as backups")
        for price in self.pricing:
            if not price.effective_at <= self.created_at < price.expires_at:
                raise ValueError("plan requires pricing current at plan creation")
        if self.plan_sha256 != _plan_hash(self):
            raise ValueError("portfolio plan hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _plan_hash(value: PortfolioPlan) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"plan_sha256"}))


def build_plan(
    *,
    created_at: datetime,
    nano_pair: ModelPair,
    core_pair: ModelPair,
    plus_pair: ModelPair,
    nano_runtime: RuntimeBinding,
    core_runtime: RuntimeBinding,
    plus_runtime: RuntimeBinding,
    dataset: DatasetBundle,
    g5_pricing: PricingEvidence,
    p4de_pricing: PricingEvidence,
    artifact_root: str,
    ceilings: CostCeilings | None = None,
) -> PortfolioPlan:
    root = _s3_prefix(artifact_root.rstrip("/") + "/", field_name="artifact root")
    protocol = training_protocol()
    candidates = (
        candidate(Tier.NANO, nano_pair),
        candidate(Tier.CORE, core_pair),
        candidate(Tier.PLUS, plus_pair),
    )
    runtimes = (nano_runtime, core_runtime, plus_runtime)
    gates = tuple(
        protocol_gate(item, runtime, protocol)
        for item, runtime in zip(candidates, runtimes, strict=True)
    )
    waves = tuple(
        build_screen_wave(
            item=item,
            gate=gate,
            runtime=runtime,
            protocol=protocol,
            dataset=dataset,
            artifact_root=root,
            surface=surface,
        )
        for item, runtime, gate in zip(candidates, runtimes, gates, strict=True)
        for surface in (Surface.LOGIT, Surface.SEQUENCE)
    )
    models = tuple(
        descriptor
        for item, item_waves in zip(
            candidates,
            (waves[0:2], waves[2:4], waves[4:6]),
            strict=True,
        )
        for matrix in item_waves
        for descriptor in descriptors(item, matrix)
    )
    artifacts = ArtifactPlan(
        root=root,
        reservations=tuple(
            ArtifactReservation(
                artifact_id=slot.artifact_id,
                model_id=slot.model_id,
                run_id=slot.run_id,
                planned_prefix=slot.artifact_prefix,
                materialization_binding_sha256=slot.materialization_binding_sha256,
            )
            for matrix in waves
            for slot in matrix.active_slots
        ),
    )
    sealed_ceilings = ceilings or CostCeilings()
    wave_costs = tuple(
        cost(
            matrix,
            g5_pricing if matrix.tier is Tier.NANO else p4de_pricing,
            protocol,
            sealed_ceilings,
        )
        for matrix in waves
    )
    experiment_cost = ExperimentCost(
        waves=wave_costs,
        aggregate_ceiling_microusd=sum(value.aggregate_ceiling_microusd for value in wave_costs),
        experiment_ceiling_microusd=sealed_ceilings.experiment_microusd,
    )
    provisional = PortfolioPlan.model_construct(
        created_at=created_at,
        protocol=protocol,
        dataset=dataset,
        candidates=candidates,
        runtimes=runtimes,
        gates=gates,
        pricing=(g5_pricing, p4de_pricing),
        ceilings=sealed_ceilings,
        screen_waves=waves,
        models=models,
        artifacts=artifacts,
        costs=experiment_cost,
        default_policy=DefaultPolicy(),
        registry=planned_registry(models),
        plan_sha256="0" * 64,
    )
    return PortfolioPlan.model_validate(
        {**provisional.model_dump(mode="python"), "plan_sha256": _plan_hash(provisional)}
    )
