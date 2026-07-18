"""Typed, comparator-bound portfolio selection and promotion evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictStr, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import (
    AwareDatetime,
    NonNegativeSafeInt,
    PositiveSafeInt,
    PrefixedSha256,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.ids import DatasetId, RunId
from distillery.proof.protocol_v2 import PROOF_PROTOCOL_ID_V2, finance_proof_v2_sha256
from experiments.aws_smoke.campaign_index import (
    AcceleratorType,
    HardwareInstanceType,
    HardwareProfileId,
)
from experiments.portfolio.plan import (
    BOOTSTRAP_RESAMPLES,
    REPLICATION_SEED,
    SCREEN_SEED,
    ModelDescriptor,
    ModelId,
    PlannedRunSlot,
    PortfolioArm,
    PortfolioPlan,
    Task,
    Tier,
    Wave,
)

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
EvaluationScope = Task | Literal["portfolio_primary_index"]


class MultiplicityPlan(FrozenModel):
    schema_version: Literal["distillery.portfolio.multiplicity.v1"] = (
        "distillery.portfolio.multiplicity.v1"
    )
    proof_protocol_id: Literal["finance-proof.v2"] = PROOF_PROTOCOL_ID_V2
    proof_protocol_sha256: Sha256Hex
    contrast_ids: tuple[StrictStr, ...] = Field(min_length=1)
    family_size: PositiveSafeInt
    family_alpha: Literal[0.05] = 0.05
    method: Literal["holm_bonferroni"] = "holm_bonferroni"
    hierarchy: Literal["tier_task_omnibus_then_preregistered_contrasts"] = (
        "tier_task_omnibus_then_preregistered_contrasts"
    )
    bootstrap_method: Literal["world_cluster_percentile"] = "world_cluster_percentile"
    bootstrap_resamples: Literal[10000] = BOOTSTRAP_RESAMPLES
    preregistered_at: AwareDatetime
    plan_sha256: Sha256Hex

    @model_validator(mode="after")
    def _bound(self) -> MultiplicityPlan:
        if self.proof_protocol_sha256 != finance_proof_v2_sha256():
            raise ValueError("multiplicity plan must bind finance-proof.v2")
        if len(self.contrast_ids) != len(set(self.contrast_ids)):
            raise ValueError("multiplicity contrasts must be unique")
        if self.family_size != len(self.contrast_ids):
            raise ValueError("multiplicity family_size must equal contrast count")
        if self.plan_sha256 != _multiplicity_plan_hash(self):
            raise ValueError("multiplicity plan hash mismatch")
        return self


def _multiplicity_plan_hash(value: MultiplicityPlan) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"plan_sha256"}))


def build_multiplicity_plan(
    plan: PortfolioPlan,
    *,
    preregistered_at: datetime,
) -> MultiplicityPlan:
    contrast_ids = tuple(
        f"contrast_{model.model_id}"
        for model in plan.models
        if model.role == "specialist"
        and model.arm in {PortfolioArm.SEQUENCE_KD, PortfolioArm.LOGIT_KD}
    )
    provisional = MultiplicityPlan.model_construct(
        proof_protocol_sha256=plan.protocol.proof_protocol_sha256,
        contrast_ids=contrast_ids,
        family_size=len(contrast_ids),
        preregistered_at=preregistered_at,
        plan_sha256="0" * 64,
    )
    return MultiplicityPlan.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "plan_sha256": _multiplicity_plan_hash(provisional),
        }
    )


class ProofInterval(FrozenModel):
    """One validation-only world-cluster interval for a predeclared contrast."""

    schema_version: Literal["distillery.portfolio.proof_interval.v1"] = (
        "distillery.portfolio.proof_interval.v1"
    )
    contrast_id: StrictStr = Field(min_length=1)
    metric: StrictStr = Field(min_length=1)
    treatment_model_id: ModelId
    comparator_model_id: ModelId
    treatment_arm: PortfolioArm
    comparator_arm: PortfolioArm
    treatment_run_id: RunId
    comparator_run_id: RunId
    treatment_protocol_sha256: Sha256Hex
    comparator_protocol_sha256: Sha256Hex
    training_seed: Literal[17, 23]
    task: EvaluationScope
    validation_view_id: DatasetId
    validation_view_sha256: Sha256Hex
    validation_split: Literal["validation"] = "validation"
    validation_split_sha256: Sha256Hex
    evaluation_manifest_sha256: Sha256Hex
    split_access_log_sha256: Sha256Hex
    evaluator_image_digest: PrefixedSha256
    world_clusters_sha256: Sha256Hex
    world_cluster_count: PositiveSafeInt
    bootstrap_method: Literal["world_cluster_percentile"] = "world_cluster_percentile"
    bootstrap_seed: NonNegativeSafeInt
    bootstrap_resamples: Literal[10000] = BOOTSTRAP_RESAMPLES
    confidence_level: Literal[0.95] = 0.95
    point: FiniteFloat
    lower: FiniteFloat
    upper: FiniteFloat
    test_dataset_sha256: None = None
    proof_protocol_id: Literal["finance-proof.v2"] = PROOF_PROTOCOL_ID_V2
    proof_protocol_sha256: Sha256Hex
    interval_sha256: Sha256Hex

    @model_validator(mode="after")
    def _interval(self) -> ProofInterval:
        if not self.lower <= self.point <= self.upper:
            raise ValueError("proof point estimate must lie inside its interval")
        if self.treatment_model_id == self.comparator_model_id:
            raise ValueError("proof interval requires distinct treatment and comparator")
        if self.treatment_run_id == self.comparator_run_id:
            raise ValueError("proof interval requires distinct treatment and comparator runs")
        if self.proof_protocol_sha256 != finance_proof_v2_sha256():
            raise ValueError("proof interval must bind finance-proof.v2")
        if self.interval_sha256 != _proof_interval_hash(self):
            raise ValueError("proof interval hash mismatch")
        return self


def _proof_interval_hash(value: ProofInterval) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"interval_sha256"}))


def proof_interval(**values: object) -> ProofInterval:
    provisional = ProofInterval.model_construct(**values, interval_sha256="0" * 64)
    return ProofInterval.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "interval_sha256": _proof_interval_hash(provisional),
        }
    )


class MultiplicityDecision(FrozenModel):
    schema_version: Literal["distillery.portfolio.multiplicity_decision.v1"] = (
        "distillery.portfolio.multiplicity_decision.v1"
    )
    multiplicity_plan_sha256: Sha256Hex
    contrast_id: StrictStr
    method: Literal["holm_bonferroni"] = "holm_bonferroni"
    family_size: PositiveSafeInt
    holm_rank: PositiveSafeInt
    raw_p_value: Probability
    adjusted_p_value: Probability
    hierarchy_gate_id: StrictStr = Field(min_length=1)
    hierarchy_gate_proof_sha256: Sha256Hex
    hierarchy_gate_adjusted_p_value: Probability
    decision_sha256: Sha256Hex

    @model_validator(mode="after")
    def _decision(self) -> MultiplicityDecision:
        if self.holm_rank > self.family_size:
            raise ValueError("Holm rank cannot exceed family size")
        if self.adjusted_p_value < self.raw_p_value:
            raise ValueError("adjusted p-value cannot be below raw p-value")
        if self.decision_sha256 != _multiplicity_decision_hash(self):
            raise ValueError("multiplicity decision hash mismatch")
        return self


def _multiplicity_decision_hash(value: MultiplicityDecision) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"decision_sha256"}))


def multiplicity_decision(**values: object) -> MultiplicityDecision:
    provisional = MultiplicityDecision.model_construct(**values, decision_sha256="0" * 64)
    return MultiplicityDecision.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "decision_sha256": _multiplicity_decision_hash(provisional),
        }
    )


class SpecialistPromotionEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.specialist_promotion.v1"] = (
        "distillery.portfolio.specialist_promotion.v1"
    )
    tier: Tier
    task: Task
    screen_interval: ProofInterval
    replication_interval: ProofInterval
    multiplicity: MultiplicityDecision
    screen_treatment_manifest_sha256: Sha256Hex
    screen_comparator_manifest_sha256: Sha256Hex
    replication_treatment_manifest_sha256: Sha256Hex
    replication_comparator_manifest_sha256: Sha256Hex
    screen_proof_report_sha256: Sha256Hex
    replication_proof_report_sha256: Sha256Hex
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash(self) -> SpecialistPromotionEvidence:
        if self.evidence_sha256 != _specialist_evidence_hash(self):
            raise ValueError("specialist promotion evidence hash mismatch")
        return self


def _specialist_evidence_hash(value: SpecialistPromotionEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"evidence_sha256"}))


def specialist_promotion_evidence(**values: object) -> SpecialistPromotionEvidence:
    provisional = SpecialistPromotionEvidence.model_construct(
        **values,
        evidence_sha256="0" * 64,
    )
    return SpecialistPromotionEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": _specialist_evidence_hash(provisional),
        }
    )


def _model_by_id(models: tuple[ModelDescriptor, ...], model_id: str) -> ModelDescriptor:
    matches = [model for model in models if model.model_id == model_id]
    if len(matches) != 1:
        raise ValueError(f"no unique planned model {model_id!r}")
    return matches[0]


def _slot_by_model(wave: Wave, model_id: str) -> PlannedRunSlot:
    matches = [slot for slot in wave.active_slots if slot.model_id == model_id]
    if len(matches) != 1:
        raise ValueError(f"no unique replication model {model_id!r}")
    return matches[0]


def _verify_interval_binding(
    interval: ProofInterval,
    treatment: ModelDescriptor | PlannedRunSlot,
    comparator: ModelDescriptor | PlannedRunSlot,
    *,
    task: Task,
    seed: int,
) -> None:
    expected = (
        treatment.model_id,
        comparator.model_id,
        treatment.arm,
        comparator.arm,
        treatment.run_id,
        comparator.run_id,
        treatment.protocol_sha256,
        comparator.protocol_sha256,
        seed,
        task,
    )
    actual = (
        interval.treatment_model_id,
        interval.comparator_model_id,
        interval.treatment_arm,
        interval.comparator_arm,
        interval.treatment_run_id,
        interval.comparator_run_id,
        interval.treatment_protocol_sha256,
        interval.comparator_protocol_sha256,
        interval.training_seed,
        interval.task,
    )
    if actual != expected:
        raise ValueError("proof interval comparator/run/protocol/task binding mismatch")


def specialist_eligible(
    plan: PortfolioPlan,
    replication_wave: Wave,
    multiplicity_plan: MultiplicityPlan,
    evidence: SpecialistPromotionEvidence,
) -> tuple[bool, str]:
    screen_treatment = _model_by_id(
        plan.models,
        evidence.screen_interval.treatment_model_id,
    )
    screen_comparator = _model_by_id(
        plan.models,
        evidence.screen_interval.comparator_model_id,
    )
    replication_treatment = _slot_by_model(
        replication_wave,
        evidence.replication_interval.treatment_model_id,
    )
    replication_comparator = _slot_by_model(
        replication_wave,
        evidence.replication_interval.comparator_model_id,
    )
    if screen_treatment.role != "specialist" or screen_comparator.role != "generalist":
        return False, "specialist promotion requires a same-tier generalist comparator"
    if (
        screen_treatment.tier,
        screen_comparator.tier,
        replication_wave.tier,
        evidence.tier,
    ) != (evidence.tier, evidence.tier, evidence.tier, evidence.tier):
        return False, "specialist promotion tier binding mismatch"
    if screen_treatment.tasks != (evidence.task,) or replication_treatment.tasks != (
        evidence.task,
    ):
        return False, "specialist task binding mismatch"
    if screen_comparator.tasks != tuple(Task) or replication_comparator.tasks != tuple(Task):
        return False, "generalist comparator must cover all four tasks"
    try:
        _verify_interval_binding(
            evidence.screen_interval,
            screen_treatment,
            screen_comparator,
            task=evidence.task,
            seed=SCREEN_SEED,
        )
        _verify_interval_binding(
            evidence.replication_interval,
            replication_treatment,
            replication_comparator,
            task=evidence.task,
            seed=REPLICATION_SEED,
        )
    except ValueError as exc:
        return False, str(exc)
    contrast_id = f"contrast_{screen_treatment.model_id}"
    if (
        evidence.multiplicity.multiplicity_plan_sha256 != multiplicity_plan.plan_sha256
        or evidence.multiplicity.contrast_id != contrast_id
        or contrast_id not in multiplicity_plan.contrast_ids
        or evidence.multiplicity.family_size != multiplicity_plan.family_size
    ):
        return False, "specialist contrast is not bound to the preregistered family"
    if (
        evidence.multiplicity.adjusted_p_value > multiplicity_plan.family_alpha
        or evidence.multiplicity.hierarchy_gate_adjusted_p_value > multiplicity_plan.family_alpha
    ):
        return False, "hierarchical or multiplicity-adjusted evidence did not clear"
    if evidence.screen_interval.lower < 0.02 or evidence.replication_interval.lower < 0.02:
        return False, "material task gain after uncertainty was not replicated"
    return True, "specialist is eligible for explicit backup; generalist remains default"


class BenchmarkMeasurement(FrozenModel):
    schema_version: Literal["distillery.portfolio.benchmark_measurement.v1"] = (
        "distillery.portfolio.benchmark_measurement.v1"
    )
    tier: Tier
    model_id: ModelId
    run_id: RunId
    manifest_sha256: Sha256Hex
    proof_report_sha256: Sha256Hex
    instance_type: HardwareInstanceType
    hardware_profile: HardwareProfileId
    accelerator: AcceleratorType
    runtime_image_digest: PrefixedSha256
    runtime_sha256: Sha256Hex
    harness_sha256: Sha256Hex
    token_count: PositiveSafeInt
    request_count: PositiveSafeInt
    throughput_tokens_per_second: FiniteFloat = Field(gt=0.0)
    total_cost_microusd: PositiveSafeInt
    cost_per_1k_tokens_microusd: PositiveSafeInt
    measured_at: AwareDatetime
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _measured(self) -> BenchmarkMeasurement:
        expected_cost = (self.total_cost_microusd * 1000 + self.token_count - 1) // (
            self.token_count
        )
        if self.cost_per_1k_tokens_microusd != expected_cost:
            raise ValueError("cost per 1k tokens does not match measured total and token count")
        if self.evidence_sha256 != _benchmark_hash(self):
            raise ValueError("benchmark measurement hash mismatch")
        return self


def _benchmark_hash(value: BenchmarkMeasurement) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"evidence_sha256"}))


def benchmark_measurement(**values: object) -> BenchmarkMeasurement:
    provisional = BenchmarkMeasurement.model_construct(**values, evidence_sha256="0" * 64)
    return BenchmarkMeasurement.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": _benchmark_hash(provisional),
        }
    )


class RatioInterval(FrozenModel):
    schema_version: Literal["distillery.portfolio.ratio_interval.v1"] = (
        "distillery.portfolio.ratio_interval.v1"
    )
    metric: Literal["throughput_ratio", "cost_ratio"]
    candidate_measurement_sha256: Sha256Hex
    incumbent_measurement_sha256: Sha256Hex
    bootstrap_method: Literal["paired_request_cluster_percentile"] = (
        "paired_request_cluster_percentile"
    )
    bootstrap_seed: NonNegativeSafeInt
    bootstrap_resamples: Literal[10000] = BOOTSTRAP_RESAMPLES
    point: FiniteFloat = Field(gt=0.0)
    lower: FiniteFloat = Field(gt=0.0)
    upper: FiniteFloat = Field(gt=0.0)
    interval_sha256: Sha256Hex

    @model_validator(mode="after")
    def _interval(self) -> RatioInterval:
        if not self.lower <= self.point <= self.upper:
            raise ValueError("ratio point estimate must lie inside its interval")
        if self.interval_sha256 != _ratio_interval_hash(self):
            raise ValueError("ratio interval hash mismatch")
        return self


def _ratio_interval_hash(value: RatioInterval) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"interval_sha256"}))


def ratio_interval(**values: object) -> RatioInterval:
    provisional = RatioInterval.model_construct(**values, interval_sha256="0" * 64)
    return RatioInterval.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "interval_sha256": _ratio_interval_hash(provisional),
        }
    )


class TierPromotionEvidence(FrozenModel):
    schema_version: Literal["distillery.portfolio.tier_promotion.v1"] = (
        "distillery.portfolio.tier_promotion.v1"
    )
    candidate: Tier
    incumbent: Tier
    quality_interval: ProofInterval
    candidate_measurement: BenchmarkMeasurement
    incumbent_measurement: BenchmarkMeasurement
    throughput_interval: RatioInterval
    cost_interval: RatioInterval
    evidence_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash(self) -> TierPromotionEvidence:
        if self.evidence_sha256 != _tier_evidence_hash(self):
            raise ValueError("tier promotion evidence hash mismatch")
        return self


def _tier_evidence_hash(value: TierPromotionEvidence) -> str:
    return content_sha256(value.model_dump(mode="json", exclude={"evidence_sha256"}))


def tier_promotion_evidence(**values: object) -> TierPromotionEvidence:
    provisional = TierPromotionEvidence.model_construct(**values, evidence_sha256="0" * 64)
    return TierPromotionEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "evidence_sha256": _tier_evidence_hash(provisional),
        }
    )


def tier_eligible(
    plan: PortfolioPlan,
    evidence: TierPromotionEvidence,
) -> tuple[bool, str]:
    candidate_model = _model_by_id(
        plan.models,
        evidence.candidate_measurement.model_id,
    )
    incumbent_model = _model_by_id(
        plan.models,
        evidence.incumbent_measurement.model_id,
    )
    if (
        candidate_model.role,
        incumbent_model.role,
        candidate_model.tier,
        incumbent_model.tier,
    ) != ("generalist", "generalist", evidence.candidate, evidence.incumbent):
        return False, "tier promotion must compare bound generalist models"
    quality = evidence.quality_interval
    if (
        quality.task != "portfolio_primary_index"
        or quality.treatment_model_id != candidate_model.model_id
        or quality.comparator_model_id != incumbent_model.model_id
        or quality.treatment_run_id != candidate_model.run_id
        or quality.comparator_run_id != incumbent_model.run_id
        or quality.treatment_protocol_sha256 != candidate_model.protocol_sha256
        or quality.comparator_protocol_sha256 != incumbent_model.protocol_sha256
    ):
        return False, "tier quality interval is not bound to candidate/incumbent protocols"
    candidate_measurement = evidence.candidate_measurement
    incumbent_measurement = evidence.incumbent_measurement
    confound_fields = (
        "instance_type",
        "hardware_profile",
        "accelerator",
        "runtime_image_digest",
        "runtime_sha256",
        "harness_sha256",
        "token_count",
        "request_count",
    )
    if any(
        getattr(candidate_measurement, field) != getattr(incumbent_measurement, field)
        for field in confound_fields
    ):
        return False, "cross-tier benchmark has a hardware/image/runtime/harness confound"
    expected_ratio_bindings = (
        candidate_measurement.evidence_sha256,
        incumbent_measurement.evidence_sha256,
    )
    for interval, metric in (
        (evidence.throughput_interval, "throughput_ratio"),
        (evidence.cost_interval, "cost_ratio"),
    ):
        if (
            interval.metric,
            interval.candidate_measurement_sha256,
            interval.incumbent_measurement_sha256,
        ) != (metric, *expected_ratio_bindings):
            return False, f"{metric} interval is not bound to measured evidence"
    quality_led = (
        quality.lower >= 0.02
        and evidence.throughput_interval.lower >= 0.80
        and evidence.cost_interval.upper <= 1.25
    )
    efficiency_led = (
        quality.lower >= -0.005
        and evidence.throughput_interval.lower >= 1.10
        and evidence.cost_interval.upper <= 0.90
    )
    if quality_led or efficiency_led:
        return True, "pre-registered measured quality/efficiency tradeoff clears"
    return False, "candidate clears neither measured tradeoff path"
