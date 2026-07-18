"""Strict typed request/response schemas compatible with the web Demo contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

FinanceTaskId = Literal[
    "transaction_review",
    "variance_analysis",
    "cash_reconciliation",
]

DemoModelArmId = Literal[
    "student_base",
    "oracle_sft",
    "sequence_kd",
    "logit_kd",
    "ce_ablation",
    "promoted_winner",
]

ArtifactKind = Literal["base", "peft_adapter", "merged"]

Provenance = Literal["live", "none"]

ValidationState = Literal["valid", "invalid", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InferRequest(StrictModel):
    """POST body for SageMaker /invocations and later /v1/demo/infer."""

    model_id: StrictStr = Field(min_length=1, max_length=256)
    artifact_id: StrictStr = Field(min_length=1, max_length=256)
    task: FinanceTaskId
    example_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    input: dict[str, Any]


class InferOkResponse(StrictModel):
    status: Literal["ok"] = "ok"
    provenance: Literal["live"] = "live"
    model_id: StrictStr
    artifact_id: StrictStr
    task: FinanceTaskId
    example_id: StrictStr | None
    structured_output: dict[str, Any]
    raw_output: StrictStr | None = None
    validation: ValidationState
    validation_detail: StrictStr | None = None
    latency_ms: StrictFloat | StrictInt
    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None


class InferErrorResponse(StrictModel):
    status: Literal["error"] = "error"
    provenance: Literal["none"] = "none"
    model_id: StrictStr | None = None
    artifact_id: StrictStr | None = None
    task: FinanceTaskId | None = None
    example_id: StrictStr | None = None
    code: StrictStr
    message: StrictStr
    retryable: StrictBool = False
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(StrictModel):
    serving_ready: StrictBool
    endpoint_id: StrictStr | None
    available_model_ids: list[StrictStr]
    loaded_model_id: StrictStr | None = None
    offline_enforced: StrictBool = True


class TeacherStudentRef(StrictModel):
    id: StrictStr
    revision: StrictStr


class ArtifactSourceProvenance(StrictModel):
    training_job_name: StrictStr | None = None
    source_uri: StrictStr = Field(min_length=1)
    manifest_sha256: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    manifest_file_sha256: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    model_tar_sha256: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    output_tar_sha256: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    weights_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    base_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    training_image_digest: StrictStr | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    validation_sha256: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ModelStats(StrictModel):
    """Evidence-backed stats. Absent evidence stays null / unknown."""

    advertised_parameter_count: StrictInt | None = None
    adapter_parameter_count: StrictInt | None = None
    compression_ratio: StrictFloat | None = None
    recipe: StrictStr | None = None
    teacher: TeacherStudentRef | None = None
    student: TeacherStudentRef | None = None
    seed: StrictInt | None = None
    data_hash: StrictStr | None = None
    manifest_hash: StrictStr | None = None
    artifact_hash: StrictStr | None = None
    training_duration_seconds: StrictFloat | None = None
    training_cost_usd: StrictFloat | None = None
    iid_primary_index: StrictFloat | None = None
    iid_ci_low: StrictFloat | None = None
    iid_ci_high: StrictFloat | None = None
    ood_retention: StrictFloat | None = None
    ood_ci_low: StrictFloat | None = None
    ood_ci_high: StrictFloat | None = None
    evaluation_scope: StrictStr | None = None
    validation_examples: StrictInt | None = None
    validation_primary_index: StrictFloat | None = None
    base_validation_primary_index: StrictFloat | None = None
    primary_index_delta: StrictFloat | None = None
    mean_latency_ms: StrictFloat | None = None
    p50_latency_ms: StrictFloat | None = None
    p95_latency_ms: StrictFloat | None = None
    proof_status: StrictStr | None = None
    promotion_status: Literal["promoted", "not_promoted", "unknown"] = "unknown"


class ServingInfo(StrictModel):
    availability: Literal["live", "unavailable"]
    endpoint_id: StrictStr | None
    artifact_id: StrictStr | None
    reason: StrictStr | None = None


class ModelEntry(StrictModel):
    model_id: StrictStr
    arm_id: DemoModelArmId
    artifact_id: StrictStr
    display_name: StrictStr
    purpose: StrictStr
    kind: ArtifactKind
    excluded: StrictBool = False
    exclusion_reason: StrictStr | None = None
    supported_tasks: list[FinanceTaskId]
    serving: ServingInfo
    stats: ModelStats
    provenance: ArtifactSourceProvenance | None = None


class ModelRegistryResponse(StrictModel):
    schema_version: Literal["distillery.demo_model_registry.v1"] = (
        "distillery.demo_model_registry.v1"
    )
    run_id: StrictStr
    dataset_id: StrictStr | None = None
    endpoint_id: StrictStr
    models: list[ModelEntry]


TASK_SCHEMA_VERSIONS: dict[FinanceTaskId, str] = {
    "transaction_review": "transaction_review.v1",
    "variance_analysis": "variance_analysis.v1",
    "cash_reconciliation": "cash_reconciliation.v1",
}

TASK_REQUIRED_FIELDS: dict[FinanceTaskId, tuple[str, ...]] = {
    "transaction_review": (
        "task",
        "schema_version",
        "gl_account",
        "journal_entry",
        "policy_action",
        "rule_ids",
        "evidence",
        "confidence",
    ),
    "variance_analysis": (
        "task",
        "schema_version",
        "profit_impact_minor",
        "direction",
        "top_drivers",
        "other_impact_minor",
        "rule_ids",
        "evidence_ids",
        "confidence",
    ),
    "cash_reconciliation": (
        "task",
        "schema_version",
        "status",
        "matched_groups",
        "exceptions",
        "adjusted_book_balance_minor",
        "adjusted_bank_balance_minor",
        "difference_minor",
        "confidence",
    ),
}


class JournalLine(StrictModel):
    account: StrictStr = Field(min_length=1)
    amount_minor: StrictInt = Field(gt=0)
    side: Literal["debit", "credit"]


class EvidenceRef(StrictModel):
    field: StrictStr = Field(min_length=1)
    source_id: StrictStr = Field(min_length=1)
    value: StrictStr = Field(min_length=1)


class TransactionReviewOutput(StrictModel):
    task: Literal["transaction_review"]
    schema_version: Literal["transaction_review.v1"]
    gl_account: StrictStr = Field(min_length=1)
    journal_entry: list[JournalLine] = Field(min_length=2)
    policy_action: Literal["approve", "review", "reject"]
    rule_ids: list[StrictStr]
    evidence: list[EvidenceRef]
    confidence: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _balanced_journal(self) -> TransactionReviewOutput:
        debits = sum(line.amount_minor for line in self.journal_entry if line.side == "debit")
        credits = sum(line.amount_minor for line in self.journal_entry if line.side == "credit")
        if debits != credits:
            raise ValueError("journal_entry debits and credits must balance")
        return self


class VarianceDriver(StrictModel):
    driver_id: StrictStr = Field(min_length=1)
    impact_minor: StrictInt
    rank: StrictInt = Field(gt=0)


class VarianceAnalysisOutput(StrictModel):
    task: Literal["variance_analysis"]
    schema_version: Literal["variance_analysis.v1"]
    profit_impact_minor: StrictInt
    direction: Literal["favorable", "unfavorable"]
    top_drivers: list[VarianceDriver] = Field(min_length=1)
    other_impact_minor: StrictInt
    rule_ids: list[StrictStr]
    evidence_ids: list[StrictStr]
    confidence: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _closed_variance(self) -> VarianceAnalysisOutput:
        ranks = [driver.rank for driver in self.top_drivers]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("top_drivers ranks must be contiguous from 1")
        identifiers = [driver.driver_id for driver in self.top_drivers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("top_drivers driver_id values must be unique")
        total = sum(driver.impact_minor for driver in self.top_drivers)
        if total + self.other_impact_minor != self.profit_impact_minor:
            raise ValueError("driver impacts plus other impact must equal profit impact")
        expected_direction = "favorable" if self.profit_impact_minor > 0 else "unfavorable"
        if self.profit_impact_minor == 0 or self.direction != expected_direction:
            raise ValueError("direction must match nonzero profit_impact_minor")
        return self


class MatchedGroup(StrictModel):
    book_ids: list[StrictStr] = Field(min_length=1)
    bank_ids: list[StrictStr] = Field(min_length=1)


class ReconciliationException(StrictModel):
    type: Literal[
        "bank_fee",
        "deposit_in_transit",
        "stale_check",
        "duplicate",
        "unexplained",
        "partial_settlement",
    ]
    event_ids: list[StrictStr] = Field(min_length=1)
    amount_minor: StrictInt


class CashReconciliationOutput(StrictModel):
    task: Literal["cash_reconciliation"]
    schema_version: Literal["cash_reconciliation.v1"]
    status: Literal["balanced", "exceptions"]
    matched_groups: list[MatchedGroup]
    exceptions: list[ReconciliationException]
    adjusted_book_balance_minor: StrictInt
    adjusted_bank_balance_minor: StrictInt
    difference_minor: StrictInt
    confidence: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _closed_reconciliation(self) -> CashReconciliationOutput:
        difference = self.adjusted_book_balance_minor - self.adjusted_bank_balance_minor
        if self.difference_minor != difference:
            raise ValueError("difference_minor must equal adjusted book minus bank")
        if self.status == "balanced" and (self.exceptions or self.difference_minor != 0):
            raise ValueError("balanced status requires no exceptions and zero difference")
        if self.status == "exceptions" and not self.exceptions:
            raise ValueError("exceptions status requires at least one exception")
        return self


TASK_OUTPUT_TYPES = {
    "transaction_review": TransactionReviewOutput,
    "variance_analysis": VarianceAnalysisOutput,
    "cash_reconciliation": CashReconciliationOutput,
}
