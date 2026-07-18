"""Strict typed request/response schemas compatible with the web Demo contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

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
