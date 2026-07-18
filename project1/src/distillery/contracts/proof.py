"""ProofReport contract. proof_status is a scientific verdict, not infra state."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ProofStatus = Literal["proved", "do_not_distill", "failed_quality", "failed_economics", "insufficient_evidence"]

PRIMARY_INDEX_WEIGHTS = {"transaction_joint_exact": 0.45, "variance_joint_exact": 0.45, "json_schema_validity": 0.10}


class ArmResult(BaseModel):
    arm: str
    predictions_sha256: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    primary_index: float | None = None
    ci_95: list[float] | None = None
    seeds: list[int] = Field(default_factory=list)


class Gates(BaseModel):
    pilot_teacher_gate: dict[str, Any] | None = None
    baseline_gate: dict[str, Any] | None = None
    trainer_gate: dict[str, Any] | None = None
    quality_gate: dict[str, Any] | None = None
    economics_gate: dict[str, Any] | None = None
    evidence_gate: dict[str, Any] | None = None
    first_failed_gate: str | None = None
    unevaluated_gates: list[str] = Field(default_factory=list)


class ProofReport(BaseModel):
    report_id: str
    protocol: dict[str, Any]
    arms: list[ArmResult]
    gates: Gates
    economics: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    proof_status: ProofStatus
    created_at: str
