"""Immutable ProofReport resource and fixed proof statuses."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictStr,
    model_validator,
)

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import AwareDatetime, Sha256Hex, content_sha256
from distillery.contracts.ids import ProofReportId, RunId

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PROOF_GATE_ORDER: tuple[str, ...] = (
    "pilot_teacher",
    "baseline",
    "trainer",
    "quality",
    "economics",
    "evidence",
)


class ProofStatus(StrEnum):
    PROVED = "proved"
    DO_NOT_DISTILL = "do_not_distill"
    FAILED_QUALITY = "failed_quality"
    FAILED_ECONOMICS = "failed_economics"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ArmResult(FrozenModel):
    arm_id: StrictStr = Field(min_length=1)
    primary_index: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    metrics: FrozenJsonObject = Field(default_factory=dict)
    prediction_sha256: Sha256Hex | None = None
    excluded: StrictBool = False
    exclusion_reason: StrictStr | None = None

    @model_validator(mode="after")
    def _exclusion_reason_consistency(self) -> ArmResult:
        if self.excluded and not self.exclusion_reason:
            raise ValueError("excluded arms require exclusion_reason")
        if not self.excluded and self.exclusion_reason is not None:
            raise ValueError("non-excluded arms cannot carry exclusion_reason")
        return self


class QualityGateResult(FrozenModel):
    gate_id: StrictStr = Field(min_length=1)
    passed: StrictBool | None
    evaluated: StrictBool
    detail: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def _evaluation_consistency(self) -> QualityGateResult:
        if self.evaluated and self.passed is None:
            raise ValueError("evaluated gates require passed=true or false")
        if not self.evaluated and self.passed is not None:
            raise ValueError("unevaluated gates require passed=null")
        return self


class ProofReport(FrozenModel):
    """Immutable scientific/economic proof report."""

    schema_version: Literal["distillery.proof_report.v1"] = (
        "distillery.proof_report.v1"
    )
    report_id: ProofReportId
    run_ids: tuple[RunId, ...] = Field(min_length=1)
    protocol_id: StrictStr = Field(min_length=1)
    protocol_sha256: Sha256Hex
    proof_status: ProofStatus
    first_failed_gate: StrictStr | None = None
    unevaluated_gates: tuple[StrictStr, ...] = ()
    arm_results: tuple[ArmResult, ...]
    quality_gates: tuple[QualityGateResult, ...]
    uncertainty: FrozenJsonObject = Field(default_factory=dict)
    economics: FrozenJsonObject = Field(default_factory=dict)
    exclusions: tuple[StrictStr, ...] = ()
    limitations: tuple[StrictStr, ...] = ()
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _proof_status_matches_ordered_gates(self) -> ProofReport:
        gate_ids = tuple(gate.gate_id for gate in self.quality_gates)
        if gate_ids != PROOF_GATE_ORDER:
            raise ValueError(
                f"quality_gates must use required order {PROOF_GATE_ORDER!r}"
            )

        failed = tuple(
            gate.gate_id
            for gate in self.quality_gates
            if gate.evaluated and gate.passed is False
        )
        if len(failed) > 1:
            raise ValueError("proof protocol stops at the first failed gate")
        expected_first_failed = failed[0] if failed else None
        if self.first_failed_gate != expected_first_failed:
            raise ValueError(
                "first_failed_gate must equal the first failed ordered gate"
            )

        expected_unevaluated = tuple(
            gate.gate_id for gate in self.quality_gates if not gate.evaluated
        )
        if self.unevaluated_gates != expected_unevaluated:
            raise ValueError(
                "unevaluated_gates must exactly match ordered unevaluated results"
            )
        if expected_first_failed is not None:
            failed_index = PROOF_GATE_ORDER.index(expected_first_failed)
            for gate in self.quality_gates[:failed_index]:
                if not gate.evaluated or gate.passed is not True:
                    raise ValueError("every gate before first_failed_gate must pass")
            for gate in self.quality_gates[failed_index + 1 :]:
                if gate.evaluated or gate.passed is not None:
                    raise ValueError(
                        "every gate after first_failed_gate must be unevaluated"
                    )

        if self.proof_status is ProofStatus.PROVED:
            if expected_first_failed is not None or expected_unevaluated:
                raise ValueError("proved cannot contain failed or unevaluated gates")
            if any(gate.passed is not True for gate in self.quality_gates):
                raise ValueError("proved requires every required gate to pass")
        elif self.proof_status is ProofStatus.DO_NOT_DISTILL:
            if expected_first_failed != "baseline":
                raise ValueError("do_not_distill requires baseline as first failed gate")
        elif self.proof_status is ProofStatus.FAILED_QUALITY:
            if expected_first_failed != "quality":
                raise ValueError("failed_quality requires quality as first failed gate")
        elif self.proof_status is ProofStatus.FAILED_ECONOMICS:
            if expected_first_failed != "economics":
                raise ValueError(
                    "failed_economics requires economics as first failed gate"
                )
        elif self.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE:
            if expected_first_failed is None and not expected_unevaluated:
                raise ValueError(
                    "insufficient_evidence requires a failed or unevaluated gate"
                )
        return self

    def resource_hash(self) -> str:
        payload = self.model_dump(mode="python", exclude={"created_at"})
        return content_sha256(payload)
