"""Sealed Finance Agent proof protocol with honest not-ready/null states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import Field, StrictStr, ValidationInfo, model_validator

from distillery.contracts.base import FrozenModel
from distillery.contracts.hashing import Sha256Hex, content_sha256
from distillery.finance_agent.contracts import AgentTrajectory

FINANCE_AGENT_METRICS: tuple[str, ...] = (
    "tool_selection_accuracy",
    "action_order_accuracy",
    "argument_exactness",
    "tool_result_exactness",
    "tool_result_use",
    "final_answer_correctness",
    "unnecessary_calls",
    "skipped_calls",
    "latency_ms",
    "cost_usd_micros",
    "end_to_end_success",
)


class LicenseDisposition(FrozenModel):
    status: Literal["unknown", "approved", "rejected"] = "unknown"
    license_id: StrictStr | None = None
    license_text_sha256: Sha256Hex | None = None
    output_use_reviewed: bool | None = None
    attribution_text: StrictStr | None = None

    @model_validator(mode="after")
    def _honest_state(self) -> Self:
        evidence = (
            self.license_id,
            self.license_text_sha256,
            self.output_use_reviewed,
        )
        if self.status == "unknown" and any(value is not None for value in evidence):
            raise ValueError("unknown license disposition cannot carry partial evidence")
        if self.status == "approved" and (
            self.license_id is None
            or self.license_text_sha256 is None
            or self.output_use_reviewed is not True
        ):
            raise ValueError("approved license requires exact text hash and output-use review")
        return self


class CostDisposition(FrozenModel):
    status: Literal["unknown", "measured"] = "unknown"
    measurement_artifact_sha256: Sha256Hex | None = None
    mean_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    total_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _honest_state(self) -> Self:
        values = (
            self.measurement_artifact_sha256,
            self.mean_latency_ms,
            self.total_cost_usd,
        )
        if self.status == "unknown" and any(value is not None for value in values):
            raise ValueError("unknown cost disposition cannot carry synthetic values")
        if self.status == "measured" and any(value is None for value in values):
            raise ValueError("measured cost requires artifact, latency, and actual cost")
        return self


class FinanceAgentProofBindings(FrozenModel):
    seed: int = Field(ge=0)
    corpus_sha256: Sha256Hex
    corpus_order_sha256: Sha256Hex
    system_prompt_set_sha256: Sha256Hex
    tool_schema_set_sha256: Sha256Hex
    trajectory_render_template_sha256: Sha256Hex
    model_id: StrictStr | None = None
    model_revision: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    model_artifact_sha256: Sha256Hex | None = None
    tokenizer_sha256: Sha256Hex | None = None
    chat_template_sha256: Sha256Hex | None = None
    license: LicenseDisposition = Field(default_factory=LicenseDisposition)
    cost: CostDisposition = Field(default_factory=CostDisposition)

    def readiness_blockers(self) -> tuple[str, ...]:
        blockers = []
        for field_name in (
            "model_id",
            "model_revision",
            "model_artifact_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
        ):
            if getattr(self, field_name) is None:
                blockers.append(f"missing_{field_name}")
        if self.license.status != "approved":
            blockers.append("license_not_approved")
        if self.cost.status != "measured":
            blockers.append("cost_not_measured")
        return tuple(blockers)


class FinanceAgentProofProtocol(FrozenModel):
    schema_version: Literal["finance_agent.proof_protocol.v1"] = "finance_agent.proof_protocol.v1"
    protocol_id: Literal["finance-agent-proof.v1"] = "finance-agent-proof.v1"
    metrics: tuple[StrictStr, ...] = FINANCE_AGENT_METRICS
    paired_evaluation: Literal[True] = True
    pairing_keys: tuple[StrictStr, ...] = (
        "example_id",
        "seed",
        "model_input_sha256",
    )
    label_source: Literal["oracle"] = "oracle"
    bindings: FinanceAgentProofBindings
    proof_status: Literal["not_ready", "ready"]
    readiness_blockers: tuple[StrictStr, ...]
    protocol_sha256: Sha256Hex

    @model_validator(mode="after")
    def _invariants(self, info: ValidationInfo) -> Self:
        blockers = self.bindings.readiness_blockers()
        expected_status = "ready" if not blockers else "not_ready"
        if self.proof_status != expected_status:
            raise ValueError("proof_status does not match binding readiness")
        if self.readiness_blockers != blockers:
            raise ValueError("readiness_blockers do not match binding evidence")
        if not info.context or not info.context.get("skip_hash_validation", False):
            if self.protocol_sha256 != content_sha256(self.canonical_payload()):
                raise ValueError("protocol_sha256 mismatch")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"protocol_sha256"})

    @classmethod
    def seal(cls, *, bindings: FinanceAgentProofBindings) -> FinanceAgentProofProtocol:
        blockers = bindings.readiness_blockers()
        provisional = cls.model_validate(
            {
                "bindings": bindings,
                "proof_status": "ready" if not blockers else "not_ready",
                "readiness_blockers": blockers,
                "protocol_sha256": "0" * 64,
            },
            context={"skip_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "protocol_sha256": content_sha256(payload)})


class PairedPrediction(FrozenModel):
    arm_id: StrictStr = Field(min_length=1)
    example_id: StrictStr = Field(min_length=1)
    seed: int = Field(ge=0)
    model_input_sha256: Sha256Hex
    trajectory: AgentTrajectory


def validate_paired_evaluation(
    predictions_by_arm: Mapping[str, Sequence[PairedPrediction]],
) -> tuple[tuple[str, int, str], ...]:
    """Require identical ordered example/seed/input keys for every proof arm."""
    if len(predictions_by_arm) < 2:
        raise ValueError("paired evaluation requires at least two arms")
    expected: tuple[tuple[str, int, str], ...] | None = None
    for arm_id in sorted(predictions_by_arm):
        rows = predictions_by_arm[arm_id]
        keys = tuple((row.example_id, row.seed, row.model_input_sha256) for row in rows)
        if len(keys) != len(set(keys)):
            raise ValueError(f"arm {arm_id} contains duplicate pairing keys")
        if any(row.arm_id != arm_id for row in rows):
            raise ValueError(f"arm {arm_id} contains rows labeled for another arm")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError(f"arm {arm_id} does not match paired order/keys")
    assert expected is not None
    return expected


__all__ = [
    "FINANCE_AGENT_METRICS",
    "CostDisposition",
    "FinanceAgentProofBindings",
    "FinanceAgentProofProtocol",
    "LicenseDisposition",
    "PairedPrediction",
    "validate_paired_evaluation",
]
