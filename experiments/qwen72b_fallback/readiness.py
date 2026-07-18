"""Integrity / image / IAM / conflict gates before any real AWS spend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from experiments.qwen72b_fallback.cost import (
    FULL_RUN_HARD_CAP_USD,
    REHEARSAL_HARD_CAP_USD,
    TRANSFER_HARD_CAP_USD,
    exact_gross_cost_usd,
)
from experiments.qwen72b_fallback.deadline import (
    FULL_MAX_RUNTIME_SECONDS,
    REHEARSAL_MAX_RUNTIME_SECONDS,
)
from experiments.qwen72b_fallback.pins import (
    DISTILLERY_BUCKET,
    MODEL_ID,
    REVISION,
    SNAPSHOT_S3_URI,
    sealed_identity,
)

GateStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    status: GateStatus
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    action: Literal["materialize", "rehearsal", "full"]
    gates: tuple[GateResult, ...]
    may_execute: bool
    blocking_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "distillery.qwen72b_fallback.readiness.v1",
            "action": self.action,
            "may_execute": self.may_execute,
            "blocking_reasons": list(self.blocking_reasons),
            "gates": [
                {"name": g.name, "status": g.status, "detail": g.detail} for g in self.gates
            ],
            "model_id": MODEL_ID,
            "revision": REVISION,
            "snapshot_s3_uri": SNAPSHOT_S3_URI,
            "bucket": DISTILLERY_BUCKET,
        }


def evaluate_readiness(
    *,
    action: Literal["materialize", "rehearsal", "full"],
    identity_ok: bool,
    inventory_ok: bool,
    license_ok: bool,
    tokenizer_family_ok: bool,
    iam_transfer_role_ok: bool,
    iam_training_role_ok: bool,
    ecr_image_digest_present: bool,
    snapshot_complete_on_s3: bool,
    conflicting_p4de_job_active: bool,
    conflicting_transfer_active: bool,
    active_g5_smoke: bool,
    active_14b_work: bool,
    materialization_projected_usd: float,
    rehearsal_projected_usd: float | None = None,
    full_projected_usd: float | None = None,
) -> ReadinessReport:
    # Touch sealed identity so pin drift fails loud even in gate assembly.
    sealed_identity()

    gates: list[GateResult] = [
        GateResult(
            "identity_pins",
            "pass" if identity_ok else "fail",
            "pinned revision/hashes verified" if identity_ok else "identity pin failed",
        ),
        GateResult(
            "weight_inventory",
            "pass" if inventory_ok else "fail",
            "37-shard inventory sealed" if inventory_ok else "inventory missing/mismatch",
        ),
        GateResult(
            "license_output_use",
            "pass" if license_ok else "fail",
            "Qwen license disposition explicit" if license_ok else "license disposition blocked",
        ),
        GateResult(
            "qwen_family_tokenizer",
            "pass" if tokenizer_family_ok else "fail",
            "tokenizer/chat-template matches Qwen2.5 family"
            if tokenizer_family_ok
            else "tokenizer family mismatch",
        ),
        GateResult(
            "no_conflicting_p4de",
            "pass" if not conflicting_p4de_job_active else "fail",
            "no InProgress ml.p4de.24xlarge job"
            if not conflicting_p4de_job_active
            else "conflicting p4de training job active",
        ),
        GateResult(
            "no_interfere_g5_or_14b",
            "pass" if not (active_g5_smoke or active_14b_work) else "fail",
            "no active g5 smoke / 14B transfer"
            if not (active_g5_smoke or active_14b_work)
            else "active g5 smoke or 14B work; refuse to interfere",
        ),
    ]

    if action == "materialize":
        gates.extend(
            [
                GateResult(
                    "iam_transfer_role",
                    "pass" if iam_transfer_role_ok else "fail",
                    "transfer role present" if iam_transfer_role_ok else "transfer IAM missing",
                ),
                GateResult(
                    "no_conflicting_transfer",
                    "pass" if not conflicting_transfer_active else "fail",
                    "no other ephemeral transfer instance"
                    if not conflicting_transfer_active
                    else "ephemeral transfer already running",
                ),
                GateResult(
                    "materialization_cost_cap",
                    "pass" if materialization_projected_usd <= TRANSFER_HARD_CAP_USD else "fail",
                    (
                        f"projected ${materialization_projected_usd:.2f} "
                        f"<= ${TRANSFER_HARD_CAP_USD:.2f}"
                    ),
                ),
            ]
        )
    else:
        gates.extend(
            [
                GateResult(
                    "iam_training_role",
                    "pass" if iam_training_role_ok else "fail",
                    "training role present" if iam_training_role_ok else "training IAM missing",
                ),
                GateResult(
                    "ecr_training_image",
                    "pass" if ecr_image_digest_present else "fail",
                    "digest-pinned training image present"
                    if ecr_image_digest_present
                    else "ECR distillery-training has no digest-pinned image",
                ),
                GateResult(
                    "s3_snapshot_complete",
                    "pass" if snapshot_complete_on_s3 else "fail",
                    "72B snapshot complete under models/"
                    if snapshot_complete_on_s3
                    else "72B snapshot not materialized on S3",
                ),
            ]
        )
        if action == "rehearsal":
            projected = rehearsal_projected_usd
            if projected is None:
                projected = exact_gross_cost_usd(
                    hourly_usd=31.5641,
                    max_runtime_seconds=REHEARSAL_MAX_RUNTIME_SECONDS,
                )
            gates.append(
                GateResult(
                    "rehearsal_cost_cap",
                    "pass" if projected <= REHEARSAL_HARD_CAP_USD else "fail",
                    f"projected ${projected:.2f} <= ${REHEARSAL_HARD_CAP_USD:.2f}",
                )
            )
        else:
            projected = full_projected_usd
            if projected is None:
                projected = exact_gross_cost_usd(
                    hourly_usd=31.5641,
                    max_runtime_seconds=FULL_MAX_RUNTIME_SECONDS,
                )
            gates.append(
                GateResult(
                    "full_run_cost_cap",
                    "pass" if projected <= FULL_RUN_HARD_CAP_USD else "fail",
                    f"projected ${projected:.2f} <= ${FULL_RUN_HARD_CAP_USD:.2f}",
                )
            )

    blocking = tuple(g.detail for g in gates if g.status == "fail")
    return ReadinessReport(
        action=action,
        gates=tuple(gates),
        may_execute=not blocking,
        blocking_reasons=blocking,
    )
