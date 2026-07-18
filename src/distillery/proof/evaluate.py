"""Evaluate immutable predictions/manifests into a ProofReport payload."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.contracts.budgets import ProofGates
from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.ids import ProofReportId, RunId
from distillery.contracts.proof import ArmResult, ProofReport
from distillery.proof.bootstrap import (
    paired_difference_ci,
    quality_retention_ci,
)
from distillery.proof.economics import compute_economics, quality_retention
from distillery.proof.gates import GateEvaluation, GateInputs, evaluate_gates
from distillery.proof.metrics import (
    ArmMetrics,
    PredictionRecord,
    compute_arm_metrics,
    iid_ood_primary,
)
from distillery.proof.systems import SystemsSummary, summarize_systems

REQUIRED_BASELINE_ARMS = ("rules", "teacher", "student_base", "cheap_off_the_shelf")


@dataclass
class ArmEvaluationInput:
    arm_id: str
    predictions: list[PredictionRecord]
    prediction_sha256: str | None = None
    systems_profile: dict[str, Any] | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    run_id: str | None = None
    seed: int | None = None


@dataclass
class ProofEvaluationInput:
    report_id: ProofReportId
    protocol_id: str
    protocol_sha256: str
    arms: list[ArmEvaluationInput]
    costs: dict[str, Any]
    run_ids: tuple[RunId, ...] = ()
    finalist_arm_id: str = "sequence_kd"
    base_arm_id: str = "student_base"
    teacher_arm_id: str = "teacher"
    seeds_present: tuple[int, ...] = ()
    teacher_cost_per_request: float | None = None
    student_cost_per_request_at_full_util: float | None = None
    evaluation_horizon_requests: int | None = None
    # Trainer gate evidence (must be supplied; missing => insufficient_evidence)
    trainer_numerical_ok: bool | None = None
    trainer_artifact_reload_ok: bool | None = None
    trainer_smoke_eval_ok: bool | None = None
    trainer_memory_cost_ok: bool | None = None
    frozen_hashes_present: bool | None = None
    bootstrap_seed: int = 17
    bootstrap_resamples: int | None = None
    limitations: tuple[str, ...] = ()
    created_at: datetime | None = None
    # Optional precomputed baseline TCO flags; if None, inferred conservatively.
    rules_projected_tco_below_distill: bool | None = None
    cheap_projected_tco_below_distill: bool | None = None
    quality_thresholds_for_baseline: dict[str, float] = field(default_factory=dict)


@dataclass
class ProofEvaluationResult:
    report: ProofReport
    arm_metrics: dict[str, ArmMetrics]
    systems: dict[str, SystemsSummary]
    gate_evaluation: GateEvaluation
    uncertainty: dict[str, Any]
    economics: dict[str, Any]


def load_predictions_jsonl(
    path: Path | str, *, arm_id: str | None = None
) -> list[PredictionRecord]:
    """Load prediction records from an immutable JSONL file."""
    records: list[PredictionRecord] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if arm_id is not None and "arm_id" not in payload:
            payload = {**payload, "arm_id": arm_id}
        records.append(PredictionRecord.model_validate(payload))
    return records


def file_sha256(path: Path | str) -> str:
    return sha256_hex(Path(path).read_bytes())


def _arm_meets_quality(metrics: ArmMetrics, th: ProofGates) -> bool:
    if metrics.json_schema_validity < th.json_schema_validity_min:
        return False
    if metrics.critical_invariant_violations > 0:
        return False
    # Baseline quality: require primary index at least teacher-gap floor above 0.5
    # is too arbitrary; use schema + no invariant breaks + joint components present.
    if metrics.transaction_joint_exact is None or metrics.variance_joint_exact is None:
        return False
    return (
        metrics.transaction_joint_exact >= 0.90
        and metrics.variance_joint_exact >= 0.90
        and metrics.primary_index >= 0.90
    )


def evaluate_proof(inp: ProofEvaluationInput) -> ProofEvaluationResult:
    """Blind-ish evaluation: metrics computed before status assignment."""
    th = ProofGates()
    arm_metrics: dict[str, ArmMetrics] = {}
    systems: dict[str, SystemsSummary] = {}
    arm_results: list[ArmResult] = []
    exclusions: list[str] = []

    for arm in inp.arms:
        if arm.excluded:
            exclusions.append(f"{arm.arm_id}: {arm.exclusion_reason or 'excluded'}")
            arm_results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    primary_index=None,
                    metrics={},
                    prediction_sha256=arm.prediction_sha256,
                    excluded=True,
                    exclusion_reason=arm.exclusion_reason,
                )
            )
            continue
        metrics = compute_arm_metrics(arm.arm_id, arm.predictions)
        arm_metrics[arm.arm_id] = metrics
        pred_hash = arm.prediction_sha256
        if pred_hash is None and arm.predictions:
            pred_hash = content_sha256(
                [p.model_dump(mode="json") for p in arm.predictions]
            )
        split_bits = iid_ood_primary(metrics.example_scores)
        arm_results.append(
            ArmResult(
                arm_id=arm.arm_id,
                primary_index=metrics.primary_index,
                metrics={
                    "json_parse_rate": metrics.json_parse_rate,
                    "json_schema_validity": metrics.json_schema_validity,
                    "refusal_empty_rate": metrics.refusal_empty_rate,
                    "transaction_joint_exact": metrics.transaction_joint_exact,
                    "variance_joint_exact": metrics.variance_joint_exact,
                    "cash_joint_exact": metrics.cash_joint_exact,
                    "critical_invariant_violations": metrics.critical_invariant_violations,
                    "calibration": {
                        "brier_score": metrics.calibration.brier_score,
                        "adaptive_ece": metrics.calibration.adaptive_ece,
                        "n": metrics.calibration.n,
                    },
                    "slices": [
                        {
                            "slice_key": s.slice_key,
                            "slice_value": s.slice_value,
                            "n": s.n,
                            "joint_exact": s.joint_exact,
                            "json_schema_validity": s.json_schema_validity,
                            "underpowered": s.underpowered,
                        }
                        for s in metrics.slices
                    ],
                    "iid_ood": split_bits,
                    "task_metrics": metrics.task_metrics,
                },
                prediction_sha256=pred_hash,
                excluded=False,
                exclusion_reason=None,
            )
        )
        if arm.systems_profile is not None:
            systems[arm.arm_id] = summarize_systems(arm.systems_profile)

    # Uncertainty: paired CIs for finalist vs teacher/base when present.
    uncertainty: dict[str, Any] = {"intervals": []}
    n_boot = inp.bootstrap_resamples
    finalist = arm_metrics.get(inp.finalist_arm_id)
    teacher = arm_metrics.get(inp.teacher_arm_id)
    base = arm_metrics.get(inp.base_arm_id)

    retention_point: float | None = None
    retention_lower: float | None = None
    teacher_gap: float | None = None
    teacher_gap_ci = None
    ood_retention: float | None = None
    max_task_regression: float | None = None

    if finalist and teacher:
        retention_point = quality_retention(finalist.primary_index, teacher.primary_index)
        try:
            rci = quality_retention_ci(
                finalist.example_scores,
                teacher.example_scores,
                n_resamples=n_boot,
                seed=inp.bootstrap_seed,
            )
            uncertainty["intervals"].append(rci.to_dict())
            retention_lower = rci.lower
        except ValueError as exc:
            uncertainty["errors"] = uncertainty.get("errors", []) + [str(exc)]

        teacher_gap = teacher.primary_index - (
            base.primary_index if base else finalist.primary_index
        )
        # Pilot uses teacher vs student_base when available.
        if base:
            try:
                teacher_gap = teacher.primary_index - base.primary_index
                teacher_gap_ci = paired_difference_ci(
                    teacher.example_scores,
                    base.example_scores,
                    n_resamples=n_boot,
                    seed=inp.bootstrap_seed,
                    metric="teacher_minus_student_base_primary",
                    arm_a=inp.teacher_arm_id,
                    arm_b=inp.base_arm_id,
                )
                uncertainty["intervals"].append(teacher_gap_ci.to_dict())
            except ValueError as exc:
                uncertainty["errors"] = uncertainty.get("errors", []) + [str(exc)]

        # OOD retention: finalist OOD primary / teacher OOD primary
        f_ood = iid_ood_primary(finalist.example_scores).get("ood_primary_index")
        t_ood = iid_ood_primary(teacher.example_scores).get("ood_primary_index")
        if f_ood is not None and t_ood is not None and t_ood != 0:
            ood_retention = f_ood / t_ood

        # Max primary-task joint-exact regression vs teacher.
        regs: list[float] = []
        if (
            finalist.transaction_joint_exact is not None
            and teacher.transaction_joint_exact is not None
        ):
            regs.append(teacher.transaction_joint_exact - finalist.transaction_joint_exact)
        if (
            finalist.variance_joint_exact is not None
            and teacher.variance_joint_exact is not None
        ):
            regs.append(teacher.variance_joint_exact - finalist.variance_joint_exact)
        max_task_regression = max(regs) if regs else None

    if finalist and base:
        try:
            dci = paired_difference_ci(
                finalist.example_scores,
                base.example_scores,
                n_resamples=n_boot,
                seed=inp.bootstrap_seed,
                metric="finalist_minus_base_primary",
                arm_a=inp.finalist_arm_id,
                arm_b=inp.base_arm_id,
            )
            uncertainty["intervals"].append(dci.to_dict())
        except ValueError as exc:
            uncertainty["errors"] = uncertainty.get("errors", []) + [str(exc)]

    # Economics
    student_idx = finalist.primary_index if finalist else 0.0
    teacher_idx = teacher.primary_index if teacher else 0.0
    base_idx = base.primary_index if base else 0.0
    batch1 = None
    batch8 = None
    if inp.finalist_arm_id in systems:
        sys_f = systems[inp.finalist_arm_id]
        if sys_f.batch_size == 1:
            batch1 = (
                float(sys_f.requests_per_second.value)
                if sys_f.requests_per_second.value is not None
                else None
            )
        # Look for batch-8 profile on any arm systems with batch_size 8
    for sid, sys in systems.items():
        if sys.batch_size == 1 and batch1 is None:
            batch1 = (
                float(sys.requests_per_second.value)
                if sys.requests_per_second.value is not None
                else None
            )
        if sys.batch_size == 8:
            batch8 = (
                float(sys.requests_per_second.value)
                if sys.requests_per_second.value is not None
                else None
            )
        _ = sid

    eco = compute_economics(
        student_primary_index=student_idx,
        teacher_primary_index=teacher_idx,
        base_primary_index=base_idx,
        costs=inp.costs,
        teacher_cost_per_request=inp.teacher_cost_per_request,
        student_cost_per_request_at_full_util=inp.student_cost_per_request_at_full_util,
        batch1_throughput_rps=batch1,
        batch8_throughput_rps=batch8,
        evaluation_horizon_requests=inp.evaluation_horizon_requests,
        economics_utilization=th.economics_utilization,
    )

    # Baseline gate inputs
    rules_m = arm_metrics.get("rules")
    cheap_m = arm_metrics.get("cheap_off_the_shelf")
    rules_ok = _arm_meets_quality(rules_m, th) if rules_m else None
    cheap_ok = _arm_meets_quality(cheap_m, th) if cheap_m else None
    # Conservative default: if TCO flags omitted, treat as False (baseline does not win)
    # but only when the arm metrics exist; missing arms stay None → insufficient_evidence.
    rules_tco = (
        inp.rules_projected_tco_below_distill
        if rules_m is not None
        else None
    )
    cheap_tco = (
        inp.cheap_projected_tco_below_distill
        if cheap_m is not None
        else None
    )
    if rules_m is not None and rules_tco is None:
        rules_tco = False
    if cheap_m is not None and cheap_tco is None:
        cheap_tco = False

    positive_savings = None
    be_within = None
    if eco.break_even_at_25pct.savings_per_request_usd is not None:
        positive_savings = eco.break_even_at_25pct.savings_per_request_usd > 0
    if eco.break_even_at_25pct.within_horizon is not None:
        be_within = eco.break_even_at_25pct.within_horizon

    cost_complete = (
        eco.gross_experiment_cost_usd.kind == "measured"
        and eco.gross_experiment_cost_usd.amount_usd is not None
        and eco.training_cost_usd.kind != "missing"
    )
    paired_ok = bool(uncertainty["intervals"])
    throughput_ok = batch1 is not None and (
        systems.get(inp.finalist_arm_id) is not None
        and systems[inp.finalist_arm_id].timed_examples >= 200
        and systems[inp.finalist_arm_id].warmup_requests >= 20
    )
    wide_span = False
    if retention_lower is not None and retention_point is not None:
        # Wide interval spanning the gate: lower below threshold while point above.
        if (
            retention_point >= th.quality_retention_point
            and retention_lower < th.quality_retention_lower_95
        ):
            wide_span = True

    baselines_present = tuple(aid for aid in REQUIRED_BASELINE_ARMS if aid in arm_metrics)

    gate_inputs = GateInputs(
        teacher_minus_student=teacher_gap,
        teacher_minus_student_ci=teacher_gap_ci,
        rules_meets_quality=rules_ok,
        cheap_meets_quality=cheap_ok,
        rules_projected_tco_below_distill=rules_tco,
        cheap_projected_tco_below_distill=cheap_tco,
        trainer_numerical_ok=inp.trainer_numerical_ok,
        trainer_artifact_reload_ok=inp.trainer_artifact_reload_ok,
        trainer_smoke_eval_ok=inp.trainer_smoke_eval_ok,
        trainer_memory_cost_ok=inp.trainer_memory_cost_ok,
        quality_retention_point=retention_point,
        quality_retention_lower_95=retention_lower,
        max_primary_task_regression=max_task_regression,
        json_schema_validity=finalist.json_schema_validity if finalist else None,
        ood_retention=ood_retention,
        critical_invariant_violations=(
            finalist.critical_invariant_violations if finalist else None
        ),
        economics=eco,
        positive_per_request_savings=positive_savings,
        break_even_within_horizon=be_within,
        seeds_present=inp.seeds_present,
        baselines_present=baselines_present,
        cost_records_complete=cost_complete,
        paired_intervals_present=paired_ok,
        frozen_hashes_present=inp.frozen_hashes_present,
        throughput_evidence_adequate=throughput_ok,
        wide_interval_spans_gate=wide_span,
    )
    gate_eval = evaluate_gates(gate_inputs, thresholds=th)

    limitations = list(inp.limitations)
    limitations.append("serving_economics_are_projected")
    if eco.notes:
        limitations.extend(eco.notes)

    report = ProofReport(
        report_id=inp.report_id,
        run_ids=inp.run_ids,
        protocol_id=inp.protocol_id,
        protocol_sha256=inp.protocol_sha256,
        proof_status=gate_eval.proof_status,
        first_failed_gate=gate_eval.first_failed_gate,
        unevaluated_gates=gate_eval.unevaluated_gates,
        arm_results=tuple(arm_results),
        quality_gates=gate_eval.quality_gates,
        uncertainty=uncertainty,
        economics=eco.to_dict(),
        exclusions=tuple(exclusions),
        limitations=tuple(dict.fromkeys(limitations)),
        created_at=inp.created_at or datetime.now(UTC),
    )
    return ProofEvaluationResult(
        report=report,
        arm_metrics=arm_metrics,
        systems=systems,
        gate_evaluation=gate_eval,
        uncertainty=uncertainty,
        economics=eco.to_dict(),
    )
