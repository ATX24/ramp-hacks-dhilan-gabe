"""Stop/go proof gate engine with first-failed and unevaluated gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from distillery.contracts.budgets import ProofGates
from distillery.contracts.proof import ProofStatus, QualityGateResult
from distillery.proof.bootstrap import BootstrapCI
from distillery.proof.economics import BREAK_EVEN_NEVER, EconomicsSummary

GATE_ORDER: tuple[str, ...] = (
    "pilot_teacher",
    "baseline",
    "trainer",
    "quality",
    "economics",
    "evidence",
)


@dataclass
class GateInputs:
    """All evidence required to evaluate proof gates.

    Missing required fields force evidence / fail-loud outcomes. They never
    silently pass.
    """

    # Pilot teacher gap (absolute primary-index points)
    teacher_minus_student: float | None = None
    teacher_minus_student_ci: BootstrapCI | None = None

    # Baseline competitiveness
    rules_meets_quality: bool | None = None
    cheap_meets_quality: bool | None = None
    rules_projected_tco_below_distill: bool | None = None
    cheap_projected_tco_below_distill: bool | None = None

    # Trainer gate
    trainer_numerical_ok: bool | None = None
    trainer_artifact_reload_ok: bool | None = None
    trainer_smoke_eval_ok: bool | None = None
    trainer_memory_cost_ok: bool | None = None

    # Quality gate
    quality_retention_point: float | None = None
    quality_retention_lower_95: float | None = None
    max_primary_task_regression: float | None = None
    json_schema_validity: float | None = None
    ood_retention: float | None = None
    critical_invariant_violations: int | None = None

    # Economics gate
    economics: EconomicsSummary | None = None
    positive_per_request_savings: bool | None = None
    break_even_within_horizon: bool | None = None

    # Evidence gate
    seeds_present: tuple[int, ...] = ()
    baselines_present: tuple[str, ...] = ()
    cost_records_complete: bool | None = None
    paired_intervals_present: bool | None = None
    frozen_hashes_present: bool | None = None
    throughput_evidence_adequate: bool | None = None
    wide_interval_spans_gate: bool | None = None

    # Optional overrides / notes
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GateEvaluation:
    proof_status: ProofStatus
    first_failed_gate: str | None
    unevaluated_gates: tuple[str, ...]
    quality_gates: tuple[QualityGateResult, ...]
    evidence_needed: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_status": self.proof_status.value,
            "first_failed_gate": self.first_failed_gate,
            "unevaluated_gates": list(self.unevaluated_gates),
            "quality_gates": [g.model_dump(mode="json") for g in self.quality_gates],
            "evidence_needed": list(self.evidence_needed),
        }


def _result(gate_id: str, passed: bool | None, evaluated: bool, detail: str) -> QualityGateResult:
    return QualityGateResult(
        gate_id=gate_id, passed=passed, evaluated=evaluated, detail=detail
    )


def evaluate_gates(
    inputs: GateInputs,
    *,
    thresholds: ProofGates | None = None,
) -> GateEvaluation:
    """Evaluate gates in order. Stop at first failure; leave the rest unevaluated."""
    th = thresholds or ProofGates()
    results: list[QualityGateResult] = []
    evidence_needed: list[str] = []
    first_failed: str | None = None
    status: ProofStatus | None = None

    def fail(gate_id: str, proof_status: ProofStatus, detail: str, needed: list[str]) -> None:
        nonlocal first_failed, status
        results.append(_result(gate_id, False, True, detail))
        first_failed = gate_id
        status = proof_status
        evidence_needed.extend(needed)

    def pass_gate(gate_id: str, detail: str) -> None:
        results.append(_result(gate_id, True, True, detail))

    # --- 1. pilot_teacher ---
    if first_failed is None:
        if inputs.teacher_minus_student is None or inputs.teacher_minus_student_ci is None:
            fail(
                "pilot_teacher",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "missing teacher-student pilot gap or paired CI",
                ["pilot_teacher_gap", "pilot_paired_ci"],
            )
        else:
            gap = inputs.teacher_minus_student
            ci = inputs.teacher_minus_student_ci
            ok = gap >= th.teacher_gap_min_abs and ci.lower > 0.0
            if ok:
                pass_gate(
                    "pilot_teacher",
                    f"teacher gap {gap:.4f} >= {th.teacher_gap_min_abs} and CI excludes 0",
                )
            else:
                fail(
                    "pilot_teacher",
                    ProofStatus.INSUFFICIENT_EVIDENCE,
                    f"teacher gap insufficient: gap={gap:.4f}, CI=[{ci.lower:.4f},{ci.upper:.4f}]",
                    ["larger_teacher_or_fallback_pair", "repeat_pilot"],
                )

    # --- 2. baseline ---
    if first_failed is None:
        missing_baseline = (
            inputs.rules_meets_quality is None
            or inputs.cheap_meets_quality is None
            or inputs.rules_projected_tco_below_distill is None
            or inputs.cheap_projected_tco_below_distill is None
        )
        if missing_baseline:
            fail(
                "baseline",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "missing rules/cheap baseline quality or TCO comparison",
                ["rules_arm", "cheap_off_the_shelf_arm", "projected_tco_comparison"],
            )
        else:
            rules_wins = (
                inputs.rules_meets_quality and inputs.rules_projected_tco_below_distill
            )
            cheap_wins = (
                inputs.cheap_meets_quality and inputs.cheap_projected_tco_below_distill
            )
            if rules_wins or cheap_wins:
                which = "rules" if rules_wins else "cheap_off_the_shelf"
                fail(
                    "baseline",
                    ProofStatus.DO_NOT_DISTILL,
                    f"{which} meets quality thresholds at lower projected TCO",
                    [],
                )
            else:
                pass_gate("baseline", "no cheaper baseline clears quality+TCO gate")

    # --- 3. trainer ---
    if first_failed is None:
        flags = (
            inputs.trainer_numerical_ok,
            inputs.trainer_artifact_reload_ok,
            inputs.trainer_smoke_eval_ok,
            inputs.trainer_memory_cost_ok,
        )
        if any(f is None for f in flags):
            fail(
                "trainer",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "missing trainer gate evidence",
                [
                    "trainer_numerical_tests",
                    "artifact_reload",
                    "deterministic_smoke_eval",
                    "memory_cost_ceilings",
                ],
            )
        elif not all(flags):
            fail(
                "trainer",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "trainer gate checks failed",
                ["fix_trainer_failures_before_full_proof"],
            )
        else:
            pass_gate("trainer", "numerical, reload, smoke, memory/cost checks passed")

    # --- 4. quality ---
    if first_failed is None:
        needed_q = []
        if inputs.quality_retention_point is None:
            needed_q.append("quality_retention_point")
        if inputs.quality_retention_lower_95 is None:
            needed_q.append("quality_retention_lower_95")
        if inputs.max_primary_task_regression is None:
            needed_q.append("max_primary_task_regression")
        if inputs.json_schema_validity is None:
            needed_q.append("json_schema_validity")
        if inputs.ood_retention is None:
            needed_q.append("ood_retention")
        if inputs.critical_invariant_violations is None:
            needed_q.append("critical_invariant_violations")
        if needed_q:
            fail(
                "quality",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "missing quality gate metrics",
                needed_q,
            )
        else:
            asserts = [
                (
                    inputs.quality_retention_point >= th.quality_retention_point,
                    f"retention_point {inputs.quality_retention_point} "
                    f"< {th.quality_retention_point}",
                ),
                (
                    inputs.quality_retention_lower_95 >= th.quality_retention_lower_95,
                    f"retention_lower_95 {inputs.quality_retention_lower_95} "
                    f"< {th.quality_retention_lower_95}",
                ),
                (
                    inputs.max_primary_task_regression <= th.max_primary_task_regression,
                    f"task regression {inputs.max_primary_task_regression} "
                    f"> {th.max_primary_task_regression}",
                ),
                (
                    inputs.json_schema_validity >= th.json_schema_validity_min,
                    f"json_schema_validity {inputs.json_schema_validity} "
                    f"< {th.json_schema_validity_min}",
                ),
                (
                    inputs.ood_retention >= th.ood_retention_min,
                    f"ood_retention {inputs.ood_retention} < {th.ood_retention_min}",
                ),
                (
                    inputs.critical_invariant_violations == 0,
                    f"critical invariant violations="
                    f"{inputs.critical_invariant_violations}",
                ),
            ]
            failed_msgs = [msg for ok, msg in asserts if not ok]
            if failed_msgs:
                fail(
                    "quality",
                    ProofStatus.FAILED_QUALITY,
                    "; ".join(failed_msgs),
                    ["improve_student_quality_or_accept_failed_quality"],
                )
            else:
                pass_gate("quality", "retention, OOD, schema, invariants within thresholds")

    # --- 5. economics ---
    if first_failed is None:
        eco = inputs.economics
        if (
            eco is None
            or inputs.positive_per_request_savings is None
            or inputs.break_even_within_horizon is None
        ):
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "missing economics summary or savings/break-even evidence",
                [
                    "gross_cost_records",
                    "per_request_savings",
                    "break_even_within_horizon",
                    "utilization_sensitivity_rows",
                ],
            )
        elif eco.gross_experiment_cost_usd.kind == "missing":
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "gross experiment cost missing; cannot pass economics",
                ["gross_experiment_cost_usd"],
            )
        elif (
            not inputs.positive_per_request_savings
            or not inputs.break_even_within_horizon
            or eco.break_even_at_25pct.break_even_requests == BREAK_EVEN_NEVER
        ):
            fail(
                "economics",
                ProofStatus.FAILED_ECONOMICS,
                "nonpositive savings or break-even outside horizon at "
                f"{th.economics_utilization:.0%} utilization",
                ["lower_serving_cost_or_accept_failed_economics"],
            )
        else:
            pass_gate(
                "economics",
                "positive savings and break-even within horizon at "
                f"{th.economics_utilization:.0%} utilization",
            )

    # --- 6. evidence ---
    if first_failed is None:
        required_seeds = {th.required_seed_screen, th.required_seed_replication}
        seeds_ok = required_seeds.issubset(set(inputs.seeds_present))
        required_baselines = {"rules", "teacher", "student_base", "cheap_off_the_shelf"}
        baselines_ok = required_baselines.issubset(set(inputs.baselines_present))
        missing: list[str] = []
        if not seeds_ok:
            missing.append(
                f"seeds_require_{th.required_seed_screen}_and_{th.required_seed_replication}"
            )
        if not baselines_ok:
            missing.append("missing_required_baseline_arms")
        if inputs.cost_records_complete is not True:
            missing.append("complete_cost_records")
        if inputs.paired_intervals_present is not True:
            missing.append("paired_bootstrap_intervals")
        if inputs.frozen_hashes_present is not True:
            missing.append("frozen_protocol_and_prediction_hashes")
        if inputs.throughput_evidence_adequate is not True:
            missing.append("adequate_throughput_evidence_for_projected_economics")
        if inputs.wide_interval_spans_gate is True:
            missing.append("narrower_intervals_not_spanning_gate")
        if inputs.cost_records_complete is None:
            # Explicit: unknown completeness never passes.
            if "complete_cost_records" not in missing:
                missing.append("complete_cost_records")
        if missing:
            fail(
                "evidence",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "evidence gate incomplete: " + ", ".join(missing),
                missing,
            )
        else:
            pass_gate(
                "evidence",
                "two seeds, baselines, costs, hashes, and paired intervals present",
            )

    # Mark remaining gates unevaluated
    evaluated_ids = {g.gate_id for g in results}
    for gate_id in GATE_ORDER:
        if gate_id not in evaluated_ids:
            results.append(
                _result(gate_id, None, False, "not evaluated (prior gate failed)")
            )

    unevaluated = tuple(g.gate_id for g in results if not g.evaluated)
    if status is None:
        # All gates passed.
        if all(g.evaluated and g.passed for g in results):
            status = ProofStatus.PROVED
        else:
            status = ProofStatus.INSUFFICIENT_EVIDENCE
            first_failed = first_failed or "evidence"

    return GateEvaluation(
        proof_status=status,
        first_failed_gate=first_failed,
        unevaluated_gates=unevaluated,
        quality_gates=tuple(results),
        evidence_needed=tuple(dict.fromkeys(evidence_needed)),
    )
