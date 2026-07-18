"""Stop/go proof gate engine with first-failed and unevaluated gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from distillery.contracts.budgets import ProofGates
from distillery.contracts.proof import ProofStatus, QualityGateResult
from distillery.proof.bootstrap import PRIMARY_METRICS, BootstrapCI
from distillery.proof.economics import BREAK_EVEN_NEVER, EconomicsSummary
from distillery.proof.evidence import EvidenceKind, evidence_kind
from distillery.proof.systems import SystemsSummary

GATE_ORDER: tuple[str, ...] = (
    "pilot_teacher",
    "baseline",
    "trainer",
    "quality",
    "economics",
    "evidence",
)
GATE_EPSILON = 1e-12


@dataclass(frozen=True)
class BaselineTCOComparison:
    """Explicit rules/cheap baseline TCO comparison evidence."""

    lower_tco_than_distillation: bool
    kind: EvidenceKind | str
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", evidence_kind(self.kind))
        if not self.detail.strip():
            raise ValueError("baseline TCO comparison detail is required")

    @property
    def usable(self) -> bool:
        return self.kind in (EvidenceKind.MEASURED, EvidenceKind.PROJECTED)


@dataclass(frozen=True)
class ArmQualityEvidence:
    """Threshold inputs for a baseline arm relative to the teacher."""

    quality_retention: float | None
    max_primary_task_regression: float | None
    json_schema_validity: float | None
    ood_retention: float | None
    critical_invariant_violations: int | None

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.quality_retention,
                self.max_primary_task_regression,
                self.json_schema_validity,
                self.ood_retention,
                self.critical_invariant_violations,
            )
        )

    def meets(self, thresholds: ProofGates) -> bool:
        if not self.complete:
            return False
        assert self.quality_retention is not None
        assert self.max_primary_task_regression is not None
        assert self.json_schema_validity is not None
        assert self.ood_retention is not None
        assert self.critical_invariant_violations is not None
        return (
            self.quality_retention >= thresholds.quality_retention_point
            and self.max_primary_task_regression
            <= thresholds.max_primary_task_regression + GATE_EPSILON
            and self.json_schema_validity >= thresholds.json_schema_validity_min
            and self.ood_retention >= thresholds.ood_retention_min
            and self.critical_invariant_violations == 0
        )


@dataclass(frozen=True)
class ArmAccountingEvidence:
    """Explicit failed/filtered/predicted example reconciliation."""

    arm_id: str
    expected_examples: int | None
    prediction_records: int
    failed_examples: int | None
    filtered_examples: int | None
    failed_example_reasons: dict[str, int] | None
    filtered_example_reasons: dict[str, int] | None

    @property
    def complete(self) -> bool:
        counts = (
            self.expected_examples,
            self.prediction_records,
            self.failed_examples,
            self.filtered_examples,
        )
        if any(count is None or count < 0 for count in counts):
            return False
        if (
            self.failed_example_reasons is None
            or self.filtered_example_reasons is None
            or any(count < 0 for count in self.failed_example_reasons.values())
            or any(count < 0 for count in self.filtered_example_reasons.values())
        ):
            return False
        assert self.expected_examples is not None
        assert self.failed_examples is not None
        assert self.filtered_examples is not None
        return (
            self.prediction_records
            + self.failed_examples
            + self.filtered_examples
            == self.expected_examples
            and sum(self.failed_example_reasons.values()) == self.failed_examples
            and sum(self.filtered_example_reasons.values())
            == self.filtered_examples
        )

    @property
    def proof_complete(self) -> bool:
        """Scored proof sets may not silently omit failed/filtered examples."""

        return (
            self.complete
            and self.failed_examples == 0
            and self.filtered_examples == 0
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
    rules_quality: ArmQualityEvidence | None = None
    cheap_quality: ArmQualityEvidence | None = None
    rules_tco_comparison: BaselineTCOComparison | None = None
    cheap_tco_comparison: BaselineTCOComparison | None = None

    # Trainer gate
    trainer_numerical_ok: bool | None = None
    trainer_artifact_reload_ok: bool | None = None
    trainer_smoke_eval_ok: bool | None = None
    trainer_memory_cost_ok: bool | None = None

    # Quality gate
    quality_retention_point: float | None = None
    quality_retention_lower_95: float | None = None
    quality_retention_upper_95: float | None = None
    max_primary_task_regression: float | None = None
    json_schema_validity: float | None = None
    ood_retention: float | None = None
    ood_retention_lower_95: float | None = None
    ood_retention_upper_95: float | None = None
    critical_invariant_violations: int | None = None

    # Economics gate
    economics: EconomicsSummary | None = None

    # Evidence gate
    seed_sets_by_arm: dict[str, tuple[int, ...]] = field(default_factory=dict)
    finalist_arm_id: str = "sequence_kd"
    teacher_arm_id: str = "teacher"
    matched_comparator_arm_id: str = "student_base"
    baselines_present: tuple[str, ...] = ()
    paired_intervals: tuple[BootstrapCI, ...] = ()
    frozen_hashes_present: bool | None = None
    finalist_systems: tuple[SystemsSummary, ...] = ()
    arm_accounting: tuple[ArmAccountingEvidence, ...] = ()
    raw_text_provenance_complete: bool | None = None

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
    wide_retention_interval = False
    wide_ood_interval = False
    systems_by_batch = {
        summary.batch_size: summary for summary in inputs.finalist_systems
    }
    systems_ok = (
        len(inputs.finalist_systems) == 2
        and set(systems_by_batch) == {1, 8}
        and all(summary.proof_ready for summary in systems_by_batch.values())
        and systems_by_batch[1].hardware == systems_by_batch[8].hardware
        and systems_by_batch[1].runtime == systems_by_batch[8].runtime
    )

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
            ok = (
                ci.defined
                and gap >= th.teacher_gap_min_abs
                and ci.lower is not None
                and ci.lower > 0.0
            )
            if ok:
                pass_gate(
                    "pilot_teacher",
                    f"teacher gap {gap:.4f} >= {th.teacher_gap_min_abs} and CI excludes 0",
                )
            else:
                fail(
                    "pilot_teacher",
                    ProofStatus.INSUFFICIENT_EVIDENCE,
                    f"teacher gap insufficient: gap={gap:.4f}, "
                    f"CI=[{ci.lower},{ci.upper}]",
                    ["larger_teacher_or_fallback_pair", "repeat_pilot"],
                )

    # --- 2. baseline ---
    if first_failed is None:
        missing_baseline = (
            inputs.rules_quality is None
            or not inputs.rules_quality.complete
            or inputs.cheap_quality is None
            or not inputs.cheap_quality.complete
            or inputs.rules_tco_comparison is None
            or inputs.cheap_tco_comparison is None
            or not inputs.rules_tco_comparison.usable
            or not inputs.cheap_tco_comparison.usable
        )
        if missing_baseline:
            fail(
                "baseline",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "missing rules/cheap baseline quality or TCO comparison",
                [
                    "rules_arm",
                    "cheap_off_the_shelf_arm",
                    "rules_tco_comparison",
                    "cheap_tco_comparison",
                ],
            )
        else:
            assert inputs.rules_tco_comparison is not None
            assert inputs.cheap_tco_comparison is not None
            assert inputs.rules_quality is not None
            assert inputs.cheap_quality is not None
            rules_wins = (
                inputs.rules_quality.meets(th)
                and inputs.rules_tco_comparison.lower_tco_than_distillation
            )
            cheap_wins = (
                inputs.cheap_quality.meets(th)
                and inputs.cheap_tco_comparison.lower_tco_than_distillation
            )
            if rules_wins or cheap_wins:
                which = "rules" if rules_wins else "cheap_off_the_shelf"
                fail(
                    "baseline",
                    ProofStatus.DO_NOT_DISTILL,
                    f"{which} meets quality thresholds at lower TCO",
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
        if inputs.quality_retention_upper_95 is None:
            needed_q.append("quality_retention_upper_95")
        if inputs.max_primary_task_regression is None:
            needed_q.append("max_primary_task_regression")
        if inputs.json_schema_validity is None:
            needed_q.append("json_schema_validity")
        if inputs.ood_retention is None:
            needed_q.append("ood_retention")
        if inputs.ood_retention_lower_95 is None:
            needed_q.append("ood_retention_lower_95")
        if inputs.ood_retention_upper_95 is None:
            needed_q.append("ood_retention_upper_95")
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
            point_failures = [
                (
                    inputs.quality_retention_point >= th.quality_retention_point,
                    f"retention_point {inputs.quality_retention_point} "
                    f"< {th.quality_retention_point}",
                ),
                (
                    inputs.max_primary_task_regression
                    <= th.max_primary_task_regression + GATE_EPSILON,
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
            failed_msgs = [msg for ok, msg in point_failures if not ok]
            if failed_msgs:
                fail(
                    "quality",
                    ProofStatus.FAILED_QUALITY,
                    "; ".join(failed_msgs),
                    ["improve_student_quality_or_accept_failed_quality"],
                )
            else:
                assert inputs.quality_retention_lower_95 is not None
                assert inputs.quality_retention_upper_95 is not None
                assert inputs.ood_retention_lower_95 is not None
                assert inputs.ood_retention_upper_95 is not None
                definitive_interval_failures: list[str] = []
                if (
                    inputs.quality_retention_upper_95
                    < th.quality_retention_lower_95
                ):
                    definitive_interval_failures.append(
                        "quality-retention interval is wholly below "
                        f"{th.quality_retention_lower_95}"
                    )
                elif (
                    inputs.quality_retention_lower_95
                    < th.quality_retention_lower_95
                ):
                    wide_retention_interval = True

                if (
                    inputs.ood_retention_upper_95
                    < th.ood_retention_min
                ):
                    definitive_interval_failures.append(
                        "OOD-retention interval is wholly below "
                        f"{th.ood_retention_min}"
                    )
                elif inputs.ood_retention_lower_95 < th.ood_retention_min:
                    wide_ood_interval = True

                if definitive_interval_failures:
                    fail(
                        "quality",
                        ProofStatus.FAILED_QUALITY,
                        "; ".join(definitive_interval_failures),
                        ["improve_student_quality_or_accept_failed_quality"],
                    )
                else:
                    pass_gate(
                        "quality",
                        "point quality passes; interval uncertainty is either "
                        "passing or deferred to evidence",
                    )

    # --- 5. economics ---
    if first_failed is None:
        eco = inputs.economics
        if eco is None:
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "economics summary missing",
                [
                    "gross_cost_records",
                    "per_request_savings",
                    "break_even_within_horizon",
                    "utilization_sensitivity_rows",
                ],
            )
        elif not systems_ok:
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "complete finalist batch 1 and 8 systems evidence is required "
                "for serving economics",
                ["complete_finalist_batch_1_and_8_systems_evidence"],
            )
        elif not eco.evaluated:
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "economics unevaluated because required inputs are missing: "
                + ", ".join(eco.unevaluated_reasons),
                list(eco.unevaluated_reasons),
            )
        elif not eco.cost_ledger.complete:
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "gross cost ledger incomplete or unreconciled",
                list(eco.cost_ledger.completeness_gaps),
            )
        elif not eco.sensitivity_complete:
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "all batch 1/8 × 5/25/50/80 utilization rows are required",
                ["complete_batch_utilization_sensitivity"],
            )
        elif any(
            float(row["observed_throughput_rps"]["value"])
            != float(systems_by_batch[int(row["batch_size"])].requests_per_second.value)
            for row in eco.utilization_rows
        ):
            fail(
                "economics",
                ProofStatus.INSUFFICIENT_EVIDENCE,
                "economics throughput does not match finalist systems evidence",
                ["recompute_economics_from_finalist_systems_profiles"],
            )
        else:
            at_gate = [
                eco.break_even_for(
                    batch_size=batch_size,
                    utilization=th.economics_utilization,
                )
                for batch_size in (1, 8)
            ]
            if any(result is None for result in at_gate):
                fail(
                    "economics",
                    ProofStatus.INSUFFICIENT_EVIDENCE,
                    "batch 1 and batch 8 break-even rows are required at gate utilization",
                    ["batch_1_and_8_break_even_at_gate_utilization"],
                )
            elif any(
                result.break_even_requests == BREAK_EVEN_NEVER
                or result.savings_per_request_usd <= 0
                or not result.within_horizon
                for result in at_gate
                if result is not None
            ):
                fail(
                    "economics",
                    ProofStatus.FAILED_ECONOMICS,
                    "nonpositive savings or break-even outside horizon at "
                    f"{th.economics_utilization:.0%} utilization for batch 1 or 8",
                    ["lower_serving_cost_or_accept_failed_economics"],
                )
            else:
                pass_gate(
                    "economics",
                    "batch 1 and 8 have positive savings and break-even within "
                    f"horizon at {th.economics_utilization:.0%} utilization",
                )

    # --- 6. evidence ---
    if first_failed is None:
        required_seeds = {th.required_seed_screen, th.required_seed_replication}
        finalist_seeds = set(
            inputs.seed_sets_by_arm.get(inputs.finalist_arm_id, ())
        )
        comparator_seeds = set(
            inputs.seed_sets_by_arm.get(inputs.matched_comparator_arm_id, ())
        )
        seeds_ok = (
            required_seeds.issubset(finalist_seeds)
            and required_seeds.issubset(comparator_seeds)
            and finalist_seeds == comparator_seeds
        )
        required_baselines = {"rules", "teacher", "student_base", "cheap_off_the_shelf"}
        baselines_ok = required_baselines.issubset(set(inputs.baselines_present))
        required_arms = {
            *required_baselines,
            inputs.finalist_arm_id,
        }
        required_pairs = {
            (inputs.teacher_arm_id, inputs.matched_comparator_arm_id),
            (inputs.finalist_arm_id, inputs.teacher_arm_id),
            (inputs.finalist_arm_id, inputs.matched_comparator_arm_id),
            ("rules", inputs.teacher_arm_id),
            ("cheap_off_the_shelf", inputs.teacher_arm_id),
        }
        required_interval_ids = {
            f"arm::{arm_id}::{metric}"
            for arm_id in required_arms
            for metric in PRIMARY_METRICS
        } | {
            f"pair::{arm_a}::{arm_b}::{metric}_difference"
            for arm_a, arm_b in required_pairs
            for metric in PRIMARY_METRICS
        } | {
            f"pair::{inputs.finalist_arm_id}::{inputs.teacher_arm_id}"
            "::quality_retention",
            f"pair::{inputs.finalist_arm_id}::{inputs.teacher_arm_id}"
            "::ood_retention",
        }
        intervals_by_id = {
            interval.interval_id: interval for interval in inputs.paired_intervals
        }
        intervals_ok = (
            len(intervals_by_id) == len(inputs.paired_intervals)
            and required_interval_ids.issubset(intervals_by_id)
            and all(
                intervals_by_id[interval_id].proof_ready
                for interval_id in required_interval_ids
            )
            and all(
                interval.proof_ready
                for interval in inputs.paired_intervals
            )
        )
        arm_accounting_ok = bool(inputs.arm_accounting) and all(
            accounting.proof_complete for accounting in inputs.arm_accounting
        )
        missing: list[str] = []
        if not seeds_ok:
            missing.append(
                f"seeds_require_{th.required_seed_screen}_and_{th.required_seed_replication}"
            )
        if not baselines_ok:
            missing.append("missing_required_baseline_arms")
        if not intervals_ok:
            missing.append("paired_bootstrap_intervals")
        if inputs.frozen_hashes_present is not True:
            missing.append("frozen_protocol_and_prediction_hashes")
        if not systems_ok:
            missing.append("complete_finalist_batch_1_and_8_systems_evidence")
        if not arm_accounting_ok:
            missing.append("failed_and_filtered_example_accounting")
        if wide_retention_interval:
            missing.append("narrower_quality_retention_interval")
        if wide_ood_interval:
            missing.append("narrower_ood_retention_interval")
        if inputs.raw_text_provenance_complete is not True:
            missing.append("captured_raw_text_provenance")
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
                "derived multi-seed coverage, complete CI catalog, raw-text "
                "provenance, accounting, systems, and hashes present",
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
