"""Every proof-status transition and missing-evidence fail-loud behavior."""

from __future__ import annotations

from distillery.contracts.proof import ProofStatus
from distillery.proof.bootstrap import BootstrapCI
from distillery.proof.economics import (
    BREAK_EVEN_NEVER,
    BreakEvenResult,
    CostValue,
    EconomicsSummary,
)
from distillery.proof.gates import GATE_ORDER, GateInputs, evaluate_gates


def _ci(estimate: float, lower: float, upper: float) -> BootstrapCI:
    return BootstrapCI(
        estimate=estimate,
        lower=lower,
        upper=upper,
        level=0.95,
        n_resamples=100,
        n_clusters=20,
        n_examples=200,
        underpowered=False,
        metric="teacher_gap",
        seed=17,
    )


def _eco(
    *,
    savings: float = 0.01,
    be: int | str = 1000,
    within: bool = True,
    gross_kind: str = "measured",
    gross: float | None = 40.0,
) -> EconomicsSummary:
    return EconomicsSummary(
        gross_experiment_cost_usd=CostValue(gross, gross_kind, "gross"),  # type: ignore[arg-type]
        teacher_generation_cost_usd=CostValue(5.0, "measured", "teacher_gen"),
        cheap_api_benchmark_cost_usd=CostValue(1.0, "measured", "cheap"),
        storage_cost_usd=CostValue(1.0, "measured", "storage"),
        training_cost_usd=CostValue(30.0, "measured", "training"),
        quality_retention=0.96,
        recovered_teacher_gap=0.9,
        recovered_teacher_gap_defined=True,
        utilization_rows=(),
        break_even_at_25pct=BreakEvenResult(
            break_even_requests=be,  # type: ignore[arg-type]
            savings_per_request_usd=savings,
            incremental_cost_usd=35.0,
            teacher_cost_per_request_usd=0.02,
            student_cost_per_request_usd=0.01,
            student_cost_kind="projected",
            utilization=0.25,
            horizon_requests=100_000,
            within_horizon=within,
        ),
    )


def _passing_inputs(**overrides) -> GateInputs:
    base = dict(
        teacher_minus_student=0.10,
        teacher_minus_student_ci=_ci(0.10, 0.06, 0.14),
        rules_meets_quality=False,
        cheap_meets_quality=False,
        rules_projected_tco_below_distill=False,
        cheap_projected_tco_below_distill=False,
        trainer_numerical_ok=True,
        trainer_artifact_reload_ok=True,
        trainer_smoke_eval_ok=True,
        trainer_memory_cost_ok=True,
        quality_retention_point=0.96,
        quality_retention_lower_95=0.92,
        max_primary_task_regression=0.02,
        json_schema_validity=0.995,
        ood_retention=0.93,
        critical_invariant_violations=0,
        economics=_eco(),
        positive_per_request_savings=True,
        break_even_within_horizon=True,
        seeds_present=(17, 23),
        baselines_present=("rules", "teacher", "student_base", "cheap_off_the_shelf"),
        cost_records_complete=True,
        paired_intervals_present=True,
        frozen_hashes_present=True,
        throughput_evidence_adequate=True,
        wide_interval_spans_gate=False,
    )
    base.update(overrides)
    return GateInputs(**base)


def test_gate_order_locked() -> None:
    assert GATE_ORDER == (
        "pilot_teacher",
        "baseline",
        "trainer",
        "quality",
        "economics",
        "evidence",
    )


def test_status_proved() -> None:
    ev = evaluate_gates(_passing_inputs())
    assert ev.proof_status is ProofStatus.PROVED
    assert ev.first_failed_gate is None
    assert ev.unevaluated_gates == ()
    assert all(g.evaluated and g.passed for g in ev.quality_gates)


def test_status_do_not_distill_from_baseline() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            rules_meets_quality=True,
            rules_projected_tco_below_distill=True,
        )
    )
    assert ev.proof_status is ProofStatus.DO_NOT_DISTILL
    assert ev.first_failed_gate == "baseline"
    assert "trainer" in ev.unevaluated_gates
    assert "quality" in ev.unevaluated_gates
    assert "economics" in ev.unevaluated_gates
    assert "evidence" in ev.unevaluated_gates


def test_status_do_not_distill_from_cheap_api() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            cheap_meets_quality=True,
            cheap_projected_tco_below_distill=True,
        )
    )
    assert ev.proof_status is ProofStatus.DO_NOT_DISTILL
    assert ev.first_failed_gate == "baseline"


def test_status_failed_quality() -> None:
    ev = evaluate_gates(_passing_inputs(quality_retention_point=0.80))
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"
    assert "economics" in ev.unevaluated_gates
    assert "evidence" in ev.unevaluated_gates


def test_status_failed_quality_invariant_violation() -> None:
    ev = evaluate_gates(_passing_inputs(critical_invariant_violations=2))
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"


def test_status_failed_economics() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            economics=_eco(savings=-0.01, be=BREAK_EVEN_NEVER, within=False),
            positive_per_request_savings=False,
            break_even_within_horizon=False,
        )
    )
    assert ev.proof_status is ProofStatus.FAILED_ECONOMICS
    assert ev.first_failed_gate == "economics"
    assert "evidence" in ev.unevaluated_gates


def test_status_insufficient_evidence_missing_seed() -> None:
    ev = evaluate_gates(_passing_inputs(seeds_present=(17,)))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "evidence"
    assert any("seeds" in e for e in ev.evidence_needed)


def test_status_insufficient_evidence_missing_baseline_arm() -> None:
    ev = evaluate_gates(
        _passing_inputs(baselines_present=("teacher", "student_base"))
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "evidence"


def test_status_insufficient_evidence_missing_costs() -> None:
    ev = evaluate_gates(_passing_inputs(cost_records_complete=False))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "evidence"


def test_missing_gross_cost_fails_economics_as_insufficient() -> None:
    ev = evaluate_gates(
        _passing_inputs(economics=_eco(gross=None, gross_kind="missing"))
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "economics"


def test_missing_pilot_never_passes() -> None:
    ev = evaluate_gates(
        _passing_inputs(teacher_minus_student=None, teacher_minus_student_ci=None)
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "pilot_teacher"
    assert set(ev.unevaluated_gates) == {
        "baseline",
        "trainer",
        "quality",
        "economics",
        "evidence",
    }


def test_pilot_gap_ci_includes_zero_insufficient() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            teacher_minus_student=0.06,
            teacher_minus_student_ci=_ci(0.06, -0.01, 0.12),
        )
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "pilot_teacher"


def test_missing_trainer_flags_never_pass() -> None:
    ev = evaluate_gates(_passing_inputs(trainer_numerical_ok=None))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "trainer"


def test_missing_baseline_comparison_never_pass() -> None:
    ev = evaluate_gates(_passing_inputs(rules_meets_quality=None))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "baseline"


def test_wide_interval_spans_gate_insufficient() -> None:
    ev = evaluate_gates(_passing_inputs(wide_interval_spans_gate=True))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "evidence"


def test_exactly_five_statuses_reachable() -> None:
    statuses = {
        evaluate_gates(_passing_inputs()).proof_status,
        evaluate_gates(
            _passing_inputs(
                rules_meets_quality=True, rules_projected_tco_below_distill=True
            )
        ).proof_status,
        evaluate_gates(_passing_inputs(quality_retention_point=0.5)).proof_status,
        evaluate_gates(
            _passing_inputs(
                positive_per_request_savings=False,
                break_even_within_horizon=False,
                economics=_eco(savings=-1, be=BREAK_EVEN_NEVER, within=False),
            )
        ).proof_status,
        evaluate_gates(_passing_inputs(seeds_present=())).proof_status,
    }
    assert statuses == {
        ProofStatus.PROVED,
        ProofStatus.DO_NOT_DISTILL,
        ProofStatus.FAILED_QUALITY,
        ProofStatus.FAILED_ECONOMICS,
        ProofStatus.INSUFFICIENT_EVIDENCE,
    }
