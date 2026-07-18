"""Every proof status and direct gate/evaluator completeness semantics."""

from __future__ import annotations

from dataclasses import replace

from distillery.contracts.budgets import ProofGates
from distillery.contracts.proof import ProofStatus
from distillery.proof.bootstrap import PRIMARY_METRICS, BootstrapCI
from distillery.proof.economics import (
    CostValue,
    compute_economics,
)
from distillery.proof.evidence import EvidenceKind
from distillery.proof.gates import (
    GATE_ORDER,
    ArmAccountingEvidence,
    ArmQualityEvidence,
    BaselineTCOComparison,
    GateInputs,
    evaluate_gates,
)
from distillery.proof.systems import summarize_systems
from distillery.proof.testing import (
    complete_cost_ledger,
    complete_systems_profile,
)


def _ci(
    estimate: float,
    lower: float,
    upper: float,
    *,
    metric: str,
    arm_a: str,
    arm_b: str | None = None,
) -> BootstrapCI:
    return BootstrapCI(
        estimate=estimate,
        lower=lower,
        upper=upper,
        level=0.95,
        n_resamples=10_000,
        valid_resamples=10_000,
        excluded_resamples=0,
        n_clusters=20,
        n_examples=200,
        underpowered=False,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seed=17,
    )


def _comparison(
    cheaper: bool,
    *,
    kind: EvidenceKind = EvidenceKind.MEASURED,
) -> BaselineTCOComparison:
    return BaselineTCOComparison(
        lower_tco_than_distillation=cheaper,
        kind=kind,
        detail="frozen comparison",
    )


def _quality(*, passing: bool = True) -> ArmQualityEvidence:
    return ArmQualityEvidence(
        quality_retention=0.96 if passing else 0.50,
        max_primary_task_regression=0.02 if passing else 0.50,
        json_schema_validity=0.995,
        ood_retention=0.93,
        critical_invariant_violations=0,
    )


def _paired_intervals() -> tuple[BootstrapCI, ...]:
    arms = (
        "rules",
        "teacher",
        "student_base",
        "cheap_off_the_shelf",
        "sequence_kd",
    )
    pairs = (
        ("teacher", "student_base"),
        ("sequence_kd", "teacher"),
        ("sequence_kd", "student_base"),
        ("rules", "teacher"),
        ("cheap_off_the_shelf", "teacher"),
    )
    intervals = [
        _ci(
            0.95,
            0.92,
            0.99,
            metric=metric,
            arm_a=arm,
        )
        for arm in arms
        for metric in PRIMARY_METRICS
    ]
    intervals.extend(
        _ci(
            0.10,
            0.06,
            0.14,
            metric=f"{metric}_difference",
            arm_a=arm_a,
            arm_b=arm_b,
        )
        for arm_a, arm_b in pairs
        for metric in PRIMARY_METRICS
    )
    intervals.extend(
        (
            _ci(
                0.96,
                0.92,
                0.99,
                metric="quality_retention",
                arm_a="sequence_kd",
                arm_b="teacher",
            ),
            _ci(
                0.93,
                0.91,
                0.98,
                metric="ood_retention",
                arm_a="sequence_kd",
                arm_b="teacher",
            ),
        )
    )
    return tuple(intervals)


def _systems():
    return (
        summarize_systems(
            complete_systems_profile(
                batch_size=1,
                requests_per_second=20.0,
            )
        ),
        summarize_systems(
            complete_systems_profile(
                batch_size=8,
                requests_per_second=80.0,
            )
        ),
    )


def _accounting() -> tuple[ArmAccountingEvidence, ...]:
    return tuple(
        ArmAccountingEvidence(
            arm_id=arm_id,
            expected_examples=200,
            prediction_records=200,
            failed_examples=0,
            filtered_examples=0,
            failed_example_reasons={},
            filtered_example_reasons={},
        )
        for arm_id in (
            "rules",
            "teacher",
            "student_base",
            "cheap_off_the_shelf",
            "sequence_kd",
        )
    )


def _eco(
    *,
    teacher_cost_per_request: float = 0.02,
    costs: dict | None = None,
):
    return compute_economics(
        student_primary_index=0.96,
        teacher_primary_index=1.0,
        base_primary_index=0.5,
        costs=complete_cost_ledger() if costs is None else costs,
        teacher_cost_per_request=CostValue(
            teacher_cost_per_request,
            EvidenceKind.PROJECTED,
            "teacher_public_price_per_request",
        ),
        student_serving_hourly_cost_usd=CostValue(
            1.20,
            EvidenceKind.PROJECTED,
            "student_instance_hour",
        ),
        observed_throughput_rps_by_batch={1: 20.0, 8: 80.0},
        evaluation_horizon_requests=100_000,
    )


def _passing_inputs(**overrides) -> GateInputs:
    intervals = _paired_intervals()
    teacher_gap_ci = next(
        interval
        for interval in intervals
        if interval.interval_id
        == "pair::teacher::student_base::primary_index_difference"
    )
    base = {
        "teacher_minus_student": 0.10,
        "teacher_minus_student_ci": teacher_gap_ci,
        "rules_quality": _quality(passing=False),
        "cheap_quality": _quality(passing=False),
        "rules_tco_comparison": _comparison(False),
        "cheap_tco_comparison": _comparison(False),
        "trainer_numerical_ok": True,
        "trainer_artifact_reload_ok": True,
        "trainer_smoke_eval_ok": True,
        "trainer_memory_cost_ok": True,
        "quality_retention_point": 0.96,
        "quality_retention_lower_95": 0.92,
        "quality_retention_upper_95": 0.99,
        "max_primary_task_regression": 0.02,
        "json_schema_validity": 0.995,
        "ood_retention": 0.93,
        "ood_retention_lower_95": 0.91,
        "ood_retention_upper_95": 0.98,
        "critical_invariant_violations": 0,
        "economics": _eco(),
        "seed_sets_by_arm": {
            arm_id: (17, 23)
            for arm_id in (
                "rules",
                "teacher",
                "student_base",
                "cheap_off_the_shelf",
                "sequence_kd",
            )
        },
        "baselines_present": (
            "rules",
            "teacher",
            "student_base",
            "cheap_off_the_shelf",
        ),
        "paired_intervals": intervals,
        "frozen_hashes_present": True,
        "finalist_systems": _systems(),
        "arm_accounting": _accounting(),
        "raw_text_provenance_complete": True,
    }
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


def test_measured_cheaper_baseline_returns_do_not_distill_first() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            rules_quality=_quality(),
            rules_tco_comparison=_comparison(True, kind=EvidenceKind.MEASURED),
        )
    )
    assert ev.proof_status is ProofStatus.DO_NOT_DISTILL
    assert ev.first_failed_gate == "baseline"
    assert ev.unevaluated_gates == (
        "trainer",
        "quality",
        "economics",
        "evidence",
    )


def test_baseline_uses_proof_gate_thresholds() -> None:
    inputs = _passing_inputs(
        rules_quality=_quality(),
        rules_tco_comparison=_comparison(True),
    )
    stricter = ProofGates(quality_retention_point=0.97)
    ev = evaluate_gates(inputs, thresholds=stricter)
    # At 0.96 the rules arm no longer clears the configurable baseline gate.
    # The finalist also fails the same point threshold at the later quality gate.
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"


def test_cheaper_cheap_api_returns_do_not_distill() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            cheap_quality=_quality(),
            cheap_tco_comparison=_comparison(True),
        )
    )
    assert ev.proof_status is ProofStatus.DO_NOT_DISTILL
    assert ev.first_failed_gate == "baseline"


def test_missing_either_tco_comparison_fails_baseline_before_quality() -> None:
    for field in ("rules_tco_comparison", "cheap_tco_comparison"):
        ev = evaluate_gates(_passing_inputs(**{field: None}))
        assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
        assert ev.first_failed_gate == "baseline"
        assert "quality" in ev.unevaluated_gates


def test_missing_baseline_arm_quality_fails_baseline() -> None:
    ev = evaluate_gates(_passing_inputs(rules_quality=None))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "baseline"


def test_incomplete_baseline_quality_is_not_treated_as_failing_threshold() -> None:
    incomplete = ArmQualityEvidence(
        quality_retention=None,
        max_primary_task_regression=0.01,
        json_schema_validity=1.0,
        ood_retention=1.0,
        critical_invariant_violations=0,
    )
    ev = evaluate_gates(_passing_inputs(rules_quality=incomplete))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "baseline"


def test_true_point_quality_failure_is_failed_quality() -> None:
    ev = evaluate_gates(_passing_inputs(quality_retention_point=0.80))
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"


def test_interval_wholly_below_lower_gate_is_failed_quality() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            quality_retention_lower_95=0.80,
            quality_retention_upper_95=0.89,
        )
    )
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"


def test_wide_interval_spanning_gate_is_insufficient_evidence() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            quality_retention_point=0.96,
            quality_retention_lower_95=0.85,
            quality_retention_upper_95=0.97,
        )
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "evidence"
    assert "narrower_quality_retention_interval" in ev.evidence_needed


def test_ood_point_pass_with_wide_interval_is_insufficient() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            ood_retention=0.93,
            ood_retention_lower_95=0.85,
            ood_retention_upper_95=0.97,
        )
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "evidence"
    assert "narrower_ood_retention_interval" in ev.evidence_needed


def test_ood_interval_wholly_below_gate_is_failed_quality() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            ood_retention=0.93,
            ood_retention_lower_95=0.80,
            ood_retention_upper_95=0.89,
        )
    )
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"


def test_invariant_violation_is_failed_quality() -> None:
    ev = evaluate_gates(_passing_inputs(critical_invariant_violations=1))
    assert ev.proof_status is ProofStatus.FAILED_QUALITY
    assert ev.first_failed_gate == "quality"


def test_nonpositive_savings_is_failed_economics() -> None:
    ev = evaluate_gates(
        _passing_inputs(economics=_eco(teacher_cost_per_request=0.000001))
    )
    assert ev.proof_status is ProofStatus.FAILED_ECONOMICS
    assert ev.first_failed_gate == "economics"


def test_incomplete_cost_ledger_is_insufficient_at_economics() -> None:
    ev = evaluate_gates(_passing_inputs(economics=_eco(costs={})))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "economics"


def test_empty_sensitivity_cannot_pass_direct_gate() -> None:
    incomplete = replace(
        _eco(),
        utilization_rows=(),
        break_even_at_25pct=(),
    )
    ev = evaluate_gates(_passing_inputs(economics=incomplete))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "economics"


def test_missing_seed_baseline_hash_systems_or_accounting_never_proves() -> None:
    cases = (
        {
            "seed_sets_by_arm": {
                arm_id: ((17,) if arm_id == "sequence_kd" else (17, 23))
                for arm_id in (
                    "rules",
                    "teacher",
                    "student_base",
                    "cheap_off_the_shelf",
                    "sequence_kd",
                )
            }
        },
        {"baselines_present": ("teacher", "student_base")},
        {"frozen_hashes_present": False},
        {"arm_accounting": ()},
        {"raw_text_provenance_complete": False},
    )
    for overrides in cases:
        ev = evaluate_gates(_passing_inputs(**overrides))
        assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
        assert ev.first_failed_gate == "evidence"


def test_missing_or_underpowered_required_interval_never_proves() -> None:
    missing = evaluate_gates(
        _passing_inputs(paired_intervals=_paired_intervals()[:-1])
    )
    assert missing.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert missing.first_failed_gate == "evidence"

    intervals = list(_paired_intervals())
    intervals[0] = replace(intervals[0], underpowered=True)
    underpowered = evaluate_gates(
        _passing_inputs(paired_intervals=tuple(intervals))
    )
    assert underpowered.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert underpowered.first_failed_gate == "evidence"

    intervals = list(_paired_intervals())
    intervals[0] = replace(
        intervals[0],
        valid_resamples=9_999,
        excluded_resamples=1,
    )
    excluded_draw = evaluate_gates(
        _passing_inputs(paired_intervals=tuple(intervals))
    )
    assert excluded_draw.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert excluded_draw.first_failed_gate == "evidence"

    extra_underpowered = replace(
        _ci(
            0.95,
            0.80,
            1.0,
            metric="primary_index",
            arm_a="extra_arm",
        ),
        underpowered=True,
    )
    extra = evaluate_gates(
        _passing_inputs(
            paired_intervals=(
                *_paired_intervals(),
                extra_underpowered,
            )
        )
    )
    assert extra.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert extra.first_failed_gate == "evidence"


def test_projected_prices_need_measured_systems_evidence() -> None:
    assert _eco().teacher_cost_per_request_usd.kind is EvidenceKind.PROJECTED
    ev = evaluate_gates(
        _passing_inputs(
            economics=_eco(),
            finalist_systems=(),
        )
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "economics"


def test_economics_cannot_use_throughput_different_from_systems() -> None:
    mismatched = _eco()
    # Recompute economics from favorable throughput while retaining the actual
    # systems profiles in GateInputs.
    teacher_price = mismatched.teacher_cost_per_request_usd
    student_hourly = mismatched.student_serving_hourly_cost_usd
    favorable = compute_economics(
        student_primary_index=0.96,
        teacher_primary_index=1.0,
        base_primary_index=0.5,
        costs=complete_cost_ledger(),
        teacher_cost_per_request=teacher_price,
        student_serving_hourly_cost_usd=student_hourly,
        observed_throughput_rps_by_batch={1: 200.0, 8: 800.0},
        evaluation_horizon_requests=100_000,
    )
    ev = evaluate_gates(_passing_inputs(economics=favorable))
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "economics"
    assert "recompute_economics_from_finalist_systems_profiles" in ev.evidence_needed


def test_missing_pilot_or_trainer_evidence_never_passes() -> None:
    pilot = evaluate_gates(
        _passing_inputs(
            teacher_minus_student=None,
            teacher_minus_student_ci=None,
        )
    )
    assert pilot.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert pilot.first_failed_gate == "pilot_teacher"

    trainer = evaluate_gates(_passing_inputs(trainer_numerical_ok=None))
    assert trainer.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert trainer.first_failed_gate == "trainer"


def test_pilot_ci_including_zero_is_insufficient() -> None:
    ev = evaluate_gates(
        _passing_inputs(
            teacher_minus_student=0.06,
            teacher_minus_student_ci=_ci(
                0.06,
                -0.01,
                0.12,
                metric="primary_index_difference",
                arm_a="teacher",
                arm_b="student_base",
            ),
        )
    )
    assert ev.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert ev.first_failed_gate == "pilot_teacher"


def test_exactly_five_statuses_reachable() -> None:
    statuses = {
        evaluate_gates(_passing_inputs()).proof_status,
        evaluate_gates(
            _passing_inputs(
                rules_quality=_quality(),
                rules_tco_comparison=_comparison(True),
            )
        ).proof_status,
        evaluate_gates(
            _passing_inputs(quality_retention_point=0.5)
        ).proof_status,
        evaluate_gates(
            _passing_inputs(economics=_eco(teacher_cost_per_request=0.000001))
        ).proof_status,
        evaluate_gates(
            _passing_inputs(seed_sets_by_arm={})
        ).proof_status,
    }
    assert statuses == {
        ProofStatus.PROVED,
        ProofStatus.DO_NOT_DISTILL,
        ProofStatus.FAILED_QUALITY,
        ProofStatus.FAILED_ECONOMICS,
        ProofStatus.INSUFFICIENT_EVIDENCE,
    }
