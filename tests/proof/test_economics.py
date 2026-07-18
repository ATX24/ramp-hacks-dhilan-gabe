"""Economics edge cases: retention, recovered gap, break-even, projected labels."""

from __future__ import annotations

import math

from distillery.proof.economics import (
    BREAK_EVEN_NEVER,
    break_even_requests,
    compute_economics,
    quality_retention,
    recovered_teacher_gap,
    utilization_cost_rows,
)


def test_quality_retention_hand_calc() -> None:
    assert quality_retention(0.95, 1.0) == 0.95
    assert quality_retention(0.0, 1.0) == 0.0
    assert quality_retention(0.5, 0.0) is None


def test_recovered_teacher_gap_undefined_nonpositive_denominator() -> None:
    # denom = teacher - base = 0.5 - 0.5 = 0 → undefined
    assert recovered_teacher_gap(0.6, 0.5, 0.5) is None
    # denom negative → undefined, never clamp
    assert recovered_teacher_gap(0.4, 0.8, 0.5) is None


def test_recovered_teacher_gap_can_exceed_one_or_be_negative() -> None:
    # student above teacher: (1.0 - 0.2) / (0.8 - 0.2) = 0.8/0.6 ≈ 1.333
    gap = recovered_teacher_gap(1.0, 0.2, 0.8)
    assert gap is not None
    assert abs(gap - (0.8 / 0.6)) < 1e-12
    # student below base: (0.1 - 0.4) / (0.9 - 0.4) = -0.3/0.5 = -0.6
    gap_neg = recovered_teacher_gap(0.1, 0.4, 0.9)
    assert gap_neg is not None
    assert abs(gap_neg - (-0.6)) < 1e-12


def test_break_even_never_when_nonpositive_savings() -> None:
    assert break_even_requests(100.0, 0.01, 0.01) == BREAK_EVEN_NEVER
    assert break_even_requests(100.0, 0.01, 0.02) == BREAK_EVEN_NEVER


def test_break_even_ceil_hand_calc() -> None:
    # cost 100, savings 0.03 → ceil(100/0.03) = ceil(3333.333...) = 3334
    assert break_even_requests(100.0, 0.05, 0.02) == 3334
    assert break_even_requests(10.0, 1.0, 0.0) == 10


def test_utilization_rows_mark_projected_and_include_all_levels() -> None:
    rows = utilization_cost_rows(
        student_cost_per_request_at_full_util=0.001,
        teacher_cost_per_request=0.01,
        incremental_cost=50.0,
        batch1_throughput_rps=12.0,
        batch8_throughput_rps=40.0,
    )
    assert [r["utilization"] for r in rows] == [0.05, 0.25, 0.50, 0.80]
    for r in rows:
        assert r["student_cost_per_request_usd"]["kind"] == "projected"
        util = r["utilization"]
        expected = 0.001 / util
        assert abs(r["student_cost_per_request_usd"]["amount_usd"] - expected) < 1e-12
    # At 25%: student=0.004, savings=0.006, BE=ceil(50/0.006)=8334
    row_25 = rows[1]
    assert row_25["break_even_requests"] == math.ceil(50.0 / 0.006)


def test_compute_economics_missing_costs_never_invented() -> None:
    eco = compute_economics(
        student_primary_index=0.9,
        teacher_primary_index=1.0,
        base_primary_index=0.5,
        costs={},
        teacher_cost_per_request=None,
        student_cost_per_request_at_full_util=None,
    )
    assert eco.gross_experiment_cost_usd.kind == "missing"
    assert eco.break_even_at_25pct.break_even_requests is None
    assert "serving_costs_are_projected_not_measured_production_savings" in eco.notes
    assert eco.recovered_teacher_gap_defined is True
    assert abs((eco.recovered_teacher_gap or 0) - 0.8) < 1e-12


def test_compute_economics_break_even_within_horizon() -> None:
    eco = compute_economics(
        student_primary_index=0.96,
        teacher_primary_index=1.0,
        base_primary_index=0.5,
        costs={
            "gross_experiment_cost_usd": 40.0,
            "training_cost_usd": 30.0,
            "teacher_generation_cost_usd": 5.0,
            "storage_cost_usd": 1.0,
            "cheap_api_benchmark_cost_usd": 2.0,
        },
        teacher_cost_per_request=0.02,
        student_cost_per_request_at_full_util=0.001,
        batch1_throughput_rps=20.0,
        batch8_throughput_rps=80.0,
        evaluation_horizon_requests=100_000,
        economics_utilization=0.25,
    )
    # student at 25% = 0.004; savings=0.016; incremental=35; BE=ceil(35/0.016)=2188
    assert eco.break_even_at_25pct.break_even_requests == math.ceil(35.0 / 0.016)
    assert eco.break_even_at_25pct.within_horizon is True
    assert eco.break_even_at_25pct.student_cost_kind == "projected"
    assert eco.gross_experiment_cost_usd.kind == "measured"
    assert len(eco.utilization_rows) == 4


def test_break_even_never_propagates_in_summary() -> None:
    eco = compute_economics(
        student_primary_index=0.9,
        teacher_primary_index=1.0,
        base_primary_index=0.4,
        costs={"gross_experiment_cost_usd": 10.0, "training_cost_usd": 10.0},
        teacher_cost_per_request=0.001,
        student_cost_per_request_at_full_util=0.01,
        evaluation_horizon_requests=1_000_000,
    )
    assert eco.break_even_at_25pct.break_even_requests == BREAK_EVEN_NEVER
    assert eco.break_even_at_25pct.within_horizon is False
