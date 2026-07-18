"""Adversarial economics, cost-ledger, and provenance tests."""

from __future__ import annotations

import math

import pytest

from distillery.proof.economics import (
    BREAK_EVEN_NEVER,
    CostValue,
    break_even_requests,
    build_cost_ledger,
    compute_economics,
    quality_retention,
    recovered_teacher_gap,
    utilization_cost_rows,
)
from distillery.proof.evidence import EvidenceKind
from distillery.proof.testing import complete_cost_ledger


def _projected_prices() -> tuple[CostValue, CostValue]:
    return (
        CostValue(
            amount_usd=0.02,
            kind=EvidenceKind.PROJECTED,
            label="teacher_public_price_per_request",
        ),
        CostValue(
            amount_usd=1.20,
            kind=EvidenceKind.PROJECTED,
            label="student_instance_hour",
        ),
    )


def _complete_economics(**overrides):
    teacher_price, student_hourly = _projected_prices()
    kwargs = {
        "student_primary_index": 0.96,
        "teacher_primary_index": 1.0,
        "base_primary_index": 0.5,
        "costs": complete_cost_ledger(),
        "teacher_cost_per_request": teacher_price,
        "student_serving_hourly_cost_usd": student_hourly,
        "observed_throughput_rps_by_batch": {1: 20.0, 8: 80.0},
        "evaluation_horizon_requests": 100_000,
    }
    kwargs.update(overrides)
    return compute_economics(**kwargs)


def test_quality_retention_hand_calc() -> None:
    assert quality_retention(0.95, 1.0) == 0.95
    assert quality_retention(0.0, 1.0) == 0.0
    assert quality_retention(0.5, 0.0) is None
    assert quality_retention(0.5, -0.1) is None


def test_recovered_teacher_gap_edge_cases_never_clamp() -> None:
    assert recovered_teacher_gap(0.6, 0.5, 0.5) is None
    assert recovered_teacher_gap(0.4, 0.8, 0.5) is None
    above_one = recovered_teacher_gap(1.0, 0.2, 0.8)
    assert above_one is not None
    assert abs(above_one - (0.8 / 0.6)) < 1e-12
    below_zero = recovered_teacher_gap(0.1, 0.4, 0.9)
    assert below_zero is not None
    assert abs(below_zero - (-0.6)) < 1e-12


def test_break_even_never_and_ceil() -> None:
    assert break_even_requests(100.0, 0.01, 0.01) == BREAK_EVEN_NEVER
    assert break_even_requests(100.0, 0.01, 0.02) == BREAK_EVEN_NEVER
    assert break_even_requests(100.0, 0.05, 0.02) == 3334


def test_sensitivity_has_all_batch_and_utilization_rows() -> None:
    teacher_price, student_hourly = _projected_prices()
    rows = utilization_cost_rows(
        student_serving_hourly_cost_usd=student_hourly,
        teacher_cost_per_request_usd=teacher_price,
        incremental_cost=49.0,
        observed_throughput_rps_by_batch={1: 20.0, 8: 80.0},
        evaluation_horizon_requests=100_000,
    )
    assert {
        (row["batch_size"], row["utilization"]) for row in rows
    } == {
        (batch, utilization)
        for batch in (1, 8)
        for utilization in (0.05, 0.25, 0.50, 0.80)
    }
    assert len(rows) == 8
    for row in rows:
        assert row["observed_throughput_rps"]["kind"] == "measured"
        assert row["student_cost_per_request_usd"]["kind"] == "projected"
        # Public teacher/API price stays projected. It is not relabeled measured.
        assert row["teacher_cost_per_request_usd"]["kind"] == "projected"
    batch1_25 = next(
        row
        for row in rows
        if row["batch_size"] == 1 and row["utilization"] == 0.25
    )
    expected_student_cpr = 1.20 / (3600 * 20 * 0.25)
    expected_be = math.ceil(49.0 / (0.02 - expected_student_cpr))
    assert batch1_25["break_even_requests"] == expected_be


def test_complete_cost_ledger_reconciles_gross() -> None:
    ledger = build_cost_ledger(complete_cost_ledger())
    assert ledger.complete is True
    assert ledger.reconciled is True
    assert ledger.component_sum_usd == ledger.gross_experiment_cost_usd.amount_usd
    # Gross 55 minus cheap benchmark 3.
    assert ledger.incremental_distillation_cost_usd == 52.0


@pytest.mark.parametrize(
    ("missing_key", "gap"),
    [
        ("billed_training_seconds", "billed_training_seconds_not_measured"),
        ("gpu_compute_cost_usd", "gpu_compute_cost_usd_not_measured"),
        ("teacher_generation_tokens", "teacher_generation_tokens_not_measured"),
        ("teacher_generation_cost_usd", "teacher_generation_cost_usd_not_measured"),
        ("cheap_api_benchmark_cost_usd", "cheap_api_benchmark_cost_usd_not_measured"),
        ("storage_cost_usd", "storage_cost_usd_not_measured"),
        ("other_costs_usd", "other_costs_not_declared"),
    ],
)
def test_each_required_ledger_component_is_fail_closed(
    missing_key: str,
    gap: str,
) -> None:
    costs = complete_cost_ledger()
    del costs[missing_key]
    ledger = build_cost_ledger(costs)
    assert ledger.complete is False
    assert gap in ledger.completeness_gaps


def test_measured_zero_teacher_generation_is_valid() -> None:
    ledger = build_cost_ledger(
        complete_cost_ledger(
            teacher_generation_tokens=0,
            teacher_generation_cost_usd=0.0,
        )
    )
    assert ledger.complete is True


def test_fractional_teacher_token_count_is_incomplete() -> None:
    costs = complete_cost_ledger()
    costs["teacher_generation_tokens"]["value"] = 0.5
    ledger = build_cost_ledger(costs)
    assert "teacher_generation_tokens_not_integer" in ledger.completeness_gaps


def test_zero_cheap_api_requires_not_run_reason() -> None:
    costs = complete_cost_ledger(cheap_api_benchmark_cost_usd=0.0)
    missing_reason = build_cost_ledger(costs)
    assert "cheap_api_zero_requires_reason" in missing_reason.completeness_gaps

    with_reason = build_cost_ledger(
        {
            **costs,
            "cheap_api_zero_reason": "not run: provider unavailable before freeze",
        }
    )
    assert with_reason.complete is True


def test_unreconciled_gross_total_is_incomplete() -> None:
    costs = complete_cost_ledger()
    costs["gross_experiment_cost_usd"]["amount_usd"] += 1.0
    ledger = build_cost_ledger(costs)
    assert ledger.complete is False
    assert "gross_total_does_not_reconcile" in ledger.completeness_gaps


def test_training_only_incremental_estimate_never_emits_economics() -> None:
    costs = {
        "gross_experiment_cost_usd": {
            "amount_usd": 40.0,
            "kind": "measured",
        },
        "billed_training_seconds": {"value": 7200, "kind": "measured"},
        "gpu_compute_cost_usd": {"amount_usd": 40.0, "kind": "measured"},
    }
    eco = _complete_economics(costs=costs)
    assert eco.evaluated is False
    assert eco.utilization_rows == ()
    assert eco.break_even_at_25pct == ()
    assert eco.cost_ledger.incremental_distillation_cost_usd is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"student_primary_index": None},
        {"teacher_primary_index": None},
        {"base_primary_index": None},
        {"teacher_cost_per_request": None},
        {"student_serving_hourly_cost_usd": None},
        {"observed_throughput_rps_by_batch": {1: 20.0}},
        {"evaluation_horizon_requests": None},
    ],
)
def test_missing_inputs_do_not_emit_break_even(overrides: dict) -> None:
    eco = _complete_economics(**overrides)
    assert eco.evaluated is False
    assert eco.utilization_rows == ()
    assert eco.break_even_at_25pct == ()
    assert eco.unevaluated_reasons


def test_complete_economics_uses_both_batches() -> None:
    eco = _complete_economics()
    assert eco.evaluated is True
    assert eco.sensitivity_complete is True
    assert len(eco.utilization_rows) == 8
    assert {row.batch_size for row in eco.break_even_at_25pct} == {1, 8}


def test_nonpositive_savings_produce_never_for_each_batch() -> None:
    eco = _complete_economics(
        teacher_cost_per_request=CostValue(
            0.000001,
            EvidenceKind.PROJECTED,
            "teacher_public_price_per_request",
        )
    )
    assert eco.evaluated is True
    assert all(
        row.break_even_requests == BREAK_EVEN_NEVER
        for row in eco.break_even_at_25pct
    )


def test_unknown_kind_fails_loud() -> None:
    with pytest.raises(ValueError, match="not a valid EvidenceKind"):
        CostValue(1.0, "estimated", "bad_kind")

    costs = complete_cost_ledger()
    costs["gpu_compute_cost_usd"]["kind"] = "observed-ish"
    with pytest.raises(ValueError, match="not a valid EvidenceKind"):
        build_cost_ledger(costs)


def test_bare_numbers_do_not_default_to_measured() -> None:
    costs = complete_cost_ledger()
    costs["gpu_compute_cost_usd"] = 40.0
    with pytest.raises(ValueError, match="explicitly declare"):
        build_cost_ledger(costs)
