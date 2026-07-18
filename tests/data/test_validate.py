"""Validator unit tests including deliberate invalid mutations."""

from __future__ import annotations

from distillery.contracts.tasks import Difficulty, TaskId
from distillery.data.oracle import solve_task
from distillery.data.validate import validate_output
from distillery.data.world import build_world


def test_rejects_unbalanced_journal() -> None:
    world = build_world(
        seed=1,
        index=0,
        split_token="v",
        task=TaskId.TRANSACTION_REVIEW,
        difficulty=Difficulty.EASY,
    )
    out = solve_task(world, TaskId.TRANSACTION_REVIEW)
    out["journal_entry"][1]["amount_minor"] = out["journal_entry"][0]["amount_minor"] - 1
    result = validate_output(TaskId.TRANSACTION_REVIEW, out)
    assert not result.ok
    assert any("unbalanced" in e or "typed_output" in e for e in result.errors)


def test_rejects_variance_non_closure() -> None:
    world = build_world(
        seed=1,
        index=0,
        split_token="v",
        task=TaskId.VARIANCE_ANALYSIS,
        difficulty=Difficulty.MEDIUM,
    )
    out = solve_task(world, TaskId.VARIANCE_ANALYSIS)
    out["other_impact_minor"] += 1
    result = validate_output(TaskId.VARIANCE_ANALYSIS, out)
    assert not result.ok


def test_rejects_bad_driver_ranking() -> None:
    out = {
        "schema_version": "variance_analysis.v1",
        "task": "variance_analysis",
        "profit_impact_minor": -200000,
        "direction": "unfavorable",
        "top_drivers": [
            {"driver_id": "beta_cost", "impact_minor": -100000, "rank": 1},
            {"driver_id": "alpha_cost", "impact_minor": -100000, "rank": 2},
        ],
        "other_impact_minor": 0,
        "rule_ids": ["VAR-TIEBREAK-001"],
        "evidence_ids": ["a", "b"],
        "confidence": 0.9,
    }
    result = validate_output(TaskId.VARIANCE_ANALYSIS, out)
    assert not result.ok
    assert any("ranking" in e for e in result.errors)


def test_rejects_cash_difference_mismatch() -> None:
    out = {
        "schema_version": "cash_reconciliation.v1",
        "task": "cash_reconciliation",
        "status": "balanced",
        "matched_groups": [],
        "exceptions": [],
        "adjusted_book_balance_minor": 100,
        "adjusted_bank_balance_minor": 100,
        "difference_minor": 5,
        "confidence": 0.9,
    }
    result = validate_output(TaskId.CASH_RECONCILIATION, out)
    assert not result.ok
