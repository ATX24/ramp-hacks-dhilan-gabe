"""Builders for proof unit tests and offline fixture assembly."""

from __future__ import annotations

from typing import Any

from distillery.proof.metrics import PredictionRecord


def txn_gold(
    *,
    gl: str = "6100",
    amount: int = 4500,
    action: str = "approve",
    rules: list[str] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    rules = rules or ["POL-MEAL-001"]
    return {
        "schema_version": "transaction_review.v1",
        "task": "transaction_review",
        "gl_account": gl,
        "journal_entry": [
            {"account": gl, "side": "debit", "amount_minor": amount},
            {"account": "2100", "side": "credit", "amount_minor": amount},
        ],
        "policy_action": action,
        "rule_ids": rules,
        "evidence": [{"source_id": "txn", "field": "amount_minor", "value": str(amount)}],
        "confidence": confidence,
    }


def var_gold(
    *,
    profit: int = -420000,
    drivers: list[dict[str, Any]] | None = None,
    other: int = -30000,
    confidence: float = 0.91,
) -> dict[str, Any]:
    if drivers is None:
        drivers = [
            {"driver_id": "cloud_usage", "impact_minor": -300000, "rank": 1},
            {"driver_id": "support_volume", "impact_minor": -90000, "rank": 2},
        ]
    direction = "favorable" if profit >= 0 else "unfavorable"
    return {
        "schema_version": "variance_analysis.v1",
        "task": "variance_analysis",
        "profit_impact_minor": profit,
        "direction": direction,
        "top_drivers": drivers,
        "other_impact_minor": other,
        "rule_ids": ["VAR-MATERIAL-005"],
        "evidence_ids": ["actual_cloud", "budget_cloud"],
        "confidence": confidence,
    }


def cash_gold() -> dict[str, Any]:
    return {
        "schema_version": "cash_reconciliation.v1",
        "task": "cash_reconciliation",
        "status": "exceptions",
        "matched_groups": [{"book_ids": ["b1"], "bank_ids": ["k9"]}],
        "exceptions": [{"type": "bank_fee", "event_ids": ["k10"], "amount_minor": 3500}],
        "adjusted_book_balance_minor": 8123400,
        "adjusted_bank_balance_minor": 8123400,
        "difference_minor": 0,
        "confidence": 0.89,
    }


def make_pred(
    *,
    example_id: str,
    world_id: str,
    task: str,
    expected: dict[str, Any],
    parsed: dict[str, Any] | None = None,
    raw_text: str | None = None,
    refused: bool = False,
    split: str = "iid_test",
    difficulty: str = "medium",
    template_family: str = "tmpl_a",
    arm_id: str = "arm",
    slices: dict[str, str] | None = None,
    latency_ms: float | None = 10.0,
    output_tokens: int | None = 40,
) -> PredictionRecord:
    return PredictionRecord(
        example_id=example_id,
        world_id=world_id,
        group_id="grp_test",
        task=task,
        difficulty=difficulty,
        split=split,
        template_family=template_family,
        arm_id=arm_id,
        raw_text=raw_text,
        parsed=parsed if parsed is not None else (None if raw_text is not None else expected),
        refused=refused,
        latency_ms=latency_ms,
        output_tokens=output_tokens,
        expected_output=expected,
        slices=slices or {},
    )
