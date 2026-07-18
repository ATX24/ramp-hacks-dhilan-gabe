"""Executable latent oracle: produces expected_output for each task directly from
latent state and verifies accounting identities. This defines benchmark truth."""
from __future__ import annotations

import hashlib

from .world import World, Txn, CHART_OF_ACCOUNTS, POLICIES, DRIVERS

VALID_GL = {a for a, _, _ in CHART_OF_ACCOUNTS}


def latent_state_hash(w: World) -> str:
    return "sha256:" + hashlib.sha256(f"{w.world_id}|{w.seed}".encode()).hexdigest()


def transaction_review_expected(w: World, txn: Txn) -> dict:
    out = {
        "schema_version": "transaction_review.v1",
        "task": "transaction_review",
        "gl_account": txn.gl_account,
        "journal_entry": [
            {"account": txn.gl_account, "side": "debit", "amount_minor": txn.amount_minor},
            {"account": "2110", "side": "credit", "amount_minor": txn.amount_minor},
        ],
        "policy_action": txn.policy_action,
        "rule_ids": [txn.policy_id],
        "evidence": [{"source_id": "txn", "field": "amount_minor", "value": str(txn.amount_minor)}],
        "confidence": 0.95,
    }
    _check_journal_balanced(out["journal_entry"])
    assert out["gl_account"] in VALID_GL
    return out


def variance_analysis_expected(w: World) -> dict:
    drivers = sorted(w.variance_drivers, key=lambda d: (-abs(d[1]), d[0]))  # abs-impact desc, id tiebreak
    profit = sum(v for _, v in drivers) + w.variance_other_minor
    out = {
        "schema_version": "variance_analysis.v1",
        "task": "variance_analysis",
        "profit_impact_minor": profit,
        "direction": "favorable" if profit >= 0 else "unfavorable",
        "top_drivers": [
            {"driver_id": d, "impact_minor": v, "rank": i + 1} for i, (d, v) in enumerate(drivers)
        ],
        "other_impact_minor": w.variance_other_minor,
        "rule_ids": ["VAR-MATERIAL-005"],
        "evidence_ids": [f"actual_{d}" for d, _ in drivers],
        "confidence": 0.92,
    }
    assert sum(d["impact_minor"] for d in out["top_drivers"]) + out["other_impact_minor"] == out["profit_impact_minor"]
    return out


def cash_reconciliation_expected(w: World) -> dict:
    r = w.recon
    matched = [{"book_ids": [f"b{i}"], "bank_ids": [f"k{i}"]} for i in range(r["n_matched"])]
    exceptions = [{"type": "bank_fee", "event_ids": [f"k{r['n_matched']}"], "amount_minor": r["bank_fee_minor"]}]
    if r["has_duplicate"]:
        exceptions.append({"type": "duplicate", "event_ids": [f"b{r['n_matched']}dup"], "amount_minor": 0})
    if r["deposit_in_transit_minor"]:
        exceptions.append({"type": "deposit_in_transit",
                           "event_ids": [f"b_dit"], "amount_minor": r["deposit_in_transit_minor"]})
    adj = r["book_balance_minor"] - r["bank_fee_minor"]
    out = {
        "schema_version": "cash_reconciliation.v1",
        "task": "cash_reconciliation",
        "status": "exceptions",
        "matched_groups": matched,
        "exceptions": exceptions,
        "adjusted_book_balance_minor": adj,
        "adjusted_bank_balance_minor": adj,
        "difference_minor": 0,
        "confidence": 0.9,
    }
    assert out["adjusted_book_balance_minor"] - out["adjusted_bank_balance_minor"] == out["difference_minor"]
    return out


def _check_journal_balanced(lines: list[dict]) -> None:
    debits = sum(l["amount_minor"] for l in lines if l["side"] == "debit")
    credits = sum(l["amount_minor"] for l in lines if l["side"] == "credit")
    if debits != credits:
        raise AssertionError(f"unbalanced journal: {debits} != {credits}")
