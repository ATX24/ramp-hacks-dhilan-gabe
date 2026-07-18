"""Renderers: turn latent state into varied task prompts (inputs). The rendered
input never contains the expected output."""
from __future__ import annotations

from .world import World, Txn, CHART_OF_ACCOUNTS, POLICIES


def _coa_excerpt(gl: str) -> list[dict]:
    # Include the true account plus near-synonym distractors.
    rows = [{"account": a, "name": n, "type": t} for a, n, t in CHART_OF_ACCOUNTS]
    return rows


def _policy_excerpts() -> list[dict]:
    return [{"rule_id": pid, "text": text} for pid, text, _, _ in POLICIES]


def render_transaction_review(w: World, txn: Txn) -> dict:
    return {
        "transaction": {
            "descriptor": txn.descriptor,
            "vendor_hint": txn.vendor,
            "amount_minor": txn.amount_minor,
            "currency": txn.currency,
            "date": txn.date,
            "department": txn.department,
            "payment_account": "2110",
        },
        "chart_of_accounts": _coa_excerpt(txn.gl_account),
        "policies": _policy_excerpts(),
    }


def render_variance_analysis(w: World) -> dict:
    lines = []
    for d, v in w.variance_drivers:
        lines.append({"driver_id": d, "budget_minor": 1_000_000, "actual_minor": 1_000_000 + (-v)})
    return {
        "period": "2026-06",
        "driver_lines": lines,
        "other_impact_minor": w.variance_other_minor,
        "sign_convention": "negative profit_impact_minor is unfavorable",
        "materiality": {"rule_id": "VAR-MATERIAL-005", "threshold_minor": 10000},
        "evidence_index": [f"actual_{d}" for d, _ in w.variance_drivers] + [f"budget_{d}" for d, _ in w.variance_drivers],
    }


def render_cash_reconciliation(w: World) -> dict:
    r = w.recon
    book = [{"id": f"b{i}", "amount_minor": 10000 * (i + 1)} for i in range(r["n_matched"])]
    bank = [{"id": f"k{i}", "amount_minor": 10000 * (i + 1)} for i in range(r["n_matched"])]
    bank.append({"id": f"k{r['n_matched']}", "amount_minor": -r["bank_fee_minor"], "memo": "SERVICE FEE"})
    if r["has_duplicate"]:
        book.append({"id": f"b{r['n_matched']}dup", "amount_minor": 10000, "memo": "possible duplicate"})
    if r["deposit_in_transit_minor"]:
        book.append({"id": "b_dit", "amount_minor": r["deposit_in_transit_minor"], "memo": "deposit in transit"})
    return {
        "book_balance_minor": r["book_balance_minor"],
        "book_entries": book,
        "bank_events": bank,
    }
