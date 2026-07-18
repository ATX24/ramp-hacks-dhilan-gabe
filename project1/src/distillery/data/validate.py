"""Deterministic validators for model/teacher outputs. Used both for synthesis
rejection and benchmark scoring."""
from __future__ import annotations

import json
from typing import Any

from .world import CHART_OF_ACCOUNTS, POLICIES

VALID_GL = {a for a, _, _ in CHART_OF_ACCOUNTS}
VALID_RULES = {pid for pid, _, _, _ in POLICIES} | {"VAR-MATERIAL-005", "POL-DEFAULT-000"}
VALID_ACTIONS = {"approve", "review", "reject"}


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def validate_transaction_review(out: dict, inp: dict) -> list[str]:
    errs = []
    if out.get("gl_account") not in VALID_GL:
        errs.append("gl_account not in chart of accounts")
    je = out.get("journal_entry") or []
    debits = sum(l.get("amount_minor", 0) for l in je if l.get("side") == "debit")
    credits = sum(l.get("amount_minor", 0) for l in je if l.get("side") == "credit")
    if debits != credits or debits == 0:
        errs.append("journal entry unbalanced")
    if any(not isinstance(l.get("amount_minor"), int) for l in je):
        errs.append("non-integer minor units")
    if out.get("policy_action") not in VALID_ACTIONS:
        errs.append("invalid policy_action")
    if not set(out.get("rule_ids") or []) <= VALID_RULES:
        errs.append("unknown rule id")
    return errs


def validate_variance_analysis(out: dict, inp: dict) -> list[str]:
    errs = []
    top = out.get("top_drivers") or []
    try:
        total = sum(d["impact_minor"] for d in top) + out.get("other_impact_minor", 0)
        if total != out.get("profit_impact_minor"):
            errs.append("arithmetic closure violated")
    except (KeyError, TypeError):
        errs.append("malformed drivers")
    pi = out.get("profit_impact_minor")
    if isinstance(pi, int):
        want = "favorable" if pi >= 0 else "unfavorable"
        if out.get("direction") != want:
            errs.append("direction/sign mismatch")
    return errs


def validate_cash_reconciliation(out: dict, inp: dict) -> list[str]:
    errs = []
    try:
        if out["adjusted_book_balance_minor"] - out["adjusted_bank_balance_minor"] != out["difference_minor"]:
            errs.append("balance difference mismatch")
    except (KeyError, TypeError):
        errs.append("missing balance fields")
    return errs


VALIDATORS = {
    "transaction_review": validate_transaction_review,
    "variance_analysis": validate_variance_analysis,
    "cash_reconciliation": validate_cash_reconciliation,
}


def validate_output(task: str, text: str, inp: dict) -> tuple[dict | None, list[str]]:
    obj = parse_json(text)
    if obj is None:
        return None, ["invalid JSON"]
    if obj.get("task") != task:
        return obj, ["wrong task field"]
    return obj, VALIDATORS[task](obj, inp)
