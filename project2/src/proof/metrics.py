"""Deterministic evaluation metrics for the merchant-normalization task."""
from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = ("merchant", "category", "is_subscription", "policy_flags")

CATEGORIES = {
    "software", "travel", "meals", "office", "advertising",
    "infrastructure", "professional_services", "other",
}


def schema_valid(out: Any) -> bool:
    if not isinstance(out, dict):
        return False
    if set(out.keys()) != set(REQUIRED_FIELDS):
        return False
    return (
        isinstance(out["merchant"], str)
        and out["category"] in CATEGORIES
        and isinstance(out["is_subscription"], bool)
        and isinstance(out["policy_flags"], list)
        and all(isinstance(f, str) for f in out["policy_flags"])
    )


def field_accuracy(pred: dict | None, gold: dict) -> float:
    """Fraction of the four fields matched exactly (flags compared as sets)."""
    if not pred or not schema_valid(pred):
        return 0.0
    hits = 0
    for f in REQUIRED_FIELDS:
        if f == "policy_flags":
            hits += set(pred[f]) == set(gold[f])
        else:
            hits += pred[f] == gold[f]
    return hits / len(REQUIRED_FIELDS)


def aggregate(preds: list[dict | None], golds: list[dict]) -> dict:
    accs = [field_accuracy(p, g) for p, g in zip(preds, golds)]
    valid = [schema_valid(p) for p in preds]
    n = max(1, len(accs))
    return {
        "field_accuracy": round(sum(accs) / n, 4),
        "schema_validity": round(sum(valid) / n, 4),
        "n": len(accs),
    }
