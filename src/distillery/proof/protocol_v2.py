"""Immutable finance-proof.v2 protocol definition (content-addressed).

finance-proof.v1 remains the active smoke/campaign protocol. This module
defines the next locked protocol for the three-primary TinyFable generalist
without mutating v1 hashes or gates.
"""

from __future__ import annotations

from typing import Any

from distillery.contracts.budgets import PRIMARY_INDEX_WEIGHTS_V2, EvaluationBudgetV2
from distillery.contracts.hashing import content_sha256

PROOF_PROTOCOL_ID_V1 = "finance-proof.v1"
PROOF_PROTOCOL_ID_V2 = "finance-proof.v2"

# Declared corpus / mixture for finance_world.v2 (see docs/decisions-finance-world-v2.md).
FINANCE_WORLD_V2_MIXTURE: dict[str, float] = {
    "transaction_review": 0.35,
    "variance_analysis": 0.35,
    "merchant_tagging": 0.20,
    "cash_reconciliation": 0.10,
}

FINANCE_WORLD_V2_FULL_SPLITS: dict[str, int] = {
    "train": 3840,
    "validation": 480,
    "iid_test": 960,
    "ood_test": 960,
}

FINANCE_WORLD_V2_SMOKE_SPLITS: dict[str, int] = {
    "train": 320,
    "validation": 80,
    "test": 160,
}


def finance_proof_v2_document() -> dict[str, Any]:
    """Canonical sealed protocol document for finance-proof.v2."""
    weights = PRIMARY_INDEX_WEIGHTS_V2
    eval_budget = EvaluationBudgetV2()
    return {
        "id": PROOF_PROTOCOL_ID_V2,
        "schema_version": "distillery.proof_protocol.v2",
        "finance_world": "finance_world.v2",
        "shared_model": "tinyfable_generalist",
        "specialist_routing": False,
        "primary_tasks": [
            "transaction_review",
            "variance_analysis",
            "merchant_tagging",
        ],
        "diagnostic_tasks": ["cash_reconciliation"],
        "task_mixture": dict(FINANCE_WORLD_V2_MIXTURE),
        "difficulty_mixture": {
            "easy": eval_budget.difficulty_easy,
            "medium": eval_budget.difficulty_medium,
            "hard": eval_budget.difficulty_hard,
        },
        "smoke_splits": dict(FINANCE_WORLD_V2_SMOKE_SPLITS),
        "full_splits": dict(FINANCE_WORLD_V2_FULL_SPLITS),
        "full_total_examples": sum(FINANCE_WORLD_V2_FULL_SPLITS.values()),
        "min_full_merchant_examples": 1000,
        "primary_index_weights": {
            "transaction_joint_exact": weights.transaction_joint_exact,
            "variance_joint_exact": weights.variance_joint_exact,
            "merchant_joint_exact": weights.merchant_joint_exact,
            "json_schema_validity": weights.json_schema_validity,
        },
        "cash_in_primary_index": False,
        "gates": {
            "teacher_gap_min_abs": 0.05,
            "quality_retention_point": 0.95,
            "quality_retention_lower_95": 0.90,
            "max_primary_task_regression": 0.05,
            "json_schema_validity_min": 0.99,
            "ood_retention_min": 0.90,
            "merchant_beats_fuzzy_baseline": True,
            "merchant_beats_frozen_student": True,
            "merchant_teacher_retention_min": 0.95,
        },
        "notes": (
            "One shared TinyFable artifact serves all primary tasks; "
            "cash_reconciliation remains diagnostic backup only."
        ),
    }


def finance_proof_v2_sha256() -> str:
    return content_sha256(finance_proof_v2_document())
