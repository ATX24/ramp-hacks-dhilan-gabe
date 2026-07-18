"""Accounting invariants and executable task-output coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.tasks import (
    CashReconciliationOutput,
    FinanceTaskEnvelope,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
    validate_task_output,
)

HASH64 = "a" * 64


def _variance(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "variance_analysis.v1",
        "task": "variance_analysis",
        "profit_impact_minor": -150,
        "direction": "unfavorable",
        "top_drivers": [
            {"driver_id": "alpha", "impact_minor": -100, "rank": 1},
            {"driver_id": "beta", "impact_minor": -50, "rank": 2},
        ],
        "other_impact_minor": 0,
        "rule_ids": ["VAR-1"],
        "evidence_ids": ["actual"],
        "confidence": 0.9,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("drivers", "message"),
    [
        (
            [
                {"driver_id": "alpha", "impact_minor": -100, "rank": 1},
                {"driver_id": "beta", "impact_minor": -50, "rank": 1},
            ],
            "unique, contiguous",
        ),
        (
            [
                {"driver_id": "alpha", "impact_minor": -100, "rank": 1},
                {"driver_id": "beta", "impact_minor": -50, "rank": 3},
            ],
            "unique, contiguous",
        ),
        (
            [
                {"driver_id": "alpha", "impact_minor": -50, "rank": 1},
                {"driver_id": "beta", "impact_minor": -100, "rank": 2},
            ],
            "descending absolute impact",
        ),
        (
            [
                {"driver_id": "beta", "impact_minor": -100, "rank": 1},
                {"driver_id": "alpha", "impact_minor": -100, "rank": 2},
            ],
            "ascending driver_id tie-break",
        ),
        (
            [
                {"driver_id": "alpha", "impact_minor": -100, "rank": 1},
                {"driver_id": "alpha", "impact_minor": -50, "rank": 2},
            ],
            "driver_id values must be unique",
        ),
    ],
)
def test_variance_rank_and_order_invariants(
    drivers: list[dict[str, object]],
    message: str,
) -> None:
    profit_impact = sum(int(driver["impact_minor"]) for driver in drivers)
    with pytest.raises(ValidationError, match=message):
        VarianceAnalysisOutput.model_validate(
            _variance(top_drivers=drivers, profit_impact_minor=profit_impact)
        )


def test_variance_closure_and_direction_invariants() -> None:
    with pytest.raises(ValidationError, match="arithmetic not closed"):
        VarianceAnalysisOutput.model_validate(_variance(profit_impact_minor=-151))
    with pytest.raises(ValidationError, match="inconsistent"):
        VarianceAnalysisOutput.model_validate(_variance(direction="favorable"))
    with pytest.raises(ValidationError, match="cannot be zero"):
        VarianceAnalysisOutput.model_validate(
            _variance(
                profit_impact_minor=0,
                top_drivers=[{"driver_id": "alpha", "impact_minor": 0, "rank": 1}],
            )
        )


def test_journal_requires_positive_debit_and_credit() -> None:
    base = {
        "gl_account": "6100",
        "policy_action": "approve",
        "rule_ids": ["POL-1"],
        "evidence": [],
        "confidence": 0.9,
    }
    with pytest.raises(ValidationError, match="positive debit"):
        TransactionReviewOutput.model_validate(
            {
                **base,
                "journal_entry": [
                    {"account": "6100", "side": "debit", "amount_minor": 0},
                    {"account": "2100", "side": "credit", "amount_minor": 0},
                ],
            }
        )
    with pytest.raises(ValidationError, match="positive credit"):
        TransactionReviewOutput.model_validate(
            {
                **base,
                "journal_entry": [
                    {"account": "6100", "side": "debit", "amount_minor": 100},
                    {"account": "2100", "side": "credit", "amount_minor": 0},
                    {"account": "1000", "side": "credit", "amount_minor": 0},
                    {"account": "1000", "side": "debit", "amount_minor": 100},
                ],
            }
        )


def _cash(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "cash_reconciliation.v1",
        "task": "cash_reconciliation",
        "status": "balanced",
        "matched_groups": [{"book_ids": ["b1"], "bank_ids": ["k1"]}],
        "exceptions": [],
        "adjusted_book_balance_minor": 100,
        "adjusted_bank_balance_minor": 100,
        "difference_minor": 0,
        "confidence": 0.9,
    }
    payload.update(updates)
    return payload


def test_cash_status_exception_consistency() -> None:
    exception = {
        "type": "bank_fee",
        "event_ids": ["k2"],
        "amount_minor": 5,
    }
    with pytest.raises(ValidationError, match="balanced reconciliation cannot"):
        CashReconciliationOutput.model_validate(_cash(exceptions=[exception]))
    with pytest.raises(ValidationError, match="requires at least one exception"):
        CashReconciliationOutput.model_validate(_cash(status="exceptions"))
    with pytest.raises(ValidationError, match="must have zero difference"):
        CashReconciliationOutput.model_validate(
            _cash(
                adjusted_book_balance_minor=101,
                adjusted_bank_balance_minor=100,
                difference_minor=1,
            )
        )


def test_task_envelope_task_must_match_output_discriminator() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        FinanceTaskEnvelope(
            example_id="ex_task_mismatch",
            world_id="world_task_mismatch",
            group_id="grp_task_mismatch",
            task="transaction_review",
            difficulty="easy",
            input={"period": "2026-Q1"},
            expected_output=_variance(),
            oracle={
                "generator_revision": "contracts-v1",
                "latent_state_hash": f"sha256:{HASH64}",
            },
            provenance={
                "split": "train",
                "template_family": "mismatch_v1",
                "label_source": "oracle",
            },
        )


def test_every_executable_task_has_output_contract() -> None:
    assert {task.value for task in TaskId} == {
        "transaction_review",
        "variance_analysis",
        "cash_reconciliation",
    }
    with pytest.raises(ValueError):
        TaskId("merchant_tagging")
    with pytest.raises(ValidationError):
        validate_task_output(
            {
                "schema_version": "merchant_tagging.v1",
                "task": "merchant_tagging",
            }
        )
