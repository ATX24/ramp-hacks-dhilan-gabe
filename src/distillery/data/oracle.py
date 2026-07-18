"""Executable latent oracle: labels come only from world state, never model output."""

from __future__ import annotations

from typing import Any, Literal

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import (
    CashReconciliationOutput,
    EvidenceRef,
    JournalLine,
    MatchedGroup,
    OracleMeta,
    ReconciliationException,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
    VarianceDriver,
)
from distillery.data.world import LatentWorld, VarianceRegime

GENERATOR_REVISION = "finance_world.v1"


def oracle_meta(world: LatentWorld) -> OracleMeta:
    return OracleMeta(
        generator_revision=GENERATOR_REVISION,
        latent_state_hash=world.latent_state_hash(),
    )


def solve_task(world: LatentWorld, task: TaskId) -> dict[str, Any]:
    """Return canonical expected_output dict for ``task`` from ``world``."""
    if task == TaskId.TRANSACTION_REVIEW:
        return solve_transaction_review(world).model_dump(mode="json")
    if task == TaskId.VARIANCE_ANALYSIS:
        return solve_variance_analysis(world).model_dump(mode="json")
    if task == TaskId.CASH_RECONCILIATION:
        return solve_cash_reconciliation(world).model_dump(mode="json")
    raise ValueError(f"oracle has no solver for task {task}")


def solve_transaction_review(world: LatentWorld) -> TransactionReviewOutput:
    txn = world.transaction
    if txn is None:
        raise ValueError("world missing transaction latent")

    amount = txn.amount_minor
    if txn.split_lines:
        journal = tuple(
            JournalLine(account=acct, side="debit", amount_minor=amt)
            for acct, amt in txn.split_lines
        ) + (JournalLine(account=txn.contra_account, side="credit", amount_minor=amount),)
        gl = txn.split_lines[0][0]
    else:
        journal = (
            JournalLine(account=txn.gl_account, side="debit", amount_minor=amount),
            JournalLine(account=txn.contra_account, side="credit", amount_minor=amount),
        )
        gl = txn.gl_account

    if txn.is_refund:
        # Refund flips sides; still balanced.
        journal = (
            JournalLine(account=txn.contra_account, side="debit", amount_minor=amount),
            JournalLine(account=gl, side="credit", amount_minor=amount),
        )

    evidence = (
        EvidenceRef(source_id="txn", field="amount_minor", value=str(amount)),
        EvidenceRef(source_id="txn", field="descriptor", value=txn.descriptor),
    )
    confidence = {
        "none": 0.97,
        "near_synonym_gl": 0.9,
        "refund": 0.88,
        "capex_opex": 0.86,
        "split_allocation": 0.84,
        "personal_looking_allowed": 0.87,
        "allowed_looking_prohibited": 0.85,
        "conflicting_rules": 0.82,
        "threshold_boundary": 0.89,
        "misleading_descriptor": 0.83,
    }.get(txn.hard_negative.value, 0.9)

    return TransactionReviewOutput(
        gl_account=gl,
        journal_entry=journal,
        policy_action=txn.policy_action.value,  # type: ignore[arg-type]
        rule_ids=txn.applied_rule_ids,
        evidence=evidence,
        confidence=confidence,
    )


def rank_drivers(
    drivers: list[tuple[str, int]],
) -> list[tuple[str, int, int]]:
    """Descending |impact|, tie-break by driver_id ascending; assign ranks 1..n."""
    ordered = sorted(drivers, key=lambda d: (-abs(d[1]), d[0]))
    return [(driver_id, impact, rank) for rank, (driver_id, impact) in enumerate(ordered, start=1)]


def solve_variance_analysis(world: LatentWorld) -> VarianceAnalysisOutput:
    var = world.variance
    if var is None:
        raise ValueError("world missing variance latent")

    ranked = rank_drivers([(d.driver_id, d.impact_minor) for d in var.drivers])
    top = tuple(
        VarianceDriver(driver_id=did, impact_minor=impact, rank=rank)
        for did, impact, rank in ranked
    )
    profit = sum(d.impact_minor for d in var.drivers) + var.other_impact_minor
    direction: Literal["favorable", "unfavorable"] = (
        "favorable" if profit >= 0 else "unfavorable"
    )
    evidence_ids = tuple(d.evidence_id for d in var.drivers)
    rule_ids = (var.materiality_rule_id,)
    if var.regime == VarianceRegime.FX:
        rule_ids = ("VAR-MATERIAL-005", "VAR-FX-002")
    elif var.regime == VarianceRegime.TIE:
        rule_ids = ("VAR-TIEBREAK-001",)

    confidence = 0.96 if var.regime == VarianceRegime.SIMPLE else 0.9
    if var.regime in {VarianceRegime.OFFSET, VarianceRegime.FX}:
        confidence = 0.86

    return VarianceAnalysisOutput(
        profit_impact_minor=profit,
        direction=direction,
        top_drivers=top,
        other_impact_minor=var.other_impact_minor,
        rule_ids=rule_ids,
        evidence_ids=evidence_ids,
        confidence=confidence,
    )


def solve_cash_reconciliation(world: LatentWorld) -> CashReconciliationOutput:
    cash = world.cash
    if cash is None:
        raise ValueError("world missing cash latent")

    matched = tuple(
        MatchedGroup(book_ids=books, bank_ids=banks) for books, banks in cash.matched
    )
    exceptions = tuple(
        ReconciliationException(
            type=exc_type,  # type: ignore[arg-type]
            event_ids=event_ids,
            amount_minor=amount,
        )
        for exc_type, event_ids, amount in cash.exceptions
    )

    adjusted_book, adjusted_bank = _adjust_balances(cash)
    difference = adjusted_book - adjusted_bank
    status: Literal["balanced", "exceptions"] = (
        "exceptions" if exceptions else "balanced"
    )
    confidence = 0.93 if status == "balanced" else 0.89

    return CashReconciliationOutput(
        status=status,
        matched_groups=matched,
        exceptions=exceptions,
        adjusted_book_balance_minor=adjusted_book,
        adjusted_bank_balance_minor=adjusted_bank,
        difference_minor=difference,
        confidence=confidence,
    )


def _adjust_balances(cash: Any) -> tuple[int, int]:
    """Apply standard reconciling items to book/bank raw balances."""
    book = cash.book_balance_minor
    bank = cash.bank_balance_minor
    for exc_type, _ids, amount in cash.exceptions:
        if exc_type == "bank_fee":
            bank += amount  # add back fee charged by bank
        elif exc_type == "deposit_in_transit":
            bank += amount
        elif exc_type == "stale_check":
            book += amount  # add back outstanding check
        elif exc_type == "duplicate":
            bank -= amount
        elif exc_type == "partial_settlement":
            # leave raw; difference reflects remaining gap unless already zeroed
            pass
        elif exc_type == "unexplained":
            pass
    return book, bank


def latent_hash_bytes(world: LatentWorld) -> str:
    return content_sha256(world.latent_payload())
