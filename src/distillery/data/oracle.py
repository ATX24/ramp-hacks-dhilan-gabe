"""Executable latent oracle: labels derive only from hidden world state."""

from __future__ import annotations

from typing import Any, Literal

from distillery.contracts.tasks import (
    CashReconciliationOutput,
    EvidenceRef,
    JournalLine,
    MatchedGroup,
    MerchantTaggingOutput,
    OracleMeta,
    ReconciliationException,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
    VarianceDriver,
)
from distillery.data.world import MCC_CATEGORY_MAP, CashLatent, LatentWorld

GENERATOR_REVISION_V1 = "finance_world.v1"
GENERATOR_REVISION_V2 = "finance_world.v2"
# Backward-compatible alias: default generator revision tracks the active
# finance_world.v1 smoke path until a campaign opts into v2.
GENERATOR_REVISION = GENERATOR_REVISION_V1


def oracle_meta(
    world: LatentWorld,
    *,
    generator_revision: str = GENERATOR_REVISION,
) -> OracleMeta:
    return OracleMeta(
        generator_revision=generator_revision,
        latent_state_hash=world.latent_state_hash(),
    )


def solve_task(world: LatentWorld, task: TaskId) -> dict[str, Any]:
    if task == TaskId.TRANSACTION_REVIEW:
        return solve_transaction_review(world).model_dump(mode="json")
    if task == TaskId.VARIANCE_ANALYSIS:
        return solve_variance_analysis(world).model_dump(mode="json")
    if task == TaskId.CASH_RECONCILIATION:
        return solve_cash_reconciliation(world).model_dump(mode="json")
    if task == TaskId.MERCHANT_TAGGING:
        return solve_merchant_tagging(world).model_dump(mode="json")
    raise ValueError(f"oracle has no solver for task {task}")


def solve_transaction_review(world: LatentWorld) -> TransactionReviewOutput:
    txn = world.transaction
    if txn is None:
        raise ValueError("world missing transaction latent")

    total = abs(txn.amount_minor)
    allocations = txn.split_lines or ((txn.gl_account, total),)
    if sum(amount for _, amount in allocations) != total:
        raise ValueError("transaction allocation does not equal absolute amount")

    if txn.amount_minor < 0:
        journal = (
            JournalLine(
                account=txn.contra_account,
                side="debit",
                amount_minor=total,
            ),
            *(
                JournalLine(account=account, side="credit", amount_minor=amount)
                for account, amount in allocations
            ),
        )
    else:
        journal = (
            *(
                JournalLine(account=account, side="debit", amount_minor=amount)
                for account, amount in allocations
            ),
            JournalLine(
                account=txn.contra_account,
                side="credit",
                amount_minor=total,
            ),
        )

    difficulty_confidence = {
        "none": 0.97,
        "near_synonym_gl": 0.91,
        "refund": 0.89,
        "chargeback": 0.87,
        "capex_opex": 0.86,
        "split_allocation": 0.85,
        "refund_split": 0.82,
        "personal_looking_allowed": 0.87,
        "allowed_looking_prohibited": 0.85,
        "conflicting_rules": 0.83,
        "threshold_boundary": 0.89,
        "misleading_descriptor": 0.84,
    }
    evidence = (
        EvidenceRef(
            source_id=txn.txn_id,
            field="amount_minor",
            value=str(txn.amount_minor),
        ),
        EvidenceRef(
            source_id=txn.txn_id,
            field="descriptor",
            value=txn.descriptor,
        ),
    )
    return TransactionReviewOutput(
        gl_account=allocations[0][0],
        journal_entry=journal,
        policy_action=txn.policy_action.value,  # type: ignore[arg-type]
        rule_ids=txn.applied_rule_ids,
        evidence=evidence,
        confidence=difficulty_confidence[txn.hard_negative.value],
    )


def rank_drivers(
    drivers: list[tuple[str, int]],
) -> list[tuple[str, int, int]]:
    """Descending absolute impact; stable tie-break by driver_id."""
    ordered = sorted(drivers, key=lambda driver: (-abs(driver[1]), driver[0]))
    return [(driver_id, impact, rank) for rank, (driver_id, impact) in enumerate(ordered, start=1)]


def solve_variance_analysis(world: LatentWorld) -> VarianceAnalysisOutput:
    variance = world.variance
    if variance is None:
        raise ValueError("world missing variance latent")

    impacts = [(driver.driver_id, driver.profit_impact_minor()) for driver in variance.drivers]
    ranked = rank_drivers(impacts)
    top = tuple(
        VarianceDriver(driver_id=driver_id, impact_minor=impact, rank=rank)
        for driver_id, impact, rank in ranked
    )
    other = sum(item.profit_impact_minor() for item in variance.unallocated)
    profit_impact = sum(impact for _, impact in impacts) + other
    pnl_delta = variance.budget_minor - variance.actual_minor
    if pnl_delta != profit_impact:
        raise ValueError(
            f"latent variance P&L mismatch: budget-actual={pnl_delta} oracle={profit_impact}"
        )
    direction: Literal["favorable", "unfavorable"] = (
        "favorable" if profit_impact >= 0 else "unfavorable"
    )
    return VarianceAnalysisOutput(
        profit_impact_minor=profit_impact,
        direction=direction,
        top_drivers=top,
        other_impact_minor=other,
        rule_ids=variance.rule_ids,
        evidence_ids=tuple(driver.source_id for driver in variance.drivers),
        confidence=0.96 if len(variance.drivers) == 1 else 0.88,
    )


def solve_cash_reconciliation(world: LatentWorld) -> CashReconciliationOutput:
    cash = world.cash
    if cash is None:
        raise ValueError("world missing cash latent")

    matched = tuple(
        MatchedGroup(book_ids=book_ids, bank_ids=bank_ids) for book_ids, bank_ids in cash.matched
    )
    exceptions = tuple(
        ReconciliationException(
            type=exception_type,  # type: ignore[arg-type]
            event_ids=event_ids,
            amount_minor=amount,
        )
        for exception_type, event_ids, amount in cash.exceptions
    )
    adjusted_book, adjusted_bank = adjust_cash_balances(
        cash.book_balance_minor,
        cash.bank_balance_minor,
        cash.exceptions,
    )
    difference = adjusted_book - adjusted_bank
    status: Literal["balanced", "exceptions"] = "exceptions" if exceptions else "balanced"
    return CashReconciliationOutput(
        status=status,
        matched_groups=matched,
        exceptions=exceptions,
        adjusted_book_balance_minor=adjusted_book,
        adjusted_bank_balance_minor=adjusted_bank,
        difference_minor=difference,
        confidence=0.93 if status == "balanced" else 0.89,
    )


def adjust_cash_balances(
    book_balance_minor: int,
    bank_balance_minor: int,
    exceptions: tuple[tuple[str, tuple[str, ...], int], ...],
) -> tuple[int, int]:
    book = book_balance_minor
    bank = bank_balance_minor
    for exception_type, _event_ids, amount in exceptions:
        if exception_type == "bank_fee":
            bank += amount
        elif exception_type == "deposit_in_transit":
            bank += amount
        elif exception_type == "stale_check":
            book += amount
        elif exception_type == "duplicate":
            bank -= amount
        elif exception_type == "partial_settlement":
            bank += amount
        elif exception_type == "unexplained":
            continue
        else:
            raise ValueError(f"unsupported reconciliation exception {exception_type}")
    return book, bank


def adjust_cash_from_output(
    cash_input: dict[str, Any],
    output: CashReconciliationOutput,
) -> tuple[int, int]:
    exceptions = tuple(
        (exception.type, exception.event_ids, exception.amount_minor)
        for exception in output.exceptions
    )
    return adjust_cash_balances(
        int(cash_input["book_balance_minor"]),
        int(cash_input["bank_balance_minor"]),
        exceptions,
    )


def cash_latent_adjusted(cash: CashLatent) -> tuple[int, int]:
    return adjust_cash_balances(
        cash.book_balance_minor,
        cash.bank_balance_minor,
        cash.exceptions,
    )


def solve_merchant_tagging(world: LatentWorld) -> MerchantTaggingOutput:
    merchant = world.merchant
    if merchant is None:
        raise ValueError("world missing merchant latent")
    expected_category = MCC_CATEGORY_MAP.get(merchant.mcc)
    if expected_category is None:
        raise ValueError(f"unknown MCC {merchant.mcc!r}")
    if merchant.spend_category != expected_category:
        raise ValueError(
            f"latent MCC/category mismatch: {merchant.mcc}->{merchant.spend_category}"
        )
    confidence = {
        "none": 0.97,
        "processor_prefix": 0.90,
        "truncated_descriptor": 0.88,
        "transposed_tokens": 0.87,
        "numeric_noise": 0.89,
        "lookalike_family": 0.84,
        "mcc_near_miss": 0.83,
        "category_collision": 0.82,
        "receipt_contradiction": 0.81,
    }[merchant.corruption_template.value]
    return MerchantTaggingOutput(
        merchant_id=merchant.merchant_id,
        merchant_name=merchant.merchant_name,
        spend_category=merchant.spend_category,
        tags=merchant.tags,
        confidence=confidence,
    )
