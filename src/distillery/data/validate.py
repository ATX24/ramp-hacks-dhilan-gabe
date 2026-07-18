"""Executable validators for accounting identities and task invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from distillery.contracts.tasks import (
    CashReconciliationOutput,
    FinanceTaskEnvelope,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
)
from distillery.data.oracle import rank_drivers


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def validate_output(task: TaskId, output: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    checks: list[str] = []

    if task == TaskId.TRANSACTION_REVIEW:
        errors.extend(_validate_transaction_review(output, checks))
    elif task == TaskId.VARIANCE_ANALYSIS:
        errors.extend(_validate_variance_analysis(output, checks))
    elif task == TaskId.CASH_RECONCILIATION:
        errors.extend(_validate_cash_reconciliation(output, checks))
    else:
        errors.append(f"unsupported task {task}")

    return ValidationResult(ok=not errors, errors=tuple(errors), checks=tuple(checks))


def validate_example(example: FinanceTaskEnvelope | dict[str, Any]) -> ValidationResult:
    if isinstance(example, dict):
        try:
            envelope = FinanceTaskEnvelope.model_validate(example)
        except ValidationError as exc:
            return ValidationResult(ok=False, errors=(f"envelope_schema:{exc}",))
    else:
        envelope = example

    result = validate_output(envelope.task, envelope.expected_output)
    extra: list[str] = list(result.errors)
    checks = list(result.checks)

    if not envelope.oracle.latent_state_hash.startswith("sha256:"):
        extra.append("oracle.latent_state_hash missing sha256 prefix")
    else:
        checks.append("oracle_hash_prefix")

    if envelope.task == TaskId.TRANSACTION_REVIEW:
        extra.extend(_validate_txn_evidence_grounding(envelope.input, envelope.expected_output))
        checks.append("evidence_grounding")

    return ValidationResult(ok=not extra, errors=tuple(extra), checks=tuple(checks))


def _validate_transaction_review(output: dict[str, Any], checks: list[str]) -> list[str]:
    errors: list[str] = []
    try:
        typed = TransactionReviewOutput.model_validate(output)
    except ValidationError as exc:
        return [f"typed_output:{exc}"]

    checks.append("typed_transaction_review")

    debits = sum(line.amount_minor for line in typed.journal_entry if line.side == "debit")
    credits = sum(
        line.amount_minor for line in typed.journal_entry if line.side == "credit"
    )
    if debits != credits:
        errors.append(f"unbalanced_journal debits={debits} credits={credits}")
    else:
        checks.append("debit_credit_identity")

    for line in typed.journal_entry:
        if line.amount_minor < 0:
            errors.append(f"negative_minor_unit account={line.account}")
        if not isinstance(line.amount_minor, int):
            errors.append(f"non_integer_minor_unit account={line.account}")
    checks.append("currency_minor_units")

    if typed.policy_action not in {"approve", "review", "reject"}:
        errors.append(f"invalid_policy_action={typed.policy_action}")
    else:
        checks.append("policy_action_enum")

    if not typed.rule_ids:
        errors.append("missing_rule_ids")
    else:
        checks.append("rule_ids_present")

    gl_in_journal = typed.gl_account in {line.account for line in typed.journal_entry}
    if not gl_in_journal:
        errors.append("gl_account_not_in_journal")
    else:
        checks.append("gl_in_journal")

    return errors


def _validate_variance_analysis(output: dict[str, Any], checks: list[str]) -> list[str]:
    errors: list[str] = []
    try:
        typed = VarianceAnalysisOutput.model_validate(output)
    except ValidationError as exc:
        return [f"typed_output:{exc}"]

    checks.append("typed_variance_analysis")

    driver_sum = sum(d.impact_minor for d in typed.top_drivers)
    if driver_sum + typed.other_impact_minor != typed.profit_impact_minor:
        errors.append("variance_arithmetic_not_closed")
    else:
        checks.append("variance_closure")

    expected_dir = "favorable" if typed.profit_impact_minor >= 0 else "unfavorable"
    if typed.direction != expected_dir:
        errors.append("direction_sign_inconsistency")
    else:
        checks.append("direction_sign")

    pairs = [(d.driver_id, d.impact_minor) for d in typed.top_drivers]
    expected_ranks = rank_drivers(pairs)
    actual = [(d.driver_id, d.impact_minor, d.rank) for d in typed.top_drivers]
    if actual != expected_ranks:
        errors.append(
            f"driver_ranking_incorrect actual={actual} expected={expected_ranks}"
        )
    else:
        checks.append("driver_ranking")

    ranks = [d.rank for d in typed.top_drivers]
    if ranks != list(range(1, len(ranks) + 1)):
        errors.append("driver_ranks_not_contiguous")
    else:
        checks.append("driver_ranks_contiguous")

    return errors


def _validate_cash_reconciliation(output: dict[str, Any], checks: list[str]) -> list[str]:
    errors: list[str] = []
    try:
        typed = CashReconciliationOutput.model_validate(output)
    except ValidationError as exc:
        return [f"typed_output:{exc}"]

    checks.append("typed_cash_reconciliation")

    expected_diff = typed.adjusted_book_balance_minor - typed.adjusted_bank_balance_minor
    if typed.difference_minor != expected_diff:
        errors.append("difference_mismatch")
    else:
        checks.append("difference_identity")

    if typed.status == "balanced" and typed.exceptions:
        errors.append("balanced_with_exceptions")
    if typed.status == "exceptions" and not typed.exceptions:
        errors.append("exceptions_status_without_exceptions")
    checks.append("status_consistency")

    seen_book: set[str] = set()
    seen_bank: set[str] = set()
    for group in typed.matched_groups:
        for bid in group.book_ids:
            if bid in seen_book:
                errors.append(f"duplicate_book_match:{bid}")
            seen_book.add(bid)
        for kid in group.bank_ids:
            if kid in seen_bank:
                errors.append(f"duplicate_bank_match:{kid}")
            seen_bank.add(kid)
    checks.append("match_uniqueness")

    return errors


def _validate_txn_evidence_grounding(
    inp: dict[str, Any],
    output: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    evidence = output.get("evidence") or []
    for item in evidence:
        field = item.get("field")
        value = item.get("value")
        if field is None or value is None:
            errors.append("evidence_missing_field_or_value")
            continue
        if field in inp and str(inp[field]) != str(value):
            # Allow evidence from nested structures via string containment.
            if str(value) not in str(inp):
                errors.append(f"evidence_not_grounded field={field} value={value}")
    return errors


def assert_policy_precedence(
    matched_rule_ids: list[str],
    precedence_by_id: dict[str, int],
    winner_id: str,
) -> ValidationResult:
    """Verify winner is the lowest-precedence matched rule (tie: rule_id asc)."""
    if winner_id not in matched_rule_ids:
        return ValidationResult(ok=False, errors=("winner_not_in_matches",))
    ordered = sorted(matched_rule_ids, key=lambda rid: (precedence_by_id[rid], rid))
    if ordered[0] != winner_id:
        return ValidationResult(
            ok=False,
            errors=(f"precedence_violation expected={ordered[0]} got={winner_id}",),
        )
    return ValidationResult(ok=True, checks=("policy_precedence",))


@dataclass
class MutationProbe:
    """Helper for leakage mutation tests."""

    name: str
    errors: list[str] = field(default_factory=list)
