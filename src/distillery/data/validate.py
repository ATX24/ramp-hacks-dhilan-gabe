"""Executable accounting, policy, grounding, and input-hygiene validators."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from distillery.contracts.tasks import (
    CashReconciliationOutput,
    FinanceTaskEnvelope,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
)
from distillery.data.oracle import adjust_cash_from_output, rank_drivers
from distillery.data.world import PolicyAction, PolicyRule, resolve_policy

FORBIDDEN_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "answer",
        "case_nonce",
        "expected_output",
        "gl_candidates",
        "hard_negative",
        "impact_hint_minor",
        "is_refund_hint",
        "label",
        "ood",
        "other_bucket_present",
        "predicted_output",
        "profit_impact_minor",
        "regime",
        "split",
        "target",
        "target_output",
        "template_family",
        "top_drivers",
    }
)
TARGET_HELPER_FRAGMENTS = (
    "answer",
    "candidate",
    "expected",
    "hint",
    "label",
    "prediction",
    "target",
)
_SPLIT_MARKER_RE = re.compile(
    r"(?i)(?:\b(?:train|training|validation|iid[_ -]?test|ood|ood[_ -]?test)\b|"
    r"(?:smk|full)_(?:tr|va|te|iid|ood)\b|"
    r"\b(?:held[- ]out|out[- ]of[- ]distribution)\b)"
)


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
        _parse_transaction(output, errors, checks)
    elif task == TaskId.VARIANCE_ANALYSIS:
        typed = _parse_variance(output, errors, checks)
        if typed is not None:
            _validate_variance_shape(typed, errors, checks)
    elif task == TaskId.CASH_RECONCILIATION:
        typed_cash = _parse_cash(output, errors, checks)
        if typed_cash is not None:
            _validate_cash_shape(typed_cash, errors, checks)
    else:
        errors.append(f"unsupported_task:{task}")
    return ValidationResult(not errors, tuple(errors), tuple(checks))


def validate_example(
    example: FinanceTaskEnvelope | dict[str, Any],
) -> ValidationResult:
    if isinstance(example, Mapping):
        try:
            envelope = FinanceTaskEnvelope.model_validate(example)
        except ValidationError as exc:
            return ValidationResult(False, (f"envelope_schema:{exc}",))
    else:
        envelope = example

    base = validate_output(envelope.task, envelope.expected_output)
    errors = list(base.errors)
    checks = list(base.checks)
    if envelope.oracle.latent_state_hash.startswith("sha256:"):
        checks.append("oracle_hash_prefix")
    else:
        errors.append("oracle_hash_prefix")

    hygiene = find_input_hygiene_errors(
        envelope.input,
        expected_output=envelope.expected_output,
    )
    errors.extend(hygiene)
    if not hygiene:
        checks.append("input_hygiene")

    if envelope.task == TaskId.TRANSACTION_REVIEW:
        _validate_transaction_context(envelope, errors, checks)
    elif envelope.task == TaskId.VARIANCE_ANALYSIS:
        _validate_variance_context(envelope, errors, checks)
    elif envelope.task == TaskId.CASH_RECONCILIATION:
        _validate_cash_context(envelope, errors, checks)
    return ValidationResult(not errors, tuple(errors), tuple(checks))


def find_input_hygiene_errors(
    finance_input: dict[str, Any],
    *,
    expected_output: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Detect answer helpers, split/OOD markers, and suspicious target copies."""
    errors: list[str] = []
    target_values = (
        _sensitive_target_values(expected_output) if expected_output is not None else set()
    )
    for path, value in _walk(finance_input):
        key = path[-1].casefold() if path else ""
        dotted = ".".join(path)
        if key in FORBIDDEN_INPUT_KEYS:
            errors.append(f"forbidden_input_field:{dotted}")
        if isinstance(value, str) and _SPLIT_MARKER_RE.search(value):
            errors.append(f"split_or_ood_marker:{dotted}")
        if any(fragment in key for fragment in TARGET_HELPER_FRAGMENTS):
            errors.append(f"target_helper_field:{dotted}")
            if _scalar_key(value) in target_values:
                errors.append(f"direct_target_copy:{dotted}")
    return tuple(sorted(set(errors)))


def assert_policy_precedence(
    matched_rule_ids: list[str],
    precedence_by_id: dict[str, int],
    winner_id: str,
) -> ValidationResult:
    if winner_id not in matched_rule_ids:
        return ValidationResult(False, ("winner_not_in_matches",))
    ordered = sorted(matched_rule_ids, key=lambda rid: (precedence_by_id[rid], rid))
    if ordered[0] != winner_id:
        return ValidationResult(
            False,
            (f"precedence_violation:expected={ordered[0]}:actual={winner_id}",),
        )
    return ValidationResult(True, checks=("policy_precedence",))


def _validate_transaction_context(
    envelope: FinanceTaskEnvelope,
    errors: list[str],
    checks: list[str],
) -> None:
    try:
        output = TransactionReviewOutput.model_validate(envelope.expected_output)
    except ValidationError:
        return
    finance_input = envelope.input
    chart = finance_input.get("chart_of_accounts")
    if not isinstance(chart, Sequence):
        errors.append("transaction_missing_chart")
        return
    allowed = {
        str(account.get("code"))
        for account in chart
        if isinstance(account, Mapping) and account.get("code")
    }
    if output.gl_account not in allowed:
        errors.append(f"unknown_gl_account:{output.gl_account}")
    for line in output.journal_entry:
        if line.account not in allowed:
            errors.append(f"unknown_journal_account:{line.account}")
    if output.gl_account in allowed and all(
        line.account in allowed for line in output.journal_entry
    ):
        checks.append("allowed_accounts")

    amount = finance_input.get("amount_minor")
    if type(amount) is not int:
        errors.append("transaction_amount_not_integer_minor_units")
        return
    absolute_amount = abs(amount)
    debits = sum(line.amount_minor for line in output.journal_entry if line.side == "debit")
    credits = sum(line.amount_minor for line in output.journal_entry if line.side == "credit")
    if debits != absolute_amount or credits != absolute_amount:
        errors.append(f"journal_amount_mismatch:{debits}:{credits}:{absolute_amount}")
    else:
        checks.append("journal_amount_identity")
    expense_side = "credit" if amount < 0 else "debit"
    expense_lines = [line for line in output.journal_entry if line.account != "2100"]
    if any(line.side != expense_side for line in expense_lines):
        errors.append("refund_or_charge_side_incorrect")
    else:
        checks.append("signed_journal_direction")

    rules = _parse_policy_rules(finance_input, errors)
    rule_by_id = {rule.rule_id: rule for rule in rules}
    missing = sorted(set(output.rule_ids) - set(rule_by_id))
    if missing:
        errors.append(f"unknown_policy_rule:{','.join(missing)}")
    excerpt = finance_input.get("policy_excerpt")
    if not isinstance(excerpt, str):
        errors.append("policy_excerpt_missing")
    else:
        for rule_id in output.rule_ids:
            if rule_id not in excerpt:
                errors.append(f"cited_rule_missing_from_excerpt:{rule_id}")

    if rules:
        try:
            expected_action, expected_gl, expected_rules = resolve_policy(
                rules,
                descriptor=str(finance_input.get("descriptor", "")),
                amount_minor=amount,
                category=str(finance_input.get("expense_category", "")),
                vendor_id=str(finance_input.get("vendor_id", "")),
            )
        except ValueError as exc:
            errors.append(f"policy_resolution:{exc}")
        else:
            if output.policy_action != expected_action.value:
                errors.append(
                    f"policy_action_inconsistent:{expected_action.value}:{output.policy_action}"
                )
            if output.gl_account != expected_gl:
                errors.append(f"policy_gl_inconsistent:{expected_gl}:{output.gl_account}")
            if tuple(output.rule_ids) != expected_rules:
                errors.append(f"policy_rule_inconsistent:{expected_rules}:{tuple(output.rule_ids)}")
            if (
                output.policy_action == expected_action.value
                and output.gl_account == expected_gl
                and tuple(output.rule_ids) == expected_rules
            ):
                checks.append("policy_applicability_and_precedence")

    txn_id = finance_input.get("txn_id")
    evidence_errors = 0
    for evidence in output.evidence:
        if evidence.source_id != txn_id:
            errors.append(f"invalid_evidence_source:{evidence.source_id}")
            evidence_errors += 1
            continue
        if evidence.field not in finance_input:
            errors.append(f"invalid_evidence_field:{evidence.field}")
            evidence_errors += 1
            continue
        if str(finance_input[evidence.field]) != evidence.value:
            errors.append(f"evidence_value_mismatch:{evidence.field}")
            evidence_errors += 1
    if evidence_errors == 0:
        checks.append("exact_evidence_grounding")


def _validate_variance_context(
    envelope: FinanceTaskEnvelope,
    errors: list[str],
    checks: list[str],
) -> None:
    try:
        output = VarianceAnalysisOutput.model_validate(envelope.expected_output)
    except ValidationError:
        return
    finance_input = envelope.input
    budget = finance_input.get("budget_minor")
    actual = finance_input.get("actual_minor")
    if type(budget) is not int or type(actual) is not int:
        errors.append("variance_budget_actual_not_integer_minor_units")
        return
    if budget - actual != output.profit_impact_minor:
        errors.append(f"variance_pnl_mismatch:{budget - actual}:{output.profit_impact_minor}")
    else:
        checks.append("variance_pnl_identity")

    observations = finance_input.get("driver_observations")
    if not isinstance(observations, Sequence):
        errors.append("variance_observations_missing")
        return
    by_driver: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    for item in observations:
        if not isinstance(item, Mapping):
            errors.append("variance_observation_not_object")
            continue
        driver_id = str(item.get("driver_id", ""))
        source_id = str(item.get("source_id", ""))
        if not driver_id or not source_id:
            errors.append("variance_observation_missing_ids")
            continue
        by_driver[driver_id] = item
        source_ids.add(source_id)
    if set(output.evidence_ids) != source_ids:
        errors.append(
            f"variance_evidence_ids_invalid:{sorted(source_ids)}:{sorted(output.evidence_ids)}"
        )
    else:
        checks.append("variance_evidence_ids")

    driver_errors = 0
    for driver in output.top_drivers:
        item = by_driver.get(driver.driver_id)
        if item is None:
            errors.append(f"variance_driver_missing:{driver.driver_id}")
            driver_errors += 1
            continue
        expected = _observation_impact(item, errors)
        if expected is not None and driver.impact_minor != expected:
            errors.append(
                f"variance_driver_impact_mismatch:{driver.driver_id}:"
                f"{expected}:{driver.impact_minor}"
            )
            driver_errors += 1
    if driver_errors == 0:
        checks.append("variance_driver_derivation")

    unallocated = finance_input.get("unallocated_line_items", [])
    if not isinstance(unallocated, Sequence):
        errors.append("variance_unallocated_not_list")
        unallocated = []
    expected_other = 0
    for item in unallocated:
        if not isinstance(item, Mapping):
            errors.append("variance_unallocated_not_object")
            continue
        impact = _observation_impact(item, errors)
        if impact is not None:
            expected_other += impact
    if output.other_impact_minor != expected_other:
        errors.append(f"variance_other_mismatch:{expected_other}:{output.other_impact_minor}")

    valid_rules = {
        str(rule.get("rule_id"))
        for rule in finance_input.get("analysis_rules", [])
        if isinstance(rule, Mapping) and rule.get("rule_id")
    }
    missing_rules = sorted(set(output.rule_ids) - valid_rules)
    if missing_rules:
        errors.append(f"variance_rule_ids_invalid:{missing_rules}")


def _validate_cash_context(
    envelope: FinanceTaskEnvelope,
    errors: list[str],
    checks: list[str],
) -> None:
    try:
        output = CashReconciliationOutput.model_validate(envelope.expected_output)
    except ValidationError:
        return
    finance_input = envelope.input
    book_items = finance_input.get("book_entries")
    bank_items = finance_input.get("bank_events")
    if not isinstance(book_items, Sequence) or not isinstance(bank_items, Sequence):
        errors.append("cash_source_lists_missing")
        return
    book = {
        str(item.get("id")): item
        for item in book_items
        if isinstance(item, Mapping) and item.get("id")
    }
    bank = {
        str(item.get("id")): item
        for item in bank_items
        if isinstance(item, Mapping) and item.get("id")
    }
    seen_book: set[str] = set()
    seen_bank: set[str] = set()
    for group in output.matched_groups:
        if not group.book_ids or not group.bank_ids:
            errors.append("cash_empty_matched_group")
        unknown_book = set(group.book_ids) - set(book)
        unknown_bank = set(group.bank_ids) - set(bank)
        if unknown_book:
            errors.append(f"cash_unknown_book_ids:{sorted(unknown_book)}")
        if unknown_bank:
            errors.append(f"cash_unknown_bank_ids:{sorted(unknown_bank)}")
        if set(group.book_ids) & seen_book:
            errors.append("cash_duplicate_book_match")
        if set(group.bank_ids) & seen_bank:
            errors.append("cash_duplicate_bank_match")
        seen_book.update(group.book_ids)
        seen_bank.update(group.bank_ids)
        if not unknown_book and not unknown_bank:
            book_sum = sum(int(book[item]["amount_minor"]) for item in group.book_ids)
            bank_sum = sum(int(bank[item]["amount_minor"]) for item in group.bank_ids)
            if book_sum != bank_sum:
                errors.append(f"cash_match_amount_mismatch:{book_sum}:{bank_sum}")

    source_ids = set(book) | set(bank)
    for exception in output.exceptions:
        unknown = set(exception.event_ids) - source_ids
        if unknown:
            errors.append(f"cash_exception_unknown_ids:{sorted(unknown)}")
    adjusted_book, adjusted_bank = adjust_cash_from_output(finance_input, output)
    if output.adjusted_book_balance_minor != adjusted_book:
        errors.append(
            f"cash_adjusted_book_mismatch:{adjusted_book}:{output.adjusted_book_balance_minor}"
        )
    if output.adjusted_bank_balance_minor != adjusted_bank:
        errors.append(
            f"cash_adjusted_bank_mismatch:{adjusted_bank}:{output.adjusted_bank_balance_minor}"
        )
    if not any(error.startswith("cash_") for error in errors):
        checks.append("cash_source_and_reconciliation")


def _parse_transaction(
    output: dict[str, Any],
    errors: list[str],
    checks: list[str],
) -> TransactionReviewOutput | None:
    try:
        typed = TransactionReviewOutput.model_validate(output)
    except ValidationError as exc:
        errors.append(f"typed_output:{exc}")
        return None
    debits = sum(line.amount_minor for line in typed.journal_entry if line.side == "debit")
    credits = sum(line.amount_minor for line in typed.journal_entry if line.side == "credit")
    if debits != credits:
        errors.append(f"unbalanced_journal:{debits}:{credits}")
    if typed.gl_account not in {line.account for line in typed.journal_entry}:
        errors.append("gl_account_not_in_journal")
    if not typed.rule_ids:
        errors.append("missing_rule_ids")
    if not errors:
        checks.extend(("typed_transaction_review", "debit_credit_identity", "minor_units"))
    return typed


def _parse_variance(
    output: dict[str, Any],
    errors: list[str],
    checks: list[str],
) -> VarianceAnalysisOutput | None:
    try:
        typed = VarianceAnalysisOutput.model_validate(output)
    except ValidationError as exc:
        errors.append(f"typed_output:{exc}")
        return None
    checks.append("typed_variance_analysis")
    return typed


def _validate_variance_shape(
    typed: VarianceAnalysisOutput,
    errors: list[str],
    checks: list[str],
) -> None:
    expected_direction = "favorable" if typed.profit_impact_minor >= 0 else "unfavorable"
    if typed.direction != expected_direction:
        errors.append("direction_sign_inconsistency")
    expected = rank_drivers(
        [(driver.driver_id, driver.impact_minor) for driver in typed.top_drivers]
    )
    actual = [(driver.driver_id, driver.impact_minor, driver.rank) for driver in typed.top_drivers]
    if actual != expected:
        errors.append(f"driver_ranking_incorrect:{actual}:{expected}")
    if not errors:
        checks.extend(("variance_closure", "driver_ranking"))


def _parse_cash(
    output: dict[str, Any],
    errors: list[str],
    checks: list[str],
) -> CashReconciliationOutput | None:
    try:
        typed = CashReconciliationOutput.model_validate(output)
    except ValidationError as exc:
        errors.append(f"typed_output:{exc}")
        return None
    checks.append("typed_cash_reconciliation")
    return typed


def _validate_cash_shape(
    typed: CashReconciliationOutput,
    errors: list[str],
    checks: list[str],
) -> None:
    expected = typed.adjusted_book_balance_minor - typed.adjusted_bank_balance_minor
    if typed.difference_minor != expected:
        errors.append("difference_mismatch")
    if typed.status == "balanced" and typed.exceptions:
        errors.append("balanced_with_exceptions")
    if typed.status == "exceptions" and not typed.exceptions:
        errors.append("exceptions_status_without_exceptions")
    if not errors:
        checks.append("difference_identity")


def _parse_policy_rules(
    finance_input: dict[str, Any],
    errors: list[str],
) -> tuple[PolicyRule, ...]:
    raw = finance_input.get("policy_rules")
    if not isinstance(raw, Sequence):
        errors.append("policy_rules_missing")
        return ()
    rules: list[PolicyRule] = []
    for item in raw:
        if not isinstance(item, Mapping):
            errors.append("policy_rule_not_object")
            continue
        try:
            action = PolicyAction(str(item["action"]))
            max_amount = item.get("max_amount_minor")
            rules.append(
                PolicyRule(
                    rule_id=str(item["rule_id"]),
                    semantic_code=str(item["rule_id"]),
                    precedence=int(item["precedence"]),
                    action=action,
                    gl_account=str(item["gl_account"]),
                    min_amount_minor=int(item["min_amount_minor"]),
                    max_amount_minor=(None if max_amount is None else int(max_amount)),
                    keywords=tuple(str(value) for value in item["keywords"]),
                    category=str(item["category"]),
                    vendor_ids=tuple(str(value) for value in item["vendor_ids"]),
                    text=str(item["text"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid_policy_rule:{exc}")
    return tuple(rules)


def _observation_impact(
    item: dict[str, Any],
    errors: list[str],
) -> int | None:
    budget = item.get("budget_minor")
    actual = item.get("actual_minor")
    pnl_type = item.get("pnl_type")
    if type(budget) is not int or type(actual) is not int:
        errors.append("variance_observation_non_integer_minor_units")
        return None
    if pnl_type == "expense":
        return budget - actual
    if pnl_type == "revenue":
        return actual - budget
    errors.append(f"variance_invalid_pnl_type:{pnl_type}")
    return None


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk(value[key], (*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))
    else:
        yield path, value


def _sensitive_target_values(output: dict[str, Any]) -> set[tuple[type, Any]]:
    keys = {
        "gl_account",
        "policy_action",
        "profit_impact_minor",
        "other_impact_minor",
        "difference_minor",
    }
    values: set[tuple[type, Any]] = set()
    for path, value in _walk(output):
        if path and path[-1] in keys:
            values.add(_scalar_key(value))
        if len(path) >= 2 and path[-1] == "impact_minor":
            values.add(_scalar_key(value))
    return values


def _scalar_key(value: Any) -> tuple[type, Any]:
    if isinstance(value, (str, int, float, bool, type(None))):
        return type(value), value
    return type(value), repr(value)
