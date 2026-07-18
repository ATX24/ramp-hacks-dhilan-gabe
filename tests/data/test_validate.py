"""Validator mutations must fail at the exact finance invariant."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from distillery.contracts.tasks import FinanceTaskEnvelope, TaskId
from distillery.data.validate import validate_example, validate_output


def _clone(
    example: FinanceTaskEnvelope,
    *,
    input_updates: dict[str, Any] | None = None,
    output_mutator=None,
) -> FinanceTaskEnvelope:
    data = example.model_dump(mode="json")
    if input_updates:
        data["input"] = {**data["input"], **input_updates}
    if output_mutator:
        output_mutator(data["expected_output"])
    return FinanceTaskEnvelope.model_validate(data)


def _example(corpus, task: TaskId) -> FinanceTaskEnvelope:
    return next(example for example in corpus.examples if example.task == task)


def _has(result, fragment: str) -> bool:
    return any(fragment in error for error in result.errors)


def test_rejects_unbalanced_journal(smoke_corpus) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)
    output = copy.deepcopy(transaction.model_dump(mode="json")["expected_output"])
    output["journal_entry"][0]["amount_minor"] += 1
    result = validate_output(TaskId.TRANSACTION_REVIEW, output)
    assert not result.ok
    assert _has(result, "typed_output") or _has(result, "unbalanced")


def test_rejects_unknown_gl_and_journal_account(smoke_corpus) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)

    def mutate(output: dict[str, Any]) -> None:
        output["gl_account"] = "9999"
        expense = next(line for line in output["journal_entry"] if line["account"] != "2100")
        expense["account"] = "9999"

    result = validate_example(_clone(transaction, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "unknown_gl_account")
    assert _has(result, "unknown_journal_account")


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda output: output.__setitem__(
                "rule_ids",
                ["POL-NOT-PRESENT-FFFFFFFF"],
            ),
            "unknown_policy_rule",
        ),
        (
            lambda output: output.__setitem__(
                "policy_action",
                "approve" if output["policy_action"] != "approve" else "reject",
            ),
            "policy_action_inconsistent",
        ),
    ],
)
def test_rejects_policy_rule_and_action_inconsistency(
    smoke_corpus,
    mutation,
    expected_error: str,
) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)
    result = validate_example(_clone(transaction, output_mutator=mutation))
    assert not result.ok
    assert _has(result, expected_error)


def test_rejects_inapplicable_policy_rule(smoke_corpus) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)
    inapplicable = next(
        rule
        for rule in transaction.input["policy_rules"]
        if rule["rule_id"] not in transaction.expected_output["rule_ids"]
    )

    def mutate(output: dict[str, Any]) -> None:
        output["rule_ids"] = [inapplicable["rule_id"]]

    result = validate_example(_clone(transaction, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "policy_rule_inconsistent")


def test_rejects_cited_rule_missing_from_excerpt(smoke_corpus) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)
    cited = transaction.expected_output["rule_ids"][0]
    mutated = _clone(
        transaction,
        input_updates={
            "policy_excerpt": transaction.input["policy_excerpt"].replace(
                cited,
                "REMOVED",
            )
        },
    )
    result = validate_example(mutated)
    assert not result.ok
    assert _has(result, "cited_rule_missing_from_excerpt")


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("source_id", "txn_aaaaaaaaaaaaaaaaaa", "invalid_evidence_source"),
        ("field", "missing_field", "invalid_evidence_field"),
        ("value", "wrong-value", "evidence_value_mismatch"),
    ],
)
def test_rejects_ungrounded_transaction_evidence(
    smoke_corpus,
    field: str,
    value: str,
    expected_error: str,
) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)

    def mutate(output: dict[str, Any]) -> None:
        output["evidence"][0][field] = value

    result = validate_example(_clone(transaction, output_mutator=mutate))
    assert not result.ok
    assert _has(result, expected_error)


def test_rejects_non_integer_input_minor_units(smoke_corpus) -> None:
    transaction = _example(smoke_corpus, TaskId.TRANSACTION_REVIEW)
    mutated = _clone(
        transaction,
        input_updates={"amount_minor": float(transaction.input["amount_minor"])},
    )
    result = validate_example(mutated)
    assert not result.ok
    assert _has(result, "not_integer_minor_units")


def test_rejects_variance_pnl_mismatch(smoke_corpus) -> None:
    variance = _example(smoke_corpus, TaskId.VARIANCE_ANALYSIS)
    mutated = _clone(
        variance,
        input_updates={"actual_minor": variance.input["actual_minor"] + 1},
    )
    result = validate_example(mutated)
    assert not result.ok
    assert _has(result, "variance_pnl_mismatch")


def test_rejects_variance_driver_impact_not_derived_from_evidence(
    smoke_corpus,
) -> None:
    variance = next(
        example
        for example in smoke_corpus.examples
        if example.task == TaskId.VARIANCE_ANALYSIS
        and len(example.expected_output["top_drivers"]) >= 2
    )

    def mutate(output: dict[str, Any]) -> None:
        output["top_drivers"][0]["impact_minor"] += 1
        output["top_drivers"][1]["impact_minor"] -= 1

    result = validate_example(_clone(variance, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "driver_impact_mismatch") or _has(
        result,
        "driver_ranking_incorrect",
    )


def test_rejects_invalid_variance_evidence_id(smoke_corpus) -> None:
    variance = _example(smoke_corpus, TaskId.VARIANCE_ANALYSIS)

    def mutate(output: dict[str, Any]) -> None:
        output["evidence_ids"][0] = "src_aaaaaaaaaaaaaaaaaa"

    result = validate_example(_clone(variance, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "variance_evidence_ids_invalid")


def test_rejects_invalid_variance_rule_id(smoke_corpus) -> None:
    variance = _example(smoke_corpus, TaskId.VARIANCE_ANALYSIS)

    def mutate(output: dict[str, Any]) -> None:
        output["rule_ids"] = ["VAR-NOT-PRESENT-FFFFFFFF"]

    result = validate_example(_clone(variance, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "variance_rule_ids_invalid")


def test_rejects_cash_unknown_matched_ids(smoke_corpus) -> None:
    cash = _example(smoke_corpus, TaskId.CASH_RECONCILIATION)

    def mutate(output: dict[str, Any]) -> None:
        output["matched_groups"][0]["book_ids"][0] = "bok_aaaaaaaaaaaaaaaaaa"

    result = validate_example(_clone(cash, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "cash_unknown_book_ids")


def test_rejects_cash_unknown_exception_ids(smoke_corpus) -> None:
    cash = next(
        example
        for example in smoke_corpus.examples
        if example.task == TaskId.CASH_RECONCILIATION and example.expected_output["exceptions"]
    )

    def mutate(output: dict[str, Any]) -> None:
        output["exceptions"][0]["event_ids"][0] = "bnk_aaaaaaaaaaaaaaaaaa"

    result = validate_example(_clone(cash, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "cash_exception_unknown_ids")


def test_rejects_cash_adjusted_balance_mismatch(smoke_corpus) -> None:
    cash = _example(smoke_corpus, TaskId.CASH_RECONCILIATION)

    def mutate(output: dict[str, Any]) -> None:
        output["adjusted_book_balance_minor"] += 1
        output["difference_minor"] += 1

    result = validate_example(_clone(cash, output_mutator=mutate))
    assert not result.ok
    assert _has(result, "cash_adjusted_book_mismatch")
