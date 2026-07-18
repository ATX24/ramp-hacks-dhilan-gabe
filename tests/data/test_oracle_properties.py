"""Semantic correctness properties for the latent world and oracle."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from distillery.contracts.tasks import Difficulty, SplitName, TaskId
from distillery.data.oracle import solve_task
from distillery.data.renderers import render_input, select_template_family
from distillery.data.validate import validate_example, validate_output
from distillery.data.world import (
    LatentWorld,
    PolicyAction,
    TxnHardNegative,
    VarianceRegime,
    build_world,
    normalize_policy_tokens,
    phrase_matches,
    resolve_policy,
)


def _find_world(
    task: TaskId,
    difficulty: Difficulty,
    *,
    ood: bool,
    predicate: Callable[[LatentWorld], bool],
) -> LatentWorld:
    for index in range(2_000):
        world = build_world(
            seed=17,
            index=index,
            split_token="semantic-alt" if ood else "semantic-main",
            task=task,
            difficulty=difficulty,
            ood=ood,
        )
        if predicate(world):
            return world
    raise AssertionError("requested latent regime was not generated")


@pytest.mark.parametrize(
    ("task", "difficulty", "ood"),
    [
        (task, difficulty, ood)
        for task in (
            TaskId.TRANSACTION_REVIEW,
            TaskId.VARIANCE_ANALYSIS,
            TaskId.CASH_RECONCILIATION,
        )
        for difficulty in (
            Difficulty.EASY,
            Difficulty.MEDIUM,
            Difficulty.HARD,
        )
        for ood in (False, True)
    ],
)
def test_oracle_outputs_validate(
    task: TaskId,
    difficulty: Difficulty,
    ood: bool,
) -> None:
    world = build_world(
        seed=23,
        index=11,
        split_token="oracle-alt" if ood else "oracle-main",
        task=task,
        difficulty=difficulty,
        ood=ood,
    )
    output = solve_task(world, task)
    assert validate_output(task, output).ok


def test_complete_token_matching_does_not_match_it_inside_deloitte() -> None:
    tokens = normalize_policy_tokens("DELOITTE CONSULTING")
    assert not phrase_matches(tokens, "it")
    assert phrase_matches(tokens, "consulting")


@pytest.mark.parametrize("ood", [False, True])
def test_every_vendor_resolves_to_its_canonical_gl(ood: bool) -> None:
    world = build_world(
        seed=17,
        index=3,
        split_token=f"vendor-{int(ood)}",
        task=TaskId.TRANSACTION_REVIEW,
        difficulty=Difficulty.EASY,
        ood=ood,
    )
    safe_amount = {
        "meals": 4_000,
        "airfare": 100_000,
        "lodging": 100_000,
        "saas": 100_000,
        "cloud": 100_000,
        "capex": 100_000,
        "services": 100_000,
        "personal": 100_000,
        "fees": 10_000,
        "facilities": 50_000,
    }
    for vendor in world.vendors:
        action, gl_account, rule_ids = resolve_policy(
            world.policies,
            descriptor=vendor.descriptors[0],
            amount_minor=safe_amount[vendor.category],
            category=vendor.category,
            vendor_id=vendor.vendor_id,
        )
        assert gl_account == vendor.default_gl, vendor.archetype
        assert rule_ids
        assert action in PolicyAction


def test_deloitte_consulting_maps_to_6600() -> None:
    world = build_world(
        seed=31,
        index=0,
        split_token="deloitte",
        task=TaskId.TRANSACTION_REVIEW,
        difficulty=Difficulty.EASY,
    )
    consulting = next(vendor for vendor in world.vendors if vendor.category == "services")
    action, gl_account, rule_ids = resolve_policy(
        world.policies,
        descriptor="DELOITTE CONSULTING",
        amount_minor=100_000,
        category="services",
        vendor_id=consulting.vendor_id,
    )
    assert gl_account == "6600"
    assert action == PolicyAction.REVIEW
    assert len(rule_ids) == 1


def test_full_corpus_service_vendors_map_to_6600(full_corpus) -> None:
    services = [
        example
        for example in full_corpus.examples
        if example.task == TaskId.TRANSACTION_REVIEW
        and example.input["expense_category"] == "services"
    ]
    deloitte = [
        example for example in services if "deloitte" in str(example.input["vendor"]).casefold()
    ]
    assert services
    assert deloitte
    assert all(example.expected_output["gl_account"] == "6600" for example in services)


def test_meal_thresholds_are_exhaustive_and_applicable() -> None:
    world = build_world(
        seed=17,
        index=0,
        split_token="thresholds",
        task=TaskId.TRANSACTION_REVIEW,
        difficulty=Difficulty.EASY,
    )
    meal_vendor = next(vendor for vendor in world.vendors if vendor.category == "meals")
    expectations = {
        5_000: ("MEAL-LOW", PolicyAction.APPROVE),
        5_001: ("MEAL-MID", PolicyAction.REVIEW),
        15_000: ("MEAL-MID", PolicyAction.REVIEW),
        15_001: ("MEAL-HIGH", PolicyAction.REJECT),
        50_000: ("MEAL-HIGH", PolicyAction.REJECT),
    }
    by_id = {rule.rule_id: rule for rule in world.policies}
    for amount, (semantic_code, expected_action) in expectations.items():
        action, gl_account, rule_ids = resolve_policy(
            world.policies,
            descriptor=meal_vendor.descriptors[0],
            amount_minor=amount,
            category="meals",
            vendor_id=meal_vendor.vendor_id,
        )
        rule = by_id[rule_ids[0]]
        assert rule.semantic_code == semantic_code
        assert rule.amount_applies(amount)
        assert action == expected_action
        assert gl_account == "6100"


@pytest.mark.parametrize("ood", [False, True])
def test_capex_opex_has_grounded_policy_in_both_regimes(ood: bool) -> None:
    world = _find_world(
        TaskId.TRANSACTION_REVIEW,
        Difficulty.HARD,
        ood=ood,
        predicate=lambda candidate: (
            candidate.transaction is not None
            and candidate.transaction.hard_negative == TxnHardNegative.CAPEX_OPEX
        ),
    )
    transaction = world.transaction
    assert transaction is not None
    cited = {
        rule.rule_id: rule
        for rule in world.policies
        if rule.rule_id in transaction.applied_rule_ids
    }
    assert cited
    assert all(rule.category == "capex" for rule in cited.values())
    assert all(rule.amount_applies(transaction.amount_minor) for rule in cited.values())
    output = solve_task(world, TaskId.TRANSACTION_REVIEW)
    assert output["gl_account"] == "1500"
    assert output["policy_action"] == "reject"


def test_every_variance_pnl_delta_matches_oracle(full_corpus) -> None:
    variances = [
        example for example in full_corpus.examples if example.task == TaskId.VARIANCE_ANALYSIS
    ]
    assert variances
    for example in variances:
        assert (
            example.input["budget_minor"] - example.input["actual_minor"]
            == example.expected_output["profit_impact_minor"]
        )


def test_offset_and_hidden_subtotal_use_distinct_mechanics() -> None:
    offset = _find_world(
        TaskId.VARIANCE_ANALYSIS,
        Difficulty.HARD,
        ood=False,
        predicate=lambda candidate: (
            candidate.variance is not None and candidate.variance.regime == VarianceRegime.OFFSET
        ),
    )
    hidden = _find_world(
        TaskId.VARIANCE_ANALYSIS,
        Difficulty.HARD,
        ood=False,
        predicate=lambda candidate: (
            candidate.variance is not None
            and candidate.variance.regime == VarianceRegime.HIDDEN_SUBTOTAL
        ),
    )
    assert offset.variance is not None
    assert hidden.variance is not None
    assert not offset.variance.unallocated
    assert hidden.variance.unallocated
    offset_impacts = {driver.profit_impact_minor() for driver in offset.variance.drivers}
    assert any(impact > 0 for impact in offset_impacts)
    assert any(impact < 0 for impact in offset_impacts)
    family = select_template_family(
        TaskId.VARIANCE_ANALYSIS,
        difficulty=Difficulty.HARD,
        ood=False,
        index=0,
        family_key=hidden.group_id,
    )
    rendered = render_input(
        hidden,
        TaskId.VARIANCE_ANALYSIS,
        template_family=family,
    )
    assert "unallocated_line_items" in rendered
    assert "impact_hint_minor" not in str(rendered)


def test_refund_split_reverses_every_allocation() -> None:
    world = _find_world(
        TaskId.TRANSACTION_REVIEW,
        Difficulty.HARD,
        ood=False,
        predicate=lambda candidate: (
            candidate.transaction is not None
            and candidate.transaction.hard_negative == TxnHardNegative.REFUND_SPLIT
        ),
    )
    transaction = world.transaction
    assert transaction is not None
    assert transaction.amount_minor < 0
    output = solve_task(world, TaskId.TRANSACTION_REVIEW)
    expense_lines = [line for line in output["journal_entry"] if line["account"] != "2100"]
    assert len(expense_lines) == 2
    assert all(line["side"] == "credit" for line in expense_lines)
    assert sum(line["amount_minor"] for line in expense_lines) == abs(transaction.amount_minor)
    assert (
        next(line for line in output["journal_entry"] if line["account"] == "2100")["side"]
        == "debit"
    )


def test_chargeback_hard_negative_is_present_and_balanced() -> None:
    world = _find_world(
        TaskId.TRANSACTION_REVIEW,
        Difficulty.HARD,
        ood=False,
        predicate=lambda candidate: (
            candidate.transaction is not None
            and candidate.transaction.hard_negative == TxnHardNegative.CHARGEBACK
        ),
    )
    assert world.transaction is not None
    assert world.transaction.amount_minor < 0
    assert "chargeback" in world.transaction.descriptor.casefold()
    output = solve_task(world, TaskId.TRANSACTION_REVIEW)
    assert validate_output(TaskId.TRANSACTION_REVIEW, output).ok


def test_cash_ood_uses_held_out_aggregation_mechanics(full_corpus) -> None:
    ood_cash = [
        example
        for example in full_corpus.examples
        if example.task == TaskId.CASH_RECONCILIATION
        and example.provenance.split == SplitName.OOD_TEST
    ]
    assert ood_cash
    assert any(
        any(
            len(group["book_ids"]) > 1 or len(group["bank_ids"]) > 1
            for group in example.expected_output["matched_groups"]
        )
        for example in ood_cash
    )


def test_all_generated_examples_pass_context_validation(full_corpus) -> None:
    for example in full_corpus.examples:
        result = validate_example(example)
        assert result.ok, (example.example_id, result.errors)
