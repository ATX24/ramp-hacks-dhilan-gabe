"""Property tests for accounting identities and oracle determinism."""

from __future__ import annotations

import pytest

from distillery.contracts.tasks import Difficulty, TaskId
from distillery.data.oracle import rank_drivers, solve_task
from distillery.data.validate import assert_policy_precedence, validate_output
from distillery.data.world import (
    PolicyAction,
    TxnHardNegative,
    build_world,
    resolve_policy,
)

TASKS = (
    TaskId.TRANSACTION_REVIEW,
    TaskId.VARIANCE_ANALYSIS,
    TaskId.CASH_RECONCILIATION,
)
DIFFS = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("difficulty", DIFFS)
@pytest.mark.parametrize("ood", [False, True])
@pytest.mark.parametrize("index", [0, 3, 11, 42])
def test_oracle_outputs_validate(
    task: TaskId,
    difficulty: Difficulty,
    ood: bool,
    index: int,
) -> None:
    world = build_world(
        seed=17,
        index=index,
        split_token="prop",
        task=task,
        difficulty=difficulty,
        ood=ood,
    )
    output = solve_task(world, task)
    result = validate_output(task, output)
    assert result.ok, result.errors


def test_debit_equals_credit_property() -> None:
    for index in range(50):
        world = build_world(
            seed=23,
            index=index,
            split_token="bal",
            task=TaskId.TRANSACTION_REVIEW,
            difficulty=DIFFS[index % 3],
        )
        out = solve_task(world, TaskId.TRANSACTION_REVIEW)
        debits = sum(
            line["amount_minor"]
            for line in out["journal_entry"]
            if line["side"] == "debit"
        )
        credits = sum(
            line["amount_minor"]
            for line in out["journal_entry"]
            if line["side"] == "credit"
        )
        assert debits == credits
        assert all(line["amount_minor"] >= 0 for line in out["journal_entry"])
        assert all(isinstance(line["amount_minor"], int) for line in out["journal_entry"])


def test_variance_closure_and_ranking() -> None:
    for index in range(40):
        world = build_world(
            seed=41,
            index=index,
            split_token="var",
            task=TaskId.VARIANCE_ANALYSIS,
            difficulty=Difficulty.HARD,
            ood=index % 2 == 0,
        )
        out = solve_task(world, TaskId.VARIANCE_ANALYSIS)
        driver_sum = sum(d["impact_minor"] for d in out["top_drivers"])
        assert driver_sum + out["other_impact_minor"] == out["profit_impact_minor"]
        ranked = rank_drivers([(d["driver_id"], d["impact_minor"]) for d in out["top_drivers"]])
        assert [
            (d["driver_id"], d["impact_minor"], d["rank"]) for d in out["top_drivers"]
        ] == ranked
        expected_dir = "favorable" if out["profit_impact_minor"] >= 0 else "unfavorable"
        assert out["direction"] == expected_dir


def test_cash_adjusted_difference_identity() -> None:
    for index in range(40):
        world = build_world(
            seed=7,
            index=index,
            split_token="csh",
            task=TaskId.CASH_RECONCILIATION,
            difficulty=DIFFS[index % 3],
            ood=index % 3 == 0,
        )
        out = solve_task(world, TaskId.CASH_RECONCILIATION)
        assert (
            out["difference_minor"]
            == out["adjusted_book_balance_minor"] - out["adjusted_bank_balance_minor"]
        )


def test_policy_precedence_winner() -> None:
    world = build_world(
        seed=17,
        index=0,
        split_token="pol",
        task=TaskId.TRANSACTION_REVIEW,
        difficulty=Difficulty.HARD,
    )
    assert world.transaction is not None
    # Reconstruct matches with conflicting rules hard-negative.
    action, gl, rule_ids = resolve_policy(
        world.policies,
        descriptor="DELL SERVER CLUSTER",
        amount_minor=5_000_000,
        category="capex",
        hard_negative=TxnHardNegative.CONFLICTING_RULES,
    )
    assert action == PolicyAction.REJECT
    assert rule_ids[0] == "POL-CAPEX-009"
    precedence = {p.rule_id: p.precedence for p in world.policies}
    matched = [p.rule_id for p in world.policies if p.rule_id in {"POL-CAPEX-009", "POL-IT-002"}]
    result = assert_policy_precedence(matched, precedence, rule_ids[0])
    assert result.ok, result.errors
    assert gl == "1500"


def test_tie_break_by_driver_id() -> None:
    ranked = rank_drivers([("beta_cost", -100_000), ("alpha_cost", -100_000)])
    assert ranked[0][0] == "alpha_cost"
    assert ranked[0][2] == 1
    assert ranked[1][0] == "beta_cost"


def test_world_hash_stable() -> None:
    a = build_world(
        seed=17,
        index=5,
        split_token="hash",
        task=TaskId.VARIANCE_ANALYSIS,
        difficulty=Difficulty.MEDIUM,
    )
    b = build_world(
        seed=17,
        index=5,
        split_token="hash",
        task=TaskId.VARIANCE_ANALYSIS,
        difficulty=Difficulty.MEDIUM,
    )
    assert a.latent_state_hash() == b.latent_state_hash()
    assert solve_task(a, TaskId.VARIANCE_ANALYSIS) == solve_task(
        b, TaskId.VARIANCE_ANALYSIS
    )
