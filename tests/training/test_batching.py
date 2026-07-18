"""Deterministic sampler / batch plan tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.training.batching import (
    DEFAULT_FINANCE_MIXTURE,
    SamplerExample,
    completion_token_budget_filter,
    largest_remainder_counts,
    plan_batches,
    sampler_order_hash,
)


def _examples() -> list[SamplerExample]:
    """20 examples => task 9/9/2; difficulty 3/3/3, 3/3/3, 1/1/0."""
    out: list[SamplerExample] = []
    task_difficulties = {
        "transaction_review": ["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3,
        "variance_analysis": ["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3,
        "cash_reconciliation": ["easy", "medium"],
    }
    index = 0
    for task, difficulties in task_difficulties.items():
        for difficulty in difficulties:
            out.append(
                SamplerExample(
                    example_id=f"ex_{index}",
                    task=task,
                    difficulty=difficulty,
                    completion_tokens=10 + index,
                    prompt_tokens=5,
                    total_tokens=15 + index,
                    completion_token_source="student_tokenizer",
                    completion_tokenizer_sha256="a" * 64,
                    record_sha256=f"{index:064x}",
                )
            )
            index += 1
    return out


def test_sampler_order_hash_is_deterministic() -> None:
    a = sampler_order_hash(["ex_1", "ex_2"], seed=17, microbatch_size=1)
    b = sampler_order_hash(["ex_1", "ex_2"], seed=17, microbatch_size=1)
    c = sampler_order_hash(["ex_1", "ex_2"], seed=23, microbatch_size=1)
    assert a == b
    assert a != c
    assert len(a) == 64


def test_plan_batches_stable_across_calls() -> None:
    examples = _examples()
    p1 = plan_batches(examples, seed=17, microbatch_size=2)
    p2 = plan_batches(examples, seed=17, microbatch_size=2)
    assert p1.order == p2.order
    assert p1.batches == p2.batches
    assert p1.sampler_order_hash == p2.sampler_order_hash
    assert all(len(batch) <= 2 for batch in p1.batches)
    assert p1.expected_task_counts == {
        "transaction_review": 9,
        "variance_analysis": 9,
        "cash_reconciliation": 2,
    }
    assert p1.expected_difficulty_counts["transaction_review"] == {
        "easy": 3,
        "medium": 3,
        "hard": 3,
    }
    assert p1.rounding_method == "largest_remainder_stable_order"
    changed_count = [
        examples[0].model_copy(
            update={
                "completion_tokens": examples[0].completion_tokens + 1,
                "total_tokens": examples[0].total_tokens + 1,
            }
        ),
        *examples[1:],
    ]
    assert plan_batches(
        changed_count, seed=17, microbatch_size=2
    ).sampler_order_hash != (
        p1.sampler_order_hash
    )


def test_plan_batches_uses_mixture_not_file_order() -> None:
    examples = _examples()
    plan = plan_batches(examples, seed=17, microbatch_size=1, mixture=DEFAULT_FINANCE_MIXTURE)
    assert set(plan.order) == {ex.example_id for ex in examples}
    # Not identical to input file order.
    assert list(plan.order) != [ex.example_id for ex in examples]


def test_completion_token_budget_filter() -> None:
    examples = _examples()
    kept, discarded = completion_token_budget_filter(examples, max_completion_tokens=14)
    assert all(ex.completion_tokens <= 14 for ex in kept)
    assert all(ex.completion_tokens > 14 for ex in discarded)
    assert len(kept) + len(discarded) == len(examples)


def test_largest_remainder_rounding_is_documented_and_stable() -> None:
    counts = largest_remainder_counts(
        7,
        {
            "transaction_review": 0.45,
            "variance_analysis": 0.45,
            "cash_reconciliation": 0.10,
        },
    )
    assert counts == {
        "transaction_review": 3,
        "variance_analysis": 3,
        "cash_reconciliation": 1,
    }


def test_wrong_task_or_difficulty_mix_is_rejected() -> None:
    examples = _examples()
    wrong_task = [
        example.model_copy(update={"task": "transaction_review"})
        if example.example_id == "ex_18"
        else example
        for example in examples
    ]
    with pytest.raises(ValueError, match="task mixture"):
        plan_batches(wrong_task, seed=17)

    wrong_difficulty = [
        example.model_copy(update={"difficulty": "hard"})
        if example.example_id == "ex_0"
        else example
        for example in examples
    ]
    with pytest.raises(ValueError, match="difficulty mixture"):
        plan_batches(wrong_difficulty, seed=17)


def test_unknown_task_and_difficulty_are_rejected() -> None:
    examples = _examples()
    unknown_task = [
        examples[0].model_copy(update={"task": "merchant_tagging"}),
        *examples[1:],
    ]
    # Default v1 mixture still rejects merchant_tagging (v2 uses FINANCE_MIXTURE_V2).
    with pytest.raises(ValueError, match="unknown tasks"):
        plan_batches(unknown_task, seed=17)
    unknown_difficulty = [
        examples[0].model_copy(update={"difficulty": "expert"}),
        *examples[1:],
    ]
    with pytest.raises(ValueError, match="unknown difficulties"):
        plan_batches(unknown_difficulty, seed=17)


def test_completion_count_must_be_tokenizer_derived() -> None:
    with pytest.raises(ValidationError):
        SamplerExample(
            example_id="ex_words",
            task="transaction_review",
            difficulty="easy",
            completion_tokens=3,
            prompt_tokens=2,
            total_tokens=5,
            completion_token_source="whitespace_words",
            completion_tokenizer_sha256="a" * 64,
            record_sha256="b" * 64,
        )
