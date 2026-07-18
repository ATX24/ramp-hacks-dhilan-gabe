"""Benchmark prompts must never expose answers or test-split leakage."""

from __future__ import annotations

import pytest

from experiments.benchmark.prompts import (
    FORBIDDEN_PROMPT_KEYS,
    assert_no_answer_leakage,
    build_benchmark_prompts,
    build_messages,
)


def test_forbidden_keys_cover_answer_channels() -> None:
    required = {
        "answer",
        "expected_output",
        "label",
        "oracle",
        "target",
        "target_output",
    }
    assert required <= {k.casefold() for k in FORBIDDEN_PROMPT_KEYS}


def test_assert_no_answer_leakage_fails_loud() -> None:
    with pytest.raises(ValueError, match="forbidden prompt field"):
        assert_no_answer_leakage({"input": {"amount_minor": 1}, "expected_output": {"x": 1}})


def test_build_messages_rejects_hygiene_violations() -> None:
    with pytest.raises(ValueError, match="input hygiene failed"):
        build_messages(
            task="transaction_review",
            example_input={"amount_minor": 1, "label": "leak"},
        )


def test_benchmark_pool_uses_train_validation_only_and_all_tasks() -> None:
    warmups, timed = build_benchmark_prompts(warmups=20, timed=200, seed=17)
    assert len(warmups) == 20
    assert len(timed) == 200
    assert {p.split for p in warmups + timed} <= {"train", "validation"}
    assert "test" not in {p.split for p in warmups + timed}
    tasks = {p.task for p in timed}
    assert tasks == {
        "transaction_review",
        "variance_analysis",
        "cash_reconciliation",
    }
    for prompt in warmups + timed:
        assert "expected_output" not in prompt.prompt_text
        assert "oracle" not in prompt.prompt_text.casefold()
        assert_no_answer_leakage({"prompt": prompt.prompt_text, "messages": prompt.messages})
