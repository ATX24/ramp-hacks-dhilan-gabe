"""Determinism and corpus size checks for Finance Agent."""

from __future__ import annotations

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.generate import CORPUS_PLANNED, CORPUS_SMOKE, generate_agent_corpus
from distillery.finance_agent.metrics import aggregate_metrics, score_episode


def test_smoke_generation_is_deterministic() -> None:
    first = generate_agent_corpus(CORPUS_SMOKE)
    second = generate_agent_corpus(CORPUS_SMOKE)
    assert first.manifest["content_sha256"] == second.manifest["content_sha256"]
    assert first.manifest["manifest_sha256"] == second.manifest["manifest_sha256"]
    assert [ex.example_id for ex in first.examples] == [ex.example_id for ex in second.examples]


def test_seed_change_changes_corpus() -> None:
    first = generate_agent_corpus(CORPUS_SMOKE, seed=17)
    second = generate_agent_corpus(CORPUS_SMOKE, seed=23)
    assert first.manifest["content_sha256"] != second.manifest["content_sha256"]


def test_smoke_exact_sizes() -> None:
    corpus = generate_agent_corpus(CORPUS_SMOKE)
    assert len(corpus.examples) == 48
    assert len(corpus.by_split[SplitName.TRAIN]) == 24
    assert len(corpus.by_split[SplitName.VALIDATION]) == 8
    assert len(corpus.by_split[SplitName.TEST]) == 16


def test_planned_corpus_counts() -> None:
    assert CORPUS_PLANNED.total_examples == 2_200
    assert CORPUS_SMOKE.total_examples == 48


def test_oracle_self_score_is_perfect() -> None:
    corpus = generate_agent_corpus(CORPUS_SMOKE)
    metrics = aggregate_metrics([score_episode(example) for example in corpus.examples])
    assert metrics["end_to_end_success_rate"] == 1.0
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["argument_exactness"] == 1.0
    assert metrics["final_answer_correctness"] == 1.0
