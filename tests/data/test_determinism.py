"""Determinism and sealed-manifest tests."""

from __future__ import annotations

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import SplitName
from distillery.data.generate import CORPUS_SMOKE, generate_corpus
from distillery.data.mixture import mixture_plan


def test_smoke_generation_is_deterministic() -> None:
    a = generate_corpus(CORPUS_SMOKE, check_near_duplicates=False)
    b = generate_corpus(CORPUS_SMOKE, check_near_duplicates=False)
    assert a.manifest["content_sha256"] == b.manifest["content_sha256"]
    assert a.manifest["split_sha256"] == b.manifest["split_sha256"]
    assert a.manifest["manifest_sha256"] == b.manifest["manifest_sha256"]
    assert [e.example_id for e in a.examples] == [e.example_id for e in b.examples]
    assert [e.oracle.latent_state_hash for e in a.examples] == [
        e.oracle.latent_state_hash for e in b.examples
    ]


def test_seed_change_changes_corpus() -> None:
    a = generate_corpus(CORPUS_SMOKE, seed=17, check_near_duplicates=False)
    b = generate_corpus(CORPUS_SMOKE, seed=23, check_near_duplicates=False)
    assert a.manifest["content_sha256"] != b.manifest["content_sha256"]


def test_smoke_counts_and_mixture(smoke_corpus) -> None:
    assert len(smoke_corpus.examples) == 560
    assert len(smoke_corpus.by_split[SplitName.TRAIN]) == 320
    assert len(smoke_corpus.by_split[SplitName.VALIDATION]) == 80
    assert len(smoke_corpus.by_split[SplitName.TEST]) == 160
    mix = smoke_corpus.manifest["mixtures"]["train"]["mixture"]["by_task"]
    assert mix["transaction_review"] == 144
    assert mix["variance_analysis"] == 144
    assert mix["cash_reconciliation"] == 32
    # Difficulty is 30/40/30 within each task; global counts follow Hamilton per task.
    expected = {"easy": 0, "medium": 0, "hard": 0}
    for _task, difficulty in mixture_plan(320):
        expected[difficulty.value] += 1
    diff = smoke_corpus.manifest["mixtures"]["train"]["mixture"]["by_difficulty"]
    assert diff == expected
    assert diff["medium"] > diff["easy"]
    assert diff["medium"] > diff["hard"]


def test_manifest_hash_covers_payload(smoke_corpus) -> None:
    payload = {
        k: v for k, v in smoke_corpus.manifest.items() if k != "manifest_sha256"
    }
    assert smoke_corpus.manifest["manifest_sha256"] == content_sha256(payload)
