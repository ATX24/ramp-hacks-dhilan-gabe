"""Corpus counts, opaque identities, shuffle, and process determinism."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import Difficulty, SplitName, TaskId
from distillery.data.generate import CORPUS_SMOKE, generate_corpus
from distillery.data.mixture import mixture_plan
from distillery.data.world import build_world

_OPAQUE_ID_RE = re.compile(r"^(?:ex|world|grp|ent|txn|vnd|src|bok|bnk)_[0-9a-f]{18}$")
_VISIBLE_SPLIT_RE = re.compile(
    r"(?i)(?:\b(?:train|training|validation|ood|iid_test)\b|"
    r"(?:smk|full)_(?:tr|va|te|iid|ood)\b)"
)


def test_smoke_generation_is_deterministic() -> None:
    first = generate_corpus(CORPUS_SMOKE, check_near_duplicates=False)
    second = generate_corpus(CORPUS_SMOKE, check_near_duplicates=False)
    assert first.manifest["content_sha256"] == second.manifest["content_sha256"]
    assert first.manifest["split_sha256"] == second.manifest["split_sha256"]
    assert first.manifest["order_sha256"] == second.manifest["order_sha256"]
    assert first.manifest["manifest_sha256"] == second.manifest["manifest_sha256"]
    assert [example.example_id for example in first.examples] == [
        example.example_id for example in second.examples
    ]


def test_seed_change_changes_corpus() -> None:
    first = generate_corpus(
        CORPUS_SMOKE,
        seed=17,
        check_near_duplicates=False,
    )
    second = generate_corpus(
        CORPUS_SMOKE,
        seed=23,
        check_near_duplicates=False,
    )
    assert first.manifest["content_sha256"] != second.manifest["content_sha256"]


def test_smoke_exact_sizes_and_mixtures(smoke_corpus) -> None:
    assert len(smoke_corpus.examples) == 560
    expected_sizes = {
        SplitName.TRAIN: 320,
        SplitName.VALIDATION: 80,
        SplitName.TEST: 160,
    }
    for split, size in expected_sizes.items():
        assert len(smoke_corpus.by_split[split]) == size
        expected = {
            "easy": 0,
            "medium": 0,
            "hard": 0,
        }
        for _task, difficulty in mixture_plan(size):
            expected[difficulty.value] += 1
        actual = smoke_corpus.manifest["mixtures"][split.value]["mixture"]
        assert actual["by_difficulty"] == expected
    assert smoke_corpus.manifest["mixtures"]["train"]["mixture"]["by_task"] == {
        "transaction_review": 144,
        "variance_analysis": 144,
        "cash_reconciliation": 32,
    }


def test_full_exact_sizes_and_mixtures(full_corpus) -> None:
    expected_sizes = {
        SplitName.TRAIN: 3_200,
        SplitName.VALIDATION: 400,
        SplitName.IID_TEST: 800,
        SplitName.OOD_TEST: 800,
    }
    assert len(full_corpus.examples) == 5_200
    for split, size in expected_sizes.items():
        assert len(full_corpus.by_split[split]) == size
        mixture = full_corpus.manifest["mixtures"][split.value]["mixture"]
        assert mixture["by_task"] == {
            "transaction_review": int(size * 0.45),
            "variance_analysis": int(size * 0.45),
            "cash_reconciliation": int(size * 0.10),
        }
        assert mixture["by_difficulty"] == {
            "easy": int(size * 0.30),
            "medium": int(size * 0.40),
            "hard": int(size * 0.30),
        }


def test_manifest_hash_covers_payload(smoke_corpus) -> None:
    payload = {
        key: value for key, value in smoke_corpus.manifest.items() if key != "manifest_sha256"
    }
    assert smoke_corpus.manifest["manifest_sha256"] == content_sha256(payload)


def test_sealed_order_is_deterministically_shuffled(full_corpus) -> None:
    for split, examples in full_corpus.by_split.items():
        ordered_plan = mixture_plan(len(examples))
        actual = [(example.task, example.difficulty) for example in examples]
        assert actual != ordered_plan, split
        assert len({example.task for example in examples[:25]}) >= 2
        assert len({example.difficulty for example in examples[:25]}) >= 2
        longest_run = 1
        current_run = 1
        for previous, current in zip(actual, actual[1:], strict=False):
            if previous == current:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        assert longest_run < 20


def test_visible_identifiers_are_opaque_and_have_no_split_markers(
    full_corpus,
) -> None:
    for example in full_corpus.examples:
        assert _OPAQUE_ID_RE.fullmatch(example.example_id)
        assert _OPAQUE_ID_RE.fullmatch(example.world_id)
        assert _OPAQUE_ID_RE.fullmatch(example.group_id)
        mutable_input = example.model_dump(mode="json")["input"]
        serialized_input = json.dumps(
            mutable_input,
            sort_keys=True,
            ensure_ascii=False,
        )
        assert not _VISIBLE_SPLIT_RE.search(serialized_input)
        assert "case_nonce" not in example.input
        assert "template_family" not in example.input
        for value in _input_ids(example.input):
            assert _OPAQUE_ID_RE.fullmatch(value), value


def test_salt_retry_changes_world_identity_and_latent_content() -> None:
    base = build_world(
        seed=17,
        index=42,
        split_token="retry-domain",
        task=TaskId.VARIANCE_ANALYSIS,
        difficulty=Difficulty.HARD,
        salt=0,
    )
    retry = build_world(
        seed=17,
        index=42,
        split_token="retry-domain",
        task=TaskId.VARIANCE_ANALYSIS,
        difficulty=Difficulty.HARD,
        salt=1,
    )
    assert base.world_id != retry.world_id
    assert base.entity_id != retry.entity_id
    assert base.latent_state_hash() != retry.latent_state_hash()


def test_hashes_are_stable_across_python_hash_seeds() -> None:
    script = (
        "from distillery.data.generate import generate_corpus;"
        "c=generate_corpus('smoke',check_near_duplicates=True);"
        "print(c.manifest['content_sha256']);"
        "print(c.manifest['manifest_sha256'])"
    )
    repo = Path(__file__).resolve().parents[2]
    outputs = []
    for hash_seed in ("1", "8675309"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip().splitlines())
    assert outputs[0] == outputs[1]


def _input_ids(finance_input: dict) -> list[str]:
    values: list[str] = []
    for key in ("entity_id", "txn_id", "vendor_id"):
        if finance_input.get(key):
            values.append(str(finance_input[key]))
    for item in finance_input.get("driver_observations", []):
        values.append(str(item["source_id"]))
    for item in finance_input.get("unallocated_line_items", []):
        values.append(str(item["source_id"]))
    for collection in ("book_entries", "bank_events"):
        for item in finance_input.get(collection, []):
            values.append(str(item["id"]))
    return values
