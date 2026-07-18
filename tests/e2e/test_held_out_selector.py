"""Fixed-seed held-out selector: no train leakage, deterministic IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from held_out_selector import (
    HELD_OUT_SPLITS,
    select_held_out,
)
from held_out_selector import load_jsonl as load_dataset_jsonl


def test_selector_only_uses_held_out_splits(golden_jsonl: Path) -> None:
    rows = load_dataset_jsonl(golden_jsonl)
    held = [
        r
        for r in rows
        if r.get("provenance", {}).get("split") in HELD_OUT_SPLITS
    ]
    tasks_present = {r["task"] for r in held}
    # Golden fixture may not cover every primary on held-out; select what exists.
    if not tasks_present:
        pytest.fail("golden fixture has no held-out rows; selector cannot be tested")

    selection = select_held_out(
        rows,
        seed=17,
        tasks=tuple(sorted(tasks_present)),
        per_task=1,
    )
    assert selection.seed == 17
    for example in selection.examples:
        split = example["provenance"]["split"]
        assert split in HELD_OUT_SPLITS
        assert example["provenance"]["split"] not in {"train", "validation"}


def test_selector_is_deterministic(golden_jsonl: Path) -> None:
    rows = load_dataset_jsonl(golden_jsonl)
    held_tasks = {
        r["task"]
        for r in rows
        if r.get("provenance", {}).get("split") in HELD_OUT_SPLITS
    }
    if len(held_tasks) < 1:
        pytest.skip("no held-out tasks in fixture")
    tasks = tuple(sorted(held_tasks))
    a = select_held_out(rows, seed=17, tasks=tasks, per_task=1)
    b = select_held_out(rows, seed=17, tasks=tasks, per_task=1)
    assert a.example_ids == b.example_ids


def test_public_view_strips_gold(golden_jsonl: Path) -> None:
    rows = load_dataset_jsonl(golden_jsonl)
    held_tasks = {
        r["task"]
        for r in rows
        if r.get("provenance", {}).get("split") in HELD_OUT_SPLITS
    }
    if not held_tasks:
        pytest.skip("no held-out tasks in fixture")
    selection = select_held_out(
        rows, seed=17, tasks=tuple(sorted(held_tasks)), per_task=1
    )
    public = selection.as_public_view()
    blob = json.dumps(public)
    assert "expected_output" not in blob
    assert "latent_state_hash" not in blob
    assert "oracle" not in blob


def test_missing_task_fails_loud() -> None:
    with pytest.raises(ValueError, match="no held-out examples"):
        select_held_out(
            [{"example_id": "ex_x", "task": "transaction_review",
              "provenance": {"split": "train"}}],
            tasks=("transaction_review",),
        )


def test_prefer_valid_over_invalid_json_fixture(golden_jsonl: Path) -> None:
    rows = load_dataset_jsonl(golden_jsonl)
    selection = select_held_out(
        rows,
        seed=17,
        tasks=("transaction_review",),
        per_task=1,
        prefer_valid=True,
    )
    assert selection.example_ids == ("ex_txn_hard_001",)
    tags = selection.examples[0].get("case_tags") or []
    assert "invalid_json" not in tags
