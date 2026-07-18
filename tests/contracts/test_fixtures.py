"""Golden fixture coverage and frozen hash manifest."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import (
    CashReconciliationOutput,
    FinanceTaskEnvelope,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
)

REQUIRED_CASE_TAGS = {
    "invalid_json",
    "unbalanced_journal",
    "sign_inversion",
    "tie_ranked",
    "reconciliation_exception",
}


def test_exactly_twelve_golden_records(golden_records: list[dict]) -> None:
    assert len(golden_records) == 12
    ids = [r["example_id"] for r in golden_records]
    assert len(set(ids)) == 12


def test_coverage_tasks_difficulties_and_tags(
    golden_records: list[dict], fixture_manifest: dict
) -> None:
    tasks = {r["task"] for r in golden_records}
    difficulties = {r["difficulty"] for r in golden_records}
    tags: set[str] = set()
    for r in golden_records:
        tags.update(r.get("case_tags", []))

    assert tasks == {
        "transaction_review",
        "variance_analysis",
        "cash_reconciliation",
    }
    assert difficulties == {"easy", "medium", "hard"}
    assert REQUIRED_CASE_TAGS <= tags
    assert set(fixture_manifest["coverage"]["required_case_tags"]) == REQUIRED_CASE_TAGS


def test_file_and_per_example_hashes_match_manifest(
    fixture_dir: Path, golden_lines: list[str], fixture_manifest: dict
) -> None:
    golden_bytes = (fixture_dir / "golden.jsonl").read_bytes()
    raw_lines = golden_bytes.splitlines(keepends=True)
    oracle_bytes = (fixture_dir / "oracle_expected.json").read_bytes()
    assert sha256_hex(golden_bytes) == fixture_manifest["files"]["golden.jsonl"]["sha256"]
    assert sha256_hex(oracle_bytes) == fixture_manifest["files"]["oracle_expected.json"]["sha256"]
    assert fixture_manifest["files"]["golden.jsonl"]["line_count"] == 12
    assert fixture_manifest["record_count"] == 12

    assert len(raw_lines) == len(golden_lines)
    assert all(line.endswith(b"\n") for line in raw_lines)
    for raw_line, _line in zip(raw_lines, golden_lines, strict=True):
        record = json.loads(raw_line)
        example_id = record["example_id"]
        semantic_hash = fixture_manifest["semantic_example_sha256"][example_id]
        raw_line_hash = fixture_manifest["raw_line_sha256"][example_id]
        assert content_sha256(record) == semantic_hash
        assert sha256_hex(raw_line) == raw_line_hash
        assert semantic_hash != raw_line_hash


def test_envelopes_validate_and_negative_outputs_fail(
    golden_records: list[dict], oracle_expected: dict
) -> None:
    for record in golden_records:
        env = FinanceTaskEnvelope.model_validate(record)
        meta = oracle_expected[env.example_id]
        tags = set(env.case_tags)

        if env.task is TaskId.TRANSACTION_REVIEW:
            TransactionReviewOutput.model_validate(env.expected_output)
        elif env.task is TaskId.VARIANCE_ANALYSIS:
            VarianceAnalysisOutput.model_validate(env.expected_output)
        elif env.task is TaskId.CASH_RECONCILIATION:
            CashReconciliationOutput.model_validate(env.expected_output)
        else:
            raise AssertionError(f"unexpected task {env.task}")
        assert meta["rejects_fixture_expected_output"] is False

        if "unbalanced_journal" in tags:
            candidate = meta["sample_invalid_prediction"]
            assert isinstance(candidate, dict)
            with pytest.raises(ValidationError):
                TransactionReviewOutput.model_validate(candidate)
        elif "sign_inversion" in tags:
            candidate = meta["sample_invalid_prediction"]
            assert isinstance(candidate, dict)
            with pytest.raises(ValidationError):
                VarianceAnalysisOutput.model_validate(candidate)
        elif "invalid_json" in tags:
            with pytest.raises(json.JSONDecodeError):
                json.loads(meta["sample_invalid_prediction"])
        assert (
            env.model_dump(mode="json")["expected_output"] == meta["expected_output"]
        )


def test_oracle_expected_keys_match_golden(
    golden_records: list[dict],
    oracle_expected: dict,
) -> None:
    golden_ids = {r["example_id"] for r in golden_records}
    assert set(oracle_expected) == golden_ids


def test_manifest_self_hash_stable_under_canonical_reload(fixture_dir: Path) -> None:
    """Re-reading bytes yields the same digest (no hidden rewrite)."""
    path = fixture_dir / "golden.jsonl"
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    second = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first == second


def test_semantic_hash_is_independent_of_raw_json_formatting(
    golden_records: list[dict],
) -> None:
    record = golden_records[0]
    compact = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    spaced = json.dumps(record, separators=(", ", ": "), sort_keys=False).encode()
    assert sha256_hex(compact) != sha256_hex(spaced)
    assert content_sha256(json.loads(compact)) == content_sha256(json.loads(spaced))


def test_model_visible_inputs_have_no_target_or_candidate_leakage(
    golden_records: list[dict],
) -> None:
    forbidden_keys = {
        "candidate",
        "candidate_output",
        "candidate_prediction",
        "expected",
        "expected_output",
        "gold",
        "gold_output",
        "label",
        "target",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for key, nested in value.items():
                assert not key.startswith(("candidate_", "expected_", "gold_", "target_"))
                assert not key.endswith("_hint")
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for record in golden_records:
        walk(record["input"])
        encoded = json.dumps(record["input"], sort_keys=True)
        assert re.search(
            r"\b(?:train|validation|test|ood|out[-_ ]of[-_ ]distribution)\b",
            encoded,
            flags=re.IGNORECASE,
        ) is None


def test_isolation_identities_are_disjoint_across_splits(
    golden_records: list[dict],
) -> None:
    identities: dict[str, set[str]] = {}

    def collect_input_identities(value: object, result: set[str]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "id" or key.endswith("_id"):
                    assert isinstance(nested, str)
                    result.add(nested)
                elif key == "period":
                    assert isinstance(nested, str)
                    result.add(f"period:{nested}")
                collect_input_identities(nested, result)
        elif isinstance(value, list):
            for nested in value:
                collect_input_identities(nested, result)

    for record in golden_records:
        split = record["provenance"]["split"]
        split_ids = identities.setdefault(split, set())
        split_ids.update(
            {
                f"example:{record['example_id']}",
                f"world:{record['world_id']}",
                f"group:{record['group_id']}",
                f"template:{record['provenance']['template_family']}",
                f"latent:{record['oracle']['latent_state_hash']}",
            }
        )
        collect_input_identities(record["input"], split_ids)

    split_names = sorted(identities)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            assert identities[split].isdisjoint(identities[other]), (
                split,
                other,
                identities[split] & identities[other],
            )


def test_approval_and_cash_oracles_are_supported(
    golden_records: list[dict],
) -> None:
    by_id = {record["example_id"]: record for record in golden_records}
    travel = by_id["ex_txn_medium_001"]
    assert travel["expected_output"]["policy_action"] == "approve"
    assert travel["input"]["preapproved"] is True

    cash = by_id["ex_cash_hard_exc_001"]
    output = cash["expected_output"]
    bank_fees = sum(
        item["amount_minor"]
        for item in output["exceptions"]
        if item["type"] == "bank_fee"
    )
    deposits = sum(
        item["amount_minor"]
        for item in output["exceptions"]
        if item["type"] == "deposit_in_transit"
    )
    assert output["adjusted_book_balance_minor"] == (
        cash["input"]["book_balance_minor"] - bank_fees
    )
    assert output["adjusted_bank_balance_minor"] == (
        cash["input"]["bank_balance_minor"] + deposits
    )
    assert output["difference_minor"] == 0
    bank_events = {event["id"]: event for event in cash["input"]["bank_events"]}
    book_entries = {entry["id"]: entry for entry in cash["input"]["book_entries"]}
    for exception in output["exceptions"]:
        for event_id in exception["event_ids"]:
            if exception["type"] == "bank_fee":
                assert abs(bank_events[event_id]["amount_minor"]) == exception[
                    "amount_minor"
                ]
            elif exception["type"] == "deposit_in_transit":
                assert book_entries[event_id]["amount_minor"] == exception[
                    "amount_minor"
                ]
