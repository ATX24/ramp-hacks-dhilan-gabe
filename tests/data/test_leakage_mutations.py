"""Mutation tests proving every isolation and answer-leak detector fires."""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from distillery.contracts.tasks import FinanceTaskEnvelope, SplitName, TaskId
from distillery.data.leakage import (
    check_leakage,
    estimated_jaccard,
    example_fingerprint,
    minhash_signature,
    normalized_content_hash,
)

_ID_RE = re.compile(
    r"\b(?:world|grp|ent|txn|vnd|src|bok|bnk|ex)_[0-9a-f]{8,}\b",
    re.IGNORECASE,
)
_RULE_RE = re.compile(r"\b(?:POL|VAR)-[A-Z0-9-]+-[A-F0-9]{8}\b")


def _clone(
    example: FinanceTaskEnvelope,
    *,
    input_updates: dict[str, Any] | None = None,
    provenance_updates: dict[str, Any] | None = None,
    **updates: Any,
) -> FinanceTaskEnvelope:
    data = example.model_dump(mode="json")
    data.update(updates)
    if input_updates:
        data["input"] = {**data["input"], **input_updates}
    if provenance_updates:
        data["provenance"] = {
            **data["provenance"],
            **provenance_updates,
        }
    return FinanceTaskEnvelope.model_validate(data)


def _different_group_pair(
    corpus,
    task: TaskId,
) -> tuple[FinanceTaskEnvelope, FinanceTaskEnvelope]:
    candidates = [example for example in corpus.examples if example.task == task]
    first = candidates[0]
    second = next(example for example in candidates if example.group_id != first.group_id)
    return first, second


def _kinds(*examples: FinanceTaskEnvelope) -> set[str]:
    return check_leakage(examples, check_near_duplicates=False).by_kind


def test_vendor_name_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    mutated = _clone(second, input_updates={"vendor": first.input["vendor"]})
    assert "vendor_name_overlap" in _kinds(first, mutated)


def test_merchant_name_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    first = _clone(first, input_updates={"merchant": "Shared Merchant"})
    second = _clone(second, input_updates={"merchant": "Shared Merchant"})
    assert "merchant_name_overlap" in _kinds(first, second)


def test_descriptor_corruption_family_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    mutated = _clone(
        second,
        input_updates={
            "descriptor": first.input["descriptor"],
            "vendor": first.input["vendor"],
        },
    )
    assert "descriptor_family_overlap" in _kinds(first, mutated)


@pytest.mark.parametrize("task", [TaskId.VARIANCE_ANALYSIS, TaskId.CASH_RECONCILIATION])
def test_period_overlap_detector(smoke_corpus, task: TaskId) -> None:
    first, second = _different_group_pair(smoke_corpus, task)
    key = "period" if task == TaskId.VARIANCE_ANALYSIS else "close_period"
    mutated = _clone(second, input_updates={key: first.input[key]})
    assert "period_overlap" in _kinds(first, mutated)


def test_driver_source_id_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.VARIANCE_ANALYSIS,
    )
    observations = copy.deepcopy(second.model_dump(mode="json")["input"]["driver_observations"])
    observations[0]["source_id"] = first.input["driver_observations"][0]["source_id"]
    mutated = _clone(
        second,
        input_updates={"driver_observations": observations},
    )
    assert "source_id_overlap" in _kinds(first, mutated)


@pytest.mark.parametrize("collection", ["book_entries", "bank_events"])
def test_book_bank_source_id_overlap_detector(
    smoke_corpus,
    collection: str,
) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.CASH_RECONCILIATION,
    )
    values = copy.deepcopy(second.model_dump(mode="json")["input"][collection])
    values[0]["id"] = first.input[collection][0]["id"]
    mutated = _clone(second, input_updates={collection: values})
    assert "source_id_overlap" in _kinds(first, mutated)


def test_policy_text_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    rules = copy.deepcopy(second.model_dump(mode="json")["input"]["policy_rules"])
    rules[0]["text"] = first.input["policy_rules"][0]["text"]
    mutated = _clone(second, input_updates={"policy_rules": rules})
    assert "policy_text_overlap" in _kinds(first, mutated)


def test_coa_description_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    chart = copy.deepcopy(second.model_dump(mode="json")["input"]["chart_of_accounts"])
    chart[0]["name"] = first.input["chart_of_accounts"][0]["name"]
    mutated = _clone(second, input_updates={"chart_of_accounts": chart})
    assert "coa_description_overlap" in _kinds(first, mutated)


def test_renderer_template_family_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    mutated = _clone(
        second,
        provenance_updates={
            "template_family": first.provenance.template_family,
        },
    )
    assert "template_family_overlap" in _kinds(first, mutated)


def test_numeric_case_overlap_detector(smoke_corpus) -> None:
    first, second = _different_group_pair(
        smoke_corpus,
        TaskId.TRANSACTION_REVIEW,
    )
    mutated = _clone(
        second,
        input_updates={
            "amount_minor": first.input["amount_minor"],
            "cost_center": first.input["cost_center"],
            "date": first.input["date"],
            "expense_category": first.input["expense_category"],
            "vendor": first.input["vendor"],
        },
    )
    assert "numeric_case_overlap" in _kinds(first, mutated)


def test_target_hint_and_direct_copy_detectors(smoke_corpus) -> None:
    variance = next(
        example for example in smoke_corpus.examples if example.task == TaskId.VARIANCE_ANALYSIS
    )
    mutated = _clone(
        variance,
        input_updates={"impact_hint_minor": variance.expected_output["profit_impact_minor"]},
    )
    kinds = _kinds(mutated)
    assert "forbidden_input_field" in kinds
    assert "target_helper_field" in kinds
    assert "direct_target_copy" in kinds


@pytest.mark.parametrize(
    ("key", "value", "expected_kind"),
    [
        ("case_nonce", "arbitrary", "forbidden_input_field"),
        ("debug_note", "full_ood", "split_or_ood_marker"),
        ("debug_note", "held-out sample", "split_or_ood_marker"),
    ],
)
def test_nonce_and_split_marker_detectors(
    smoke_corpus,
    key: str,
    value: str,
    expected_kind: str,
) -> None:
    example = smoke_corpus.examples[0]
    mutated = _clone(example, input_updates={key: value})
    assert expected_kind in _kinds(mutated)


def test_semantic_clone_changed_ids_and_nonce_is_detected(smoke_corpus) -> None:
    base = next(
        example for example in smoke_corpus.examples if example.task == TaskId.TRANSACTION_REVIEW
    )
    data = base.model_dump(mode="json")
    counter = 0

    def replace_string(value: str) -> str:
        nonlocal counter

        def replace_id(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            prefix = match.group(0).split("_", maxsplit=1)[0]
            return f"{prefix}_{counter:018x}"

        def replace_rule(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            stem = match.group(0).rsplit("-", maxsplit=1)[0]
            return f"{stem}-{counter:08X}"

        return _RULE_RE.sub(replace_rule, _ID_RE.sub(replace_id, value))

    def rekey(value: Any) -> Any:
        if isinstance(value, str):
            return replace_string(value)
        if isinstance(value, list):
            return [rekey(item) for item in value]
        if isinstance(value, dict):
            return {key: rekey(item) for key, item in value.items()}
        return value

    clone_data = rekey(data)
    clone_data["input"]["case_nonce"] = "identity-only-change"
    clone = FinanceTaskEnvelope.model_validate(clone_data)
    assert normalized_content_hash(base) == normalized_content_hash(clone)
    report = check_leakage([base, clone], check_near_duplicates=False)
    assert "exact_normalized_duplicate" in report.by_kind
    assert "forbidden_input_field" in report.by_kind


@pytest.mark.parametrize("cross_split", [False, True])
def test_near_duplicate_detector_is_fatal_within_or_across_split(
    smoke_corpus,
    cross_split: bool,
) -> None:
    base = next(
        example for example in smoke_corpus.examples if example.task == TaskId.TRANSACTION_REVIEW
    )
    updates: dict[str, Any] = {
        "world_id": "world_aaaaaaaaaaaaaaaaaa",
        "group_id": "grp_bbbbbbbbbbbbbbbbbb",
        "example_id": "ex_cccccccccccccccccc",
    }
    provenance = (
        {"split": SplitName.TEST.value} if cross_split else {"split": base.provenance.split.value}
    )
    near = _clone(
        base,
        input_updates={"amount_minor": base.input["amount_minor"] + 1},
        provenance_updates=provenance,
        **updates,
    )
    score = estimated_jaccard(
        minhash_signature(example_fingerprint(base)),
        minhash_signature(example_fingerprint(near)),
    )
    assert score >= 0.85
    report = check_leakage([base, near], check_near_duplicates=True)
    expected = "cross_split_near_duplicate" if cross_split else "same_split_near_duplicate"
    assert expected in report.by_kind
    assert not report.ok


def test_full_corpus_has_no_leakage_findings(full_corpus) -> None:
    assert full_corpus.leakage.ok
    assert full_corpus.leakage.findings == []
    assert full_corpus.leakage.near_duplicate_pairs == 0
