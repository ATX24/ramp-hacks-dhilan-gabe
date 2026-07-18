"""Grouped isolation and unlabeled OOD combination holdouts."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

from distillery.contracts.tasks import FinanceTaskEnvelope, SplitName, TaskId


def test_identity_and_semantic_assets_are_group_isolated(full_corpus) -> None:
    extractors: dict[str, Callable[[FinanceTaskEnvelope], list[str]]] = {
        "world": lambda example: [example.world_id],
        "entity": lambda example: [str(example.input.get("entity_id", ""))],
        "vendor": lambda example: [str(example.input.get("vendor", ""))],
        "period": _periods,
        "source": _source_ids,
        "policy_text": _policy_texts,
        "coa_description": _coa_descriptions,
        "template_family": lambda example: [example.provenance.template_family],
    }
    for name, extractor in extractors.items():
        owners: dict[str, set[str]] = defaultdict(set)
        splits: dict[str, set[SplitName]] = defaultdict(set)
        for example in full_corpus.examples:
            for value in extractor(example):
                if value:
                    owners[value].add(example.group_id)
                    splits[value].add(example.provenance.split)
        assert all(len(groups) == 1 for groups in owners.values()), name
        assert all(len(values) == 1 for values in splits.values()), name


def test_group_ids_never_cross_splits(full_corpus) -> None:
    owners: dict[str, set[SplitName]] = defaultdict(set)
    for example in full_corpus.examples:
        owners[example.group_id].add(example.provenance.split)
    assert all(len(splits) == 1 for splits in owners.values())


def test_ood_prompts_and_inputs_do_not_label_distribution(full_corpus) -> None:
    forbidden = re.compile(
        r"\bood\b|held[- ]out|out[- ]of[- ]distribution",
        re.IGNORECASE,
    )
    for example in full_corpus.by_split[SplitName.OOD_TEST]:
        rendered = str(example.input)
        assert not forbidden.search(rendered)
        assert "regime" not in example.input


def test_ood_holds_out_policy_combinations(full_corpus) -> None:
    iid_actions: set[str] = set()
    ood_actions: set[str] = set()
    for example in full_corpus.examples:
        if example.task != TaskId.TRANSACTION_REVIEW:
            continue
        base_cloud_rules = [
            rule
            for rule in example.input["policy_rules"]
            if rule["category"] == "cloud" and rule["min_amount_minor"] == 0
        ]
        if not base_cloud_rules:
            continue
        target = ood_actions if example.provenance.split == SplitName.OOD_TEST else iid_actions
        target.update(rule["action"] for rule in base_cloud_rules)
    assert iid_actions == {"approve"}
    assert ood_actions == {"review"}


def test_ood_holds_out_gl_description_combinations(full_corpus) -> None:
    iid: set[tuple[str, str]] = set()
    ood: set[tuple[str, str]] = set()
    for example in full_corpus.examples:
        if example.task != TaskId.TRANSACTION_REVIEW:
            continue
        target = ood if example.provenance.split == SplitName.OOD_TEST else iid
        target.update(
            (str(account["code"]), str(account["name"]))
            for account in example.input["chart_of_accounts"]
        )
    assert iid
    assert ood
    assert iid.isdisjoint(ood)


def test_ood_holds_out_variance_driver_sign_mixtures(full_corpus) -> None:
    iid: set[tuple[str, str, int]] = set()
    ood: set[tuple[str, str, int]] = set()
    for example in full_corpus.examples:
        if example.task != TaskId.VARIANCE_ANALYSIS:
            continue
        target = ood if example.provenance.split == SplitName.OOD_TEST else iid
        for item in example.input["driver_observations"]:
            raw = (
                item["budget_minor"] - item["actual_minor"]
                if item["pnl_type"] == "expense"
                else item["actual_minor"] - item["budget_minor"]
            )
            sign = 1 if raw > 0 else -1 if raw < 0 else 0
            target.add((item["driver_id"], item["pnl_type"], sign))
    assert iid
    assert ood
    assert iid.isdisjoint(ood)


def test_ood_holds_out_cash_aggregation_patterns(full_corpus) -> None:
    iid_patterns: set[tuple[int, int]] = set()
    ood_patterns: set[tuple[int, int]] = set()
    for example in full_corpus.examples:
        if example.task != TaskId.CASH_RECONCILIATION:
            continue
        target = ood_patterns if example.provenance.split == SplitName.OOD_TEST else iid_patterns
        target.update(
            (len(group["book_ids"]), len(group["bank_ids"]))
            for group in example.expected_output["matched_groups"]
        )
    assert (1, 2) in ood_patterns
    assert (2, 1) in ood_patterns
    assert (1, 2) not in iid_patterns
    assert (2, 1) not in iid_patterns


def test_envelopes_round_trip_foundation_contract(full_corpus) -> None:
    for example in full_corpus.examples[::257]:
        round_trip = FinanceTaskEnvelope.model_validate(example.model_dump(mode="json"))
        assert round_trip == example


def _periods(example: FinanceTaskEnvelope) -> list[str]:
    return [str(example.input[key]) for key in ("period", "close_period") if example.input.get(key)]


def _source_ids(example: FinanceTaskEnvelope) -> list[str]:
    values: list[str] = []
    if example.input.get("txn_id"):
        values.append(str(example.input["txn_id"]))
    for item in example.input.get("driver_observations", []):
        values.append(str(item["source_id"]))
    for item in example.input.get("unallocated_line_items", []):
        values.append(str(item["source_id"]))
    for collection in ("book_entries", "bank_events"):
        values.extend(str(item["id"]) for item in example.input.get(collection, []))
    return values


def _policy_texts(example: FinanceTaskEnvelope) -> list[str]:
    return [str(rule["text"]) for rule in example.input.get("policy_rules", [])]


def _coa_descriptions(example: FinanceTaskEnvelope) -> list[str]:
    return [str(account["name"]) for account in example.input.get("chart_of_accounts", [])]
