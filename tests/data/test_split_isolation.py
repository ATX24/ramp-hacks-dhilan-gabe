"""Grouped split isolation and OOD holdout tests."""

from __future__ import annotations

from distillery.contracts.tasks import SplitName, TaskId
from distillery.data.generate import CORPUS_FULL, generate_corpus
from distillery.data.world import IID_TEMPLATE_FAMILIES, OOD_TEMPLATE_FAMILIES


def test_smoke_no_cross_split_ids(smoke_corpus) -> None:
    by_split: dict[SplitName, set[str]] = {}
    for split, items in smoke_corpus.by_split.items():
        keys: set[str] = set()
        for ex in items:
            keys.add(ex.world_id)
            keys.add(ex.group_id)
            if ent := ex.input.get("entity_id"):
                keys.add(f"ent:{ent}")
            if txn := ex.input.get("txn_id"):
                keys.add(f"txn:{txn}")
        by_split[split] = keys

    splits = list(by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = by_split[left] & by_split[right]
            assert not overlap, f"{left}∩{right}={sorted(overlap)[:5]}"


def test_full_ood_holds_out_templates_and_regimes() -> None:
    corpus = generate_corpus(CORPUS_FULL, check_near_duplicates=True)
    assert len(corpus.examples) == 5200
    assert len(corpus.by_split[SplitName.TRAIN]) == 3200
    assert len(corpus.by_split[SplitName.VALIDATION]) == 400
    assert len(corpus.by_split[SplitName.IID_TEST]) == 800
    assert len(corpus.by_split[SplitName.OOD_TEST]) == 800

    in_dist_templates: set[str] = set()
    ood_templates: set[str] = set()
    for ex in corpus.examples:
        if ex.provenance.split == SplitName.OOD_TEST:
            ood_templates.add(ex.provenance.template_family)
        else:
            in_dist_templates.add(ex.provenance.template_family)

    assert in_dist_templates.isdisjoint(ood_templates)
    for task in (
        TaskId.TRANSACTION_REVIEW,
        TaskId.VARIANCE_ANALYSIS,
        TaskId.CASH_RECONCILIATION,
    ):
        ood_set = set(OOD_TEMPLATE_FAMILIES[task])
        assert ood_set
        assert ood_set.isdisjoint(in_dist_templates)
        assert set(IID_TEMPLATE_FAMILIES[task]).isdisjoint(ood_templates)

    # Mixture targets recorded.
    train_mix = corpus.manifest["mixtures"]["train"]["mixture"]["by_task"]
    assert train_mix == {
        "transaction_review": 1440,
        "variance_analysis": 1440,
        "cash_reconciliation": 320,
    }


def test_envelopes_round_trip_contract(smoke_corpus) -> None:
    from distillery.contracts.tasks import FinanceTaskEnvelope

    for ex in smoke_corpus.examples[:20]:
        again = FinanceTaskEnvelope.model_validate(ex.model_dump(mode="json"))
        assert again.example_id == ex.example_id
        assert again.oracle.latent_state_hash.startswith("sha256:")
