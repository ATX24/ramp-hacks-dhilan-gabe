"""Mutation tests: each leakage detector must catch a deliberate inject."""

from __future__ import annotations

from distillery.contracts.tasks import FinanceTaskEnvelope, SplitName, TaskId
from distillery.data.generate import CORPUS_SMOKE, generate_corpus
from distillery.data.leakage import (
    check_leakage,
    estimated_jaccard,
    example_fingerprint,
    minhash_signature,
    normalized_content_hash,
)
from distillery.data.world import IID_TEMPLATE_FAMILIES, OOD_TEMPLATE_FAMILIES


def _clone(ex: FinanceTaskEnvelope, **updates: object) -> FinanceTaskEnvelope:
    data = ex.model_dump(mode="json")
    for key, value in updates.items():
        if key == "provenance" and isinstance(value, dict):
            data["provenance"] = {**data["provenance"], **value}
        elif key == "input" and isinstance(value, dict):
            data["input"] = {**data["input"], **value}
        else:
            data[key] = value
    return FinanceTaskEnvelope.model_validate(data)


def test_exact_normalized_duplicate_detected(smoke_corpus) -> None:
    base = smoke_corpus.examples[0]
    twin = _clone(base, example_id="ex_mut_dup_00001", world_id="world_mut_dup_1")
    # Force identical normalized material (same input/output/task/difficulty).
    report = check_leakage([base, twin], check_near_duplicates=False)
    assert not report.ok
    assert any(f.kind == "exact_normalized_duplicate" for f in report.findings)


def test_cross_split_world_id_detected(smoke_corpus) -> None:
    train = next(
        e for e in smoke_corpus.examples if e.provenance.split == SplitName.TRAIN
    )
    leaked = _clone(
        next(e for e in smoke_corpus.examples if e.provenance.split == SplitName.TEST),
        example_id="ex_mut_world_leak_001",
        world_id=train.world_id,
        group_id="grp_mut_world_leak",
        input={**train.input, "case_nonce": "mut-world-leak", "entity_id": "ent_mut_world"},
    )
    report = check_leakage(
        [train, leaked],
        check_near_duplicates=False,
    )
    assert any(f.kind == "cross_split_id" for f in report.findings)


def test_cross_split_group_id_detected(smoke_corpus) -> None:
    train = next(
        e for e in smoke_corpus.examples if e.provenance.split == SplitName.TRAIN
    )
    val = next(
        e for e in smoke_corpus.examples if e.provenance.split == SplitName.VALIDATION
    )
    leaked = _clone(
        val,
        example_id="ex_mut_grp_leak_001",
        group_id=train.group_id,
        world_id="world_mut_grp_leak",
        input={**val.input, "case_nonce": "mut-grp", "entity_id": "ent_mut_grp"},
    )
    report = check_leakage([train, leaked], check_near_duplicates=False)
    assert any(
        f.kind == "cross_split_id" and "group_id" in f.detail for f in report.findings
    )


def test_ood_template_leak_detected() -> None:
    corpus = generate_corpus(CORPUS_SMOKE, check_near_duplicates=False)
    train = next(e for e in corpus.examples if e.provenance.split == SplitName.TRAIN)
    ood_family = OOD_TEMPLATE_FAMILIES[TaskId.TRANSACTION_REVIEW][0]
    assert ood_family not in IID_TEMPLATE_FAMILIES[TaskId.TRANSACTION_REVIEW]

    ood_clone = _clone(
        train,
        example_id="ex_mut_ood_tmpl_001",
        world_id="world_mut_ood_tmpl",
        group_id="grp_mut_ood_tmpl",
        provenance={"split": SplitName.OOD_TEST.value, "template_family": ood_family},
        input={**train.input, "case_nonce": "ood-tmpl", "entity_id": "ent_mut_ood"},
    )
    contaminated = _clone(
        train,
        example_id="ex_mut_ood_tmpl_002",
        world_id="world_mut_ood_tmpl2",
        group_id="grp_mut_ood_tmpl2",
        provenance={"template_family": ood_family},
        input={**train.input, "case_nonce": "iid-with-ood-tmpl", "entity_id": "ent_mut_ood2"},
    )
    report = check_leakage([ood_clone, contaminated], check_near_duplicates=False)
    assert any(f.kind == "cross_split_template" for f in report.findings)


def test_near_duplicate_cross_split_detected(smoke_corpus) -> None:
    train = next(
        e for e in smoke_corpus.examples if e.provenance.split == SplitName.TRAIN
    )
    # Near-copy into test: tiny descriptor edit, same numeric skeleton.
    near = _clone(
        train,
        example_id="ex_mut_near_001",
        world_id="world_mut_near_001",
        group_id="grp_mut_near_001",
        provenance={"split": SplitName.TEST.value},
        input={
            **train.input,
            "case_nonce": train.input["case_nonce"],  # identical nonce → near copy
            "entity_id": "ent_mut_near",
            "descriptor": train.input.get("descriptor", "") + " ",
        },
    )
    score = estimated_jaccard(
        minhash_signature(example_fingerprint(train)),
        minhash_signature(example_fingerprint(near)),
    )
    assert score >= 0.85
    report = check_leakage([train, near], check_near_duplicates=True)
    assert any(f.kind == "cross_split_near_duplicate" for f in report.findings)


def test_normalized_hash_ignores_example_id(smoke_corpus) -> None:
    a = smoke_corpus.examples[0]
    b = _clone(a, example_id="ex_mut_hash_ignore_001")
    assert normalized_content_hash(a) == normalized_content_hash(b)


def test_smoke_corpus_passes_leakage(smoke_corpus) -> None:
    report = check_leakage(smoke_corpus.examples, check_near_duplicates=True)
    fatal = [
        f
        for f in report.findings
        if f.kind
        in {
            "exact_normalized_duplicate",
            "cross_split_id",
            "cross_split_template",
            "cross_split_near_duplicate",
        }
    ]
    assert fatal == []
