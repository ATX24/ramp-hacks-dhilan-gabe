"""Property and adversarial coverage for merchant_tagging (Primary C)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest
from pydantic import ValidationError

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import (
    MERCHANT_FINANCE_TAGS,
    MERCHANT_SPEND_CATEGORIES,
    Difficulty,
    FinanceTaskEnvelope,
    LabelSource,
    MerchantTaggingOutput,
    Provenance,
    SplitName,
    TaskId,
    validate_task_output,
)
from distillery.data.campaign import build_campaign_manifest
from distillery.data.generate import (
    CORPUS_FULL_V2,
    CORPUS_SMOKE,
    CORPUS_SMOKE_V2,
    generate_corpus,
)
from distillery.data.hard_negatives import MERCHANT_HARD_NEGATIVES
from distillery.data.mixture import (
    TASK_MIXTURE,
    TASK_MIXTURE_V2,
    TASK_ORDER,
    TASK_ORDER_V2,
    mixture_plan,
    task_counts,
)
from distillery.data.oracle import (
    GENERATOR_REVISION_V2,
    oracle_meta,
    solve_merchant_tagging,
    solve_task,
)
from distillery.data.renderers import render_input, select_template_family
from distillery.data.validate import validate_example, validate_output
from distillery.data.world import (
    IID_MERCHANT_FAMILY_SPECS,
    MCC_CATEGORY_MAP,
    OOD_MERCHANT_FAMILY_SPECS,
    MerchantHardNegative,
    build_world,
)
from distillery.proof.metrics import (
    PredictionRecord,
    compute_arm_metrics,
    compute_primary_index,
    compute_primary_index_v2,
    score_prediction,
)
from distillery.proof.protocol_v2 import finance_proof_v2_document, finance_proof_v2_sha256
from distillery.training.batching import DEFAULT_FINANCE_MIXTURE, FINANCE_MIXTURE_V2


def _merchant_world(
    *,
    index: int = 0,
    difficulty: Difficulty = Difficulty.HARD,
    ood: bool = False,
    salt: int = 0,
):
    return build_world(
        seed=17,
        index=index,
        split_token="merch_probe",
        task=TaskId.MERCHANT_TAGGING,
        difficulty=difficulty,
        ood=ood,
        salt=salt,
    )


def test_mcc_category_map_is_closed_and_consistent() -> None:
    assert set(MCC_CATEGORY_MAP.values()) <= MERCHANT_SPEND_CATEGORIES
    for specs in (IID_MERCHANT_FAMILY_SPECS, OOD_MERCHANT_FAMILY_SPECS):
        for family_id, _name, mcc, tags in specs:
            assert mcc in MCC_CATEGORY_MAP, family_id
            assert set(tags) <= MERCHANT_FINANCE_TAGS, family_id


def test_oracle_enforces_mcc_category_and_sorted_tags() -> None:
    world = _merchant_world(difficulty=Difficulty.EASY)
    output = solve_merchant_tagging(world)
    assert output.spend_category == MCC_CATEGORY_MAP[world.merchant.mcc]
    assert list(output.tags) == sorted(output.tags)
    assert set(output.tags) <= MERCHANT_FINANCE_TAGS


@pytest.mark.parametrize("corruption", sorted(MERCHANT_HARD_NEGATIVES, key=lambda c: c.value))
def test_hard_negative_corruption_templates_render_without_label_leak(
    corruption: MerchantHardNegative,
) -> None:
    found = False
    for index in range(80):
        world = _merchant_world(index=index, difficulty=Difficulty.HARD)
        if world.merchant is None or world.merchant.corruption_template != corruption:
            continue
        found = True
        family = select_template_family(
            TaskId.MERCHANT_TAGGING,
            difficulty=Difficulty.HARD,
            ood=False,
            index=index,
            family_key=world.group_id,
        )
        rendered = render_input(world, TaskId.MERCHANT_TAGGING, template_family=family)
        expected = solve_task(world, TaskId.MERCHANT_TAGGING)
        assert "merchant_id" not in rendered
        assert "merchant_name" not in rendered
        assert "spend_category" not in rendered
        assert "tags" not in rendered
        assert rendered["descriptor"]
        assert rendered["mcc"]
        result = validate_output(TaskId.MERCHANT_TAGGING, expected)
        assert result.ok, result.errors
        break
    assert found, f"did not sample corruption {corruption}"


def test_held_out_ood_merchant_families_disjoint_from_iid() -> None:
    iid = {family_id for family_id, *_ in IID_MERCHANT_FAMILY_SPECS}
    ood = {family_id for family_id, *_ in OOD_MERCHANT_FAMILY_SPECS}
    assert iid.isdisjoint(ood)
    iid_world = _merchant_world(index=3, ood=False)
    ood_world = _merchant_world(index=3, ood=True)
    assert iid_world.merchant is not None
    assert ood_world.merchant is not None
    assert iid_world.merchant.family_id in iid
    assert ood_world.merchant.family_id in ood


def test_noisy_descriptors_differ_from_canonical_name_under_corruption() -> None:
    world = _merchant_world(index=11, difficulty=Difficulty.HARD)
    assert world.merchant is not None
    if world.merchant.corruption_template == MerchantHardNegative.NONE:
        pytest.skip("sampled none corruption")
    assert world.merchant.noisy_descriptor.casefold() != world.merchant.merchant_name.casefold()


def test_exact_match_and_tag_macro_f1_metrics() -> None:
    world = _merchant_world(index=5, difficulty=Difficulty.MEDIUM)
    gold = solve_task(world, TaskId.MERCHANT_TAGGING)
    perfect = PredictionRecord(
        example_id="ex_merch_perfect",
        world_id=world.world_id,
        group_id=world.group_id,
        task=TaskId.MERCHANT_TAGGING.value,
        difficulty="medium",
        split="iid_test",
        arm_id="oracle",
        seed=17,
        raw_text=json.dumps(gold, sort_keys=True),
        raw_text_provenance="fixture_serialization",
        expected_output=gold,
    )
    wrong = dict(gold)
    wrong["merchant_id"] = "mrc_000000000000000000"
    wrong["spend_category"] = "saas" if gold["spend_category"] != "saas" else "meals"
    wrong["tags"] = sorted(["software", "recurring"])
    imperfect = PredictionRecord(
        example_id="ex_merch_imperfect",
        world_id=world.world_id,
        group_id=world.group_id,
        task=TaskId.MERCHANT_TAGGING.value,
        difficulty="medium",
        split="iid_test",
        arm_id="student",
        seed=17,
        raw_text=json.dumps(wrong, sort_keys=True),
        raw_text_provenance="fixture_serialization",
        expected_output=gold,
    )
    perfect_score = score_prediction(perfect)
    imperfect_score = score_prediction(imperfect)
    assert perfect_score.joint_exact is True
    assert perfect_score.components["merchant_exact"] == 1.0
    assert imperfect_score.joint_exact is False
    metrics = compute_arm_metrics("probe", [perfect, imperfect])
    assert metrics.merchant_joint_exact == 0.5
    assert metrics.task_metrics["merchant_tagging"]["category_macro_f1"] is not None
    assert metrics.task_metrics["merchant_tagging"]["tag_macro_f1"] is not None
    assert metrics.primary_index_v2 is not None
    assert metrics.primary_index_v2 == compute_primary_index_v2(
        None, None, 0.5, metrics.json_schema_validity
    )


def test_v1_primary_index_unchanged_when_merchant_present() -> None:
    # Merchant examples must not silently rewrite the v1 index formula.
    assert compute_primary_index(1.0, 0.0, 1.0) == 0.55
    assert compute_primary_index_v2(1.0, 0.0, 1.0, 1.0) == 0.65


def test_smoke_v2_mixture_and_leakage() -> None:
    corpus = generate_corpus(CORPUS_SMOKE_V2, check_near_duplicates=True)
    assert corpus.manifest["envelope_schema_version"] == "finance_world.v2"
    assert corpus.manifest["generator_revision"] == GENERATOR_REVISION_V2
    counts = Counter(example.task for example in corpus.examples)
    assert counts[TaskId.MERCHANT_TAGGING] == 112
    assert counts[TaskId.TRANSACTION_REVIEW] == 196
    assert counts[TaskId.VARIANCE_ANALYSIS] == 196
    assert counts[TaskId.CASH_RECONCILIATION] == 56
    assert corpus.leakage.ok
    assert all(example.schema_version == "finance_world.v2" for example in corpus.examples)


def test_full_v2_has_at_least_1000_merchant_examples() -> None:
    expected = task_counts(6240, mixture=TASK_MIXTURE_V2, order=TASK_ORDER_V2)
    assert expected[TaskId.MERCHANT_TAGGING] >= 1000
    assert expected[TaskId.MERCHANT_TAGGING] == 1248
    assert expected[TaskId.TRANSACTION_REVIEW] == 2184
    assert expected[TaskId.VARIANCE_ANALYSIS] == 2184
    plan = mixture_plan(960, mixture=TASK_MIXTURE_V2, order=TASK_ORDER_V2)
    assert sum(1 for task, _ in plan if task == TaskId.MERCHANT_TAGGING) == 192


def test_v1_smoke_hashes_still_generate_and_exclude_merchant() -> None:
    corpus = generate_corpus(CORPUS_SMOKE, check_near_duplicates=True)
    assert corpus.manifest["schema_version"] == "finance_world.v1.corpus_manifest"
    assert TaskId.MERCHANT_TAGGING not in {example.task for example in corpus.examples}
    assert corpus.manifest["task_mixture_target"] == {
        "transaction_review": 0.45,
        "variance_analysis": 0.45,
        "cash_reconciliation": 0.10,
    }


def test_corpus_specs_enforce_versioned_task_sets() -> None:
    with pytest.raises(ValueError, match="finance_world.v2 task set"):
        replace(
            CORPUS_SMOKE_V2,
            task_mixture=TASK_MIXTURE,
            task_order=TASK_ORDER,
        )
    with pytest.raises(ValueError, match="finance_world.v1 task set"):
        replace(
            CORPUS_SMOKE,
            task_mixture=TASK_MIXTURE_V2,
            task_order=TASK_ORDER_V2,
        )


def test_v2_campaign_rejects_v1_corpus(smoke_corpus) -> None:
    with pytest.raises(ValueError, match="does not match campaign world"):
        build_campaign_manifest(
            world="finance_world.v2",
            corpus="smoke",
            campaign_id="camp_wrong_world",
            generated=smoke_corpus,
        )


def test_same_artifact_no_specialist_routing_in_campaign_manifest() -> None:
    smoke = generate_corpus(CORPUS_SMOKE_V2, check_near_duplicates=False)
    manifest = build_campaign_manifest(
        world="finance_world.v2",
        corpus="smoke",
        campaign_id="camp_tinyfable_v2_probe",
        generated=smoke,
    )
    assert manifest["specialist_routing"] is False
    assert manifest["routing_policy"] == "same_artifact_all_primary_tasks"
    assert manifest["shared_artifact_id"] == "tinyfable_generalist"
    assert manifest["proof_protocol"]["id"] == "finance-proof.v2"
    assert manifest["proof_protocol"]["sha256"] == finance_proof_v2_sha256()
    assert set(FINANCE_MIXTURE_V2.task_weights) == set(manifest["sampler_mixture"])
    assert "merchant_tagging" not in DEFAULT_FINANCE_MIXTURE.task_weights
    # Re-seal stability.
    again = build_campaign_manifest(
        world="finance_world.v2",
        corpus="smoke",
        campaign_id="camp_tinyfable_v2_probe",
        generated=smoke,
    )
    assert again["manifest_sha256"] == manifest["manifest_sha256"]


def test_proof_protocol_v2_document_is_content_addressed() -> None:
    doc = finance_proof_v2_document()
    assert doc["id"] == "finance-proof.v2"
    assert doc["specialist_routing"] is False
    assert doc["cash_in_primary_index"] is False
    assert doc["full_total_examples"] == 6240
    assert doc["min_full_merchant_examples"] == 1000
    assert finance_proof_v2_sha256() == content_sha256(doc)


def test_merchant_output_rejects_unsorted_and_unknown_tags() -> None:
    with pytest.raises(ValidationError):
        MerchantTaggingOutput(
            merchant_id="mrc_abcdef0123456789ab",
            merchant_name="Probe",
            spend_category="meals",
            tags=("entertainment", "employee_spend"),
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        validate_task_output(
            {
                "schema_version": "merchant_tagging.v1",
                "task": "merchant_tagging",
                "merchant_id": "mrc_abcdef0123456789ab",
                "merchant_name": "Probe",
                "spend_category": "meals",
                "tags": ["not_a_tag"],
                "confidence": 0.5,
            }
        )


def test_mcc_near_miss_tolerated_in_example_validation() -> None:
    for index in range(120):
        world = _merchant_world(index=index, difficulty=Difficulty.HARD)
        if (
            world.merchant is None
            or world.merchant.corruption_template != MerchantHardNegative.MCC_NEAR_MISS
        ):
            continue
        family = select_template_family(
            TaskId.MERCHANT_TAGGING,
            difficulty=Difficulty.HARD,
            ood=False,
            index=index,
            family_key=world.group_id,
        )
        env_input = render_input(world, TaskId.MERCHANT_TAGGING, template_family=family)
        expected = solve_task(world, TaskId.MERCHANT_TAGGING)
        assert env_input["mcc"] != world.merchant.mcc
        assert expected["spend_category"] == world.merchant.spend_category

        envelope = FinanceTaskEnvelope(
            schema_version="finance_world.v2",
            example_id="ex_mcc_near_miss_probe",
            world_id=world.world_id,
            group_id=world.group_id,
            task=TaskId.MERCHANT_TAGGING,
            difficulty=Difficulty.HARD,
            input=env_input,
            expected_output=expected,
            oracle=oracle_meta(world, generator_revision=GENERATOR_REVISION_V2),
            provenance=Provenance(
                split=SplitName.TRAIN,
                template_family=family,
                label_source=LabelSource.ORACLE,
            ),
        )
        result = validate_example(envelope)
        assert result.ok, result.errors
        assert "mcc_category_near_miss_tolerated" in result.checks
        return
    pytest.fail("failed to sample MCC_NEAR_MISS")


def test_full_corpus_spec_declared_sizes() -> None:
    assert CORPUS_FULL_V2.total_examples == 6240
    assert CORPUS_SMOKE_V2.total_examples == 560
