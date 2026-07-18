"""Typed proof, measured tier comparison, and registry publication gates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.hashing import content_sha256
from experiments.portfolio.plan import PortfolioArm, Task, Tier
from experiments.portfolio.registry import (
    artifact_publication_evidence,
    generalist_approval,
    promote_specialist_registry,
    publish_registry,
    register_replication_wave,
    registry_publication_gate,
)
from experiments.portfolio.selection import (
    BenchmarkMeasurement,
    ProofInterval,
    benchmark_measurement,
    build_multiplicity_plan,
    multiplicity_decision,
    proof_interval,
    ratio_interval,
    specialist_eligible,
    specialist_promotion_evidence,
    tier_eligible,
    tier_promotion_evidence,
)
from tests.portfolio.support import IMAGE_DIGEST, NOW, H, make_replication_wave


def _specialist_selection(plan):
    replication = make_replication_wave(plan)
    multiplicity = build_multiplicity_plan(plan, preregistered_at=NOW)
    screen_wave = plan.screen_waves[0]
    screen_treatment = next(
        slot
        for slot in screen_wave.active_slots
        if slot.role == "specialist"
        and slot.tasks == (Task.TRANSACTION_REVIEW,)
        and slot.arm is PortfolioArm.LOGIT_KD
    )
    screen_comparator = next(
        slot
        for slot in screen_wave.active_slots
        if slot.role == "generalist" and slot.arm is PortfolioArm.LOGIT_KD
    )
    replication_treatment = next(
        slot
        for slot in replication.active_slots
        if slot.role == "specialist" and slot.arm is PortfolioArm.LOGIT_KD
    )
    replication_comparator = next(
        slot
        for slot in replication.active_slots
        if slot.role == "generalist" and slot.arm is PortfolioArm.LOGIT_KD
    )
    view = plan.dataset.view_for((Task.TRANSACTION_REVIEW,))
    contrast_id = f"contrast_{screen_treatment.model_id}"
    interval_common = {
        "contrast_id": contrast_id,
        "metric": "transaction_review_primary_delta",
        "task": Task.TRANSACTION_REVIEW,
        "validation_view_id": view.view_id,
        "validation_view_sha256": view.view_sha256,
        "validation_split_sha256": view.split_sha256["validation"],
        "evaluation_manifest_sha256": H["9"],
        "split_access_log_sha256": H["a"],
        "evaluator_image_digest": IMAGE_DIGEST,
        "world_clusters_sha256": H["1"],
        "world_cluster_count": 80,
        "bootstrap_seed": 20260718,
        "point": 0.035,
        "lower": 0.021,
        "upper": 0.051,
        "proof_protocol_sha256": plan.protocol.proof_protocol_sha256,
    }
    screen_interval = proof_interval(
        **interval_common,
        treatment_model_id=screen_treatment.model_id,
        comparator_model_id=screen_comparator.model_id,
        treatment_arm=screen_treatment.arm,
        comparator_arm=screen_comparator.arm,
        treatment_run_id=screen_treatment.run_id,
        comparator_run_id=screen_comparator.run_id,
        treatment_protocol_sha256=screen_treatment.protocol_sha256,
        comparator_protocol_sha256=screen_comparator.protocol_sha256,
        training_seed=17,
    )
    replication_interval = proof_interval(
        **interval_common,
        treatment_model_id=replication_treatment.model_id,
        comparator_model_id=replication_comparator.model_id,
        treatment_arm=replication_treatment.arm,
        comparator_arm=replication_comparator.arm,
        treatment_run_id=replication_treatment.run_id,
        comparator_run_id=replication_comparator.run_id,
        treatment_protocol_sha256=replication_treatment.protocol_sha256,
        comparator_protocol_sha256=replication_comparator.protocol_sha256,
        training_seed=23,
    )
    decision = multiplicity_decision(
        multiplicity_plan_sha256=multiplicity.plan_sha256,
        contrast_id=contrast_id,
        family_size=multiplicity.family_size,
        holm_rank=1,
        raw_p_value=0.001,
        adjusted_p_value=0.024,
        hierarchy_gate_id="omnibus_nano_transaction_review",
        hierarchy_gate_proof_sha256=H["2"],
        hierarchy_gate_adjusted_p_value=0.03,
    )
    evidence = specialist_promotion_evidence(
        tier=Tier.NANO,
        task=Task.TRANSACTION_REVIEW,
        screen_interval=screen_interval,
        replication_interval=replication_interval,
        multiplicity=decision,
        screen_treatment_manifest_sha256=H["3"],
        screen_comparator_manifest_sha256=H["4"],
        replication_treatment_manifest_sha256=H["5"],
        replication_comparator_manifest_sha256=H["6"],
        screen_proof_report_sha256=H["7"],
        replication_proof_report_sha256=H["8"],
    )
    return replication, multiplicity, evidence


def test_specialist_promotion_requires_bound_seed17_and_seed23_evidence(plan) -> None:
    replication, multiplicity, evidence = _specialist_selection(plan)
    eligible, reason = specialist_eligible(
        plan,
        replication,
        multiplicity,
        evidence,
    )
    assert eligible is True
    assert "generalist remains default" in reason
    assert evidence.screen_interval.bootstrap_resamples == 10_000
    assert evidence.replication_interval.bootstrap_resamples == 10_000
    assert evidence.screen_interval.training_seed == 17
    assert evidence.replication_interval.training_seed == 23


def test_proof_interval_rejects_free_floats_test_data_and_wrong_bootstrap(plan) -> None:
    _, _, evidence = _specialist_selection(plan)
    payload = evidence.screen_interval.model_dump(
        mode="python",
        exclude={"interval_sha256"},
    )
    with pytest.raises(ValidationError, match="inside"):
        proof_interval(**{**payload, "point": 0.99})
    with pytest.raises(ValidationError, match="10000"):
        proof_interval(**{**payload, "bootstrap_resamples": 9_999})
    with pytest.raises(ValidationError, match="none_required"):
        ProofInterval.model_validate(
            {
                **payload,
                "test_dataset_sha256": H["1"],
                "interval_sha256": H["2"],
            }
        )


def test_specialist_promotion_rejects_unbound_comparator_and_multiplicity(plan) -> None:
    replication, multiplicity, evidence = _specialist_selection(plan)
    interval_payload = evidence.replication_interval.model_dump(
        mode="python",
        exclude={"interval_sha256"},
    )
    wrong_interval = proof_interval(**{**interval_payload, "comparator_protocol_sha256": H["f"]})
    wrong_evidence_payload = evidence.model_dump(
        mode="python",
        exclude={"evidence_sha256"},
    )
    wrong_evidence_payload.update(
        {
            "screen_interval": evidence.screen_interval,
            "replication_interval": wrong_interval,
            "multiplicity": evidence.multiplicity,
        }
    )
    wrong_evidence = specialist_promotion_evidence(**wrong_evidence_payload)
    assert (
        specialist_eligible(
            plan,
            replication,
            multiplicity,
            wrong_evidence,
        )[0]
        is False
    )
    failed_decision_payload = evidence.multiplicity.model_dump(
        mode="python",
        exclude={"decision_sha256"},
    )
    failed_decision = multiplicity_decision(
        **{
            **failed_decision_payload,
            "adjusted_p_value": 0.06,
            "hierarchy_gate_adjusted_p_value": 0.07,
        }
    )
    failed_evidence = specialist_promotion_evidence(
        **{
            **wrong_evidence_payload,
            "replication_interval": evidence.replication_interval,
            "multiplicity": failed_decision,
        }
    )
    assert (
        specialist_eligible(
            plan,
            replication,
            multiplicity,
            failed_evidence,
        )[0]
        is False
    )


def _generalist(plan, tier: Tier):
    return next(
        model
        for model in plan.models
        if model.tier is tier and model.role == "generalist" and model.arm is PortfolioArm.LOGIT_KD
    )


def _measurement(model, *, tier: Tier, throughput: float, cost: int):
    token_count = 100_000
    return benchmark_measurement(
        tier=tier,
        model_id=model.model_id,
        run_id=model.run_id,
        manifest_sha256=H["1"],
        proof_report_sha256=H["2"],
        instance_type="ml.p4de.24xlarge",
        hardware_profile="p4de-24xlarge-8xa100-80gb-independent-v1",
        accelerator="NVIDIA A100 80GB",
        runtime_image_digest=IMAGE_DIGEST,
        runtime_sha256=H["3"],
        harness_sha256=H["4"],
        token_count=token_count,
        request_count=200,
        throughput_tokens_per_second=throughput,
        total_cost_microusd=cost,
        cost_per_1k_tokens_microusd=(cost * 1000 + token_count - 1) // token_count,
        measured_at=NOW,
    )


def _tier_evidence(plan):
    candidate = _generalist(plan, Tier.CORE)
    incumbent = _generalist(plan, Tier.NANO)
    view = plan.dataset.view_for(tuple(Task))
    quality = proof_interval(
        contrast_id="contrast_core_vs_nano_generalist",
        metric="portfolio_primary_index_delta",
        treatment_model_id=candidate.model_id,
        comparator_model_id=incumbent.model_id,
        treatment_arm=candidate.arm,
        comparator_arm=incumbent.arm,
        treatment_run_id=candidate.run_id,
        comparator_run_id=incumbent.run_id,
        treatment_protocol_sha256=candidate.protocol_sha256,
        comparator_protocol_sha256=incumbent.protocol_sha256,
        training_seed=17,
        task="portfolio_primary_index",
        validation_view_id=view.view_id,
        validation_view_sha256=view.view_sha256,
        validation_split_sha256=view.split_sha256["validation"],
        evaluation_manifest_sha256=H["6"],
        split_access_log_sha256=H["7"],
        evaluator_image_digest=IMAGE_DIGEST,
        world_clusters_sha256=H["5"],
        world_cluster_count=120,
        bootstrap_seed=41,
        point=0.03,
        lower=0.021,
        upper=0.045,
        proof_protocol_sha256=plan.protocol.proof_protocol_sha256,
    )
    candidate_measurement = _measurement(
        candidate,
        tier=Tier.CORE,
        throughput=90.0,
        cost=1_800_000,
    )
    incumbent_measurement = _measurement(
        incumbent,
        tier=Tier.NANO,
        throughput=100.0,
        cost=1_500_000,
    )
    throughput = ratio_interval(
        metric="throughput_ratio",
        candidate_measurement_sha256=candidate_measurement.evidence_sha256,
        incumbent_measurement_sha256=incumbent_measurement.evidence_sha256,
        bootstrap_seed=42,
        point=0.90,
        lower=0.85,
        upper=0.95,
    )
    cost = ratio_interval(
        metric="cost_ratio",
        candidate_measurement_sha256=candidate_measurement.evidence_sha256,
        incumbent_measurement_sha256=incumbent_measurement.evidence_sha256,
        bootstrap_seed=43,
        point=1.20,
        lower=1.10,
        upper=1.24,
    )
    return tier_promotion_evidence(
        candidate=Tier.CORE,
        incumbent=Tier.NANO,
        quality_interval=quality,
        candidate_measurement=candidate_measurement,
        incumbent_measurement=incumbent_measurement,
        throughput_interval=throughput,
        cost_interval=cost,
    )


def test_tier_promotion_requires_measured_same_hardware_harness_and_tokens(plan) -> None:
    evidence = _tier_evidence(plan)
    assert tier_eligible(plan, evidence)[0] is True
    assert evidence.candidate_measurement.throughput_tokens_per_second == 90.0
    assert evidence.candidate_measurement.cost_per_1k_tokens_microusd > 0
    with pytest.raises(ValidationError):
        BenchmarkMeasurement.model_validate(
            evidence.candidate_measurement.model_dump(
                mode="python",
                exclude={"throughput_tokens_per_second"},
            )
        )
    candidate_payload = evidence.candidate_measurement.model_dump(
        mode="python",
        exclude={"evidence_sha256"},
    )
    confounded = benchmark_measurement(
        **{
            **candidate_payload,
            "instance_type": "ml.g5.48xlarge",
            "hardware_profile": "g5-48xlarge-8xa10g-independent-v1",
            "accelerator": "NVIDIA A10G",
        }
    )
    throughput_payload = evidence.throughput_interval.model_dump(
        mode="python",
        exclude={"interval_sha256"},
    )
    cost_payload = evidence.cost_interval.model_dump(
        mode="python",
        exclude={"interval_sha256"},
    )
    confounded_evidence = tier_promotion_evidence(
        candidate=Tier.CORE,
        incumbent=Tier.NANO,
        quality_interval=evidence.quality_interval,
        candidate_measurement=confounded,
        incumbent_measurement=evidence.incumbent_measurement,
        throughput_interval=ratio_interval(
            **{
                **throughput_payload,
                "candidate_measurement_sha256": confounded.evidence_sha256,
            }
        ),
        cost_interval=ratio_interval(
            **{
                **cost_payload,
                "candidate_measurement_sha256": confounded.evidence_sha256,
            }
        ),
    )
    eligible, reason = tier_eligible(plan, confounded_evidence)
    assert eligible is False
    assert "confound" in reason


def _artifact(model_id, run_id, *, manifest_sha=H["5"], proof_sha=H["8"]):
    return artifact_publication_evidence(
        model_id=model_id,
        run_id=run_id,
        manifest_sha256=manifest_sha,
        adapter_uri=f"s3://portfolio-published/{model_id}/adapter/",
        adapter_checksum_sha256=H["9"],
        adapter_size_bytes=1024,
        artifact_inventory_sha256=content_sha256({"model_id": model_id, "manifest": manifest_sha}),
        artifact_exists_evidence_sha256=H["a"],
        proof_report_sha256=proof_sha,
        verified_at=NOW,
    )


def test_registry_promotion_and_publish_are_artifact_and_proof_gated(plan) -> None:
    replication, multiplicity, evidence = _specialist_selection(plan)
    registry = register_replication_wave(
        registry=plan.registry,
        plan=plan,
        wave=replication,
    )
    treatment = next(
        slot
        for slot in replication.active_slots
        if slot.model_id == evidence.replication_interval.treatment_model_id
    )
    artifact = _artifact(
        treatment.model_id,
        treatment.run_id,
        manifest_sha=evidence.replication_treatment_manifest_sha256,
        proof_sha=evidence.replication_proof_report_sha256,
    )
    promotion = promote_specialist_registry(
        plan=plan,
        registry=registry,
        replication_wave=replication,
        multiplicity_plan=multiplicity,
        evidence=evidence,
        artifact=artifact,
    )
    assert promotion.status == "specialist_backup"
    assert promotion.routing == "explicit_user_switch_only"
    approvals = []
    for tier in Tier:
        model = next(
            model
            for model in plan.models
            if model.tier is tier
            and model.role == "generalist"
            and model.arm is PortfolioArm.ORACLE_SFT
        )
        model_artifact = _artifact(model.model_id, model.run_id, proof_sha=H["7"])
        approvals.append(
            generalist_approval(
                tier=tier,
                model_id=model.model_id,
                proof_report_sha256=H["7"],
                quality_gate_evidence_sha256=H["6"],
                artifact_evidence=model_artifact,
            )
        )
    inventory_hash = content_sha256(
        sorted(
            [
                *(approval.artifact_evidence.artifact_inventory_sha256 for approval in approvals),
                promotion.artifact_evidence.artifact_inventory_sha256,
            ]
        )
    )
    gate = registry_publication_gate(
        planned_registry_sha256=registry.registry_sha256,
        artifact_inventory_sha256=inventory_hash,
        checksum_verifier_sha256=H["5"],
        proof_protocol_sha256=plan.protocol.proof_protocol_sha256,
        published_at=NOW,
    )
    bundle = publish_registry(
        plan=plan,
        registry=registry,
        gate=gate,
        generalists=tuple(approvals),
        specialist_promotions=(promotion,),
    )
    assert len(bundle.entries) == 4
    assert bundle.entries[-1].status == "specialist_backup"
    assert bundle.active_default_model_id == approvals[0].model_id
    wrong_gate_payload = gate.model_dump(
        mode="python",
        exclude={"gate_evidence_sha256"},
    )
    wrong_gate = registry_publication_gate(
        **{**wrong_gate_payload, "artifact_inventory_sha256": H["f"]}
    )
    with pytest.raises(ValueError, match="exact artifact inventory"):
        publish_registry(
            plan=plan,
            registry=registry,
            gate=wrong_gate,
            generalists=tuple(approvals),
            specialist_promotions=(promotion,),
        )
