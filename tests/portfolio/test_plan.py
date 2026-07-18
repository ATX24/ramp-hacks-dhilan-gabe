"""Portfolio wave design, readiness, determinism, and cost regressions."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from distillery.contracts.budgets import ProofGates, TrainingBudget
from experiments.portfolio.plan import (
    ACCOUNT_CEILING_MICROUSD,
    BOOTSTRAP_RESAMPLES,
    REPLICATION_SEED,
    SCREEN_SEED,
    CostCeilings,
    FinalistPair,
    NotStartedSlot,
    PlannedRunSlot,
    PortfolioArm,
    Surface,
    Task,
    Tier,
    build_replication_wave,
    cost,
    memory_probe,
    pricing_evidence,
    readiness_evidence,
    replication_selection_lock,
    validate_readiness,
)
from tests.portfolio.support import NOW, H, make_plan, make_readiness


def test_protocol_uses_proof_and_training_constants_not_smoke(plan) -> None:
    protocol = plan.protocol
    proof = ProofGates()
    budget = TrainingBudget()
    assert protocol.screen_seed == proof.required_seed_screen == SCREEN_SEED
    assert protocol.replication_seed == proof.required_seed_replication == REPLICATION_SEED
    assert (
        protocol.bootstrap_resamples == proof.bootstrap_resamples == BOOTSTRAP_RESAMPLES == 10_000
    )
    assert (
        protocol.max_length,
        protocol.max_completion,
        protocol.max_steps,
        protocol.grad_accumulation,
        protocol.lora_rank,
    ) == (
        budget.max_length,
        budget.max_completion,
        budget.max_steps,
        budget.grad_accumulation,
        budget.lora_rank,
    )
    assert protocol.max_completion != 160
    assert protocol.grad_accumulation != 8


def test_exact_nano_logit_screen_and_preserved_idle_slots(plan) -> None:
    wave = plan.screen_waves[0]
    assert (wave.tier, wave.surface, wave.seed) == (Tier.NANO, Surface.LOGIT, 17)
    assert [(slot.slot, slot.node, slot.gpu) for slot in wave.slots] == [
        (index, index % 2, index // 2) for index in range(16)
    ]
    active = wave.active_slots
    assert [
        (slot.role, slot.arm, tuple(task.value for task in slot.tasks)) for slot in active[:4]
    ] == [
        ("generalist", PortfolioArm.ORACLE_SFT, tuple(task.value for task in Task)),
        ("generalist", PortfolioArm.SEQUENCE_KD, tuple(task.value for task in Task)),
        ("generalist", PortfolioArm.LOGIT_KD, tuple(task.value for task in Task)),
        ("generalist", PortfolioArm.CE_ABLATION, tuple(task.value for task in Task)),
    ]
    assert [(slot.arm, slot.tasks[0]) for slot in active[4:]] == [
        pair
        for task in Task
        for pair in (
            (PortfolioArm.LOGIT_KD, task),
            (PortfolioArm.CE_ABLATION, task),
        )
    ]
    assert all(isinstance(slot, NotStartedSlot) for slot in wave.slots[12:])
    assert all(slot.cost_included for slot in wave.slots[12:])


def test_every_specialist_treatment_has_same_recipe_control(plan) -> None:
    assert [(wave.tier, wave.surface) for wave in plan.screen_waves] == [
        (tier, surface) for tier in Tier for surface in (Surface.LOGIT, Surface.SEQUENCE)
    ]
    for wave in plan.screen_waves:
        groups: dict[str, list[PlannedRunSlot]] = {}
        for slot in wave.active_slots:
            groups.setdefault(slot.comparison_id, []).append(slot)
            if slot.role == "generalist":
                assert slot.tasks == tuple(Task)
            else:
                assert len(slot.tasks) == 1
        for pair in groups.values():
            treatment = next(slot for slot in pair if slot.comparison_position == "treatment")
            control = next(slot for slot in pair if slot.comparison_position == "control")
            assert (
                treatment.recipe,
                treatment.seed,
                treatment.tasks,
                treatment.dataset_view_sha256,
                treatment.comparison_sha256,
            ) == (
                control.recipe,
                control.seed,
                control.tasks,
                control.dataset_view_sha256,
                control.comparison_sha256,
            )
            assert treatment.arm in {
                PortfolioArm.SEQUENCE_KD,
                PortfolioArm.LOGIT_KD,
            }
            if treatment.arm is PortfolioArm.SEQUENCE_KD:
                assert control.arm in {
                    PortfolioArm.ORACLE_SFT,
                    PortfolioArm.SEQUENCE_CE_CONTROL,
                }
            else:
                assert control.arm is PortfolioArm.CE_ABLATION


def test_generalist_and_specialist_dataset_views_are_sealed(plan) -> None:
    generalist = plan.dataset.view_for(tuple(Task))
    assert generalist.task_filter == tuple(Task)
    for task in Task:
        specialist = plan.dataset.view_for((task,))
        assert specialist.task_filter == (task,)
        assert specialist.parent_bundle_sha256 == plan.dataset.content_sha256
        assert specialist.view_sha256 != generalist.view_sha256
    generalist_slot = plan.screen_waves[0].active_slots[0]
    with pytest.raises(ValidationError, match="all four"):
        generalist_slot.model_copy(update={"tasks": (Task.MERCHANT_TAGGING,)})


def test_specialists_start_planned_without_registry_uris_or_statuses(plan) -> None:
    assert plan.registry.publishable is False
    assert not hasattr(plan.registry, "adapter_uris")
    assert not hasattr(plan.registry, "merged_uris")
    assert not hasattr(plan.registry, "routing_statuses")
    specialists = [entry for entry in plan.registry.entries if entry.role == "specialist"]
    assert specialists
    assert {entry.state for entry in specialists} == {"planned"}
    assert all(not hasattr(entry, "status") for entry in specialists)
    assert all(not hasattr(entry, "adapter_uri") for entry in specialists)


def _replication_lock(plan):
    source = plan.screen_waves[0]
    specialist = [
        slot
        for slot in source.active_slots
        if slot.role == "specialist" and slot.tasks == (Task.TRANSACTION_REVIEW,)
    ]
    generalist_logit = [
        slot
        for slot in source.active_slots
        if slot.role == "generalist" and slot.comparison_id == "cmp_generalist_logit"
    ]
    pairs = (
        FinalistPair(
            treatment_model_id=next(
                slot.model_id for slot in specialist if slot.comparison_position == "treatment"
            ),
            control_model_id=next(
                slot.model_id for slot in specialist if slot.comparison_position == "control"
            ),
        ),
        FinalistPair(
            treatment_model_id=next(
                slot.model_id
                for slot in generalist_logit
                if slot.comparison_position == "treatment"
            ),
            control_model_id=next(
                slot.model_id for slot in generalist_logit if slot.comparison_position == "control"
            ),
        ),
    )
    return replication_selection_lock(
        tier=Tier.NANO,
        source_wave_sha256=(source.matrix_sha256,),
        validation_split_sha256=plan.dataset.split_sha256["validation"],
        selection_protocol_sha256=H["1"],
        selected_pairs=pairs,
        locked_at=NOW,
    )


def test_seed23_replication_binds_new_ids_artifacts_and_protocol(plan) -> None:
    replication = build_replication_wave(plan=plan, selection=_replication_lock(plan))
    assert replication.seed == 23
    assert replication.phase == "replication"
    assert len(replication.active_slots) == 4
    screen_ids = {model.model_id for model in plan.models}
    screen_runs = {model.run_id for model in plan.models}
    screen_artifacts = {reservation.artifact_id for reservation in plan.artifacts.reservations}
    assert not screen_ids.intersection(slot.model_id for slot in replication.active_slots)
    assert not screen_runs.intersection(slot.run_id for slot in replication.active_slots)
    assert not screen_artifacts.intersection(slot.artifact_id for slot in replication.active_slots)
    assert all("s23" in slot.model_id for slot in replication.active_slots)
    assert all(slot.seed == 23 for slot in replication.active_slots)
    assert all(
        slot.protocol_sha256 not in {model.protocol_sha256 for model in plan.models}
        for slot in replication.active_slots
    )
    assert all(isinstance(slot, NotStartedSlot) for slot in replication.slots[4:])


def test_readiness_binds_roles_licenses_output_image_and_memory(plan) -> None:
    evidence = make_readiness(plan, Tier.CORE)
    validate_readiness(
        plan.candidates[1],
        plan.runtimes[1],
        plan.gates[1],
        evidence,
    )
    mismatched_payload = evidence.model_dump(
        mode="python",
        exclude={"readiness_sha256"},
    )
    mismatched_payload["student_role_evidence_sha256"] = H["1"]
    mismatched_payload["probes"] = evidence.probes
    mismatched = readiness_evidence(**mismatched_payload)
    with pytest.raises(ValueError, match="role/license/output/image"):
        validate_readiness(
            plan.candidates[1],
            plan.runtimes[1],
            plan.gates[1],
            mismatched,
        )
    first = evidence.probes[0]
    peak = plan.gates[1].capacity_bytes * 86 // 100
    bad_probe_payload = first.model_dump(mode="python", exclude={"probe_sha256"})
    bad_probe_payload.update(
        {
            "peak_bytes": peak,
            "headroom_bytes": plan.gates[1].capacity_bytes - peak,
        }
    )
    bad_probe = memory_probe(**bad_probe_payload)
    bad_evidence_payload = evidence.model_dump(
        mode="python",
        exclude={"readiness_sha256"},
    )
    bad_evidence_payload["probes"] = (bad_probe, *evidence.probes[1:])
    bad_evidence = readiness_evidence(**bad_evidence_payload)
    with pytest.raises(ValueError, match="85%"):
        validate_readiness(
            plan.candidates[1],
            plan.runtimes[1],
            plan.gates[1],
            bad_evidence,
        )


def test_pricing_bytes_freshness_and_all_cost_ceilings(plan) -> None:
    price = plan.pricing[0]
    source = b'{"instance":"ml.g5.48xlarge","hourly_usd":"20.36"}'
    price.verify_evidence_bytes(source)
    with pytest.raises(ValueError, match="hash mismatch"):
        price.verify_evidence_bytes(b"x" * len(source))
    expensive = pricing_evidence(
        source_uri="https://pricing.example.test/expensive.json",
        region="us-east-1",
        instance_type="ml.g5.48xlarge",
        current_hourly_price_microusd=100_000_000,
        attestor="test",
        attested_at=NOW,
        effective_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
        evidence_bytes=b"expensive",
    )
    with pytest.raises(ValueError, match="per-run|per-wave"):
        cost(
            plan.screen_waves[0],
            expensive,
            plan.protocol,
            plan.ceilings,
        )
    with pytest.raises(ValidationError, match="ordered"):
        CostCeilings(
            per_run_microusd=25_000_000,
            per_wave_microusd=250_000_000,
            experiment_microusd=ACCOUNT_CEILING_MICROUSD + 1,
        )
    assert plan.costs.aggregate_ceiling_microusd <= plan.ceilings.experiment_microusd
    assert plan.costs.aggregate_ceiling_microusd <= plan.ceilings.account_microusd
    assert len(plan.costs.waves) == 6
    for wave_cost in plan.costs.waves:
        assert len(wave_cost.slots) == 16
        assert (
            sum(slot.allocated_ceiling_microusd for slot in wave_cost.slots)
            == wave_cost.aggregate_ceiling_microusd
        )


def test_plan_is_deterministic_and_role_reuse_is_role_specific(plan) -> None:
    rebuilt = make_plan()
    assert rebuilt.plan_sha256 == plan.plan_sha256
    assert rebuilt.canonical_bytes() == plan.canonical_bytes()
    nano_teacher = plan.candidates[0].pair.teacher
    core_student = plan.candidates[1].pair.student
    assert (nano_teacher.model_id, nano_teacher.revision) == (
        core_student.model_id,
        core_student.revision,
    )
    assert nano_teacher.evidence_sha256 != core_student.evidence_sha256
    assert all(candidate.throughput_tokens_per_second is None for candidate in plan.candidates)
    assert all(candidate.cost_per_1k_tokens_microusd is None for candidate in plan.candidates)
