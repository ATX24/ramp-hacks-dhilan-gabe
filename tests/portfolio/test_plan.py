"""Tiered portfolio determinism, safety, gates, and accounting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiments.portfolio.plan import (
    A10G_BYTES,
    A100_BYTES,
    REQUIRED_PROBES,
    ArtifactPlan,
    MemoryProbe,
    Pricing,
    ReadinessEvidence,
    SpecialistEvidence,
    Task,
    Tier,
    TierEvidence,
    build_plan,
    model_pair,
    role_evidence,
    specialist_eligible,
    tier_eligible,
    validate_readiness,
)

H = {key: key * 64 for key in "123456789"}


def _role(role: str, model: str, revision: str):
    return role_evidence(
        role=role,  # type: ignore[arg-type]
        model_id=model,
        revision=revision,
        model_config_sha256=H["1"],
        tokenizer_sha256=H["2"],
        chat_template_sha256=H["3"],
        special_tokens_sha256=H["4"],
        license_sha256=H["5"],
        output_use_sha256=H["6"],
    )


@pytest.fixture
def plan():
    # Same 1.5B revision, separately attested as Nano teacher and Core student.
    nano_teacher = _role("teacher", "Qwen/Qwen2.5-1.5B-Instruct", "b" * 40)
    core_student = _role("student", "Qwen/Qwen2.5-1.5B-Instruct", "b" * 40)
    return build_plan(
        nano_pair=model_pair(
            nano_teacher,
            _role("student", "Qwen/Qwen2.5-0.5B-Instruct", "a" * 40),
        ),
        core_pair=model_pair(
            _role("teacher", "Qwen/Qwen2.5-7B-Instruct", "c" * 40),
            core_student,
        ),
        plus_pair=model_pair(
            _role("teacher", "Qwen/Qwen2.5-14B-Instruct", "e" * 40),
            _role("student", "Qwen/Qwen2.5-3B-Instruct", "d" * 40),
        ),
        g5_pricing=Pricing(
            instance_type="ml.g5.48xlarge",
            hourly_microusd=20_360_000,
            evidence_sha256=H["7"],
        ),
        p4de_pricing=Pricing(
            instance_type="ml.p4de.24xlarge",
            hourly_microusd=31_564_107,
            evidence_sha256=H["8"],
        ),
        artifact_root="s3://portfolio-test/v2/",
    )


def test_role_specific_reuse_and_tier_metadata(plan) -> None:
    nano_teacher = plan.candidates[0].pair.teacher
    core_student = plan.candidates[1].pair.student
    assert (nano_teacher.model_id, nano_teacher.revision) == (
        core_student.model_id,
        core_student.revision,
    )
    assert nano_teacher.evidence_sha256 != core_student.evidence_sha256
    with pytest.raises(ValidationError, match="role-specific"):
        model_pair(plan.candidates[1].pair.teacher, nano_teacher)
    assert [item.name for item in plan.candidates] == [
        "TinyFable Nano",
        "TinyFable Core",
        "TinyFable Plus",
    ]
    assert all(item.larger_is_better_assumption is False for item in plan.candidates)
    assert all(item.throughput_tokens_per_second is None for item in plan.candidates)


def test_exact_wave1_matrix_and_round_robin(plan) -> None:
    wave = plan.waves[0]
    assert wave.tier == Tier.NANO
    assert wave.instance_type == "ml.g5.48xlarge"
    assert [(slot.slot, slot.node, slot.gpu) for slot in wave.slots] == [
        (index, index % 2, index // 2) for index in range(16)
    ]
    actual = [
        (slot.role, slot.arm, tuple(task.value for task in slot.tasks)) for slot in wave.slots
    ]
    assert actual == [
        *[
            ("generalist", arm, tuple(task.value for task in Task))
            for arm in ("oracle_sft", "sequence_kd", "logit_kd", "ce_ablation")
        ],
        *[
            ("specialist", arm, (task.value,))
            for task in Task
            for arm in ("sequence_kd", "logit_kd", "ce_ablation")
        ],
    ]


def test_all_tiers_cover_tasks_with_matched_controls(plan) -> None:
    assert [wave.instance_type for wave in plan.waves] == [
        "ml.g5.48xlarge",
        "ml.p4de.24xlarge",
        "ml.p4de.24xlarge",
    ]
    assert {wave.seed for wave in plan.waves} == {17}
    for wave in plan.waves:
        assert len(wave.slots) == 16
        for task in Task:
            task_slots = [
                slot for slot in wave.slots if slot.role == "specialist" and slot.tasks == (task,)
            ]
            assert {slot.arm for slot in task_slots} == {
                "sequence_kd",
                "logit_kd",
                "ce_ablation",
            }
            logit = next(slot for slot in task_slots if slot.arm == "logit_kd")
            control = next(slot for slot in task_slots if slot.arm == "ce_ablation")
            assert (logit.seed, logit.recipe) == (control.seed, control.recipe)
            assert control.matched_control_of == "logit_kd"


def test_registry_defaults_to_generalists_without_hidden_routing(plan) -> None:
    registry = plan.registry
    assert registry.active_default_tier == Tier.NANO
    assert registry.silent_task_routing_forbidden
    assert registry.silent_tier_routing_forbidden
    by_id = {entry.model_id: entry for entry in registry.entries}
    assert all(by_id[model_id].role == "generalist" for model_id in registry.tier_default_model_ids)
    assert all(
        entry.status == "specialist_backup"
        for entry in registry.entries
        if entry.role == "specialist"
    )


def _readiness(plan, tier: Tier) -> ReadinessEvidence:
    item = plan.candidates[list(Tier).index(tier)]
    gate = plan.gates[list(Tier).index(tier)]
    peak = gate.capacity_bytes * 80 // 100
    return ReadinessEvidence(
        tier=tier,
        gate_sha256=gate.gate_sha256,
        probes=tuple(
            MemoryProbe(
                kind=kind,
                candidate_sha256=item.descriptor_sha256,
                pair_sha256=item.pair.binding_sha256,
                gate_sha256=gate.gate_sha256,
                instance_type=gate.instance_type,
                device=gate.device,
                runtime_image_digest=f"sha256:{'f' * 64}",
                peak_bytes=peak,
                capacity_bytes=gate.capacity_bytes,
                headroom_bytes=gate.capacity_bytes - peak,
                manifest_sha256=H["9"],
            )
            for kind in REQUIRED_PROBES
        ),
        manifest_sha256=H["9"],
    )


@pytest.mark.parametrize(
    ("tier", "capacity", "headroom"),
    [
        (Tier.NANO, A10G_BYTES, 4 * 1024**3),
        (Tier.CORE, A100_BYTES, 8 * 1024**3),
        (Tier.PLUS, A100_BYTES, 8 * 1024**3),
    ],
)
def test_exact_measured_memory_gates(plan, tier, capacity, headroom) -> None:
    index = list(Tier).index(tier)
    gate = plan.gates[index]
    assert (gate.capacity_bytes, gate.min_headroom_bytes) == (capacity, headroom)
    assert gate.max_peak_basis_points == 8500
    assert gate.estimates_cannot_pass
    validate_readiness(plan.candidates[index], gate, _readiness(plan, tier))


def test_memory_gate_rejects_over_85_percent(plan) -> None:
    gate = plan.gates[2]
    evidence = _readiness(plan, Tier.PLUS)
    peak = gate.capacity_bytes * 86 // 100
    bad_probe = evidence.probes[0].model_copy(
        update={
            "peak_bytes": peak,
            "headroom_bytes": gate.capacity_bytes - peak,
        }
    )
    with pytest.raises(ValueError, match="85%"):
        validate_readiness(
            plan.candidates[2],
            gate,
            evidence.model_copy(update={"probes": (bad_probe, *evidence.probes[1:])}),
        )


def test_artifact_isolation_and_duplicate_rejection(plan) -> None:
    artifacts = plan.artifacts.artifacts
    assert len({item.adapter_uri for item in artifacts}) == 48
    assert len({item.merged_uri_optional for item in artifacts}) == 48
    assert all(item.mutate_v1_forbidden for item in artifacts)
    duplicate = artifacts[1].model_copy(update={"adapter_uri": artifacts[0].adapter_uri})
    with pytest.raises(ValidationError, match="duplicate artifact adapter_uri"):
        ArtifactPlan(
            root=plan.artifacts.root,
            artifacts=(artifacts[0], duplicate, *artifacts[2:]),
        )


def test_specialist_promotion_requires_material_uncertain_gain(plan) -> None:
    weak = SpecialistEvidence(
        tier=Tier.NANO,
        task=Task.MERCHANT_TAGGING,
        quality_delta=0.04,
        quality_ci_lower=0.019,
        quality_ci_upper=0.06,
        validation_protocol_sha256=H["9"],
    )
    strong = weak.model_copy(update={"quality_ci_lower": 0.021})
    assert specialist_eligible(weak, plan.promotion)[0] is False
    eligible, reason = specialist_eligible(strong, plan.promotion)
    assert eligible is True
    assert "generalist remains default" in reason


def test_tier_promotion_uses_quality_throughput_and_cost_not_size(plan) -> None:
    evidence = TierEvidence(
        candidate=Tier.CORE,
        incumbent=Tier.NANO,
        quality_ci_lower=0.021,
        quality_ci_upper=0.04,
        throughput_ratio_ci_lower=0.85,
        throughput_ratio_ci_upper=0.95,
        cost_ratio_ci_lower=1.02,
        cost_ratio_ci_upper=1.20,
        measurement_protocol_sha256=H["9"],
    )
    assert plan.promotion.no_size_prior
    assert tier_eligible(evidence, plan.promotion)[0] is True
    too_slow = evidence.model_copy(
        update={
            "throughput_ratio_ci_lower": 0.65,
            "throughput_ratio_ci_upper": 0.75,
        }
    )
    assert tier_eligible(too_slow, plan.promotion)[0] is False


def test_costs_sum_exactly_and_plan_is_deterministic(plan) -> None:
    for item in plan.costs:
        assert sum(slot.ceiling_microusd for slot in item.slots) == (
            item.aggregate_ceiling_microusd
        )
        for node in (0, 1):
            assert (
                sum(slot.ceiling_microusd for slot in item.slots if slot.node == node)
                == item.parent_ceiling_microusd
            )
        assert item.throughput_tokens_per_second is None
    assert plan.canonical_bytes().endswith(b"}")
    with pytest.raises(ValidationError):
        plan.mode = "train"
