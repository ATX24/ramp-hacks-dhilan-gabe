"""Production-shaped portfolio contract fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from distillery.contracts.hashing import content_sha256
from distillery.contracts.manifest import ManifestCompletionEvidence, SealedRunManifest
from experiments.portfolio.materialize import (
    SlotMaterializationEvidence,
    slot_materialization_evidence,
)
from experiments.portfolio.plan import (
    REQUIRED_PROBES,
    TASKS,
    DatasetBundle,
    FinalistPair,
    PlannedRunSlot,
    PortfolioPlan,
    ReadinessEvidence,
    Task,
    Tier,
    build_dataset_bundle,
    build_plan,
    build_replication_wave,
    dataset_view,
    memory_probe,
    model_pair,
    pricing_evidence,
    readiness_evidence,
    replication_selection_lock,
    role_evidence,
    runtime_binding,
)

NOW = datetime(2026, 7, 18, 17, 0, tzinfo=UTC)
ACCOUNT = "225989358036"
H = {character: character * 64 for character in "123456789abcdef"}
IMAGE_DIGEST = f"sha256:{H['a']}"
PACKAGE_LOCK_HASH = H["b"]
SOURCE_REVISION = "f" * 40
SPECIAL_TOKEN_MAP = {"eos_token": 151645, "pad_token": 151643}


def _role(role: str, model_id: str, revision: str):
    return role_evidence(
        role=role,  # type: ignore[arg-type]
        model_id=model_id,
        revision=revision,
        model_config_sha256=H["1"],
        tokenizer_sha256=H["2"],
        chat_template_sha256=H["3"],
        special_tokens_sha256=H["4"],
        license_sha256=H["5"],
        output_use_sha256=H["6"],
    )


def make_dataset_bundle() -> DatasetBundle:
    bundle_hash = H["7"]
    filters = (TASKS, *((task,) for task in TASKS))
    views = tuple(
        dataset_view(
            view_id=f"ds_pf_view_{index:02d}",
            parent_bundle_sha256=bundle_hash,
            relative_prefix=f"views/view-{index:02d}/",
            task_filter=tasks,
            content_digest=str(index + 1) * 64,
            split_sha256={"train": H["8"], "validation": H["9"]},
        )
        for index, tasks in enumerate(filters)
    )
    return build_dataset_bundle(
        bundle_id="ds_pf_bundle_v2",
        uri=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/dataset-bundle/",
        content_digest=bundle_hash,
        split_sha256={"train": H["8"], "validation": H["9"]},
        views=views,
    )


def make_plan() -> PortfolioPlan:
    nano_pair = model_pair(
        _role("teacher", "Qwen/Qwen2.5-1.5B-Instruct", "b" * 40),
        _role("student", "Qwen/Qwen2.5-0.5B-Instruct", "a" * 40),
    )
    core_pair = model_pair(
        _role("teacher", "Qwen/Qwen2.5-7B-Instruct", "c" * 40),
        _role("student", "Qwen/Qwen2.5-1.5B-Instruct", "b" * 40),
    )
    plus_pair = model_pair(
        _role("teacher", "Qwen/Qwen2.5-14B-Instruct", "e" * 40),
        _role("student", "Qwen/Qwen2.5-3B-Instruct", "d" * 40),
    )
    runtimes = tuple(
        runtime_binding(
            tier=tier,
            runtime_image_digest=IMAGE_DIGEST,
            package_lock_hash=PACKAGE_LOCK_HASH,
            source_revision=SOURCE_REVISION,
        )
        for tier in Tier
    )
    g5_price = pricing_evidence(
        source_uri="https://pricing.example.test/g5-48xlarge.json",
        region="us-east-1",
        instance_type="ml.g5.48xlarge",
        current_hourly_price_microusd=20_360_000,
        attestor="portfolio-test",
        attested_at=NOW,
        effective_at=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=1),
        evidence_bytes=b'{"instance":"ml.g5.48xlarge","hourly_usd":"20.36"}',
    )
    p4de_price = pricing_evidence(
        source_uri="https://pricing.example.test/p4de-24xlarge.json",
        region="us-east-1",
        instance_type="ml.p4de.24xlarge",
        current_hourly_price_microusd=31_564_107,
        attestor="portfolio-test",
        attested_at=NOW,
        effective_at=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=1),
        evidence_bytes=b'{"instance":"ml.p4de.24xlarge","hourly_usd":"31.564107"}',
    )
    return build_plan(
        created_at=NOW,
        nano_pair=nano_pair,
        core_pair=core_pair,
        plus_pair=plus_pair,
        nano_runtime=runtimes[0],
        core_runtime=runtimes[1],
        plus_runtime=runtimes[2],
        dataset=make_dataset_bundle(),
        g5_pricing=g5_price,
        p4de_pricing=p4de_price,
        artifact_root=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/v3/",
    )


def make_readiness(plan: PortfolioPlan, tier: Tier) -> ReadinessEvidence:
    index = tuple(Tier).index(tier)
    item = plan.candidates[index]
    runtime = plan.runtimes[index]
    gate = plan.gates[index]
    peak = gate.capacity_bytes * 70 // 100
    common = {
        "tier": tier,
        "candidate_sha256": item.descriptor_sha256,
        "pair_sha256": item.pair.binding_sha256,
        "gate_sha256": gate.gate_sha256,
        "teacher_role_evidence_sha256": item.pair.teacher.evidence_sha256,
        "student_role_evidence_sha256": item.pair.student.evidence_sha256,
        "teacher_license_evidence_sha256": item.pair.teacher.license_evidence_sha256,
        "student_license_evidence_sha256": item.pair.student.license_evidence_sha256,
        "teacher_output_use_evidence_sha256": (item.pair.teacher.output_use_evidence_sha256),
        "student_output_use_evidence_sha256": (item.pair.student.output_use_evidence_sha256),
        "runtime_sha256": runtime.runtime_sha256,
        "runtime_image_digest": runtime.runtime_image_digest,
        "image_evidence_sha256": H["c"],
        "instance_type": gate.instance_type,
        "accelerator": gate.accelerator,
        "peak_bytes": peak,
        "capacity_bytes": gate.capacity_bytes,
        "headroom_bytes": gate.capacity_bytes - peak,
        "measured_at": NOW,
        "attestor": "portfolio-test",
        "raw_evidence_sha256": H["d"],
        "raw_evidence_size_bytes": 1024,
    }
    probes = tuple(memory_probe(kind=kind, **common) for kind in REQUIRED_PROBES)
    return readiness_evidence(
        tier=tier,
        candidate_sha256=item.descriptor_sha256,
        pair_sha256=item.pair.binding_sha256,
        gate_sha256=gate.gate_sha256,
        runtime_sha256=runtime.runtime_sha256,
        runtime_image_digest=runtime.runtime_image_digest,
        image_evidence_sha256=H["c"],
        image_evidence_size_bytes=2048,
        teacher_role_evidence_sha256=item.pair.teacher.evidence_sha256,
        student_role_evidence_sha256=item.pair.student.evidence_sha256,
        teacher_license_evidence_sha256=item.pair.teacher.license_evidence_sha256,
        student_license_evidence_sha256=item.pair.student.license_evidence_sha256,
        teacher_output_use_evidence_sha256=(item.pair.teacher.output_use_evidence_sha256),
        student_output_use_evidence_sha256=(item.pair.student.output_use_evidence_sha256),
        probes=probes,
        evidence_manifest_sha256=H["e"],
    )


def make_materialization_evidence(
    slot: PlannedRunSlot,
) -> SlotMaterializationEvidence:
    example_ids = ("ex_pf_001", "ex_pf_002")
    target_source = "pre_materialized_teacher" if slot.arm.value == "sequence_kd" else "oracle"
    label_source = "teacher" if target_source == "pre_materialized_teacher" else "oracle"
    records = {
        example_id: content_sha256(
            {
                "example_id": example_id,
                "model_id": slot.model_id,
                "target_source": target_source,
            }
        )
        for example_id in example_ids
    }
    completion = ManifestCompletionEvidence(
        source_file_sha256=H["1"],
        canonical_records_sha256=content_sha256(records),
        record_sha256=records,
        provenance_sha256=content_sha256(
            {"records": records, "slot_protocol": slot.protocol_sha256}
        ),
        completion_token_counts={"ex_pf_001": 24, "ex_pf_002": 31},
        completion_tokenizer_sha256=H["2"],
        label_source_counts={label_source: 2},
        accepted_example_count=2,
    )
    return slot_materialization_evidence(
        model_id=slot.model_id,
        created_at=NOW,
        target_source=target_source,
        teacher_responses_sha256=(H["f"] if target_source == "pre_materialized_teacher" else None),
        completion_evidence=completion,
        special_token_map=SPECIAL_TOKEN_MAP,
        sampler_order_hash=content_sha256(
            {"seed": slot.seed, "model_id": slot.model_id, "order": list(example_ids)}
        ),
        source_records_sha256=content_sha256(records),
    )


def make_replication_wave(plan: PortfolioPlan):
    source = plan.screen_waves[0]
    specialist = [
        slot
        for slot in source.active_slots
        if slot.role == "specialist" and slot.tasks == (Task.TRANSACTION_REVIEW,)
    ]
    generalist = [
        slot
        for slot in source.active_slots
        if slot.role == "generalist" and slot.comparison_id == "cmp_generalist_logit"
    ]
    selection = replication_selection_lock(
        tier=Tier.NANO,
        source_wave_sha256=(source.matrix_sha256,),
        validation_split_sha256=plan.dataset.split_sha256["validation"],
        selection_protocol_sha256=H["1"],
        selected_pairs=(
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
                    slot.model_id for slot in generalist if slot.comparison_position == "treatment"
                ),
                control_model_id=next(
                    slot.model_id for slot in generalist if slot.comparison_position == "control"
                ),
            ),
        ),
        locked_at=NOW,
    )
    return build_replication_wave(plan=plan, selection=selection)


def write_manifest(path: Path, manifest: SealedRunManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
