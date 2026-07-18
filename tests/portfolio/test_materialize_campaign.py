"""Real manifest materialization and g5/p4de campaign adapter regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distillery.contracts.hashing import sha256_hex
from distillery.contracts.manifest import SealedRunManifest
from experiments.aws_smoke.campaign_index import matched_protocol_inputs_sha256
from experiments.portfolio.campaign import (
    CAMPAIGN_ORCHESTRATOR_MODULE,
    CONTAINER_PYTHON,
    REQUIRED_IMAGE_MODULES,
    ContainerStagingEvidence,
    PortfolioAwsEvidence,
    SlotOutcome,
    build_execution_ledger,
    build_portfolio_launch_plan,
    container_staging_evidence,
    stage_portfolio_wave,
)
from experiments.portfolio.materialize import (
    materialize_slot,
    validate_materialized_manifest,
)
from experiments.portfolio.plan import PortfolioArm, Task, Tier
from experiments.portfolio.task_filter_runtime import resolve_task_dataset_dir
from tests.portfolio.support import (
    ACCOUNT,
    IMAGE_DIGEST,
    H,
    make_materialization_evidence,
    make_readiness,
    write_manifest,
)


def _materialize_wave(plan, wave, root: Path) -> dict[str, Path]:
    readiness = make_readiness(plan, wave.tier)
    paths: dict[str, Path] = {}
    for slot in wave.active_slots:
        manifest = materialize_slot(
            plan=plan,
            wave=wave,
            slot=slot,
            readiness=readiness,
            evidence=make_materialization_evidence(slot),
        )
        paths[slot.model_id] = write_manifest(
            root / f"slot-{slot.slot:02d}" / "manifest.json",
            manifest,
        )
    return paths


def test_slots_materialize_to_production_shaped_sealed_manifests(plan, tmp_path) -> None:
    wave = plan.screen_waves[0]
    paths = _materialize_wave(plan, wave, tmp_path)
    manifests = [SealedRunManifest.model_validate_json(path.read_text()) for path in paths.values()]
    assert len(manifests) == 12
    assert len({manifest.seal_sha256() for manifest in manifests}) == 12
    assert len({matched_protocol_inputs_sha256(manifest) for manifest in manifests}) == 1
    assert {manifest.tags["PortfolioArm"] for manifest in manifests} == {
        PortfolioArm.ORACLE_SFT.value,
        PortfolioArm.SEQUENCE_KD.value,
        PortfolioArm.LOGIT_KD.value,
        PortfolioArm.CE_ABLATION.value,
    }
    for slot, manifest in zip(wave.active_slots, manifests, strict=True):
        validate_materialized_manifest(plan=plan, wave=wave, slot=slot, manifest=manifest)
        assert manifest.tags["RunMode"] == "portfolio-v2"
        assert "EmergencyProfile" not in manifest.tags
        assert manifest.proof_protocol.id == "finance-proof.v2"
        assert manifest.training.max_steps == plan.protocol.max_steps
        assert manifest.training.seed == 17
        task_filter = json.loads(manifest.tags["PortfolioTaskFilter"])
        assert task_filter == [task.value for task in slot.tasks]
        if slot.role == "generalist":
            assert task_filter == [task.value for task in Task]
        else:
            assert len(task_filter) == 1
        if slot.recipe == "logit.v1":
            assert manifest.training.qlora.capability_evidence.memory_dry_run is not None


def test_materializer_rejects_v1_smoke_knobs_and_mixed_proof(plan) -> None:
    wave = plan.screen_waves[0]
    slot = wave.active_slots[0]
    manifest = materialize_slot(
        plan=plan,
        wave=wave,
        slot=slot,
        readiness=make_readiness(plan, Tier.NANO),
        evidence=make_materialization_evidence(slot),
    )
    smoke_training = manifest.training.model_copy(
        update={"max_steps": 8},
    )
    mixed = manifest.model_copy(update={"training": smoke_training})
    with pytest.raises(ValueError, match="differs from portfolio"):
        validate_materialized_manifest(plan=plan, wave=wave, slot=slot, manifest=mixed)
    mixed_proof = manifest.model_copy(
        update={
            "proof_protocol": manifest.proof_protocol.model_copy(
                update={"id": "finance-proof.v1", "sha256": H["1"]}
            )
        }
    )
    with pytest.raises(ValueError, match="differs from portfolio"):
        validate_materialized_manifest(
            plan=plan,
            wave=wave,
            slot=slot,
            manifest=mixed_proof,
        )


@pytest.mark.parametrize(
    ("wave_index", "expected_instance", "expected_accelerator"),
    [
        (0, "ml.g5.48xlarge", "NVIDIA A10G"),
        (3, "ml.p4de.24xlarge", "NVIDIA A100 80GB"),
    ],
)
def test_portfolio_campaign_adapter_supports_g5_and_p4de(
    plan,
    tmp_path,
    wave_index,
    expected_instance,
    expected_accelerator,
) -> None:
    wave = plan.screen_waves[wave_index]
    paths = _materialize_wave(plan, wave, tmp_path / "source")
    bundle = stage_portfolio_wave(
        destination=tmp_path / "bundle",
        plan=plan,
        wave=wave,
        manifest_paths_by_model_id=paths,
        wave_input_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/input/",
        wave_output_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/output/",
    )
    assert bundle.index.hardware.instance_type == expected_instance
    assert bundle.index.hardware.accelerator == expected_accelerator
    assert len(bundle.index.slots) == 16
    assert [slot.gpu for slot in bundle.index.slots if slot.node == 0] == list(range(8))
    assert [slot.gpu for slot in bundle.index.slots if slot.node == 1] == list(range(8))
    active = [slot for slot in bundle.index.slots if slot.state == "planned"]
    assert len(active) == len(wave.active_slots)
    assert all(slot.manifest_sha256 is not None for slot in active)


def test_failed_and_not_started_slots_never_repartition_and_include_cost(plan, tmp_path) -> None:
    wave = plan.screen_waves[0]
    bundle = stage_portfolio_wave(
        destination=tmp_path / "bundle",
        plan=plan,
        wave=wave,
        manifest_paths_by_model_id=_materialize_wave(plan, wave, tmp_path / "source"),
        wave_input_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/input/",
        wave_output_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/output/",
    )
    outcomes = {}
    for binding in bundle.index.slots:
        if binding.state == "not_started":
            outcome = SlotOutcome(slot=binding.slot, state="not_started")
        elif binding.slot == 5:
            outcome = SlotOutcome(
                slot=binding.slot,
                state="failed",
                error="synthetic test failure",
            )
        else:
            outcome = SlotOutcome(
                slot=binding.slot,
                state="succeeded",
                artifact_checksum_sha256=H["1"],
                proof_report_sha256=H["2"],
            )
        outcomes[binding.slot] = outcome
    ledger = build_execution_ledger(
        bundle.index,
        outcomes=outcomes,
        parent_actual_cost_microusd=(10_001, 20_002),
    )
    assert [(slot.slot, slot.node, slot.gpu) for slot in ledger.slots] == [
        (index, index % 2, index // 2) for index in range(16)
    ]
    assert ledger.slots[5].state == "failed"
    assert (ledger.slots[6].node, ledger.slots[6].gpu) == (0, 3)
    assert all(ledger.slots[index].state == "not_started" for index in range(12, 16))
    assert sum(slot.allocated_actual_cost_microusd for slot in ledger.slots) == 30_003


def _container_evidence(plan) -> ContainerStagingEvidence:
    inventory = b'{"staged":["campaign","portfolio-task-filter"]}'
    return container_staging_evidence(
        runtime_image_digest=IMAGE_DIGEST,
        entrypoint=(CONTAINER_PYTHON, "-m", CAMPAIGN_ORCHESTRATOR_MODULE),
        module_sha256={module: H["3"] for module in REQUIRED_IMAGE_MODULES},
        source_inventory_sha256=sha256_hex(inventory),
        source_inventory_size_bytes=len(inventory),
        portfolio_task_filter_integration_sha256=H["4"],
        training_protocol_sha256=plan.protocol.protocol_sha256,
    )


def test_launch_plan_is_dry_run_and_requires_container_staging(plan, tmp_path) -> None:
    wave = plan.screen_waves[3]
    bundle = stage_portfolio_wave(
        destination=tmp_path / "bundle",
        plan=plan,
        wave=wave,
        manifest_paths_by_model_id=_materialize_wave(plan, wave, tmp_path / "source"),
        wave_input_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/input/",
        wave_output_s3_prefix=f"s3://distillery-artifacts-{ACCOUNT}/portfolio/output/",
    )
    container = _container_evidence(plan)
    evidence = PortfolioAwsEvidence(
        aws_account_id=ACCOUNT,
        iam_role_arn=f"arn:aws:iam::{ACCOUNT}:role/DistillerySageMakerTrainingRole",
        ecr_image_uri=(
            f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/distillery-training@{IMAGE_DIGEST}"
        ),
        models_s3_uri=f"s3://distillery-artifacts-{ACCOUNT}/models/",
        dataset_bundle_s3_uri=plan.dataset.uri,
        container=container,
    )
    launch = build_portfolio_launch_plan(
        bundle=bundle,
        plan=plan,
        wave=wave,
        evidence=evidence,
    )
    assert launch.dry_run is True
    assert len(launch.jobs) == 2
    assert {
        job["ResourceConfig"]["InstanceType"]  # type: ignore[index]
        for job in launch.jobs
    } == {"ml.p4de.24xlarge"}
    assert all(job["EnableNetworkIsolation"] is True for job in launch.jobs)
    with pytest.raises(ValidationError, match="missing staged modules"):
        container_staging_evidence(
            runtime_image_digest=IMAGE_DIGEST,
            entrypoint=(CONTAINER_PYTHON, "-m", CAMPAIGN_ORCHESTRATOR_MODULE),
            module_sha256={"experiments.aws_smoke.train": H["3"]},
            source_inventory_sha256=H["4"],
            source_inventory_size_bytes=1,
            portfolio_task_filter_integration_sha256=H["5"],
            training_protocol_sha256=plan.protocol.protocol_sha256,
        )


def test_task_filter_runtime_resolves_only_sealed_view(plan, tmp_path) -> None:
    wave = plan.screen_waves[1]
    slot = next(
        slot
        for slot in wave.active_slots
        if slot.tasks == (Task.MERCHANT_TAGGING,) and slot.arm is PortfolioArm.SEQUENCE_KD
    )
    manifest = materialize_slot(
        plan=plan,
        wave=wave,
        slot=slot,
        readiness=make_readiness(plan, Tier.NANO),
        evidence=make_materialization_evidence(slot),
    )
    payload = json.loads(manifest.tags["PortfolioDatasetView"])
    view_dir = tmp_path.joinpath(*Path(payload["relative_prefix"]).parts)
    view_dir.mkdir(parents=True)
    train = b'{"task":"merchant_tagging","split":"train"}\n'
    validation = b'{"task":"merchant_tagging","split":"validation"}\n'
    (view_dir / "train.jsonl").write_bytes(train)
    (view_dir / "validation.jsonl").write_bytes(validation)
    payload["split_sha256"] = {
        "train": sha256_hex(train),
        "validation": sha256_hex(validation),
    }
    tags = dict(manifest.tags)
    tags["PortfolioDatasetView"] = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    runtime_manifest = manifest.model_copy(update={"tags": tags})
    assert resolve_task_dataset_dir(runtime_manifest, tmp_path) == view_dir.resolve()
    (view_dir / "train.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="train hash mismatch"):
        resolve_task_dataset_dir(runtime_manifest, tmp_path)
