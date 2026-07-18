"""Builders for isolated multi-GPU campaign tests."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from distillery.contracts.hashing import content_sha256
from distillery.contracts.manifest import (
    ManifestCompletionEvidence,
    ManifestCost,
    ManifestDatasetRef,
    ManifestModels,
    ManifestModelSpec,
    ManifestOutput,
    ManifestProofProtocol,
    ManifestQLoRAConfig,
    ManifestRecipe,
    ManifestRuntime,
    ManifestTraining,
    SealedRunManifest,
)
from experiments.aws_smoke.campaign_index import (
    MICRO_USD_PER_USD,
    CampaignHardwareProfile,
    CampaignPricingEvidenceReference,
    HardwareProfileId,
    VerifiedCampaignBundle,
    campaign_hardware_profile,
    stage_campaign_bundle,
)
from experiments.aws_smoke.campaign_wave import VerifiedWaveBundle, stage_two_job_wave
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import ALL_ARMS

FIXED_TIME = datetime(2026, 7, 18, 17, 0, tzinfo=UTC)
G5_12_PRICE_MICROUSD = 7_090_000
G5_48_PRICE_MICROUSD = 20_360_000
P4DE_PRICE_MICROUSD = 31_564_107
TEST_RUNTIME_SECONDS = 20


def _price_text(microusd: int) -> str:
    whole, fractional = divmod(microusd, MICRO_USD_PER_USD)
    return f"{whole}.{fractional:06d}".rstrip("0").rstrip(".")


def write_test_manifests(
    root: Path,
    evidence: EmergencyEvidence,
    *,
    hardware: CampaignHardwareProfile,
    arm_count: int = 4,
    hourly_price_microusd: int = G5_12_PRICE_MICROUSD,
    max_runtime_seconds: int = TEST_RUNTIME_SECONDS,
) -> list[Path]:
    paths: list[Path] = []
    campaign_output = evidence.artifact_s3_prefix.rstrip("/") + "/campaigns/campaign_test/output/"
    example_id = "ex_campaign01"
    completion_record = content_sha256(
        {
            "example_id": example_id,
            "target_text": '{"ok":true}',
            "target_source": "oracle",
        }
    )
    for ordinal in range(arm_count):
        arm = ALL_ARMS[ordinal % len(ALL_ARMS)]
        run_id = f"run_campaign-{ordinal}-{arm.replace('_', '-')}"
        protocol_sha256 = content_sha256({"run_id": run_id, "arm": arm, "protocol": "test"})
        emergency_config = {
            "arm": arm,
            "max_runtime_seconds": max_runtime_seconds,
        }
        max_cost = (
            math.ceil(hourly_price_microusd * max_runtime_seconds / 3600 / MICRO_USD_PER_USD * 100)
            / 100
        )
        manifest = SealedRunManifest(
            run_id=run_id,
            created_at=FIXED_TIME,
            dataset=ManifestDatasetRef(
                dataset_id="ds_campaign01",
                uri=evidence.dataset_s3_uri,
                sha256="1" * 64,
                split_sha256={"train": "2" * 64, "validation": "3" * 64},
            ),
            models=ManifestModels(
                teacher=ManifestModelSpec(
                    id=evidence.teacher_model_id,
                    revision=evidence.teacher_revision,
                    tokenizer_sha256=evidence.teacher_tokenizer_sha256,
                    chat_template_sha256=evidence.teacher_chat_template_sha256,
                ),
                student=ManifestModelSpec(
                    id=evidence.student_model_id,
                    revision=evidence.student_revision,
                    tokenizer_sha256=evidence.student_tokenizer_sha256,
                    chat_template_sha256=evidence.student_chat_template_sha256,
                ),
            ),
            recipe=ManifestRecipe(
                requested="sequence.v1",
                resolved="sequence.v1",
                resolver_reasons=("explicit_request",),
            ),
            training=ManifestTraining(
                seed=17 + ordinal,
                max_steps=8,
                token_budget=0,
                max_length=512,
                qlora=ManifestQLoRAConfig(),
                completion_evidence=ManifestCompletionEvidence(
                    source_file_sha256="4" * 64,
                    canonical_records_sha256=content_sha256({"records": [completion_record]}),
                    record_sha256={example_id: completion_record},
                    provenance_sha256=content_sha256(
                        {"run_id": run_id, "completion": completion_record}
                    ),
                    completion_token_counts={example_id: 8},
                    completion_tokenizer_sha256=evidence.student_tokenizer_sha256,
                    label_source_counts={"oracle": 1},
                    accepted_example_count=1,
                ),
            ),
            proof_protocol=ManifestProofProtocol(
                id=evidence.proof_protocol_id,
                sha256=evidence.proof_protocol_sha256,
            ),
            runtime=ManifestRuntime(
                backend="sagemaker",
                region=evidence.aws_region,
                instance_type=hardware.instance_type,
                image_digest=evidence.image_digest,
            ),
            cost=ManifestCost(
                max_run_usd=max_cost,
                estimate_low_usd=0.0,
                estimate_high_usd=max_cost,
            ),
            output=ManifestOutput(
                prefix=f"{campaign_output}arms/{run_id}/",
            ),
            package_lock_hash=evidence.package_lock_hash,
            source_revision=evidence.source_revision,
            license_dispositions={
                evidence.student_model_id: evidence.license_disposition,
                evidence.teacher_model_id: evidence.license_disposition,
            },
            tags={
                "Arm": arm,
                "EnableNetworkIsolation": "true",
                "EmergencyConfig": json.dumps(
                    emergency_config,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "EmergencyConfigSha256": content_sha256(emergency_config),
                "HourlyUsd": _price_text(hourly_price_microusd),
                "MaxRuntimeInSeconds": str(max_runtime_seconds),
                "TrainingProtocolSha256": protocol_sha256,
            },
            sampler_order_hash="5" * 64,
        )
        path = root / f"{ordinal:02d}-{arm}" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def stage_test_campaign(
    tmp_path: Path,
    evidence: EmergencyEvidence,
    *,
    profile_id: HardwareProfileId = "g5-12xlarge-4xa10g-independent-v1",
    arm_count: int | None = None,
    hourly_price_microusd: int = G5_12_PRICE_MICROUSD,
    max_runtime_seconds: int = TEST_RUNTIME_SECONDS,
) -> VerifiedCampaignBundle:
    hardware = campaign_hardware_profile(profile_id)
    arm_count = hardware.gpu_count if arm_count is None else arm_count
    manifests = write_test_manifests(
        tmp_path / "source",
        evidence,
        hardware=hardware,
        arm_count=arm_count,
        hourly_price_microusd=hourly_price_microusd,
        max_runtime_seconds=max_runtime_seconds,
    )
    pricing = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/attestations/g5-12xlarge.json",
        evidence_sha256="6" * 64,
        region=evidence.aws_region,
        instance_type=hardware.instance_type,
        hourly_price_microusd=hourly_price_microusd,
        attested_by="campaign-test",
        attested_at=FIXED_TIME,
    )
    output_prefix = evidence.artifact_s3_prefix.rstrip("/") + "/campaigns/campaign_test/output/"
    return stage_campaign_bundle(
        destination=tmp_path / "bundle",
        campaign_id="campaign_test",
        created_at=FIXED_TIME,
        ordered_manifest_paths=manifests,
        hardware=hardware,
        pricing=pricing,
        input_s3_prefix=(
            evidence.artifact_s3_prefix.rstrip("/") + "/campaigns/campaign_test/input/"
        ),
        campaign_output_prefix=output_prefix,
    )


def replace_arm(
    bundle: VerifiedCampaignBundle,
    ordinal: int,
    **updates: object,
) -> tuple:
    arms = list(bundle.index.arms)
    arms[ordinal] = arms[ordinal].model_copy(update=updates)
    return tuple(arms)


def stage_test_wave(
    tmp_path: Path,
    evidence: EmergencyEvidence,
    *,
    run_count: int = 16,
) -> VerifiedWaveBundle:
    hardware = campaign_hardware_profile("g5-48xlarge-8xa10g-independent-v1")
    manifests = write_test_manifests(
        tmp_path / "wave-source",
        evidence,
        hardware=hardware,
        arm_count=run_count,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    pricing = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/attestations/g5-48xlarge.json",
        evidence_sha256="7" * 64,
        region=evidence.aws_region,
        instance_type=hardware.instance_type,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
        attested_by="campaign-test",
        attested_at=FIXED_TIME,
    )
    return stage_two_job_wave(
        destination=tmp_path / "wave-bundle",
        wave_id="wave_test",
        created_at=FIXED_TIME,
        ordered_manifest_paths=manifests,
        hardware=hardware,
        pricing=pricing,
        wave_input_s3_prefix=(evidence.artifact_s3_prefix.rstrip("/") + "/waves/wave_test/input/"),
        wave_output_s3_prefix=(
            evidence.artifact_s3_prefix.rstrip("/") + "/waves/wave_test/output/"
        ),
    )
