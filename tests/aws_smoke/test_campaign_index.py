"""Sealed campaign index topology, tamper, and collision tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.aws_smoke.campaign_index import (
    SEALED_CAMPAIGN_INDEX_FILENAME,
    CampaignArmBinding,
    CampaignPricingEvidenceReference,
    SealedCampaignIndex,
    campaign_hardware_profile,
    campaign_protocol_sha256,
    load_sealed_campaign_index,
    stage_campaign_bundle,
    verify_campaign_bundle,
)
from experiments.aws_smoke.memory import (
    A100_80GB_VRAM_BYTES,
    EmergencyMemoryProbeEvidence,
    select_precision_mode,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from tests.aws_smoke.campaign_support import (
    FIXED_TIME,
    G5_12_PRICE_MICROUSD,
    G5_48_PRICE_MICROUSD,
    P4DE_PRICE_MICROUSD,
    stage_test_campaign,
    write_test_manifests,
)


def test_g5_48_campaign_seals_eight_ordered_slots(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(
        tmp_path,
        valid_evidence,
        profile_id="g5-48xlarge-8xa10g-independent-v1",
        arm_count=8,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    assert bundle.index.hardware.instance_type == "ml.g5.48xlarge"
    assert bundle.index.hardware.gpu_count == 8
    assert [arm.gpu_slot for arm in bundle.index.arms] == list(range(8))
    assert len({arm.run_id for arm in bundle.index.arms}) == 8
    assert len({arm.output_prefix for arm in bundle.index.arms}) == 8
    assert (
        load_sealed_campaign_index(bundle.root / SEALED_CAMPAIGN_INDEX_FILENAME).seal_sha256()
        == bundle.index_sha256
    )


@pytest.mark.parametrize(
    ("profile_id", "price_microusd", "instance_type", "gpu_count"),
    [
        (
            "g5-12xlarge-4xa10g-independent-v1",
            G5_12_PRICE_MICROUSD,
            "ml.g5.12xlarge",
            4,
        ),
        (
            "p4de-24xlarge-8xa100-80gb-independent-v1",
            P4DE_PRICE_MICROUSD,
            "ml.p4de.24xlarge",
            8,
        ),
    ],
)
def test_fallback_and_stretch_profiles_are_strictly_bound(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
    profile_id: str,
    price_microusd: int,
    instance_type: str,
    gpu_count: int,
) -> None:
    bundle = stage_test_campaign(
        tmp_path,
        valid_evidence,
        profile_id=profile_id,  # type: ignore[arg-type]
        hourly_price_microusd=price_microusd,
    )
    assert bundle.index.hardware.instance_type == instance_type
    assert bundle.index.hardware.gpu_count == gpu_count
    assert bundle.index.pricing.hourly_price_microusd == price_microusd
    verify_campaign_bundle(bundle.root, expected_index_sha256=bundle.index_sha256)


def test_p4de_memory_evidence_uses_a100_capacity(
    valid_evidence: EmergencyEvidence,
) -> None:
    base = valid_evidence.memory_probe_evidence
    peak = 20 * 1024**3
    evidence = EmergencyMemoryProbeEvidence.model_validate(
        {
            **base.model_dump(mode="python"),
            "precision_mode": "bf16_lora",
            "device_type": "NVIDIA A100-SXM4-80GB",
            "peak_memory_bytes": peak,
            "capacity_memory_bytes": A100_80GB_VRAM_BYTES,
            "headroom_bytes": A100_80GB_VRAM_BYTES - peak,
            "instance_type": "ml.p4de.24xlarge",
        }
    )
    estimate = select_precision_mode(
        sealed_mode="bf16_lora",
        nf4_kernel_probe_passed=False,
        bf16_memory_evidence=evidence,
        max_length=512,
        microbatch=1,
        lora_rank=16,
        load_teacher=True,
    )
    assert estimate.fits is True
    assert estimate.device_bytes == A100_80GB_VRAM_BYTES

    with pytest.raises(ValidationError, match="A100-SXM4-80GB"):
        EmergencyMemoryProbeEvidence.model_validate(
            {
                **evidence.model_dump(mode="python"),
                "device_type": "NVIDIA A10G",
            }
        )


def test_tampered_index_and_manifest_fail_closed(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path, valid_evidence)
    index_path = bundle.root / SEALED_CAMPAIGN_INDEX_FILENAME
    index_path.write_bytes(index_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        verify_campaign_bundle(bundle.root)

    second = stage_test_campaign(tmp_path / "second", valid_evidence)
    manifest_path = second.root / second.index.arms[0].manifest_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["training"]["seed"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        verify_campaign_bundle(second.root)


def test_extra_cross_arm_file_is_rejected(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path, valid_evidence)
    leaked = bundle.root / "arms" / bundle.index.arms[0].run_id / "other-arm.txt"
    leaked.write_text("cross-arm leakage\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_campaign_bundle(bundle.root)


def test_duplicate_slots_runs_prefixes_and_unsafe_paths_are_rejected(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path, valid_evidence)
    first, second, *rest = bundle.index.arms

    duplicate_slot = second.model_copy(update={"gpu_slot": first.gpu_slot})
    with pytest.raises(ValidationError, match="duplicate gpu_slot"):
        bundle.index.model_copy(update={"arms": (first, duplicate_slot, *rest)})

    duplicate_prefix = second.model_copy(update={"output_prefix": first.output_prefix})
    with pytest.raises(ValidationError, match="duplicate output_prefix"):
        bundle.index.model_copy(update={"arms": (first, duplicate_prefix, *rest)})

    duplicate_run = CampaignArmBinding(
        ordinal=second.ordinal,
        arm=second.arm,
        run_id=first.run_id,
        manifest_path=first.manifest_path,
        manifest_sha256=second.manifest_sha256,
        protocol_sha256=second.protocol_sha256,
        gpu_slot=second.gpu_slot,
        output_prefix=second.output_prefix,
    )
    with pytest.raises(ValidationError, match="duplicate run_id"):
        bundle.index.model_copy(update={"arms": (first, duplicate_run, *rest)})

    with pytest.raises(ValidationError, match="manifest_path"):
        first.model_copy(update={"manifest_path": "../manifest.json"})


def test_arm_count_must_fit_hardware(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    hardware = campaign_hardware_profile("g5-12xlarge-4xa10g-independent-v1")
    manifests = write_test_manifests(
        tmp_path / "source",
        valid_evidence,
        hardware=hardware,
        arm_count=5,
        hourly_price_microusd=G5_12_PRICE_MICROUSD,
    )
    pricing = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/g5-12.json",
        evidence_sha256="8" * 64,
        region=valid_evidence.aws_region,
        instance_type=hardware.instance_type,
        hourly_price_microusd=G5_12_PRICE_MICROUSD,
        attested_by="test",
        attested_at=FIXED_TIME,
    )
    with pytest.raises(ValueError, match="between one and 4"):
        stage_campaign_bundle(
            destination=tmp_path / "bundle",
            campaign_id="campaign_too-many",
            created_at=FIXED_TIME,
            ordered_manifest_paths=manifests,
            hardware=hardware,
            pricing=pricing,
            input_s3_prefix="s3://test-bucket/input/",
            campaign_output_prefix="s3://test-bucket/output/",
        )


def test_hardware_and_price_mismatches_are_rejected(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path, valid_evidence)
    wrong_hardware = campaign_hardware_profile("g5-48xlarge-8xa10g-independent-v1")
    with pytest.raises(ValidationError, match="pricing instance_type"):
        bundle.index.model_copy(update={"hardware": wrong_hardware})

    wrong_pricing = bundle.index.pricing.model_copy(
        update={"hourly_price_microusd": G5_48_PRICE_MICROUSD}
    )
    wrong_index = SealedCampaignIndex(
        **{
            **bundle.index.model_dump(mode="python"),
            "pricing": wrong_pricing,
            "protocol_sha256": campaign_protocol_sha256(bundle.index.arms),
        }
    )
    index_path = bundle.root / SEALED_CAMPAIGN_INDEX_FILENAME
    index_path.write_bytes(wrong_index.canonical_bytes())
    (bundle.root / "sealed_campaign_index.sha256").write_text(
        wrong_index.seal_sha256() + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="attested hourly price mismatch"):
        verify_campaign_bundle(bundle.root)
