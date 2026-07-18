"""Two-node g5.48 wave partition and aggregate-cost tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aws_smoke.campaign_index import (
    CampaignPricingEvidenceReference,
    campaign_hardware_profile,
)
from experiments.aws_smoke.campaign_wave import (
    ParentCampaignCost,
    aggregate_wave_cost,
    partition_wave_manifests,
    verify_wave_bundle,
)
from experiments.aws_smoke.channels import load_manifest
from experiments.aws_smoke.pins import EmergencyEvidence
from tests.aws_smoke.campaign_support import (
    FIXED_TIME,
    G5_48_PRICE_MICROUSD,
    stage_test_wave,
    write_test_manifests,
)


@pytest.mark.parametrize(
    ("run_count", "expected_sizes"),
    [(12, (6, 6)), (13, (7, 6)), (16, (8, 8))],
)
def test_wave_partitions_are_balanced_disjoint_and_deterministic(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
    run_count: int,
    expected_sizes: tuple[int, int],
) -> None:
    wave = stage_test_wave(tmp_path, valid_evidence, run_count=run_count)
    assert tuple(len(campaign.index.arms) for campaign in wave.campaigns) == (expected_sizes)
    node_runs = [tuple(arm.run_id for arm in campaign.index.arms) for campaign in wave.campaigns]
    assert node_runs[0] == wave.index.ordered_run_ids[0::2]
    assert node_runs[1] == wave.index.ordered_run_ids[1::2]
    assert set(node_runs[0]).isdisjoint(node_runs[1])
    assert wave.campaigns[0].index.hardware == wave.campaigns[1].index.hardware
    assert wave.campaigns[0].index.pricing == wave.campaigns[1].index.pricing
    verify_wave_bundle(wave.root, expected_index_sha256=wave.index_sha256)


@pytest.mark.parametrize("run_count", [11, 17])
def test_wave_requires_twelve_to_sixteen_runs(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
    run_count: int,
) -> None:
    with pytest.raises(ValueError, match="12–16"):
        stage_test_wave(tmp_path, valid_evidence, run_count=run_count)


def test_cross_node_duplicate_runs_and_prefixes_fail_before_staging(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    hardware = campaign_hardware_profile("g5-48xlarge-8xa10g-independent-v1")
    manifests = write_test_manifests(
        tmp_path / "source",
        valid_evidence,
        hardware=hardware,
        arm_count=12,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    pricing = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/g5-48.json",
        evidence_sha256="7" * 64,
        region=valid_evidence.aws_region,
        instance_type=hardware.instance_type,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
        attested_by="test",
        attested_at=FIXED_TIME,
    )
    duplicated = [*manifests[:-1], manifests[0]]
    with pytest.raises(ValueError, match="duplicate run_id"):
        partition_wave_manifests(
            duplicated,
            hardware=hardware,
            pricing=pricing,
        )


def test_wave_rejects_cross_node_output_and_protocol_mismatches(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    hardware = campaign_hardware_profile("g5-48xlarge-8xa10g-independent-v1")
    pricing = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/g5-48.json",
        evidence_sha256="7" * 64,
        region=valid_evidence.aws_region,
        instance_type=hardware.instance_type,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
        attested_by="test",
        attested_at=FIXED_TIME,
    )

    output_manifests = write_test_manifests(
        tmp_path / "output-source",
        valid_evidence,
        hardware=hardware,
        arm_count=12,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    first = load_manifest(output_manifests[0])
    last = load_manifest(output_manifests[-1]).model_copy(update={"output": first.output})
    output_manifests[-1].write_text(
        json.dumps(last.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate output_prefix"):
        partition_wave_manifests(
            output_manifests,
            hardware=hardware,
            pricing=pricing,
        )

    protocol_manifests = write_test_manifests(
        tmp_path / "protocol-source",
        valid_evidence,
        hardware=hardware,
        arm_count=12,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    final = load_manifest(protocol_manifests[-1])
    mismatched = final.model_copy(
        update={
            "training": final.training.model_copy(
                update={"max_steps": final.training.max_steps + 1}
            )
        }
    )
    protocol_manifests[-1].write_text(
        json.dumps(mismatched.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matched protocol inputs"):
        partition_wave_manifests(
            protocol_manifests,
            hardware=hardware,
            pricing=pricing,
        )


def test_wave_rejects_wrong_hardware_and_price(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    hardware = campaign_hardware_profile("g5-48xlarge-8xa10g-independent-v1")
    manifests = write_test_manifests(
        tmp_path / "source",
        valid_evidence,
        hardware=hardware,
        arm_count=12,
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    wrong_hardware = campaign_hardware_profile("p4de-24xlarge-8xa100-80gb-independent-v1")
    wrong_pricing = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/p4de.json",
        evidence_sha256="9" * 64,
        region=valid_evidence.aws_region,
        instance_type=wrong_hardware.instance_type,
        hourly_price_microusd=31_564_107,
        attested_by="test",
        attested_at=FIXED_TIME,
    )
    with pytest.raises(ValueError, match="requires 8-GPU g5.48"):
        partition_wave_manifests(
            manifests,
            hardware=wrong_hardware,
            pricing=wrong_pricing,
        )

    mismatched_price = CampaignPricingEvidenceReference(
        reference="https://pricing.example.test/g5-48-wrong.json",
        evidence_sha256="a" * 64,
        region=valid_evidence.aws_region,
        instance_type=hardware.instance_type,
        hourly_price_microusd=G5_48_PRICE_MICROUSD + 1,
        attested_by="test",
        attested_at=FIXED_TIME,
    )
    with pytest.raises(ValueError, match="wave price mismatch"):
        partition_wave_manifests(
            manifests,
            hardware=hardware,
            pricing=mismatched_price,
        )


def test_aggregate_cost_counts_each_parent_exactly_once(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    wave = stage_test_wave(tmp_path, valid_evidence)
    first_sha = wave.index.campaigns[0].campaign_index_sha256
    second_sha = wave.index.campaigns[1].campaign_index_sha256
    aggregate = aggregate_wave_cost(
        wave,
        [
            ParentCampaignCost(
                campaign_index_sha256=second_sha,
                parent_cost_microusd=20_360_001,
            ),
            ParentCampaignCost(
                campaign_index_sha256=first_sha,
                parent_cost_microusd=20_360_000,
            ),
        ],
    )
    assert aggregate.aggregate_parent_cost_microusd == 40_720_001
    assert [cost.campaign_index_sha256 for cost in aggregate.parents] == [
        first_sha,
        second_sha,
    ]

    duplicate = ParentCampaignCost(
        campaign_index_sha256=first_sha,
        parent_cost_microusd=1,
    )
    with pytest.raises(ValueError, match="double count"):
        aggregate_wave_cost(wave, [duplicate, duplicate])
