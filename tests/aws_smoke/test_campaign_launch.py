"""Plan-only SageMaker requests for approved campaign hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.aws_smoke.campaign_launch import (
    CAMPAIGN_ENTRYPOINT,
    CONTAINER_PYTHON,
    build_campaign_launch_plan,
    build_wave_launch_plan,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from tests.aws_smoke.campaign_support import (
    G5_12_PRICE_MICROUSD,
    G5_48_PRICE_MICROUSD,
    P4DE_PRICE_MICROUSD,
    TEST_RUNTIME_SECONDS,
    stage_test_campaign,
    stage_test_wave,
)


def test_primary_wave_plans_two_identical_eight_gpu_jobs(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    wave = stage_test_wave(tmp_path, valid_evidence, run_count=16)
    plan = build_wave_launch_plan(
        wave_root=wave.root,
        evidence=valid_evidence,
        expected_index_sha256=wave.index_sha256,
    )
    assert plan.dry_run is True
    assert len(plan.jobs) == 2
    assert all(job.instance_type == "ml.g5.48xlarge" for job in plan.jobs)
    assert all(job.instance_count == 1 for job in plan.jobs)
    assert all(job.gpu_count == 8 for job in plan.jobs)
    assert all(job.hourly_price_microusd == G5_48_PRICE_MICROUSD for job in plan.jobs)
    assert plan.aggregate_max_parent_cost_microusd == sum(
        job.max_parent_cost_microusd for job in plan.jobs
    )
    assert len({job.create_training_job_request["TrainingJobName"] for job in plan.jobs}) == 2

    for job in plan.jobs:
        request = job.create_training_job_request
        assert request["EnableNetworkIsolation"] is True
        assert request["ResourceConfig"]["InstanceType"] == "ml.g5.48xlarge"
        assert request["ResourceConfig"]["InstanceCount"] == 1
        assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == (CAMPAIGN_ENTRYPOINT)
        assert Path(CAMPAIGN_ENTRYPOINT[0]).is_absolute()
        assert CAMPAIGN_ENTRYPOINT[0] == CONTAINER_PYTHON
        channels = {channel["ChannelName"] for channel in request["InputDataConfig"]}
        assert channels == {"campaign", "dataset", "models"}
        assert request["HyperParameters"]["hourly_price_microusd"] == str(G5_48_PRICE_MICROUSD)


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
def test_single_campaign_plans_fallback_and_stretch_profiles(
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
    plan = build_campaign_launch_plan(
        campaign_root=bundle.root,
        evidence=valid_evidence,
        expected_index_sha256=bundle.index_sha256,
    )
    assert plan.instance_type == instance_type
    assert plan.gpu_count == gpu_count
    assert plan.hourly_price_microusd == price_microusd
    assert plan.create_training_job_request["StoppingCondition"] == {
        "MaxRuntimeInSeconds": TEST_RUNTIME_SECONDS
    }
    expected_max = (price_microusd * TEST_RUNTIME_SECONDS + 3600 - 1) // 3600
    assert plan.max_parent_cost_microusd == expected_max
