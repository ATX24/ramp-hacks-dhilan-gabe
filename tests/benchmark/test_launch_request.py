"""Launch request safety: digest pin, MaxRuntime, tags, cost cap."""

from __future__ import annotations

import pytest

from experiments.benchmark.launch import (
    HARD_CAP_USD,
    LaunchConfig,
    build_create_training_job_request,
    estimate_cost_usd,
)


def _cfg(**overrides: object) -> LaunchConfig:
    base = dict(
        instance_type="ml.g5.xlarge",
        image_uri=(
            "225989358036.dkr.ecr.us-east-1.amazonaws.com/"
            "distillery-training@sha256:" + ("a" * 64)
        ),
        max_runtime_seconds=3600,
        dtype="bf16",
        warmups=20,
        timed=200,
        max_new_tokens=128,
        batch_sizes="1,8",
        dry_run=True,
        execute=False,
        cost_cap_usd=HARD_CAP_USD,
    )
    base.update(overrides)
    return LaunchConfig(**base)  # type: ignore[arg-type]


def test_estimate_g5_under_cap() -> None:
    cost = estimate_cost_usd("ml.g5.xlarge", 5400)
    assert cost < HARD_CAP_USD
    assert abs(cost - 1.408 * 1.5) < 1e-9


def test_request_is_tagged_finite_and_isolated() -> None:
    request = build_create_training_job_request(_cfg(), job_name="dist-bench-test")
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600
    assert request["EnableNetworkIsolation"] is True
    assert request["ResourceConfig"]["InstanceCount"] == 1
    tags = {t["Key"]: t["Value"] for t in request["Tags"]}
    assert tags["Component"] == "distillery-benchmark"
    assert tags["CostCapUSD"] == "100"
    assert tags["AutoCleanup"] == "true"
    assert "@sha256:" in request["AlgorithmSpecification"]["TrainingImage"]
    assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == [
        "bash",
        "/opt/ml/input/data/code/sm_entrypoint.sh",
    ]
    assert any(ch["ChannelName"] == "code" for ch in request["InputDataConfig"])


def test_unpinned_image_rejected() -> None:
    with pytest.raises(ValueError, match="digest-pinned"):
        build_create_training_job_request(
            _cfg(image_uri="225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training:latest"),
            job_name="x",
        )


def test_cost_cap_blocks_long_p4de() -> None:
    # 4 hours on p4de exceeds $100.
    with pytest.raises(ValueError, match="hard cap"):
        build_create_training_job_request(
            _cfg(instance_type="ml.p4de.24xlarge", max_runtime_seconds=4 * 3600),
            job_name="x",
        )
