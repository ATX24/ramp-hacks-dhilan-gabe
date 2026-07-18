from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiments.qwen72b_fallback.cost import (
    P4DE_HOURLY_USD,
    P4DE_PRICE_SOURCE,
    ActiveResourceCost,
    CostAction,
    ResourceKind,
    seal_cost_evidence,
)
from experiments.qwen72b_fallback.launch import (
    launch_training_job,
    stop_training_job_and_verify,
)
from experiments.qwen72b_fallback.materializer import (
    REQUIREMENTS_LOCK_PATH,
    ROOT_VOLUME_GIB,
    _terminate_client_token_instances,
    build_materialization_plan,
    render_bootstrap,
    terminate_orphan,
)
from experiments.qwen72b_fallback.materializer_worker import MIN_VOLUME_BYTES
from experiments.qwen72b_fallback.profile import rehearsal_profile
from experiments.qwen72b_fallback.readiness import ExecutionAction

ROOT = Path(__file__).resolve().parents[2]


def test_materializer_bootstrap_is_pinned_large_and_not_renderable_as_bypass(
    authorization_factory,
) -> None:
    authorization = authorization_factory(
        action=ExecutionAction.MATERIALIZE,
        profile=None,
        launch_name="qwen72b-transfer-test",
    )
    plan = build_materialization_plan(authorization)
    bootstrap = render_bootstrap(plan, authorization)
    lock = REQUIREMENTS_LOCK_PATH.read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "qwen72b" / "materialize_to_s3.py").read_text(encoding="utf-8")
    assert plan.hf_transfer_enabled is False
    assert "HF_HUB_ENABLE_HF_TRANSFER=0" in bootstrap
    assert "uv archive checksum mismatch" in bootstrap
    assert "--require-hashes" in bootstrap
    assert "boto3==1.35.99" in lock
    assert MIN_VOLUME_BYTES == 200 * 1024**3
    assert ROOT_VOLUME_GIB >= 200
    assert "render-worker" not in script
    assert len(bootstrap.encode()) <= 16 * 1024


def test_cost_evidence_includes_orphan_exposure_and_forbids_retries() -> None:
    orphan = ActiveResourceCost(
        resource_id="qwen72b-orphan-job",
        resource_kind=ResourceKind.P4DE_TRAINING_JOB,
        age_seconds=3600,
        hourly_usd=P4DE_HOURLY_USD,
        accrued_usd=31.57,
    )
    evidence = seal_cost_evidence(
        action=CostAction.REHEARSAL,
        instance_type="ml.p4de.24xlarge",
        hourly_usd=P4DE_HOURLY_USD,
        price_source=P4DE_PRICE_SOURCE,
        max_runtime_seconds=3600,
        hard_cap_usd=100.0,
        active_resources=(orphan,),
    )
    assert evidence.active_orphan_accrued_usd == 31.57
    assert evidence.total_worst_case_usd == 63.14
    assert evidence.max_launch_attempts == 1
    assert evidence.retry_budget_usd == 0.0


class Ec2Orphan:
    def __init__(self) -> None:
        self.terminated = False

    def describe_instances(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-12345678",
                            "State": {"Name": "terminated" if self.terminated else "running"},
                            "Tags": [
                                {
                                    "Key": "DistilleryWorkstream",
                                    "Value": "qwen72b-fallback",
                                }
                            ],
                            "BlockDeviceMappings": [{"Ebs": {"DeleteOnTermination": True}}],
                        }
                    ]
                }
            ]
        }

    def terminate_instances(self, **_kwargs: Any) -> None:
        self.terminated = True


def test_orphan_cleanup_requires_typed_confirmation_and_verifies_termination() -> None:
    ec2 = Ec2Orphan()
    with pytest.raises(ValueError, match="typed confirmation"):
        terminate_orphan(
            ec2=ec2,
            instance_id="i-12345678",
            typed_confirmation="wrong",
            sleep=lambda _seconds: None,
        )
    evidence = terminate_orphan(
        ec2=ec2,
        instance_id="i-12345678",
        typed_confirmation="TERMINATE QWEN72B TRANSFER i-12345678",
        sleep=lambda _seconds: None,
    )
    assert evidence.final_state == "terminated"
    assert evidence.delete_on_termination is True


def test_ambiguous_ec2_launch_discovers_and_terminates_client_token_orphan() -> None:
    class AmbiguousEc2:
        terminated = False

        def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
            if "Filters" in kwargs:
                return {"Reservations": [{"Instances": [{"InstanceId": "i-abcdef12345678901"}]}]}
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-abcdef12345678901",
                                "State": {"Name": "terminated" if self.terminated else "running"},
                                "BlockDeviceMappings": [{"Ebs": {"DeleteOnTermination": True}}],
                            }
                        ]
                    }
                ]
            }

        def terminate_instances(self, **_kwargs: Any) -> None:
            self.terminated = True

    client = AmbiguousEc2()
    terminated = _terminate_client_token_instances(
        client,
        "q72b-test-token",
        sleep=lambda _seconds: None,
    )
    assert terminated == ("i-abcdef12345678901",)
    assert client.terminated is True


class ExistingJob:
    def describe_training_job(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "TrainingJobName": kwargs["TrainingJobName"],
            "TrainingJobStatus": "InProgress",
        }


class NoS3Writes:
    def put_object(self, **_kwargs: Any) -> None:
        raise AssertionError("duplicate launch must fail before S3 writes")


def test_duplicate_training_launch_is_rejected_before_side_effects(
    authorization_factory,
) -> None:
    profile = rehearsal_profile()
    authorization = authorization_factory(
        action=ExecutionAction.REHEARSAL,
        profile=profile,
        launch_name="qwen72b-rehearsal-duplicate",
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        launch_training_job(
            sagemaker=ExistingJob(),
            s3=NoS3Writes(),
            authorization=authorization,
            profile=profile,
        )


class StoppableJob:
    def __init__(self) -> None:
        self.stopped = False

    def describe_training_job(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:225989358036:training-job/x",
            "TrainingJobStatus": "Stopped" if self.stopped else "InProgress",
        }

    def list_tags(self, **_kwargs: Any) -> dict[str, Any]:
        return {"Tags": [{"Key": "DistilleryWorkstream", "Value": "qwen72b-fallback"}]}

    def stop_training_job(self, **_kwargs: Any) -> None:
        self.stopped = True


def test_training_stop_is_explicit_and_terminal_state_is_verified() -> None:
    job = StoppableJob()
    result = stop_training_job_and_verify(
        sagemaker=job,
        job_name="qwen72b-rehearsal-stop",
        typed_confirmation="STOP QWEN72B TRAINING qwen72b-rehearsal-stop",
        sleep=lambda _seconds: None,
    )
    assert result.stop_verified is True
    assert result.final_status == "Stopped"


def test_materializer_source_verifies_uploaded_bodies_merges_and_wipes() -> None:
    source = (ROOT / "experiments" / "qwen72b_fallback" / "materializer_worker.py").read_text(
        encoding="utf-8"
    )
    assert "sha256_stream" in source
    assert "merge_materialization_manifest" in source
    assert "IfMatch" in source and "IfNoneMatch" in source
    assert "materialization destination prefix is not empty" in source
    assert "delete_objects" in source
    assert "shutil.rmtree(WORK_ROOT" in source
    assert "local_wipe_complete" in source
