"""Job control: distinct names, serial resume, mock SageMaker client."""

from __future__ import annotations

from typing import Any

from experiments.aws_smoke.job_control import (
    JobStatusView,
    next_resumable_arm,
    status_campaign,
    stop_job,
)
from experiments.aws_smoke.manifests import job_name_for_arm
from experiments.aws_smoke.profile import REQUIRED_ARMS


class FakeSageMaker:
    def __init__(self, jobs: dict[str, dict[str, Any]] | None = None) -> None:
        self.jobs = jobs or {}
        self.stop_calls: list[str] = []

    def describe_training_job(self, *, TrainingJobName: str) -> dict[str, Any]:
        if TrainingJobName not in self.jobs:
            error = Exception(f"Could not find training job {TrainingJobName}")
            error.response = {  # type: ignore[attr-defined]
                "Error": {"Code": "ValidationException", "Message": "not found"}
            }
            raise error
        return self.jobs[TrainingJobName]

    def stop_training_job(self, *, TrainingJobName: str) -> dict[str, Any]:
        self.stop_calls.append(TrainingJobName)
        current = self.jobs[TrainingJobName]
        current["TrainingJobStatus"] = "Stopping"
        return {}

    def list_training_jobs(self, **kwargs: Any) -> dict[str, Any]:
        status = kwargs.get("StatusEquals")
        summaries = [
            {"TrainingJobName": name}
            for name, payload in self.jobs.items()
            if payload.get("TrainingJobStatus") == status
        ]
        return {"TrainingJobSummaries": summaries}


def test_job_names_are_arm_specific() -> None:
    names = {
        arm: job_name_for_arm(arm, manifest_sha256=("a" * 64))
        for arm in REQUIRED_ARMS
    }
    assert len(set(names.values())) == 3
    assert "oracle-sft" in names["oracle_sft"]
    assert "ce-ablation" in names["ce_ablation"]
    assert "logit-kd" in names["logit_kd"]


def test_next_resumable_serial_quota() -> None:
    statuses = {
        "oracle_sft": JobStatusView(
            job_name="j1",
            arm="oracle_sft",
            status="Completed",
            secondary_status=None,
            failure_reason=None,
            terminal=True,
            exists=True,
        ),
        "ce_ablation": JobStatusView(
            job_name="j2",
            arm="ce_ablation",
            status="NotStarted",
            secondary_status=None,
            failure_reason=None,
            terminal=False,
            exists=False,
        ),
        "logit_kd": JobStatusView(
            job_name="j3",
            arm="logit_kd",
            status="NotStarted",
            secondary_status=None,
            failure_reason=None,
            terminal=False,
            exists=False,
        ),
    }
    assert next_resumable_arm(statuses, ordered_arms=REQUIRED_ARMS) == "ce_ablation"

    statuses["ce_ablation"] = JobStatusView(
        job_name="j2",
        arm="ce_ablation",
        status="InProgress",
        secondary_status="Training",
        failure_reason=None,
        terminal=False,
        exists=True,
    )
    assert next_resumable_arm(statuses, ordered_arms=REQUIRED_ARMS) is None


def test_status_and_stop_with_fake_client() -> None:
    client = FakeSageMaker(
        {
            "aws-smoke-oracle-sft-aaa": {
                "TrainingJobStatus": "InProgress",
                "SecondaryStatus": "Training",
            }
        }
    )
    views = status_campaign(
        client,
        {
            "oracle_sft": "aws-smoke-oracle-sft-aaa",
            "ce_ablation": "aws-smoke-ce-ablation-bbb",
            "logit_kd": "aws-smoke-logit-kd-ccc",
        },
    )
    assert views["oracle_sft"].exists is True
    assert views["ce_ablation"].exists is False
    stopped = stop_job(client, "aws-smoke-oracle-sft-aaa")
    assert stopped.status == "Stopping"
    assert client.stop_calls == ["aws-smoke-oracle-sft-aaa"]
