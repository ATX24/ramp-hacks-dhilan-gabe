"""Launch request isolation, canonical channels, and distinct-signal planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.aws_smoke.channels import discover_and_load_manifest
from experiments.aws_smoke.launch_plan import (
    CONTAINER_MANIFEST_PATH,
    ENTRYPOINT,
    build_create_training_job_request,
    discover_generated_manifest_paths,
    plan_serial_launch,
    plan_to_dict,
    stage_manifest_for_job,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import (
    CONTROL_LAUNCH_ORDER,
    DEFAULT_EMERGENCY_PROFILE,
    DEFAULT_UNIQUE_LAUNCH_ORDER,
)
from experiments.aws_smoke.safety import CONFIRM_PHRASE, CallerIdentity
from tests.aws_smoke.support import build_campaign


class FakeS3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, str] | None]] = []

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, str] | None = None,
    ) -> None:
        self.calls.append((filename, bucket, key, ExtraArgs))


def test_request_is_network_isolated_and_uses_canonical_manifest(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    manifest = discover_and_load_manifest(paths["logit_kd"].parent)
    request = build_create_training_job_request(
        manifest=manifest,
        evidence=valid_evidence,
        arm="logit_kd",
    )
    assert request["EnableNetworkIsolation"] is True
    assert request["ResourceConfig"] == {
        "InstanceType": "ml.g5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 30,
    }
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 900
    assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == ENTRYPOINT
    arguments = request["AlgorithmSpecification"]["ContainerArguments"]
    assert arguments[:2] == ["--manifest", CONTAINER_MANIFEST_PATH]
    assert request["RoleArn"] == valid_evidence.iam_role_arn
    assert (
        request["AlgorithmSpecification"]["TrainingImage"]
        == valid_evidence.ecr_image_uri
    )
    channels = {item["ChannelName"] for item in request["InputDataConfig"]}
    assert channels == {"manifest", "dataset", "models"}
    s3 = FakeS3()
    target = stage_manifest_for_job(
        s3,
        local_manifest_path=paths["logit_kd"],
        request=request,
    )
    assert target.endswith("/manifest/manifest.json")
    assert s3.calls == [
        (
            str(paths["logit_kd"]),
            f"distillery-artifacts-{valid_evidence.aws_account_id}",
            "artifacts/runs/run_awssmoke-logit-kd/manifest/manifest.json",
            {"ContentType": "application/json"},
        )
    ]


def test_default_plan_fails_without_third_distinct_signal(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    with pytest.raises(ValueError, match="sequence_kd"):
        plan_serial_launch(
            manifest_paths=paths,
            evidence=valid_evidence,
            profile_name="gabriel-cli",
            confirm=None,
            dry_run=True,
            identity_provider=None,
        )


def test_launch_rejects_dataset_hash_evidence_mismatch(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    manifest = discover_and_load_manifest(paths["logit_kd"].parent)
    payload = valid_evidence.model_dump(mode="json")
    payload["data_content_sha256"] = "f" * 64
    mismatched = EmergencyEvidence.model_validate(payload)
    with pytest.raises(ValueError, match="dataset hash"):
        build_create_training_job_request(
            manifest=manifest,
            evidence=mismatched,
            arm="logit_kd",
        )


def test_default_plan_prefers_three_distinct_signals(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    build_campaign(tmp_path, valid_evidence, include_sequence_kd=True)
    paths = discover_generated_manifest_paths(tmp_path / "manifests")
    plan = plan_serial_launch(
        manifest_paths=paths,
        evidence=valid_evidence,
        profile_name="gabriel-cli",
        confirm=None,
        dry_run=True,
        identity_provider=None,
    )
    assert tuple(job.arm for job in plan.jobs) == DEFAULT_UNIQUE_LAUNCH_ORDER
    assert plan.distinct_signal_count == 3
    assert all(job.distinct_training_signal for job in plan.jobs)
    assert len({job.job_name for job in plan.jobs}) == 3
    assert plan.quota_instance_count == 1
    assert plan.total_ceiling_usd == pytest.approx(
        DEFAULT_EMERGENCY_PROFILE.max_run_usd * 3
    )
    assert plan_to_dict(plan)["distinct_signal_count"] == 3


def test_explicit_control_plan_discloses_equivalence(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(tmp_path, valid_evidence)
    plan = plan_serial_launch(
        manifest_paths=paths,
        evidence=valid_evidence,
        profile_name="gabriel-cli",
        confirm=None,
        dry_run=True,
        identity_provider=None,
        arms=CONTROL_LAUNCH_ORDER,
        require_three_distinct=False,
    )
    assert plan.distinct_signal_count == 2
    ce = next(job for job in plan.jobs if job.arm == "ce_ablation")
    assert ce.distinct_training_signal is False
    assert ce.equivalent_to == "oracle_sft"


def test_execute_requires_confirmation(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths, _, _ = build_campaign(
        tmp_path,
        valid_evidence,
        include_sequence_kd=True,
    )

    def identity() -> CallerIdentity:
        return CallerIdentity(
            account=valid_evidence.aws_account_id,
            arn=f"arn:aws:iam::{valid_evidence.aws_account_id}:user/gabriel-cli",
            user_id="AIDATEST",
        )

    with pytest.raises(PermissionError, match="safety gates failed"):
        plan_serial_launch(
            manifest_paths=paths,
            evidence=valid_evidence,
            profile_name="gabriel-cli",
            confirm=None,
            dry_run=False,
            identity_provider=identity,
        )
    plan = plan_serial_launch(
        manifest_paths=paths,
        evidence=valid_evidence,
        profile_name="gabriel-cli",
        confirm=CONFIRM_PHRASE,
        dry_run=False,
        identity_provider=identity,
    )
    assert tuple(job.arm for job in plan.jobs) == DEFAULT_UNIQUE_LAUNCH_ORDER
