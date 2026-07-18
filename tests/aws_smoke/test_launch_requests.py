"""Launch request shape, cost ceiling, and separate job names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aws_smoke.dataset_subset import materialize_emergency_subset
from experiments.aws_smoke.launch_plan import (
    ENTRYPOINT,
    build_create_training_job_request,
    load_manifest,
    plan_serial_launch,
    plan_to_dict,
)
from experiments.aws_smoke.manifests import write_arm_manifests
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE, REQUIRED_ARMS
from experiments.aws_smoke.safety import CONFIRM_PHRASE, CallerIdentity


def _write_campaign(tmp_path: Path, evidence: EmergencyEvidence) -> dict:
    subset = materialize_emergency_subset(tmp_path / "subset")
    rows = [
        json.loads(line)
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths = write_arm_manifests(
        output_dir=tmp_path / "manifests",
        evidence=evidence,
        dataset_id="ds_awssmoke01",
        dataset_uri=evidence.dataset_s3_uri,
        dataset_sha256=subset.content_sha256,
        split_sha256=subset.split_sha256,
        example_ids=[str(r["example_id"]) for r in rows],
        tasks=[str(r["task"]) for r in rows],
        difficulties=[str(r["difficulty"]) for r in rows],
    )
    return paths


def test_create_training_job_request_bounds(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path, valid_evidence)
    manifest = load_manifest(paths["logit_kd"])
    request = build_create_training_job_request(
        manifest=manifest,
        evidence=valid_evidence,
        arm="logit_kd",
    )
    assert request["ResourceConfig"]["InstanceType"] == "ml.g5.xlarge"
    assert request["ResourceConfig"]["InstanceCount"] == 1
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] <= 15 * 60
    assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == ENTRYPOINT
    assert request["RoleArn"] == valid_evidence.iam_role_arn
    assert request["AlgorithmSpecification"]["TrainingImage"] == valid_evidence.ecr_image_uri
    channels = {c["ChannelName"] for c in request["InputDataConfig"]}
    assert channels == {"manifest", "dataset", "models"}


def test_serial_plan_has_three_distinct_jobs(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path, valid_evidence)
    plan = plan_serial_launch(
        manifest_paths=paths,
        evidence=valid_evidence,
        profile_name="gabriel-cli",
        confirm=None,
        dry_run=True,
        identity_provider=None,
    )
    assert len(plan.jobs) == 3
    names = [job.job_name for job in plan.jobs]
    assert len(set(names)) == 3
    assert plan.quota_instance_count == 1
    expected_ceiling = DEFAULT_EMERGENCY_PROFILE.max_run_usd * 3
    assert plan.total_ceiling_usd == pytest.approx(expected_ceiling)
    # Per-run ceiling from $1.408/hr * 15/60h = $0.352 → ceil cents = $0.36
    assert DEFAULT_EMERGENCY_PROFILE.max_run_usd == pytest.approx(0.36)
    payload = plan_to_dict(plan)
    assert payload["dry_run"] is True


def test_execute_requires_confirmation(valid_evidence: EmergencyEvidence, tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path, valid_evidence)

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
    assert plan.dry_run is False
    assert [job.arm for job in plan.jobs] == list(REQUIRED_ARMS)
