"""Serial launcher / dry-run planner for three emergency arms (quota=1)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.backends.safety import parse_digest_pinned_ecr_image
from distillery.contracts.manifest import SealedRunManifest
from experiments.aws_smoke.manifests import job_name_for_arm
from experiments.aws_smoke.pins import EmergencyEvidence
from experiments.aws_smoke.profile import (
    DEFAULT_EMERGENCY_PROFILE,
    REQUIRED_ARMS,
    EmergencyTrainingProfile,
    RunArm,
)
from experiments.aws_smoke.safety import (
    CallerIdentity,
    enforce_safety_gates,
)

ENTRYPOINT = ["python", "-m", "experiments.aws_smoke.train"]


@dataclass(frozen=True, slots=True)
class PlannedJob:
    arm: RunArm
    run_id: str
    job_name: str
    manifest_sha256: str
    output_prefix: str
    max_runtime_seconds: int
    max_run_usd: float
    hourly_usd: float
    create_training_job_request: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SerialLaunchPlan:
    dry_run: bool
    profile: str
    region: str
    instance_type: str
    quota_instance_count: int
    jobs: tuple[PlannedJob, ...]
    total_ceiling_usd: float
    identity: CallerIdentity | None


def load_manifest(path: Path) -> SealedRunManifest:
    return SealedRunManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def build_create_training_job_request(
    *,
    manifest: SealedRunManifest,
    evidence: EmergencyEvidence,
    arm: RunArm,
    profile: EmergencyTrainingProfile | None = None,
) -> dict[str, Any]:
    """Pure request builder. Does not call AWS."""
    p = profile or DEFAULT_EMERGENCY_PROFILE
    if manifest.runtime.instance_type != p.instance_type:
        raise ValueError(
            f"instance_type must be {p.instance_type}, got {manifest.runtime.instance_type}"
        )
    if p.quota_instance_count != 1:
        raise ValueError("emergency path requires quota_instance_count == 1")
    if p.max_runtime_seconds > 15 * 60:
        raise ValueError("MaxRuntime must be <= 15 minutes for emergency path")
    identity = parse_digest_pinned_ecr_image(evidence.ecr_image_uri)
    if identity.digest != manifest.runtime.image_digest:
        raise ValueError("manifest image_digest does not match evidence ecr image")

    job_name = job_name_for_arm(arm, manifest_sha256=manifest.seal_sha256())
    output_prefix = manifest.output.prefix.rstrip("/") + "/"
    hyperparams = {
        "manifest_sha256": manifest.seal_sha256(),
        "run_id": manifest.run_id,
        "arm": arm,
        "max_run_usd": str(manifest.cost.max_run_usd),
        "hourly_usd": str(p.hourly_usd),
        "emergency_profile": p.name,
    }
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": evidence.ecr_image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": list(ENTRYPOINT),
            "ContainerArguments": [
                "--manifest",
                "/opt/ml/input/data/manifest/manifest.json",
                "--arm",
                arm,
            ],
        },
        "RoleArn": evidence.iam_role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "manifest",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"{output_prefix}manifest/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/json",
                "InputMode": "File",
            },
            {
                "ChannelName": "dataset",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": manifest.dataset.uri.rstrip("/") + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            },
            {
                "ChannelName": "models",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": evidence.artifact_s3_prefix.rstrip("/") + "/models/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"{output_prefix}sagemaker-output/",
        },
        "ResourceConfig": {
            "InstanceType": p.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": p.max_runtime_seconds,
        },
        "HyperParameters": hyperparams,
        "EnableNetworkIsolation": False,
        "EnableManagedSpotTraining": False,
        "Tags": [
            {"Key": "Project", "Value": "RampHackathon"},
            {"Key": "Owner", "Value": "Dhilan"},
            {"Key": "TTL", "Value": "2026-07-20"},
            {"Key": "RunId", "Value": manifest.run_id},
            {"Key": "Arm", "Value": arm},
            {"Key": "Recipe", "Value": manifest.recipe.resolved},
            {"Key": "ManifestSha256", "Value": manifest.seal_sha256()},
            {"Key": "Backend", "Value": "sagemaker"},
            {"Key": "EmergencyProfile", "Value": p.name},
        ],
    }


def plan_serial_launch(
    *,
    manifest_paths: Mapping[RunArm, Path],
    evidence: EmergencyEvidence,
    profile_name: str,
    confirm: str | None,
    dry_run: bool = True,
    identity_provider: Any | None = None,
    training_profile: EmergencyTrainingProfile | None = None,
    arms: Sequence[RunArm] = REQUIRED_ARMS,
) -> SerialLaunchPlan:
    """
    Build a serial (quota=1) launch plan for the three required arms.

    Default is dry-run. Mutation requires safety gates + confirmation phrase.
    """
    p = training_profile or DEFAULT_EMERGENCY_PROFILE
    identity = enforce_safety_gates(
        profile=profile_name,
        confirm=confirm,
        evidence=evidence,
        identity_provider=identity_provider,
        dry_run=dry_run,
    )
    jobs: list[PlannedJob] = []
    seen_names: set[str] = set()
    for arm in arms:
        path = manifest_paths[arm]
        manifest = load_manifest(path)
        request = build_create_training_job_request(
            manifest=manifest,
            evidence=evidence,
            arm=arm,
            profile=p,
        )
        job_name = str(request["TrainingJobName"])
        if job_name in seen_names:
            raise ValueError(f"duplicate TrainingJobName across arms: {job_name}")
        seen_names.add(job_name)
        jobs.append(
            PlannedJob(
                arm=arm,
                run_id=manifest.run_id,
                job_name=job_name,
                manifest_sha256=manifest.seal_sha256(),
                output_prefix=manifest.output.prefix,
                max_runtime_seconds=p.max_runtime_seconds,
                max_run_usd=manifest.cost.max_run_usd,
                hourly_usd=p.hourly_usd,
                create_training_job_request=request,
            )
        )
    return SerialLaunchPlan(
        dry_run=dry_run,
        profile=profile_name,
        region=evidence.aws_region,
        instance_type=p.instance_type,
        quota_instance_count=p.quota_instance_count,
        jobs=tuple(jobs),
        total_ceiling_usd=sum(job.max_run_usd for job in jobs),
        identity=identity,
    )


def plan_to_dict(plan: SerialLaunchPlan) -> dict[str, Any]:
    return {
        "dry_run": plan.dry_run,
        "profile": plan.profile,
        "region": plan.region,
        "instance_type": plan.instance_type,
        "quota_instance_count": plan.quota_instance_count,
        "total_ceiling_usd": plan.total_ceiling_usd,
        "planned_at": datetime.now(tz=UTC).isoformat(),
        "identity_arn": plan.identity.arn if plan.identity is not None else None,
        "jobs": [
            {
                "arm": job.arm,
                "run_id": job.run_id,
                "job_name": job.job_name,
                "manifest_sha256": job.manifest_sha256,
                "output_prefix": job.output_prefix,
                "max_runtime_seconds": job.max_runtime_seconds,
                "max_run_usd": job.max_run_usd,
                "hourly_usd": job.hourly_usd,
                "create_training_job_request": job.create_training_job_request,
            }
            for job in plan.jobs
        ],
        "serial_note": (
            "Submit jobs one at a time; wait for terminal state before the next "
            "(quota InstanceCount=1 on ml.g5.xlarge)."
        ),
    }
