"""Serial launcher plan with canonical manifest channels and isolated networking."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from distillery.contracts.manifest import SealedRunManifest
from distillery.training.entrypoint import EXECUTE_ACKNOWLEDGEMENT
from experiments.aws_smoke.channels import (
    CANONICAL_MANIFEST_FILENAME,
    load_manifest,
)
from experiments.aws_smoke.manifests import (
    job_name_for_arm,
    manifest_arm,
    manifest_emergency_config,
    manifest_objective,
)
from experiments.aws_smoke.pins import EmergencyEvidence, parse_digest_pinned_ecr_image
from experiments.aws_smoke.profile import (
    DEFAULT_EMERGENCY_PROFILE,
    EmergencyTrainingProfile,
    RunArm,
    default_launch_order,
)
from experiments.aws_smoke.safety import CallerIdentity, enforce_safety_gates

ENTRYPOINT = ["python", "/opt/distillery/container_entrypoint.py"]
CONTAINER_MANIFEST_PATH = (
    f"/opt/ml/input/data/manifest/{CANONICAL_MANIFEST_FILENAME}"
)
CONTAINER_RESPONSES_PATH = "/opt/ml/input/data/responses/responses.jsonl"
CONTAINER_OUTPUT_DIR = "/opt/ml/output/data"
CONTAINER_MODEL_OUTPUT_DIR = "/opt/ml/model"


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
    scientific_role: str
    distinct_training_signal: bool
    equivalent_to: str | None
    local_manifest_path: Path
    manifest_object_uri: str
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
    distinct_signal_count: int
    identity: CallerIdentity | None


class S3UploadClient(Protocol):
    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, str] | None = None,
    ) -> Any: ...


def discover_generated_manifest_paths(
    manifests_dir: Path,
) -> dict[RunArm, Path]:
    """Read canonical paths from the generated campaign index."""
    index_path = manifests_dir / "campaign_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, dict):
        raise ValueError("campaign index missing arms map")
    result: dict[RunArm, Path] = {}
    for arm, metadata in raw_arms.items():
        if arm not in {"oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"}:
            raise ValueError(f"campaign index contains unknown arm {arm!r}")
        if not isinstance(metadata, dict):
            raise ValueError(f"campaign index arm {arm} metadata must be an object")
        relative = metadata.get("manifest")
        if not isinstance(relative, str):
            raise ValueError(f"campaign index arm {arm} lacks manifest path")
        path = manifests_dir / relative
        if not path.resolve().is_relative_to(manifests_dir.resolve()):
            raise ValueError(f"campaign index arm {arm} manifest escapes campaign dir")
        if path.name != CANONICAL_MANIFEST_FILENAME:
            raise ValueError(f"campaign index arm {arm} uses noncanonical manifest name")
        if not path.is_file():
            raise FileNotFoundError(f"campaign index manifest missing: {path}")
        result[arm] = path  # type: ignore[index]
    return result


def manifest_object_uri(request: Mapping[str, Any]) -> str:
    channels = {
        str(channel["ChannelName"]): channel
        for channel in request["InputDataConfig"]
    }
    prefix = channels["manifest"]["DataSource"]["S3DataSource"]["S3Uri"]
    return str(prefix).rstrip("/") + f"/{CANONICAL_MANIFEST_FILENAME}"


def stage_manifest_for_job(
    client: S3UploadClient,
    *,
    local_manifest_path: Path,
    request: Mapping[str, Any],
) -> str:
    """Upload exactly canonical manifest.json to the request's channel prefix."""
    manifest = load_manifest(local_manifest_path)
    expected_sha256 = str(request["HyperParameters"]["manifest_sha256"])
    if manifest.seal_sha256() != expected_sha256:
        raise ValueError("local manifest seal differs from CreateTrainingJob request")
    target = manifest_object_uri(request)
    expected_target = (
        manifest.output.prefix.rstrip("/")
        + f"/manifest/{CANONICAL_MANIFEST_FILENAME}"
    )
    if target != expected_target:
        raise ValueError("manifest staging target differs from sealed output prefix")
    parsed = urlparse(target)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid manifest staging S3 URI: {target}")
    client.upload_file(
        str(local_manifest_path),
        parsed.netloc,
        parsed.path.lstrip("/"),
        ExtraArgs={"ContentType": "application/json"},
    )
    return target


def build_create_training_job_request(
    *,
    manifest: SealedRunManifest,
    evidence: EmergencyEvidence,
    arm: RunArm,
    profile: EmergencyTrainingProfile | None = None,
) -> dict[str, Any]:
    """Build one finite, network-isolated request without calling AWS."""
    p = profile or DEFAULT_EMERGENCY_PROFILE
    if manifest.runtime.instance_type != p.instance_type:
        raise ValueError(
            f"instance_type must be {p.instance_type}, got {manifest.runtime.instance_type}"
        )
    if p.quota_instance_count != 1:
        raise ValueError("emergency path requires quota_instance_count == 1")
    if p.max_runtime_seconds > 15 * 60:
        raise ValueError("MaxRuntime must be <= 15 minutes for emergency path")
    if manifest.tags.get("EnableNetworkIsolation") != "true":
        raise ValueError("manifest must seal EnableNetworkIsolation=true")
    try:
        tagged_runtime_seconds = int(manifest.tags["MaxRuntimeInSeconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest must seal an integer MaxRuntimeInSeconds tag") from exc
    emergency_config = manifest_emergency_config(manifest)
    configured_runtime_seconds = emergency_config.get("max_runtime_seconds")
    if (
        tagged_runtime_seconds != p.max_runtime_seconds
        or configured_runtime_seconds != p.max_runtime_seconds
    ):
        raise ValueError(
            "SageMaker runtime, sealed MaxRuntimeInSeconds tag, and "
            "EmergencyConfig max_runtime_seconds must be exactly equal"
        )
    identity = parse_digest_pinned_ecr_image(evidence.ecr_image_uri)
    if identity.digest != manifest.runtime.image_digest:
        raise ValueError("manifest image_digest does not match evidence ECR image")
    if manifest.dataset.sha256 != evidence.data_content_sha256:
        raise ValueError("manifest dataset hash does not match operator evidence")
    if manifest_arm(manifest) != arm:
        raise ValueError("request arm does not match sealed manifest arm")

    objective = manifest_objective(manifest)
    job_name = job_name_for_arm(arm, manifest_sha256=manifest.seal_sha256())
    output_prefix = manifest.output.prefix.rstrip("/") + "/"
    hyperparams = {
        "manifest_sha256": manifest.seal_sha256(),
        "run_id": manifest.run_id,
        "arm": arm,
        "max_run_usd": str(manifest.cost.max_run_usd),
        "hourly_usd": str(p.hourly_usd),
        "max_runtime_seconds": str(tagged_runtime_seconds),
        "emergency_profile": p.name,
        "initialization_fingerprint": str(
            manifest.tags["InitializationFingerprint"]
        ),
    }
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": evidence.ecr_image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": list(ENTRYPOINT),
            "ContainerArguments": [
                "--execution-mode",
                "emergency-smoke",
                "--manifest",
                CONTAINER_MANIFEST_PATH,
                "--responses",
                CONTAINER_RESPONSES_PATH,
                "--output-dir",
                CONTAINER_OUTPUT_DIR,
                "--model-output-dir",
                CONTAINER_MODEL_OUTPUT_DIR,
                "--arm",
                arm,
                "--execute",
                "--execute-acknowledgement",
                EXECUTE_ACKNOWLEDGEMENT,
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
                "ChannelName": "responses",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": manifest.dataset.uri.rstrip("/") + "/responses/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/jsonlines",
                "InputMode": "File",
            },
            {
                "ChannelName": "models",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": evidence.models_s3_uri.rstrip("/") + "/",
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
            "MaxRuntimeInSeconds": tagged_runtime_seconds,
        },
        "HyperParameters": hyperparams,
        "EnableNetworkIsolation": True,
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
            {"Key": "MaxRuntimeInSeconds", "Value": str(tagged_runtime_seconds)},
            {"Key": "ScientificRole", "Value": str(objective["scientific_role"])},
            {
                "Key": "DistinctTrainingSignal",
                "Value": str(objective["distinct_training_signal"]).lower(),
            },
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
    arms: Sequence[RunArm] | None = None,
    require_three_distinct: bool = True,
) -> SerialLaunchPlan:
    """Plan one-at-a-time jobs; default order requires three distinct signals."""
    p = training_profile or DEFAULT_EMERGENCY_PROFILE
    launch_arms = (
        tuple(arms)
        if arms is not None
        else default_launch_order(
            set(manifest_paths),
            require_three_distinct=require_three_distinct,
        )
    )
    identity = enforce_safety_gates(
        profile=profile_name,
        confirm=confirm,
        evidence=evidence,
        identity_provider=identity_provider,
        dry_run=dry_run,
    )
    jobs: list[PlannedJob] = []
    seen_names: set[str] = set()
    for arm in launch_arms:
        try:
            path = manifest_paths[arm]
        except KeyError:
            raise ValueError(f"launch arm {arm} lacks a generated manifest") from None
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
        objective = manifest_objective(manifest)
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
                scientific_role=str(objective["scientific_role"]),
                distinct_training_signal=bool(objective["distinct_training_signal"]),
                equivalent_to=objective["equivalent_to"],
                local_manifest_path=path,
                manifest_object_uri=manifest_object_uri(request),
                create_training_job_request=request,
            )
        )
    distinct_count = sum(job.distinct_training_signal for job in jobs)
    if require_three_distinct and distinct_count < 3:
        raise ValueError(
            f"default launch requires 3 distinct signals, planned only {distinct_count}; "
            "ce_ablation is an oracle_sft-equivalent control"
        )
    return SerialLaunchPlan(
        dry_run=dry_run,
        profile=profile_name,
        region=evidence.aws_region,
        instance_type=p.instance_type,
        quota_instance_count=p.quota_instance_count,
        jobs=tuple(jobs),
        total_ceiling_usd=sum(job.max_run_usd for job in jobs),
        distinct_signal_count=distinct_count,
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
        "distinct_signal_count": plan.distinct_signal_count,
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
                "scientific_role": job.scientific_role,
                "distinct_training_signal": job.distinct_training_signal,
                "equivalent_to": job.equivalent_to,
                "local_manifest_path": str(job.local_manifest_path),
                "manifest_object_uri": job.manifest_object_uri,
                "create_training_job_request": job.create_training_job_request,
            }
            for job in plan.jobs
        ],
        "serial_note": (
            "Submit one job at a time and wait for terminal state because quota is "
            "one ml.g5.xlarge instance."
        ),
        "control_disclosure": (
            "ce_ablation is identical to oracle_sft when both use oracle hard "
            "targets; it is a replication/control, not a third distinct method."
        ),
    }
