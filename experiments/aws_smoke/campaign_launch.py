"""Plan-only SageMaker requests for sealed 4/8-GPU campaigns and waves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.aws_smoke.campaign_index import (
    MICRO_USD_PER_USD,
    VerifiedCampaignBundle,
    verify_campaign_bundle,
)
from experiments.aws_smoke.campaign_wave import verify_wave_bundle
from experiments.aws_smoke.manifests import manifest_emergency_config
from experiments.aws_smoke.pins import EmergencyEvidence, parse_digest_pinned_ecr_image

CAMPAIGN_CHANNEL_NAME = "campaign"
CONTAINER_CAMPAIGN_ROOT = "/opt/ml/input/data/campaign"
CONTAINER_DATASET_ROOT = "/opt/ml/input/data/dataset"
CONTAINER_MODELS_ROOT = "/opt/ml/input/data/models"
CONTAINER_OUTPUT_ROOT = "/opt/ml/output/data"
CONTAINER_MODEL_ROOT = "/opt/ml/model"
CONTAINER_RUNTIME_ROOT = "/tmp/distillery-campaign"
CONTAINER_PYTHON = "/opt/conda/bin/python"
CAMPAIGN_ENTRYPOINT = [
    CONTAINER_PYTHON,
    "-m",
    "experiments.aws_smoke.campaign_orchestrator",
]


@dataclass(frozen=True, slots=True)
class CampaignLaunchPlan:
    dry_run: bool
    campaign_id: str
    campaign_index_sha256: str
    instance_type: str
    instance_count: int
    gpu_count: int
    hourly_price_microusd: int
    max_parent_cost_microusd: int
    create_training_job_request: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WaveLaunchPlan:
    dry_run: bool
    wave_id: str
    wave_index_sha256: str
    jobs: tuple[CampaignLaunchPlan, CampaignLaunchPlan]
    aggregate_max_parent_cost_microusd: int


def _ceiling_parent_cost_microusd(
    *,
    hourly_price_microusd: int,
    max_runtime_seconds: int,
) -> int:
    numerator = hourly_price_microusd * max_runtime_seconds
    return (numerator + 3600 - 1) // 3600


def _validate_launch_evidence(
    bundle: VerifiedCampaignBundle,
    evidence: EmergencyEvidence,
) -> None:
    index = bundle.index
    identity = parse_digest_pinned_ecr_image(evidence.ecr_image_uri)
    if identity.digest != evidence.image_digest:
        raise ValueError("ECR image URI does not match attested image digest")
    if identity.region != index.pricing.region or evidence.aws_region != index.pricing.region:
        raise ValueError("campaign pricing region and AWS evidence region must match")
    if identity.account_id != evidence.aws_account_id:
        raise ValueError("ECR image account does not match AWS evidence account")
    if any(manifest.runtime.image_digest != evidence.image_digest for manifest in bundle.manifests):
        raise ValueError("campaign manifest image digest does not match AWS evidence")
    if any(
        manifest.dataset.uri.rstrip("/") != evidence.dataset_s3_uri.rstrip("/")
        for manifest in bundle.manifests
    ):
        raise ValueError("campaign dataset channel does not match AWS evidence")
    for manifest in bundle.manifests:
        try:
            tagged_runtime = int(manifest.tags["MaxRuntimeInSeconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("campaign manifest has invalid MaxRuntimeInSeconds tag") from exc
        configured_runtime = manifest_emergency_config(manifest).get(
            "max_runtime_seconds"
        )
        if (
            tagged_runtime != index.max_runtime_seconds
            or configured_runtime != index.max_runtime_seconds
        ):
            raise ValueError(
                "campaign stopping condition, manifest runtime tag, and "
                "EmergencyConfig runtime must be exactly equal"
            )


def build_campaign_launch_plan(
    *,
    campaign_root: Path,
    evidence: EmergencyEvidence,
    expected_index_sha256: str | None = None,
    volume_size_gb: int = 60,
) -> CampaignLaunchPlan:
    """Build, but never submit, one network-isolated CreateTrainingJob request."""
    if volume_size_gb < 30:
        raise ValueError("campaign volume_size_gb must be at least 30")
    bundle = verify_campaign_bundle(
        campaign_root,
        expected_index_sha256=expected_index_sha256,
    )
    _validate_launch_evidence(bundle, evidence)
    index = bundle.index
    index_sha256 = bundle.index_sha256
    max_parent_cost = _ceiling_parent_cost_microusd(
        hourly_price_microusd=index.pricing.hourly_price_microusd,
        max_runtime_seconds=index.max_runtime_seconds,
    )
    dataset_uri = bundle.manifests[0].dataset.uri.rstrip("/") + "/"
    model_uri = evidence.models_s3_uri.rstrip("/") + "/"
    job_name = f"distillery-{index.hardware.gpu_count}gpu-{index_sha256[:16]}"
    request = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": evidence.ecr_image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": list(CAMPAIGN_ENTRYPOINT),
            "ContainerArguments": [
                "--campaign-root",
                CONTAINER_CAMPAIGN_ROOT,
                "--expected-index-sha256",
                index_sha256,
                "--dataset-dir",
                CONTAINER_DATASET_ROOT,
                "--models-dir",
                CONTAINER_MODELS_ROOT,
                "--output-root",
                CONTAINER_OUTPUT_ROOT,
                "--model-root",
                CONTAINER_MODEL_ROOT,
                "--runtime-root",
                CONTAINER_RUNTIME_ROOT,
                "--python-executable",
                CONTAINER_PYTHON,
                "--timeout-seconds",
                str(index.max_runtime_seconds),
            ],
        },
        "RoleArn": evidence.iam_role_arn,
        "InputDataConfig": [
            {
                "ChannelName": CAMPAIGN_CHANNEL_NAME,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": index.input_s3_prefix,
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
                        "S3Uri": dataset_uri,
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
                        "S3Uri": model_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": index.campaign_output_prefix,
        },
        "ResourceConfig": {
            "InstanceType": index.hardware.instance_type,
            "InstanceCount": index.hardware.instance_count,
            "VolumeSizeInGB": volume_size_gb,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": index.max_runtime_seconds,
        },
        "HyperParameters": {
            "campaign_id": index.campaign_id,
            "campaign_index_sha256": index_sha256,
            "campaign_protocol_sha256": index.protocol_sha256,
            "pricing_evidence_sha256": index.pricing.evidence_sha256,
            "hourly_price_microusd": str(index.pricing.hourly_price_microusd),
            "max_parent_cost_microusd": str(max_parent_cost),
            "max_runtime_seconds": str(index.max_runtime_seconds),
            "execution_mode": index.hardware.execution_mode,
        },
        "EnableNetworkIsolation": True,
        "EnableManagedSpotTraining": False,
        "Tags": [
            {"Key": "Project", "Value": "RampHackathon"},
            {"Key": "Backend", "Value": "sagemaker"},
            {"Key": "CampaignId", "Value": index.campaign_id},
            {"Key": "CampaignIndexSha256", "Value": index_sha256},
            {"Key": "CampaignProtocolSha256", "Value": index.protocol_sha256},
            {"Key": "InstanceTopology", "Value": index.hardware.profile_id},
            {"Key": "ArmCount", "Value": str(len(index.arms))},
            {"Key": "ExecutionMode", "Value": index.hardware.execution_mode},
            {"Key": "MaxRuntimeInSeconds", "Value": str(index.max_runtime_seconds)},
        ],
    }
    if max_parent_cost <= 0 or max_parent_cost > 10_000 * MICRO_USD_PER_USD:
        raise ValueError("sealed campaign parent cost exceeds the account-wide ceiling")
    return CampaignLaunchPlan(
        dry_run=True,
        campaign_id=index.campaign_id,
        campaign_index_sha256=index_sha256,
        instance_type=index.hardware.instance_type,
        instance_count=index.hardware.instance_count,
        gpu_count=index.hardware.gpu_count,
        hourly_price_microusd=index.pricing.hourly_price_microusd,
        max_parent_cost_microusd=max_parent_cost,
        create_training_job_request=request,
    )


def build_wave_launch_plan(
    *,
    wave_root: Path,
    evidence: EmergencyEvidence,
    expected_index_sha256: str | None = None,
    volume_size_gb: int = 60,
) -> WaveLaunchPlan:
    """Build exactly two g5.48xlarge requests without submitting either job."""
    wave = verify_wave_bundle(
        wave_root,
        expected_index_sha256=expected_index_sha256,
    )
    jobs = tuple(
        build_campaign_launch_plan(
            campaign_root=campaign.root,
            evidence=evidence,
            expected_index_sha256=campaign.index_sha256,
            volume_size_gb=volume_size_gb,
        )
        for campaign in wave.campaigns
    )
    if len(jobs) != 2:
        raise RuntimeError("sealed two-node wave did not produce exactly two jobs")
    aggregate = sum(job.max_parent_cost_microusd for job in jobs)
    if aggregate > 10_000 * MICRO_USD_PER_USD:
        raise ValueError("wave maximum cost exceeds the account-wide ceiling")
    return WaveLaunchPlan(
        dry_run=True,
        wave_id=wave.index.wave_id,
        wave_index_sha256=wave.index_sha256,
        jobs=(jobs[0], jobs[1]),
        aggregate_max_parent_cost_microusd=aggregate,
    )
