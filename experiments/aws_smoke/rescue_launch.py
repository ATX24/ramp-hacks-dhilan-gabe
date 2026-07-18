"""Plan-only CreateTrainingJob builder for Script Mode oracle_sft rescue."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from distillery.contracts.manifest import SealedRunManifest
from experiments.aws_smoke.manifests import manifest_arm, manifest_emergency_config
from experiments.aws_smoke.pins import EmergencyEvidence, parse_digest_pinned_ecr_image
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE, EmergencyTrainingProfile

# Pinned official AWS HF PyTorch training DLC (us-east-1, digest-locked).
RESCUE_DLC_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
RESCUE_DLC_DIGEST = (
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
RESCUE_DLC_TAG = "2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04"
RESCUE_DLC_ECR_ARN = (
    "arn:aws:ecr:us-east-1:763104351884:repository/huggingface-pytorch-training"
)

CONTAINER_CODE_ROOT = "/opt/ml/input/data/code"
CONTAINER_MANIFEST_PATH = "/opt/ml/input/data/manifest/manifest.json"
CONTAINER_RESPONSES_PATH = "/opt/ml/input/data/responses/responses.jsonl"
CONTAINER_DATASET_ROOT = "/opt/ml/input/data/dataset"
CONTAINER_MODELS_ROOT = "/opt/ml/input/data/models"
CONTAINER_OUTPUT_DIR = "/opt/ml/output/data"
CONTAINER_MODEL_OUTPUT_DIR = "/opt/ml/model"


def rescue_job_name(*, manifest_sha256: str) -> str:
    base = f"rescue-oracle-sft-{manifest_sha256[:12]}"
    return base[:63].rstrip("-")


def build_rescue_create_training_job_request(
    *,
    manifest: SealedRunManifest,
    evidence: EmergencyEvidence,
    code_s3_uri: str,
    profile: EmergencyTrainingProfile | None = None,
) -> dict[str, Any]:
    """Build one finite, network-isolated Script Mode request without calling AWS."""
    p = profile or DEFAULT_EMERGENCY_PROFILE
    if manifest_arm(manifest) != "oracle_sft":
        raise ValueError("rescue path only submits oracle_sft")
    if manifest.runtime.instance_type != p.instance_type:
        raise ValueError(
            f"instance_type must be {p.instance_type}, got {manifest.runtime.instance_type}"
        )
    if p.max_runtime_seconds > 20 * 60:
        raise ValueError("MaxRuntime must be <= 20 minutes for rescue path")
    if p.max_run_usd > 50.0:
        raise ValueError("max gross cost must be <= $50 for rescue path")
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
    if not identity.is_aws_hf_training_dlc:
        raise ValueError("rescue Script Mode requires the pinned AWS HF training DLC")
    if identity.digest != RESCUE_DLC_DIGEST or evidence.image_digest != RESCUE_DLC_DIGEST:
        raise ValueError("rescue image digest does not match the sealed DLC pin")
    if evidence.ecr_image_uri != RESCUE_DLC_IMAGE_URI:
        raise ValueError("rescue ecr_image_uri does not match the sealed DLC pin")
    if manifest.runtime.image_digest != RESCUE_DLC_DIGEST:
        raise ValueError("manifest image_digest does not match the sealed DLC pin")
    if manifest.dataset.sha256 != evidence.data_content_sha256:
        raise ValueError("manifest dataset hash does not match operator evidence")
    parsed_code = urlparse(code_s3_uri)
    if parsed_code.scheme != "s3" or not parsed_code.netloc or not parsed_code.path.strip("/"):
        raise ValueError(f"invalid code S3 URI: {code_s3_uri}")

    job_name = rescue_job_name(manifest_sha256=manifest.seal_sha256())
    output_prefix = manifest.output.prefix.rstrip("/") + "/"
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": evidence.ecr_image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "python",
                f"{CONTAINER_CODE_ROOT}/rescue_entry.py",
            ],
            "ContainerArguments": [
                "--arm",
                "oracle_sft",
                "--manifest",
                CONTAINER_MANIFEST_PATH,
                "--dataset-dir",
                CONTAINER_DATASET_ROOT,
                "--models-dir",
                CONTAINER_MODELS_ROOT,
                "--responses",
                CONTAINER_RESPONSES_PATH,
                "--output-dir",
                CONTAINER_OUTPUT_DIR,
                "--model-output-dir",
                CONTAINER_MODEL_OUTPUT_DIR,
            ],
        },
        "RoleArn": evidence.iam_role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "code",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": code_s3_uri.rstrip("/") + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            },
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
        "HyperParameters": {
            "manifest_sha256": manifest.seal_sha256(),
            "run_id": manifest.run_id,
            "arm": "oracle_sft",
            "max_run_usd": str(manifest.cost.max_run_usd),
            "hourly_usd": str(p.hourly_usd),
            "max_runtime_seconds": str(tagged_runtime_seconds),
            "rescue_path": "script_mode_hf_dlc",
            "rescue_dlc_tag": RESCUE_DLC_TAG,
            "source_revision": evidence.source_revision,
        },
        "EnableNetworkIsolation": True,
        "EnableManagedSpotTraining": False,
        "Tags": [
            {"Key": "Project", "Value": "RampHackathon"},
            {"Key": "Owner", "Value": "Gabriel"},
            {"Key": "TTL", "Value": "2026-07-20"},
            {"Key": "RunId", "Value": manifest.run_id},
            {"Key": "Arm", "Value": "oracle_sft"},
            {"Key": "RescuePath", "Value": "script_mode_hf_dlc"},
            {"Key": "ManifestSha256", "Value": manifest.seal_sha256()},
            {"Key": "Backend", "Value": "sagemaker"},
            {"Key": "MaxRuntimeInSeconds", "Value": str(tagged_runtime_seconds)},
            {"Key": "ImageDigest", "Value": RESCUE_DLC_DIGEST},
        ],
    }


def list_active_distillery_jobs(client: Any) -> list[dict[str, str]]:
    """Return Starting/InProgress Distillery jobs (any naming convention)."""
    active: list[dict[str, str]] = []
    for status in ("InProgress", "Starting"):
        response = client.list_training_jobs(
            StatusEquals=status,
            MaxResults=100,
            SortBy="CreationTime",
            SortOrder="Descending",
        )
        for summary in response.get("TrainingJobSummaries", []):
            name = str(summary.get("TrainingJobName", ""))
            lowered = name.lower()
            if (
                lowered.startswith("rescue-")
                or lowered.startswith("aws-smoke-")
                or lowered.startswith("distillery-")
            ):
                active.append(
                    {
                        "name": name,
                        "status": str(summary.get("TrainingJobStatus", status)),
                        "creation_time": str(summary.get("CreationTime", "")),
                    }
                )
    return active
