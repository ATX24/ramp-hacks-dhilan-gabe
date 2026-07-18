"""Gate-authorized SageMaker launch and verified stop paths for Qwen72B."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.qwen72b_fallback.bindings import TRAINING_ROLE_ARN
from experiments.qwen72b_fallback.deadline import phase_budget_for
from experiments.qwen72b_fallback.evidence import sha256_bytes
from experiments.qwen72b_fallback.finance_world_targets import write_corpus_channel
from experiments.qwen72b_fallback.pins import (
    DISTILLERY_BUCKET,
    REVISION,
)
from experiments.qwen72b_fallback.profile import Qwen72BTrainingProfile
from experiments.qwen72b_fallback.readiness import (
    ExecutionAction,
    ExecutionAuthorization,
)

TERMINAL_TRAINING_STATUSES = frozenset({"Completed", "Failed", "Stopped"})
MODEL_S3_URI = f"s3://{DISTILLERY_BUCKET}/models/Qwen/Qwen2.5-72B-Instruct/{REVISION}/"
INPUT_PREFIX = "qwen72b/inputs"
OUTPUT_PREFIX = f"s3://{DISTILLERY_BUCKET}/qwen72b/runs"


@dataclass(frozen=True, slots=True)
class TrainingLaunchResult:
    job_name: str
    creation_time: str
    request_sha256: str
    uploaded_body_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class TrainingStopResult:
    job_name: str
    final_status: str
    stop_verified: bool


def _action_for_profile(profile: Qwen72BTrainingProfile) -> ExecutionAction:
    return {
        "rehearsal": ExecutionAction.REHEARSAL,
        "full": ExecutionAction.FULL,
    }[profile.kind.value]


def _put_and_verify(
    s3: Any,
    *,
    key: str,
    body: bytes,
    content_type: str,
) -> str:
    expected = sha256_bytes(body)
    s3.put_object(
        Bucket=DISTILLERY_BUCKET,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata={"sha256": expected},
    )
    response = s3.get_object(
        Bucket=DISTILLERY_BUCKET,
        Key=key,
        ChecksumMode="ENABLED",
    )
    actual_body = response["Body"].read()
    if actual_body != body or sha256_bytes(actual_body) != expected:
        raise RuntimeError(f"uploaded control-channel body mismatch: {key}")
    return expected


def upload_control_channels(
    *,
    s3: Any,
    authorization: ExecutionAuthorization,
    profile: Qwen72BTrainingProfile,
) -> tuple[str, dict[str, str]]:
    finance = authorization.evidence_bundle.finance_world_data
    if finance is None:
        raise ValueError("training authorization lacks finance-world evidence")
    prefix = f"{INPUT_PREFIX}/{authorization.launch_name}"
    bodies: dict[str, bytes] = {
        "control/profile.json": (
            json.dumps(profile.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode(),
        "control/authorization.json": (
            json.dumps(
                authorization.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    }
    with tempfile.TemporaryDirectory(prefix="qwen72b-data-") as temporary:
        root = Path(temporary)
        write_corpus_channel(finance, root)
        bodies["data/train.jsonl"] = (root / "train.jsonl").read_bytes()
        bodies["data/finance_world_evidence.json"] = (
            root / "finance_world_evidence.json"
        ).read_bytes()
    if authorization.evidence_bundle.memory_probe is not None:
        bodies["control/memory-probe.json"] = (
            json.dumps(
                authorization.evidence_bundle.memory_probe.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    hashes = {
        relative: _put_and_verify(
            s3,
            key=f"{prefix}/{relative}",
            body=body,
            content_type=(
                "application/jsonl" if relative.endswith(".jsonl") else "application/json"
            ),
        )
        for relative, body in sorted(bodies.items())
    }
    return prefix, hashes


def build_training_request(
    *,
    authorization: ExecutionAuthorization,
    profile: Qwen72BTrainingProfile,
    input_prefix: str,
    mode: str | None = None,
) -> dict[str, Any]:
    action = (
        ExecutionAction.MEMORY_PROBE if mode == "memory_probe" else _action_for_profile(profile)
    )
    authorization.require_current(
        action=action,
        launch_name=authorization.launch_name,
    )
    if authorization.evidence_bundle.target_profile_sha256 != profile.profile_sha256:
        raise ValueError("launch profile differs from execution authorization")
    image = authorization.evidence_bundle.ecr_image
    if image is None:
        raise ValueError("training authorization lacks exact ECR image evidence")
    effective_mode = mode or profile.kind.value
    arguments = [
        "--execution-mode",
        "qwen72b",
        "--qwen72b-mode",
        effective_mode,
        "--launch-name",
        authorization.launch_name,
        "--profile",
        "/opt/ml/input/data/control/profile.json",
        "--authorization",
        "/opt/ml/input/data/control/authorization.json",
        "--models-dir",
        "/opt/ml/input/data/models",
        "--data-dir",
        "/opt/ml/input/data/data",
        "--output-dir",
        "/opt/ml/model",
        "--runtime-image-digest",
        image.image_digest,
        "--execute",
    ]
    if authorization.evidence_bundle.memory_probe is not None:
        arguments.extend(
            [
                "--memory-probe",
                "/opt/ml/input/data/control/memory-probe.json",
            ]
        )
    return {
        "TrainingJobName": authorization.launch_name,
        "AlgorithmSpecification": {
            "TrainingImage": image.image_uri,
            "TrainingInputMode": "File",
            "ContainerArguments": arguments,
        },
        "RoleArn": TRAINING_ROLE_ARN,
        "InputDataConfig": [
            {
                "ChannelName": "models",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": MODEL_S3_URI,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "control",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{DISTILLERY_BUCKET}/{input_prefix}/control/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "data",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{DISTILLERY_BUCKET}/{input_prefix}/data/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "OutputDataConfig": {"S3OutputPath": f"{OUTPUT_PREFIX}/{authorization.launch_name}/"},
        "ResourceConfig": {
            "InstanceType": "ml.p4de.24xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 500,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": (
                phase_budget_for("memory_probe").max_runtime_seconds
                if effective_mode == "memory_probe"
                else profile.max_runtime_seconds
            ),
        },
        "EnableNetworkIsolation": True,
        "EnableManagedSpotTraining": False,
        "RetryStrategy": {"MaximumRetryAttempts": 0},
        "Tags": [
            {"Key": "DistilleryWorkstream", "Value": "qwen72b-fallback"},
            {"Key": "RunMode", "Value": effective_mode},
            {
                "Key": "AuthorizationSha256",
                "Value": authorization.evidence_sha256,
            },
            {"Key": "ProfileSha256", "Value": profile.profile_sha256},
            {"Key": "MaxCostUSD", "Value": f"{profile.hard_cap_usd:.2f}"},
        ],
    }


def _job_exists(sagemaker: Any, job_name: str) -> bool:
    try:
        sagemaker.describe_training_job(TrainingJobName=job_name)
    except Exception as exc:  # noqa: BLE001 - normalize boto and injected clients
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in {"ValidationException", "ResourceNotFound", "ResourceNotFoundException"}:
            return False
        raise
    return True


def _verify_created_job(
    sagemaker: Any,
    *,
    request: dict[str, Any],
    authorization: ExecutionAuthorization,
    effective_mode: str,
) -> dict[str, Any]:
    described = sagemaker.describe_training_job(TrainingJobName=authorization.launch_name)
    expected_fields = {
        "TrainingJobName": request["TrainingJobName"],
        "AlgorithmSpecification": request["AlgorithmSpecification"],
        "RoleArn": request["RoleArn"],
        "InputDataConfig": request["InputDataConfig"],
        "OutputDataConfig": request["OutputDataConfig"],
        "ResourceConfig": request["ResourceConfig"],
        "StoppingCondition": request["StoppingCondition"],
        "EnableNetworkIsolation": True,
        "EnableManagedSpotTraining": False,
        "RetryStrategy": {"MaximumRetryAttempts": 0},
    }
    mismatches = [
        name for name, expected in expected_fields.items() if described.get(name) != expected
    ]
    tags = sagemaker.list_tags(ResourceArn=described["TrainingJobArn"]).get(
        "Tags",
        [],
    )
    tag_map = {str(tag["Key"]): str(tag["Value"]) for tag in tags}
    expected_tags = {
        "DistilleryWorkstream": "qwen72b-fallback",
        "RunMode": effective_mode,
        "AuthorizationSha256": authorization.evidence_sha256,
    }
    if any(tag_map.get(key) != value for key, value in expected_tags.items()):
        mismatches.append("Tags")
    if mismatches:
        raise RuntimeError(
            f"created SageMaker job differs from sealed request: {sorted(mismatches)}"
        )
    return described


def launch_training_job(
    *,
    sagemaker: Any,
    s3: Any,
    authorization: ExecutionAuthorization,
    profile: Qwen72BTrainingProfile,
    mode: str | None = None,
) -> TrainingLaunchResult:
    action = (
        ExecutionAction.MEMORY_PROBE if mode == "memory_probe" else _action_for_profile(profile)
    )
    authorization.require_current(
        action=action,
        launch_name=authorization.launch_name,
    )
    if authorization.evidence_bundle.target_profile_sha256 != profile.profile_sha256:
        raise ValueError("launch profile differs from execution authorization")
    if _job_exists(sagemaker, authorization.launch_name):
        raise RuntimeError(f"duplicate SageMaker launch: {authorization.launch_name}")
    input_prefix, uploaded = upload_control_channels(
        s3=s3,
        authorization=authorization,
        profile=profile,
    )
    request = build_training_request(
        authorization=authorization,
        profile=profile,
        input_prefix=input_prefix,
        mode=mode,
    )
    try:
        sagemaker.create_training_job(**request)
    except BaseException:
        if _job_exists(sagemaker, authorization.launch_name):
            stop_training_job_and_verify(
                sagemaker=sagemaker,
                job_name=authorization.launch_name,
                typed_confirmation=(f"STOP QWEN72B TRAINING {authorization.launch_name}"),
            )
        raise
    effective_mode = mode or profile.kind.value
    try:
        described = _verify_created_job(
            sagemaker,
            request=request,
            authorization=authorization,
            effective_mode=effective_mode,
        )
    except BaseException:
        if _job_exists(sagemaker, authorization.launch_name):
            stop_training_job_and_verify(
                sagemaker=sagemaker,
                job_name=authorization.launch_name,
                typed_confirmation=f"STOP QWEN72B TRAINING {authorization.launch_name}",
            )
        raise
    creation = described.get("CreationTime")
    return TrainingLaunchResult(
        job_name=authorization.launch_name,
        creation_time=str(creation),
        request_sha256=content_sha256_for_request(request),
        uploaded_body_sha256=uploaded,
    )


def content_sha256_for_request(request: dict[str, Any]) -> str:
    from distillery.contracts.hashing import content_sha256

    return content_sha256(request)


def stop_training_job_and_verify(
    *,
    sagemaker: Any,
    job_name: str,
    typed_confirmation: str,
    sleep: Any = time.sleep,
    attempts: int = 90,
) -> TrainingStopResult:
    expected = f"STOP QWEN72B TRAINING {job_name}"
    if typed_confirmation != expected:
        raise ValueError(f"typed confirmation must equal: {expected}")
    job = sagemaker.describe_training_job(TrainingJobName=job_name)
    tags = sagemaker.list_tags(ResourceArn=job["TrainingJobArn"]).get("Tags", [])
    tag_map = {str(tag["Key"]): str(tag["Value"]) for tag in tags}
    if tag_map.get("DistilleryWorkstream") != "qwen72b-fallback":
        raise RuntimeError("refusing to stop a non-Qwen72B training job")
    status = str(job.get("TrainingJobStatus"))
    if status not in TERMINAL_TRAINING_STATUSES and status != "Stopping":
        sagemaker.stop_training_job(TrainingJobName=job_name)
    for _attempt in range(attempts):
        status = str(
            sagemaker.describe_training_job(TrainingJobName=job_name).get("TrainingJobStatus")
        )
        if status in TERMINAL_TRAINING_STATUSES:
            return TrainingStopResult(
                job_name=job_name,
                final_status=status,
                stop_verified=True,
            )
        sleep(10)
    raise RuntimeError(f"SageMaker stop was not verified: {job_name}")
