#!/usr/bin/env python3
"""Launch a finite, tagged SageMaker Training Job for Transformers benchmarks.

Hard cost ceiling: $100. Guaranteed MaxRuntime. Tags every resource. Does not
touch the emergency training launch path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import boto3
from botocore.exceptions import ClientError

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

AWS_PROFILE = "gabriel-cli"
AWS_REGION = "us-east-1"
AWS_ACCOUNT = "225989358036"
ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT}:role/distillery-sagemaker-training"
BUCKET = f"distillery-{AWS_ACCOUNT}-{AWS_REGION}"
MODELS_PREFIX = f"s3://{BUCKET}/models/"
STUDENT_PREFIX = (
    f"s3://{BUCKET}/models/Qwen/Qwen2.5-0.5B-Instruct/"
    "7ae557604adf67be50417f59c2c2f167def9a775/"
)
TEACHER_PREFIX = (
    f"s3://{BUCKET}/models/Qwen/Qwen2.5-1.5B-Instruct/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306/"
)
HARD_CAP_USD = 100.0
HOURLY_USD = {
    "ml.g5.xlarge": 1.408,
    "ml.p4de.24xlarge": 31.5641075,
}
HARDWARE_LABEL = {
    "ml.g5.xlarge": "NVIDIA-A10G-24GB",
    "ml.p4de.24xlarge": "NVIDIA-A100-80GB-x8-use-gpu0",
}


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    instance_type: str
    image_uri: str
    max_runtime_seconds: int
    dtype: str
    warmups: int
    timed: int
    max_new_tokens: int
    batch_sizes: str
    dry_run: bool
    execute: bool
    cost_cap_usd: float


def estimate_cost_usd(instance_type: str, max_runtime_seconds: int) -> float:
    hourly = HOURLY_USD[instance_type]
    return hourly * (max_runtime_seconds / 3600.0)


def build_job_name(instance_type: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    short = instance_type.replace(".", "").replace("ml", "")
    return f"dist-bench-{short}-{stamp}"[:63]


def build_create_training_job_request(cfg: LaunchConfig, *, job_name: str) -> dict[str, Any]:
    if cfg.instance_type not in HOURLY_USD:
        raise ValueError(f"unsupported instance_type {cfg.instance_type}")
    if "@sha256:" not in cfg.image_uri:
        raise ValueError("image_uri must be digest-pinned (...@sha256:<hex>)")
    digest = cfg.image_uri.rsplit("@", 1)[-1]
    if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
        raise ValueError("image_uri digest must be sha256:<64 hex>")
    est = estimate_cost_usd(cfg.instance_type, cfg.max_runtime_seconds)
    if est > cfg.cost_cap_usd:
        raise ValueError(
            f"estimated max cost ${est:.2f} exceeds hard cap ${cfg.cost_cap_usd:.2f}"
        )
    if est > HARD_CAP_USD:
        raise ValueError(f"estimated max cost ${est:.2f} exceeds absolute hard cap $100")
    output_prefix = f"s3://{BUCKET}/benchmarks/{job_name}/sagemaker-output/"
    return {
        "TrainingJobName": job_name,
        "RoleArn": ROLE_ARN,
        "AlgorithmSpecification": {
            "TrainingImage": cfg.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "python",
                "/opt/distillery/experiments/benchmark/run.py",
            ],
            "ContainerArguments": [
                "--models-root",
                "/opt/ml/input/data/models",
                "--output-dir",
                "/opt/ml/output/data",
                "--dtype",
                cfg.dtype,
                "--max-new-tokens",
                str(cfg.max_new_tokens),
                "--warmups",
                str(cfg.warmups),
                "--timed",
                str(cfg.timed),
                "--batch-sizes",
                cfg.batch_sizes,
                "--hardware",
                HARDWARE_LABEL[cfg.instance_type],
                "--instance-type",
                cfg.instance_type,
            ],
        },
        "HyperParameters": {
            "benchmark": "transformers_throughput",
            "cost_cap_usd": str(cfg.cost_cap_usd),
            "estimated_max_usd": f"{est:.4f}",
        },
        "InputDataConfig": [
            {
                "ChannelName": "models",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": MODELS_PREFIX,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": output_prefix,
        },
        "ResourceConfig": {
            "InstanceType": cfg.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": cfg.max_runtime_seconds,
        },
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": True,
        "Environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DISTILLERY_BENCHMARK_INSTANCE_TYPE": cfg.instance_type,
            "DISTILLERY_BENCHMARK_HARDWARE": HARDWARE_LABEL[cfg.instance_type],
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "Tags": [
            {"Key": "Project", "Value": "RampHackathon"},
            {"Key": "Component", "Value": "distillery-benchmark"},
            {"Key": "Owner", "Value": "gabriel"},
            {"Key": "Purpose", "Value": "tokens-per-second"},
            {"Key": "Runtime", "Value": "transformers"},
            {"Key": "CostCapUSD", "Value": str(int(cfg.cost_cap_usd))},
            {"Key": "MaxRuntimeInSeconds", "Value": str(cfg.max_runtime_seconds)},
            {"Key": "TTL", "Value": "2026-07-20"},
            {"Key": "AutoCleanup", "Value": "true"},
        ],
    }


def inspect_quotas(session: boto3.session.Session, instance_type: str) -> dict[str, Any]:
    client = session.client("service-quotas")
    wanted = {
        f"{instance_type} for training job usage",
        f"{instance_type} for endpoint usage",
        f"{instance_type} for processing job usage",
    }
    found: dict[str, float] = {}
    paginator = client.get_paginator("list_service_quotas")
    for page in paginator.paginate(ServiceCode="sagemaker"):
        for quota in page.get("Quotas", []):
            name = quota["QuotaName"]
            if name in wanted or (
                instance_type in name and "training job usage" in name
            ):
                found[name] = float(quota["Value"])
    training_key = f"{instance_type} for training job usage"
    if found.get(training_key, 0.0) < 1.0:
        raise RuntimeError(
            f"insufficient training quota for {instance_type}: {found.get(training_key, 0.0)}"
        )
    return found


def stop_job(sm: Any, job_name: str) -> None:
    try:
        sm.stop_training_job(TrainingJobName=job_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {"ValidationException", "ResourceNotFound"}:
            raise


def wait_and_cleanup(sm: Any, job_name: str, *, poll_s: int = 30) -> dict[str, Any]:
    terminal = {
        "Completed",
        "Failed",
        "Stopped",
        "Stopping",
    }
    while True:
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        print(
            json.dumps(
                {
                    "job_name": job_name,
                    "status": status,
                    "secondary": desc.get("SecondaryStatus"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if status in terminal and status != "Stopping":
            # Training jobs are finite; no endpoint to delete. Ensure stopped if failed mid-flight.
            if status not in {"Completed", "Failed", "Stopped"}:
                stop_job(sm, job_name)
            return desc
        time.sleep(poll_s)


def billed_cost_usd(desc: dict[str, Any]) -> float | None:
    seconds = desc.get("BillableTimeInSeconds")
    instance = desc.get("ResourceConfig", {}).get("InstanceType")
    if seconds is None or instance not in HOURLY_USD:
        return None
    return HOURLY_USD[instance] * (float(seconds) / 3600.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmark-launch")
    parser.add_argument("--instance-type", default="ml.g5.xlarge", choices=sorted(HOURLY_USD))
    parser.add_argument("--image-uri", required=True, help="Digest-pinned ECR image URI")
    parser.add_argument("--max-runtime-seconds", type=int, default=5400)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--cost-cap-usd", type=float, default=HARD_CAP_USD)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--wait", action="store_true", default=False)
    parser.add_argument("--profile", default=AWS_PROFILE)
    parser.add_argument("--region", default=AWS_REGION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("choose exactly one of --dry-run or --execute")
    if not args.execute and not args.dry_run:
        args.dry_run = True
    cfg = LaunchConfig(
        instance_type=args.instance_type,
        image_uri=args.image_uri,
        max_runtime_seconds=args.max_runtime_seconds,
        dtype=args.dtype,
        warmups=args.warmups,
        timed=args.timed,
        max_new_tokens=args.max_new_tokens,
        batch_sizes=args.batch_sizes,
        dry_run=args.dry_run,
        execute=args.execute,
        cost_cap_usd=args.cost_cap_usd,
    )
    job_name = build_job_name(cfg.instance_type)
    request = build_create_training_job_request(cfg, job_name=job_name)
    est = estimate_cost_usd(cfg.instance_type, cfg.max_runtime_seconds)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    quotas = inspect_quotas(session, cfg.instance_type)
    plan = {
        "dry_run": cfg.dry_run,
        "execute": cfg.execute,
        "job_name": job_name,
        "estimated_max_usd": est,
        "hard_cap_usd": HARD_CAP_USD,
        "quotas": quotas,
        "student_prefix": STUDENT_PREFIX,
        "teacher_prefix": TEACHER_PREFIX,
        "create_training_job_request": request,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if cfg.dry_run:
        return 0
    sm = session.client("sagemaker")
    try:
        sm.create_training_job(**request)
    except Exception:
        # Best-effort stop if partially created.
        stop_job(sm, job_name)
        raise
    if args.wait:
        try:
            desc = wait_and_cleanup(sm, job_name)
        except Exception:
            stop_job(sm, job_name)
            raise
        cost = billed_cost_usd(desc)
        result = {
            "job_name": job_name,
            "status": desc["TrainingJobStatus"],
            "failure_reason": desc.get("FailureReason"),
            "billable_time_seconds": desc.get("BillableTimeInSeconds"),
            "billed_cost_usd": cost,
            "model_artifacts": desc.get("ModelArtifacts"),
            "output_path": request["OutputDataConfig"]["S3OutputPath"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if desc["TrainingJobStatus"] == "Completed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
