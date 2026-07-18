#!/usr/bin/env python3
"""Evaluate 72B materialization/rehearsal gates against live AWS state.

Never prints credentials. Exits 0 with a JSON report; may_execute may still be false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.qwen72b_fallback.cost import (  # noqa: E402
    P4DE_HOURLY_USD,
    TRANSFER_HOURLY_USD,
    exact_gross_cost_usd,
)
from experiments.qwen72b_fallback.deadline import (  # noqa: E402
    FULL_MAX_RUNTIME_SECONDS,
    REHEARSAL_MAX_RUNTIME_SECONDS,
)
from experiments.qwen72b_fallback.pins import (  # noqa: E402
    DISTILLERY_BUCKET,
    MODEL_ID,
    REVISION,
    TOKENIZER_SHA256,
    load_weight_inventory,
    sealed_identity,
)
from experiments.qwen72b_fallback.readiness import evaluate_readiness  # noqa: E402

TRANSFER_STATUS_KEY = "models/_ephemeral-transfer/14b-32b-status.json"
AWS_PROFILE = "gabriel-cli"
AWS_REGION = "us-east-1"
TRANSFER_ROLE = "distillery-model-transfer-20260718172924"
TRAINING_ROLE = "distillery-sagemaker-training"
ECR_REPO = "distillery-training"


def _session() -> boto3.session.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def _role_exists(iam: Any, name: str) -> bool:
    try:
        iam.get_role(RoleName=name)
        return True
    except ClientError:
        return False


def _ecr_has_digest(ecr: Any) -> bool:
    try:
        paginator = ecr.get_paginator("describe_images")
        for page in paginator.paginate(repositoryName=ECR_REPO):
            for image in page.get("imageDetails", []):
                if image.get("imageDigest", "").startswith("sha256:"):
                    return True
        return False
    except ClientError:
        return False


def _snapshot_complete(s3: Any, inventory: dict[str, Any]) -> bool:
    prefix = f"models/Qwen/Qwen2.5-72B-Instruct/{REVISION}/"
    try:
        listed = s3.list_objects_v2(Bucket=DISTILLERY_BUCKET, Prefix=prefix)
    except ClientError:
        return False
    contents = listed.get("Contents") or []
    present = {item["Key"].removeprefix(prefix): int(item["Size"]) for item in contents}
    for name, meta in inventory["files"].items():
        if name not in present or present[name] != int(meta["size"]):
            return False
    return True


def _conflicting_p4de(sm: Any) -> bool:
    try:
        jobs = sm.list_training_jobs(StatusEquals="InProgress", MaxResults=50)[
            "TrainingJobSummaries"
        ]
    except ClientError:
        return True
    for job in jobs:
        name = job["TrainingJobName"]
        try:
            detail = sm.describe_training_job(TrainingJobName=name)
        except ClientError:
            return True
        instance = detail.get("ResourceConfig", {}).get("InstanceType")
        if instance == "ml.p4de.24xlarge":
            return True
    return False


def _active_transfer(ec2: Any, s3: Any) -> bool:
    try:
        reservations = ec2.describe_instances(
            Filters=[{"Name": "instance-state-name", "Values": ["pending", "running"]}]
        )["Reservations"]
    except ClientError:
        return True
    for reservation in reservations:
        for instance in reservation.get("Instances", []):
            tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
            name = tags.get("Name", "")
            if "distillery-model-transfer" in name:
                return True
    try:
        obj = s3.get_object(Bucket=DISTILLERY_BUCKET, Key=TRANSFER_STATUS_KEY)
        status = json.loads(obj["Body"].read().decode())
        if status.get("ok") is False and status.get("phase") not in {"done", "failed"}:
            return True
    except ClientError:
        pass
    return False


def _active_g5_smoke(sm: Any) -> bool:
    try:
        jobs = sm.list_training_jobs(StatusEquals="InProgress", MaxResults=50)[
            "TrainingJobSummaries"
        ]
    except ClientError:
        return True
    for job in jobs:
        name = job["TrainingJobName"].lower()
        if "g5" in name or "smoke" in name or "tinyfable" in name:
            return True
        try:
            detail = sm.describe_training_job(TrainingJobName=job["TrainingJobName"])
        except ClientError:
            return True
        instance = detail.get("ResourceConfig", {}).get("InstanceType", "")
        if instance.startswith("ml.g5."):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(prog="check_gates")
    parser.add_argument(
        "--action",
        choices=("materialize", "rehearsal", "full"),
        required=True,
    )
    args = parser.parse_args()

    identity = sealed_identity()
    inventory = load_weight_inventory()
    identity_ok = (
        identity.model_id == MODEL_ID
        and identity.revision == REVISION
        and identity.tokenizer_sha256 == TOKENIZER_SHA256
    )
    inventory_ok = int(inventory["n_safetensors_shards"]) == 37
    license_ok = "qwen-license" in identity.license_disposition.lower()
    tokenizer_family_ok = bool(inventory.get("qwen_family_tokenizer_compatible"))

    session = _session()
    iam = session.client("iam")
    ecr = session.client("ecr")
    s3 = session.client("s3")
    sm = session.client("sagemaker")
    ec2 = session.client("ec2")

    report = evaluate_readiness(
        action=args.action,
        identity_ok=identity_ok,
        inventory_ok=inventory_ok,
        license_ok=license_ok,
        tokenizer_family_ok=tokenizer_family_ok,
        iam_transfer_role_ok=_role_exists(iam, TRANSFER_ROLE),
        iam_training_role_ok=_role_exists(iam, TRAINING_ROLE),
        ecr_image_digest_present=_ecr_has_digest(ecr),
        snapshot_complete_on_s3=_snapshot_complete(s3, inventory),
        conflicting_p4de_job_active=_conflicting_p4de(sm),
        conflicting_transfer_active=_active_transfer(ec2, s3),
        active_g5_smoke=_active_g5_smoke(sm),
        active_14b_work=_active_transfer(ec2, s3),
        materialization_projected_usd=exact_gross_cost_usd(
            hourly_usd=TRANSFER_HOURLY_USD,
            max_runtime_seconds=3 * 3600,
        ),
        rehearsal_projected_usd=exact_gross_cost_usd(
            hourly_usd=P4DE_HOURLY_USD,
            max_runtime_seconds=REHEARSAL_MAX_RUNTIME_SECONDS,
        ),
        full_projected_usd=exact_gross_cost_usd(
            hourly_usd=P4DE_HOURLY_USD,
            max_runtime_seconds=FULL_MAX_RUNTIME_SECONDS,
        ),
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
