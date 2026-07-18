"""Standalone EC2 worker for body-hash-verified Qwen72B materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

WORK_ROOT = Path("/var/lib/distillery-transfer/qwen72b")
MIN_VOLUME_BYTES = 200 * 1024**3
REGION = "us-east-1"
DOWNLOAD_WORKERS = 8
HF_TRANSFER_ENV = "HF_HUB_ENABLE_HF_TRANSFER"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(body: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
        if chunk:
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    body = path.read_bytes()
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, body


def validate_inputs(plan: dict[str, Any], auth: dict[str, Any], auth_bytes: bytes) -> None:
    required_plan = {
        "schema_version",
        "launch_name",
        "model_id",
        "revision",
        "bucket",
        "prefix",
        "status_key",
        "materialization_manifest_key",
        "inventory_sha256",
        "expected",
        "authorization_file_sha256",
        "hourly_usd",
        "hard_cap_usd",
        "max_runtime_seconds",
        "hf_transfer_enabled",
        "evidence_sha256",
    }
    if set(plan) != required_plan:
        raise ValueError("materialization plan field set differs from worker contract")
    if plan["schema_version"] != "distillery.qwen72b_fallback.materialization_plan.v2":
        raise ValueError("materialization plan schema mismatch")
    if plan["hf_transfer_enabled"] is not False:
        raise ValueError("hf_transfer must remain explicitly disabled")
    if not isinstance(plan["evidence_sha256"], str) or len(plan["evidence_sha256"]) != 64:
        raise ValueError("materialization plan lacks a valid evidence hash")
    if os.environ.get(HF_TRANSFER_ENV, "0") != "0":
        raise ValueError("HF_HUB_ENABLE_HF_TRANSFER must be disabled")
    if hashlib.sha256(auth_bytes).hexdigest() != plan["authorization_file_sha256"]:
        raise ValueError("authorization file body hash differs from materialization plan")
    if auth.get("action") != "materialize":
        raise ValueError("worker authorization action must be materialize")
    if auth.get("launch_name") != plan["launch_name"]:
        raise ValueError("worker authorization launch name mismatch")
    if auth.get("source") != "live_aws":
        raise ValueError("worker requires live-AWS authorization evidence")
    if int(auth.get("expires_unix_seconds", 0)) <= int(time.time()):
        raise ValueError("worker authorization expired before bootstrap")
    bundle = auth.get("evidence_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("worker authorization lacks evidence bundle")
    reviews = bundle.get("reviews", {}).get("review_packet_sha256")
    if not isinstance(reviews, list) or len(reviews) != 2 or len(set(reviews)) != 2:
        raise ValueError("worker authorization lacks two independent review packets")
    conflicts = bundle.get("conflicts")
    if not isinstance(conflicts, dict):
        raise ValueError("worker authorization lacks live conflict evidence")
    for field in (
        "active_p4de_jobs",
        "active_g5_jobs",
        "active_14b_or_32b_jobs",
        "active_transfer_instance_ids",
        "duplicate_launches",
        "orphan_resource_ids",
    ):
        if conflicts.get(field) != []:
            raise ValueError(f"worker authorization has non-empty conflict field {field}")
    confirmation = bundle.get("confirmation", {})
    expected_confirmation = f"EXECUTE QWEN72B MATERIALIZE {plan['launch_name']}"
    if confirmation.get("typed_text") != expected_confirmation:
        raise ValueError("worker authorization lacks exact typed confirmation")
    expected = plan["expected"]
    if not isinstance(expected, dict) or len(expected) < 40:
        raise ValueError("materialization plan lacks complete inventory")
    for name, metadata in expected.items():
        if not isinstance(name, str) or "/" in name or name in {"", ".", ".."}:
            raise ValueError(f"unsafe materialization filename: {name!r}")
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"sha256", "size"}
            or not isinstance(metadata["size"], int)
            or metadata["size"] <= 0
            or not isinstance(metadata["sha256"], str)
            or len(metadata["sha256"]) != 64
        ):
            raise ValueError(f"invalid expected metadata for {name}")


def verify_volume(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(path)
    capacity = stats.f_frsize * stats.f_blocks
    free = stats.f_frsize * stats.f_bavail
    if capacity < MIN_VOLUME_BYTES or free < MIN_VOLUME_BYTES:
        raise RuntimeError(
            f"materializer requires >=200 GiB capacity and free space; "
            f"capacity={capacity} free={free}"
        )


def download_one(plan: dict[str, Any], name: str, destination: Path) -> None:
    encoded_name = urllib.parse.quote(name, safe="")
    url = (
        f"https://huggingface.co/{plan['model_id']}/resolve/"
        f"{plan['revision']}/{encoded_name}?download=true"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "distillery-qwen72b-materializer/2"},
    )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    metadata = plan["expected"][name]
    if temporary.stat().st_size != metadata["size"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download size mismatch: {name}")
    if sha256_file(temporary) != metadata["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download body hash mismatch: {name}")
    os.replace(temporary, destination)


def status_writer(s3: Any, plan: dict[str, Any], started: float, phase: str, **extra: Any) -> None:
    payload = {
        "schema_version": "distillery.qwen72b_fallback.transfer_status.v2",
        "launch_name": plan["launch_name"],
        "model_id": plan["model_id"],
        "revision": plan["revision"],
        "phase": phase,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        **extra,
    }
    s3.put_object(
        Bucket=plan["bucket"],
        Key=plan["status_key"],
        Body=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
    )


def check_budget(plan: dict[str, Any], started: float) -> None:
    elapsed = time.monotonic() - started
    if elapsed >= int(plan["max_runtime_seconds"]):
        raise RuntimeError("materializer exceeded sealed runtime")
    accrued = float(plan["hourly_usd"]) * elapsed / 3600.0
    if accrued >= float(plan["hard_cap_usd"]):
        raise RuntimeError("materializer exceeded sealed cost cap")


def merge_materialization_manifest(
    s3: Any,
    plan: dict[str, Any],
    object_hashes: dict[str, str],
) -> tuple[str, str]:
    bucket = plan["bucket"]
    key = plan["materialization_manifest_key"]
    etag: str | None = None
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        previous_body = response["Body"].read()
        etag = str(response["ETag"])
        previous = json.loads(previous_body)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
            raise
        previous = {
            "schema_version": "distillery.model_materialization.v2",
            "models": [],
        }
    models = previous.get("models")
    if not isinstance(models, list):
        raise ValueError("existing materialization manifest lacks models list")
    retained = [
        entry
        for entry in models
        if not (
            isinstance(entry, dict)
            and entry.get("model_id") == plan["model_id"]
            and entry.get("revision") == plan["revision"]
        )
    ]
    entry = {
        "model_id": plan["model_id"],
        "revision": plan["revision"],
        "inventory_sha256": plan["inventory_sha256"],
        "object_body_sha256": object_hashes,
        "materialized_at_utc": datetime.now(UTC).isoformat(),
    }
    payload = {
        **previous,
        "schema_version": "distillery.model_materialization.v2",
        "models": sorted(
            [*retained, entry],
            key=lambda item: (str(item.get("model_id")), str(item.get("revision"))),
        ),
    }
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": "application/json",
    }
    if etag is None:
        kwargs["IfNoneMatch"] = "*"
    else:
        kwargs["IfMatch"] = etag
    put_response = s3.put_object(**kwargs)
    version_id = put_response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise RuntimeError("versioned materialization manifest lacks VersionId")
    uploaded = s3.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
    )["Body"].read()
    if uploaded != body:
        raise RuntimeError("materialization manifest body changed after conditional merge")
    return hashlib.sha256(body).hexdigest(), version_id


def run(plan_path: Path, authorization_path: Path) -> None:
    started = time.monotonic()
    plan, _plan_bytes = load_json(plan_path)
    authorization, authorization_bytes = load_json(authorization_path)
    validate_inputs(plan, authorization, authorization_bytes)
    verify_volume(WORK_ROOT)
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=32,
        ),
    )
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    completion: dict[str, Any] | None = None
    uploaded_keys: list[str] = []
    try:
        if s3.get_bucket_versioning(Bucket=plan["bucket"]).get("Status") != "Enabled":
            raise RuntimeError("materialization requires S3 bucket versioning")
        existing = s3.list_objects_v2(
            Bucket=plan["bucket"],
            Prefix=f"{plan['prefix']}/",
            MaxKeys=1,
        )
        if existing.get("KeyCount", len(existing.get("Contents", []))) != 0:
            raise RuntimeError("materialization destination prefix is not empty")
        status_writer(s3, plan, started, "downloading", ok=False)
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = [
                pool.submit(download_one, plan, name, WORK_ROOT / name)
                for name in sorted(plan["expected"])
            ]
            for future in futures:
                future.result()
                check_budget(plan, started)

        status_writer(s3, plan, started, "uploading", ok=False)
        for name in sorted(plan["expected"]):
            metadata = plan["expected"][name]
            s3.upload_file(
                str(WORK_ROOT / name),
                plan["bucket"],
                f"{plan['prefix']}/{name}",
                ExtraArgs={
                    "Metadata": {
                        "sha256": metadata["sha256"],
                        "model-id": plan["model_id"],
                        "revision": plan["revision"],
                    }
                },
                Config=transfer,
            )
            uploaded_keys.append(f"{plan['prefix']}/{name}")
            check_budget(plan, started)

        status_writer(s3, plan, started, "body_hash_verification", ok=False)
        object_hashes: dict[str, str] = {}
        for name, metadata in sorted(plan["expected"].items()):
            response = s3.get_object(
                Bucket=plan["bucket"],
                Key=f"{plan['prefix']}/{name}",
                ChecksumMode="ENABLED",
            )
            actual_hash, actual_size = sha256_stream(response["Body"])
            if actual_hash != metadata["sha256"] or actual_size != metadata["size"]:
                raise RuntimeError(f"uploaded S3 body mismatch: {name}")
            object_hashes[name] = actual_hash
            check_budget(plan, started)

        sums = (
            "\n".join(f"{digest}  {name}" for name, digest in sorted(object_hashes.items())) + "\n"
        ).encode()
        snapshot = {
            "schema_version": "distillery.model_snapshot.v2",
            "model_id": plan["model_id"],
            "revision": plan["revision"],
            "inventory_sha256": plan["inventory_sha256"],
            "object_body_sha256": object_hashes,
            "authorization_file_sha256": plan["authorization_file_sha256"],
            "materialized_at_utc": datetime.now(UTC).isoformat(),
        }
        snapshot_body = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
        s3.put_object(
            Bucket=plan["bucket"],
            Key=f"{plan['prefix']}/SHA256SUMS",
            Body=sums,
            ContentType="text/plain",
        )
        uploaded_keys.append(f"{plan['prefix']}/SHA256SUMS")
        s3.put_object(
            Bucket=plan["bucket"],
            Key=f"{plan['prefix']}/snapshot-manifest.json",
            Body=snapshot_body,
            ContentType="application/json",
        )
        uploaded_keys.append(f"{plan['prefix']}/snapshot-manifest.json")
        uploaded_sums = s3.get_object(
            Bucket=plan["bucket"],
            Key=f"{plan['prefix']}/SHA256SUMS",
        )["Body"].read()
        uploaded_snapshot = s3.get_object(
            Bucket=plan["bucket"],
            Key=f"{plan['prefix']}/snapshot-manifest.json",
        )["Body"].read()
        if uploaded_sums != sums or uploaded_snapshot != snapshot_body:
            raise RuntimeError("uploaded control object body hash mismatch")
        manifest_hash, manifest_version_id = merge_materialization_manifest(
            s3,
            plan,
            object_hashes,
        )
        completion = {
            "object_count": len(object_hashes),
            "object_body_sha256": object_hashes,
            "sha256sums_body_sha256": hashlib.sha256(sums).hexdigest(),
            "snapshot_manifest_sha256": hashlib.sha256(snapshot_body).hexdigest(),
            "materialization_manifest_sha256": manifest_hash,
            "materialization_manifest_version_id": manifest_version_id,
        }
    except BaseException as exc:
        try:
            status_writer(
                s3,
                plan,
                started,
                "failed",
                ok=False,
                error_type=type(exc).__name__,
                error=str(exc)[:512],
            )
        finally:
            if uploaded_keys:
                for offset in range(0, len(uploaded_keys), 1000):
                    s3.delete_objects(
                        Bucket=plan["bucket"],
                        Delete={
                            "Objects": [
                                {"Key": key} for key in uploaded_keys[offset : offset + 1000]
                            ],
                            "Quiet": True,
                        },
                    )
            raise
    finally:
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
    assert completion is not None
    status_writer(
        s3,
        plan,
        started,
        "complete",
        ok=True,
        **completion,
        local_wipe_complete=not WORK_ROOT.exists(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.authorization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
