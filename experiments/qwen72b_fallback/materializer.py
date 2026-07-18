"""Gate-bound EC2 coordinator with dual termination and wipe controls."""

from __future__ import annotations

import base64
import io
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from experiments.qwen72b_fallback.bindings import load_execution_bindings
from experiments.qwen72b_fallback.cost import (
    TRANSFER_HARD_CAP_USD,
    TRANSFER_HOURLY_USD,
)
from experiments.qwen72b_fallback.evidence import (
    SHA256_PATTERN,
    HashBoundEvidence,
    sha256_bytes,
)
from experiments.qwen72b_fallback.pins import (
    DISTILLERY_BUCKET,
    MODEL_ID,
    REVISION,
    load_weight_inventory,
)
from experiments.qwen72b_fallback.readiness import (
    ExecutionAction,
    ExecutionAuthorization,
)

PACKAGE_DIR = Path(__file__).resolve().parent
WORKER_PATH = PACKAGE_DIR / "materializer_worker.py"
REQUIREMENTS_LOCK_PATH = (
    PACKAGE_DIR.parents[1] / "containers" / "materializer" / "requirements.lock"
)
MODEL_PREFIX = f"models/Qwen/Qwen2.5-72B-Instruct/{REVISION}"
STATUS_PREFIX = "models/_ephemeral-transfer/qwen72b"
MAX_RUNTIME_SECONDS = 3 * 3600
ROOT_VOLUME_GIB = 250
UV_VERSION = "0.6.14"
MATERIALIZER_PYTHON_VERSION = "3.11.9"
UV_ARCHIVE_SHA256 = "0aaf451c391d3913823bfb8ed354b446dcfd0553a32ed8266611e4181c61fd51"
MAX_USER_DATA_BYTES = 16 * 1024


class MaterializationObject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sha256: str = Field(pattern=SHA256_PATTERN)
    size: int = Field(gt=0)


class MaterializationPlan(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.materialization_plan.v2"] = (
        "distillery.qwen72b_fallback.materialization_plan.v2"
    )
    launch_name: str = Field(pattern=r"^qwen72b-transfer-[a-z0-9-]{1,40}$")
    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    revision: Literal["495f39366efef23836d0cfae4fbe635880d2be31"] = REVISION
    bucket: Literal["distillery-225989358036-us-east-1"] = DISTILLERY_BUCKET
    prefix: Literal["models/Qwen/Qwen2.5-72B-Instruct/495f39366efef23836d0cfae4fbe635880d2be31"] = (
        MODEL_PREFIX
    )
    status_key: str
    materialization_manifest_key: Literal["models/materialization.json"] = (
        "models/materialization.json"
    )
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    expected: dict[str, MaterializationObject]
    authorization_file_sha256: str = Field(pattern=SHA256_PATTERN)
    hourly_usd: Literal[1.944] = TRANSFER_HOURLY_USD
    hard_cap_usd: Literal[500.0] = TRANSFER_HARD_CAP_USD
    max_runtime_seconds: Literal[10800] = MAX_RUNTIME_SECONDS
    hf_transfer_enabled: Literal[False] = False


class TerminationEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.termination.v1"] = (
        "distillery.qwen72b_fallback.termination.v1"
    )
    instance_id: str = Field(pattern=r"^i-[0-9a-f]{8,17}$")
    final_state: Literal["terminated"]
    delete_on_termination: Literal[True]
    local_wipe_required: Literal[True]
    instance_initiated_shutdown_behavior: Literal["terminate"]


@dataclass(frozen=True, slots=True)
class MaterializationLaunchResult:
    instance_id: str
    status: dict[str, Any]
    termination: TerminationEvidence


def build_materialization_plan(
    authorization: ExecutionAuthorization,
) -> MaterializationPlan:
    authorization.require_current(
        action=ExecutionAction.MATERIALIZE,
        launch_name=authorization.launch_name,
    )
    inventory = load_weight_inventory()
    auth_bytes = (
        json.dumps(
            authorization.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return MaterializationPlan.seal(
        launch_name=authorization.launch_name,
        status_key=f"{STATUS_PREFIX}/{authorization.launch_name}.json",
        inventory_sha256=inventory.inventory_sha256,
        expected={
            name: MaterializationObject(sha256=item.sha256, size=item.size)
            for name, item in inventory.files.items()
        },
        authorization_file_sha256=sha256_bytes(auth_bytes),
    )


def _tar_payload(
    plan: MaterializationPlan,
    authorization: ExecutionAuthorization,
) -> bytes:
    files = {
        "materializer_worker.py": WORKER_PATH.read_bytes(),
        "requirements.lock": REQUIREMENTS_LOCK_PATH.read_bytes(),
        "plan.json": (
            json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode(),
        "authorization.json": (
            json.dumps(
                authorization.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    }
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, body in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o444
            info.mtime = 0
            archive.addfile(info, io.BytesIO(body))
    return output.getvalue()


def render_bootstrap(
    plan: MaterializationPlan,
    authorization: ExecutionAuthorization,
) -> str:
    """Internal user-data renderer. No CLI exposes a standalone bypass worker."""
    payload = _tar_payload(plan, authorization)
    encoded = base64.b64encode(payload).decode("ascii")
    payload_sha256 = sha256_bytes(payload)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
WORK=/opt/distillery-qwen72b-materializer
cleanup() {{
  rm -rf "$WORK" /var/lib/distillery-transfer/qwen72b
  shutdown -h now || true
}}
trap cleanup EXIT INT TERM
mkdir -p "$WORK"
python3 - <<'PY'
import base64, hashlib, io, tarfile
from pathlib import Path
payload = base64.b64decode({encoded!r})
if hashlib.sha256(payload).hexdigest() != {payload_sha256!r}:
    raise SystemExit("bootstrap payload hash mismatch")
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    for member in archive.getmembers():
        if not member.isfile() or "/" in member.name or member.name in {{".", ".."}}:
            raise SystemExit("unsafe bootstrap archive member")
    archive.extractall("/opt/distillery-qwen72b-materializer")
PY
python3 - <<'PY'
import hashlib, os, tarfile, urllib.request
from pathlib import Path
version = {UV_VERSION!r}
expected = {UV_ARCHIVE_SHA256!r}
url = f"https://github.com/astral-sh/uv/releases/download/{{version}}/uv-x86_64-unknown-linux-gnu.tar.gz"
body = urllib.request.urlopen(url, timeout=120).read()
if hashlib.sha256(body).hexdigest() != expected:
    raise SystemExit("uv archive checksum mismatch")
archive_path = Path("/tmp/uv.tar.gz")
archive_path.write_bytes(body)
with tarfile.open(archive_path, mode="r:gz") as archive:
    source = archive.extractfile("uv-x86_64-unknown-linux-gnu/uv")
    if source is None:
        raise SystemExit("uv archive lacks executable")
    target = Path("/usr/local/bin/uv")
    target.write_bytes(source.read())
    os.chmod(target, 0o755)
archive_path.unlink()
PY
uv python install {MATERIALIZER_PYTHON_VERSION}
uv venv --python {MATERIALIZER_PYTHON_VERSION} --python-preference only-managed "$WORK/.venv"
uv pip install --python "$WORK/.venv/bin/python" --require-hashes -r "$WORK/requirements.lock"
export HF_HUB_ENABLE_HF_TRANSFER=0
timeout --signal=TERM --kill-after=120 {MAX_RUNTIME_SECONDS} \
  "$WORK/.venv/bin/python" "$WORK/materializer_worker.py" \
  --plan "$WORK/plan.json" --authorization "$WORK/authorization.json"
"""
    if len(script.encode()) > MAX_USER_DATA_BYTES:
        raise ValueError(
            f"EC2 user data is {len(script.encode())} bytes; limit is {MAX_USER_DATA_BYTES}"
        )
    return script


def _assert_hardened_transfer_resources(ec2: Any) -> dict[str, str]:
    bindings = load_execution_bindings()
    missing = [
        name
        for name, value in {
            "transfer_ami_id": bindings.transfer_ami_id,
            "transfer_instance_profile_arn": bindings.transfer_instance_profile_arn,
            "transfer_security_group_id": bindings.transfer_security_group_id,
            "transfer_subnet_id": bindings.transfer_subnet_id,
        }.items()
        if value is None
    ]
    if missing:
        raise RuntimeError(f"sealed transfer resources are absent: {missing}")
    image = ec2.describe_images(ImageIds=[bindings.transfer_ami_id]).get("Images", [])
    if (
        len(image) != 1
        or image[0].get("ImageId") != bindings.transfer_ami_id
        or image[0].get("State") != "available"
        or image[0].get("Architecture") != "x86_64"
        or image[0].get("RootDeviceType") != "ebs"
    ):
        raise RuntimeError("sealed transfer AMI failed exact live verification")
    root_device_name = image[0].get("RootDeviceName")
    if not isinstance(root_device_name, str) or not root_device_name.startswith("/dev/"):
        raise RuntimeError("sealed transfer AMI lacks an exact root device name")
    group = ec2.describe_security_groups(GroupIds=[bindings.transfer_security_group_id]).get(
        "SecurityGroups", []
    )
    if len(group) != 1 or group[0].get("IpPermissions"):
        raise RuntimeError("transfer security group must have no ingress rules")
    subnet = ec2.describe_subnets(SubnetIds=[bindings.transfer_subnet_id]).get(
        "Subnets",
        [],
    )
    if (
        len(subnet) != 1
        or subnet[0].get("State") != "available"
        or subnet[0].get("MapPublicIpOnLaunch") is not False
    ):
        raise RuntimeError("transfer subnet must be available and private")
    return {
        "ami_id": str(bindings.transfer_ami_id),
        "profile_name": str(bindings.transfer_instance_profile_arn).rsplit("/", 1)[-1],
        "security_group_id": str(bindings.transfer_security_group_id),
        "subnet_id": str(bindings.transfer_subnet_id),
        "root_device_name": root_device_name,
    }


def terminate_and_verify(
    ec2: Any,
    instance_id: str,
    *,
    sleep: Any = time.sleep,
    attempts: int = 60,
) -> TerminationEvidence:
    ec2.terminate_instances(InstanceIds=[instance_id])
    for _attempt in range(attempts):
        reservations = ec2.describe_instances(InstanceIds=[instance_id]).get(
            "Reservations",
            [],
        )
        instances = [
            instance
            for reservation in reservations
            for instance in reservation.get("Instances", [])
        ]
        if len(instances) == 1 and instances[0].get("State", {}).get("Name") == "terminated":
            mappings = instances[0].get("BlockDeviceMappings", [])
            delete_on_termination = bool(mappings) and all(
                mapping.get("Ebs", {}).get("DeleteOnTermination") is True for mapping in mappings
            )
            if not delete_on_termination:
                raise RuntimeError("terminated transfer instance left an EBS volume orphan")
            return TerminationEvidence.seal(
                instance_id=instance_id,
                final_state="terminated",
                delete_on_termination=True,
                local_wipe_required=True,
                instance_initiated_shutdown_behavior="terminate",
            )
        sleep(10)
    raise RuntimeError(f"instance termination was not verified: {instance_id}")


def _terminate_client_token_instances(
    ec2: Any,
    client_token: str,
    *,
    sleep: Any,
    discovery_attempts: int = 6,
) -> tuple[str, ...]:
    instance_ids: set[str] = set()
    for _attempt in range(discovery_attempts):
        reservations = ec2.describe_instances(
            Filters=[
                {"Name": "client-token", "Values": [client_token]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping", "stopped"],
                },
            ]
        ).get("Reservations", [])
        instance_ids.update(
            str(instance["InstanceId"])
            for reservation in reservations
            for instance in reservation.get("Instances", [])
            if instance.get("InstanceId")
        )
        if instance_ids:
            break
        sleep(5)
    for instance_id in sorted(instance_ids):
        terminate_and_verify(ec2, instance_id, sleep=sleep)
    return tuple(sorted(instance_ids))


def launch_materialization(
    *,
    ec2: Any,
    s3: Any,
    authorization: ExecutionAuthorization,
    sleep: Any = time.sleep,
) -> MaterializationLaunchResult:
    authorization.require_current(
        action=ExecutionAction.MATERIALIZE,
        launch_name=authorization.launch_name,
    )
    plan = build_materialization_plan(authorization)
    resources = _assert_hardened_transfer_resources(ec2)
    user_data = render_bootstrap(plan, authorization)
    client_token = f"q72b-{authorization.evidence_sha256[:48]}"
    try:
        response = ec2.run_instances(
            ClientToken=client_token,
            ImageId=resources["ami_id"],
            InstanceType="c5n.9xlarge",
            MinCount=1,
            MaxCount=1,
            IamInstanceProfile={"Name": resources["profile_name"]},
            InstanceInitiatedShutdownBehavior="terminate",
            MetadataOptions={
                "HttpTokens": "required",
                "HttpEndpoint": "enabled",
                "HttpPutResponseHopLimit": 1,
            },
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": resources["subnet_id"],
                    "Groups": [resources["security_group_id"]],
                    "AssociatePublicIpAddress": False,
                    "DeleteOnTermination": True,
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": resources["root_device_name"],
                    "Ebs": {
                        "VolumeSize": ROOT_VOLUME_GIB,
                        "VolumeType": "gp3",
                        "Encrypted": True,
                        "DeleteOnTermination": True,
                    },
                }
            ],
            UserData=user_data,
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": authorization.launch_name},
                        {"Key": "LaunchName", "Value": authorization.launch_name},
                        {"Key": "DistilleryWorkstream", "Value": "qwen72b-fallback"},
                        {
                            "Key": "AuthorizationSha256",
                            "Value": authorization.evidence_sha256,
                        },
                        {"Key": "MaxRuntimeSeconds", "Value": str(MAX_RUNTIME_SECONDS)},
                    ],
                },
                {
                    "ResourceType": "volume",
                    "Tags": [
                        {"Key": "LaunchName", "Value": authorization.launch_name},
                        {"Key": "DistilleryWorkstream", "Value": "qwen72b-fallback"},
                    ],
                },
            ],
        )
    except BaseException:
        _terminate_client_token_instances(ec2, client_token, sleep=sleep)
        raise
    instances = response.get("Instances", [])
    if len(instances) != 1:
        _terminate_client_token_instances(ec2, client_token, sleep=sleep)
        raise RuntimeError("EC2 launch did not return exactly one transfer instance")
    instance_id = str(instances[0]["InstanceId"])
    status: dict[str, Any] = {}
    termination: TerminationEvidence | None = None
    deadline = time.monotonic() + MAX_RUNTIME_SECONDS
    try:
        while time.monotonic() < deadline:
            try:
                body = s3.get_object(
                    Bucket=DISTILLERY_BUCKET,
                    Key=plan.status_key,
                )["Body"].read()
                status = json.loads(body)
                if (
                    status.get("launch_name") == authorization.launch_name
                    and status.get("phase") == "complete"
                    and status.get("ok") is True
                    and status.get("local_wipe_complete") is True
                ):
                    break
                if (
                    status.get("launch_name") == authorization.launch_name
                    and status.get("phase") == "failed"
                ):
                    raise RuntimeError(
                        f"materializer worker failed: {status.get('error', 'unknown')}"
                    )
            except Exception as exc:  # noqa: BLE001 - normalize injected/boto clients
                code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                if code not in {"NoSuchKey", "404"}:
                    raise
            sleep(15)
        else:
            raise RuntimeError("materializer coordinator deadline expired")
    finally:
        termination = terminate_and_verify(ec2, instance_id, sleep=sleep)
    if status.get("local_wipe_complete") is not True:
        raise RuntimeError("materializer did not attest local wipe completion")
    assert termination is not None
    return MaterializationLaunchResult(
        instance_id=instance_id,
        status=status,
        termination=termination,
    )


def terminate_orphan(
    *,
    ec2: Any,
    instance_id: str,
    typed_confirmation: str,
    sleep: Any = time.sleep,
) -> TerminationEvidence:
    expected = f"TERMINATE QWEN72B TRANSFER {instance_id}"
    if typed_confirmation != expected:
        raise ValueError(f"typed confirmation must equal: {expected}")
    reservations = ec2.describe_instances(InstanceIds=[instance_id]).get(
        "Reservations",
        [],
    )
    instances = [
        instance for reservation in reservations for instance in reservation.get("Instances", [])
    ]
    if len(instances) != 1:
        raise RuntimeError("orphan cleanup requires exactly one live instance identity")
    tags = {str(tag["Key"]): str(tag["Value"]) for tag in instances[0].get("Tags", [])}
    if tags.get("DistilleryWorkstream") != "qwen72b-fallback":
        raise RuntimeError("refusing to terminate a non-Qwen72B instance")
    return terminate_and_verify(ec2, instance_id, sleep=sleep)
