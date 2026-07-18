"""Concrete read-only AWS probes for the 72B readiness engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.bindings import (
    TRAINING_ROLE_ARN,
    load_execution_bindings,
)
from experiments.qwen72b_fallback.cost import (
    FULL_RUN_HARD_CAP_USD,
    P4DE_HOURLY_USD,
    PROBE_HARD_CAP_USD,
    REHEARSAL_HARD_CAP_USD,
    TRANSFER_HARD_CAP_USD,
    TRANSFER_HOURLY_USD,
    TRANSFER_PRICE_SOURCE,
    ActiveResourceCost,
    CostAction,
    ResourceKind,
    exact_gross_cost_usd,
    seal_cost_evidence,
)
from experiments.qwen72b_fallback.evidence import sha256_bytes
from experiments.qwen72b_fallback.finance_world_targets import (
    FinanceWorldCorpusEvidence,
    build_finance_world_targets,
)
from experiments.qwen72b_fallback.license_policy import verify_license_artifacts
from experiments.qwen72b_fallback.memory import (
    Qwen72BMemoryProbeEvidence,
    require_measured_probe,
)
from experiments.qwen72b_fallback.pins import (
    AWS_REGION,
    DISTILLERY_ACCOUNT_ID,
    DISTILLERY_BUCKET,
    EXECUTION_BINDINGS_PATH,
    MODEL_ID,
    REVISION,
    load_weight_inventory,
    sealed_identity,
)
from experiments.qwen72b_fallback.profile import Qwen72BTrainingProfile, RunKind
from experiments.qwen72b_fallback.readiness import (
    ConflictEvidence,
    EcrImageEvidence,
    ExecutionAction,
    GateCode,
    IamScopeEvidence,
    LocalPolicyEvidence,
    ReviewClearanceEvidence,
    S3SnapshotEvidence,
    VerificationFailure,
)
from experiments.qwen72b_fallback.tokenizer_compat import (
    TOKENIZER_FILENAMES,
    TokenizerCompatibilityEvidence,
    load_target_registry,
    seal_compatibility,
    verify_tokenizer_pair,
)

MODEL_PREFIX = f"models/Qwen/Qwen2.5-72B-Instruct/{REVISION}"
MATERIALIZATION_MANIFEST_KEY = "models/materialization.json"
TRANSFER_MAX_SECONDS = 3 * 3600
ACTIVE_TRAINING_STATUSES = ("InProgress", "Stopping")
ACTIVE_EC2_STATES = ("pending", "running", "stopping", "shutting-down")


def _read_streaming_body(response: dict[str, Any]) -> bytes:
    body = response.get("Body")
    if body is None:
        raise ValueError("S3 get_object response lacks Body")
    chunks: list[bytes] = []
    if hasattr(body, "iter_chunks"):
        chunks.extend(chunk for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024) if chunk)
    else:
        data = body.read()
        if not isinstance(data, bytes):
            raise ValueError("S3 Body.read() did not return bytes")
        chunks.append(data)
    return b"".join(chunks)


def _hash_streaming_body(response: dict[str, Any]) -> tuple[str, int]:
    body = response.get("Body")
    if body is None:
        raise ValueError("S3 get_object response lacks Body")
    digest = hashlib.sha256()
    size = 0
    if hasattr(body, "iter_chunks"):
        chunks: Iterable[bytes] = body.iter_chunks(chunk_size=8 * 1024 * 1024)
    else:
        chunks = iter(lambda: body.read(8 * 1024 * 1024), b"")
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise ValueError("S3 streaming body yielded a non-bytes chunk")
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _s3_uri_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _statement_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError("IAM policy Action/Resource must be string or string list")


class AwsLiveVerifier:
    """Injectable AWS client set; every returned artifact is hash-bound."""

    def __init__(
        self,
        *,
        repo_root: Any,
        sts: Any,
        s3: Any,
        ecr: Any,
        iam: Any,
        ec2: Any,
        sagemaker: Any,
        now: Any = lambda: datetime.now(UTC),
        open_url: Any = urlopen,
    ) -> None:
        self.repo_root = repo_root
        self.sts = sts
        self.s3 = s3
        self.ecr = ecr
        self.iam = iam
        self.ec2 = ec2
        self.sagemaker = sagemaker
        self.now = now
        self.open_url = open_url

    @classmethod
    def from_boto3(cls, *, repo_root: Any, profile_name: str | None = None) -> Any:
        import boto3

        session = boto3.Session(profile_name=profile_name, region_name=AWS_REGION)
        return cls(
            repo_root=repo_root,
            sts=session.client("sts"),
            s3=session.client("s3"),
            ecr=session.client("ecr"),
            iam=session.client("iam"),
            ec2=session.client("ec2"),
            sagemaker=session.client("sagemaker"),
        )

    def verify_local_policy(self) -> LocalPolicyEvidence:
        identity = sealed_identity()
        license_evidence = verify_license_artifacts(self.repo_root)
        if identity.license_file_sha256 != license_evidence.model_license_body_sha256:
            raise VerificationFailure(
                GateCode.LICENSE_ARTIFACTS,
                "identity and Qwen license artifact hashes differ",
            )
        return LocalPolicyEvidence.seal(
            identity=identity,
            license=license_evidence,
            execution_bindings_bytes_sha256=sha256_bytes(EXECUTION_BINDINGS_PATH.read_bytes()),
        )

    def verify_reviews(self) -> ReviewClearanceEvidence:
        bindings = load_execution_bindings()
        if not bindings.both_reviews_clear:
            raise VerificationFailure(
                GateCode.EXECUTION_REVIEWS,
                "both independent execution review packet hashes are still absent",
            )
        packets = bindings.review_packet_sha256
        assert len(packets) == 2
        return ReviewClearanceEvidence.seal(
            review_packet_sha256=(packets[0], packets[1]),
            execution_bindings_bytes_sha256=bindings.file_sha256,
        )

    def _identity(self) -> dict[str, Any]:
        identity = self.sts.get_caller_identity()
        if identity.get("Account") != DISTILLERY_ACCOUNT_ID:
            raise VerificationFailure(
                GateCode.IAM_SCOPE,
                "STS caller belongs to the wrong AWS account",
            )
        arn = str(identity.get("Arn", ""))
        if arn.endswith(":root"):
            raise VerificationFailure(GateCode.IAM_SCOPE, "AWS account root is forbidden")
        return identity

    def _policy_documents(self, role_name: str) -> tuple[dict[str, Any], ...]:
        documents: list[dict[str, Any]] = []
        inline = self.iam.list_role_policies(RoleName=role_name)
        for name in inline.get("PolicyNames", []):
            payload = self.iam.get_role_policy(RoleName=role_name, PolicyName=name)
            document = payload.get("PolicyDocument")
            if not isinstance(document, dict):
                raise ValueError(f"inline role policy {name} lacks a document")
            documents.append(document)
        attached = self.iam.list_attached_role_policies(RoleName=role_name)
        for item in attached.get("AttachedPolicies", []):
            arn = item.get("PolicyArn")
            policy = self.iam.get_policy(PolicyArn=arn).get("Policy", {})
            version = self.iam.get_policy_version(
                PolicyArn=arn,
                VersionId=policy.get("DefaultVersionId"),
            )
            document = version.get("PolicyVersion", {}).get("Document")
            if not isinstance(document, dict):
                raise ValueError(f"attached role policy {arn} lacks a document")
            documents.append(document)
        if not documents:
            raise VerificationFailure(GateCode.IAM_SCOPE, "IAM role has no policies")
        return tuple(documents)

    def _verify_transfer_scope(
        self,
        documents: tuple[dict[str, Any], ...],
    ) -> tuple[str, ...]:
        permitted_bucket = f"arn:aws:s3:::{DISTILLERY_BUCKET}"
        permitted_prefixes = (
            f"{permitted_bucket}/models/Qwen/Qwen2.5-72B-Instruct/{REVISION}",
            f"{permitted_bucket}/models/_ephemeral-transfer/qwen72b",
            f"{permitted_bucket}/models/materialization.json",
        )
        allowed_actions = {
            "s3:AbortMultipartUpload",
            "s3:GetObject",
            "s3:ListBucket",
            "s3:ListBucketMultipartUploads",
            "s3:ListMultipartUploadParts",
            "s3:PutObject",
        }
        resources: set[str] = set()
        for document in documents:
            statements = document.get("Statement")
            if isinstance(statements, dict):
                statements = [statements]
            if not isinstance(statements, list):
                raise ValueError("IAM policy lacks Statement list")
            for statement in statements:
                if not isinstance(statement, dict) or statement.get("Effect") != "Allow":
                    continue
                actions = _statement_values(statement.get("Action", ()))
                statement_resources = _statement_values(statement.get("Resource", ()))
                unexpected_actions = set(actions) - allowed_actions
                if unexpected_actions:
                    raise VerificationFailure(
                        GateCode.IAM_SCOPE,
                        "transfer instance profile has non-allowlisted actions: "
                        f"{sorted(unexpected_actions)}",
                    )
                if actions:
                    for resource in statement_resources:
                        if resource == "*":
                            raise VerificationFailure(
                                GateCode.IAM_SCOPE,
                                "transfer instance profile grants wildcard S3 access",
                            )
                        if resource != permitted_bucket and not any(
                            resource == prefix or resource.startswith(f"{prefix}/")
                            for prefix in permitted_prefixes
                        ):
                            raise VerificationFailure(
                                GateCode.IAM_SCOPE,
                                f"transfer S3 resource is outside sealed prefixes: {resource}",
                            )
                        resources.add(resource)
        if not resources:
            raise VerificationFailure(
                GateCode.IAM_SCOPE,
                "transfer instance profile has no scoped S3 resources",
            )
        return tuple(sorted(resources))

    @staticmethod
    def _observed_resources(
        documents: tuple[dict[str, Any], ...],
    ) -> tuple[str, ...]:
        resources: set[str] = set()
        for document in documents:
            statements = document.get("Statement")
            if isinstance(statements, dict):
                statements = [statements]
            if not isinstance(statements, list):
                raise ValueError("IAM policy lacks Statement list")
            for statement in statements:
                if not isinstance(statement, dict) or statement.get("Effect") != "Allow":
                    continue
                resources.update(_statement_values(statement.get("Resource", ())))
        if not resources:
            raise VerificationFailure(
                GateCode.IAM_SCOPE,
                "IAM role has no observed allow resources",
            )
        return tuple(sorted(resources))

    def verify_iam(self, action: ExecutionAction) -> IamScopeEvidence:
        caller = self._identity()
        bindings = load_execution_bindings()
        instance_profile_arn: str | None = None
        transfer_ami_id: str | None = None
        transfer_subnet_id: str | None = None
        transfer_security_group_id: str | None = None
        if action is ExecutionAction.MATERIALIZE:
            transfer_values = {
                "instance profile": bindings.transfer_instance_profile_arn,
                "AMI": bindings.transfer_ami_id,
                "subnet": bindings.transfer_subnet_id,
                "security group": bindings.transfer_security_group_id,
            }
            missing = [name for name, value in transfer_values.items() if value is None]
            if missing:
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    f"sealed transfer resources are absent: {missing}",
                )
            transfer_ami_id = str(bindings.transfer_ami_id)
            transfer_subnet_id = str(bindings.transfer_subnet_id)
            transfer_security_group_id = str(bindings.transfer_security_group_id)
            profile_name = bindings.transfer_instance_profile_arn.rsplit("/", 1)[-1]
            profile = self.iam.get_instance_profile(InstanceProfileName=profile_name).get(
                "InstanceProfile",
                {},
            )
            if profile.get("Arn") != bindings.transfer_instance_profile_arn:
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    "live transfer instance profile ARN differs from sealed ARN",
                )
            roles = profile.get("Roles", [])
            if not isinstance(roles, list) or len(roles) != 1:
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    "transfer instance profile must contain exactly one role",
                )
            role = roles[0]
            instance_profile_arn = bindings.transfer_instance_profile_arn
            images = self.ec2.describe_images(ImageIds=[transfer_ami_id]).get(
                "Images",
                [],
            )
            if (
                len(images) != 1
                or images[0].get("ImageId") != transfer_ami_id
                or images[0].get("State") != "available"
                or images[0].get("Architecture") != "x86_64"
                or images[0].get("RootDeviceType") != "ebs"
            ):
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    "transfer AMI failed exact live verification",
                )
            groups = self.ec2.describe_security_groups(GroupIds=[transfer_security_group_id]).get(
                "SecurityGroups", []
            )
            if len(groups) != 1 or groups[0].get("IpPermissions"):
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    "transfer security group must exact-match a no-ingress group",
                )
            subnets = self.ec2.describe_subnets(SubnetIds=[transfer_subnet_id]).get("Subnets", [])
            if (
                len(subnets) != 1
                or subnets[0].get("State") != "available"
                or subnets[0].get("MapPublicIpOnLaunch") is not False
            ):
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    "transfer subnet must exact-match an available private subnet",
                )
        else:
            role_name = TRAINING_ROLE_ARN.rsplit("/", 1)[-1]
            role = self.iam.get_role(RoleName=role_name).get("Role", {})
            if role.get("Arn") != TRAINING_ROLE_ARN:
                raise VerificationFailure(
                    GateCode.IAM_SCOPE,
                    "live SageMaker role differs from exact sealed role ARN",
                )
        role_name = str(role.get("RoleName", ""))
        role_arn = str(role.get("Arn", ""))
        role_id = str(role.get("RoleId", ""))
        if not role_name or not role_arn or not role_id:
            raise VerificationFailure(GateCode.IAM_SCOPE, "IAM role identity is incomplete")
        documents = self._policy_documents(role_name)
        if action is ExecutionAction.MATERIALIZE:
            resources = self._verify_transfer_scope(documents)
        else:
            resources = self._observed_resources(documents)
        return IamScopeEvidence.seal(
            action=action,
            caller_account_id=DISTILLERY_ACCOUNT_ID,
            caller_arn=str(caller["Arn"]),
            role_arn=role_arn,
            role_id=role_id,
            instance_profile_arn=instance_profile_arn,
            transfer_ami_id=transfer_ami_id,
            transfer_subnet_id=transfer_subnet_id,
            transfer_security_group_id=transfer_security_group_id,
            policy_document_sha256=tuple(
                sorted(content_sha256(document) for document in documents)
            ),
            allowed_s3_resources=resources,
        )

    def _active_training_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for status in ACTIVE_TRAINING_STATUSES:
            token: str | None = None
            while True:
                kwargs: dict[str, Any] = {
                    "StatusEquals": status,
                    "MaxResults": 100,
                }
                if token is not None:
                    kwargs["NextToken"] = token
                response = self.sagemaker.list_training_jobs(**kwargs)
                for summary in response.get("TrainingJobSummaries", []):
                    name = str(summary["TrainingJobName"])
                    jobs.append(self.sagemaker.describe_training_job(TrainingJobName=name))
                token = response.get("NextToken")
                if not token:
                    break
        return jobs

    def _active_transfer_instances(self) -> list[dict[str, Any]]:
        response = self.ec2.describe_instances(
            Filters=[
                {
                    "Name": "tag:DistilleryWorkstream",
                    "Values": ["qwen72b-fallback"],
                },
                {"Name": "instance-state-name", "Values": list(ACTIVE_EC2_STATES)},
            ]
        )
        return [
            instance
            for reservation in response.get("Reservations", [])
            for instance in reservation.get("Instances", [])
        ]

    def _all_active_ec2_instances(self) -> list[dict[str, Any]]:
        response = self.ec2.describe_instances(
            Filters=[{"Name": "instance-state-name", "Values": list(ACTIVE_EC2_STATES)}]
        )
        return [
            instance
            for reservation in response.get("Reservations", [])
            for instance in reservation.get("Instances", [])
        ]

    @staticmethod
    def _tags(resource: dict[str, Any]) -> dict[str, str]:
        tags = resource.get("Tags", [])
        if isinstance(tags, list):
            return {
                str(tag["Key"]): str(tag["Value"])
                for tag in tags
                if isinstance(tag, dict) and "Key" in tag and "Value" in tag
            }
        return {}

    def verify_conflicts(
        self,
        action: ExecutionAction,
        launch_name: str,
    ) -> ConflictEvidence:
        jobs = self._active_training_jobs()
        transfers = self._active_transfer_instances()
        active_ec2 = self._all_active_ec2_instances()
        active_p4de: list[str] = []
        active_g5: list[str] = []
        active_large: list[str] = []
        duplicates: list[str] = []
        orphans: list[str] = []
        for job in jobs:
            name = str(job.get("TrainingJobName", ""))
            instance_type = str(job.get("ResourceConfig", {}).get("InstanceType", ""))
            if instance_type == "ml.p4de.24xlarge":
                active_p4de.append(name)
            if ".g5." in instance_type or "g5" in name.lower():
                active_g5.append(name)
            if "14b" in name.lower() or "32b" in name.lower():
                active_large.append(name)
            if name == launch_name:
                duplicates.append(name)
        for instance in active_ec2:
            instance_id = str(instance.get("InstanceId", ""))
            tags = self._tags(instance)
            searchable = " ".join(
                [
                    instance_id,
                    str(instance.get("InstanceType", "")),
                    *tags.keys(),
                    *tags.values(),
                ]
            ).lower()
            if "g5" in searchable:
                active_g5.append(instance_id)
            if "14b" in searchable or "32b" in searchable:
                active_large.append(instance_id)
        transfer_ids: list[str] = []
        for instance in transfers:
            instance_id = str(instance.get("InstanceId", ""))
            transfer_ids.append(instance_id)
            tags = self._tags(instance)
            if tags.get("LaunchName") == launch_name:
                duplicates.append(instance_id)
            if not tags.get("LaunchName"):
                orphans.append(instance_id)
        if action is ExecutionAction.MATERIALIZE:
            versioning = self.s3.get_bucket_versioning(Bucket=DISTILLERY_BUCKET)
            if versioning.get("Status") != "Enabled":
                raise VerificationFailure(
                    GateCode.S3_SNAPSHOT,
                    "materialization requires S3 bucket versioning to be enabled",
                )
            existing_snapshot_keys = self._listed_keys(
                DISTILLERY_BUCKET,
                f"{MODEL_PREFIX}/",
            )
            orphans.extend(
                f"s3://{DISTILLERY_BUCKET}/{key}" for key in sorted(existing_snapshot_keys)
            )
        return ConflictEvidence.seal(
            action=action,
            requested_launch_name=launch_name,
            active_p4de_jobs=tuple(sorted(active_p4de)),
            active_g5_jobs=tuple(sorted(active_g5)),
            active_14b_or_32b_jobs=tuple(sorted(active_large)),
            active_transfer_instance_ids=tuple(sorted(transfer_ids)),
            duplicate_launches=tuple(sorted(set(duplicates))),
            orphan_resource_ids=tuple(sorted(orphans)),
        )

    def _active_resource_costs(self) -> tuple[ActiveResourceCost, ...]:
        now = self.now()
        costs: list[ActiveResourceCost] = []
        for job in self._active_training_jobs():
            instance_type = str(job.get("ResourceConfig", {}).get("InstanceType", ""))
            if instance_type != "ml.p4de.24xlarge":
                continue
            created = job.get("CreationTime")
            age = int(max(0.0, (now - created).total_seconds())) if created else 0
            costs.append(
                ActiveResourceCost(
                    resource_id=str(job.get("TrainingJobName")),
                    resource_kind=ResourceKind.P4DE_TRAINING_JOB,
                    age_seconds=age,
                    hourly_usd=P4DE_HOURLY_USD,
                    accrued_usd=exact_gross_cost_usd(
                        hourly_usd=P4DE_HOURLY_USD,
                        max_runtime_seconds=max(age, 1),
                    ),
                )
            )
        for instance in self._active_transfer_instances():
            launched = instance.get("LaunchTime")
            age = int(max(0.0, (now - launched).total_seconds())) if launched else 0
            costs.append(
                ActiveResourceCost(
                    resource_id=str(instance.get("InstanceId")),
                    resource_kind=ResourceKind.TRANSFER_EC2,
                    age_seconds=age,
                    hourly_usd=TRANSFER_HOURLY_USD,
                    accrued_usd=exact_gross_cost_usd(
                        hourly_usd=TRANSFER_HOURLY_USD,
                        max_runtime_seconds=max(age, 1),
                    ),
                )
            )
        return tuple(sorted(costs, key=lambda item: item.resource_id))

    def verify_cost(
        self,
        action: ExecutionAction,
        profile: Qwen72BTrainingProfile | None,
    ) -> Any:
        resources = self._active_resource_costs()
        if action is ExecutionAction.MATERIALIZE:
            return seal_cost_evidence(
                action=CostAction.MATERIALIZE,
                instance_type="c5n.9xlarge",
                hourly_usd=TRANSFER_HOURLY_USD,
                price_source=TRANSFER_PRICE_SOURCE,
                max_runtime_seconds=TRANSFER_MAX_SECONDS,
                hard_cap_usd=TRANSFER_HARD_CAP_USD,
                active_resources=resources,
            )
        if profile is None:
            raise VerificationFailure(
                GateCode.COST_EXPOSURE,
                "training cost verification requires an exact profile",
            )
        mapping = {
            ExecutionAction.MEMORY_PROBE: (
                CostAction.MEMORY_PROBE,
                PROBE_HARD_CAP_USD,
            ),
            ExecutionAction.REHEARSAL: (
                CostAction.REHEARSAL,
                REHEARSAL_HARD_CAP_USD,
            ),
            ExecutionAction.FULL: (CostAction.FULL, FULL_RUN_HARD_CAP_USD),
            ExecutionAction.TEACHER_TRAJECTORIES: (
                CostAction.REHEARSAL,
                REHEARSAL_HARD_CAP_USD,
            ),
        }
        cost_action, hard_cap = mapping[action]
        return seal_cost_evidence(
            action=cost_action,
            instance_type=profile.instance_type,
            hourly_usd=profile.hourly_usd,
            price_source=profile.price_source,
            max_runtime_seconds=profile.max_runtime_seconds,
            hard_cap_usd=hard_cap,
            active_resources=resources,
        )

    def _get_s3_bytes(self, bucket: str, key: str) -> bytes:
        return _read_streaming_body(
            self.s3.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        )

    def _hash_s3_object(self, bucket: str, key: str) -> tuple[str, int]:
        return _hash_streaming_body(
            self.s3.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        )

    def _listed_keys(self, bucket: str, prefix: str) -> set[str]:
        keys: set[str] = set()
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            response = self.s3.list_objects_v2(**kwargs)
            keys.update(str(item["Key"]) for item in response.get("Contents", []))
            token = response.get("NextContinuationToken")
            if not token:
                return keys

    def verify_s3_snapshot(self) -> S3SnapshotEvidence:
        inventory = load_weight_inventory()
        expected_keys = {f"{MODEL_PREFIX}/{name}" for name in inventory.files}
        control_keys = {
            f"{MODEL_PREFIX}/SHA256SUMS",
            f"{MODEL_PREFIX}/snapshot-manifest.json",
        }
        actual_keys = self._listed_keys(DISTILLERY_BUCKET, f"{MODEL_PREFIX}/")
        if actual_keys != expected_keys | control_keys:
            missing = sorted((expected_keys | control_keys) - actual_keys)
            extra = sorted(actual_keys - (expected_keys | control_keys))
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                f"S3 snapshot key set mismatch; missing={missing} extra={extra}",
            )
        hashes: dict[str, str] = {}
        sizes: dict[str, int] = {}
        for name, expected in sorted(inventory.files.items()):
            actual_hash, actual_size = self._hash_s3_object(
                DISTILLERY_BUCKET,
                f"{MODEL_PREFIX}/{name}",
            )
            if actual_size != expected.size or actual_hash != expected.sha256:
                raise VerificationFailure(
                    GateCode.S3_BODY_HASHES,
                    f"S3 object body mismatch: {name}",
                )
            hashes[name] = actual_hash
            sizes[name] = actual_size
        sums_body = self._get_s3_bytes(
            DISTILLERY_BUCKET,
            f"{MODEL_PREFIX}/SHA256SUMS",
        )
        expected_sums = (
            "\n".join(f"{item.sha256}  {name}" for name, item in sorted(inventory.files.items()))
            + "\n"
        ).encode()
        if sums_body != expected_sums:
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                "S3 SHA256SUMS body differs from exact inventory",
            )
        snapshot_body = self._get_s3_bytes(
            DISTILLERY_BUCKET,
            f"{MODEL_PREFIX}/snapshot-manifest.json",
        )
        snapshot = json.loads(snapshot_body)
        expected_snapshot_fields = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "inventory_sha256": inventory.inventory_sha256,
            "object_body_sha256": hashes,
        }
        if any(snapshot.get(key) != value for key, value in expected_snapshot_fields.items()):
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                "S3 snapshot manifest does not bind exact object body hashes",
            )
        materialization_body = self._get_s3_bytes(
            DISTILLERY_BUCKET,
            MATERIALIZATION_MANIFEST_KEY,
        )
        materialization = json.loads(materialization_body)
        entries = materialization.get("models")
        if not isinstance(entries, list):
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                "materialization manifest lacks models list",
            )
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("model_id") == MODEL_ID
            and entry.get("revision") == REVISION
        ]
        if len(matches) != 1:
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                "materialization manifest lacks exactly one 72B identity",
            )
        if matches[0].get("inventory_sha256") != inventory.inventory_sha256:
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                "materialization manifest inventory hash mismatch",
            )
        if matches[0].get("object_body_sha256") != hashes:
            raise VerificationFailure(
                GateCode.S3_BODY_HASHES,
                "materialization manifest object hash map mismatch",
            )
        return S3SnapshotEvidence.seal(
            bucket=DISTILLERY_BUCKET,
            prefix=MODEL_PREFIX,
            inventory_sha256=inventory.inventory_sha256,
            object_body_sha256=hashes,
            object_sizes=sizes,
            snapshot_manifest_body_sha256=sha256_bytes(snapshot_body),
            sha256sums_body_sha256=sha256_bytes(sums_body),
            materialization_manifest_body_sha256=sha256_bytes(materialization_body),
        )

    def _tokenizer_bodies(self, model_id: str, revision: str) -> dict[str, bytes]:
        org, name = model_id.split("/", 1)
        prefix = f"models/{org}/{name}/{revision}"
        return {
            filename: self._get_s3_bytes(
                DISTILLERY_BUCKET,
                f"{prefix}/{filename}",
            )
            for filename in TOKENIZER_FILENAMES
        }

    def verify_tokenizer_compatibility(self) -> TokenizerCompatibilityEvidence:
        teacher_bodies = self._tokenizer_bodies(MODEL_ID, REVISION)
        registry = load_target_registry()
        pairs = tuple(
            verify_tokenizer_pair(
                target=target,
                teacher_bodies=teacher_bodies,
                target_bodies=self._tokenizer_bodies(
                    target.model_id,
                    target.revision,
                ),
            )
            for target in registry.targets
        )
        return seal_compatibility(pairs)

    def verify_ecr_image(self) -> EcrImageEvidence:
        bindings = load_execution_bindings()
        binding = bindings.ecr_image
        if binding is None:
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "sealed 72B-capable ECR image binding is absent",
            )
        repository = self.ecr.describe_repositories(
            registryId=binding.account_id,
            repositoryNames=[binding.repository],
        ).get("repositories", [])
        if len(repository) != 1:
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "ECR repository lookup did not return exactly one repository",
            )
        expected_uri = (
            f"{binding.account_id}.dkr.ecr.{binding.region}.amazonaws.com/{binding.repository}"
        )
        if repository[0].get("repositoryUri") != expected_uri:
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "live ECR repository URI differs from exact account/region/repository",
            )
        details = self.ecr.describe_images(
            registryId=binding.account_id,
            repositoryName=binding.repository,
            imageIds=[{"imageDigest": binding.image_digest}],
        ).get("imageDetails", [])
        if len(details) != 1 or details[0].get("imageDigest") != binding.image_digest:
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "live ECR digest differs from sealed 72B-capable image digest",
            )
        images = self.ecr.batch_get_image(
            registryId=binding.account_id,
            repositoryName=binding.repository,
            imageIds=[{"imageDigest": binding.image_digest}],
            acceptedMediaTypes=[
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            ],
        ).get("images", [])
        if (
            len(images) != 1
            or images[0].get("imageId", {}).get("imageDigest") != binding.image_digest
        ):
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "ECR batch manifest lookup did not bind the exact digest",
            )
        manifest = json.loads(images[0].get("imageManifest", ""))
        config_digest = manifest.get("config", {}).get("digest")
        if not isinstance(config_digest, str) or not config_digest.startswith("sha256:"):
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "ECR image manifest lacks a config digest",
            )
        config_url = self.ecr.get_download_url_for_layer(
            registryId=binding.account_id,
            repositoryName=binding.repository,
            layerDigest=config_digest,
        ).get("downloadUrl")
        if not isinstance(config_url, str) or not config_url.startswith("https://"):
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "ECR image config download URL is missing",
            )
        with self.open_url(config_url, timeout=60) as response:
            config_body = response.read()
        if f"sha256:{sha256_bytes(config_body)}" != config_digest:
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "ECR image config body differs from manifest digest",
            )
        config = json.loads(config_body)
        labels = config.get("config", {}).get("Labels", {})
        expected_labels = {
            "distillery.source.sha": binding.source_revision,
            "distillery.source.tree.sha256": binding.source_tree_sha256,
            "distillery.package.lock.sha256": binding.package_lock_sha256,
            "distillery.qwen72b.trainer": "experiments.qwen72b_fallback.train",
            "distillery.qwen72b.attention.backend": "sdpa_math",
            "distillery.qwen72b.flash_attention_2": "false",
        }
        if not isinstance(labels, dict) or any(
            labels.get(name) != expected for name, expected in expected_labels.items()
        ):
            raise VerificationFailure(
                GateCode.ECR_EXACT_IMAGE,
                "live ECR config labels do not prove the sealed 72B trainer",
            )
        binding_hash = content_sha256(binding.model_dump(mode="json"))
        return EcrImageEvidence.seal(
            account_id=binding.account_id,
            region=binding.region,
            repository=binding.repository,
            image_digest=binding.image_digest,
            image_uri=binding.image_uri,
            image_binding_sha256=binding_hash,
            source_revision=binding.source_revision,
            package_lock_sha256=binding.package_lock_sha256,
            source_tree_sha256=binding.source_tree_sha256,
            qwen72b_trainer_packaged=binding.qwen72b_trainer_packaged,
            attention_backend=binding.attention_backend,
            flash_attention_2_packaged=binding.flash_attention_2_packaged,
        )

    def verify_memory_probe(
        self,
        *,
        profile: Qwen72BTrainingProfile,
        image: EcrImageEvidence,
        local_policy: LocalPolicyEvidence,
    ) -> Qwen72BMemoryProbeEvidence:
        binding = load_execution_bindings().memory_probe
        if binding is None:
            raise VerificationFailure(
                GateCode.MEMORY_PROBE,
                "measured target-device QLoRA memory probe binding is absent",
            )
        bucket, key = _s3_uri_parts(binding.s3_uri)
        body = self._get_s3_bytes(bucket, key)
        if sha256_bytes(body) != binding.body_sha256:
            raise VerificationFailure(
                GateCode.MEMORY_PROBE,
                "memory probe S3 body differs from sealed binding hash",
            )
        probe = Qwen72BMemoryProbeEvidence.model_validate_json(body)
        require_measured_probe(
            probe,
            profile_sha256=profile.profile_sha256,
            model_identity_sha256=local_policy.identity.evidence_sha256,
            image_binding_sha256=image.image_binding_sha256,
            runtime_image_digest=image.image_digest,
        )
        return probe

    def verify_finance_world_data(
        self,
        profile: Qwen72BTrainingProfile,
    ) -> FinanceWorldCorpusEvidence:
        if profile.kind in {RunKind.MEMORY_PROBE, RunKind.REHEARSAL}:
            evidence = build_finance_world_targets(
                source_corpus="smoke_v2",
                per_task=6,
            )
        else:
            evidence = build_finance_world_targets(
                source_corpus="full_v2",
                per_task=None,
            )
        if len(evidence.records) != profile.train_examples:
            raise VerificationFailure(
                GateCode.FINANCE_WORLD_DATA,
                "finance-world record count differs from exact profile",
            )
        return evidence
