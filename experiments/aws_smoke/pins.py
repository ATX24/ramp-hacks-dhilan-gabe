"""Exact pin and evidence gates. No default placeholders that can pass.

Operators must supply a filled evidence JSON. Missing, empty, or sentinel
placeholder values fail loud before any CreateTrainingJob request is built.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from distillery.recipes.base import require_pinned_revision
from experiments.aws_smoke.memory import EmergencyMemoryProbeEvidence

UNSET_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "UNSET",
        "TODO",
        "REPLACE_ME",
        "PENDING",
        "placeholder",
        "PLACEHOLDER",
        "TBD",
        "xxx",
        "XXXX",
        "0" * 40,
        "0" * 64,
        "sha256:" + ("0" * 64),
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_ARN_RE = re.compile(
    r"^arn:aws:iam::(?P<account>[0-9]{12}):role/(?P<name>[A-Za-z0-9+=,.@_-]+)$"
)
_S3_URI_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9](?:/.*)?$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_ECR_IMAGE_RE = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\."
    r"(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9]+(?:[._/-][a-z0-9]+)*)@"
    r"(?P<digest>sha256:[0-9a-f]{64})$"
)


# Official AWS Hugging Face training DLC account (us-east-1 / global DLC).
AWS_HF_DLC_ACCOUNT_ID = "763104351884"
AWS_HF_DLC_TRAINING_REPOSITORY = "huggingface-pytorch-training"


@dataclass(frozen=True, slots=True)
class EcrImageIdentity:
    account_id: str
    region: str
    repository: str
    digest: str

    @property
    def is_aws_hf_training_dlc(self) -> bool:
        return (
            self.account_id == AWS_HF_DLC_ACCOUNT_ID
            and self.repository == AWS_HF_DLC_TRAINING_REPOSITORY
        )


def parse_digest_pinned_ecr_image(uri: str) -> EcrImageIdentity:
    match = _ECR_IMAGE_RE.fullmatch(uri)
    if match is None:
        raise ValueError(
            "ECR image must be a concrete account/region repository URI pinned "
            "by sha256 digest"
        )
    return EcrImageIdentity(
        account_id=match.group("account"),
        region=match.group("region"),
        repository=match.group("repository"),
        digest=match.group("digest"),
    )


def _reject_unset(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if stripped in UNSET_SENTINELS or stripped.upper() in UNSET_SENTINELS:
        raise ValueError(f"{field_name} is unset or a forbidden placeholder: {value!r}")
    return stripped


class EmergencyEvidence(BaseModel):
    """Operator-attested pins required before any AWS smoke job may launch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.aws_smoke.evidence.v1"] = (
        "distillery.aws_smoke.evidence.v1"
    )

    aws_account_id: StrictStr
    aws_region: StrictStr = Field(min_length=1)
    aws_profile: StrictStr = Field(min_length=1)
    iam_role_arn: StrictStr
    artifact_s3_prefix: StrictStr
    dataset_s3_uri: StrictStr
    models_s3_uri: StrictStr
    model_materialization_uri: StrictStr
    model_materialization_sha256: StrictStr
    ecr_image_uri: StrictStr
    image_digest: StrictStr
    student_model_id: StrictStr = Field(min_length=1)
    student_revision: StrictStr
    student_model_config_sha256: StrictStr
    teacher_model_id: StrictStr = Field(min_length=1)
    teacher_revision: StrictStr
    teacher_model_config_sha256: StrictStr
    student_tokenizer_sha256: StrictStr
    teacher_tokenizer_sha256: StrictStr
    student_chat_template_sha256: StrictStr
    teacher_chat_template_sha256: StrictStr
    student_special_token_map: dict[str, int]
    teacher_special_token_map: dict[str, int]
    package_lock_hash: StrictStr
    source_revision: StrictStr
    proof_protocol_id: StrictStr = Field(min_length=1)
    proof_protocol_sha256: StrictStr
    license_disposition: StrictStr = Field(min_length=1)
    output_use_disposition: StrictStr = Field(min_length=1)
    data_content_sha256: StrictStr
    price_source: StrictStr = Field(min_length=1)
    hourly_usd: float = Field(gt=0.0)
    evidence_attested_by: StrictStr = Field(min_length=1)
    evidence_notes: StrictStr = ""
    memory_probe_evidence: EmergencyMemoryProbeEvidence | None

    @field_validator(
        "aws_account_id",
        "aws_region",
        "aws_profile",
        "iam_role_arn",
        "artifact_s3_prefix",
        "dataset_s3_uri",
        "models_s3_uri",
        "model_materialization_uri",
        "model_materialization_sha256",
        "ecr_image_uri",
        "image_digest",
        "student_model_id",
        "student_revision",
        "student_model_config_sha256",
        "teacher_model_id",
        "teacher_revision",
        "teacher_model_config_sha256",
        "student_tokenizer_sha256",
        "teacher_tokenizer_sha256",
        "student_chat_template_sha256",
        "teacher_chat_template_sha256",
        "package_lock_hash",
        "source_revision",
        "proof_protocol_id",
        "proof_protocol_sha256",
        "license_disposition",
        "output_use_disposition",
        "data_content_sha256",
        "price_source",
        "evidence_attested_by",
        mode="before",
    )
    @classmethod
    def _no_placeholders(cls, value: Any, info: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string")
        return _reject_unset(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _cross_field_gates(self) -> EmergencyEvidence:
        if not _ACCOUNT_RE.fullmatch(self.aws_account_id):
            raise ValueError("aws_account_id must be a 12-digit account id")
        if self.aws_profile != "gabriel-cli":
            raise ValueError(
                "aws_profile must be exactly 'gabriel-cli' for this emergency path"
            )
        role = _ROLE_ARN_RE.fullmatch(self.iam_role_arn)
        if role is None:
            raise ValueError("iam_role_arn must be a concrete iam role ARN")
        if role.group("account") != self.aws_account_id:
            raise ValueError("iam_role_arn account must match aws_account_id")
        if not _S3_URI_RE.fullmatch(self.artifact_s3_prefix):
            raise ValueError("artifact_s3_prefix must be an s3:// URI")
        if not _S3_URI_RE.fullmatch(self.dataset_s3_uri):
            raise ValueError("dataset_s3_uri must be an s3:// URI")
        if not _S3_URI_RE.fullmatch(self.models_s3_uri):
            raise ValueError("models_s3_uri must be an s3:// URI")
        if not _S3_URI_RE.fullmatch(self.model_materialization_uri):
            raise ValueError("model_materialization_uri must be an s3:// URI")
        if not _SHA256_RE.fullmatch(self.model_materialization_sha256):
            raise ValueError("model_materialization_sha256 must be 64 lowercase hex")
        models = urlparse(self.models_s3_uri)
        materialization = urlparse(self.model_materialization_uri)
        if models.netloc != materialization.netloc:
            raise ValueError("model prefix and materialization manifest must share a bucket")
        if models.path.rstrip("/") != "/models":
            raise ValueError("models_s3_uri must select the exact models channel root")
        if materialization.path != "/models/materialization.json":
            raise ValueError(
                "model_materialization_uri must select models/materialization.json"
            )
        require_pinned_revision(self.student_revision, role="student")
        require_pinned_revision(self.teacher_revision, role="teacher")
        if not _SHA256_RE.fullmatch(self.student_tokenizer_sha256):
            raise ValueError("student_tokenizer_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.student_model_config_sha256):
            raise ValueError("student_model_config_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.teacher_model_config_sha256):
            raise ValueError("teacher_model_config_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.teacher_tokenizer_sha256):
            raise ValueError("teacher_tokenizer_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.student_chat_template_sha256):
            raise ValueError("student_chat_template_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.teacher_chat_template_sha256):
            raise ValueError("teacher_chat_template_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.package_lock_hash):
            raise ValueError("package_lock_hash must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.proof_protocol_sha256):
            raise ValueError("proof_protocol_sha256 must be 64 lowercase hex")
        if not _SHA256_RE.fullmatch(self.data_content_sha256):
            raise ValueError("data_content_sha256 must be 64 lowercase hex")
        if not self.image_digest.startswith("sha256:"):
            raise ValueError("image_digest must be sha256:<64-hex>")
        digest_hex = self.image_digest.removeprefix("sha256:")
        if not _SHA256_RE.fullmatch(digest_hex):
            raise ValueError("image_digest must be sha256:<64-hex>")
        identity = parse_digest_pinned_ecr_image(self.ecr_image_uri)
        if identity.digest != self.image_digest:
            raise ValueError(
                "ecr_image_uri digest does not match image_digest evidence field"
            )
        if identity.region != self.aws_region:
            raise ValueError("ecr_image_uri region does not match aws_region")
        if identity.account_id != self.aws_account_id and not identity.is_aws_hf_training_dlc:
            raise ValueError(
                "ecr_image_uri account must match aws_account_id or be the "
                "pinned AWS Hugging Face training DLC account"
            )
        if self.student_tokenizer_sha256 != self.teacher_tokenizer_sha256:
            raise ValueError(
                "logit KD requires identical teacher/student tokenizer sha256 evidence"
            )
        if self.student_chat_template_sha256 != self.teacher_chat_template_sha256:
            raise ValueError(
                "logit KD requires identical teacher/student chat_template sha256 evidence"
            )
        if not self.student_special_token_map or not self.teacher_special_token_map:
            raise ValueError("teacher/student special-token maps must be nonempty")
        if self.student_special_token_map != self.teacher_special_token_map:
            raise ValueError(
                "logit KD requires identical teacher/student special-token map evidence"
            )
        license_ok = self.license_disposition.lower()
        if "blocked" in license_ok or "unknown" in license_ok or "pending" in license_ok:
            raise ValueError(
                "license_disposition must be an explicit approved disposition, "
                f"got {self.license_disposition!r}"
            )
        use_ok = self.output_use_disposition.lower()
        if "blocked" in use_ok or "unknown" in use_ok or "pending" in use_ok:
            raise ValueError(
                "output_use_disposition must be an explicit approved disposition, "
                f"got {self.output_use_disposition!r}"
            )
        if abs(float(self.hourly_usd) - 1.408) > 1e-9:
            raise ValueError(
                "hourly_usd must equal the locked emergency price 1.408 USD/hr "
                f"(got {self.hourly_usd})"
            )
        return self


def load_evidence(path: Path) -> EmergencyEvidence:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evidence file must be a JSON object")
    return EmergencyEvidence.model_validate(raw)


def evidence_schema_template() -> dict[str, Any]:
    """Schema keys only. Values are forbidden sentinels that cannot pass gates."""
    return {
        "schema_version": "distillery.aws_smoke.evidence.v1",
        "aws_account_id": "UNSET",
        "aws_region": "UNSET",
        "aws_profile": "gabriel-cli",
        "iam_role_arn": "UNSET",
        "artifact_s3_prefix": "UNSET",
        "dataset_s3_uri": "UNSET",
        "models_s3_uri": "UNSET",
        "model_materialization_uri": "UNSET",
        "model_materialization_sha256": "UNSET",
        "ecr_image_uri": "UNSET",
        "image_digest": "UNSET",
        "student_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "student_revision": "UNSET",
        "student_model_config_sha256": "UNSET",
        "teacher_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "teacher_revision": "UNSET",
        "teacher_model_config_sha256": "UNSET",
        "student_tokenizer_sha256": "UNSET",
        "teacher_tokenizer_sha256": "UNSET",
        "student_chat_template_sha256": "UNSET",
        "teacher_chat_template_sha256": "UNSET",
        "student_special_token_map": {},
        "teacher_special_token_map": {},
        "package_lock_hash": "UNSET",
        "source_revision": "UNSET",
        "proof_protocol_id": "finance-proof.v1",
        "proof_protocol_sha256": "UNSET",
        "license_disposition": "UNSET",
        "output_use_disposition": "UNSET",
        "data_content_sha256": "UNSET",
        "price_source": "UNSET",
        "hourly_usd": 1.408,
        "evidence_attested_by": "UNSET",
        "evidence_notes": "Fill every UNSET field from measured evidence before launch.",
        "memory_probe_evidence": {
            "schema_version": "distillery.aws_smoke.memory_probe.v2",
            "passed": False,
            "precision_mode": "qlora_nf4",
            "device_type": "NVIDIA A10G",
            "peak_memory_bytes": 0,
            "capacity_memory_bytes": 0,
            "headroom_bytes": 0,
            "probe_id": "UNSET",
            "student_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "student_revision": "UNSET",
            "teacher_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "teacher_revision": "UNSET",
            "max_length": 640,
            "max_completion": 128,
            "vocab_chunk_size": 4096,
            "microbatch": 1,
            "grad_accumulation": 1,
            "runtime_image_digest": "UNSET",
            "instance_type": "ml.g5.xlarge",
        },
    }
