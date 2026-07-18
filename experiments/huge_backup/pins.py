"""Operator-attested pins for huge_backup. No unset sentinels may pass."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from experiments.aws_smoke.pins import UNSET_SENTINELS, parse_digest_pinned_ecr_image
from experiments.huge_backup.memory import HugeBackupMemoryProbeEvidence
from experiments.huge_backup.profile import (
    FALLBACK_STUDENT_MODEL_ID,
    STUDENT_MODEL_ID,
    TEACHER_MODEL_ID,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REV_RE = re.compile(r"^[0-9a-f]{40}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")


def _reject_unset(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if stripped in UNSET_SENTINELS or stripped.upper() in UNSET_SENTINELS:
        raise ValueError(f"{field_name} is unset or a forbidden placeholder: {value!r}")
    return stripped


class HugeBackupEvidence(BaseModel):
    """Pins required before a real rehearsal or warm launch may proceed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["distillery.huge_backup.evidence.v1"] = (
        "distillery.huge_backup.evidence.v1"
    )
    aws_account_id: StrictStr
    aws_region: StrictStr = Field(min_length=1)
    aws_profile: StrictStr = Field(min_length=1)
    iam_role_arn: StrictStr
    artifact_s3_prefix: StrictStr
    dataset_s3_uri: StrictStr
    models_s3_uri: StrictStr
    teacher_responses_s3_uri: StrictStr
    ecr_image_uri: StrictStr
    image_digest: StrictStr
    student_model_id: Literal["Qwen/Qwen2.5-14B-Instruct"] = STUDENT_MODEL_ID
    student_revision: StrictStr
    student_model_config_sha256: StrictStr
    teacher_model_id: Literal["Qwen/Qwen2.5-32B-Instruct"] = TEACHER_MODEL_ID
    teacher_revision: StrictStr
    teacher_model_config_sha256: StrictStr
    teacher_responses_sha256: StrictStr
    student_tokenizer_sha256: StrictStr
    teacher_tokenizer_sha256: StrictStr
    package_lock_hash: StrictStr
    source_revision: StrictStr
    license_disposition: StrictStr = Field(min_length=1)
    output_use_disposition: StrictStr = Field(min_length=1)
    price_source: StrictStr
    hourly_usd: float = Field(gt=0.0)
    evidence_attested_by: StrictStr = Field(min_length=1)
    flash_attention_2_attested: bool
    memory_probe_evidence: HugeBackupMemoryProbeEvidence | None = None
    fallback_student_model_id: Literal["Qwen/Qwen2.5-7B-Instruct"] = FALLBACK_STUDENT_MODEL_ID

    @field_validator(
        "aws_account_id",
        "aws_region",
        "aws_profile",
        "iam_role_arn",
        "artifact_s3_prefix",
        "dataset_s3_uri",
        "models_s3_uri",
        "teacher_responses_s3_uri",
        "ecr_image_uri",
        "image_digest",
        "student_revision",
        "teacher_revision",
        "student_model_config_sha256",
        "teacher_model_config_sha256",
        "teacher_responses_sha256",
        "student_tokenizer_sha256",
        "teacher_tokenizer_sha256",
        "package_lock_hash",
        "source_revision",
        "license_disposition",
        "output_use_disposition",
        "price_source",
        "evidence_attested_by",
        mode="before",
    )
    @classmethod
    def _no_placeholders(cls, value: object, info: object) -> object:
        if isinstance(value, str):
            field_name = getattr(info, "field_name", "field")
            return _reject_unset(value, field_name=str(field_name))
        return value

    @model_validator(mode="after")
    def _validate_pins(self) -> HugeBackupEvidence:
        if not _ACCOUNT_RE.fullmatch(self.aws_account_id):
            raise ValueError("aws_account_id must be a 12-digit account id")
        if not _REV_RE.fullmatch(self.student_revision):
            raise ValueError("student_revision must be a 40-char git sha")
        if not _REV_RE.fullmatch(self.teacher_revision):
            raise ValueError("teacher_revision must be a 40-char git sha")
        if not _REV_RE.fullmatch(self.source_revision):
            raise ValueError("source_revision must be a 40-char git sha")
        for name in (
            "student_model_config_sha256",
            "teacher_model_config_sha256",
            "teacher_responses_sha256",
            "student_tokenizer_sha256",
            "teacher_tokenizer_sha256",
            "package_lock_hash",
        ):
            digest = getattr(self, name)
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{name} must be 64-char lowercase hex")
        if not self.image_digest.startswith("sha256:") or not _SHA256_RE.fullmatch(
            self.image_digest.removeprefix("sha256:")
        ):
            raise ValueError("image_digest must be sha256:<64 hex>")
        identity = parse_digest_pinned_ecr_image(self.ecr_image_uri)
        if identity.digest != self.image_digest:
            raise ValueError("ecr_image_uri digest must match image_digest")
        if identity.account_id != self.aws_account_id:
            raise ValueError("ecr image account must match aws_account_id")
        if self.aws_profile != "gabriel-cli":
            raise ValueError("aws_profile must be gabriel-cli")
        return self
