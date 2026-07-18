"""Operator-filled exact execution bindings; null means explicitly unavailable."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.qwen72b_fallback.evidence import (
    PREFIXED_SHA256_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    reject_placeholders,
    sha256_bytes,
)
from experiments.qwen72b_fallback.pins import (
    AWS_REGION,
    DISTILLERY_ACCOUNT_ID,
    ECR_REPOSITORY,
    EXECUTION_BINDINGS_PATH,
)

TRAINING_ROLE_ARN = "arn:aws:iam::225989358036:role/distillery-sagemaker-training"


class EcrImageBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    account_id: Literal["225989358036"] = DISTILLERY_ACCOUNT_ID
    region: Literal["us-east-1"] = AWS_REGION
    repository: Literal["distillery-training"] = ECR_REPOSITORY
    image_digest: str = Field(pattern=PREFIXED_SHA256_PATTERN)
    image_uri: str = Field(
        pattern=(
            r"^225989358036\.dkr\.ecr\.us-east-1\.amazonaws\.com/"
            r"distillery-training@sha256:[0-9a-f]{64}$"
        )
    )
    source_revision: str = Field(pattern=REVISION_PATTERN)
    package_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    source_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    qwen72b_trainer_packaged: Literal[True]
    attention_backend: Literal["sdpa_math"] = "sdpa_math"
    flash_attention_2_packaged: Literal[False] = False

    @model_validator(mode="after")
    def _uri_matches_digest(self) -> EcrImageBinding:
        if self.image_uri != (
            f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/"
            f"{self.repository}@{self.image_digest}"
        ):
            raise ValueError("ECR image URI differs from exact binding fields")
        return self


class MemoryProbeBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    s3_uri: str = Field(
        pattern=(
            r"^s3://distillery-225989358036-us-east-1/"
            r"qwen72b/evidence/memory-probe/[a-z0-9._/-]+\.json$"
        )
    )
    body_sha256: str = Field(pattern=SHA256_PATTERN)


class ExecutionBindings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.qwen72b_fallback.execution_bindings.v1"]
    ecr_image: EcrImageBinding | None
    memory_probe: MemoryProbeBinding | None
    review_packet_sha256: tuple[str, ...]
    training_role_arn: Literal["arn:aws:iam::225989358036:role/distillery-sagemaker-training"] = (
        TRAINING_ROLE_ARN
    )
    transfer_ami_id: str | None = Field(
        default=None,
        pattern=r"^ami-[0-9a-f]{17}$",
    )
    transfer_instance_profile_arn: str | None = Field(
        default=None,
        pattern=(
            r"^arn:aws:iam::225989358036:instance-profile/"
            r"distillery-qwen72b-transfer-[A-Za-z0-9+=,.@_-]+$"
        ),
    )
    transfer_security_group_id: str | None = Field(
        default=None,
        pattern=r"^sg-[0-9a-f]{17}$",
    )
    transfer_subnet_id: str | None = Field(
        default=None,
        pattern=r"^subnet-[0-9a-f]{17}$",
    )

    @model_validator(mode="after")
    def _review_and_placeholder_invariants(self) -> ExecutionBindings:
        reject_placeholders(self.model_dump(mode="json"))
        if len(self.review_packet_sha256) not in {0, 2}:
            raise ValueError("execution requires either zero pending or exactly two review packets")
        if len(set(self.review_packet_sha256)) != len(self.review_packet_sha256):
            raise ValueError("execution review packet hashes must be distinct")
        for digest in self.review_packet_sha256:
            if not __import__("re").fullmatch(SHA256_PATTERN, digest):
                raise ValueError("review packet hash must be 64 lowercase hex")
        return self

    @property
    def both_reviews_clear(self) -> bool:
        return len(self.review_packet_sha256) == 2

    @property
    def file_sha256(self) -> str:
        return sha256_bytes(EXECUTION_BINDINGS_PATH.read_bytes())


@lru_cache(maxsize=1)
def load_execution_bindings() -> ExecutionBindings:
    return ExecutionBindings.model_validate_json(EXECUTION_BINDINGS_PATH.read_bytes())
