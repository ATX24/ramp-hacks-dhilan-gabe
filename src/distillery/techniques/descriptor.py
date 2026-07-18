"""Immutable versioned technique descriptors (BYODT core type)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, StrictStr, ValidationInfo, field_validator, model_validator

from distillery.backends.safety import parse_digest_pinned_ecr_image
from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import (
    GitCommitSha,
    PrefixedSha256,
    Sha256Hex,
    content_sha256,
)
from distillery.techniques.capabilities import (
    require_known_capabilities,
    require_known_evidence,
)
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error

TECHNIQUE_ID_PATTERN = re.compile(
    r"^(?:sequence\.v1|logit\.v1|[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)$"
)
TECHNIQUE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
DESCRIPTOR_SCHEMA_VERSION: Literal["distillery.technique.v1"] = "distillery.technique.v1"


class TeacherSignal(StrEnum):
    HARD_TARGET_SEQUENCE = "hard_target_sequence"
    FULL_LOGITS = "full_logits"
    CUSTOM = "custom"


class TokenizerConstraint(StrEnum):
    STUDENT_ONLY = "student_only"
    EXACT_MATCH = "exact_match"
    NONE = "none"


class ExecutionKind(StrEnum):
    BUILTIN = "builtin"
    EXTERNAL_CONTAINER = "external_container"


class ArtifactContract(FrozenModel):
    """Outputs a technique must emit for Distillery artifact ingestion."""

    schema_version: Literal["distillery.technique.artifacts.v1"] = (
        "distillery.technique.artifacts.v1"
    )
    required_outputs: tuple[StrictStr, ...] = Field(min_length=1)
    optional_outputs: tuple[StrictStr, ...] = ()
    checksum_manifest: Literal["SHA256SUMS"] = "SHA256SUMS"


class HardwareRequirements(FrozenModel):
    min_gpu_memory_gib: int = Field(ge=1, le=1024)
    approved_instance_types: tuple[StrictStr, ...] = Field(min_length=1)
    requires_network_isolation: bool = True


class CostModel(FrozenModel):
    estimator: Literal["runtime_hours_x_hourly"] = "runtime_hours_x_hourly"
    default_max_runtime_seconds: int = Field(ge=60, le=86_400)
    default_max_run_usd: float = Field(gt=0.0, allow_inf_nan=False)
    currency: Literal["USD"] = "USD"


class PluginImageBinding(FrozenModel):
    """Digest-pinned plugin/container identity. Tags are rejected."""

    image_uri: StrictStr = Field(min_length=1)
    image_digest: PrefixedSha256

    @model_validator(mode="after")
    def _digest_pinned_only(self) -> Self:
        if "@sha256:" not in self.image_uri:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID,
                "plugin image URI must be digest-pinned (@sha256:...), never a tag",
                details={"image_uri": self.image_uri},
            )
        try:
            identity = parse_digest_pinned_ecr_image(self.image_uri)
        except Exception as exc:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID,
                "plugin image URI must be a digest-pinned private ECR image",
                details={"image_uri": self.image_uri, "error": str(exc)},
            )
        if identity.digest != self.image_digest:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DIGEST_INVALID,
                "plugin image_uri digest does not match image_digest field",
                details={
                    "image_uri_digest": identity.digest,
                    "image_digest": self.image_digest,
                },
            )
        return self


class ReviewedSourceBinding(FrozenModel):
    """Reviewed source tree bound into the sealed descriptor."""

    repository_uri: StrictStr = Field(min_length=1)
    commit_sha: GitCommitSha
    source_tree_sha256: Sha256Hex
    review_record_sha256: Sha256Hex


class TechniqueDescriptor(FrozenModel):
    """
    Immutable versioned technique descriptor.

    Callers never mutate a sealed descriptor. Integrity is bound by
    ``descriptor_sha256`` over the canonical payload excluding that field.
    """

    schema_version: Literal["distillery.technique.v1"] = DESCRIPTOR_SCHEMA_VERSION
    technique_id: StrictStr
    version: StrictStr
    display_name: StrictStr = Field(min_length=1)
    summary: StrictStr = Field(min_length=1)
    execution: ExecutionKind
    teacher_signal: TeacherSignal
    tokenizer_constraint: TokenizerConstraint
    capabilities: tuple[StrictStr, ...] = Field(min_length=1)
    evidence_requirements: tuple[StrictStr, ...] = Field(min_length=1)
    config_schema: FrozenJsonObject
    artifact_contract: ArtifactContract
    metrics: tuple[StrictStr, ...] = Field(min_length=1)
    hardware: HardwareRequirements
    cost_model: CostModel
    plugin_image: PluginImageBinding | None = None
    reviewed_source: ReviewedSourceBinding | None = None
    descriptor_sha256: Sha256Hex

    @field_validator("technique_id")
    @classmethod
    def _technique_id_shape(cls, value: str) -> str:
        if TECHNIQUE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("technique_id must be sequence.v1, logit.v1, or dotted lowercase id")
        return value

    @field_validator("version")
    @classmethod
    def _version_shape(cls, value: str) -> str:
        if TECHNIQUE_VERSION_PATTERN.fullmatch(value) is None:
            raise ValueError("version must be MAJOR.MINOR.PATCH digits")
        return value

    @model_validator(mode="after")
    def _invariants(self, info: ValidationInfo) -> Self:
        require_known_capabilities(self.capabilities)
        require_known_evidence(self.evidence_requirements)
        _validate_config_schema_shape(self.config_schema)
        if self.execution is ExecutionKind.BUILTIN:
            if self.plugin_image is not None or self.reviewed_source is not None:
                raise ValueError("builtin techniques cannot bind plugin/source images")
            if self.technique_id not in {"sequence.v1", "logit.v1"}:
                raise ValueError("builtin execution is reserved for sequence.v1/logit.v1")
        else:
            if self.plugin_image is None or self.reviewed_source is None:
                raise ValueError(
                    "external_container techniques require plugin_image and reviewed_source"
                )
            if not self.hardware.requires_network_isolation:
                raise ValueError("external_container techniques require network isolation")
            if "network_isolated_plugin" not in self.capabilities:
                raise ValueError(
                    "external_container techniques must declare network_isolated_plugin"
                )
        if not info.context or not info.context.get("skip_descriptor_hash_validation", False):
            self.assert_integrity()
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"descriptor_sha256"})

    def assert_integrity(self) -> None:
        expected = content_sha256(self.canonical_payload())
        if self.descriptor_sha256 != expected:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "descriptor_sha256 does not match canonical technique descriptor",
                details={
                    "expected": expected,
                    "actual": self.descriptor_sha256,
                    "technique_id": self.technique_id,
                    "version": self.version,
                },
            )

    @classmethod
    def seal(cls, **data: Any) -> TechniqueDescriptor:
        """Validate fields, then bind the complete canonical descriptor hash."""
        provisional = cls.model_validate(
            {**data, "descriptor_sha256": "0" * 64},
            context={"skip_descriptor_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "descriptor_sha256": content_sha256(payload)})

    @property
    def technique_key(self) -> str:
        return f"{self.technique_id}@{self.version}"


def _validate_config_schema_shape(schema: Mapping[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValueError("config_schema.type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("config_schema.properties must be an object")
    if schema.get("additionalProperties") is not False:
        raise ValueError("config_schema.additionalProperties must be false")


__all__ = [
    "DESCRIPTOR_SCHEMA_VERSION",
    "ArtifactContract",
    "CostModel",
    "ExecutionKind",
    "HardwareRequirements",
    "PluginImageBinding",
    "ReviewedSourceBinding",
    "TeacherSignal",
    "TechniqueDescriptor",
    "TokenizerConstraint",
]
