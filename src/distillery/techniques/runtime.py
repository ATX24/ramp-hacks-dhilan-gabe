"""Sealed technique plan and internal adapter seam."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, Protocol, Self

from pydantic import Field, StrictBool, StrictStr, ValidationInfo, model_validator

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import Sha256Hex, content_sha256
from distillery.techniques.compatibility import (
    CompatibilityContext,
    CompatibilityDecision,
)
from distillery.techniques.descriptor import (
    ArtifactContract,
    ExecutionKind,
    TechniqueDescriptor,
)
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error
from distillery.techniques.lifecycle import TechniqueLifecycle
from distillery.training.models import TrainingLoadPlan


class LossContract(FrozenModel):
    objective: StrictStr = Field(min_length=1)
    signal: StrictStr = Field(min_length=1)
    mode: StrictStr = Field(min_length=1)
    temperature: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    kd_weight: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    hard_ce_weight: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    fields: FrozenJsonObject = Field(default_factory=dict)


class ExternalExecutionPlan(FrozenModel):
    """Plan-only identity for future external-container backend integration."""

    execution: Literal["external_container"] = "external_container"
    image_uri: StrictStr
    image_digest: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    network_isolation_required: Literal[True] = True
    channel_schema_version: Literal["distillery.technique.channel.v2"] = (
        "distillery.technique.channel.v2"
    )
    channel_plan_filename: Literal["technique_plan.json"] = "technique_plan.json"
    reviewed_source_repository: StrictStr
    reviewed_source_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    reviewed_source_tree_sha256: Sha256Hex
    reviewed_source_review_record_sha256: Sha256Hex


class ResolvedHardwarePlan(FrozenModel):
    backend_kind: Literal["local", "sagemaker"]
    instance_type: StrictStr = Field(min_length=1)
    approved_instance_types: tuple[StrictStr, ...] = Field(min_length=1)
    min_gpu_memory_gib: int = Field(ge=1)
    network_isolation_required: StrictBool
    network_isolation_claimed: StrictBool


class TechniquePlan(FrozenModel):
    """Complete resolved identity for exactly one executable plan shape."""

    schema_version: Literal["distillery.technique.plan.v2"] = "distillery.technique.plan.v2"
    technique_id: StrictStr
    version: StrictStr
    descriptor: TechniqueDescriptor
    descriptor_sha256: Sha256Hex
    resolved_config: FrozenJsonObject
    config_sha256: Sha256Hex
    environment: CompatibilityContext
    protocol_sha256: Sha256Hex
    lifecycle: TechniqueLifecycle
    lifecycle_history: tuple[TechniqueLifecycle, ...]
    compatibility: CompatibilityDecision
    hardware: ResolvedHardwarePlan
    artifact_contract: ArtifactContract
    loss: LossContract
    adapter_config: FrozenJsonObject
    training_load_plan: TrainingLoadPlan | None = None
    external_execution: ExternalExecutionPlan | None = None
    objective_fields: FrozenJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sealed_identity(self, info: ValidationInfo) -> Self:
        if (self.training_load_plan is None) == (self.external_execution is None):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "TechniquePlan must contain exactly one execution plan",
            )
        if (
            self.technique_id != self.descriptor.technique_id
            or self.version != self.descriptor.version
            or self.descriptor_sha256 != self.descriptor.descriptor_sha256
        ):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "TechniquePlan descriptor identity mismatch",
            )
        if self.config_sha256 != content_sha256(dict(self.resolved_config)):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
                "TechniquePlan resolved_config does not match config_sha256",
            )
        claims = self.environment.model_dump(mode="json")
        if (
            self.compatibility.environment_sha256 != content_sha256(claims)
            or dict(self.compatibility.negotiated_claims) != claims
        ):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
                "TechniquePlan compatibility claims do not match environment",
            )
        if self.hardware.instance_type != self.environment.instance_type:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
                "TechniquePlan hardware does not match negotiated instance",
            )
        if self.artifact_contract != self.descriptor.artifact_contract:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH,
                "TechniquePlan artifact contract differs from descriptor",
            )
        self._validate_execution_identity()
        if not info.context or not info.context.get("skip_protocol_validation", False):
            from distillery.techniques.protocol import verify_protocol_hash

            verify_protocol_hash(self)
        return self

    def _validate_execution_identity(self) -> None:
        if self.training_load_plan is not None:
            if self.descriptor.execution is not ExecutionKind.BUILTIN:
                raise_technique_error(
                    TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                    "external descriptor cannot contain builtin training_load_plan",
                )
            student = self.training_load_plan.student.ref
            if (
                student.model_id != self.environment.student_model_id
                or student.revision != self.environment.student_revision
                or student.tokenizer_sha256 != self.environment.tokenizer_sha256_student
                or student.chat_template_sha256 != self.environment.chat_template_sha256_student
            ):
                raise_technique_error(
                    TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
                    "builtin student load identity differs from environment",
                )
            teacher = self.training_load_plan.teacher
            if teacher is not None and (
                teacher.ref.model_id != self.environment.teacher_model_id
                or teacher.ref.revision != self.environment.teacher_revision
                or teacher.ref.tokenizer_sha256 != self.environment.tokenizer_sha256_teacher
                or teacher.ref.chat_template_sha256 != self.environment.chat_template_sha256_teacher
            ):
                raise_technique_error(
                    TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
                    "builtin teacher load identity differs from environment",
                )
            return
        external = self.external_execution
        if (
            external is None
            or self.descriptor.execution is not ExecutionKind.EXTERNAL_CONTAINER
            or self.descriptor.plugin_image is None
            or self.descriptor.reviewed_source is None
        ):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "external execution identity is incomplete",
            )
        source = self.descriptor.reviewed_source
        image = self.descriptor.plugin_image
        if (
            external.image_uri != image.image_uri
            or external.image_digest != image.image_digest
            or external.reviewed_source_repository != source.repository_uri
            or external.reviewed_source_commit != source.commit_sha
            or external.reviewed_source_tree_sha256 != source.source_tree_sha256
            or external.reviewed_source_review_record_sha256 != source.review_record_sha256
        ):
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "external execution image/source identity differs from descriptor",
            )

    @classmethod
    def seal(cls, **data: Any) -> TechniquePlan:
        from distillery.techniques.protocol import recompute_protocol_hash

        provisional = cls.model_validate(
            {**data, "protocol_sha256": "0" * 64},
            context={"skip_protocol_validation": True},
        )
        return cls.model_validate(
            {
                **provisional.model_dump(mode="json"),
                "protocol_sha256": recompute_protocol_hash(provisional),
            }
        )

    def plan_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))

    def validate_artifacts(self, produced: Mapping[str, str]) -> None:
        allowed = {
            *self.artifact_contract.required_outputs,
            *self.artifact_contract.optional_outputs,
        }
        names = set(produced)
        missing = sorted(set(self.artifact_contract.required_outputs) - names)
        unexpected = sorted(names - allowed)
        malformed = sorted(
            name
            for name, digest in produced.items()
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        )
        if missing or unexpected or malformed:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_ARTIFACT_CONTRACT_MISMATCH,
                "produced artifacts do not satisfy the sealed artifact contract",
                details={
                    "missing": missing,
                    "unexpected": unexpected,
                    "malformed_checksums": malformed,
                },
            )


class RuntimeAdapter(Protocol):
    descriptor: TechniqueDescriptor

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan: ...


__all__ = [
    "ExternalExecutionPlan",
    "LossContract",
    "ResolvedHardwarePlan",
    "RuntimeAdapter",
    "TechniquePlan",
]
