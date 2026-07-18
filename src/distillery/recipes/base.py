"""Shared recipe abstractions for Distillery trainers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload
from distillery.contracts.hashing import content_sha256
from distillery.contracts.recipes import RecipeId
from distillery.contracts.tasks import LabelSource

PINNED_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def is_pinned_revision(revision: str | None) -> bool:
    return revision is not None and PINNED_REVISION_PATTERN.fullmatch(revision) is not None


def require_pinned_revision(
    revision: str | None,
    *,
    role: str,
    run_id: str | None = None,
) -> str:
    """Require an immutable lowercase 40-hex model/source revision."""
    if not is_pinned_revision(revision):
        raise_recipe_error(
            DistilleryErrorCode.MODEL_REVISION_UNPINNED,
            f"{role} revision must be exactly 40 lowercase hexadecimal characters",
            details={"role": role, "revision": revision},
            run_id=run_id,
        )
    return revision


class RecipeMode(StrEnum):
    """Training objective mode for a resolved recipe arm."""

    SEQUENCE_CE = "sequence_ce"
    LOGIT_KD = "logit_kd"
    CE_ABLATION = "ce_ablation"


class JointTokenizationEvidence(BaseModel):
    """Joint prompt+target tokenization with an offset-derived loss boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.joint_tokenization.v1"] = (
        "distillery.joint_tokenization.v1"
    )
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    add_special_tokens: Literal[False] = False
    boundary_method: Literal["offset_mapping_v1"] = "offset_mapping_v1"
    joint_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_ids: tuple[int, ...]
    input_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    offset_mapping: tuple[tuple[int, int], ...]
    offset_mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completion_start_index: int = Field(ge=0)
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=1)
    total_token_count: int = Field(ge=1)

    @field_validator("input_ids", mode="before")
    @classmethod
    def _strict_input_ids(cls, value: Any) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("input_ids must be a nonempty list or tuple")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in value
        ):
            raise ValueError("input_ids must contain non-negative integers")
        return tuple(value)

    @field_validator("offset_mapping", mode="before")
    @classmethod
    def _strict_offsets(cls, value: Any) -> tuple[tuple[int, int], ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("offset_mapping must be a nonempty list or tuple")
        offsets: list[tuple[int, int]] = []
        for offset in value:
            if (
                not isinstance(offset, (list, tuple))
                or len(offset) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, int)
                    for part in offset
                )
            ):
                raise ValueError("offset_mapping entries must be integer pairs")
            start, end = offset
            if start < 0 or end <= start:
                raise ValueError("offset_mapping entries must satisfy 0 <= start < end")
            offsets.append((start, end))
        return tuple(offsets)

    @model_validator(mode="after")
    def _validate_internal_binding(self) -> JointTokenizationEvidence:
        if len(self.input_ids) != len(self.offset_mapping):
            raise ValueError("input_ids and offset_mapping lengths must match")
        if any(
            current[0] < previous[0] or current[1] < previous[1]
            for previous, current in zip(
                self.offset_mapping,
                self.offset_mapping[1:],
                strict=False,
            )
        ):
            raise ValueError("offset_mapping must be monotonic")
        if self.total_token_count != len(self.input_ids):
            raise ValueError("total_token_count must equal len(input_ids)")
        if self.prompt_token_count != self.completion_start_index:
            raise ValueError(
                "prompt_token_count must equal completion_start_index"
            )
        if self.completion_token_count != (
            self.total_token_count - self.completion_start_index
        ):
            raise ValueError(
                "completion_token_count must equal total minus completion start"
            )
        if self.completion_start_index >= self.total_token_count:
            raise ValueError("joint tokenization must contain a completion token")
        if self.input_ids_sha256 != content_sha256(
            {"input_ids": list(self.input_ids)}
        ):
            raise ValueError("input_ids_sha256 does not match input_ids")
        if self.offset_mapping_sha256 != content_sha256(
            {"offset_mapping": [list(offset) for offset in self.offset_mapping]}
        ):
            raise ValueError("offset_mapping_sha256 does not match offset_mapping")
        return self


class ResponseRecord(BaseModel):
    """One training response with provenance for sequence/logit materialization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    task: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    response_text: str = Field(min_length=1)
    selected_target_text: str = Field(min_length=1)
    label_source: LabelSource
    tokenization: JointTokenizationEvidence
    teacher_model_id: str | None = None
    teacher_revision: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    imported_source_id: str | None = None
    imported_source_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    oracle_generator_revision: str | None = None
    oracle_latent_state_hash: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    rule_ids: tuple[str, ...] = ()
    transformation_lineage: tuple[str, ...] = ()
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_record(self, info: ValidationInfo) -> ResponseRecord:
        teacher_fields = (self.teacher_model_id, self.teacher_revision)
        imported_fields = (self.imported_source_id, self.imported_source_sha256)
        oracle_fields = (
            self.oracle_generator_revision,
            self.oracle_latent_state_hash,
        )

        if self.label_source is LabelSource.TEACHER:
            if (
                not all(teacher_fields)
                or self.teacher_model_id is None
                or not self.teacher_model_id.strip()
            ):
                raise ValueError(
                    "teacher provenance requires teacher_model_id and teacher_revision"
                )
            if not is_pinned_revision(self.teacher_revision):
                raise ValueError(
                    "teacher_revision must be exactly 40 lowercase hex characters"
                )
            if not self.generation_params:
                raise ValueError(
                    "teacher provenance requires nonempty generation_params"
                )
            if not self.transformation_lineage or any(
                not item.strip() for item in self.transformation_lineage
            ):
                raise ValueError(
                    "teacher provenance requires nonempty transformation_lineage"
                )
            if any(imported_fields) or any(oracle_fields) or self.rule_ids:
                raise ValueError("teacher provenance contains fields for another source")
        elif self.label_source is LabelSource.IMPORTED:
            if (
                not all(imported_fields)
                or self.imported_source_id is None
                or not self.imported_source_id.strip()
            ):
                raise ValueError(
                    "imported provenance requires imported_source_id "
                    "and imported_source_sha256"
                )
            if any(teacher_fields) or any(oracle_fields) or self.rule_ids:
                raise ValueError("imported provenance contains fields for another source")
            if self.generation_params:
                raise ValueError(
                    "imported provenance cannot contain teacher generation_params"
                )
        elif self.label_source is LabelSource.ORACLE:
            if not all(oracle_fields):
                raise ValueError(
                    "oracle provenance requires oracle_generator_revision "
                    "and oracle_latent_state_hash"
                )
            if not is_pinned_revision(self.oracle_generator_revision):
                raise ValueError(
                    "oracle_generator_revision must be exactly 40 lowercase hex characters"
                )
            if not self.rule_ids or any(
                not rule_id.strip() for rule_id in self.rule_ids
            ):
                raise ValueError("oracle provenance requires nonempty rule_ids")
            if any(teacher_fields) or any(imported_fields):
                raise ValueError("oracle provenance contains fields for another source")
            if self.generation_params:
                raise ValueError(
                    "oracle provenance cannot contain teacher generation_params"
                )
        elif self.label_source is LabelSource.RULES:
            if not self.rule_ids or any(
                not rule_id.strip() for rule_id in self.rule_ids
            ):
                raise ValueError("rules provenance requires nonempty rule_ids")
            if any(teacher_fields) or any(imported_fields) or any(oracle_fields):
                raise ValueError("rules provenance contains fields for another source")
            if self.generation_params:
                raise ValueError(
                    "rules provenance cannot contain teacher generation_params"
                )
        else:
            raise ValueError(f"unsupported label source {self.label_source!r}")

        if (
            self.selected_target_text != self.response_text
            and not self.transformation_lineage
        ):
            raise ValueError(
                "selected target changes require transformation_lineage"
            )
        joint_text = self.prompt_text + self.selected_target_text
        if self.tokenization.joint_text_sha256 != content_sha256(joint_text):
            raise ValueError(
                "joint tokenization hash does not match canonical prompt+target"
            )
        boundary = len(self.prompt_text)
        offsets = self.tokenization.offset_mapping
        if any(end > len(joint_text) for _, end in offsets):
            raise ValueError("token offsets exceed canonical joint text")
        expected_completion_start = next(
            (
                index
                for index, (_start, end) in enumerate(offsets)
                if end > boundary
            ),
            len(offsets),
        )
        if expected_completion_start != self.tokenization.completion_start_index:
            raise ValueError(
                "completion boundary does not match joint token offset mapping"
            )
        if expected_completion_start == len(offsets):
            raise ValueError("joint tokenization has no selected-target token")
        if (
            not info.context
            or not info.context.get("skip_record_hash_validation", False)
        ):
            self.assert_integrity()
        return self

    @property
    def completion_token_count(self) -> int:
        return self.tokenization.completion_token_count

    @property
    def completion_token_count_source(self) -> Literal["student_tokenizer"]:
        return "student_tokenizer"

    @property
    def completion_tokenizer_sha256(self) -> str:
        return self.tokenization.tokenizer_sha256

    @property
    def prompt_token_count(self) -> int:
        return self.tokenization.prompt_token_count

    @property
    def total_token_count(self) -> int:
        return self.tokenization.total_token_count

    def provenance_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        return {
            "label_source": self.label_source.value,
            "teacher_model_id": payload["teacher_model_id"],
            "teacher_revision": payload["teacher_revision"],
            "generation_params": payload["generation_params"],
            "imported_source_id": payload["imported_source_id"],
            "imported_source_sha256": payload["imported_source_sha256"],
            "oracle_generator_revision": payload["oracle_generator_revision"],
            "oracle_latent_state_hash": payload["oracle_latent_state_hash"],
            "rule_ids": payload["rule_ids"],
            "transformation_lineage": payload["transformation_lineage"],
        }

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"record_sha256"})

    def assert_integrity(self) -> None:
        expected_hash = content_sha256(self.canonical_payload())
        if self.record_sha256 != expected_hash:
            raise ValueError(
                "record_sha256 does not match canonical content and provenance"
            )

    @classmethod
    def seal(cls, **data: Any) -> ResponseRecord:
        """Validate fields, apply defaults, then bind the complete canonical record."""
        provisional = cls.model_validate(
            {**data, "record_sha256": "0" * 64},
            context={"skip_record_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate(
            {**payload, "record_sha256": content_sha256(payload)}
        )


class MaterializedExample(BaseModel):
    """Validated prompt/response pair ready for tokenization and masking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    task: str
    difficulty: str
    prompt_text: str
    response_text: str
    source_response_text: str
    selected_target_text: str
    label_source: LabelSource
    tokenization: JointTokenizationEvidence
    provenance: dict[str, Any]
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rejected: bool = False
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def completion_token_count(self) -> int:
        return self.tokenization.completion_token_count

    @property
    def completion_token_count_source(self) -> Literal["student_tokenizer"]:
        return "student_tokenizer"

    @property
    def completion_tokenizer_sha256(self) -> str:
        return self.tokenization.tokenizer_sha256

    @property
    def total_token_count(self) -> int:
        return self.tokenization.total_token_count


class MaterializationReport(BaseModel):
    """Deterministic validation/rejection summary for a materialization pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: tuple[MaterializedExample, ...]
    rejected: tuple[MaterializedExample, ...]
    label_source_counts: dict[str, int]
    recipe_id: str


class CompletionMask(BaseModel):
    """Token-level loss mask: 1 on completion positions, 0 on prompt/pad."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[float, ...]
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class RecipeContext:
    """Runtime context shared by recipe validate/prepare steps (no model loads)."""

    run_id: str
    seed: int
    max_length: int
    max_completion: int
    student_model_id: str
    student_revision: str
    teacher_model_id: str | None = None
    teacher_revision: str | None = None
    tokenizer_sha256_student: str | None = None
    tokenizer_sha256_teacher: str | None = None
    chat_template_sha256_student: str | None = None
    chat_template_sha256_teacher: str | None = None
    special_token_map_student: Mapping[str, int] = field(default_factory=dict)
    special_token_map_teacher: Mapping[str, int] = field(default_factory=dict)
    memory_dry_run_evidence: Mapping[str, Any] | None = None
    capability_binding_sha256: str | None = None
    training_config_sha256: str | None = None
    teacher_model_config_sha256: str | None = None
    student_model_config_sha256: str | None = None
    length_config_sha256: str | None = None
    runtime_image_digest: str | None = None
    runtime_instance_type: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class Recipe(ABC):
    """Versioned recipe interface. Implementations must not silently downgrade."""

    recipe_id: RecipeId

    @abstractmethod
    def validate_capabilities(self, context: RecipeContext) -> None:
        """Raise DistilleryError on hard-gate failure."""

    @abstractmethod
    def materialize(
        self,
        records: Sequence[ResponseRecord],
        *,
        context: RecipeContext,
    ) -> MaterializationReport:
        """Validate and materialize training examples."""

    @abstractmethod
    def objective_fields(self) -> dict[str, Any]:
        """Fields that uniquely identify the training objective for this recipe/mode."""


def raise_recipe_error(
    code: DistilleryErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    raise DistilleryError(
        ErrorPayload.from_code(code, message, details=details, run_id=run_id)
    )


def count_label_sources(examples: Sequence[MaterializedExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        key = example.label_source.value
        counts[key] = counts.get(key, 0) + 1
    return counts
