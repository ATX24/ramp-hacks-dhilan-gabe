"""logit.v1: exact full-vocabulary forward KL with hard-CE mixing and CE ablation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from distillery.contracts.errors import DistilleryErrorCode
from distillery.contracts.hashing import content_sha256
from distillery.contracts.recipes import RecipeId
from distillery.recipes.base import (
    MaterializationReport,
    Recipe,
    RecipeContext,
    RecipeMode,
    ResponseRecord,
    raise_recipe_error,
    require_pinned_revision,
)
from distillery.recipes.sequence_v1 import SequenceV1Config, materialize_sequence_examples


class LogitV1Config(BaseModel):
    """Objective configuration for logit.v1 and its matched CE-only ablation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recipe_id: str = RecipeId.LOGIT_V1.value
    mode: RecipeMode = RecipeMode.LOGIT_KD
    temperature: float = Field(default=2.0, gt=0.0)
    kd_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    hard_ce_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vocab_chunk_size: int = Field(default=4096, ge=1)
    max_completion: int = Field(default=160, ge=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> LogitV1Config:
        finite_values = {
            "temperature": self.temperature,
            "kd_weight": self.kd_weight,
            "hard_ce_weight": self.hard_ce_weight,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        total = self.kd_weight + self.hard_ce_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"kd_weight + hard_ce_weight must equal 1.0, got {total}"
            )
        if self.mode is RecipeMode.CE_ABLATION and self.kd_weight != 0.0:
            raise ValueError("ce_ablation mode requires kd_weight == 0.0")
        if self.mode is RecipeMode.CE_ABLATION and abs(self.hard_ce_weight - 1.0) > 1e-9:
            raise ValueError("ce_ablation mode requires hard_ce_weight == 1.0")
        return self

    def as_ce_ablation(self) -> LogitV1Config:
        """Return the matched CE-only control: only the KD term is zeroed."""
        return self.model_copy(
            update={
                "mode": RecipeMode.CE_ABLATION,
                "kd_weight": 0.0,
                "hard_ce_weight": 1.0,
            }
        )


class MemoryDryRunEvidence(BaseModel):
    """Precomputed, configuration-bound evidence. This class never runs a probe."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.memory_dry_run.v2"] = (
        "distillery.memory_dry_run.v2"
    )
    passed: bool
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher_model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    student_model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    instance_type: str = Field(min_length=1)
    recipe_id: Literal["logit.v1"] = "logit.v1"
    teacher_model_id: str = Field(min_length=1)
    teacher_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    student_model_id: str = Field(min_length=1)
    student_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(ge=1)
    max_completion: int = Field(ge=1)
    vocab_chunk_size: int = Field(ge=1)
    peak_memory_bytes: int = Field(ge=1)
    capacity_memory_bytes: int = Field(ge=1)
    headroom_bytes: int = Field(ge=1)
    device_type: str = Field(min_length=1)
    probe_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_bound_memory_measurement(self) -> MemoryDryRunEvidence:
        if self.peak_memory_bytes >= self.capacity_memory_bytes:
            raise ValueError("peak memory must be below measured capacity")
        if self.headroom_bytes != (
            self.capacity_memory_bytes - self.peak_memory_bytes
        ):
            raise ValueError("headroom_bytes must equal capacity minus peak")
        if self.evidence_sha256 != memory_dry_run_evidence_sha256(
            self.model_dump(mode="json")
        ):
            raise ValueError(
                "evidence_sha256 does not bind the complete memory evidence record"
            )
        return self


def memory_dry_run_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    """Hash every memory evidence field except the self-referential digest."""
    payload = dict(evidence)
    payload.pop("evidence_sha256", None)
    return content_sha256(payload)


# Fields that may differ between logit_kd and matched ce_ablation manifests.
_OBJECTIVE_DIFF_PATHS = frozenset(
    {
        "mode",
        "kd_weight",
        "hard_ce_weight",
        "objective",
        "signal",
        "training.objective.mode",
        "training.objective.kd_weight",
        "training.objective.hard_ce_weight",
        "training.objective.objective",
        "training.objective.signal",
    }
)


def assert_tokenizer_compatible(
    *,
    student_tokenizer_sha256: str,
    teacher_tokenizer_sha256: str,
    student_chat_template_sha256: str,
    teacher_chat_template_sha256: str,
    student_special_token_map: Mapping[str, int],
    teacher_special_token_map: Mapping[str, int],
    run_id: str | None = None,
) -> None:
    """Fail loud on tokenizer / special-token / chat-template mismatch."""
    hash_inputs = {
        "student_tokenizer_sha256": student_tokenizer_sha256,
        "teacher_tokenizer_sha256": teacher_tokenizer_sha256,
        "student_chat_template_sha256": student_chat_template_sha256,
        "teacher_chat_template_sha256": teacher_chat_template_sha256,
    }
    invalid_hashes = [
        name
        for name, value in hash_inputs.items()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None
    ]
    if invalid_hashes:
        raise_recipe_error(
            DistilleryErrorCode.TOKENIZER_MISMATCH,
            "logit.v1 tokenizer/chat-template evidence must use SHA-256 hex",
            details={"invalid_fields": invalid_hashes},
            run_id=run_id,
        )
    if student_tokenizer_sha256 != teacher_tokenizer_sha256:
        raise_recipe_error(
            DistilleryErrorCode.TOKENIZER_MISMATCH,
            "logit.v1 requires exactly equal tokenizer fingerprints",
            details={
                "student_tokenizer_sha256": student_tokenizer_sha256,
                "teacher_tokenizer_sha256": teacher_tokenizer_sha256,
            },
            run_id=run_id,
        )
    student_map = dict(student_special_token_map)
    teacher_map = dict(teacher_special_token_map)
    if not student_map or not teacher_map:
        raise_recipe_error(
            DistilleryErrorCode.TOKENIZER_MISMATCH,
            "logit.v1 requires explicit nonempty teacher/student special-token maps",
            details={
                "student_special_token_evidence_present": bool(student_map),
                "teacher_special_token_evidence_present": bool(teacher_map),
                "integration_requirement": (
                    "add sealed special-token maps or fingerprints to the run manifest"
                ),
            },
            run_id=run_id,
        )
    for role, token_map in (("student", student_map), ("teacher", teacher_map)):
        invalid_entries = [
            key
            for key, value in token_map.items()
            if not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ]
        if invalid_entries:
            raise_recipe_error(
                DistilleryErrorCode.TOKENIZER_MISMATCH,
                f"logit.v1 {role} special-token map is malformed",
                details={"role": role, "invalid_keys": invalid_entries},
                run_id=run_id,
            )
    if student_map != teacher_map:
        raise_recipe_error(
            DistilleryErrorCode.TOKENIZER_MISMATCH,
            "logit.v1 requires identical special-token maps",
            details={
                "student_special_token_map": student_map,
                "teacher_special_token_map": teacher_map,
            },
            run_id=run_id,
        )
    if student_chat_template_sha256 != teacher_chat_template_sha256:
        raise_recipe_error(
            DistilleryErrorCode.CHAT_TEMPLATE_MISMATCH,
            "logit.v1 requires compatible chat-template fingerprints",
            details={
                "student_chat_template_sha256": student_chat_template_sha256,
                "teacher_chat_template_sha256": teacher_chat_template_sha256,
            },
            run_id=run_id,
        )


def validate_memory_dry_run_evidence(
    context: RecipeContext,
    config: LogitV1Config,
) -> MemoryDryRunEvidence:
    """Validate precomputed memory evidence against the exact run/model/config."""
    if context.memory_dry_run_evidence is None:
        raise_recipe_error(
            DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
            "logit.v1 requires structured passed memory dry-run evidence",
            details={
                "missing": "memory_dry_run_evidence",
                "required_manifest_field": (
                    "training.qlora.capability_evidence.memory_dry_run"
                ),
            },
            run_id=context.run_id,
        )
    try:
        evidence = MemoryDryRunEvidence.model_validate(context.memory_dry_run_evidence)
    except ValidationError as exc:
        validation_errors = [
            {
                "type": error["type"],
                "loc": list(error["loc"]),
                "msg": error["msg"],
                "input": repr(error.get("input")),
            }
            for error in exc.errors(include_url=False)
        ]
        raise_recipe_error(
            DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
            "logit.v1 memory dry-run evidence is malformed",
            details={"validation_errors": validation_errors},
            run_id=context.run_id,
        )

    expected = {
        "binding_sha256": context.capability_binding_sha256,
        "training_config_sha256": context.training_config_sha256,
        "teacher_model_config_sha256": context.teacher_model_config_sha256,
        "student_model_config_sha256": context.student_model_config_sha256,
        "length_config_sha256": context.length_config_sha256,
        "runtime_image_digest": context.runtime_image_digest,
        "instance_type": context.runtime_instance_type,
        "teacher_model_id": context.teacher_model_id,
        "teacher_revision": context.teacher_revision,
        "student_model_id": context.student_model_id,
        "student_revision": context.student_revision,
        "max_length": context.max_length,
        "max_completion": context.max_completion,
        "vocab_chunk_size": config.vocab_chunk_size,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(evidence, name)}
        for name, value in expected.items()
        if value is None or getattr(evidence, name) != value
    }
    if not evidence.passed:
        mismatches["passed"] = {"expected": True, "actual": False}
    if mismatches:
        raise_recipe_error(
            DistilleryErrorCode.MEMORY_DRY_RUN_FAILED,
            "logit.v1 memory dry-run evidence does not match the sealed configuration",
            details={"mismatches": mismatches, "probe_id": evidence.probe_id},
            run_id=context.run_id,
        )
    return evidence


def assert_frozen_teacher(
    *,
    requires_grad: bool,
    training: bool,
    has_optimizer_state: bool,
    run_id: str | None = None,
) -> None:
    """Enforce frozen-teacher invariants for white-box logit KD."""
    violations: list[str] = []
    if requires_grad:
        violations.append("requires_grad_true")
    if training:
        violations.append("teacher_in_train_mode")
    if has_optimizer_state:
        violations.append("teacher_has_optimizer_state")
    if violations:
        raise_recipe_error(
            DistilleryErrorCode.RECIPE_INCOMPATIBLE,
            "logit.v1 teacher must be frozen (eval, no grad, no optimizer)",
            details={"violations": violations},
            run_id=run_id,
        )


def compare_matched_ce_ablation_manifests(
    logit_manifest: Mapping[str, Any],
    ce_manifest: Mapping[str, Any],
    *,
    allowed_diff_paths: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """
    Compare two sealed-manifest-like dicts for matched CE ablation parity.

    Returns a tuple of violation messages. Empty means only objective fields differ.
    Nested dicts are compared recursively. Only exact dotted paths in
    ``allowed_diff_paths`` may differ; matching leaf names elsewhere do not.
    """
    allowed = (
        allowed_diff_paths
        if allowed_diff_paths is not None
        else _OBJECTIVE_DIFF_PATHS
    )
    violations: list[str] = []

    def _walk(left: Any, right: Any, path: str) -> None:
        if path in allowed:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            keys = set(left) | set(right)
            for key in sorted(keys):
                child = f"{path}.{key}" if path else key
                if key not in left:
                    if child in allowed:
                        continue
                    violations.append(f"missing_in_logit:{child}")
                    continue
                if key not in right:
                    if child in allowed:
                        continue
                    violations.append(f"missing_in_ce:{child}")
                    continue
                _walk(left[key], right[key], child)
            return
        if left != right:
            violations.append(f"mismatch:{path}:{left!r}!={right!r}")

    _walk(dict(logit_manifest), dict(ce_manifest), "")
    return tuple(violations)


def assert_matched_ce_ablation(
    logit_manifest: Mapping[str, Any],
    ce_manifest: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> None:
    violations = compare_matched_ce_ablation_manifests(logit_manifest, ce_manifest)
    if violations:
        raise_recipe_error(
            DistilleryErrorCode.RECIPE_INCOMPATIBLE,
            "ce_ablation must match logit.v1 except objective fields",
            details={"violations": list(violations)},
            run_id=run_id,
        )


class LogitV1Recipe(Recipe):
    """Implemented logit.v1 recipe (exact full-vocab forward KL + hard CE)."""

    recipe_id = RecipeId.LOGIT_V1

    def __init__(self, config: LogitV1Config | None = None) -> None:
        self.config = config or LogitV1Config()

    def validate_capabilities(self, context: RecipeContext) -> None:
        if not context.teacher_model_id or not context.teacher_revision:
            raise_recipe_error(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "logit.v1 requires a local teacher model id and pinned revision",
                details={
                    "teacher_model_id": context.teacher_model_id,
                    "teacher_revision": context.teacher_revision,
                },
                run_id=context.run_id,
            )
        require_pinned_revision(
            context.teacher_revision,
            role="teacher",
            run_id=context.run_id,
        )
        require_pinned_revision(
            context.student_revision,
            role="student",
            run_id=context.run_id,
        )
        if (
            not context.tokenizer_sha256_student
            or not context.tokenizer_sha256_teacher
            or not context.chat_template_sha256_student
            or not context.chat_template_sha256_teacher
        ):
            raise_recipe_error(
                DistilleryErrorCode.TOKENIZER_MISMATCH,
                "logit.v1 requires teacher and student tokenizer/chat-template hashes",
                run_id=context.run_id,
            )
        assert_tokenizer_compatible(
            student_tokenizer_sha256=context.tokenizer_sha256_student,
            teacher_tokenizer_sha256=context.tokenizer_sha256_teacher,
            student_chat_template_sha256=context.chat_template_sha256_student,
            teacher_chat_template_sha256=context.chat_template_sha256_teacher,
            student_special_token_map=context.special_token_map_student,
            teacher_special_token_map=context.special_token_map_teacher,
            run_id=context.run_id,
        )
        validate_memory_dry_run_evidence(context, self.config)

    def materialize(
        self,
        records: Sequence[ResponseRecord],
        *,
        context: RecipeContext,
    ) -> MaterializationReport:
        self.validate_capabilities(context)
        # Teacher-forced prefixes still require schema-valid response text for CE mixing.
        seq_cfg = SequenceV1Config(
            max_completion=self.config.max_completion,
            max_length=context.max_length,
        )
        report = materialize_sequence_examples(records, config=seq_cfg)
        if not report.accepted:
            raise_recipe_error(
                DistilleryErrorCode.UNSUPPORTED_LABEL_SOURCE,
                "logit.v1 has no usable accepted responses after validation",
                details={"rejected": len(report.rejected)},
                run_id=context.run_id,
            )
        return report.model_copy(update={"recipe_id": self.recipe_id.value})

    def objective_fields(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id.value,
            "mode": self.config.mode.value,
            "objective": (
                "hard_ce_only"
                if self.config.mode is RecipeMode.CE_ABLATION
                else "forward_kl_plus_hard_ce"
            ),
            "signal": (
                "hard_target_sequence"
                if self.config.mode is RecipeMode.CE_ABLATION
                else "full_logits"
            ),
            "temperature": self.config.temperature,
            "kd_weight": self.config.kd_weight,
            "hard_ce_weight": self.config.hard_ce_weight,
            "vocab_chunk_size": self.config.vocab_chunk_size,
        }
