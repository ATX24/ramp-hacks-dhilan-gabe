"""Built-in adapters to the existing sequence/logit training contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from distillery.recipes.logit_v1 import LogitV1Config, LogitV1Recipe
from distillery.recipes.sequence_v1 import SequenceV1Config, SequenceV1Recipe
from distillery.techniques.compatibility import (
    CompatibilityContext,
    negotiate_compatibility,
)
from distillery.techniques.descriptor import TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error
from distillery.techniques.lifecycle import TechniqueLifecycle
from distillery.techniques.runtime import (
    LossContract,
    ResolvedHardwarePlan,
    TechniquePlan,
)
from distillery.techniques.schema import validate_config_against_schema
from distillery.training.models import (
    ModelRef,
    StudentLoadConfig,
    TeacherLoadConfig,
    TrainingLoadPlan,
)
from distillery.training.qlora import qlora_from_smoke_budget

_LIFECYCLE = (
    TechniqueLifecycle.REGISTERED,
    TechniqueLifecycle.COMPATIBLE,
    TechniqueLifecycle.PLANNED,
)


def _hardware(
    descriptor: TechniqueDescriptor,
    context: CompatibilityContext,
) -> ResolvedHardwarePlan:
    return ResolvedHardwarePlan(
        backend_kind=context.backend_kind,
        instance_type=context.instance_type,
        approved_instance_types=descriptor.hardware.approved_instance_types,
        min_gpu_memory_gib=descriptor.hardware.min_gpu_memory_gib,
        network_isolation_required=descriptor.hardware.requires_network_isolation,
        network_isolation_claimed=context.network_isolation,
    )


def _assert_config_identity(
    config: Mapping[str, Any],
    context: CompatibilityContext,
    *,
    teacher: bool,
) -> None:
    comparisons = {
        "student_model_id": context.student_model_id,
        "student_revision": context.student_revision,
        "student_tokenizer_sha256": context.tokenizer_sha256_student,
        "student_chat_template_sha256": context.chat_template_sha256_student,
    }
    if teacher:
        comparisons.update(
            {
                "teacher_model_id": context.teacher_model_id,
                "teacher_revision": context.teacher_revision,
                "teacher_tokenizer_sha256": context.tokenizer_sha256_teacher,
                "teacher_chat_template_sha256": context.chat_template_sha256_teacher,
            }
        )
    mismatches = sorted(key for key, expected in comparisons.items() if config.get(key) != expected)
    if mismatches:
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
            "schema-bound model identity differs from compatibility context",
            details={"mismatched_fields": mismatches},
            run_id=context.run_id,
        )


class BuiltinSequenceAdapter:
    def __init__(self, descriptor: TechniqueDescriptor) -> None:
        if descriptor.technique_id != "sequence.v1":
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "BuiltinSequenceAdapter requires sequence.v1 descriptor",
            )
        self.descriptor = descriptor

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        resolved, config_sha256 = validate_config_against_schema(
            config,
            self.descriptor.config_schema,
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
        )
        _assert_config_identity(resolved, context, teacher=False)
        compatibility = negotiate_compatibility(self.descriptor, context)
        try:
            recipe_config = SequenceV1Config(
                max_length=resolved["max_length"],
                max_completion=resolved["max_completion"],
                require_nonempty_response=resolved["require_nonempty_response"],
                require_json_object_response=resolved["require_json_object_response"],
                pad_token_id=resolved["pad_token_id"],
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
                "sequence.v1 config failed recipe invariant checks",
                details={"error": str(exc)},
            )
        objective = SequenceV1Recipe(recipe_config).objective_fields()
        load_plan = TrainingLoadPlan(
            teacher=None,
            student=StudentLoadConfig(
                ref=ModelRef(
                    model_id=resolved["student_model_id"],
                    revision=resolved["student_revision"],
                    tokenizer_sha256=resolved["student_tokenizer_sha256"],
                    chat_template_sha256=resolved["student_chat_template_sha256"],
                ),
                qlora=qlora_from_smoke_budget(),
            ),
            recipe="sequence.v1",
            seed=resolved["seed"],
        )
        return TechniquePlan.seal(
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
            descriptor=self.descriptor,
            descriptor_sha256=self.descriptor.descriptor_sha256,
            resolved_config=resolved,
            config_sha256=config_sha256,
            environment=context,
            lifecycle=TechniqueLifecycle.PLANNED,
            lifecycle_history=_LIFECYCLE,
            compatibility=compatibility,
            hardware=_hardware(self.descriptor, context),
            artifact_contract=self.descriptor.artifact_contract,
            loss=LossContract(
                objective=objective["objective"],
                signal=objective["signal"],
                mode=objective["mode"],
                fields=objective,
            ),
            adapter_config=recipe_config.model_dump(mode="json"),
            training_load_plan=load_plan,
            external_execution=None,
            objective_fields=objective,
        )


class BuiltinLogitAdapter:
    def __init__(self, descriptor: TechniqueDescriptor) -> None:
        if descriptor.technique_id != "logit.v1":
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "BuiltinLogitAdapter requires logit.v1 descriptor",
            )
        self.descriptor = descriptor

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        resolved, config_sha256 = validate_config_against_schema(
            config,
            self.descriptor.config_schema,
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
        )
        _assert_config_identity(resolved, context, teacher=True)
        compatibility = negotiate_compatibility(self.descriptor, context)
        try:
            recipe_config = LogitV1Config(
                temperature=resolved["temperature"],
                kd_weight=resolved["kd_weight"],
                hard_ce_weight=resolved["hard_ce_weight"],
                vocab_chunk_size=resolved["vocab_chunk_size"],
                max_completion=resolved["max_completion"],
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
                "logit.v1 config failed recipe invariant checks",
                details={"error": str(exc)},
            )
        objective = LogitV1Recipe(recipe_config).objective_fields()
        load_plan = TrainingLoadPlan(
            teacher=TeacherLoadConfig(
                ref=ModelRef(
                    model_id=resolved["teacher_model_id"],
                    revision=resolved["teacher_revision"],
                    tokenizer_sha256=resolved["teacher_tokenizer_sha256"],
                    chat_template_sha256=resolved["teacher_chat_template_sha256"],
                )
            ),
            student=StudentLoadConfig(
                ref=ModelRef(
                    model_id=resolved["student_model_id"],
                    revision=resolved["student_revision"],
                    tokenizer_sha256=resolved["student_tokenizer_sha256"],
                    chat_template_sha256=resolved["student_chat_template_sha256"],
                ),
                qlora=qlora_from_smoke_budget(),
            ),
            recipe="logit.v1",
            seed=resolved["seed"],
        )
        return TechniquePlan.seal(
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
            descriptor=self.descriptor,
            descriptor_sha256=self.descriptor.descriptor_sha256,
            resolved_config=resolved,
            config_sha256=config_sha256,
            environment=context,
            lifecycle=TechniqueLifecycle.PLANNED,
            lifecycle_history=_LIFECYCLE,
            compatibility=compatibility,
            hardware=_hardware(self.descriptor, context),
            artifact_contract=self.descriptor.artifact_contract,
            loss=LossContract(
                objective=objective["objective"],
                signal=objective["signal"],
                mode=objective["mode"],
                temperature=objective["temperature"],
                kd_weight=objective["kd_weight"],
                hard_ce_weight=objective["hard_ce_weight"],
                fields=objective,
            ),
            adapter_config={
                **recipe_config.model_dump(mode="json"),
                "max_length": resolved["max_length"],
            },
            training_load_plan=load_plan,
            external_execution=None,
            objective_fields=objective,
        )


__all__ = ["BuiltinLogitAdapter", "BuiltinSequenceAdapter"]
