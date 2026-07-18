"""Built-in adapters wrapping sequence.v1 / logit.v1 recipe contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from distillery.recipes.base import RecipeMode
from distillery.recipes.logit_v1 import LogitV1Config
from distillery.recipes.sequence_v1 import SequenceV1Config
from distillery.techniques.compatibility import (
    CompatibilityContext,
    negotiate_compatibility,
)
from distillery.techniques.descriptor import TechniqueDescriptor
from distillery.techniques.errors import TechniqueErrorCode, raise_technique_error
from distillery.techniques.lifecycle import TechniqueLifecycle
from distillery.techniques.protocol import assert_protocol_deterministic
from distillery.techniques.runtime import LossContract, TechniquePlan
from distillery.techniques.schema import validate_config_against_schema
from distillery.training.models import (
    ModelRef,
    StudentLoadConfig,
    TeacherLoadConfig,
    TrainingLoadPlan,
)
from distillery.training.qlora import qlora_from_smoke_budget


def _require_config_str(config: Mapping[str, Any], key: str, fallback: str) -> str:
    value = config.get(key, fallback)
    if not isinstance(value, str) or not value.strip():
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
            f"config.{key} must be a nonempty string",
            details={"key": key},
        )
    return value


def _require_revision(config: Mapping[str, Any], key: str, fallback: str) -> str:
    value = _require_config_str(config, key, fallback)
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise_technique_error(
            TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
            f"config.{key} must be a 40-char lowercase git revision",
            details={"key": key, "value": value},
        )
    return value


class BuiltinSequenceAdapter:
    """Adapter: sequence.v1 technique → existing CE training plan/loss contract."""

    def __init__(self, descriptor: TechniqueDescriptor) -> None:
        if descriptor.technique_id != "sequence.v1":
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "BuiltinSequenceAdapter requires sequence.v1 descriptor",
                details={"technique_id": descriptor.technique_id},
            )
        self.descriptor = descriptor

    def validate(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> str:
        del context
        return validate_config_against_schema(
            config,
            self.descriptor.config_schema,
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
        )

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        config_sha256 = self.validate(config, context)
        compatibility = negotiate_compatibility(self.descriptor, context)
        protocol_sha256 = assert_protocol_deterministic(
            descriptor=self.descriptor,
            config_sha256=config_sha256,
            compatibility=compatibility,
        )
        student_id = _require_config_str(config, "student_model_id", context.student_model_id)
        student_revision = _require_revision(config, "student_revision", context.student_revision)
        # Bind length bounds into the recipe config surface for parity checks.
        SequenceV1Config(
            max_length=int(config["max_length"]),
            max_completion=int(config["max_completion"]),
            require_json_object_response=bool(config.get("require_json_object_response", True)),
        )
        objective = {
            "recipe_id": "sequence.v1",
            "mode": RecipeMode.SEQUENCE_CE.value,
            "objective": "ce",
            "signal": "hard_target_sequence",
        }
        loss = LossContract(
            objective="ce",
            signal="hard_target_sequence",
            mode=RecipeMode.SEQUENCE_CE.value,
            fields=objective,
        )
        load_plan = TrainingLoadPlan(
            teacher=None,
            student=StudentLoadConfig(
                ref=ModelRef(
                    model_id=student_id,
                    revision=student_revision,
                    tokenizer_sha256=context.tokenizer_sha256_student,
                ),
                qlora=qlora_from_smoke_budget(),
            ),
            recipe="sequence.v1",
            seed=int(config["seed"]),
        )
        return TechniquePlan(
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
            descriptor_sha256=self.descriptor.descriptor_sha256,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            lifecycle=TechniqueLifecycle.PLANNED,
            compatibility=compatibility,
            loss=loss,
            training_load_plan=load_plan,
            external_execution=None,
            objective_fields=objective,
            channel_contract=None,
        )


class BuiltinLogitAdapter:
    """Adapter: logit.v1 technique → existing KD+CE training plan/loss contract."""

    def __init__(self, descriptor: TechniqueDescriptor) -> None:
        if descriptor.technique_id != "logit.v1":
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_DESCRIPTOR_INVALID,
                "BuiltinLogitAdapter requires logit.v1 descriptor",
                details={"technique_id": descriptor.technique_id},
            )
        self.descriptor = descriptor

    def validate(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> str:
        del context
        config_sha256 = validate_config_against_schema(
            config,
            self.descriptor.config_schema,
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
        )
        # Reuse LogitV1Config weight invariants for parity with the recipe seam.
        try:
            LogitV1Config(
                temperature=float(config["temperature"]),
                kd_weight=float(config["kd_weight"]),
                hard_ce_weight=float(config["hard_ce_weight"]),
                max_completion=int(config["max_completion"]),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_CONFIG_MISMATCH,
                "logit.v1 config failed recipe invariant checks",
                details={"error": str(exc)},
            )
        return config_sha256

    def plan(
        self,
        config: Mapping[str, Any],
        context: CompatibilityContext,
    ) -> TechniquePlan:
        config_sha256 = self.validate(config, context)
        compatibility = negotiate_compatibility(self.descriptor, context)
        protocol_sha256 = assert_protocol_deterministic(
            descriptor=self.descriptor,
            config_sha256=config_sha256,
            compatibility=compatibility,
        )
        if context.teacher_model_id is None or context.teacher_revision is None:
            raise_technique_error(
                TechniqueErrorCode.TECHNIQUE_INCOMPATIBLE,
                "logit.v1 plan requires teacher model identity",
                run_id=context.run_id,
            )
        student_id = _require_config_str(config, "student_model_id", context.student_model_id)
        student_revision = _require_revision(config, "student_revision", context.student_revision)
        teacher_id = _require_config_str(config, "teacher_model_id", context.teacher_model_id)
        teacher_revision = _require_revision(config, "teacher_revision", context.teacher_revision)
        temperature = float(config["temperature"])
        kd_weight = float(config["kd_weight"])
        hard_ce_weight = float(config["hard_ce_weight"])
        objective = {
            "recipe_id": "logit.v1",
            "mode": RecipeMode.LOGIT_KD.value,
            "objective": "forward_kl_plus_hard_ce",
            "signal": "full_logits",
            "temperature": temperature,
            "kd_weight": kd_weight,
            "hard_ce_weight": hard_ce_weight,
        }
        loss = LossContract(
            objective="forward_kl_plus_hard_ce",
            signal="full_logits",
            mode=RecipeMode.LOGIT_KD.value,
            temperature=temperature,
            kd_weight=kd_weight,
            hard_ce_weight=hard_ce_weight,
            fields=objective,
        )
        load_plan = TrainingLoadPlan(
            teacher=TeacherLoadConfig(
                ref=ModelRef(
                    model_id=teacher_id,
                    revision=teacher_revision,
                    tokenizer_sha256=context.tokenizer_sha256_teacher,
                    chat_template_sha256=context.chat_template_sha256_teacher,
                )
            ),
            student=StudentLoadConfig(
                ref=ModelRef(
                    model_id=student_id,
                    revision=student_revision,
                    tokenizer_sha256=context.tokenizer_sha256_student,
                    chat_template_sha256=context.chat_template_sha256_student,
                ),
                qlora=qlora_from_smoke_budget(),
            ),
            recipe="logit.v1",
            seed=int(config["seed"]),
        )
        return TechniquePlan(
            technique_id=self.descriptor.technique_id,
            version=self.descriptor.version,
            descriptor_sha256=self.descriptor.descriptor_sha256,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            lifecycle=TechniqueLifecycle.PLANNED,
            compatibility=compatibility,
            loss=loss,
            training_load_plan=load_plan,
            external_execution=None,
            objective_fields=objective,
            channel_contract=None,
        )


__all__ = ["BuiltinLogitAdapter", "BuiltinSequenceAdapter"]
