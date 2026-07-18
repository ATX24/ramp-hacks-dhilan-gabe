"""Model loading configuration builders (no weight download or instantiation)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload
from distillery.recipes.base import require_pinned_revision
from distillery.recipes.logit_v1 import assert_frozen_teacher
from distillery.training.qlora import (
    QLoRAConfig,
    QuantizationConfigSpec,
    default_student_quantization,
    qlora_from_manifest_dict,
)


class ModelRef(BaseModel):
    """Pinned model identity. Floating ``main`` revisions are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    chat_template_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

class TeacherLoadConfig(BaseModel):
    """Frozen teacher load plan: BF16, eval, no optimizer, no grad."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: ModelRef
    dtype: Literal["bfloat16"] = "bfloat16"
    device_map: str | dict[str, Any] = "auto"
    requires_grad: Literal[False] = False
    eval_mode: Literal[True] = True
    optimizer_state_enabled: Literal[False] = False
    trust_remote_code: bool = False

    def to_from_pretrained_kwargs(self) -> dict[str, Any]:
        return {
            "pretrained_model_name_or_path": self.ref.model_id,
            "revision": self.ref.revision,
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }

    def freeze_assertions(self) -> dict[str, bool]:
        return {
            "requires_grad": self.requires_grad,
            "training": not self.eval_mode,
            "has_optimizer_state": self.optimizer_state_enabled,
        }


class StudentLoadConfig(BaseModel):
    """QLoRA student load plan: 4-bit NF4 base + LoRA adapters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: ModelRef
    quantization: QuantizationConfigSpec = Field(default_factory=default_student_quantization)
    qlora: QLoRAConfig
    gradient_checkpointing: bool = True
    device_map: str | dict[str, Any] = "auto"
    trust_remote_code: bool = False

    def to_from_pretrained_kwargs(self) -> dict[str, Any]:
        return {
            "pretrained_model_name_or_path": self.ref.model_id,
            "revision": self.ref.revision,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "quantization_config": self.quantization.to_bitsandbytes_dict(),
        }


class TrainingLoadPlan(BaseModel):
    """Paired teacher/student load plan for a sealed run (configs only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    teacher: TeacherLoadConfig | None
    student: StudentLoadConfig
    recipe: str
    seed: int


class RuntimeParameter(Protocol):
    requires_grad: bool


class RuntimeTeacher(Protocol):
    training: bool

    def requires_grad_(self, requires_grad: bool) -> Any: ...

    def eval(self) -> Any: ...

    def parameters(self) -> Iterable[RuntimeParameter]: ...


@dataclass(frozen=True, slots=True)
class FrozenTeacherRuntimeGuard:
    """Proof that a concrete teacher was frozen and checked immediately."""

    teacher_identity: int
    parameter_identities: tuple[int, ...]
    passed: bool = True


OptimizerT = TypeVar("OptimizerT")


def assert_teacher_load_plan_frozen(
    plan: TrainingLoadPlan,
    *,
    run_id: str | None = None,
) -> None:
    """Prove the pre-load teacher configuration is eval/no-grad/no-optimizer."""
    if plan.recipe != "logit.v1":
        return
    if plan.teacher is None:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "logit.v1 requires a teacher load plan",
                run_id=run_id,
            )
        )
    assert_frozen_teacher(**plan.teacher.freeze_assertions(), run_id=run_id)


def assert_runtime_teacher_frozen(
    teacher: RuntimeTeacher,
    *,
    guard: FrozenTeacherRuntimeGuard | None = None,
) -> tuple[RuntimeParameter, ...]:
    """Verify a runtime teacher immediately before optimizer construction."""
    parameters = tuple(teacher.parameters())
    violations: list[str] = []
    if teacher.training:
        violations.append("teacher_in_train_mode")
    if not parameters:
        violations.append("teacher_has_no_parameters")
    if any(parameter.requires_grad for parameter in parameters):
        violations.append("teacher_parameter_requires_grad")
    if guard is not None:
        if not guard.passed or guard.teacher_identity != id(teacher):
            violations.append("invalid_frozen_teacher_guard")
        if guard.parameter_identities != tuple(id(parameter) for parameter in parameters):
            violations.append("teacher_parameters_changed_after_guard")
    if violations:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "runtime teacher is not safely frozen",
                details={"violations": violations},
            )
        )
    return parameters


def freeze_and_assert_runtime_teacher(
    teacher: RuntimeTeacher,
) -> FrozenTeacherRuntimeGuard:
    """Set requires_grad=False/eval and return a checked guard token."""
    teacher.requires_grad_(False)
    teacher.eval()
    parameters = assert_runtime_teacher_frozen(teacher)
    return FrozenTeacherRuntimeGuard(
        teacher_identity=id(teacher),
        parameter_identities=tuple(id(parameter) for parameter in parameters),
    )


def construct_optimizer_after_teacher_guard(
    optimizer_factory: Callable[[Iterable[Any]], OptimizerT],
    student_parameters: Iterable[Any],
    *,
    teacher: RuntimeTeacher,
    teacher_guard: FrozenTeacherRuntimeGuard | None,
) -> OptimizerT:
    """Only call an optimizer factory after re-checking a valid teacher guard."""
    if teacher_guard is None:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "optimizer construction requires a frozen-teacher runtime guard",
                details={"missing": "teacher_guard"},
            )
        )
    teacher_parameters = assert_runtime_teacher_frozen(
        teacher,
        guard=teacher_guard,
    )
    student = tuple(student_parameters)
    teacher_ids = {id(parameter) for parameter in teacher_parameters}
    overlap = [id(parameter) for parameter in student if id(parameter) in teacher_ids]
    if overlap:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "student optimizer parameters include frozen teacher parameters",
                details={"overlapping_parameter_count": len(overlap)},
            )
        )
    return optimizer_factory(student)


def build_teacher_load_config(
    *,
    model_id: str,
    revision: str,
    tokenizer_sha256: str | None = None,
    chat_template_sha256: str | None = None,
) -> TeacherLoadConfig:
    require_pinned_revision(revision, role="teacher")
    return TeacherLoadConfig(
        ref=ModelRef(
            model_id=model_id,
            revision=revision,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
        )
    )


def build_student_load_config(
    *,
    model_id: str,
    revision: str,
    qlora: QLoRAConfig | dict[str, Any] | None = None,
    tokenizer_sha256: str | None = None,
    chat_template_sha256: str | None = None,
    gradient_checkpointing: bool = True,
) -> StudentLoadConfig:
    require_pinned_revision(revision, role="student")
    if qlora is None:
        qlora_cfg = qlora_from_manifest_dict({})
    elif isinstance(qlora, QLoRAConfig):
        qlora_cfg = qlora
    else:
        qlora_cfg = qlora_from_manifest_dict(qlora)
    return StudentLoadConfig(
        ref=ModelRef(
            model_id=model_id,
            revision=revision,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
        ),
        qlora=qlora_cfg,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_training_load_plan(
    *,
    recipe: str,
    seed: int,
    student_id: str,
    student_revision: str,
    teacher_id: str | None = None,
    teacher_revision: str | None = None,
    qlora: dict[str, Any] | None = None,
    student_tokenizer_sha256: str | None = None,
    teacher_tokenizer_sha256: str | None = None,
    student_chat_template_sha256: str | None = None,
    teacher_chat_template_sha256: str | None = None,
) -> TrainingLoadPlan:
    """Build load configs from manifest fields without touching the network."""
    student = build_student_load_config(
        model_id=student_id,
        revision=student_revision,
        qlora=qlora,
        tokenizer_sha256=student_tokenizer_sha256,
        chat_template_sha256=student_chat_template_sha256,
    )
    teacher: TeacherLoadConfig | None = None
    if teacher_id is not None and teacher_revision is not None:
        teacher = build_teacher_load_config(
            model_id=teacher_id,
            revision=teacher_revision,
            tokenizer_sha256=teacher_tokenizer_sha256,
            chat_template_sha256=teacher_chat_template_sha256,
        )
    elif recipe == "logit.v1":
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.CAPABILITY_UNAVAILABLE,
                "logit.v1 load plan requires teacher id and revision",
                details={"recipe": recipe},
            )
        )
    return TrainingLoadPlan(teacher=teacher, student=student, recipe=recipe, seed=seed)
