"""QLoRA and model load-config builder tests (no weight download)."""

from __future__ import annotations

import pytest

from distillery.contracts.budgets import SmokeTrainingBudget, TrainingBudget
from distillery.contracts.errors import DistilleryError, DistilleryErrorCode
from distillery.training.models import (
    FrozenTeacherRuntimeGuard,
    TeacherLoadConfig,
    TrainingLoadPlan,
    assert_teacher_load_plan_frozen,
    build_student_load_config,
    build_teacher_load_config,
    build_training_load_plan,
    construct_optimizer_after_teacher_guard,
    freeze_and_assert_runtime_teacher,
)
from distillery.training.qlora import (
    DEFAULT_LORA_TARGET_MODULES,
    QLoRAConfig,
    qlora_from_manifest_dict,
    qlora_from_smoke_budget,
    qlora_from_training_budget,
)

TEACHER_REVISION = "a" * 40
STUDENT_REVISION = "b" * 40


class _Parameter:
    def __init__(self, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad


class _Teacher:
    def __init__(self, *, obey_freeze: bool = True) -> None:
        self.training = True
        self._parameters = [_Parameter(), _Parameter()]
        self._obey_freeze = obey_freeze

    def requires_grad_(self, requires_grad: bool) -> _Teacher:
        if self._obey_freeze:
            for parameter in self._parameters:
                parameter.requires_grad = requires_grad
        return self

    def eval(self) -> _Teacher:
        if self._obey_freeze:
            self.training = False
        return self

    def parameters(self) -> list[_Parameter]:
        return self._parameters


def test_qlora_smoke_and_full_budgets() -> None:
    smoke = qlora_from_smoke_budget(SmokeTrainingBudget())
    full = qlora_from_training_budget(TrainingBudget())
    assert smoke.rank == 8
    assert smoke.alpha == 16
    assert full.rank == 16
    assert full.alpha == 32
    assert smoke.dropout == pytest.approx(0.05)
    peft = smoke.to_peft_dict()
    assert peft["r"] == 8
    assert peft["target_modules"] == list(DEFAULT_LORA_TARGET_MODULES)


def test_qlora_from_manifest_dict_defaults() -> None:
    cfg = qlora_from_manifest_dict({"rank": 4})
    assert cfg.rank == 4
    assert cfg.alpha == 16  # smoke default


@pytest.mark.parametrize(
    "values",
    [
        {"rank": "8"},
        {"alpha": "16"},
        {"dropout": "0.05"},
        {"target_modules": "q_proj"},
        {"use_rslora": "false"},
        {"modules_to_save": "lm_head"},
    ],
)
def test_qlora_manifest_parsing_rejects_coercions(values: dict) -> None:
    with pytest.raises(DistilleryError) as exc:
        qlora_from_manifest_dict(values)
    assert exc.value.code is DistilleryErrorCode.RECIPE_INCOMPATIBLE
    assert "malformed" in exc.value.payload.message


def test_qlora_direct_model_is_strict() -> None:
    with pytest.raises(ValueError):
        QLoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules="q_proj",
        )
    with pytest.raises(ValueError):
        QLoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            use_rslora="false",
        )


def test_teacher_load_config_is_frozen_bf16() -> None:
    teacher = build_teacher_load_config(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision=TEACHER_REVISION,
    )
    assert teacher.dtype == "bfloat16"
    assert teacher.requires_grad is False
    assert teacher.eval_mode is True
    asserts = teacher.freeze_assertions()
    assert asserts == {
        "requires_grad": False,
        "training": False,
        "has_optimizer_state": False,
    }


def test_student_load_config_has_nf4_quantization() -> None:
    student = build_student_load_config(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision=STUDENT_REVISION,
        qlora={"rank": 8, "alpha": 16, "dropout": 0.05},
    )
    q = student.to_from_pretrained_kwargs()["quantization_config"]
    assert q["load_in_4bit"] is True
    assert q["bnb_4bit_quant_type"] == "nf4"
    assert q["bnb_4bit_use_double_quant"] is True
    assert student.gradient_checkpointing is True


@pytest.mark.parametrize(
    "revision",
    ["refs/main", "abcdef1", "A" * 40, "a" * 39, "a" * 41],
)
def test_non_40_lowercase_hex_revision_rejected(revision: str) -> None:
    with pytest.raises(DistilleryError) as exc:
        build_student_load_config(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=revision,
        )
    assert exc.value.code is DistilleryErrorCode.MODEL_REVISION_UNPINNED


def test_logit_load_plan_requires_teacher() -> None:
    with pytest.raises(DistilleryError) as exc:
        build_training_load_plan(
            recipe="logit.v1",
            seed=17,
            student_id="Qwen/Qwen2.5-0.5B-Instruct",
            student_revision=STUDENT_REVISION,
            teacher_id=None,
            teacher_revision=None,
        )
    assert exc.value.code is DistilleryErrorCode.CAPABILITY_UNAVAILABLE


def test_sequence_load_plan_teacher_optional() -> None:
    plan = build_training_load_plan(
        recipe="sequence.v1",
        seed=17,
        student_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=STUDENT_REVISION,
    )
    assert plan.teacher is None
    assert plan.student.qlora.rank == 8


def test_runtime_teacher_guard_precedes_optimizer_factory() -> None:
    teacher = _Teacher()
    student_parameter = _Parameter()
    calls: list[tuple[_Parameter, ...]] = []

    def factory(parameters: object) -> str:
        calls.append(tuple(parameters))
        return "optimizer"

    with pytest.raises(DistilleryError, match="runtime guard"):
        construct_optimizer_after_teacher_guard(
            factory,
            [student_parameter],
            teacher=teacher,
            teacher_guard=None,
        )
    assert calls == []

    guard = freeze_and_assert_runtime_teacher(teacher)
    result = construct_optimizer_after_teacher_guard(
        factory,
        [student_parameter],
        teacher=teacher,
        teacher_guard=guard,
    )
    assert result == "optimizer"
    assert calls == [(student_parameter,)]
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_runtime_teacher_cannot_bypass_freeze_calls() -> None:
    teacher = _Teacher(obey_freeze=False)
    with pytest.raises(DistilleryError, match="safely frozen"):
        freeze_and_assert_runtime_teacher(teacher)


def test_invalid_or_stale_guard_blocks_optimizer() -> None:
    teacher = _Teacher()
    valid_guard = freeze_and_assert_runtime_teacher(teacher)
    stale_guard = FrozenTeacherRuntimeGuard(
        teacher_identity=valid_guard.teacher_identity,
        parameter_identities=(123,),
    )
    factory_called = False

    def factory(parameters: object) -> object:
        nonlocal factory_called
        factory_called = True
        return parameters

    with pytest.raises(DistilleryError, match="safely frozen"):
        construct_optimizer_after_teacher_guard(
            factory,
            [_Parameter()],
            teacher=teacher,
            teacher_guard=stale_guard,
        )
    assert factory_called is False


def test_validation_guard_catches_unsafe_teacher_config_bypass() -> None:
    safe = build_teacher_load_config(
        model_id="teacher",
        revision=TEACHER_REVISION,
    )
    unsafe = TeacherLoadConfig.model_construct(
        ref=safe.ref,
        dtype="bfloat16",
        device_map="auto",
        requires_grad=True,
        eval_mode=False,
        optimizer_state_enabled=True,
        trust_remote_code=False,
    )
    student = build_student_load_config(
        model_id="student",
        revision=STUDENT_REVISION,
    )
    bypassed_plan = TrainingLoadPlan.model_construct(
        teacher=unsafe,
        student=student,
        recipe="logit.v1",
        seed=17,
    )
    with pytest.raises(DistilleryError) as exc:
        assert_teacher_load_plan_frozen(bypassed_plan)
    assert exc.value.code is DistilleryErrorCode.RECIPE_INCOMPATIBLE
