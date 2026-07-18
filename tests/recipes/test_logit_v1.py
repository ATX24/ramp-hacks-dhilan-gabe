"""logit.v1 capability gates, frozen teacher, and CE-ablation manifest tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode
from distillery.recipes.base import RecipeContext, RecipeMode
from distillery.recipes.logit_v1 import (
    LogitV1Config,
    LogitV1Recipe,
    MemoryDryRunEvidence,
    assert_frozen_teacher,
    assert_matched_ce_ablation,
    assert_tokenizer_compatible,
    compare_matched_ce_ablation_manifests,
    memory_dry_run_evidence_sha256,
)

TEACHER_REVISION = "1" * 40
STUDENT_REVISION = "2" * 40
DIGEST = "a" * 64
BINDING = "b" * 64
TRAINING_CONFIG = "c" * 64
TEACHER_MODEL_CONFIG = "d" * 64
STUDENT_MODEL_CONFIG = "e" * 64
LENGTH_CONFIG = "f" * 64
IMAGE_DIGEST = "sha256:" + ("1" * 64)
INSTANCE_TYPE = "ml.g5.xlarge"


def _context(
    *,
    special_token_map_student: dict[str, int] | None = None,
    special_token_map_teacher: dict[str, int] | None = None,
    memory_overrides: dict | None = None,
) -> RecipeContext:
    memory = {
        "schema_version": "distillery.memory_dry_run.v2",
        "passed": True,
        "binding_sha256": BINDING,
        "training_config_sha256": TRAINING_CONFIG,
        "teacher_model_config_sha256": TEACHER_MODEL_CONFIG,
        "student_model_config_sha256": STUDENT_MODEL_CONFIG,
        "length_config_sha256": LENGTH_CONFIG,
        "runtime_image_digest": IMAGE_DIGEST,
        "instance_type": INSTANCE_TYPE,
        "recipe_id": "logit.v1",
        "teacher_model_id": "teacher",
        "teacher_revision": TEACHER_REVISION,
        "student_model_id": "student",
        "student_revision": STUDENT_REVISION,
        "max_length": 512,
        "max_completion": 160,
        "vocab_chunk_size": 4096,
        "peak_memory_bytes": 1024,
        "capacity_memory_bytes": 2048,
        "headroom_bytes": 1024,
        "device_type": "synthetic-a10g-profile",
        "probe_id": "probe-1",
    }
    memory.update(memory_overrides or {})
    memory["evidence_sha256"] = memory_dry_run_evidence_sha256(memory)
    memory = MemoryDryRunEvidence.model_validate(memory).model_dump(mode="json")
    return RecipeContext(
        run_id="run_logit",
        seed=17,
        max_length=512,
        max_completion=160,
        student_model_id="student",
        student_revision=STUDENT_REVISION,
        teacher_model_id="teacher",
        teacher_revision=TEACHER_REVISION,
        tokenizer_sha256_student=DIGEST,
        tokenizer_sha256_teacher=DIGEST,
        chat_template_sha256_student=DIGEST,
        chat_template_sha256_teacher=DIGEST,
        special_token_map_student=(
            {"eos": 1, "pad": 0}
            if special_token_map_student is None
            else special_token_map_student
        ),
        special_token_map_teacher=(
            {"eos": 1, "pad": 0}
            if special_token_map_teacher is None
            else special_token_map_teacher
        ),
        memory_dry_run_evidence=memory,
        capability_binding_sha256=BINDING,
        training_config_sha256=TRAINING_CONFIG,
        teacher_model_config_sha256=TEACHER_MODEL_CONFIG,
        student_model_config_sha256=STUDENT_MODEL_CONFIG,
        length_config_sha256=LENGTH_CONFIG,
        runtime_image_digest=IMAGE_DIGEST,
        runtime_instance_type=INSTANCE_TYPE,
    )


def test_logit_config_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        LogitV1Config(kd_weight=0.5, hard_ce_weight=0.4)
    with pytest.raises(ValueError, match="finite"):
        LogitV1Config(temperature=float("inf"))


def test_ce_ablation_zeros_kd_term_only() -> None:
    base = LogitV1Config(temperature=2.0, kd_weight=0.7, hard_ce_weight=0.3)
    ablation = base.as_ce_ablation()
    assert ablation.mode is RecipeMode.CE_ABLATION
    assert ablation.kd_weight == 0.0
    assert ablation.hard_ce_weight == 1.0
    assert ablation.temperature == base.temperature
    assert ablation.vocab_chunk_size == base.vocab_chunk_size


def test_tokenizer_mismatch_fails_loud() -> None:
    with pytest.raises(DistilleryError) as exc:
        assert_tokenizer_compatible(
            student_tokenizer_sha256="a" * 64,
            teacher_tokenizer_sha256="b" * 64,
            student_chat_template_sha256="c" * 64,
            teacher_chat_template_sha256="c" * 64,
            student_special_token_map={"eos": 1},
            teacher_special_token_map={"eos": 1},
        )
    assert exc.value.code is DistilleryErrorCode.TOKENIZER_MISMATCH


def test_chat_template_mismatch_fails_loud() -> None:
    with pytest.raises(DistilleryError) as exc:
        assert_tokenizer_compatible(
            student_tokenizer_sha256="a" * 64,
            teacher_tokenizer_sha256="a" * 64,
            student_chat_template_sha256="c" * 64,
            teacher_chat_template_sha256="d" * 64,
            student_special_token_map={"eos": 1},
            teacher_special_token_map={"eos": 1},
        )
    assert exc.value.code is DistilleryErrorCode.CHAT_TEMPLATE_MISMATCH


def test_special_token_map_mismatch_fails_loud() -> None:
    with pytest.raises(DistilleryError) as exc:
        assert_tokenizer_compatible(
            student_tokenizer_sha256="a" * 64,
            teacher_tokenizer_sha256="a" * 64,
            student_chat_template_sha256="c" * 64,
            teacher_chat_template_sha256="c" * 64,
            student_special_token_map={"eos": 1},
            teacher_special_token_map={"eos": 2},
        )
    assert exc.value.code is DistilleryErrorCode.TOKENIZER_MISMATCH


def test_missing_special_token_evidence_is_not_vacuously_compatible() -> None:
    with pytest.raises(DistilleryError) as exc:
        assert_tokenizer_compatible(
            student_tokenizer_sha256=DIGEST,
            teacher_tokenizer_sha256=DIGEST,
            student_chat_template_sha256=DIGEST,
            teacher_chat_template_sha256=DIGEST,
            student_special_token_map={},
            teacher_special_token_map={},
        )
    assert exc.value.code is DistilleryErrorCode.TOKENIZER_MISMATCH
    assert "nonempty" in exc.value.payload.message


def test_frozen_teacher_assertions() -> None:
    assert_frozen_teacher(
        requires_grad=False,
        training=False,
        has_optimizer_state=False,
    )
    with pytest.raises(DistilleryError) as exc:
        assert_frozen_teacher(
            requires_grad=True,
            training=True,
            has_optimizer_state=True,
        )
    assert exc.value.code is DistilleryErrorCode.RECIPE_INCOMPATIBLE
    assert "requires_grad_true" in exc.value.payload.details["violations"]


def test_matched_ce_ablation_allows_only_objective_diffs() -> None:
    logit = {
        "seed": 17,
        "max_steps": 30,
        "qlora": {"rank": 8, "alpha": 16},
        "mode": "logit_kd",
        "kd_weight": 0.7,
        "hard_ce_weight": 0.3,
        "temperature": 2.0,
    }
    ce = {
        "seed": 17,
        "max_steps": 30,
        "qlora": {"rank": 8, "alpha": 16},
        "mode": "ce_ablation",
        "kd_weight": 0.0,
        "hard_ce_weight": 1.0,
        "temperature": 2.0,
    }
    assert compare_matched_ce_ablation_manifests(logit, ce) == ()
    assert_matched_ce_ablation(logit, ce)


def test_matched_ce_ablation_detects_seed_drift() -> None:
    logit = {"seed": 17, "mode": "logit_kd", "kd_weight": 0.7, "hard_ce_weight": 0.3}
    ce = {"seed": 23, "mode": "ce_ablation", "kd_weight": 0.0, "hard_ce_weight": 1.0}
    violations = compare_matched_ce_ablation_manifests(logit, ce)
    assert any(v.startswith("mismatch:seed:") for v in violations)


def test_ablation_does_not_allow_nested_leaf_name_wildcards() -> None:
    logit = {
        "training": {
            "objective": {"kd_weight": 0.7},
            "diagnostics": {"kd_weight": 0.7},
        }
    }
    ce = {
        "training": {
            "objective": {"kd_weight": 0.0},
            "diagnostics": {"kd_weight": 0.0},
        }
    }
    violations = compare_matched_ce_ablation_manifests(logit, ce)
    assert violations == ("mismatch:training.diagnostics.kd_weight:0.7!=0.0",)


def test_logit_recipe_validate_capabilities_ok() -> None:
    recipe = LogitV1Recipe()
    recipe.validate_capabilities(_context())


def test_logit_recipe_rejects_missing_memory_evidence() -> None:
    recipe = LogitV1Recipe()
    context = replace(_context(), memory_dry_run_evidence=None)
    with pytest.raises(DistilleryError) as exc:
        recipe.validate_capabilities(context)
    assert exc.value.code is DistilleryErrorCode.MEMORY_DRY_RUN_FAILED


@pytest.mark.parametrize(
    "overrides",
    [
        {"passed": False},
        {"binding_sha256": "d" * 64},
        {"student_revision": "3" * 40},
        {"vocab_chunk_size": 2048},
    ],
)
def test_logit_recipe_rejects_failed_or_mismatched_memory_evidence(
    overrides: dict,
) -> None:
    with pytest.raises(DistilleryError) as exc:
        LogitV1Recipe().validate_capabilities(_context(memory_overrides=overrides))
    assert exc.value.code is DistilleryErrorCode.MEMORY_DRY_RUN_FAILED


def test_logit_objective_fields_differ_for_ablation() -> None:
    kd = LogitV1Recipe(LogitV1Config())
    ce = LogitV1Recipe(LogitV1Config().as_ce_ablation())
    assert kd.objective_fields()["kd_weight"] == 0.7
    assert ce.objective_fields()["kd_weight"] == 0.0
    assert ce.objective_fields()["objective"] == "hard_ce_only"
