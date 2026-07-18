from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distillery.contracts.tasks import SplitName
from distillery.data.generate import generate_corpus
from experiments.qwen72b_fallback.finance_world_targets import (
    ALL_TASKS,
    rehearsal_corpus,
)
from experiments.qwen72b_fallback.license_policy import (
    ATTRIBUTION_PLAN_PATH,
    QWEN_NOTICE_BYTES,
    QWEN_NOTICE_PATH,
    DistributionAttribution,
    verify_license_artifacts,
)
from experiments.qwen72b_fallback.pins import (
    CHAT_TEMPLATE_SHA256,
    MODEL_CONFIG_SHA256,
    REVISION,
    SPECIAL_TOKEN_IDS,
    TOKENIZER_FILE_SHA256,
    WeightInventory,
    load_weight_inventory,
    sealed_identity,
)
from experiments.qwen72b_fallback.roles import (
    Qwen72BAdaptedFallbackRole,
    Qwen72BTeacherRole,
    TrajectoryState,
    validate_role,
)
from experiments.qwen72b_fallback.tokenizer_compat import load_target_registry
from experiments.qwen72b_fallback.trajectories import (
    TeacherTrajectoryAbsent,
    seal_trajectory_bundle,
    trajectory_absent,
)

ROOT = Path(__file__).resolve().parents[2]


def test_full_identity_inventory_and_special_tokens_are_hash_bound() -> None:
    inventory = load_weight_inventory()
    identity = sealed_identity()
    assert inventory.revision == REVISION
    assert len([name for name in inventory.files if name.endswith(".safetensors")]) == 37
    assert inventory.model_config_sha256 == MODEL_CONFIG_SHA256
    assert inventory.tokenizer_files_sha256 == TOKENIZER_FILE_SHA256
    assert inventory.chat_template_sha256 == CHAT_TEMPLATE_SHA256
    assert inventory.special_token_ids == SPECIAL_TOKEN_IDS
    assert identity.inventory_sha256 == inventory.inventory_sha256

    tampered = inventory.model_dump(mode="python")
    tampered["files"]["config.json"]["sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="canonical hash"):
        WeightInventory.model_validate(tampered)


def test_qwen_license_notice_attribution_and_mit_code_license_are_distinct() -> None:
    evidence = verify_license_artifacts(ROOT)
    plan = json.loads(ATTRIBUTION_PLAN_PATH.read_bytes())
    assert QWEN_NOTICE_PATH.read_bytes() == QWEN_NOTICE_BYTES
    assert plan["base_model_attribution"] == DistributionAttribution.BASE.value
    assert plan["derived_model_attribution"] == DistributionAttribution.DERIVED.value
    assert evidence.repo_code_license.value == "MIT"
    assert evidence.model_license.value.startswith("Qwen LICENSE AGREEMENT")
    assert evidence.model_license_body_sha256 == sealed_identity().license_file_sha256


def test_closed_roles_disambiguate_teacher_fallback_and_students() -> None:
    identity_hash = sealed_identity().evidence_sha256
    fallback = Qwen72BAdaptedFallbackRole(model_identity_sha256=identity_hash)
    teacher = Qwen72BTeacherRole(
        model_identity_sha256=identity_hash,
        trajectory_state=TrajectoryState.ABSENT,
    )
    assert fallback.is_distilled_student is False
    assert teacher.ready is False
    with pytest.raises(ValidationError):
        Qwen72BAdaptedFallbackRole(
            model_identity_sha256=identity_hash,
            is_distilled_student=True,
        )
    with pytest.raises(ValidationError):
        validate_role({"model_role": "teacher", "model_identity_sha256": identity_hash})


def test_empty_teacher_trajectories_are_explicitly_absent_and_never_ready() -> None:
    state = trajectory_absent()
    assert isinstance(state, TeacherTrajectoryAbsent)
    assert state.record_count == 0
    assert state.ready is False
    with pytest.raises(ValueError, match="empty teacher trajectories"):
        seal_trajectory_bundle(
            teacher_identity_sha256=sealed_identity().evidence_sha256,
            records=(),
        )


def test_rehearsal_targets_are_real_finance_world_v2_latent_oracles() -> None:
    corpus = rehearsal_corpus()
    assert len(corpus.records) == 24
    assert set(corpus.task_counts) == ALL_TASKS
    assert set(corpus.task_counts.values()) == {6}
    for record in corpus.records:
        assert record.generator_revision == "finance_world.v2"
        assert record.latent_state_hash == record.envelope.oracle.latent_state_hash
        assert record.envelope.schema_version == "finance_world.v2"
        assert record.envelope_sha256


def test_finance_world_v1_semantics_remain_unchanged() -> None:
    v1 = generate_corpus("smoke")
    v2 = generate_corpus("smoke_v2")
    assert {example.schema_version for split in v1.by_split.values() for example in split} == {
        "finance_world.v1"
    }
    assert {example.schema_version for split in v2.by_split.values() for example in split} == {
        "finance_world.v2"
    }
    assert v1.by_split[SplitName.TRAIN] != v2.by_split[SplitName.TRAIN]


def test_tokenizer_registry_requires_actual_pair_components_not_a_true_flag() -> None:
    raw = json.loads(
        (ROOT / "experiments" / "qwen72b_fallback" / "tokenizer_targets.json").read_bytes()
    )
    registry = load_target_registry()
    assert "compatible" not in raw
    assert "qwen_family_tokenizer_compatible" not in raw
    assert len(registry.targets) == 5
    assert registry.tokenizer_file_sha256 == TOKENIZER_FILE_SHA256
    assert registry.special_token_ids == SPECIAL_TOKEN_IDS
