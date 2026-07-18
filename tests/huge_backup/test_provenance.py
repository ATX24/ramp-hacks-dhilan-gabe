"""Teacher provenance, label bans, and model-role checks."""

from __future__ import annotations

import pytest

from experiments.huge_backup.profile import HugeBackupTrainingProfile
from experiments.huge_backup.provenance import (
    TeacherProvenanceError,
    assert_model_role,
    assert_no_label_keys,
    build_teacher_record,
    load_teacher_responses,
)
from tests.huge_backup.fakes import materialize_channels


def test_record_hash_binding() -> None:
    from pydantic import ValidationError

    record = build_teacher_record(
        example_id="ex-1",
        prompt_text="p",
        response_text="r",
        teacher_model_id="Qwen/Qwen2.5-32B-Instruct",
        teacher_revision="f" * 40,
    )
    with pytest.raises(ValidationError, match="record_sha256 mismatch"):
        type(record)(
            example_id=record.example_id,
            prompt_text=record.prompt_text,
            response_text=record.response_text,
            teacher_model_id=record.teacher_model_id,
            teacher_revision=record.teacher_revision,
            record_sha256="0" * 64,
        )


def test_forbids_test_labels() -> None:
    with pytest.raises(TeacherProvenanceError, match="forbidden label key"):
        assert_no_label_keys({"example_id": "x", "expected_output": {"a": 1}})


def test_wrong_model_role() -> None:
    with pytest.raises(TeacherProvenanceError, match="wrong model role"):
        assert_model_role("student", expected="teacher")


def test_bad_teacher_provenance_on_load(
    tmp_path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(
        tmp_path / "channels",
        mini_profile,
        include_label_key=True,
    )
    with pytest.raises(TeacherProvenanceError):
        load_teacher_responses(channels.teacher_responses / "teacher_responses.json")


def test_wrong_role_on_load(tmp_path, mini_profile: HugeBackupTrainingProfile) -> None:
    channels, _, _ = materialize_channels(
        tmp_path / "channels",
        mini_profile,
        wrong_role=True,
    )
    with pytest.raises(TeacherProvenanceError, match="wrong model role"):
        load_teacher_responses(channels.teacher_responses / "teacher_responses.json")
