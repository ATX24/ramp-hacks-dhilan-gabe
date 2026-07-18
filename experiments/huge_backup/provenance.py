"""Teacher-response provenance: hash-bound, no test labels, pre-timer only."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256, sha256_hex

FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "answer",
        "expected_output",
        "label",
        "oracle",
        "predicted_output",
        "target",
        "target_output",
        "test_label",
        "gold",
        "ground_truth",
    }
)

TARGET_SOURCE = "pre_materialized_teacher"
TEACHER_ROLE = "teacher"
STUDENT_ROLE = "student"


class TeacherProvenanceError(ValueError):
    pass


class TeacherResponseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    response_text: str = Field(min_length=1)
    teacher_model_id: str = Field(min_length=1)
    teacher_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_source: Literal["pre_materialized_teacher"] = TARGET_SOURCE
    model_role: Literal["teacher"] = TEACHER_ROLE
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _bind_hash(self) -> TeacherResponseRecord:
        expected = completion_record_sha256(
            example_id=self.example_id,
            prompt_text=self.prompt_text,
            response_text=self.response_text,
            teacher_model_id=self.teacher_model_id,
            teacher_revision=self.teacher_revision,
        )
        if self.record_sha256 != expected:
            raise TeacherProvenanceError(
                f"record_sha256 mismatch for {self.example_id}: "
                f"sealed={self.record_sha256} computed={expected}"
            )
        return self


def assert_no_label_keys(payload: Mapping[str, Any], *, path: str = "$") -> None:
    for key, value in payload.items():
        if key in FORBIDDEN_LABEL_KEYS:
            raise TeacherProvenanceError(
                f"forbidden label key {key!r} at {path}; teacher synthesis "
                "must not consume test/oracle labels"
            )
        if isinstance(value, Mapping):
            assert_no_label_keys(value, path=f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    assert_no_label_keys(item, path=f"{path}.{key}[{index}]")


def assert_model_role(role: str, *, expected: str) -> None:
    if role != expected:
        raise TeacherProvenanceError(f"wrong model role: expected={expected!r} actual={role!r}")


def completion_record_sha256(
    *,
    example_id: str,
    prompt_text: str,
    response_text: str,
    teacher_model_id: str,
    teacher_revision: str,
) -> str:
    return content_sha256(
        {
            "example_id": example_id,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "teacher_model_id": teacher_model_id,
            "teacher_revision": teacher_revision,
            "target_source": TARGET_SOURCE,
            "model_role": TEACHER_ROLE,
        }
    )


def build_teacher_record(
    *,
    example_id: str,
    prompt_text: str,
    response_text: str,
    teacher_model_id: str,
    teacher_revision: str,
) -> TeacherResponseRecord:
    digest = completion_record_sha256(
        example_id=example_id,
        prompt_text=prompt_text,
        response_text=response_text,
        teacher_model_id=teacher_model_id,
        teacher_revision=teacher_revision,
    )
    return TeacherResponseRecord(
        example_id=example_id,
        prompt_text=prompt_text,
        response_text=response_text,
        teacher_model_id=teacher_model_id,
        teacher_revision=teacher_revision,
        record_sha256=digest,
    )


def canonical_responses_sha256(records: Sequence[TeacherResponseRecord]) -> str:
    ordered = sorted(records, key=lambda row: row.example_id)
    return content_sha256([row.model_dump(mode="json") for row in ordered])


def load_teacher_responses(path: Path) -> list[TeacherResponseRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise TeacherProvenanceError(f"teacher responses must be a nonempty list: {path}")
    records: list[TeacherResponseRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TeacherProvenanceError(f"row {index} is not an object")
        assert_no_label_keys(row, path=f"$[{index}]")
        assert_model_role(str(row.get("model_role", "")), expected=TEACHER_ROLE)
        record = TeacherResponseRecord.model_validate(row)
        if record.example_id in seen:
            raise TeacherProvenanceError(f"duplicate example_id {record.example_id!r}")
        seen.add(record.example_id)
        records.append(record)
    return records


def write_teacher_responses(path: Path, records: Sequence[TeacherResponseRecord]) -> str:
    ordered = sorted(records, key=lambda row: row.example_id)
    payload = [row.model_dump(mode="json") for row in ordered]
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_hex(text.encode("utf-8"))
