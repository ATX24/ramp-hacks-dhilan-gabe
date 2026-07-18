"""Hash-bound Qwen72B teacher trajectories with explicit absence semantics."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.evidence import (
    REVISION_PATTERN,
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
)
from experiments.qwen72b_fallback.pins import MODEL_ID, REVISION


class TrajectoryStatus(StrEnum):
    ABSENT = "absent"
    VERIFIED_NONEMPTY = "verified_nonempty"


class TrajectoryAbsenceReason(StrEnum):
    NOT_GENERATED = "not_generated"
    NOT_VERIFIED = "not_verified"


class TeacherTrajectoryRecord(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.teacher_trajectory.v1"] = (
        "distillery.qwen72b_fallback.teacher_trajectory.v1"
    )
    source: Literal[VerificationSource.TARGET_DEVICE] = VerificationSource.TARGET_DEVICE
    teacher_model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    teacher_revision: str = Field(pattern=REVISION_PATTERN)
    teacher_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    finance_target_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_text: str = Field(min_length=1)
    response_text: str = Field(min_length=1)
    generation_config_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _exact_teacher(self) -> TeacherTrajectoryRecord:
        if self.teacher_revision != REVISION:
            raise ValueError("teacher trajectory uses the wrong Qwen72B revision")
        return self


class TeacherTrajectoryAbsent(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.teacher_state.v1"] = (
        "distillery.qwen72b_fallback.teacher_state.v1"
    )
    status: Literal[TrajectoryStatus.ABSENT] = TrajectoryStatus.ABSENT
    reason: TrajectoryAbsenceReason
    record_count: Literal[0] = 0

    @property
    def ready(self) -> bool:
        return False


class TeacherTrajectoryBundle(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.teacher_state.v1"] = (
        "distillery.qwen72b_fallback.teacher_state.v1"
    )
    status: Literal[TrajectoryStatus.VERIFIED_NONEMPTY] = TrajectoryStatus.VERIFIED_NONEMPTY
    teacher_model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    teacher_revision: str = Field(pattern=REVISION_PATTERN)
    teacher_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    records: tuple[TeacherTrajectoryRecord, ...] = Field(min_length=1)
    record_set_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _verify_bundle(self) -> TeacherTrajectoryBundle:
        if self.teacher_revision != REVISION:
            raise ValueError("teacher bundle uses the wrong Qwen72B revision")
        if any(
            record.teacher_identity_sha256 != self.teacher_identity_sha256
            for record in self.records
        ):
            raise ValueError("teacher record identity differs from bundle identity")
        expected = content_sha256([record.model_dump(mode="json") for record in self.records])
        if self.record_set_sha256 != expected:
            raise ValueError("teacher trajectory record_set_sha256 mismatch")
        return self

    @property
    def ready(self) -> bool:
        return True


TeacherTrajectoryState = Annotated[
    TeacherTrajectoryAbsent | TeacherTrajectoryBundle,
    Field(discriminator="status"),
]
TRAJECTORY_STATE_ADAPTER = TypeAdapter(TeacherTrajectoryState)


def trajectory_absent(
    reason: TrajectoryAbsenceReason = TrajectoryAbsenceReason.NOT_GENERATED,
) -> TeacherTrajectoryAbsent:
    return TeacherTrajectoryAbsent.seal(reason=reason)


def seal_trajectory_bundle(
    *,
    teacher_identity_sha256: str,
    records: tuple[TeacherTrajectoryRecord, ...],
) -> TeacherTrajectoryBundle:
    if not records:
        raise ValueError("empty teacher trajectories cannot seal as ready")
    return TeacherTrajectoryBundle.seal(
        teacher_revision=REVISION,
        teacher_identity_sha256=teacher_identity_sha256,
        records=records,
        record_set_sha256=content_sha256([record.model_dump(mode="json") for record in records]),
    )
