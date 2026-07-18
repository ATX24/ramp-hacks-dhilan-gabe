"""Closed role vocabulary for Qwen72B, TinyFable, and demo training arms."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from experiments.qwen72b_fallback.evidence import (
    REVISION_PATTERN,
    SHA256_PATTERN,
)


class ModelRole(StrEnum):
    QWEN72B_TEACHER = "qwen72b_teacher"
    QWEN72B_ADAPTED_FALLBACK = "qwen72b_adapted_finance_fallback"
    TINYFABLE_STUDENT = "tinyfable_student"


class SupervisionKind(StrEnum):
    FINANCE_WORLD_V2_LATENT_ORACLE = "finance_world_v2_latent_oracle"
    QWEN72B_HASH_BOUND_TRAJECTORY = "qwen72b_hash_bound_trajectory"


class TinyFableTier(StrEnum):
    NANO = "nano"
    CORE = "core"
    PLUS = "plus"
    LARGE_CANDIDATE = "large_candidate"


class TrainingArm(StrEnum):
    FINANCE_WORLD_V2_ORACLE_SFT = "finance_world_v2_oracle_sft"
    QWEN72B_SEQUENCE_SFT = "qwen72b_sequence_sft"


class DemoArmName(StrEnum):
    ORACLE_SFT = "demo_oracle_sft"
    SEQUENCE_KD = "demo_sequence_kd"
    LOGIT_KD = "demo_logit_kd"
    CE_ABLATION = "demo_ce_ablation"


class TrajectoryState(StrEnum):
    ABSENT = "absent"
    VERIFIED_NONEMPTY = "verified_nonempty"


class _RoleBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Qwen72BTeacherRole(_RoleBase):
    model_role: Literal[ModelRole.QWEN72B_TEACHER] = ModelRole.QWEN72B_TEACHER
    model_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    supervision_kind: Literal[SupervisionKind.QWEN72B_HASH_BOUND_TRAJECTORY] = (
        SupervisionKind.QWEN72B_HASH_BOUND_TRAJECTORY
    )
    supervised_role: Literal[ModelRole.TINYFABLE_STUDENT] = ModelRole.TINYFABLE_STUDENT
    trajectory_state: TrajectoryState
    trajectory_bundle_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _trajectory_invariant(self) -> Qwen72BTeacherRole:
        if self.trajectory_state is TrajectoryState.ABSENT:
            if self.trajectory_bundle_sha256 is not None:
                raise ValueError("absent teacher trajectories cannot carry a bundle hash")
        elif self.trajectory_bundle_sha256 is None:
            raise ValueError("verified teacher role requires a non-empty trajectory bundle")
        return self

    @property
    def ready(self) -> bool:
        return self.trajectory_state is TrajectoryState.VERIFIED_NONEMPTY


class Qwen72BAdaptedFallbackRole(_RoleBase):
    model_role: Literal[ModelRole.QWEN72B_ADAPTED_FALLBACK] = ModelRole.QWEN72B_ADAPTED_FALLBACK
    model_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    training_arm: Literal[TrainingArm.FINANCE_WORLD_V2_ORACLE_SFT] = (
        TrainingArm.FINANCE_WORLD_V2_ORACLE_SFT
    )
    supervision_kind: Literal[SupervisionKind.FINANCE_WORLD_V2_LATENT_ORACLE] = (
        SupervisionKind.FINANCE_WORLD_V2_LATENT_ORACLE
    )
    larger_teacher_model_id: None = None
    larger_teacher_revision: None = None
    is_distilled_student: Literal[False] = False


class TinyFableStudentRole(_RoleBase):
    model_role: Literal[ModelRole.TINYFABLE_STUDENT] = ModelRole.TINYFABLE_STUDENT
    tier: TinyFableTier
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=REVISION_PATTERN)
    teacher_model_id: Literal["Qwen/Qwen2.5-72B-Instruct"]
    teacher_revision: str = Field(pattern=REVISION_PATTERN)
    supervision_kind: Literal[SupervisionKind.QWEN72B_HASH_BOUND_TRAJECTORY] = (
        SupervisionKind.QWEN72B_HASH_BOUND_TRAJECTORY
    )
    trajectory_bundle_sha256: str = Field(pattern=SHA256_PATTERN)


RoleBinding = Annotated[
    Qwen72BTeacherRole | Qwen72BAdaptedFallbackRole | TinyFableStudentRole,
    Field(discriminator="model_role"),
]
ROLE_ADAPTER = TypeAdapter(RoleBinding)


def validate_role(value: object) -> RoleBinding:
    return ROLE_ADAPTER.validate_python(value)
