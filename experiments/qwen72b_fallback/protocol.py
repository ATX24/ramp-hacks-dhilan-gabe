"""Typed protocol seal for the adapted fallback and unavailable teacher path."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from experiments.qwen72b_fallback.evidence import HashBoundEvidence
from experiments.qwen72b_fallback.profile import Qwen72BTrainingProfile
from experiments.qwen72b_fallback.readiness import (
    ExecutionAction,
    ExecutionAuthorization,
)
from experiments.qwen72b_fallback.roles import (
    Qwen72BAdaptedFallbackRole,
    Qwen72BTeacherRole,
    TrajectoryState,
)
from experiments.qwen72b_fallback.trajectories import (
    TeacherTrajectoryAbsent,
    TeacherTrajectoryBundle,
    TeacherTrajectoryState,
    trajectory_absent,
)


class Qwen72BRunProtocol(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.protocol.v2"] = (
        "distillery.qwen72b_fallback.protocol.v2"
    )
    profile: Qwen72BTrainingProfile
    authorization: ExecutionAuthorization
    adapted_fallback_role: Qwen72BAdaptedFallbackRole
    teacher_role: Qwen72BTeacherRole
    teacher_trajectory_state: TeacherTrajectoryAbsent | TeacherTrajectoryBundle

    @model_validator(mode="after")
    def _role_invariants(self) -> Qwen72BRunProtocol:
        identity_hash = self.authorization.evidence_bundle.local_policy.identity.evidence_sha256
        if self.adapted_fallback_role.model_identity_sha256 != identity_hash:
            raise ValueError("adapted fallback role identity differs from authorization")
        if self.teacher_role.model_identity_sha256 != identity_hash:
            raise ValueError("teacher role identity differs from authorization")
        if self.teacher_trajectory_state.ready != self.teacher_role.ready:
            raise ValueError("teacher role readiness differs from trajectory state")
        if isinstance(self.teacher_trajectory_state, TeacherTrajectoryBundle):
            if self.teacher_trajectory_state.teacher_identity_sha256 != identity_hash:
                raise ValueError("teacher trajectory identity differs from authorization")
            if (
                self.teacher_role.trajectory_bundle_sha256
                != self.teacher_trajectory_state.evidence_sha256
            ):
                raise ValueError("teacher role trajectory hash differs from bundle")
        elif self.teacher_role.trajectory_bundle_sha256 is not None:
            raise ValueError("absent teacher state cannot carry a trajectory bundle hash")
        evidence = self.authorization.evidence_bundle
        if self.profile.profile_sha256 != evidence.target_profile_sha256:
            raise ValueError("profile/authorization binding mismatch")
        if (
            evidence.memory_probe is not None
            and evidence.memory_probe.profile_sha256 != self.profile.profile_sha256
        ):
            raise ValueError("profile/memory-probe binding mismatch")
        return self


def build_protocol(
    *,
    profile: Qwen72BTrainingProfile,
    authorization: ExecutionAuthorization,
    teacher_trajectories: TeacherTrajectoryState | None = None,
) -> Qwen72BRunProtocol:
    if authorization.action not in {
        ExecutionAction.MEMORY_PROBE,
        ExecutionAction.REHEARSAL,
        ExecutionAction.FULL,
    }:
        raise ValueError("run protocol requires a probe/rehearsal/full authorization")
    state = teacher_trajectories or trajectory_absent()
    identity_hash = authorization.evidence_bundle.local_policy.identity.evidence_sha256
    bundle_hash = state.evidence_sha256 if isinstance(state, TeacherTrajectoryBundle) else None
    teacher = Qwen72BTeacherRole(
        model_identity_sha256=identity_hash,
        trajectory_state=(
            TrajectoryState.VERIFIED_NONEMPTY
            if isinstance(state, TeacherTrajectoryBundle)
            else TrajectoryState.ABSENT
        ),
        trajectory_bundle_sha256=bundle_hash,
    )
    fallback = Qwen72BAdaptedFallbackRole(
        model_identity_sha256=identity_hash,
    )
    return Qwen72BRunProtocol.seal(
        profile=profile,
        authorization=authorization,
        adapted_fallback_role=fallback,
        teacher_role=teacher,
        teacher_trajectory_state=state,
    )
