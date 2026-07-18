"""Monotonic DistillationRun state machine."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload


class RunState(StrEnum):
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    SYNTHESIZING = "SYNTHESIZING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    FINALIZING = "FINALIZING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES: frozenset[RunState] = frozenset(
    {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
)

# Monotonic forward edges. Terminal states admit no further transitions.
# SYNTHESIZING may be skipped: STARTING -> TRAINING with skip_reason recorded on the run.
# QUEUED -> SUCCEEDED is reserved by DistillationRun for do_not_distill reports.
ALLOWED_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = MappingProxyType(
    {
    RunState.QUEUED: frozenset(
        {RunState.STARTING, RunState.SUCCEEDED, RunState.CANCELLED, RunState.FAILED}
    ),
    RunState.STARTING: frozenset(
        {RunState.SYNTHESIZING, RunState.TRAINING, RunState.CANCELLED, RunState.FAILED}
    ),
    RunState.SYNTHESIZING: frozenset(
        {RunState.TRAINING, RunState.CANCELLED, RunState.FAILED}
    ),
    RunState.TRAINING: frozenset(
        {RunState.EVALUATING, RunState.CANCELLED, RunState.FAILED}
    ),
    RunState.EVALUATING: frozenset(
        {RunState.FINALIZING, RunState.CANCELLED, RunState.FAILED}
    ),
    # Once finalization begins, completion/failure wins any cancellation race.
    RunState.FINALIZING: frozenset({RunState.SUCCEEDED, RunState.FAILED}),
    RunState.SUCCEEDED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
    }
)


class InvalidTransitionError(DistilleryError):
    def __init__(
        self,
        current: RunState,
        target: RunState,
        *,
        reason: str | None = None,
    ) -> None:
        message = f"invalid transition {current.value} -> {target.value}"
        if reason is not None:
            message = f"{message}: {reason}"
        super().__init__(
            ErrorPayload.from_code(
                DistilleryErrorCode.INVALID_TRANSITION,
                message,
                details={
                    "current": current.value,
                    "target": target.value,
                    **({"reason": reason} if reason is not None else {}),
                },
                retryable=False,
            )
        )


def next_states(current: RunState) -> frozenset[RunState]:
    return ALLOWED_TRANSITIONS[current]


def assert_transition(current: RunState, target: RunState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(current, target)
