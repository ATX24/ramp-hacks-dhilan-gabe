"""Explicit channel/load/train/save/reload/cleanup budgets for 72B jobs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class RunPhase(StrEnum):
    CHANNEL_VERIFY = "channel_verify"
    MODEL_LOAD = "model_load"
    TRAIN_STEPS = "train_steps"
    ADAPTER_SAVE = "adapter_save"
    ADAPTER_RELOAD = "adapter_reload"
    CLEANUP_ARTIFACTS = "cleanup_artifacts"
    SHUTDOWN = "shutdown"


PHASE_ORDER = (
    RunPhase.CHANNEL_VERIFY,
    RunPhase.MODEL_LOAD,
    RunPhase.TRAIN_STEPS,
    RunPhase.ADAPTER_SAVE,
    RunPhase.ADAPTER_RELOAD,
    RunPhase.CLEANUP_ARTIFACTS,
    RunPhase.SHUTDOWN,
)


@dataclass(frozen=True, slots=True)
class PhaseBudget:
    kind: Literal["memory_probe", "rehearsal", "full"]
    max_runtime_seconds: int
    phase_seconds: dict[RunPhase, int]

    def __post_init__(self) -> None:
        if set(self.phase_seconds) != set(PHASE_ORDER):
            raise ValueError("phase budget must explicitly cover every run phase")
        if any(seconds < 0 for seconds in self.phase_seconds.values()):
            raise ValueError("phase budgets must be nonnegative")
        if sum(self.phase_seconds.values()) > self.max_runtime_seconds:
            raise ValueError("phase budgets exceed max runtime")

    @property
    def unallocated_seconds(self) -> int:
        return self.max_runtime_seconds - sum(self.phase_seconds.values())

    def phase_deadline_offset(self, phase: RunPhase) -> int:
        index = PHASE_ORDER.index(phase)
        return self.unallocated_seconds + sum(
            self.phase_seconds[item] for item in PHASE_ORDER[: index + 1]
        )


_BUDGETS = {
    "memory_probe": PhaseBudget(
        kind="memory_probe",
        max_runtime_seconds=3600,
        phase_seconds={
            RunPhase.CHANNEL_VERIFY: 300,
            RunPhase.MODEL_LOAD: 900,
            RunPhase.TRAIN_STEPS: 600,
            RunPhase.ADAPTER_SAVE: 180,
            RunPhase.ADAPTER_RELOAD: 0,
            RunPhase.CLEANUP_ARTIFACTS: 300,
            RunPhase.SHUTDOWN: 120,
        },
    ),
    "rehearsal": PhaseBudget(
        kind="rehearsal",
        max_runtime_seconds=3600,
        phase_seconds={
            RunPhase.CHANNEL_VERIFY: 300,
            RunPhase.MODEL_LOAD: 900,
            RunPhase.TRAIN_STEPS: 600,
            RunPhase.ADAPTER_SAVE: 300,
            RunPhase.ADAPTER_RELOAD: 900,
            RunPhase.CLEANUP_ARTIFACTS: 300,
            RunPhase.SHUTDOWN: 120,
        },
    ),
    "full": PhaseBudget(
        kind="full",
        max_runtime_seconds=5400,
        phase_seconds={
            RunPhase.CHANNEL_VERIFY: 240,
            RunPhase.MODEL_LOAD: 720,
            RunPhase.TRAIN_STEPS: 3000,
            RunPhase.ADAPTER_SAVE: 240,
            RunPhase.ADAPTER_RELOAD: 720,
            RunPhase.CLEANUP_ARTIFACTS: 240,
            RunPhase.SHUTDOWN: 120,
        },
    ),
}


class DeadlineExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunDeadline:
    budget: PhaseBudget
    started_monotonic: float
    clock: Callable[[], float] = time.monotonic

    def require_phase_time(self, phase: RunPhase, operation: str) -> None:
        elapsed = self.clock() - self.started_monotonic
        deadline = self.budget.phase_deadline_offset(phase)
        if elapsed >= deadline:
            raise DeadlineExceeded(
                f"{phase.value} deadline reached before {operation}; "
                "later save/reload/cleanup reserves remain protected"
            )

    def remaining_seconds(self, phase: RunPhase) -> float:
        elapsed = self.clock() - self.started_monotonic
        return max(0.0, self.budget.phase_deadline_offset(phase) - elapsed)


def phase_budget_for(kind: str) -> PhaseBudget:
    try:
        return _BUDGETS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown 72B run kind: {kind!r}") from exc


def build_deadline(
    kind: Literal["memory_probe", "rehearsal", "full"],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> RunDeadline:
    return RunDeadline(
        budget=phase_budget_for(kind),
        started_monotonic=clock(),
        clock=clock,
    )


def time_arithmetic(kind: Literal["memory_probe", "rehearsal", "full"]) -> dict[str, object]:
    budget = phase_budget_for(kind)
    return {
        "kind": kind,
        "max_runtime_seconds": budget.max_runtime_seconds,
        "unallocated_seconds": budget.unallocated_seconds,
        "phase_seconds": {phase.value: budget.phase_seconds[phase] for phase in PHASE_ORDER},
        "phase_deadline_offsets": {
            phase.value: budget.phase_deadline_offset(phase) for phase in PHASE_ORDER
        },
    }
