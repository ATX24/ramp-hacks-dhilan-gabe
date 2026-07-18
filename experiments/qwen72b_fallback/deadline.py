"""Warm-job deadlines for rehearsal and full 72B runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

# Rehearsal: 3 optimizer steps, hard wall well under the $100 cap.
REHEARSAL_MAX_RUNTIME_SECONDS = 20 * 60  # 1200
# Full run profile window: 30–90 minutes sealed at the upper bound.
FULL_MAX_RUNTIME_SECONDS = 90 * 60  # 5400
ARTIFACT_RESERVE_SECONDS = 8 * 60  # adapter save, rank-0 reload, manifest, smoke
SHUTDOWN_MARGIN_SECONDS = 30
ProfileKind = Literal["rehearsal", "full"]


class DeadlineExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Deadline:
    started_monotonic: float
    training_deadline_monotonic: float
    hard_deadline_monotonic: float
    clock: Callable[[], float] = time.monotonic

    def require_training_time(self, phase: str) -> None:
        if self.clock() >= self.training_deadline_monotonic:
            raise DeadlineExceeded(
                f"training deadline reached before {phase}; artifact reserve preserved"
            )

    def require_finalize_time(self, phase: str) -> None:
        if self.clock() >= self.hard_deadline_monotonic:
            raise DeadlineExceeded(f"hard runtime deadline reached before {phase}")


def build_deadline(
    *,
    kind: ProfileKind,
    artifact_reserve_seconds: int = ARTIFACT_RESERVE_SECONDS,
    shutdown_margin_seconds: int = SHUTDOWN_MARGIN_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Deadline:
    max_runtime = (
        REHEARSAL_MAX_RUNTIME_SECONDS if kind == "rehearsal" else FULL_MAX_RUNTIME_SECONDS
    )
    if artifact_reserve_seconds + shutdown_margin_seconds >= max_runtime:
        raise ValueError("artifact reserve plus shutdown margin exceeds runtime")
    started = clock()
    hard = started + max_runtime - shutdown_margin_seconds
    return Deadline(
        started_monotonic=started,
        training_deadline_monotonic=hard - artifact_reserve_seconds,
        hard_deadline_monotonic=hard,
        clock=clock,
    )


def time_arithmetic(kind: ProfileKind) -> dict[str, int | float | str]:
    max_runtime = (
        REHEARSAL_MAX_RUNTIME_SECONDS if kind == "rehearsal" else FULL_MAX_RUNTIME_SECONDS
    )
    training_window = max_runtime - SHUTDOWN_MARGIN_SECONDS - ARTIFACT_RESERVE_SECONDS
    return {
        "kind": kind,
        "max_runtime_seconds": max_runtime,
        "artifact_reserve_seconds": ARTIFACT_RESERVE_SECONDS,
        "shutdown_margin_seconds": SHUTDOWN_MARGIN_SECONDS,
        "training_window_seconds": training_window,
        "training_window_minutes": training_window / 60.0,
    }
