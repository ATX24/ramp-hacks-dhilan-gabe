"""Monotonic emergency deadline with artifact-finalization reserve."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


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
    max_runtime_seconds: int,
    artifact_reserve_seconds: int,
    shutdown_margin_seconds: int = 30,
    clock: Callable[[], float] = time.monotonic,
) -> Deadline:
    if max_runtime_seconds > 900:
        raise ValueError("emergency trainer max runtime may not exceed 900 seconds")
    if artifact_reserve_seconds < 120:
        raise ValueError("artifact reserve must be at least 120 seconds")
    if artifact_reserve_seconds >= max_runtime_seconds:
        raise ValueError("artifact reserve must be shorter than max runtime")
    if shutdown_margin_seconds < 15:
        raise ValueError("shutdown margin must be at least 15 seconds")
    if artifact_reserve_seconds + shutdown_margin_seconds >= max_runtime_seconds:
        raise ValueError("artifact reserve plus shutdown margin exceeds runtime")
    started = clock()
    hard_deadline = started + max_runtime_seconds - shutdown_margin_seconds
    return Deadline(
        started_monotonic=started,
        training_deadline_monotonic=hard_deadline - artifact_reserve_seconds,
        hard_deadline_monotonic=hard_deadline,
        clock=clock,
    )
