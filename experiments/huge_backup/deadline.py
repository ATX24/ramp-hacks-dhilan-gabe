"""30-minute warm deadline with a frozen 5-minute artifact reserve."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

# Frozen warm-job arithmetic (seconds).
MAX_RUNTIME_SECONDS = 30 * 60  # 1800
ARTIFACT_RESERVE_SECONDS = 5 * 60  # 300 — adapter save, rank-0 reload, manifest, smoke
SHUTDOWN_MARGIN_SECONDS = 30
# Training window = 1800 - 30 - 300 = 1470s (24.5 minutes).
TRAINING_WINDOW_SECONDS = MAX_RUNTIME_SECONDS - SHUTDOWN_MARGIN_SECONDS - ARTIFACT_RESERVE_SECONDS
HARD_DEADLINE_OFFSET_SECONDS = MAX_RUNTIME_SECONDS - SHUTDOWN_MARGIN_SECONDS  # 1770
TRAINING_DEADLINE_OFFSET_SECONDS = HARD_DEADLINE_OFFSET_SECONDS - ARTIFACT_RESERVE_SECONDS


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
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS,
    artifact_reserve_seconds: int = ARTIFACT_RESERVE_SECONDS,
    shutdown_margin_seconds: int = SHUTDOWN_MARGIN_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Deadline:
    if max_runtime_seconds != MAX_RUNTIME_SECONDS:
        raise ValueError(f"huge_backup warm job requires exactly {MAX_RUNTIME_SECONDS}s runtime")
    if artifact_reserve_seconds != ARTIFACT_RESERVE_SECONDS:
        raise ValueError(
            f"huge_backup requires exactly {ARTIFACT_RESERVE_SECONDS}s artifact reserve"
        )
    if shutdown_margin_seconds != SHUTDOWN_MARGIN_SECONDS:
        raise ValueError(f"huge_backup requires exactly {SHUTDOWN_MARGIN_SECONDS}s shutdown margin")
    if artifact_reserve_seconds + shutdown_margin_seconds >= max_runtime_seconds:
        raise ValueError("artifact reserve plus shutdown margin exceeds runtime")
    started = clock()
    hard_deadline = started + HARD_DEADLINE_OFFSET_SECONDS
    return Deadline(
        started_monotonic=started,
        training_deadline_monotonic=hard_deadline - artifact_reserve_seconds,
        hard_deadline_monotonic=hard_deadline,
        clock=clock,
    )


def warm_time_arithmetic() -> dict[str, int | float]:
    """Exact sealed arithmetic for operator briefing / manifests."""
    return {
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "artifact_reserve_seconds": ARTIFACT_RESERVE_SECONDS,
        "shutdown_margin_seconds": SHUTDOWN_MARGIN_SECONDS,
        "hard_deadline_offset_seconds": HARD_DEADLINE_OFFSET_SECONDS,
        "training_deadline_offset_seconds": TRAINING_DEADLINE_OFFSET_SECONDS,
        "training_window_seconds": TRAINING_WINDOW_SECONDS,
        "training_window_minutes": TRAINING_WINDOW_SECONDS / 60.0,
        "updates": 200,
        "max_median_step_seconds_for_budget": TRAINING_WINDOW_SECONDS / 200.0,
        "rehearsal_median_step_fail_seconds": 8.0,
        "rehearsal_budget_breach_at_fail_gate_seconds": 200 * 8.0,
    }
