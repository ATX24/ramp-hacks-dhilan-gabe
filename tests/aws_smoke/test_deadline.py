"""Emergency runtime deadline preserves time for required artifacts."""

from __future__ import annotations

import pytest

from experiments.aws_smoke.deadline import DeadlineExceeded, build_deadline


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_training_stops_at_690_with_finalize_and_shutdown_reserves() -> None:
    clock = FakeClock()
    deadline = build_deadline(
        max_runtime_seconds=900,
        artifact_reserve_seconds=180,
        clock=clock,
    )
    clock.value = 689.9
    deadline.require_training_time("last step")
    clock.value = 690.0
    with pytest.raises(DeadlineExceeded, match="artifact reserve"):
        deadline.require_training_time("next step")
    deadline.require_finalize_time("adapter save")
    clock.value = 870.0
    with pytest.raises(DeadlineExceeded, match="hard runtime"):
        deadline.require_finalize_time("checksums")


def test_deadline_rejects_unsafe_reserve() -> None:
    with pytest.raises(ValueError, match="at least 120"):
        build_deadline(max_runtime_seconds=900, artifact_reserve_seconds=60)
    with pytest.raises(ValueError, match="may not exceed 900"):
        build_deadline(max_runtime_seconds=901, artifact_reserve_seconds=180)
