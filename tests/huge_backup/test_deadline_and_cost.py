"""Warm-time arithmetic and exact gross cost."""

from __future__ import annotations

import pytest

from experiments.huge_backup.cost import build_gross_cost_artifact, exact_gross_cost_usd
from experiments.huge_backup.deadline import (
    ARTIFACT_RESERVE_SECONDS,
    HARD_DEADLINE_OFFSET_SECONDS,
    MAX_RUNTIME_SECONDS,
    SHUTDOWN_MARGIN_SECONDS,
    TRAINING_DEADLINE_OFFSET_SECONDS,
    TRAINING_WINDOW_SECONDS,
    DeadlineExceeded,
    build_deadline,
    warm_time_arithmetic,
)
from experiments.huge_backup.profile import DEFAULT_HUGE_BACKUP_PROFILE, assert_production_seal


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_frozen_warm_time_arithmetic() -> None:
    assert MAX_RUNTIME_SECONDS == 1800
    assert ARTIFACT_RESERVE_SECONDS == 300
    assert SHUTDOWN_MARGIN_SECONDS == 30
    assert HARD_DEADLINE_OFFSET_SECONDS == 1770
    assert TRAINING_DEADLINE_OFFSET_SECONDS == 1470
    assert TRAINING_WINDOW_SECONDS == 1470
    arithmetic = warm_time_arithmetic()
    assert arithmetic["training_window_seconds"] == 1470
    assert arithmetic["max_median_step_seconds_for_budget"] == pytest.approx(7.35)
    assert arithmetic["rehearsal_budget_breach_at_fail_gate_seconds"] == 1600.0


def test_training_stops_before_five_minute_reserve() -> None:
    clock = FakeClock()
    deadline = build_deadline(clock=clock)
    clock.value = 1469.9
    deadline.require_training_time("last update")
    clock.value = 1470.0
    with pytest.raises(DeadlineExceeded, match="artifact reserve"):
        deadline.require_training_time("overflow update")
    deadline.require_finalize_time("adapter save")
    clock.value = 1770.0
    with pytest.raises(DeadlineExceeded, match="hard runtime"):
        deadline.require_finalize_time("late checksum")


def test_exact_gross_cost_ceiling() -> None:
    # 31.5641 * 0.5h = 15.78205 -> ceil cents = 15.79
    assert exact_gross_cost_usd(max_runtime_seconds=1800) == 15.79
    artifact = build_gross_cost_artifact(DEFAULT_HUGE_BACKUP_PROFILE)
    assert artifact["gross_cost_usd"] == 15.79
    assert artifact["max_run_usd"] == 15.79


def test_production_seal() -> None:
    assert_production_seal(DEFAULT_HUGE_BACKUP_PROFILE)
