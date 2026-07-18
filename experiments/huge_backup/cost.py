"""Exact gross cost for the sealed warm job ceiling."""

from __future__ import annotations

import math
from typing import Any

from experiments.huge_backup.profile import HOURLY_USD, PRICE_SOURCE, HugeBackupTrainingProfile


def exact_gross_cost_usd(
    *,
    hourly_usd: float = HOURLY_USD,
    max_runtime_seconds: int,
) -> float:
    """Gross = hourly * (max_runtime_seconds/3600), rounded up to cents."""
    if hourly_usd <= 0:
        raise ValueError("hourly_usd must be positive")
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    gross = hourly_usd * (max_runtime_seconds / 3600.0)
    return math.ceil(gross * 100.0) / 100.0


def build_gross_cost_artifact(profile: HugeBackupTrainingProfile) -> dict[str, Any]:
    gross = exact_gross_cost_usd(
        hourly_usd=profile.hourly_usd,
        max_runtime_seconds=profile.max_runtime_seconds,
    )
    if abs(gross - profile.max_run_usd) > 1e-9:
        raise RuntimeError("gross cost diverged from profile.max_run_usd")
    return {
        "schema_version": "distillery.huge_backup.gross_cost.v1",
        "instance_type": profile.instance_type,
        "hourly_usd": profile.hourly_usd,
        "price_source": PRICE_SOURCE,
        "max_runtime_seconds": profile.max_runtime_seconds,
        "billing_hours": profile.max_runtime_seconds / 3600.0,
        "gross_cost_usd": gross,
        "max_run_usd": profile.max_run_usd,
        "currency": "USD",
        "notes": "exact ceiling for the sealed 30-minute warm job; teacher pre-timer excluded",
    }
