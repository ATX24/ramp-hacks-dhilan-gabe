"""Gross cost ceilings for materialization, rehearsal, and full runs."""

from __future__ import annotations

import math
from typing import Any, Literal

# Operator-attested SageMaker on-demand us-east-1 (same pin as huge_backup / campaign).
P4DE_HOURLY_USD = 31.5641
P4DE_PRICE_SOURCE = "operator_attested_ml.p4de.24xlarge_us-east-1_31.5641"
# Ephemeral transfer instance class used by Distillery model materialization.
TRANSFER_HOURLY_USD = 1.944
TRANSFER_PRICE_SOURCE = "operator_attested_c5n.9xlarge_us-east-1_1.944"
TRANSFER_HARD_CAP_USD = 500.0
REHEARSAL_HARD_CAP_USD = 100.0
FULL_RUN_HARD_CAP_USD = 500.0


def exact_gross_cost_usd(*, hourly_usd: float, max_runtime_seconds: int) -> float:
    if hourly_usd <= 0:
        raise ValueError("hourly_usd must be positive")
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    gross = hourly_usd * (max_runtime_seconds / 3600.0)
    return math.ceil(gross * 100.0) / 100.0


def assert_under_cap(*, gross_usd: float, hard_cap_usd: float, label: str) -> None:
    if gross_usd > hard_cap_usd + 1e-9:
        raise RuntimeError(
            f"{label} gross ${gross_usd:.2f} exceeds hard cap ${hard_cap_usd:.2f}"
        )


def build_cost_artifact(
    *,
    kind: Literal["materialization", "rehearsal", "full"],
    max_runtime_seconds: int,
    hourly_usd: float,
    price_source: str,
    hard_cap_usd: float,
    instance_type: str,
) -> dict[str, Any]:
    gross = exact_gross_cost_usd(
        hourly_usd=hourly_usd,
        max_runtime_seconds=max_runtime_seconds,
    )
    assert_under_cap(gross_usd=gross, hard_cap_usd=hard_cap_usd, label=kind)
    return {
        "schema_version": "distillery.qwen72b_fallback.gross_cost.v1",
        "kind": kind,
        "instance_type": instance_type,
        "hourly_usd": hourly_usd,
        "price_source": price_source,
        "max_runtime_seconds": max_runtime_seconds,
        "billing_hours": max_runtime_seconds / 3600.0,
        "gross_cost_usd": gross,
        "hard_cap_usd": hard_cap_usd,
        "currency": "USD",
        "under_cap": True,
    }
