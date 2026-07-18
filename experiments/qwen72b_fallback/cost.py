"""Worst-case cost controls including active orphans and retry exposure."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.qwen72b_fallback.evidence import (
    HashBoundEvidence,
    VerificationSource,
)

P4DE_HOURLY_USD = 31.5641
P4DE_PRICE_SOURCE = "operator_attested_ml.p4de.24xlarge_us-east-1_31.5641"
TRANSFER_HOURLY_USD = 1.944
TRANSFER_PRICE_SOURCE = "operator_attested_c5n.9xlarge_us-east-1_1.944"
TRANSFER_HARD_CAP_USD = 500.0
PROBE_HARD_CAP_USD = 100.0
REHEARSAL_HARD_CAP_USD = 100.0
FULL_RUN_HARD_CAP_USD = 500.0


class CostAction(StrEnum):
    MATERIALIZE = "materialize"
    MEMORY_PROBE = "memory_probe"
    REHEARSAL = "rehearsal"
    FULL = "full"


class ResourceKind(StrEnum):
    TRANSFER_EC2 = "transfer_ec2"
    P4DE_TRAINING_JOB = "p4de_training_job"


class ActiveResourceCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    resource_id: str = Field(min_length=1)
    resource_kind: ResourceKind
    age_seconds: int = Field(ge=0)
    hourly_usd: float = Field(gt=0)
    accrued_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def _exact_accrual(self) -> ActiveResourceCost:
        expected = exact_gross_cost_usd(
            hourly_usd=self.hourly_usd,
            max_runtime_seconds=max(self.age_seconds, 1),
        )
        if self.accrued_usd != expected:
            raise ValueError("active resource accrued cost arithmetic mismatch")
        return self


class CostAuthorizationEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.cost_evidence.v2"] = (
        "distillery.qwen72b_fallback.cost_evidence.v2"
    )
    source: Literal[VerificationSource.LIVE_AWS] = VerificationSource.LIVE_AWS
    action: CostAction
    instance_type: str
    hourly_usd: float = Field(gt=0)
    price_source: str
    max_runtime_seconds: int = Field(gt=0)
    max_launch_attempts: Literal[1] = 1
    retry_budget_usd: Literal[0.0] = 0.0
    new_run_worst_case_usd: float = Field(gt=0)
    active_resources: tuple[ActiveResourceCost, ...]
    active_orphan_accrued_usd: float = Field(ge=0)
    total_worst_case_usd: float = Field(gt=0)
    hard_cap_usd: float = Field(gt=0)

    @model_validator(mode="after")
    def _cost_invariants(self) -> CostAuthorizationEvidence:
        expected_new = exact_gross_cost_usd(
            hourly_usd=self.hourly_usd,
            max_runtime_seconds=self.max_runtime_seconds,
        )
        if self.new_run_worst_case_usd != expected_new:
            raise ValueError("new-run cost arithmetic mismatch")
        expected_orphans = round_up_cents(
            sum(resource.accrued_usd for resource in self.active_resources)
        )
        if self.active_orphan_accrued_usd != expected_orphans:
            raise ValueError("orphan cost arithmetic mismatch")
        expected_total = round_up_cents(expected_new + expected_orphans + self.retry_budget_usd)
        if self.total_worst_case_usd != expected_total:
            raise ValueError("total worst-case cost arithmetic mismatch")
        if self.total_worst_case_usd > self.hard_cap_usd:
            raise ValueError("total worst-case cost exceeds the hard cap")
        return self


def round_up_cents(value: float) -> float:
    return math.ceil(value * 100.0) / 100.0


def exact_gross_cost_usd(*, hourly_usd: float, max_runtime_seconds: int) -> float:
    if hourly_usd <= 0:
        raise ValueError("hourly_usd must be positive")
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    return round_up_cents(hourly_usd * (max_runtime_seconds / 3600.0))


def assert_under_cap(*, gross_usd: float, hard_cap_usd: float, label: str) -> None:
    if gross_usd > hard_cap_usd + 1e-9:
        raise RuntimeError(f"{label} gross ${gross_usd:.2f} exceeds hard cap ${hard_cap_usd:.2f}")


def seal_cost_evidence(
    *,
    action: CostAction,
    instance_type: str,
    hourly_usd: float,
    price_source: str,
    max_runtime_seconds: int,
    hard_cap_usd: float,
    active_resources: tuple[ActiveResourceCost, ...],
) -> CostAuthorizationEvidence:
    new_cost = exact_gross_cost_usd(
        hourly_usd=hourly_usd,
        max_runtime_seconds=max_runtime_seconds,
    )
    orphan_cost = round_up_cents(sum(resource.accrued_usd for resource in active_resources))
    return CostAuthorizationEvidence.seal(
        action=action,
        instance_type=instance_type,
        hourly_usd=hourly_usd,
        price_source=price_source,
        max_runtime_seconds=max_runtime_seconds,
        active_resources=active_resources,
        active_orphan_accrued_usd=orphan_cost,
        new_run_worst_case_usd=new_cost,
        total_worst_case_usd=round_up_cents(new_cost + orphan_cost),
        hard_cap_usd=hard_cap_usd,
    )
