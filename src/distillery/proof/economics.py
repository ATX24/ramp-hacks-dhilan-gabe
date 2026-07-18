"""Gross cost, quality retention/gap, and utilization-sensitive break-even."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

BREAK_EVEN_NEVER: Literal["never"] = "never"

CostKind = Literal["measured", "projected", "missing"]
UTILIZATION_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.80)


@dataclass(frozen=True)
class CostValue:
    amount_usd: float | None
    kind: CostKind
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount_usd": self.amount_usd,
            "kind": self.kind,
            "label": self.label,
        }


@dataclass(frozen=True)
class BreakEvenResult:
    break_even_requests: int | Literal["never"] | None
    savings_per_request_usd: float | None
    incremental_cost_usd: float | None
    teacher_cost_per_request_usd: float | None
    student_cost_per_request_usd: float | None
    student_cost_kind: CostKind
    utilization: float
    horizon_requests: int | None
    within_horizon: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "break_even_requests": self.break_even_requests,
            "savings_per_request_usd": self.savings_per_request_usd,
            "incremental_cost_usd": self.incremental_cost_usd,
            "teacher_cost_per_request_usd": self.teacher_cost_per_request_usd,
            "student_cost_per_request_usd": self.student_cost_per_request_usd,
            "student_cost_kind": self.student_cost_kind,
            "utilization": self.utilization,
            "horizon_requests": self.horizon_requests,
            "within_horizon": self.within_horizon,
        }


@dataclass(frozen=True)
class EconomicsSummary:
    gross_experiment_cost_usd: CostValue
    teacher_generation_cost_usd: CostValue
    cheap_api_benchmark_cost_usd: CostValue
    storage_cost_usd: CostValue
    training_cost_usd: CostValue
    quality_retention: float | None
    recovered_teacher_gap: float | None
    recovered_teacher_gap_defined: bool
    utilization_rows: tuple[dict[str, Any], ...]
    break_even_at_25pct: BreakEvenResult
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gross_experiment_cost_usd": self.gross_experiment_cost_usd.to_dict(),
            "teacher_generation_cost_usd": self.teacher_generation_cost_usd.to_dict(),
            "cheap_api_benchmark_cost_usd": self.cheap_api_benchmark_cost_usd.to_dict(),
            "storage_cost_usd": self.storage_cost_usd.to_dict(),
            "training_cost_usd": self.training_cost_usd.to_dict(),
            "quality_retention": self.quality_retention,
            "recovered_teacher_gap": self.recovered_teacher_gap,
            "recovered_teacher_gap_defined": self.recovered_teacher_gap_defined,
            "utilization_rows": list(self.utilization_rows),
            "break_even_at_25pct": self.break_even_at_25pct.to_dict(),
            "notes": list(self.notes),
        }


def quality_retention(
    student_primary_index: float, teacher_primary_index: float
) -> float | None:
    """student / teacher. Undefined when teacher index is 0."""
    if teacher_primary_index == 0.0:
        return None
    return student_primary_index / teacher_primary_index


def recovered_teacher_gap(
    student_primary_index: float,
    base_primary_index: float,
    teacher_primary_index: float,
) -> float | None:
    """(student - base) / (teacher - base). Undefined when denominator <= 0.

    Never clamps negative or >1 values.
    """
    denom = teacher_primary_index - base_primary_index
    if denom <= 0:
        return None
    return (student_primary_index - base_primary_index) / denom


def break_even_requests(
    total_incremental_distillation_cost: float,
    teacher_cost_per_request: float,
    student_projected_cost_per_request: float,
) -> int | Literal["never"]:
    """Requests until distillation pays back. ``never`` if savings <= 0."""
    savings = teacher_cost_per_request - student_projected_cost_per_request
    if savings <= 0:
        return BREAK_EVEN_NEVER
    return int(math.ceil(total_incremental_distillation_cost / savings))


def utilization_cost_rows(
    *,
    student_cost_per_request_at_full_util: float,
    teacher_cost_per_request: float,
    incremental_cost: float,
    batch1_throughput_rps: float | None,
    batch8_throughput_rps: float | None,
    utilizations: tuple[float, ...] = UTILIZATION_LEVELS,
) -> tuple[dict[str, Any], ...]:
    """Projected serving cost rows at disclosed utilizations.

    Cost scales as ``full_util_cost / utilization`` (idle capacity allocated).
    All serving costs are explicitly ``projected``.
    """
    rows: list[dict[str, Any]] = []
    for util in utilizations:
        if util <= 0:
            raise ValueError(f"utilization must be positive, got {util}")
        student_cpr = student_cost_per_request_at_full_util / util
        be = break_even_requests(incremental_cost, teacher_cost_per_request, student_cpr)
        row: dict[str, Any] = {
            "utilization": util,
            "student_cost_per_request_usd": {
                "amount_usd": student_cpr,
                "kind": "projected",
                "label": f"student_serving_at_{int(util * 100)}pct_util",
            },
            "teacher_cost_per_request_usd": {
                "amount_usd": teacher_cost_per_request,
                "kind": "measured",
                "label": "teacher_or_api_cost_per_request",
            },
            "savings_per_request_usd": teacher_cost_per_request - student_cpr,
            "break_even_requests": be,
            "batch1_throughput_rps": {
                "value": batch1_throughput_rps,
                "kind": "measured" if batch1_throughput_rps is not None else "missing",
            },
            "batch8_throughput_rps": {
                "value": batch8_throughput_rps,
                "kind": "measured" if batch8_throughput_rps is not None else "missing",
            },
        }
        rows.append(row)
    return tuple(rows)


def _cost_from_record(
    costs: dict[str, Any], key: str, *, default_kind: CostKind = "measured"
) -> CostValue:
    if key not in costs or costs[key] is None:
        return CostValue(amount_usd=None, kind="missing", label=key)
    raw = costs[key]
    if isinstance(raw, dict):
        amount = raw.get("amount_usd", raw.get("usd"))
        kind = raw.get("kind", default_kind)
        if kind not in ("measured", "projected", "missing"):
            kind = default_kind
        return CostValue(
            amount_usd=float(amount) if amount is not None else None,
            kind=kind,  # type: ignore[arg-type]
            label=str(raw.get("label", key)),
        )
    return CostValue(amount_usd=float(raw), kind=default_kind, label=key)


def compute_economics(
    *,
    student_primary_index: float,
    teacher_primary_index: float,
    base_primary_index: float,
    costs: dict[str, Any],
    teacher_cost_per_request: float | None,
    student_cost_per_request_at_full_util: float | None,
    batch1_throughput_rps: float | None = None,
    batch8_throughput_rps: float | None = None,
    evaluation_horizon_requests: int | None = None,
    economics_utilization: float = 0.25,
) -> EconomicsSummary:
    notes: list[str] = []
    gross = _cost_from_record(costs, "gross_experiment_cost_usd")
    teacher_gen = _cost_from_record(costs, "teacher_generation_cost_usd")
    cheap_api = _cost_from_record(costs, "cheap_api_benchmark_cost_usd")
    storage = _cost_from_record(costs, "storage_cost_usd")
    training = _cost_from_record(costs, "training_cost_usd")

    retention = quality_retention(student_primary_index, teacher_primary_index)
    gap = recovered_teacher_gap(
        student_primary_index, base_primary_index, teacher_primary_index
    )
    if gap is None:
        notes.append("recovered_teacher_gap_undefined_nonpositive_denominator")

    incremental = None
    if training.amount_usd is not None and teacher_gen.amount_usd is not None:
        incremental = training.amount_usd + teacher_gen.amount_usd
    elif training.amount_usd is not None:
        incremental = training.amount_usd
        notes.append("incremental_cost_uses_training_only_teacher_gen_missing")
    else:
        notes.append("incremental_distillation_cost_missing")

    if teacher_cost_per_request is None:
        notes.append("teacher_cost_per_request_missing")
    if student_cost_per_request_at_full_util is None:
        notes.append("student_projected_cost_per_request_missing")

    if (
        incremental is not None
        and teacher_cost_per_request is not None
        and student_cost_per_request_at_full_util is not None
    ):
        rows = utilization_cost_rows(
            student_cost_per_request_at_full_util=student_cost_per_request_at_full_util,
            teacher_cost_per_request=teacher_cost_per_request,
            incremental_cost=incremental,
            batch1_throughput_rps=batch1_throughput_rps,
            batch8_throughput_rps=batch8_throughput_rps,
        )
        student_at_util = student_cost_per_request_at_full_util / economics_utilization
        be_val = break_even_requests(
            incremental, teacher_cost_per_request, student_at_util
        )
        savings = teacher_cost_per_request - student_at_util
        within: bool | None
        if be_val == BREAK_EVEN_NEVER:
            within = False
        elif evaluation_horizon_requests is None:
            within = None
            notes.append("evaluation_horizon_requests_missing")
        else:
            within = be_val <= evaluation_horizon_requests
        be = BreakEvenResult(
            break_even_requests=be_val,
            savings_per_request_usd=savings,
            incremental_cost_usd=incremental,
            teacher_cost_per_request_usd=teacher_cost_per_request,
            student_cost_per_request_usd=student_at_util,
            student_cost_kind="projected",
            utilization=economics_utilization,
            horizon_requests=evaluation_horizon_requests,
            within_horizon=within,
        )
    else:
        rows = ()
        be = BreakEvenResult(
            break_even_requests=None,
            savings_per_request_usd=None,
            incremental_cost_usd=incremental,
            teacher_cost_per_request_usd=teacher_cost_per_request,
            student_cost_per_request_usd=(
                None
                if student_cost_per_request_at_full_util is None
                else student_cost_per_request_at_full_util / economics_utilization
            ),
            student_cost_kind="projected"
            if student_cost_per_request_at_full_util is not None
            else "missing",
            utilization=economics_utilization,
            horizon_requests=evaluation_horizon_requests,
            within_horizon=None,
        )

    notes.append("serving_costs_are_projected_not_measured_production_savings")
    return EconomicsSummary(
        gross_experiment_cost_usd=gross,
        teacher_generation_cost_usd=teacher_gen,
        cheap_api_benchmark_cost_usd=cheap_api,
        storage_cost_usd=storage,
        training_cost_usd=training,
        quality_retention=retention,
        recovered_teacher_gap=gap,
        recovered_teacher_gap_defined=gap is not None,
        utilization_rows=rows,
        break_even_at_25pct=be,
        notes=tuple(notes),
    )
