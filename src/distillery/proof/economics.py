"""Gross cost, quality retention/gap, and utilization-sensitive break-even."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from distillery.proof.evidence import EvidenceKind, LabeledValue, evidence_kind

BREAK_EVEN_NEVER: Literal["never"] = "never"
CostKind = EvidenceKind
UTILIZATION_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.80)
REQUIRED_BATCH_SIZES: tuple[int, ...] = (1, 8)


@dataclass(frozen=True)
class CostValue:
    """A USD amount with exhaustive evidence provenance."""

    amount_usd: float | None
    kind: EvidenceKind | str
    label: str
    reason: str | None = None

    def __post_init__(self) -> None:
        kind = evidence_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is EvidenceKind.MISSING and self.amount_usd is not None:
            raise ValueError("missing cost evidence must not carry an amount")
        if kind is not EvidenceKind.MISSING and self.amount_usd is None:
            raise ValueError(f"{kind.value} cost evidence requires an amount")
        if self.amount_usd is not None and self.amount_usd < 0:
            raise ValueError("cost amounts must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount_usd": self.amount_usd,
            "kind": self.kind.value,
            "label": self.label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GrossCostLedger:
    """Complete, reconciled, pre-credit experiment cost ledger."""

    gross_experiment_cost_usd: CostValue
    billed_training_seconds: LabeledValue
    gpu_compute_cost_usd: CostValue
    teacher_generation_tokens: LabeledValue
    teacher_generation_cost_usd: CostValue
    cheap_api_benchmark_cost_usd: CostValue
    cheap_api_zero_reason: str | None
    storage_cost_usd: CostValue
    other_costs_usd: tuple[CostValue, ...]
    other_costs_declared: bool
    component_sum_usd: float | None
    reconciled: bool | None
    completeness_gaps: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.completeness_gaps

    @property
    def incremental_distillation_cost_usd(self) -> float | None:
        """Gross experiment cost excluding the cheap-API comparison run."""

        if not self.complete:
            return None
        assert self.gross_experiment_cost_usd.amount_usd is not None
        assert self.cheap_api_benchmark_cost_usd.amount_usd is not None
        return (
            self.gross_experiment_cost_usd.amount_usd
            - self.cheap_api_benchmark_cost_usd.amount_usd
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gross_experiment_cost_usd": self.gross_experiment_cost_usd.to_dict(),
            "billed_training_seconds": self.billed_training_seconds.to_dict(),
            "gpu_compute_cost_usd": self.gpu_compute_cost_usd.to_dict(),
            "teacher_generation_tokens": self.teacher_generation_tokens.to_dict(),
            "teacher_generation_cost_usd": self.teacher_generation_cost_usd.to_dict(),
            "cheap_api_benchmark_cost_usd": (
                self.cheap_api_benchmark_cost_usd.to_dict()
            ),
            "cheap_api_zero_reason": self.cheap_api_zero_reason,
            "storage_cost_usd": self.storage_cost_usd.to_dict(),
            "other_costs_usd": [cost.to_dict() for cost in self.other_costs_usd],
            "other_costs_declared": self.other_costs_declared,
            "component_sum_usd": self.component_sum_usd,
            "reconciled": self.reconciled,
            "incremental_distillation_cost_usd": (
                self.incremental_distillation_cost_usd
            ),
            "complete": self.complete,
            "completeness_gaps": list(self.completeness_gaps),
        }


@dataclass(frozen=True)
class BreakEvenResult:
    batch_size: int
    utilization: float
    break_even_requests: int | Literal["never"]
    savings_per_request_usd: float
    incremental_cost_usd: float
    teacher_cost_per_request_usd: float
    teacher_cost_kind: EvidenceKind | str
    student_cost_per_request_usd: float
    student_cost_kind: EvidenceKind | str
    observed_throughput_rps: float
    horizon_requests: int
    within_horizon: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "teacher_cost_kind",
            evidence_kind(self.teacher_cost_kind),
        )
        object.__setattr__(
            self,
            "student_cost_kind",
            evidence_kind(self.student_cost_kind),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "utilization": self.utilization,
            "break_even_requests": self.break_even_requests,
            "savings_per_request_usd": self.savings_per_request_usd,
            "incremental_cost_usd": self.incremental_cost_usd,
            "teacher_cost_per_request_usd": self.teacher_cost_per_request_usd,
            "teacher_cost_kind": self.teacher_cost_kind.value,
            "student_cost_per_request_usd": self.student_cost_per_request_usd,
            "student_cost_kind": self.student_cost_kind.value,
            "observed_throughput_rps": self.observed_throughput_rps,
            "horizon_requests": self.horizon_requests,
            "within_horizon": self.within_horizon,
        }


@dataclass(frozen=True)
class EconomicsSummary:
    cost_ledger: GrossCostLedger
    teacher_cost_per_request_usd: CostValue
    student_serving_hourly_cost_usd: CostValue
    quality_retention: float | None
    recovered_teacher_gap: float | None
    recovered_teacher_gap_defined: bool
    evaluated: bool
    unevaluated_reasons: tuple[str, ...]
    utilization_rows: tuple[dict[str, Any], ...]
    break_even_at_25pct: tuple[BreakEvenResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def sensitivity_complete(self) -> bool:
        expected = {
            (batch_size, utilization)
            for batch_size in REQUIRED_BATCH_SIZES
            for utilization in UTILIZATION_LEVELS
        }
        actual = {
            (int(row["batch_size"]), float(row["utilization"]))
            for row in self.utilization_rows
        }
        return self.evaluated and actual == expected

    def break_even_for(
        self,
        *,
        batch_size: int,
        utilization: float,
    ) -> BreakEvenResult | None:
        for result in self.break_even_at_25pct:
            if (
                result.batch_size == batch_size
                and result.utilization == utilization
            ):
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        ledger = self.cost_ledger
        return {
            "cost_ledger": ledger.to_dict(),
            # Keep top-level component records readable for report consumers.
            "gross_experiment_cost_usd": (
                ledger.gross_experiment_cost_usd.to_dict()
            ),
            "gpu_compute_cost_usd": ledger.gpu_compute_cost_usd.to_dict(),
            "teacher_generation_cost_usd": (
                ledger.teacher_generation_cost_usd.to_dict()
            ),
            "cheap_api_benchmark_cost_usd": (
                ledger.cheap_api_benchmark_cost_usd.to_dict()
            ),
            "storage_cost_usd": ledger.storage_cost_usd.to_dict(),
            "teacher_cost_per_request_usd": (
                self.teacher_cost_per_request_usd.to_dict()
            ),
            "student_serving_hourly_cost_usd": (
                self.student_serving_hourly_cost_usd.to_dict()
            ),
            "quality_retention": self.quality_retention,
            "recovered_teacher_gap": self.recovered_teacher_gap,
            "recovered_teacher_gap_defined": self.recovered_teacher_gap_defined,
            "evaluated": self.evaluated,
            "unevaluated_reasons": list(self.unevaluated_reasons),
            "utilization_rows": list(self.utilization_rows),
            "sensitivity_complete": self.sensitivity_complete,
            "break_even_at_25pct": [
                result.to_dict() for result in self.break_even_at_25pct
            ],
            "notes": list(self.notes),
        }


def quality_retention(
    student_primary_index: float, teacher_primary_index: float
) -> float | None:
    """student / teacher. Undefined for a nonpositive teacher index."""
    if teacher_primary_index <= 0.0:
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
    student_serving_hourly_cost_usd: CostValue,
    teacher_cost_per_request_usd: CostValue,
    incremental_cost: float,
    observed_throughput_rps_by_batch: dict[int, float],
    evaluation_horizon_requests: int,
    utilizations: tuple[float, ...] = UTILIZATION_LEVELS,
) -> tuple[dict[str, Any], ...]:
    """Build every batch/utilization sensitivity row from observed throughput."""

    if student_serving_hourly_cost_usd.amount_usd is None:
        raise ValueError("student hourly serving cost is required")
    if teacher_cost_per_request_usd.amount_usd is None:
        raise ValueError("teacher cost per request is required")
    if set(observed_throughput_rps_by_batch) != set(REQUIRED_BATCH_SIZES):
        raise ValueError("observed throughput must contain exactly batch 1 and batch 8")

    rows: list[dict[str, Any]] = []
    for batch_size in REQUIRED_BATCH_SIZES:
        throughput = observed_throughput_rps_by_batch[batch_size]
        if throughput <= 0:
            raise ValueError(f"batch {batch_size} throughput must be positive")
        for util in utilizations:
            if util <= 0:
                raise ValueError(f"utilization must be positive, got {util}")
            student_cpr = (
                student_serving_hourly_cost_usd.amount_usd
                / (3600.0 * throughput * util)
            )
            teacher_cpr = teacher_cost_per_request_usd.amount_usd
            be = break_even_requests(incremental_cost, teacher_cpr, student_cpr)
            within_horizon = (
                False
                if be == BREAK_EVEN_NEVER
                else be <= evaluation_horizon_requests
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "utilization": util,
                    "observed_throughput_rps": {
                        "value": throughput,
                        "kind": EvidenceKind.MEASURED.value,
                        "unit": "req/s",
                    },
                    "student_serving_hourly_cost_usd": (
                        student_serving_hourly_cost_usd.to_dict()
                    ),
                    "student_cost_per_request_usd": {
                        "amount_usd": student_cpr,
                        "kind": EvidenceKind.PROJECTED.value,
                        "label": (
                            f"student_batch_{batch_size}_at_"
                            f"{int(util * 100)}pct_util"
                        ),
                    },
                    "teacher_cost_per_request_usd": (
                        teacher_cost_per_request_usd.to_dict()
                    ),
                    "savings_per_request_usd": teacher_cpr - student_cpr,
                    "break_even_requests": be,
                    "horizon_requests": evaluation_horizon_requests,
                    "within_horizon": within_horizon,
                }
            )
    return tuple(rows)


def _missing_cost(key: str) -> CostValue:
    return CostValue(
        amount_usd=None,
        kind=EvidenceKind.MISSING,
        label=key,
    )


def _cost_from_record(costs: dict[str, Any], key: str) -> CostValue:
    if key not in costs or costs[key] is None:
        return _missing_cost(key)
    raw = costs[key]
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must explicitly declare amount_usd and kind")
    if "kind" not in raw:
        raise ValueError(f"{key} must explicitly declare kind")
    amount = raw.get("amount_usd", raw.get("usd"))
    return CostValue(
        amount_usd=float(amount) if amount is not None else None,
        kind=evidence_kind(raw["kind"]),
        label=str(raw.get("label", key)),
        reason=str(raw["reason"]) if raw.get("reason") else None,
    )


def _quantity_from_record(
    costs: dict[str, Any],
    key: str,
    *,
    unit: str,
) -> LabeledValue:
    if key not in costs or costs[key] is None:
        return LabeledValue(
            value=None,
            kind=EvidenceKind.MISSING,
            unit=unit,
            label=key,
        )
    raw = costs[key]
    if not isinstance(raw, dict) or "kind" not in raw:
        raise ValueError(f"{key} must explicitly declare value and kind")
    value = raw.get("value")
    labeled = LabeledValue(
        value=float(value) if value is not None else None,
        kind=evidence_kind(raw["kind"]),
        unit=unit,
        label=str(raw.get("label", key)),
        reason=str(raw["reason"]) if raw.get("reason") else None,
    )
    if labeled.value is not None and float(labeled.value) < 0:
        raise ValueError(f"{key} must be nonnegative")
    return labeled


def build_cost_ledger(costs: dict[str, Any]) -> GrossCostLedger:
    """Parse and validate a complete gross cost ledger."""

    gross = _cost_from_record(costs, "gross_experiment_cost_usd")
    billed_seconds = _quantity_from_record(
        costs,
        "billed_training_seconds",
        unit="s",
    )
    gpu = _cost_from_record(costs, "gpu_compute_cost_usd")
    teacher_tokens = _quantity_from_record(
        costs,
        "teacher_generation_tokens",
        unit="tokens",
    )
    teacher = _cost_from_record(costs, "teacher_generation_cost_usd")
    cheap = _cost_from_record(costs, "cheap_api_benchmark_cost_usd")
    storage = _cost_from_record(costs, "storage_cost_usd")
    cheap_zero_reason = (
        str(costs["cheap_api_zero_reason"])
        if costs.get("cheap_api_zero_reason")
        else None
    )

    other_declared = "other_costs_usd" in costs
    raw_other = costs.get("other_costs_usd", {})
    if not isinstance(raw_other, dict):
        raise ValueError("other_costs_usd must be a mapping of named cost records")
    other_costs = tuple(
        _cost_from_record({name: raw}, name)
        for name, raw in sorted(raw_other.items())
    )

    gaps: list[str] = []
    required_costs = {
        "gross_experiment_cost_usd": gross,
        "gpu_compute_cost_usd": gpu,
        "teacher_generation_cost_usd": teacher,
        "cheap_api_benchmark_cost_usd": cheap,
        "storage_cost_usd": storage,
    }
    for name, cost in required_costs.items():
        if cost.kind is not EvidenceKind.MEASURED:
            gaps.append(f"{name}_not_measured")
    if billed_seconds.kind is not EvidenceKind.MEASURED:
        gaps.append("billed_training_seconds_not_measured")
    if teacher_tokens.kind is not EvidenceKind.MEASURED:
        gaps.append("teacher_generation_tokens_not_measured")
    elif not float(teacher_tokens.value).is_integer():
        gaps.append("teacher_generation_tokens_not_integer")
    if (
        cheap.kind is EvidenceKind.MEASURED
        and cheap.amount_usd == 0
        and not cheap_zero_reason
    ):
        gaps.append("cheap_api_zero_requires_reason")
    if not other_declared:
        gaps.append("other_costs_not_declared")
    for cost in other_costs:
        if cost.kind is not EvidenceKind.MEASURED:
            gaps.append(f"other_cost_{cost.label}_not_measured")

    components = (gpu, teacher, cheap, storage, *other_costs)
    if all(cost.amount_usd is not None for cost in components):
        component_sum = sum(float(cost.amount_usd) for cost in components)
    else:
        component_sum = None
    if gross.amount_usd is not None and component_sum is not None:
        tolerance = max(1e-9, abs(gross.amount_usd) * 1e-9)
        reconciled = abs(gross.amount_usd - component_sum) <= tolerance
        if not reconciled:
            gaps.append("gross_total_does_not_reconcile")
    else:
        reconciled = None
        gaps.append("gross_total_reconciliation_unavailable")

    return GrossCostLedger(
        gross_experiment_cost_usd=gross,
        billed_training_seconds=billed_seconds,
        gpu_compute_cost_usd=gpu,
        teacher_generation_tokens=teacher_tokens,
        teacher_generation_cost_usd=teacher,
        cheap_api_benchmark_cost_usd=cheap,
        cheap_api_zero_reason=cheap_zero_reason,
        storage_cost_usd=storage,
        other_costs_usd=other_costs,
        other_costs_declared=other_declared,
        component_sum_usd=component_sum,
        reconciled=reconciled,
        completeness_gaps=tuple(dict.fromkeys(gaps)),
    )


def _cost_input(
    value: CostValue | dict[str, Any] | None,
    *,
    label: str,
) -> CostValue:
    if value is None:
        return _missing_cost(label)
    if isinstance(value, CostValue):
        return value
    return _cost_from_record({label: value}, label)


def compute_economics(
    *,
    student_primary_index: float | None,
    teacher_primary_index: float | None,
    base_primary_index: float | None,
    costs: dict[str, Any],
    teacher_cost_per_request: CostValue | dict[str, Any] | None,
    student_serving_hourly_cost_usd: CostValue | dict[str, Any] | None,
    observed_throughput_rps_by_batch: dict[int, float] | None = None,
    evaluation_horizon_requests: int | None = None,
    economics_utilization: float = 0.25,
) -> EconomicsSummary:
    notes: list[str] = []
    ledger = build_cost_ledger(costs)
    teacher_cpr = _cost_input(
        teacher_cost_per_request,
        label="teacher_cost_per_request_usd",
    )
    student_hourly = _cost_input(
        student_serving_hourly_cost_usd,
        label="student_serving_hourly_cost_usd",
    )

    retention = (
        quality_retention(student_primary_index, teacher_primary_index)
        if student_primary_index is not None and teacher_primary_index is not None
        else None
    )
    gap = (
        recovered_teacher_gap(
            student_primary_index,
            base_primary_index,
            teacher_primary_index,
        )
        if (
            student_primary_index is not None
            and base_primary_index is not None
            and teacher_primary_index is not None
        )
        else None
    )
    if gap is None:
        notes.append("recovered_teacher_gap_missing_or_undefined")

    reasons: list[str] = []
    if student_primary_index is None:
        reasons.append("finalist_primary_index_missing")
    if teacher_primary_index is None:
        reasons.append("teacher_primary_index_missing")
    if base_primary_index is None:
        reasons.append("base_primary_index_missing")
    reasons.extend(ledger.completeness_gaps)
    if teacher_cpr.kind is EvidenceKind.MISSING:
        notes.append("teacher_cost_per_request_missing")
        reasons.append("teacher_cost_per_request_missing")
    if student_hourly.kind is EvidenceKind.MISSING:
        notes.append("student_serving_hourly_cost_missing")
        reasons.append("student_serving_hourly_cost_missing")
    throughput = observed_throughput_rps_by_batch or {}
    if set(throughput) != set(REQUIRED_BATCH_SIZES):
        reasons.append("finalist_batch_1_and_8_throughput_required")
    elif any(value <= 0 for value in throughput.values()):
        reasons.append("finalist_throughput_must_be_positive")
    if evaluation_horizon_requests is None:
        reasons.append("evaluation_horizon_requests_missing")
    elif evaluation_horizon_requests <= 0:
        reasons.append("evaluation_horizon_requests_nonpositive")

    incremental = ledger.incremental_distillation_cost_usd
    if incremental is None:
        reasons.append("incremental_distillation_cost_unevaluated")

    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        assert incremental is not None
        assert evaluation_horizon_requests is not None
        rows = utilization_cost_rows(
            student_serving_hourly_cost_usd=student_hourly,
            teacher_cost_per_request_usd=teacher_cpr,
            incremental_cost=incremental,
            observed_throughput_rps_by_batch=throughput,
            evaluation_horizon_requests=evaluation_horizon_requests,
        )
        break_even = tuple(
            BreakEvenResult(
                batch_size=int(row["batch_size"]),
                utilization=float(row["utilization"]),
                break_even_requests=row["break_even_requests"],
                savings_per_request_usd=float(row["savings_per_request_usd"]),
                incremental_cost_usd=incremental,
                teacher_cost_per_request_usd=float(
                    teacher_cpr.amount_usd
                ),
                teacher_cost_kind=teacher_cpr.kind,
                student_cost_per_request_usd=float(
                    row["student_cost_per_request_usd"]["amount_usd"]
                ),
                student_cost_kind=EvidenceKind.PROJECTED,
                observed_throughput_rps=float(
                    row["observed_throughput_rps"]["value"]
                ),
                horizon_requests=evaluation_horizon_requests,
                within_horizon=bool(row["within_horizon"]),
            )
            for row in rows
            if float(row["utilization"]) == economics_utilization
        )
        evaluated = True
    else:
        rows = ()
        break_even = ()
        evaluated = False

    notes.append("serving_costs_are_projected_not_measured_production_savings")
    return EconomicsSummary(
        cost_ledger=ledger,
        teacher_cost_per_request_usd=teacher_cpr,
        student_serving_hourly_cost_usd=student_hourly,
        quality_retention=retention,
        recovered_teacher_gap=gap,
        recovered_teacher_gap_defined=gap is not None,
        evaluated=evaluated,
        unevaluated_reasons=tuple(reasons),
        utilization_rows=rows,
        break_even_at_25pct=break_even,
        notes=tuple(notes),
    )
