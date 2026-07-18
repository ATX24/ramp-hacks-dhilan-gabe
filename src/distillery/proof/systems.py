"""Latency, throughput, and resource summaries from systems profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distillery.contracts.budgets import EvaluationBudget
from distillery.contracts.tasks import TaskId
from distillery.proof.evidence import EvidenceKind, LabeledValue, evidence_kind

PRIMARY_TASKS = (
    TaskId.TRANSACTION_REVIEW.value,
    TaskId.VARIANCE_ANALYSIS.value,
)


@dataclass(frozen=True)
class SystemsSummary:
    """One arm/batch systems profile with explicit evidence provenance."""

    hardware: str | None
    runtime: str | None
    batch_size: int
    warmup_requests: int
    timed_examples: int
    warmup_requests_by_task: dict[str, int]
    timed_examples_by_task: dict[str, int]
    latency_p50_ms: LabeledValue
    latency_p95_ms: LabeledValue
    requests_per_second: LabeledValue
    output_tokens_per_second: LabeledValue
    failure_rate: LabeledValue
    peak_vram_allocated_gb: LabeledValue
    peak_vram_reserved_gb: LabeledValue
    peak_cpu_ram_gb: LabeledValue
    wall_time_seconds: LabeledValue
    billed_training_seconds: LabeledValue
    gpu_hours: LabeledValue
    notes: tuple[str, ...] = ()

    def proof_evidence_gaps(self) -> tuple[str, ...]:
        """Return every reason this profile cannot support a proof claim."""

        budget = EvaluationBudget()
        gaps: list[str] = []
        if not self.hardware:
            gaps.append("hardware_missing")
        if not self.runtime:
            gaps.append("runtime_missing")
        if self.batch_size not in (1, 8):
            gaps.append("batch_size_must_be_1_or_8")
        for task in PRIMARY_TASKS:
            if self.warmup_requests_by_task.get(task, 0) < budget.warmup_requests:
                gaps.append(f"{task}_warmups_below_{budget.warmup_requests}")
            if (
                self.timed_examples_by_task.get(task, 0)
                < budget.timed_examples_per_primary_task
            ):
                gaps.append(
                    f"{task}_timed_examples_below_"
                    f"{budget.timed_examples_per_primary_task}"
                )
        if self.warmup_requests < sum(self.warmup_requests_by_task.values()):
            gaps.append("aggregate_warmups_below_per_task_sum")
        if self.timed_examples < sum(self.timed_examples_by_task.values()):
            gaps.append("aggregate_timed_examples_below_per_task_sum")
        required_measurements = {
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "requests_per_second": self.requests_per_second,
            "output_tokens_per_second": self.output_tokens_per_second,
            "failure_rate": self.failure_rate,
        }
        for name, value in required_measurements.items():
            if value.kind is not EvidenceKind.MEASURED:
                gaps.append(f"{name}_not_measured")
        if (
            self.requests_per_second.kind is EvidenceKind.MEASURED
            and float(self.requests_per_second.value) <= 0
        ):
            gaps.append("requests_per_second_nonpositive")
        return tuple(gaps)

    @property
    def proof_ready(self) -> bool:
        return not self.proof_evidence_gaps()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware,
            "runtime": self.runtime,
            "batch_size": self.batch_size,
            "warmup_requests": self.warmup_requests,
            "timed_examples": self.timed_examples,
            "warmup_requests_by_task": dict(self.warmup_requests_by_task),
            "timed_examples_by_task": dict(self.timed_examples_by_task),
            "latency_p50_ms": self.latency_p50_ms.to_dict(),
            "latency_p95_ms": self.latency_p95_ms.to_dict(),
            "requests_per_second": self.requests_per_second.to_dict(),
            "output_tokens_per_second": self.output_tokens_per_second.to_dict(),
            "failure_rate": self.failure_rate.to_dict(),
            "peak_vram_allocated_gb": self.peak_vram_allocated_gb.to_dict(),
            "peak_vram_reserved_gb": self.peak_vram_reserved_gb.to_dict(),
            "peak_cpu_ram_gb": self.peak_cpu_ram_gb.to_dict(),
            "wall_time_seconds": self.wall_time_seconds.to_dict(),
            "billed_training_seconds": self.billed_training_seconds.to_dict(),
            "gpu_hours": self.gpu_hours.to_dict(),
            "proof_ready": self.proof_ready,
            "proof_evidence_gaps": list(self.proof_evidence_gaps()),
            "notes": list(self.notes),
        }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _labeled(
    value: Any,
    unit: str | None = None,
    *,
    label: str | None = None,
) -> LabeledValue:
    if value is None:
        return LabeledValue(
            value=None,
            kind=EvidenceKind.MISSING,
            unit=unit,
            label=label,
        )
    if isinstance(value, dict):
        raw_value = value.get("value")
        kind = evidence_kind(value.get("kind", EvidenceKind.MISSING))
        normalized_value = (
            float(raw_value)
            if raw_value is not None and isinstance(raw_value, (int, float, str))
            else raw_value
        )
        return LabeledValue(
            value=normalized_value,
            kind=kind,
            unit=str(value.get("unit", unit)) if value.get("unit", unit) else None,
            label=str(value.get("label", label)) if value.get("label", label) else None,
            reason=str(value["reason"]) if value.get("reason") else None,
        )
    return LabeledValue(
        value=float(value),
        kind=EvidenceKind.MEASURED,
        unit=unit,
        label=label,
    )


def summarize_systems(
    profile: dict[str, Any],
    *,
    latencies_ms: list[float] | None = None,
    batch_size: int = 1,
) -> SystemsSummary:
    """Build a systems summary from a systems/profile.json-like dict.

    Missing required measured fields are labeled ``missing`` (never invented).
    """
    notes: list[str] = []
    hardware = profile.get("hardware") or profile.get("instance_type")
    runtime = profile.get("runtime")
    warmups_by_task = {
        str(k): int(v)
        for k, v in (profile.get("warmup_requests_by_task") or {}).items()
    }
    timed_by_task = {
        str(k): int(v)
        for k, v in (profile.get("timed_examples_by_task") or {}).items()
    }
    warmup = int(profile.get("warmup_requests", sum(warmups_by_task.values())) or 0)
    timed = int(profile.get("timed_examples", sum(timed_by_task.values())) or 0)

    lats = latencies_ms
    if lats is None and "latencies_ms" in profile:
        lats = [float(x) for x in profile["latencies_ms"]]
    if lats is None and "latency_samples_ms" in profile:
        lats = [float(x) for x in profile["latency_samples_ms"]]

    if lats:
        p50 = _percentile(lats, 0.50)
        p95 = _percentile(lats, 0.95)
    else:
        p50 = profile.get("latency_p50_ms")
        p95 = profile.get("latency_p95_ms")

    rps = profile.get("requests_per_second")
    tok_s = profile.get("output_tokens_per_second")
    failure = profile.get("failure_rate")
    if failure is None and "failures" in profile and timed:
        failure = float(profile["failures"]) / float(timed)

    budget = EvaluationBudget()
    if timed < budget.timed_examples_per_primary_task:
        notes.append(
            "timed_examples_below_protocol_minimum_"
            f"{budget.timed_examples_per_primary_task}"
        )
    if warmup < budget.warmup_requests:
        notes.append(
            f"warmup_requests_below_protocol_minimum_{budget.warmup_requests}"
        )

    wall = profile.get("wall_time_seconds")
    billed = profile.get("billed_training_seconds")
    gpu_hours = profile.get("gpu_hours")
    if gpu_hours is None and billed is not None:
        billed_value = _labeled(
            billed,
            "s",
            label="billed_training_seconds",
        )
        if billed_value.kind is EvidenceKind.MEASURED:
            gpu_hours = float(billed_value.value) / 3600.0
            notes.append("gpu_hours_derived_from_billed_training_seconds")

    return SystemsSummary(
        hardware=str(hardware) if hardware is not None else None,
        runtime=str(runtime) if runtime is not None else None,
        batch_size=int(profile.get("batch_size", batch_size)),
        warmup_requests=warmup,
        timed_examples=timed,
        warmup_requests_by_task=warmups_by_task,
        timed_examples_by_task=timed_by_task,
        latency_p50_ms=_labeled(p50, "ms", label="latency_p50_ms"),
        latency_p95_ms=_labeled(p95, "ms", label="latency_p95_ms"),
        requests_per_second=_labeled(
            rps,
            "req/s",
            label="requests_per_second",
        ),
        output_tokens_per_second=_labeled(
            tok_s,
            "tok/s",
            label="output_tokens_per_second",
        ),
        failure_rate=_labeled(failure, "rate", label="failure_rate"),
        peak_vram_allocated_gb=_labeled(
            profile.get("peak_vram_allocated_gb"),
            "GiB",
            label="peak_vram_allocated_gb",
        ),
        peak_vram_reserved_gb=_labeled(
            profile.get("peak_vram_reserved_gb"),
            "GiB",
            label="peak_vram_reserved_gb",
        ),
        peak_cpu_ram_gb=_labeled(
            profile.get("peak_cpu_ram_gb"),
            "GiB",
            label="peak_cpu_ram_gb",
        ),
        wall_time_seconds=_labeled(
            wall,
            "s",
            label="wall_time_seconds",
        ),
        billed_training_seconds=_labeled(
            billed,
            "s",
            label="billed_training_seconds",
        ),
        gpu_hours=_labeled(
            gpu_hours,
            "GPU-h",
            label="gpu_hours",
        ),
        notes=tuple(notes),
    )
