"""Latency, throughput, and resource summaries from systems profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ValueKind = Literal["measured", "projected", "missing"]


@dataclass(frozen=True)
class LabeledValue:
    value: float | int | str | None
    kind: ValueKind
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "kind": self.kind, "unit": self.unit}


@dataclass(frozen=True)
class SystemsSummary:
    """Systems metrics with explicit measured vs projected labeling."""

    hardware: str | None
    batch_size: int
    warmup_requests: int
    timed_examples: int
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware,
            "batch_size": self.batch_size,
            "warmup_requests": self.warmup_requests,
            "timed_examples": self.timed_examples,
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


def _measured(value: float | int | None, unit: str | None = None) -> LabeledValue:
    if value is None:
        return LabeledValue(value=None, kind="missing", unit=unit)
    return LabeledValue(value=value, kind="measured", unit=unit)


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
    warmup = int(profile.get("warmup_requests", 0) or 0)
    timed = int(profile.get("timed_examples", 0) or 0)

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
        if p50 is not None:
            p50 = float(p50)
        if p95 is not None:
            p95 = float(p95)

    rps = profile.get("requests_per_second")
    tok_s = profile.get("output_tokens_per_second")
    failure = profile.get("failure_rate")
    if failure is None and "failures" in profile and timed:
        failure = float(profile["failures"]) / float(timed)

    if timed < 200:
        notes.append("timed_examples_below_protocol_minimum_200")
    if warmup < 20:
        notes.append("warmup_requests_below_protocol_minimum_20")

    wall = profile.get("wall_time_seconds")
    billed = profile.get("billed_training_seconds")
    gpu_hours = profile.get("gpu_hours")
    if gpu_hours is None and billed is not None:
        gpu_hours = float(billed) / 3600.0
        notes.append("gpu_hours_derived_from_billed_training_seconds")

    return SystemsSummary(
        hardware=str(hardware) if hardware is not None else None,
        batch_size=int(profile.get("batch_size", batch_size)),
        warmup_requests=warmup,
        timed_examples=timed,
        latency_p50_ms=_measured(p50, "ms"),
        latency_p95_ms=_measured(p95, "ms"),
        requests_per_second=_measured(float(rps) if rps is not None else None, "req/s"),
        output_tokens_per_second=_measured(
            float(tok_s) if tok_s is not None else None, "tok/s"
        ),
        failure_rate=_measured(float(failure) if failure is not None else None, "rate"),
        peak_vram_allocated_gb=_measured(
            float(profile["peak_vram_allocated_gb"])
            if profile.get("peak_vram_allocated_gb") is not None
            else None,
            "GiB",
        ),
        peak_vram_reserved_gb=_measured(
            float(profile["peak_vram_reserved_gb"])
            if profile.get("peak_vram_reserved_gb") is not None
            else None,
            "GiB",
        ),
        peak_cpu_ram_gb=_measured(
            float(profile["peak_cpu_ram_gb"])
            if profile.get("peak_cpu_ram_gb") is not None
            else None,
            "GiB",
        ),
        wall_time_seconds=_measured(float(wall) if wall is not None else None, "s"),
        billed_training_seconds=_measured(float(billed) if billed is not None else None, "s"),
        gpu_hours=_measured(float(gpu_hours) if gpu_hours is not None else None, "GPU-h"),
        notes=tuple(notes),
    )
