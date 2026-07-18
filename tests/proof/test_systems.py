"""Systems summary measured/missing labeling."""

from __future__ import annotations

from distillery.proof.systems import summarize_systems


def test_systems_computes_percentiles_from_samples() -> None:
    # 1..100 → p50≈50.5, p95≈95.05 with linear interpolation
    lats = [float(i) for i in range(1, 101)]
    summary = summarize_systems(
        {
            "hardware": "ml.g5.xlarge",
            "batch_size": 1,
            "warmup_requests": 20,
            "timed_examples": 200,
            "requests_per_second": 15.0,
            "output_tokens_per_second": 400.0,
            "failure_rate": 0.0,
            "peak_vram_allocated_gb": 18.0,
            "peak_vram_reserved_gb": 20.0,
            "peak_cpu_ram_gb": 8.0,
            "billed_training_seconds": 3600,
        },
        latencies_ms=lats,
    )
    assert summary.latency_p50_ms.kind == "measured"
    assert summary.latency_p95_ms.kind == "measured"
    assert abs(float(summary.latency_p50_ms.value) - 50.5) < 1e-9
    assert summary.gpu_hours.kind == "measured"
    assert float(summary.gpu_hours.value) == 1.0
    assert "gpu_hours_derived_from_billed_training_seconds" in summary.notes


def test_systems_marks_missing_fields() -> None:
    summary = summarize_systems({"batch_size": 8, "warmup_requests": 5, "timed_examples": 10})
    assert summary.latency_p50_ms.kind == "missing"
    assert summary.requests_per_second.kind == "missing"
    assert "timed_examples_below_protocol_minimum_200" in summary.notes
    assert "warmup_requests_below_protocol_minimum_20" in summary.notes
