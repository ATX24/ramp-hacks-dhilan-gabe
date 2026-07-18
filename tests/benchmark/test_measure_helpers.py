"""Pure helpers for benchmark measurement (no GPU / no network)."""

from __future__ import annotations

from experiments.benchmark.stats import percentile


def test_percentile_interpolation() -> None:
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 0.50) == 50.5
    assert percentile([], 0.50) is None
    assert percentile([7.0], 0.95) == 7.0
