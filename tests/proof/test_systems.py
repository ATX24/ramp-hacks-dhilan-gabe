"""Systems profile completeness and provenance tests."""

from __future__ import annotations

import pytest

from distillery.proof.evidence import EvidenceKind, LabeledValue
from distillery.proof.systems import summarize_systems
from distillery.proof.testing import complete_systems_profile


def test_systems_computes_percentiles_from_samples() -> None:
    lats = [float(i) for i in range(1, 101)]
    profile = complete_systems_profile(batch_size=1, requests_per_second=15.0)
    profile.pop("latency_p50_ms")
    profile.pop("latency_p95_ms")
    summary = summarize_systems(profile, latencies_ms=lats)
    assert summary.latency_p50_ms.kind is EvidenceKind.MEASURED
    assert summary.latency_p95_ms.kind is EvidenceKind.MEASURED
    assert abs(float(summary.latency_p50_ms.value) - 50.5) < 1e-9
    assert summary.gpu_hours.kind is EvidenceKind.MEASURED
    assert float(summary.gpu_hours.value) == 2.0
    assert summary.proof_ready is True


@pytest.mark.parametrize("batch_size", [1, 8])
def test_each_required_batch_profile_is_proof_ready(batch_size: int) -> None:
    summary = summarize_systems(
        complete_systems_profile(
            batch_size=batch_size,
            requests_per_second=20.0 * batch_size,
        )
    )
    assert summary.hardware == "ml.g5.xlarge"
    assert summary.runtime == "transformers-4.46/cuda-12.4"
    assert summary.proof_ready is True
    assert summary.proof_evidence_gaps() == ()


@pytest.mark.parametrize(
    ("mutation", "gap"),
    [
        ({"hardware": None}, "hardware_missing"),
        ({"runtime": None}, "runtime_missing"),
        (
            {"warmup_requests_by_task": {"transaction_review": 19, "variance_analysis": 20}},
            "transaction_review_warmups_below_20",
        ),
        (
            {
                "timed_examples_by_task": {
                    "transaction_review": 200,
                    "variance_analysis": 199,
                }
            },
            "variance_analysis_timed_examples_below_200",
        ),
        ({"requests_per_second": None}, "requests_per_second_not_measured"),
    ],
)
def test_missing_profile_evidence_is_explicit(
    mutation: dict,
    gap: str,
) -> None:
    profile = complete_systems_profile(batch_size=1, requests_per_second=20.0)
    profile.update(mutation)
    summary = summarize_systems(profile)
    assert summary.proof_ready is False
    assert gap in summary.proof_evidence_gaps()


def test_aggregate_counts_do_not_substitute_for_per_task_counts() -> None:
    profile = complete_systems_profile(batch_size=1, requests_per_second=20.0)
    profile.pop("warmup_requests_by_task")
    profile.pop("timed_examples_by_task")
    profile["warmup_requests"] = 100
    profile["timed_examples"] = 10_000
    summary = summarize_systems(profile)
    assert summary.proof_ready is False
    assert "transaction_review_warmups_below_20" in summary.proof_evidence_gaps()
    assert (
        "variance_analysis_timed_examples_below_200"
        in summary.proof_evidence_gaps()
    )


def test_aggregate_counts_cannot_contradict_per_task_counts() -> None:
    profile = complete_systems_profile(batch_size=1, requests_per_second=20.0)
    profile["warmup_requests"] = 20
    profile["timed_examples"] = 200
    summary = summarize_systems(profile)
    assert summary.proof_ready is False
    assert "aggregate_warmups_below_per_task_sum" in summary.proof_evidence_gaps()
    assert (
        "aggregate_timed_examples_below_per_task_sum"
        in summary.proof_evidence_gaps()
    )


def test_projected_throughput_does_not_count_as_observed() -> None:
    profile = complete_systems_profile(batch_size=8, requests_per_second=80.0)
    profile["requests_per_second"] = {
        "value": 80.0,
        "kind": "projected",
    }
    summary = summarize_systems(profile)
    assert summary.requests_per_second.kind is EvidenceKind.PROJECTED
    assert summary.proof_ready is False
    assert "requests_per_second_not_measured" in summary.proof_evidence_gaps()


def test_unknown_systems_kind_fails_loud() -> None:
    profile = complete_systems_profile(batch_size=1, requests_per_second=20.0)
    profile["requests_per_second"] = {
        "value": 20.0,
        "kind": "approximate",
    }
    with pytest.raises(ValueError, match="not a valid EvidenceKind"):
        summarize_systems(profile)


def test_labeled_value_rejects_inconsistent_missing_state() -> None:
    with pytest.raises(ValueError, match="missing evidence"):
        LabeledValue(1.0, EvidenceKind.MISSING)
