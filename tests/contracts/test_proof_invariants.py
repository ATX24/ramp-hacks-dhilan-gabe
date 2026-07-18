"""Cross-field proof-report and ordered-gate invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.proof import (
    PROOF_GATE_ORDER,
    ArmResult,
    ProofReport,
    ProofStatus,
    QualityGateResult,
)

HEX64 = "a" * 64


def _gates(
    *,
    failed_gate: str | None = None,
) -> tuple[QualityGateResult, ...]:
    failed_index = PROOF_GATE_ORDER.index(failed_gate) if failed_gate is not None else None
    results: list[QualityGateResult] = []
    for index, gate_id in enumerate(PROOF_GATE_ORDER):
        if failed_index is None or index < failed_index:
            results.append(
                QualityGateResult(
                    gate_id=gate_id,
                    passed=True,
                    evaluated=True,
                    detail="passed",
                )
            )
        elif index == failed_index:
            results.append(
                QualityGateResult(
                    gate_id=gate_id,
                    passed=False,
                    evaluated=True,
                    detail="failed",
                )
            )
        else:
            results.append(
                QualityGateResult(
                    gate_id=gate_id,
                    passed=None,
                    evaluated=False,
                    detail="not evaluated",
                )
            )
    return tuple(results)


def _report(**updates: object) -> ProofReport:
    payload: dict[str, object] = {
        "report_id": "prf_proof_invariants_001",
        "run_ids": ("run_proof_invariants_001",),
        "protocol_id": "finance-proof.v1",
        "protocol_sha256": HEX64,
        "proof_status": ProofStatus.PROVED,
        "first_failed_gate": None,
        "unevaluated_gates": (),
        "arm_results": (
            ArmResult(
                arm_id="student",
                primary_index=0.95,
                metrics={"nested": {"samples": [1, 2]}},
            ),
        ),
        "quality_gates": _gates(),
        "uncertainty": {"bootstrap": {"resamples": 10_000}},
        "economics": {"break_even_requests": 1_000},
        "created_at": datetime(2026, 7, 18, tzinfo=UTC),
    }
    payload.update(updates)
    return ProofReport.model_validate(payload)


def test_proved_requires_every_required_gate_to_pass() -> None:
    report = _report()
    assert report.proof_status is ProofStatus.PROVED
    with pytest.raises(ValidationError, match="proved cannot"):
        _report(
            quality_gates=_gates(failed_gate="quality"),
            first_failed_gate="quality",
            unevaluated_gates=PROOF_GATE_ORDER[4:],
        )


def test_first_failed_and_unevaluated_lists_must_match_results() -> None:
    gates = _gates(failed_gate="quality")
    with pytest.raises(ValidationError, match="first_failed_gate"):
        _report(
            proof_status=ProofStatus.FAILED_QUALITY,
            quality_gates=gates,
            first_failed_gate="economics",
            unevaluated_gates=PROOF_GATE_ORDER[4:],
        )
    with pytest.raises(ValidationError, match="unevaluated_gates"):
        _report(
            proof_status=ProofStatus.FAILED_QUALITY,
            quality_gates=gates,
            first_failed_gate="quality",
            unevaluated_gates=("evidence",),
        )


def test_gate_results_are_complete_unique_and_ordered() -> None:
    reversed_gates = tuple(reversed(_gates()))
    with pytest.raises(ValidationError, match="required order"):
        _report(quality_gates=reversed_gates)
    with pytest.raises(ValidationError, match="required order"):
        _report(quality_gates=_gates()[:-1])


@pytest.mark.parametrize(
    ("status", "failed_gate"),
    [
        (ProofStatus.DO_NOT_DISTILL, "baseline"),
        (ProofStatus.FAILED_QUALITY, "quality"),
        (ProofStatus.FAILED_ECONOMICS, "economics"),
        (ProofStatus.INSUFFICIENT_EVIDENCE, "evidence"),
    ],
)
def test_nonproved_status_matches_failed_gate(
    status: ProofStatus,
    failed_gate: str,
) -> None:
    index = PROOF_GATE_ORDER.index(failed_gate)
    report = _report(
        proof_status=status,
        quality_gates=_gates(failed_gate=failed_gate),
        first_failed_gate=failed_gate,
        unevaluated_gates=PROOF_GATE_ORDER[index + 1 :],
    )
    assert report.first_failed_gate == failed_gate


def test_nested_proof_evidence_is_deeply_immutable() -> None:
    report = _report()
    before = report.resource_hash()
    with pytest.raises(TypeError):
        report.uncertainty["bootstrap"]["resamples"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        report.arm_results[0].metrics["nested"]["samples"][0] = 9  # type: ignore[index]
    assert report.resource_hash() == before
