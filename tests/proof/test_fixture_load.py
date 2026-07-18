"""Load immutable prediction fixtures into metrics/report path."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from distillery.contracts.hashing import content_sha256
from distillery.proof.evaluate import (
    ArmEvaluationInput,
    ProofEvaluationInput,
    evaluate_proof,
    file_sha256,
    load_predictions_jsonl,
)
from distillery.proof.metrics import PredictionRecord, compute_arm_metrics
from distillery.proof.report import render_html_report, render_json_report

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_predictions.jsonl"


def test_load_sample_predictions_fixture() -> None:
    records = load_predictions_jsonl(FIXTURE, arm_id="teacher")
    assert len(records) == 4
    metrics = compute_arm_metrics("teacher", records)
    # 2/4 schema-valid (invalid json + cash/txn/var: wait 3 valid + 1 invalid)
    # txn easy, var easy, cash medium valid; invalid_json fails parse
    assert metrics.json_parse_rate == 0.75
    assert metrics.json_schema_validity == 0.75
    assert metrics.transaction_joint_exact == 0.5  # 1 ok, 1 invalid among txn
    assert metrics.cash_joint_exact == 1.0
    digest = file_sha256(FIXTURE)
    assert len(digest) == 64


def test_loader_requires_raw_text_and_explicit_provenance() -> None:
    record = load_predictions_jsonl(FIXTURE)[0]
    payload = record.model_dump()
    payload.pop("raw_text_provenance")
    with pytest.raises(ValueError, match="raw_text_provenance"):
        PredictionRecord.model_validate(payload)
    payload = record.model_dump()
    payload.pop("raw_text")
    with pytest.raises(ValueError, match="raw_text"):
        PredictionRecord.model_validate(payload)


def test_fixture_feeds_report_render() -> None:
    records = load_predictions_jsonl(FIXTURE)
    # Minimal multi-arm harness using the same fixture rows as every arm.
    arms = [
        ArmEvaluationInput(
            arm_id=aid,
            predictions=[
                record.model_copy(update={"arm_id": aid, "seed": seed})
                for seed in (17, 23)
                for record in records
            ],
        )
        for aid in ("rules", "teacher", "student_base", "cheap_off_the_shelf", "sequence_kd")
    ]
    result = evaluate_proof(
        ProofEvaluationInput(
            report_id="prf_fixture_smoke",
            protocol_id="finance-proof.v1",
            protocol_sha256=content_sha256({"id": "finance-proof.v1"}),
            arms=arms,
            costs={},
            created_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
            run_ids=("run_fixture_smoke",),
            bootstrap_resamples=20,
            frozen_hashes_present=False,
        )
    )
    doc = render_json_report(result)
    html = render_html_report(result)
    assert doc["schema_version"] == "distillery.proof_report.v1"
    assert "prf_fixture_smoke" in html
    # Missing seeds/costs must not prove.
    assert doc["proof_status"] == "insufficient_evidence"
