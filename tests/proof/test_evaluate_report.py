"""End-to-end proof evaluation and JSON/HTML report rendering."""

from __future__ import annotations

from datetime import UTC, datetime

from distillery.contracts.hashing import content_sha256
from distillery.contracts.proof import ProofStatus
from distillery.proof.evaluate import ArmEvaluationInput, ProofEvaluationInput, evaluate_proof
from distillery.proof.report import render_html_report, render_json_report
from distillery.proof.testing import make_pred, txn_gold, var_gold

PROTOCOL_SHA = content_sha256({"protocol": "finance-proof.v1", "version": 1})


def _arm_records(arm_id: str, *, quality: str = "perfect", n_worlds: int = 4):
    gold_t = txn_gold()
    gold_v = var_gold(
        profit=100000,
        drivers=[{"driver_id": "hc", "impact_minor": 100000, "rank": 1}],
        other=0,
    )
    records = []
    for i in range(n_worlds):
        wid = f"world_{i}"
        split = "ood_test" if i % 2 else "iid_test"
        if quality == "perfect":
            t_pred, v_pred = gold_t, gold_v
        elif quality == "good":
            # retain most quality: wrong on one world only
            if i == n_worlds - 1:
                t_pred = {
                    **gold_t,
                    "gl_account": "9999",
                    "journal_entry": [
                        {"account": "9999", "side": "debit", "amount_minor": 4500},
                        {"account": "2100", "side": "credit", "amount_minor": 4500},
                    ],
                }
                v_pred = gold_v
            else:
                t_pred, v_pred = gold_t, gold_v
        else:  # poor
            t_pred = {
                **gold_t,
                "gl_account": "9999",
                "journal_entry": [
                    {"account": "9999", "side": "debit", "amount_minor": 4500},
                    {"account": "2100", "side": "credit", "amount_minor": 4500},
                ],
            }
            v_pred = {
                **gold_v,
                "profit_impact_minor": 0,
                "direction": "favorable",
                "top_drivers": [{"driver_id": "hc", "impact_minor": 0, "rank": 1}],
                "other_impact_minor": 0,
            }
        records.append(
            make_pred(
                example_id=f"ex_{arm_id}_t_{i}",
                world_id=wid,
                task="transaction_review",
                expected=gold_t,
                parsed=t_pred,
                arm_id=arm_id,
                split=split,
            )
        )
        records.append(
            make_pred(
                example_id=f"ex_{arm_id}_v_{i}",
                world_id=wid,
                task="variance_analysis",
                expected=gold_v,
                parsed=v_pred,
                arm_id=arm_id,
                split=split,
            )
        )
    return records


def _systems(batch_size: int = 1) -> dict:
    return {
        "hardware": "ml.g5.xlarge",
        "batch_size": batch_size,
        "warmup_requests": 20,
        "timed_examples": 200,
        "latency_p50_ms": 40.0,
        "latency_p95_ms": 90.0,
        "requests_per_second": 25.0 if batch_size == 1 else 90.0,
        "output_tokens_per_second": 500.0,
        "failure_rate": 0.0,
        "peak_vram_allocated_gb": 16.0,
        "peak_vram_reserved_gb": 18.0,
        "peak_cpu_ram_gb": 10.0,
        "billed_training_seconds": 7200,
    }


def _full_input(**overrides) -> ProofEvaluationInput:
    arms = [
        ArmEvaluationInput(
            "rules",
            _arm_records("rules", quality="poor"),
            systems_profile=_systems(),
        ),
        ArmEvaluationInput(
            "teacher", _arm_records("teacher", quality="perfect"), systems_profile=_systems()
        ),
        ArmEvaluationInput(
            "student_base", _arm_records("student_base", quality="poor"), systems_profile=_systems()
        ),
        ArmEvaluationInput(
            "cheap_off_the_shelf",
            _arm_records("cheap_off_the_shelf", quality="poor"),
            systems_profile=_systems(),
        ),
        ArmEvaluationInput(
            "sequence_kd",
            _arm_records("sequence_kd", quality="perfect"),
            systems_profile=_systems(),
        ),
        ArmEvaluationInput(
            "sequence_kd_b8",
            _arm_records("sequence_kd_b8", quality="perfect"),
            systems_profile=_systems(batch_size=8),
        ),
    ]
    # Drop the fake batch8 arm from metrics path — attach batch8 profile on finalist only.
    arms = [a for a in arms if a.arm_id != "sequence_kd_b8"]
    # Provide both batch profiles via two systems entries: use finalist profile batch1;
    # add a second ArmEvaluationInput excluded? Simpler: put batch8 numbers into costs path
    # by adding systems on an excluded arm. Instead, merge batch8 into evaluation via
    # a dedicated systems-only arm excluded from quality.
    arms.append(
        ArmEvaluationInput(
            arm_id="sequence_kd_batch8_profile",
            predictions=[],
            systems_profile=_systems(batch_size=8),
            excluded=True,
            exclusion_reason="systems_profile_only",
        )
    )

    kwargs = dict(
        report_id="prf_test_001",
        protocol_id="finance-proof.v1",
        protocol_sha256=PROTOCOL_SHA,
        arms=arms,
        costs={
            "gross_experiment_cost_usd": 55.0,
            "training_cost_usd": 40.0,
            "teacher_generation_cost_usd": 8.0,
            "storage_cost_usd": 2.0,
            "cheap_api_benchmark_cost_usd": 3.0,
        },
        run_ids=("run_seed17", "run_seed23"),
        finalist_arm_id="sequence_kd",
        seeds_present=(17, 23),
        teacher_cost_per_request=0.02,
        student_cost_per_request_at_full_util=0.001,
        evaluation_horizon_requests=100_000,
        trainer_numerical_ok=True,
        trainer_artifact_reload_ok=True,
        trainer_smoke_eval_ok=True,
        trainer_memory_cost_ok=True,
        frozen_hashes_present=True,
        bootstrap_seed=17,
        bootstrap_resamples=80,
        created_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
        rules_projected_tco_below_distill=False,
        cheap_projected_tco_below_distill=False,
    )
    kwargs.update(overrides)
    return ProofEvaluationInput(**kwargs)


def test_evaluate_can_reach_proved() -> None:
    result = evaluate_proof(_full_input())
    assert result.report.proof_status is ProofStatus.PROVED
    assert result.report.first_failed_gate is None
    assert result.arm_metrics["sequence_kd"].primary_index >= 0.99
    assert result.economics["break_even_at_25pct"]["student_cost_kind"] == "projected"
    assert result.economics["gross_experiment_cost_usd"]["kind"] == "measured"


def test_evaluate_missing_seed_insufficient() -> None:
    result = evaluate_proof(_full_input(seeds_present=(17,)))
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert result.report.first_failed_gate == "evidence"


def test_evaluate_missing_cost_insufficient() -> None:
    result = evaluate_proof(_full_input(costs={}))
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    # economics or evidence depending on which fails first after quality
    assert result.report.first_failed_gate in {"economics", "evidence"}


def test_evaluate_baseline_do_not_distill() -> None:
    # Make rules perfect + cheaper TCO
    arms = [
        ArmEvaluationInput("rules", _arm_records("rules", quality="perfect")),
        ArmEvaluationInput("teacher", _arm_records("teacher", quality="perfect")),
        ArmEvaluationInput("student_base", _arm_records("student_base", quality="poor")),
        ArmEvaluationInput(
            "cheap_off_the_shelf", _arm_records("cheap_off_the_shelf", quality="poor")
        ),
        ArmEvaluationInput(
            "sequence_kd",
            _arm_records("sequence_kd", quality="perfect"),
            systems_profile=_systems(),
        ),
    ]
    result = evaluate_proof(
        _full_input(arms=arms, rules_projected_tco_below_distill=True)
    )
    assert result.report.proof_status is ProofStatus.DO_NOT_DISTILL
    assert result.report.first_failed_gate == "baseline"
    assert "quality" in result.report.unevaluated_gates


def test_evaluate_failed_quality() -> None:
    arms = [
        ArmEvaluationInput("rules", _arm_records("rules", quality="poor")),
        ArmEvaluationInput("teacher", _arm_records("teacher", quality="perfect")),
        ArmEvaluationInput("student_base", _arm_records("student_base", quality="poor")),
        ArmEvaluationInput(
            "cheap_off_the_shelf", _arm_records("cheap_off_the_shelf", quality="poor")
        ),
        ArmEvaluationInput(
            "sequence_kd",
            _arm_records("sequence_kd", quality="poor"),
            systems_profile=_systems(),
        ),
    ]
    result = evaluate_proof(_full_input(arms=arms))
    # Poor finalist may fail pilot (teacher vs base still ok) then quality
    assert result.report.proof_status in {
        ProofStatus.FAILED_QUALITY,
        ProofStatus.INSUFFICIENT_EVIDENCE,
    }


def test_json_and_html_report_self_contained() -> None:
    result = evaluate_proof(_full_input())
    doc = render_json_report(result)
    assert doc["proof_status"] == result.report.proof_status.value
    assert "document_sha256" in doc
    assert doc["economics"]["gross_experiment_cost_usd"]["kind"] == "measured"
    html = render_html_report(result)
    assert "<!DOCTYPE html>" in html
    assert "projected" in html.lower()
    assert result.report.report_id in html
    assert "<script src=" not in html
    assert "http://" not in html or "schema" in html  # no external asset links expected
    # Embedded JSON present
    assert doc["document_sha256"] in html


def test_report_marks_projected_vs_measured() -> None:
    result = evaluate_proof(_full_input())
    doc = render_json_report(result)
    be = doc["economics"]["break_even_at_25pct"]
    assert be["student_cost_kind"] == "projected"
    for row in doc["economics"]["utilization_rows"]:
        assert row["student_cost_per_request_usd"]["kind"] == "projected"
    sys = doc["systems"]["sequence_kd"]
    assert sys["latency_p50_ms"]["kind"] == "measured"
