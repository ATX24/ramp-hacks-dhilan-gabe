"""End-to-end fail-closed evaluation and deterministic report rendering."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from distillery.contracts.hashing import content_sha256
from distillery.contracts.proof import ProofStatus
from distillery.proof.economics import CostValue
from distillery.proof.evaluate import (
    ArmEvaluationInput,
    ProofEvaluationInput,
    SystemsProfileInput,
    evaluate_proof,
)
from distillery.proof.evidence import EvidenceKind
from distillery.proof.gates import BaselineTCOComparison
from distillery.proof.report import render_html_report, render_json_report
from distillery.proof.testing import (
    complete_cost_ledger,
    complete_systems_profile,
    make_pred,
    txn_gold,
    var_gold,
)

PROTOCOL_SHA = content_sha256({"protocol": "finance-proof.v1", "version": 1})
CREATED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _arm_records(
    arm_id: str,
    *,
    quality: str = "perfect",
    n_worlds: int = 20,
    seeds: tuple[int, ...] = (17, 23),
):
    if not seeds:
        return []
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
        elif quality == "one_error":
            if i == n_worlds - 1:
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
                    "top_drivers": [
                        {
                            "driver_id": "wrong_driver",
                            "impact_minor": 100000,
                            "rank": 1,
                        }
                    ],
                }
            else:
                t_pred, v_pred = gold_t, gold_v
        else:
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
                "top_drivers": [
                    {"driver_id": "hc", "impact_minor": 0, "rank": 1}
                ],
                "other_impact_minor": 0,
            }
        # Example ids are identical across arms so bootstrap pairing is explicit.
        records.append(
            make_pred(
                example_id=f"ex_t_{i}",
                world_id=wid,
                task="transaction_review",
                expected=gold_t,
                parsed=t_pred,
                arm_id=arm_id,
                seed=seeds[0],
                split=split,
            )
        )
        records.append(
            make_pred(
                example_id=f"ex_v_{i}",
                world_id=wid,
                task="variance_analysis",
                expected=gold_v,
                parsed=v_pred,
                arm_id=arm_id,
                seed=seeds[0],
                split=split,
            )
        )
    return [
        (
            record
            if seed == seeds[0]
            else record.model_copy(update={"seed": seed})
        )
        for seed in seeds
        for record in records
    ]


def _arm(
    arm_id: str,
    *,
    quality: str,
    n_worlds: int = 20,
    seeds: tuple[int, ...] = (17, 23),
    failed_examples: int = 0,
    filtered_examples: int = 0,
) -> ArmEvaluationInput:
    predictions = _arm_records(
        arm_id,
        quality=quality,
        n_worlds=n_worlds,
        seeds=seeds,
    )
    return ArmEvaluationInput(
        arm_id=arm_id,
        predictions=predictions,
        expected_examples=(
            len(predictions) + failed_examples + filtered_examples
        ),
        failed_examples=failed_examples,
        filtered_examples=filtered_examples,
        failed_example_reasons=(
            {"runtime_error": failed_examples} if failed_examples else {}
        ),
        filtered_example_reasons=(
            {"protocol_filter": filtered_examples} if filtered_examples else {}
        ),
    )


def _systems_inputs(
    *,
    include_batch1: bool = True,
    include_batch8: bool = True,
    batch1_arm: str = "sequence_kd",
    batch8_arm: str = "sequence_kd",
    batch1_profile: dict | None = None,
    batch8_profile: dict | None = None,
) -> tuple[SystemsProfileInput, ...]:
    profiles: list[SystemsProfileInput] = []
    if include_batch1:
        profiles.append(
            SystemsProfileInput(
                arm_id=batch1_arm,
                profile=batch1_profile
                or complete_systems_profile(
                    batch_size=1,
                    requests_per_second=20.0,
                ),
            )
        )
    if include_batch8:
        profiles.append(
            SystemsProfileInput(
                arm_id=batch8_arm,
                profile=batch8_profile
                or complete_systems_profile(
                    batch_size=8,
                    requests_per_second=80.0,
                ),
            )
        )
    return tuple(profiles)


def _full_input(**overrides) -> ProofEvaluationInput:
    arms = [
        _arm("rules", quality="poor"),
        _arm("teacher", quality="perfect"),
        _arm("student_base", quality="poor"),
        _arm("cheap_off_the_shelf", quality="poor"),
        _arm("sequence_kd", quality="perfect"),
    ]
    kwargs = {
        "report_id": "prf_test_001",
        "protocol_id": "finance-proof.v1",
        "protocol_sha256": PROTOCOL_SHA,
        "arms": arms,
        "costs": complete_cost_ledger(),
        "created_at": CREATED_AT,
        "run_ids": ("run_seed17", "run_seed23"),
        "finalist_arm_id": "sequence_kd",
        "systems_profiles": _systems_inputs(),
        "teacher_cost_per_request": CostValue(
            0.02,
            EvidenceKind.PROJECTED,
            "teacher_public_price_per_request",
        ),
        "student_serving_hourly_cost_usd": CostValue(
            1.20,
            EvidenceKind.PROJECTED,
            "student_instance_hour",
        ),
        "evaluation_horizon_requests": 100_000,
        "trainer_numerical_ok": True,
        "trainer_artifact_reload_ok": True,
        "trainer_smoke_eval_ok": True,
        "trainer_memory_cost_ok": True,
        "frozen_hashes_present": True,
        "bootstrap_seed": 17,
        "bootstrap_resamples": 100,
        "rules_tco_comparison": BaselineTCOComparison(
            False,
            EvidenceKind.MEASURED,
            "rules TCO frozen before evaluation",
        ),
        "cheap_tco_comparison": BaselineTCOComparison(
            False,
            EvidenceKind.PROJECTED,
            "cheap API public price comparison",
        ),
    }
    kwargs.update(overrides)
    return ProofEvaluationInput(**kwargs)


def test_complete_evaluation_reaches_proved() -> None:
    result = evaluate_proof(_full_input(bootstrap_resamples=None))
    assert result.report.proof_status is ProofStatus.PROVED
    assert result.report.first_failed_gate is None
    assert result.economics["evaluated"] is True
    assert result.economics["sensitivity_complete"] is True
    assert len(result.economics["utilization_rows"]) == 8
    assert result.uncertainty["interval_inventory"] == {
        "count": 52,
        "all_defined": True,
        "all_proof_ready": True,
        "underpowered_interval_ids": [],
        "undefined_interval_ids": [],
    }
    interval_ids = {
        interval["interval_id"]
        for interval in result.uncertainty["intervals"]
    }
    assert "arm::sequence_kd::transaction_joint_exact" in interval_ids
    assert "arm::teacher::ood_primary_index" in interval_ids
    assert (
        "pair::sequence_kd::teacher::json_schema_validity_difference"
        in interval_ids
    )
    assert "pair::sequence_kd::teacher::ood_retention" in interval_ids
    assert (
        result.uncertainty["methodology"]["method"]
        == "percentile_linear_interpolation"
    )
    assert "percentile_intervals_are_not_bias_corrected_or_accelerated" in (
        result.uncertainty["methodology"]["limitations"]
    )
    assert set(result.systems) >= {
        "sequence_kd:batch_1",
        "sequence_kd:batch_8",
    }


def test_omitted_tco_comparison_never_defaults_not_cheaper() -> None:
    for field in ("rules_tco_comparison", "cheap_tco_comparison"):
        result = evaluate_proof(_full_input(**{field: None}))
        assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
        assert result.report.first_failed_gate == "baseline"
        assert "quality" in result.report.unevaluated_gates


def test_measured_cheaper_rules_baseline_wins_before_later_gates() -> None:
    arms = [
        _arm("rules", quality="perfect"),
        _arm("teacher", quality="perfect"),
        _arm("student_base", quality="poor"),
        _arm("cheap_off_the_shelf", quality="poor"),
        _arm("sequence_kd", quality="perfect"),
    ]
    result = evaluate_proof(
        _full_input(
            arms=arms,
            rules_tco_comparison=BaselineTCOComparison(
                True,
                EvidenceKind.MEASURED,
                "measured rules TCO below distillation",
            ),
        )
    )
    assert result.report.proof_status is ProofStatus.DO_NOT_DISTILL
    assert result.report.first_failed_gate == "baseline"
    assert "quality" in result.report.unevaluated_gates


def test_wide_retention_interval_routes_to_evidence_not_failed_quality() -> None:
    arms = [
        _arm("rules", quality="poor", n_worlds=20),
        _arm("teacher", quality="perfect", n_worlds=20),
        _arm("student_base", quality="poor", n_worlds=20),
        _arm("cheap_off_the_shelf", quality="poor", n_worlds=20),
        _arm("sequence_kd", quality="one_error", n_worlds=20),
    ]
    result = evaluate_proof(_full_input(arms=arms))
    retention = next(
        interval
        for interval in result.uncertainty["intervals"]
        if interval["metric"] == "quality_retention"
    )
    assert retention["estimate"] >= 0.95
    assert retention["lower"] < 0.90 <= retention["upper"]
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert result.report.first_failed_gate == "evidence"
    assert "narrower_quality_retention_interval" in (
        result.gate_evaluation.evidence_needed
    )
    assert "narrower_ood_retention_interval" in (
        result.gate_evaluation.evidence_needed
    )


def test_missing_batch8_finalist_profile_is_incomplete() -> None:
    result = evaluate_proof(
        _full_input(systems_profiles=_systems_inputs(include_batch8=False))
    )
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert result.report.first_failed_gate == "economics"
    assert result.economics["utilization_rows"] == []


def test_never_borrows_batch8_throughput_from_other_arm() -> None:
    result = evaluate_proof(
        _full_input(
            systems_profiles=_systems_inputs(batch8_arm="teacher"),
        )
    )
    assert "teacher:batch_8" in result.systems
    assert "sequence_kd:batch_8" not in result.systems
    assert result.economics["evaluated"] is False
    assert result.economics["utilization_rows"] == []
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE


def test_excluded_systems_only_arm_profile_is_still_processed() -> None:
    profile_only = ArmEvaluationInput(
        arm_id="profile_only",
        predictions=[],
        systems_profile=complete_systems_profile(
            batch_size=8,
            requests_per_second=99.0,
        ),
        excluded=True,
        exclusion_reason="systems-only",
    )
    inp = _full_input(arms=[*_full_input().arms, profile_only])
    result = evaluate_proof(inp)
    assert "profile_only:batch_8" in result.systems


def test_aggregate_counts_do_not_satisfy_per_task_profile_requirement() -> None:
    profile = complete_systems_profile(batch_size=8, requests_per_second=80.0)
    profile.pop("warmup_requests_by_task")
    profile.pop("timed_examples_by_task")
    profile["warmup_requests"] = 1_000
    profile["timed_examples"] = 10_000
    result = evaluate_proof(
        _full_input(
            systems_profiles=_systems_inputs(batch8_profile=profile),
        )
    )
    assert result.economics["evaluated"] is False
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE


def test_mismatched_hardware_or_runtime_is_incomplete() -> None:
    batch8 = complete_systems_profile(
        batch_size=8,
        requests_per_second=80.0,
        runtime="different-runtime",
    )
    result = evaluate_proof(
        _full_input(
            systems_profiles=_systems_inputs(batch8_profile=batch8),
        )
    )
    assert result.economics["evaluated"] is False
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE


def test_missing_finalist_throughput_does_not_emit_economics() -> None:
    batch1 = complete_systems_profile(batch_size=1, requests_per_second=20.0)
    batch1["requests_per_second"] = None
    result = evaluate_proof(
        _full_input(
            systems_profiles=_systems_inputs(batch1_profile=batch1),
        )
    )
    assert result.economics["evaluated"] is False
    assert result.economics["utilization_rows"] == []


def test_missing_cost_component_or_unreconciled_gross_never_proves() -> None:
    missing = complete_cost_ledger()
    del missing["teacher_generation_tokens"]
    missing_result = evaluate_proof(_full_input(costs=missing))
    assert missing_result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert missing_result.report.first_failed_gate == "economics"

    unreconciled = complete_cost_ledger()
    unreconciled["gross_experiment_cost_usd"]["amount_usd"] += 1.0
    unreconciled_result = evaluate_proof(_full_input(costs=unreconciled))
    assert unreconciled_result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert unreconciled_result.economics["evaluated"] is False


def test_missing_teacher_arm_uses_none_not_zero_economics_placeholder() -> None:
    arms = [arm for arm in _full_input().arms if arm.arm_id != "teacher"]
    result = evaluate_proof(_full_input(arms=arms))
    assert result.economics["quality_retention"] is None
    assert result.economics["evaluated"] is False
    assert result.economics["utilization_rows"] == []
    assert "teacher_primary_index_missing" in result.economics["unevaluated_reasons"]
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE


def test_missing_prediction_seed_is_rejected_not_attested() -> None:
    arms = list(_full_input().arms)
    arms[-1] = _arm("sequence_kd", quality="perfect", seeds=(17,))
    with pytest.raises(ValueError, match="required seeds"):
        evaluate_proof(_full_input(arms=arms))


def test_missing_or_duplicate_seed_example_record_is_rejected() -> None:
    missing_arms = list(_full_input().arms)
    missing_finalist = _arm("sequence_kd", quality="perfect")
    missing_finalist.predictions = [
        record
        for record in missing_finalist.predictions
        if not (record.seed == 23 and record.example_id == "ex_t_0")
    ]
    missing_arms[-1] = missing_finalist
    with pytest.raises(ValueError, match="prediction identities differ"):
        evaluate_proof(_full_input(arms=missing_arms))

    duplicate_arms = list(_full_input().arms)
    duplicate_finalist = _arm("sequence_kd", quality="perfect")
    duplicate_finalist.predictions.append(
        duplicate_finalist.predictions[0]
    )
    duplicate_arms[-1] = duplicate_finalist
    with pytest.raises(ValueError, match="duplicate seed/example"):
        evaluate_proof(_full_input(arms=duplicate_arms))


def test_mismatched_paired_example_identity_is_rejected() -> None:
    arms = list(_full_input().arms)
    finalist = _arm("sequence_kd", quality="perfect")
    finalist.predictions = [
        (
            record.model_copy(update={"example_id": "changed_example"})
            if record.example_id == "ex_t_0"
            else record
        )
        for record in finalist.predictions
    ]
    arms[-1] = finalist
    with pytest.raises(ValueError, match="paired prediction identities differ"):
        evaluate_proof(_full_input(arms=arms))


def test_missing_arm_accounting_never_proves() -> None:
    arms = list(_full_input().arms)
    arms[-1] = ArmEvaluationInput(
        arm_id="sequence_kd",
        predictions=_arm_records("sequence_kd"),
        # Counts intentionally omitted.
    )
    missing_accounting = evaluate_proof(_full_input(arms=arms))
    assert missing_accounting.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert missing_accounting.report.first_failed_gate == "evidence"


def test_duplicate_benchmark_arm_ids_fail_loud() -> None:
    arms = [*_full_input().arms, _arm("sequence_kd", quality="perfect")]
    with pytest.raises(ValueError, match="arm_ids must be unique"):
        evaluate_proof(_full_input(arms=arms))


def test_failed_and_filtered_counts_must_reconcile() -> None:
    arms = list(_full_input().arms)
    finalist = _arm("sequence_kd", quality="perfect", failed_examples=2)
    finalist.expected_examples += 1
    arms[-1] = finalist
    result = evaluate_proof(_full_input(arms=arms))
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert "failed_and_filtered_example_accounting" in (
        result.gate_evaluation.evidence_needed
    )


def test_nonzero_failed_examples_require_reason_reconciliation() -> None:
    arms = list(_full_input().arms)
    finalist = _arm("sequence_kd", quality="perfect", failed_examples=1)
    finalist.failed_example_reasons = None
    arms[-1] = finalist
    result = evaluate_proof(_full_input(arms=arms))
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert result.report.first_failed_gate == "evidence"


def test_reconciled_but_unscored_failed_example_cannot_prove() -> None:
    arms = list(_full_input().arms)
    arms[-1] = _arm("sequence_kd", quality="perfect", failed_examples=1)
    result = evaluate_proof(_full_input(arms=arms))
    accounting = next(
        arm for arm in result.report.arm_results if arm.arm_id == "sequence_kd"
    ).metrics["example_accounting"]
    assert accounting["complete"] is True
    assert accounting["proof_complete"] is False
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert result.report.first_failed_gate == "evidence"


def test_non_captured_raw_text_provenance_cannot_prove() -> None:
    arms = list(_full_input().arms)
    finalist = _arm("sequence_kd", quality="perfect")
    finalist.predictions = [
        record.model_copy(
            update={"raw_text_provenance": "fixture_serialization"}
        )
        for record in finalist.predictions
    ]
    arms[-1] = finalist
    result = evaluate_proof(_full_input(arms=arms))
    assert result.report.proof_status is ProofStatus.INSUFFICIENT_EVIDENCE
    assert "captured_raw_text_provenance" in (
        result.gate_evaluation.evidence_needed
    )


def test_explicit_failed_and_filtered_zero_counts_can_pass() -> None:
    result = evaluate_proof(_full_input())
    finalist = next(
        arm for arm in result.report.arm_results if arm.arm_id == "sequence_kd"
    )
    accounting = finalist.metrics["example_accounting"]
    assert accounting == {
        "expected_examples": 80,
        "prediction_records": 80,
        "failed_examples": 0,
        "filtered_examples": 0,
        "failed_example_reasons": {},
        "filtered_example_reasons": {},
        "complete": True,
        "proof_complete": True,
    }


def test_created_at_is_required_and_timezone_aware() -> None:
    kwargs = _full_input().__dict__.copy()
    kwargs["created_at"] = datetime(2026, 7, 18, 12, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_proof(ProofEvaluationInput(**kwargs))


def test_report_hash_is_reproducible_for_identical_inputs() -> None:
    first = render_json_report(evaluate_proof(_full_input()))
    second = render_json_report(evaluate_proof(_full_input()))
    assert first["created_at"] == CREATED_AT.isoformat()
    assert first["document_sha256"] == second["document_sha256"]
    assert first["resource_hash"] == second["resource_hash"]


def test_json_and_html_report_are_self_contained_and_honest() -> None:
    result = evaluate_proof(_full_input(bootstrap_resamples=None))
    doc = render_json_report(result)
    html = render_html_report(result)
    assert doc["proof_status"] == "proved"
    assert doc["economics"]["gross_experiment_cost_usd"]["kind"] == "measured"
    assert doc["economics"]["teacher_cost_per_request_usd"]["kind"] == "projected"
    assert len(doc["economics"]["utilization_rows"]) == 8
    assert "<!DOCTYPE html>" in html
    assert "projected" in html.lower()
    assert "batch 1" in html
    assert "batch 8" in html
    assert "percentile_linear_interpolation" in html
    assert "arm::sequence_kd::primary_index" in html
    assert "<script src=" not in html
    assert doc["document_sha256"] in html
