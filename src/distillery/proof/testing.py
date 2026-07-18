"""Builders for proof unit tests and offline fixture assembly."""

from __future__ import annotations

import json
from typing import Any

from distillery.proof.metrics import PredictionRecord


def txn_gold(
    *,
    gl: str = "6100",
    amount: int = 4500,
    action: str = "approve",
    rules: list[str] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    rules = rules or ["POL-MEAL-001"]
    return {
        "schema_version": "transaction_review.v1",
        "task": "transaction_review",
        "gl_account": gl,
        "journal_entry": [
            {"account": gl, "side": "debit", "amount_minor": amount},
            {"account": "2100", "side": "credit", "amount_minor": amount},
        ],
        "policy_action": action,
        "rule_ids": rules,
        "evidence": [{"source_id": "txn", "field": "amount_minor", "value": str(amount)}],
        "confidence": confidence,
    }


def var_gold(
    *,
    profit: int = -420000,
    drivers: list[dict[str, Any]] | None = None,
    other: int = -30000,
    confidence: float = 0.91,
) -> dict[str, Any]:
    if drivers is None:
        drivers = [
            {"driver_id": "cloud_usage", "impact_minor": -300000, "rank": 1},
            {"driver_id": "support_volume", "impact_minor": -90000, "rank": 2},
        ]
    direction = "favorable" if profit >= 0 else "unfavorable"
    return {
        "schema_version": "variance_analysis.v1",
        "task": "variance_analysis",
        "profit_impact_minor": profit,
        "direction": direction,
        "top_drivers": drivers,
        "other_impact_minor": other,
        "rule_ids": ["VAR-MATERIAL-005"],
        "evidence_ids": ["actual_cloud", "budget_cloud"],
        "confidence": confidence,
    }


def cash_gold() -> dict[str, Any]:
    return {
        "schema_version": "cash_reconciliation.v1",
        "task": "cash_reconciliation",
        "status": "exceptions",
        "matched_groups": [{"book_ids": ["b1"], "bank_ids": ["k9"]}],
        "exceptions": [{"type": "bank_fee", "event_ids": ["k10"], "amount_minor": 3500}],
        "adjusted_book_balance_minor": 8123400,
        "adjusted_bank_balance_minor": 8123400,
        "difference_minor": 0,
        "confidence": 0.89,
    }


def make_pred(
    *,
    example_id: str,
    world_id: str,
    task: str,
    expected: dict[str, Any],
    parsed: dict[str, Any] | None = None,
    raw_text: str | None = None,
    refused: bool = False,
    split: str = "iid_test",
    difficulty: str = "medium",
    template_family: str = "tmpl_a",
    arm_id: str = "arm",
    seed: int = 17,
    raw_text_provenance: str = "captured_model_output",
    slices: dict[str, str] | None = None,
    latency_ms: float | None = 10.0,
    output_tokens: int | None = 40,
) -> PredictionRecord:
    resolved_parsed = parsed
    if resolved_parsed is None and raw_text is None and not refused:
        resolved_parsed = expected
    resolved_raw_text = raw_text
    if resolved_raw_text is None:
        resolved_raw_text = (
            "I refuse"
            if refused
            else json.dumps(resolved_parsed, sort_keys=True)
        )
    return PredictionRecord(
        example_id=example_id,
        world_id=world_id,
        group_id="grp_test",
        task=task,
        difficulty=difficulty,
        split=split,
        template_family=template_family,
        arm_id=arm_id,
        seed=seed,
        raw_text=resolved_raw_text,
        raw_text_provenance=raw_text_provenance,
        parsed=resolved_parsed,
        refused=refused,
        latency_ms=latency_ms,
        output_tokens=output_tokens,
        expected_output=expected,
        slices=slices or {},
    )


def measured_cost(amount_usd: float, label: str) -> dict[str, Any]:
    return {
        "amount_usd": amount_usd,
        "kind": "measured",
        "label": label,
    }


def complete_cost_ledger(
    *,
    gpu_compute_cost_usd: float = 40.0,
    teacher_generation_cost_usd: float = 8.0,
    cheap_api_benchmark_cost_usd: float = 3.0,
    storage_cost_usd: float = 2.0,
    other_costs_usd: dict[str, float] | None = None,
    teacher_generation_tokens: int = 10_000,
    billed_training_seconds: int = 7_200,
    cheap_api_zero_reason: str | None = None,
) -> dict[str, Any]:
    other = other_costs_usd if other_costs_usd is not None else {"evaluation": 2.0}
    gross = (
        gpu_compute_cost_usd
        + teacher_generation_cost_usd
        + cheap_api_benchmark_cost_usd
        + storage_cost_usd
        + sum(other.values())
    )
    payload: dict[str, Any] = {
        "gross_experiment_cost_usd": measured_cost(gross, "gross_experiment"),
        "billed_training_seconds": {
            "value": billed_training_seconds,
            "kind": "measured",
        },
        "gpu_compute_cost_usd": measured_cost(
            gpu_compute_cost_usd,
            "gpu_compute",
        ),
        "teacher_generation_tokens": {
            "value": teacher_generation_tokens,
            "kind": "measured",
        },
        "teacher_generation_cost_usd": measured_cost(
            teacher_generation_cost_usd,
            "teacher_generation",
        ),
        "cheap_api_benchmark_cost_usd": measured_cost(
            cheap_api_benchmark_cost_usd,
            "cheap_api_benchmark",
        ),
        "storage_cost_usd": measured_cost(storage_cost_usd, "storage"),
        "other_costs_usd": {
            name: measured_cost(amount, name)
            for name, amount in other.items()
        },
    }
    if cheap_api_zero_reason is not None:
        payload["cheap_api_zero_reason"] = cheap_api_zero_reason
    return payload


def complete_systems_profile(
    *,
    batch_size: int,
    requests_per_second: float,
    hardware: str = "ml.g5.xlarge",
    runtime: str = "transformers-4.46/cuda-12.4",
) -> dict[str, Any]:
    return {
        "hardware": hardware,
        "runtime": runtime,
        "batch_size": batch_size,
        "warmup_requests": 40,
        "timed_examples": 400,
        "warmup_requests_by_task": {
            "transaction_review": 20,
            "variance_analysis": 20,
        },
        "timed_examples_by_task": {
            "transaction_review": 200,
            "variance_analysis": 200,
        },
        "latency_p50_ms": 40.0,
        "latency_p95_ms": 90.0,
        "requests_per_second": requests_per_second,
        "output_tokens_per_second": 500.0,
        "failure_rate": 0.0,
        "peak_vram_allocated_gb": 16.0,
        "peak_vram_reserved_gb": 18.0,
        "peak_cpu_ram_gb": 10.0,
        "billed_training_seconds": 7_200,
    }
