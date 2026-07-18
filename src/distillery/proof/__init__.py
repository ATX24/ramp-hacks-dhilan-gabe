"""Proof, uncertainty, systems, and economics evaluation (Workstream 5).

Consumes immutable prediction files and run manifests. Never trains models.
"""

from __future__ import annotations

from distillery.proof.bootstrap import (
    BootstrapCI,
    paired_cluster_bootstrap,
    paired_difference_ci,
)
from distillery.proof.economics import (
    BREAK_EVEN_NEVER,
    BreakEvenResult,
    CostKind,
    CostValue,
    EconomicsSummary,
    break_even_requests,
    compute_economics,
    quality_retention,
    recovered_teacher_gap,
    utilization_cost_rows,
)
from distillery.proof.evaluate import (
    ArmEvaluationInput,
    ProofEvaluationInput,
    evaluate_proof,
)
from distillery.proof.gates import (
    GATE_ORDER,
    GateEvaluation,
    evaluate_gates,
)
from distillery.proof.metrics import (
    PRIMARY_INDEX_WEIGHTS,
    ArmMetrics,
    CalibrationMetrics,
    ExampleScore,
    PredictionRecord,
    SliceReport,
    compute_arm_metrics,
    compute_primary_index,
    score_prediction,
)
from distillery.proof.report import render_html_report, render_json_report
from distillery.proof.systems import SystemsSummary, summarize_systems

__all__ = [
    "BREAK_EVEN_NEVER",
    "GATE_ORDER",
    "PRIMARY_INDEX_WEIGHTS",
    "ArmEvaluationInput",
    "ArmMetrics",
    "BootstrapCI",
    "BreakEvenResult",
    "CalibrationMetrics",
    "CostKind",
    "CostValue",
    "EconomicsSummary",
    "ExampleScore",
    "GateEvaluation",
    "PredictionRecord",
    "ProofEvaluationInput",
    "SliceReport",
    "SystemsSummary",
    "break_even_requests",
    "compute_arm_metrics",
    "compute_economics",
    "compute_primary_index",
    "evaluate_gates",
    "evaluate_proof",
    "paired_cluster_bootstrap",
    "paired_difference_ci",
    "quality_retention",
    "recovered_teacher_gap",
    "render_html_report",
    "render_json_report",
    "score_prediction",
    "summarize_systems",
    "utilization_cost_rows",
]
