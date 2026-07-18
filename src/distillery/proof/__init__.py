"""Proof, uncertainty, systems, and economics evaluation (Workstream 5).

Consumes immutable prediction files and run manifests. Never trains models.
"""

from __future__ import annotations

from distillery.proof.bootstrap import (
    PRIMARY_METRICS,
    BootstrapCI,
    arm_metric_ci,
    paired_cluster_bootstrap,
    paired_difference_ci,
    quality_retention_ci,
    ratio_metric_ci,
    validate_paired_scores,
)
from distillery.proof.economics import (
    BREAK_EVEN_NEVER,
    BreakEvenResult,
    CostKind,
    CostValue,
    EconomicsSummary,
    GrossCostLedger,
    break_even_requests,
    build_cost_ledger,
    compute_economics,
    quality_retention,
    recovered_teacher_gap,
    utilization_cost_rows,
)
from distillery.proof.evaluate import (
    ArmEvaluationInput,
    ProofEvaluationInput,
    SystemsProfileInput,
    evaluate_proof,
)
from distillery.proof.evidence import EvidenceKind, LabeledValue
from distillery.proof.gates import (
    GATE_ORDER,
    ArmAccountingEvidence,
    ArmQualityEvidence,
    BaselineTCOComparison,
    GateEvaluation,
    evaluate_gates,
)
from distillery.proof.metrics import (
    PRIMARY_INDEX_WEIGHTS,
    PRIMARY_INDEX_WEIGHTS_V2,
    ArmMetrics,
    CalibrationMetrics,
    ExampleScore,
    PredictionRecord,
    RawTextProvenance,
    SliceReport,
    compute_arm_metrics,
    compute_primary_index,
    compute_primary_index_v2,
    score_prediction,
)
from distillery.proof.protocol_v2 import (
    PROOF_PROTOCOL_ID_V2,
    finance_proof_v2_document,
    finance_proof_v2_sha256,
)
from distillery.proof.report import render_html_report, render_json_report
from distillery.proof.systems import SystemsSummary, summarize_systems

__all__ = [
    "BREAK_EVEN_NEVER",
    "GATE_ORDER",
    "PRIMARY_INDEX_WEIGHTS",
    "PRIMARY_INDEX_WEIGHTS_V2",
    "PROOF_PROTOCOL_ID_V2",
    "PRIMARY_METRICS",
    "ArmEvaluationInput",
    "ArmAccountingEvidence",
    "ArmMetrics",
    "ArmQualityEvidence",
    "BaselineTCOComparison",
    "BootstrapCI",
    "BreakEvenResult",
    "CalibrationMetrics",
    "CostKind",
    "CostValue",
    "EconomicsSummary",
    "EvidenceKind",
    "ExampleScore",
    "GateEvaluation",
    "GrossCostLedger",
    "LabeledValue",
    "PredictionRecord",
    "RawTextProvenance",
    "arm_metric_ci",
    "ProofEvaluationInput",
    "SliceReport",
    "SystemsSummary",
    "SystemsProfileInput",
    "break_even_requests",
    "build_cost_ledger",
    "compute_arm_metrics",
    "compute_economics",
    "compute_primary_index",
    "compute_primary_index_v2",
    "finance_proof_v2_document",
    "finance_proof_v2_sha256",
    "evaluate_gates",
    "evaluate_proof",
    "paired_cluster_bootstrap",
    "paired_difference_ci",
    "quality_retention_ci",
    "quality_retention",
    "ratio_metric_ci",
    "recovered_teacher_gap",
    "render_html_report",
    "render_json_report",
    "score_prediction",
    "summarize_systems",
    "utilization_cost_rows",
    "validate_paired_scores",
]
