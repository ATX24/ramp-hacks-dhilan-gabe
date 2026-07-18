"""Evaluate immutable predictions/manifests into a ProofReport payload."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from distillery.contracts.budgets import ProofGates
from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.ids import ProofReportId, RunId
from distillery.contracts.proof import ArmResult, ProofReport
from distillery.proof.bootstrap import (
    PERCENTILE_LIMITATIONS,
    PERCENTILE_METHOD,
    PRIMARY_METRICS,
    BootstrapCI,
    arm_metric_ci,
    paired_difference_ci,
    quality_retention_ci,
    ratio_metric_ci,
    validate_paired_scores,
)
from distillery.proof.economics import (
    CostValue,
    compute_economics,
    quality_retention,
)
from distillery.proof.evidence import EvidenceKind
from distillery.proof.gates import (
    ArmAccountingEvidence,
    ArmQualityEvidence,
    BaselineTCOComparison,
    GateEvaluation,
    GateInputs,
    evaluate_gates,
)
from distillery.proof.metrics import (
    ArmMetrics,
    PredictionRecord,
    compute_arm_metrics,
    iid_ood_primary,
)
from distillery.proof.systems import SystemsSummary, summarize_systems

REQUIRED_BASELINE_ARMS = ("rules", "teacher", "student_base", "cheap_off_the_shelf")


@dataclass
class ArmEvaluationInput:
    arm_id: str
    predictions: list[PredictionRecord]
    prediction_sha256: str | None = None
    systems_profile: dict[str, Any] | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    run_id: str | None = None
    expected_examples: int | None = None
    failed_examples: int | None = None
    filtered_examples: int | None = None
    failed_example_reasons: dict[str, int] | None = None
    filtered_example_reasons: dict[str, int] | None = None

    @property
    def example_accounting_complete(self) -> bool:
        counts = (
            self.expected_examples,
            self.failed_examples,
            self.filtered_examples,
        )
        if any(count is None or count < 0 for count in counts):
            return False
        if (
            self.failed_example_reasons is None
            or self.filtered_example_reasons is None
            or any(count < 0 for count in self.failed_example_reasons.values())
            or any(count < 0 for count in self.filtered_example_reasons.values())
        ):
            return False
        assert self.expected_examples is not None
        assert self.failed_examples is not None
        assert self.filtered_examples is not None
        return (
            len(self.predictions) + self.failed_examples + self.filtered_examples
            == self.expected_examples
            and sum(self.failed_example_reasons.values()) == self.failed_examples
            and sum(self.filtered_example_reasons.values())
            == self.filtered_examples
        )

    @property
    def example_accounting_proof_complete(self) -> bool:
        return (
            self.example_accounting_complete
            and self.failed_examples == 0
            and self.filtered_examples == 0
        )


@dataclass(frozen=True)
class SystemsProfileInput:
    """Systems-only profile explicitly attached to an evaluated arm."""

    arm_id: str
    profile: dict[str, Any]


@dataclass
class ProofEvaluationInput:
    report_id: ProofReportId
    protocol_id: str
    protocol_sha256: str
    arms: list[ArmEvaluationInput]
    costs: dict[str, Any]
    created_at: datetime
    run_ids: tuple[RunId, ...] = ()
    finalist_arm_id: str = "sequence_kd"
    base_arm_id: str = "student_base"
    teacher_arm_id: str = "teacher"
    systems_profiles: tuple[SystemsProfileInput, ...] = ()
    teacher_cost_per_request: CostValue | dict[str, Any] | None = None
    student_serving_hourly_cost_usd: CostValue | dict[str, Any] | None = None
    evaluation_horizon_requests: int | None = None
    # Trainer gate evidence (must be supplied; missing => insufficient_evidence)
    trainer_numerical_ok: bool | None = None
    trainer_artifact_reload_ok: bool | None = None
    trainer_smoke_eval_ok: bool | None = None
    trainer_memory_cost_ok: bool | None = None
    frozen_hashes_present: bool | None = None
    bootstrap_seed: int = 17
    bootstrap_resamples: int | None = None
    limitations: tuple[str, ...] = ()
    rules_tco_comparison: BaselineTCOComparison | None = None
    cheap_tco_comparison: BaselineTCOComparison | None = None


@dataclass
class ProofEvaluationResult:
    report: ProofReport
    arm_metrics: dict[str, ArmMetrics]
    systems: dict[str, SystemsSummary]
    gate_evaluation: GateEvaluation
    uncertainty: dict[str, Any]
    economics: dict[str, Any]


def load_predictions_jsonl(
    path: Path | str, *, arm_id: str | None = None
) -> list[PredictionRecord]:
    """Load immutable rows without synthesizing raw text or provenance."""
    records: list[PredictionRecord] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if arm_id is not None and "arm_id" not in payload:
            payload = {**payload, "arm_id": arm_id}
        records.append(PredictionRecord.model_validate(payload))
    return records


def file_sha256(path: Path | str) -> str:
    return sha256_hex(Path(path).read_bytes())


def _quality_comparison(
    metrics: ArmMetrics,
    teacher: ArmMetrics,
) -> tuple[float | None, float | None, float | None]:
    retention = quality_retention(metrics.primary_index, teacher.primary_index)
    regressions: list[float] = []
    if (
        metrics.transaction_joint_exact is not None
        and teacher.transaction_joint_exact is not None
    ):
        regressions.append(
            teacher.transaction_joint_exact - metrics.transaction_joint_exact
        )
    if (
        metrics.variance_joint_exact is not None
        and teacher.variance_joint_exact is not None
    ):
        regressions.append(
            teacher.variance_joint_exact - metrics.variance_joint_exact
        )
    max_regression = max(regressions) if regressions else None

    arm_ood = iid_ood_primary(metrics.example_scores).get("ood_primary_index")
    teacher_ood = iid_ood_primary(teacher.example_scores).get("ood_primary_index")
    ood_retention = (
        arm_ood / teacher_ood
        if arm_ood is not None and teacher_ood not in (None, 0.0)
        else None
    )
    return retention, max_regression, ood_retention


def _arm_quality_evidence(
    metrics: ArmMetrics,
    teacher: ArmMetrics,
) -> ArmQualityEvidence:
    retention, max_regression, ood_retention = _quality_comparison(
        metrics,
        teacher,
    )
    return ArmQualityEvidence(
        quality_retention=retention,
        max_primary_task_regression=max_regression,
        json_schema_validity=metrics.json_schema_validity,
        ood_retention=ood_retention,
        critical_invariant_violations=metrics.critical_invariant_violations,
    )


def _systems_key(arm_id: str, batch_size: int) -> str:
    return f"{arm_id}:batch_{batch_size}"


def _validate_arm_seed_records(arm: ArmEvaluationInput) -> tuple[int, ...]:
    if not arm.predictions:
        raise ValueError(f"{arm.arm_id}: no prediction records")
    by_seed: dict[int, dict[str, tuple[str, ...]]] = {}
    for record in arm.predictions:
        if record.arm_id != arm.arm_id:
            raise ValueError(
                f"{arm.arm_id}: prediction arm_id mismatch: {record.arm_id}"
            )
        seed_records = by_seed.setdefault(record.seed, {})
        if record.example_id in seed_records:
            raise ValueError(
                f"{arm.arm_id}: duplicate seed/example record "
                f"({record.seed}, {record.example_id})"
            )
        seed_records[record.example_id] = (
            record.world_id,
            record.task,
            record.split,
            record.group_id,
            record.difficulty,
            record.template_family,
            content_sha256(record.expected_output),
        )
    seeds = tuple(sorted(by_seed))
    reference_seed = seeds[0]
    reference = by_seed[reference_seed]
    for seed in seeds[1:]:
        if by_seed[seed] != reference:
            raise ValueError(
                f"{arm.arm_id}: seed {seed} prediction identities differ "
                f"from seed {reference_seed}"
            )
    return seeds


def _required_comparison_pairs(
    inp: ProofEvaluationInput,
) -> tuple[tuple[str, str], ...]:
    return (
        (inp.teacher_arm_id, inp.base_arm_id),
        (inp.finalist_arm_id, inp.teacher_arm_id),
        (inp.finalist_arm_id, inp.base_arm_id),
        ("rules", inp.teacher_arm_id),
        ("cheap_off_the_shelf", inp.teacher_arm_id),
    )


def _validate_paired_prediction_records(
    arm_a: ArmEvaluationInput,
    arm_b: ArmEvaluationInput,
) -> None:
    def paired_map(
        arm: ArmEvaluationInput,
    ) -> dict[tuple[int, str, str], tuple[str, str, str]]:
        return {
            (record.seed, record.example_id, record.world_id): (
                record.task,
                record.split,
                content_sha256(record.expected_output),
            )
            for record in arm.predictions
        }

    records_a = paired_map(arm_a)
    records_b = paired_map(arm_b)
    if records_a != records_b:
        raise ValueError(
            f"{arm_a.arm_id}/{arm_b.arm_id}: paired prediction identities "
            "differ or immutable gold records differ"
        )


def evaluate_proof(inp: ProofEvaluationInput) -> ProofEvaluationResult:
    """Blind-ish evaluation: metrics computed before status assignment."""
    if inp.created_at.tzinfo is None or inp.created_at.utcoffset() is None:
        raise ValueError("created_at must be explicit and timezone-aware")
    included_arm_ids = [arm.arm_id for arm in inp.arms if not arm.excluded]
    if len(included_arm_ids) != len(set(included_arm_ids)):
        raise ValueError("non-excluded benchmark arm_ids must be unique")
    seed_sets_by_arm = {
        arm.arm_id: _validate_arm_seed_records(arm)
        for arm in inp.arms
        if not arm.excluded
    }
    arm_inputs_by_id = {
        arm.arm_id: arm for arm in inp.arms if not arm.excluded
    }
    required_seeds = {
        ProofGates().required_seed_screen,
        ProofGates().required_seed_replication,
    }
    for arm_id in (inp.finalist_arm_id, inp.base_arm_id):
        if arm_id not in seed_sets_by_arm:
            continue
        seeds = set(seed_sets_by_arm.get(arm_id, ()))
        if not required_seeds.issubset(seeds):
            raise ValueError(
                f"{arm_id}: required seeds {sorted(required_seeds)} missing; "
                f"found {sorted(seeds)}"
            )
    if (
        inp.finalist_arm_id in seed_sets_by_arm
        and inp.base_arm_id in seed_sets_by_arm
        and seed_sets_by_arm[inp.finalist_arm_id]
        != seed_sets_by_arm[inp.base_arm_id]
    ):
        raise ValueError("finalist and matched comparator seed sets differ")

    th = ProofGates()
    arm_metrics: dict[str, ArmMetrics] = {}
    systems: dict[str, SystemsSummary] = {}
    arm_results: list[ArmResult] = []
    exclusions: list[str] = []

    def add_systems_profile(arm_id: str, profile: dict[str, Any]) -> None:
        summary = summarize_systems(profile)
        key = _systems_key(arm_id, summary.batch_size)
        if key in systems:
            raise ValueError(f"duplicate systems profile for {key}")
        systems[key] = summary

    for profile_input in inp.systems_profiles:
        add_systems_profile(profile_input.arm_id, profile_input.profile)

    for arm in inp.arms:
        # Systems evidence is independent of benchmark-arm inclusion. Process it
        # before excluded arms are skipped.
        if arm.systems_profile is not None:
            add_systems_profile(arm.arm_id, arm.systems_profile)
        if arm.excluded:
            exclusions.append(f"{arm.arm_id}: {arm.exclusion_reason or 'excluded'}")
            arm_results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    primary_index=None,
                    metrics={},
                    prediction_sha256=arm.prediction_sha256,
                    excluded=True,
                    exclusion_reason=arm.exclusion_reason,
                )
            )
            continue
        metrics = compute_arm_metrics(arm.arm_id, arm.predictions)
        arm_metrics[arm.arm_id] = metrics
        pred_hash = arm.prediction_sha256
        if pred_hash is None and arm.predictions:
            pred_hash = content_sha256(
                [p.model_dump(mode="json") for p in arm.predictions]
            )
        split_bits = iid_ood_primary(metrics.example_scores)
        arm_results.append(
            ArmResult(
                arm_id=arm.arm_id,
                primary_index=metrics.primary_index,
                metrics={
                    "json_parse_rate": metrics.json_parse_rate,
                    "json_schema_validity": metrics.json_schema_validity,
                    "refusal_empty_rate": metrics.refusal_empty_rate,
                    "transaction_joint_exact": metrics.transaction_joint_exact,
                    "variance_joint_exact": metrics.variance_joint_exact,
                    "cash_joint_exact": metrics.cash_joint_exact,
                    "critical_invariant_violations": metrics.critical_invariant_violations,
                    "seeds": list(metrics.seeds),
                    "seed_metrics": {
                        str(seed): values
                        for seed, values in metrics.seed_metrics.items()
                    },
                    "raw_text_provenance": {
                        provenance: sum(
                            1
                            for record in arm.predictions
                            if record.raw_text_provenance == provenance
                        )
                        for provenance in (
                            "captured_model_output",
                            "fixture_serialization",
                        )
                    },
                    "calibration": {
                        "brier_score": metrics.calibration.brier_score,
                        "adaptive_ece": metrics.calibration.adaptive_ece,
                        "n": metrics.calibration.n,
                    },
                    "slices": [
                        {
                            "slice_key": s.slice_key,
                            "slice_value": s.slice_value,
                            "n": s.n,
                            "joint_exact": s.joint_exact,
                            "json_schema_validity": s.json_schema_validity,
                            "underpowered": s.underpowered,
                        }
                        for s in metrics.slices
                    ],
                    "iid_ood": split_bits,
                    "task_metrics": metrics.task_metrics,
                    "example_accounting": {
                        "expected_examples": arm.expected_examples,
                        "prediction_records": len(arm.predictions),
                        "failed_examples": arm.failed_examples,
                        "filtered_examples": arm.filtered_examples,
                        "failed_example_reasons": arm.failed_example_reasons,
                        "filtered_example_reasons": arm.filtered_example_reasons,
                        "complete": arm.example_accounting_complete,
                        "proof_complete": arm.example_accounting_proof_complete,
                    },
                },
                prediction_sha256=pred_hash,
                excluded=False,
                exclusion_reason=None,
            )
        )

    # Uncertainty: all arm metrics plus every required strict paired difference.
    uncertainty: dict[str, Any] = {
        "source_timestamp": inp.created_at.isoformat(),
        "methodology": {
            "confidence_level": 0.95,
            "method": PERCENTILE_METHOD,
            "cluster_unit": "world_id",
            "seed_handling": (
                "seed replicates are aggregated within example and remain "
                "inside their world_id cluster"
            ),
            "limitations": list(PERCENTILE_LIMITATIONS),
        },
        "intervals": [],
    }
    bootstrap_intervals: list[BootstrapCI] = []
    n_boot = inp.bootstrap_resamples
    finalist = arm_metrics.get(inp.finalist_arm_id)
    teacher = arm_metrics.get(inp.teacher_arm_id)
    base = arm_metrics.get(inp.base_arm_id)

    for arm_id, metrics in sorted(arm_metrics.items()):
        for metric in PRIMARY_METRICS:
            bootstrap_intervals.append(
                arm_metric_ci(
                    metrics.example_scores,
                    metric,
                    arm_id=arm_id,
                    n_resamples=n_boot,
                    seed=inp.bootstrap_seed,
                )
            )

    for arm_a, arm_b in _required_comparison_pairs(inp):
        metrics_a = arm_metrics.get(arm_a)
        metrics_b = arm_metrics.get(arm_b)
        if metrics_a is None or metrics_b is None:
            continue
        _validate_paired_prediction_records(
            arm_inputs_by_id[arm_a],
            arm_inputs_by_id[arm_b],
        )
        validate_paired_scores(
            metrics_a.example_scores,
            metrics_b.example_scores,
            arm_a=arm_a,
            arm_b=arm_b,
        )
        for score_metric in PRIMARY_METRICS:
            bootstrap_intervals.append(
                paired_difference_ci(
                    metrics_a.example_scores,
                    metrics_b.example_scores,
                    score_metric=score_metric,
                    n_resamples=n_boot,
                    seed=inp.bootstrap_seed,
                    metric=f"{score_metric}_difference",
                    arm_a=arm_a,
                    arm_b=arm_b,
                )
            )

    retention_point: float | None = None
    retention_lower: float | None = None
    retention_upper: float | None = None
    teacher_gap: float | None = None
    teacher_gap_ci: BootstrapCI | None = None
    ood_retention: float | None = None
    ood_retention_lower: float | None = None
    ood_retention_upper: float | None = None
    max_task_regression: float | None = None

    if finalist and teacher:
        retention_point = quality_retention(finalist.primary_index, teacher.primary_index)
        retention_ci = quality_retention_ci(
            finalist.example_scores,
            teacher.example_scores,
            n_resamples=n_boot,
            seed=inp.bootstrap_seed,
            arm_a=inp.finalist_arm_id,
            arm_b=inp.teacher_arm_id,
        )
        bootstrap_intervals.append(retention_ci)
        retention_lower = retention_ci.lower
        retention_upper = retention_ci.upper

        ood_retention_ci = ratio_metric_ci(
            finalist.example_scores,
            teacher.example_scores,
            score_metric="ood_primary_index",
            metric="ood_retention",
            arm_a=inp.finalist_arm_id,
            arm_b=inp.teacher_arm_id,
            n_resamples=n_boot,
            seed=inp.bootstrap_seed,
        )
        bootstrap_intervals.append(ood_retention_ci)
        ood_retention = ood_retention_ci.estimate
        ood_retention_lower = ood_retention_ci.lower
        ood_retention_upper = ood_retention_ci.upper

        teacher_gap = teacher.primary_index - (
            base.primary_index if base else finalist.primary_index
        )
        if base:
            teacher_gap = teacher.primary_index - base.primary_index
            teacher_gap_ci = next(
                (
                    interval
                    for interval in bootstrap_intervals
                    if interval.arm_a == inp.teacher_arm_id
                    and interval.arm_b == inp.base_arm_id
                    and interval.metric == "primary_index_difference"
                ),
                None,
            )

        # Max primary-task joint-exact regression vs teacher.
        regs: list[float] = []
        if (
            finalist.transaction_joint_exact is not None
            and teacher.transaction_joint_exact is not None
        ):
            regs.append(teacher.transaction_joint_exact - finalist.transaction_joint_exact)
        if (
            finalist.variance_joint_exact is not None
            and teacher.variance_joint_exact is not None
        ):
            regs.append(teacher.variance_joint_exact - finalist.variance_joint_exact)
        max_task_regression = max(regs) if regs else None

    uncertainty["intervals"] = [
        interval.to_dict() for interval in bootstrap_intervals
    ]
    uncertainty["interval_inventory"] = {
        "count": len(bootstrap_intervals),
        "all_defined": all(interval.defined for interval in bootstrap_intervals),
        "all_proof_ready": all(
            interval.proof_ready for interval in bootstrap_intervals
        ),
        "underpowered_interval_ids": [
            interval.interval_id
            for interval in bootstrap_intervals
            if interval.underpowered
        ],
        "undefined_interval_ids": [
            interval.interval_id
            for interval in bootstrap_intervals
            if not interval.defined
        ],
    }

    # Systems evidence belongs to the finalist only. Never borrow throughput
    # from another arm, even when hardware happens to match.
    finalist_systems = {
        batch_size: systems.get(_systems_key(inp.finalist_arm_id, batch_size))
        for batch_size in (1, 8)
    }
    systems_evidence_complete = all(
        summary is not None and summary.proof_ready
        for summary in finalist_systems.values()
    )
    if systems_evidence_complete:
        batch1_systems = finalist_systems[1]
        batch8_systems = finalist_systems[8]
        assert batch1_systems is not None
        assert batch8_systems is not None
        systems_evidence_complete = (
            batch1_systems.hardware == batch8_systems.hardware
            and batch1_systems.runtime == batch8_systems.runtime
        )

    observed_throughput: dict[int, float] = {}
    if systems_evidence_complete:
        for batch_size, summary in finalist_systems.items():
            assert summary is not None
            assert summary.requests_per_second.kind is EvidenceKind.MEASURED
            observed_throughput[batch_size] = float(
                summary.requests_per_second.value
            )

    eco = compute_economics(
        student_primary_index=finalist.primary_index if finalist else None,
        teacher_primary_index=teacher.primary_index if teacher else None,
        base_primary_index=base.primary_index if base else None,
        costs=inp.costs,
        teacher_cost_per_request=inp.teacher_cost_per_request,
        student_serving_hourly_cost_usd=inp.student_serving_hourly_cost_usd,
        observed_throughput_rps_by_batch=observed_throughput,
        evaluation_horizon_requests=inp.evaluation_horizon_requests,
        economics_utilization=th.economics_utilization,
    )

    # Baseline gate inputs
    rules_m = arm_metrics.get("rules")
    cheap_m = arm_metrics.get("cheap_off_the_shelf")
    rules_quality = (
        _arm_quality_evidence(rules_m, teacher)
        if rules_m is not None and teacher is not None
        else None
    )
    cheap_quality = (
        _arm_quality_evidence(cheap_m, teacher)
        if cheap_m is not None and teacher is not None
        else None
    )

    baselines_present = tuple(aid for aid in REQUIRED_BASELINE_ARMS if aid in arm_metrics)

    included_arms = [arm for arm in inp.arms if not arm.excluded]
    arm_accounting = tuple(
        ArmAccountingEvidence(
            arm_id=arm.arm_id,
            expected_examples=arm.expected_examples,
            prediction_records=len(arm.predictions),
            failed_examples=arm.failed_examples,
            filtered_examples=arm.filtered_examples,
            failed_example_reasons=arm.failed_example_reasons,
            filtered_example_reasons=arm.filtered_example_reasons,
        )
        for arm in included_arms
    )
    prediction_hashes_complete = all(
        result.excluded or result.prediction_sha256 is not None
        for result in arm_results
    )

    gate_inputs = GateInputs(
        teacher_minus_student=teacher_gap,
        teacher_minus_student_ci=teacher_gap_ci,
        rules_quality=rules_quality,
        cheap_quality=cheap_quality,
        rules_tco_comparison=inp.rules_tco_comparison,
        cheap_tco_comparison=inp.cheap_tco_comparison,
        trainer_numerical_ok=inp.trainer_numerical_ok,
        trainer_artifact_reload_ok=inp.trainer_artifact_reload_ok,
        trainer_smoke_eval_ok=inp.trainer_smoke_eval_ok,
        trainer_memory_cost_ok=inp.trainer_memory_cost_ok,
        quality_retention_point=retention_point,
        quality_retention_lower_95=retention_lower,
        quality_retention_upper_95=retention_upper,
        max_primary_task_regression=max_task_regression,
        json_schema_validity=finalist.json_schema_validity if finalist else None,
        ood_retention=ood_retention,
        ood_retention_lower_95=ood_retention_lower,
        ood_retention_upper_95=ood_retention_upper,
        critical_invariant_violations=(
            finalist.critical_invariant_violations if finalist else None
        ),
        economics=eco,
        seed_sets_by_arm=seed_sets_by_arm,
        finalist_arm_id=inp.finalist_arm_id,
        teacher_arm_id=inp.teacher_arm_id,
        matched_comparator_arm_id=inp.base_arm_id,
        baselines_present=baselines_present,
        paired_intervals=tuple(bootstrap_intervals),
        frozen_hashes_present=(
            inp.frozen_hashes_present is True and prediction_hashes_complete
        ),
        finalist_systems=tuple(
            summary
            for summary in finalist_systems.values()
            if summary is not None
        ),
        arm_accounting=arm_accounting,
        raw_text_provenance_complete=all(
            record.raw_text_provenance == "captured_model_output"
            for arm in included_arms
            for record in arm.predictions
        ),
    )
    gate_eval = evaluate_gates(gate_inputs, thresholds=th)

    limitations = list(inp.limitations)
    limitations.append("serving_economics_are_projected")
    limitations.extend(PERCENTILE_LIMITATIONS)
    if eco.notes:
        limitations.extend(eco.notes)

    report = ProofReport(
        report_id=inp.report_id,
        run_ids=inp.run_ids,
        protocol_id=inp.protocol_id,
        protocol_sha256=inp.protocol_sha256,
        proof_status=gate_eval.proof_status,
        first_failed_gate=gate_eval.first_failed_gate,
        unevaluated_gates=gate_eval.unevaluated_gates,
        arm_results=tuple(arm_results),
        quality_gates=gate_eval.quality_gates,
        uncertainty=uncertainty,
        economics=eco.to_dict(),
        exclusions=tuple(exclusions),
        limitations=tuple(dict.fromkeys(limitations)),
        created_at=inp.created_at,
    )
    return ProofEvaluationResult(
        report=report,
        arm_metrics=arm_metrics,
        systems=systems,
        gate_evaluation=gate_eval,
        uncertainty=uncertainty,
        economics=eco.to_dict(),
    )
