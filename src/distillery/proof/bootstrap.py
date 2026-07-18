"""Strict paired multi-seed bootstrap clustered by world_id."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from distillery.contracts.budgets import ProofGates
from distillery.contracts.tasks import TaskId
from distillery.proof.metrics import ExampleScore, compute_primary_index

PRIMARY_METRICS: tuple[str, ...] = (
    "transaction_joint_exact",
    "variance_joint_exact",
    "json_schema_validity",
    "primary_index",
    "ood_primary_index",
)
PERCENTILE_METHOD = "percentile_linear_interpolation"
PERCENTILE_LIMITATIONS: tuple[str, ...] = (
    "percentile_intervals_are_not_bias_corrected_or_accelerated",
    "finite_resamples_add_monte_carlo_error",
    "coverage_can_be_unreliable_with_few_clusters_or_boundary_metrics",
    "world_id_is_the_resampling_unit_and_seed_replicates_stay_with_their_world",
)
Statistic = Callable[[Sequence[ExampleScore]], float | None]


@dataclass(frozen=True)
class BootstrapCI:
    """A scalar clustered-bootstrap interval with validity accounting."""

    estimate: float | None
    lower: float | None
    upper: float | None
    level: float
    n_resamples: int
    valid_resamples: int
    excluded_resamples: int
    n_clusters: int
    n_examples: int
    underpowered: bool
    metric: str
    arm_a: str | None = None
    arm_b: str | None = None
    seed: int | None = None
    undefined_reason: str | None = None
    percentile_method: str = PERCENTILE_METHOD
    limitations: tuple[str, ...] = PERCENTILE_LIMITATIONS

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError("bootstrap confidence level must be in (0, 1)")
        if self.n_resamples <= 0:
            raise ValueError("bootstrap n_resamples must be positive")
        if self.valid_resamples < 0 or self.excluded_resamples < 0:
            raise ValueError("bootstrap resample counts must be nonnegative")
        if self.valid_resamples + self.excluded_resamples != self.n_resamples:
            raise ValueError(
                "valid_resamples + excluded_resamples must equal n_resamples"
            )
        if self.n_clusters < 0 or self.n_examples < 0:
            raise ValueError("cluster and example counts must be nonnegative")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("bootstrap bounds must both be defined or missing")
        if self.estimate is None and self.lower is not None:
            raise ValueError("undefined point estimate cannot have CI bounds")

    @property
    def defined(self) -> bool:
        return (
            self.estimate is not None
            and self.lower is not None
            and self.upper is not None
        )

    @property
    def proof_ready(self) -> bool:
        return (
            self.defined
            and not self.underpowered
            and self.excluded_resamples == 0
            and self.valid_resamples == self.n_resamples
            and self.n_resamples == ProofGates().bootstrap_resamples
            and self.level == 0.95
            and self.percentile_method == PERCENTILE_METHOD
        )

    @property
    def interval_id(self) -> str:
        if self.arm_a is None:
            return self.metric
        if self.arm_b is None:
            return f"arm::{self.arm_a}::{self.metric}"
        return f"pair::{self.arm_a}::{self.arm_b}::{self.metric}"

    def excludes_zero(self) -> bool:
        if self.lower is None or self.upper is None:
            return False
        return self.lower > 0.0 or self.upper < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_id": self.interval_id,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "defined": self.defined,
            "level": self.level,
            "n_resamples": self.n_resamples,
            "valid_resamples": self.valid_resamples,
            "excluded_resamples": self.excluded_resamples,
            "n_clusters": self.n_clusters,
            "n_examples": self.n_examples,
            "underpowered": self.underpowered,
            "proof_ready": self.proof_ready,
            "metric": self.metric,
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "seed": self.seed,
            "undefined_reason": self.undefined_reason,
            "percentile_method": self.percentile_method,
            "limitations": list(self.limitations),
            "excludes_zero": self.excludes_zero(),
        }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile on a sorted finite sample."""

    if not sorted_vals:
        raise ValueError("empty sample for percentile")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    floor = int(k)
    ceil = min(floor + 1, len(sorted_vals) - 1)
    if floor == ceil:
        return sorted_vals[floor]
    return sorted_vals[floor] + (
        sorted_vals[ceil] - sorted_vals[floor]
    ) * (k - floor)


def _validate_unique_scores(
    scores: Sequence[ExampleScore],
    *,
    arm: str,
) -> None:
    seen: set[tuple[int, str]] = set()
    example_metadata: dict[str, tuple[str, str, str]] = {}
    for score in scores:
        key = (score.seed, score.example_id)
        if key in seen:
            raise ValueError(
                f"{arm}: duplicate seed/example record {key}"
            )
        seen.add(key)
        metadata = (score.world_id, score.task, score.split)
        previous = example_metadata.setdefault(score.example_id, metadata)
        if previous != metadata:
            raise ValueError(
                f"{arm}: example_id {score.example_id} changes "
                "world_id/task/split across seeds"
            )


def validate_paired_scores(
    scores_a: Sequence[ExampleScore],
    scores_b: Sequence[ExampleScore],
    *,
    arm_a: str,
    arm_b: str,
) -> None:
    """Require exact seed/example/world/task/split identity across two arms."""

    _validate_unique_scores(scores_a, arm=arm_a)
    _validate_unique_scores(scores_b, arm=arm_b)

    def identities(scores: Sequence[ExampleScore]) -> set[tuple[int, str, str, str, str]]:
        return {
            (
                score.seed,
                score.example_id,
                score.world_id,
                score.task,
                score.split,
            )
            for score in scores
        }

    ids_a = identities(scores_a)
    ids_b = identities(scores_b)
    if ids_a != ids_b:
        raise ValueError(
            f"{arm_a}/{arm_b}: paired prediction identities differ "
            f"(only_{arm_a}={len(ids_a - ids_b)}, "
            f"only_{arm_b}={len(ids_b - ids_a)})"
        )


def _cluster_map(
    scores: Sequence[ExampleScore],
    *,
    arm: str,
) -> dict[str, list[ExampleScore]]:
    _validate_unique_scores(scores, arm=arm)
    clusters: dict[str, list[ExampleScore]] = defaultdict(list)
    for score in scores:
        clusters[score.world_id].append(score)
    return {
        world_id: sorted(
            members,
            key=lambda score: (score.seed, score.example_id),
        )
        for world_id, members in clusters.items()
    }


def metric_value(
    scores: Sequence[ExampleScore],
    metric: str,
) -> float | None:
    """Compute a primary metric using the same point-estimate semantics."""

    selected = list(scores)
    if metric == "ood_primary_index":
        selected = [score for score in selected if score.split == "ood_test"]
        metric = "primary_index"
    if not selected:
        return None

    if metric == "transaction_joint_exact":
        values = [
            float(score.joint_exact)
            for score in selected
            if score.task == TaskId.TRANSACTION_REVIEW.value
        ]
        return sum(values) / len(values) if values else None
    if metric == "variance_joint_exact":
        values = [
            float(score.joint_exact)
            for score in selected
            if score.task == TaskId.VARIANCE_ANALYSIS.value
        ]
        return sum(values) / len(values) if values else None
    if metric == "json_schema_validity":
        return sum(
            float(score.json_schema_valid) for score in selected
        ) / len(selected)
    if metric == "primary_index":
        transaction = metric_value(
            selected,
            "transaction_joint_exact",
        )
        variance = metric_value(selected, "variance_joint_exact")
        if transaction is None or variance is None:
            return None
        schema = metric_value(selected, "json_schema_validity")
        assert schema is not None
        return compute_primary_index(transaction, variance, schema)
    raise ValueError(f"unsupported bootstrap metric: {metric}")


def _build_ci(
    *,
    point: float | None,
    samples: list[float],
    level: float,
    n_resamples: int,
    n_clusters: int,
    n_examples: int,
    min_clusters_powered: int,
    metric: str,
    arm_a: str | None,
    arm_b: str | None,
    seed: int,
    undefined_reason: str | None = None,
) -> BootstrapCI:
    valid = len(samples)
    excluded = n_resamples - valid
    samples.sort()
    if point is None or not samples:
        lower = None
        upper = None
        reason = undefined_reason or "point_or_all_bootstrap_draws_undefined"
    else:
        alpha = 1.0 - level
        lower = _percentile(samples, alpha / 2.0)
        upper = _percentile(samples, 1.0 - alpha / 2.0)
        reason = (
            undefined_reason
            or (
                "some_bootstrap_draws_undefined"
                if excluded
                else None
            )
        )
    return BootstrapCI(
        estimate=point,
        lower=lower,
        upper=upper,
        level=level,
        n_resamples=n_resamples,
        valid_resamples=valid,
        excluded_resamples=excluded,
        n_clusters=n_clusters,
        n_examples=n_examples,
        underpowered=n_clusters < min_clusters_powered or valid == 0,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seed=seed,
        undefined_reason=reason,
    )


def _empty_ci(
    *,
    metric: str,
    arm_a: str,
    arm_b: str | None,
    n_resamples: int | None,
    level: float,
    seed: int,
    reason: str,
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    resolved_resamples = (
        ProofGates().bootstrap_resamples
        if n_resamples is None
        else n_resamples
    )
    return _build_ci(
        point=None,
        samples=[],
        level=level,
        n_resamples=resolved_resamples,
        n_clusters=0,
        n_examples=0,
        min_clusters_powered=min_clusters_powered,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seed=seed,
        undefined_reason=reason,
    )


def paired_cluster_bootstrap(
    scores: Sequence[ExampleScore],
    statistic: Statistic,
    *,
    n_resamples: int | None = None,
    level: float = 0.95,
    seed: int = 17,
    metric: str = "metric",
    arm_id: str = "arm",
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    """Cluster bootstrap one arm; all seed replicates stay in each world."""

    gates = ProofGates()
    n_resamples = gates.bootstrap_resamples if n_resamples is None else n_resamples
    clusters = _cluster_map(scores, arm=arm_id)
    world_ids = sorted(clusters)
    if not world_ids:
        raise ValueError("no scores for bootstrap")

    point = statistic(scores)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(world_ids) for _ in world_ids]
        resampled = [
            score
            for world_id in drawn
            for score in clusters[world_id]
        ]
        value = statistic(resampled)
        if value is not None:
            samples.append(value)
    return _build_ci(
        point=point,
        samples=samples,
        level=level,
        n_resamples=n_resamples,
        n_clusters=len(world_ids),
        n_examples=len(scores),
        min_clusters_powered=min_clusters_powered,
        metric=metric,
        arm_a=arm_id,
        arm_b=None,
        seed=seed,
    )


def arm_metric_ci(
    scores: Sequence[ExampleScore],
    metric: str,
    *,
    arm_id: str,
    n_resamples: int | None = None,
    level: float = 0.95,
    seed: int = 17,
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    metric_scores = (
        [score for score in scores if score.split == "ood_test"]
        if metric == "ood_primary_index"
        else scores
    )
    statistic_metric = (
        "primary_index" if metric == "ood_primary_index" else metric
    )
    if not metric_scores:
        return _empty_ci(
            metric=metric,
            arm_a=arm_id,
            arm_b=None,
            n_resamples=n_resamples,
            level=level,
            seed=seed,
            reason="no_examples_for_metric",
            min_clusters_powered=min_clusters_powered,
        )
    return paired_cluster_bootstrap(
        metric_scores,
        lambda sample: metric_value(sample, statistic_metric),
        n_resamples=n_resamples,
        level=level,
        seed=seed,
        metric=metric,
        arm_id=arm_id,
        min_clusters_powered=min_clusters_powered,
    )


def paired_difference_ci(
    scores_a: Sequence[ExampleScore],
    scores_b: Sequence[ExampleScore],
    *,
    statistic: Statistic | None = None,
    score_metric: str = "primary_index",
    n_resamples: int | None = None,
    level: float = 0.95,
    seed: int = 17,
    metric: str = "primary_index_difference",
    arm_a: str = "a",
    arm_b: str = "b",
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    """Strict paired difference with no intersections or fallback means."""

    if score_metric == "ood_primary_index":
        scores_a = [
            score for score in scores_a if score.split == "ood_test"
        ]
        scores_b = [
            score for score in scores_b if score.split == "ood_test"
        ]
        score_metric = "primary_index"
    if not scores_a and not scores_b:
        return _empty_ci(
            metric=metric,
            arm_a=arm_a,
            arm_b=arm_b,
            n_resamples=n_resamples,
            level=level,
            seed=seed,
            reason="no_paired_examples_for_metric",
            min_clusters_powered=min_clusters_powered,
        )
    validate_paired_scores(
        scores_a,
        scores_b,
        arm_a=arm_a,
        arm_b=arm_b,
    )
    gates = ProofGates()
    n_resamples = gates.bootstrap_resamples if n_resamples is None else n_resamples
    stat = statistic or (lambda sample: metric_value(sample, score_metric))
    by_a = _cluster_map(scores_a, arm=arm_a)
    by_b = _cluster_map(scores_b, arm=arm_b)
    world_ids = sorted(by_a)
    if not world_ids:
        raise ValueError("no paired world_id clusters")

    point_a = stat(scores_a)
    point_b = stat(scores_b)
    point = (
        point_a - point_b
        if point_a is not None and point_b is not None
        else None
    )
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(world_ids) for _ in world_ids]
        resampled_a = [
            score
            for world_id in drawn
            for score in by_a[world_id]
        ]
        resampled_b = [
            score
            for world_id in drawn
            for score in by_b[world_id]
        ]
        value_a = stat(resampled_a)
        value_b = stat(resampled_b)
        if value_a is not None and value_b is not None:
            samples.append(value_a - value_b)
    return _build_ci(
        point=point,
        samples=samples,
        level=level,
        n_resamples=n_resamples,
        n_clusters=len(world_ids),
        n_examples=len(scores_a),
        min_clusters_powered=min_clusters_powered,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seed=seed,
    )


def ratio_metric_ci(
    numerator_scores: Sequence[ExampleScore],
    denominator_scores: Sequence[ExampleScore],
    *,
    score_metric: str,
    metric: str,
    arm_a: str,
    arm_b: str,
    n_resamples: int | None = None,
    level: float = 0.95,
    seed: int = 17,
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    """Strict paired ratio; nonpositive denominators are undefined."""

    if score_metric == "ood_primary_index":
        numerator_scores = [
            score
            for score in numerator_scores
            if score.split == "ood_test"
        ]
        denominator_scores = [
            score
            for score in denominator_scores
            if score.split == "ood_test"
        ]
        score_metric = "primary_index"
    if not numerator_scores and not denominator_scores:
        return _empty_ci(
            metric=metric,
            arm_a=arm_a,
            arm_b=arm_b,
            n_resamples=n_resamples,
            level=level,
            seed=seed,
            reason="no_paired_examples_for_metric",
            min_clusters_powered=min_clusters_powered,
        )
    validate_paired_scores(
        numerator_scores,
        denominator_scores,
        arm_a=arm_a,
        arm_b=arm_b,
    )
    gates = ProofGates()
    n_resamples = gates.bootstrap_resamples if n_resamples is None else n_resamples
    by_num = _cluster_map(numerator_scores, arm=arm_a)
    by_den = _cluster_map(denominator_scores, arm=arm_b)
    world_ids = sorted(by_num)
    if not world_ids:
        raise ValueError("no paired world_id clusters")

    numerator = metric_value(numerator_scores, score_metric)
    denominator = metric_value(denominator_scores, score_metric)
    point = (
        numerator / denominator
        if numerator is not None
        and denominator is not None
        and denominator > 0
        else None
    )
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(world_ids) for _ in world_ids]
        num_sample = [
            score
            for world_id in drawn
            for score in by_num[world_id]
        ]
        den_sample = [
            score
            for world_id in drawn
            for score in by_den[world_id]
        ]
        num_value = metric_value(num_sample, score_metric)
        den_value = metric_value(den_sample, score_metric)
        if (
            num_value is not None
            and den_value is not None
            and den_value > 0
        ):
            samples.append(num_value / den_value)
    undefined_reason = (
        "nonpositive_or_missing_point_denominator"
        if point is None
        and (denominator is None or denominator <= 0)
        else None
    )
    return _build_ci(
        point=point,
        samples=samples,
        level=level,
        n_resamples=n_resamples,
        n_clusters=len(world_ids),
        n_examples=len(numerator_scores),
        min_clusters_powered=min_clusters_powered,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seed=seed,
        undefined_reason=undefined_reason,
    )


def quality_retention_ci(
    student_scores: Sequence[ExampleScore],
    teacher_scores: Sequence[ExampleScore],
    *,
    n_resamples: int | None = None,
    seed: int = 17,
    arm_a: str = "student",
    arm_b: str = "teacher",
) -> BootstrapCI:
    """Bootstrap student-primary / teacher-primary with undefined ratios."""

    return ratio_metric_ci(
        student_scores,
        teacher_scores,
        score_metric="primary_index",
        metric="quality_retention",
        arm_a=arm_a,
        arm_b=arm_b,
        n_resamples=n_resamples,
        seed=seed,
    )
