"""Paired 95% bootstrap CIs clustered by world_id (10,000 resamples default)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from distillery.contracts.budgets import ProofGates
from distillery.contracts.tasks import TaskId
from distillery.proof.metrics import ExampleScore, compute_primary_index


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a scalar metric or arm difference."""

    estimate: float
    lower: float
    upper: float
    level: float
    n_resamples: int
    n_clusters: int
    n_examples: int
    underpowered: bool
    metric: str
    arm_a: str | None = None
    arm_b: str | None = None
    seed: int | None = None

    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "n_resamples": self.n_resamples,
            "n_clusters": self.n_clusters,
            "n_examples": self.n_examples,
            "underpowered": self.underpowered,
            "metric": self.metric,
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "seed": self.seed,
            "excludes_zero": self.excludes_zero(),
        }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear interpolation percentile on a sorted list (inclusive)."""
    if not sorted_vals:
        raise ValueError("empty sample for percentile")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _cluster_map(scores: Sequence[ExampleScore]) -> dict[str, list[ExampleScore]]:
    clusters: dict[str, list[ExampleScore]] = defaultdict(list)
    for s in scores:
        clusters[s.world_id].append(s)
    return dict(clusters)


def paired_cluster_bootstrap(
    scores: Sequence[ExampleScore],
    statistic: Callable[[Sequence[ExampleScore]], float],
    *,
    n_resamples: int | None = None,
    level: float = 0.95,
    seed: int = 17,
    metric: str = "metric",
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    """Cluster bootstrap by ``world_id`` with replacement.

    Entire worlds are resampled so examples sharing a world stay together.
    """
    gates = ProofGates()
    n_resamples = gates.bootstrap_resamples if n_resamples is None else n_resamples
    clusters = _cluster_map(scores)
    world_ids = sorted(clusters)
    if not world_ids:
        raise ValueError("no scores for bootstrap")

    point = statistic(scores)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(world_ids) for _ in world_ids]
        resampled: list[ExampleScore] = []
        for wid in drawn:
            resampled.extend(clusters[wid])
        samples.append(statistic(resampled))

    samples.sort()
    alpha = 1.0 - level
    lower = _percentile(samples, alpha / 2.0)
    upper = _percentile(samples, 1.0 - alpha / 2.0)
    return BootstrapCI(
        estimate=point,
        lower=lower,
        upper=upper,
        level=level,
        n_resamples=n_resamples,
        n_clusters=len(world_ids),
        n_examples=len(scores),
        underpowered=len(world_ids) < min_clusters_powered,
        metric=metric,
        seed=seed,
    )


def _mean_joint(scores: Sequence[ExampleScore]) -> float:
    if not scores:
        return 0.0
    return sum(1.0 if s.joint_exact else 0.0 for s in scores) / len(scores)


def _mean_schema(scores: Sequence[ExampleScore]) -> float:
    if not scores:
        return 0.0
    return sum(1.0 if s.json_schema_valid else 0.0 for s in scores) / len(scores)


def _primary_from_scores(scores: Sequence[ExampleScore]) -> float:
    if not scores:
        return 0.0
    txn = [s for s in scores if s.task == TaskId.TRANSACTION_REVIEW.value]
    var = [s for s in scores if s.task == TaskId.VARIANCE_ANALYSIS.value]
    txn_j = _mean_joint(txn) if txn else None
    var_j = _mean_joint(var) if var else None
    schema = _mean_schema(scores)
    return compute_primary_index(txn_j, var_j, schema)


def paired_difference_ci(
    scores_a: Sequence[ExampleScore],
    scores_b: Sequence[ExampleScore],
    *,
    statistic: Callable[[Sequence[ExampleScore]], float] | None = None,
    n_resamples: int | None = None,
    level: float = 0.95,
    seed: int = 17,
    metric: str = "primary_index_diff",
    arm_a: str = "a",
    arm_b: str = "b",
    min_clusters_powered: int = 10,
) -> BootstrapCI:
    """Paired cluster bootstrap of ``stat(A) - stat(B)`` on shared world_ids.

    Pairing is by ``world_id`` (and within-world example_id alignment when both
    arms share the same example ids). Worlds missing from either arm are dropped.
    """
    gates = ProofGates()
    n_resamples = gates.bootstrap_resamples if n_resamples is None else n_resamples
    stat = statistic or _primary_from_scores

    by_a = _cluster_map(scores_a)
    by_b = _cluster_map(scores_b)
    world_ids = sorted(set(by_a) & set(by_b))
    if not world_ids:
        raise ValueError("no overlapping world_id clusters for paired bootstrap")

    # Align example-level pairs within each world when example_ids match.
    aligned_a: list[ExampleScore] = []
    aligned_b: list[ExampleScore] = []
    for wid in world_ids:
        map_a = {s.example_id: s for s in by_a[wid]}
        map_b = {s.example_id: s for s in by_b[wid]}
        shared_ex = sorted(set(map_a) & set(map_b))
        if shared_ex:
            for eid in shared_ex:
                aligned_a.append(map_a[eid])
                aligned_b.append(map_b[eid])
        else:
            # Fall back to full world contents (same cluster membership).
            aligned_a.extend(by_a[wid])
            aligned_b.extend(by_b[wid])

    point = stat(aligned_a) - stat(aligned_b)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(world_ids) for _ in world_ids]
        ra: list[ExampleScore] = []
        rb: list[ExampleScore] = []
        for wid in drawn:
            map_a = {s.example_id: s for s in by_a[wid]}
            map_b = {s.example_id: s for s in by_b[wid]}
            shared_ex = sorted(set(map_a) & set(map_b))
            if shared_ex:
                for eid in shared_ex:
                    ra.append(map_a[eid])
                    rb.append(map_b[eid])
            else:
                ra.extend(by_a[wid])
                rb.extend(by_b[wid])
        samples.append(stat(ra) - stat(rb))

    samples.sort()
    alpha = 1.0 - level
    lower = _percentile(samples, alpha / 2.0)
    upper = _percentile(samples, 1.0 - alpha / 2.0)
    return BootstrapCI(
        estimate=point,
        lower=lower,
        upper=upper,
        level=level,
        n_resamples=n_resamples,
        n_clusters=len(world_ids),
        n_examples=len(aligned_a),
        underpowered=len(world_ids) < min_clusters_powered,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seed=seed,
    )


def quality_retention_ci(
    student_scores: Sequence[ExampleScore],
    teacher_scores: Sequence[ExampleScore],
    *,
    n_resamples: int | None = None,
    seed: int = 17,
) -> BootstrapCI:
    """Bootstrap CI for student_primary / teacher_primary (paired by world)."""

    def retention(pair_scores: Sequence[ExampleScore]) -> float:
        # Not used directly; we bootstrap via paired_difference-style resampling.
        raise NotImplementedError

    _ = retention
    gates = ProofGates()
    n_resamples = gates.bootstrap_resamples if n_resamples is None else n_resamples
    by_s = _cluster_map(student_scores)
    by_t = _cluster_map(teacher_scores)
    world_ids = sorted(set(by_s) & set(by_t))
    if not world_ids:
        raise ValueError("no overlapping worlds for retention CI")

    def _align(worlds: list[str]) -> tuple[list[ExampleScore], list[ExampleScore]]:
        sa: list[ExampleScore] = []
        ta: list[ExampleScore] = []
        for wid in worlds:
            sa.extend(by_s[wid])
            ta.extend(by_t[wid])
        return sa, ta

    s0, t0 = _align(world_ids)
    t_primary = _primary_from_scores(t0)
    s_primary = _primary_from_scores(s0)
    point = 0.0 if t_primary == 0.0 else s_primary / t_primary

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = [rng.choice(world_ids) for _ in world_ids]
        s_r, t_r = _align(drawn)
        tp = _primary_from_scores(t_r)
        sp = _primary_from_scores(s_r)
        samples.append(0.0 if tp == 0.0 else sp / tp)
    samples.sort()
    alpha = 0.05
    return BootstrapCI(
        estimate=point,
        lower=_percentile(samples, alpha / 2.0),
        upper=_percentile(samples, 1.0 - alpha / 2.0),
        level=0.95,
        n_resamples=n_resamples,
        n_clusters=len(world_ids),
        n_examples=len(s0),
        underpowered=len(world_ids) < 10,
        metric="quality_retention",
        arm_a="student",
        arm_b="teacher",
        seed=seed,
    )
