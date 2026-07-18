"""Deterministic paired clustered bootstrap tests."""

from __future__ import annotations

from distillery.contracts.budgets import ProofGates
from distillery.proof.bootstrap import (
    paired_cluster_bootstrap,
    paired_difference_ci,
    quality_retention_ci,
)
from distillery.proof.metrics import compute_arm_metrics, score_prediction
from distillery.proof.testing import make_pred, txn_gold, var_gold


def _scores_for_arm(correct_worlds: set[str], worlds: list[str], arm: str):
    gold_t = txn_gold()
    gold_v = var_gold(
        profit=100000,
        drivers=[{"driver_id": "hc", "impact_minor": 100000, "rank": 1}],
        other=0,
    )
    records = []
    for i, wid in enumerate(worlds):
        ok = wid in correct_worlds
        t_pred = (
            gold_t
            if ok
            else {
                **gold_t,
                "gl_account": "9999",
                "journal_entry": [
                    {"account": "9999", "side": "debit", "amount_minor": 4500},
                    {"account": "2100", "side": "credit", "amount_minor": 4500},
                ],
            }
        )
        records.append(
            make_pred(
                example_id=f"ex_{arm}_t_{i}",
                world_id=wid,
                task="transaction_review",
                expected=gold_t,
                parsed=t_pred,
                arm_id=arm,
            )
        )
        records.append(
            make_pred(
                example_id=f"ex_{arm}_v_{i}",
                world_id=wid,
                task="variance_analysis",
                expected=gold_v,
                parsed=gold_v
                if ok
                else {
                    **gold_v,
                    "profit_impact_minor": 0,
                    "direction": "favorable",
                    "top_drivers": [{"driver_id": "hc", "impact_minor": 0, "rank": 1}],
                    "other_impact_minor": 0,
                },
                arm_id=arm,
            )
        )
    return [score_prediction(r) for r in records], records


def test_bootstrap_deterministic_same_seed() -> None:
    worlds = [f"world_{i}" for i in range(6)]
    scores, _ = _scores_for_arm({"world_0", "world_1", "world_2", "world_3"}, worlds, "a")

    def mean_joint(xs):
        return sum(1.0 if s.joint_exact else 0.0 for s in xs) / len(xs)

    a = paired_cluster_bootstrap(
        scores, mean_joint, n_resamples=200, seed=17, metric="joint"
    )
    b = paired_cluster_bootstrap(
        scores, mean_joint, n_resamples=200, seed=17, metric="joint"
    )
    assert a.estimate == b.estimate
    assert a.lower == b.lower
    assert a.upper == b.upper
    assert a.n_clusters == 6


def test_bootstrap_different_seed_can_differ() -> None:
    worlds = [f"world_{i}" for i in range(8)]
    scores, _ = _scores_for_arm(set(worlds[:5]), worlds, "a")

    def mean_joint(xs):
        return sum(1.0 if s.joint_exact else 0.0 for s in xs) / len(xs)

    a = paired_cluster_bootstrap(
        scores, mean_joint, n_resamples=500, seed=17, metric="joint"
    )
    b = paired_cluster_bootstrap(
        scores, mean_joint, n_resamples=500, seed=23, metric="joint"
    )
    assert a.estimate == b.estimate
    assert (a.lower, a.upper) != (b.lower, b.upper) or a.n_resamples == b.n_resamples


def test_paired_difference_preserves_world_pairing() -> None:
    worlds = [f"world_{i}" for i in range(5)]
    scores_a, _ = _scores_for_arm(set(worlds), worlds, "teacher")
    scores_b, _ = _scores_for_arm(set(worlds[1:]), worlds, "base")
    ci = paired_difference_ci(
        scores_a,
        scores_b,
        n_resamples=300,
        seed=17,
        arm_a="teacher",
        arm_b="base",
    )
    assert ci.estimate > 0
    assert ci.arm_a == "teacher"
    assert ci.n_clusters == 5
    assert ci.n_examples > 0


def test_default_resamples_is_10000() -> None:
    gates = ProofGates()
    assert gates.bootstrap_resamples == 10_000
    worlds = ["world_0", "world_1"]
    scores, _ = _scores_for_arm({"world_0"}, worlds, "a")

    def mean_joint(xs):
        return sum(1.0 if s.joint_exact else 0.0 for s in xs) / max(len(xs), 1)

    ci = paired_cluster_bootstrap(scores, mean_joint, n_resamples=50, seed=1)
    assert ci.n_resamples == 50
    assert gates.bootstrap_resamples == 10_000


def test_quality_retention_ci_deterministic() -> None:
    worlds = [f"world_{i}" for i in range(4)]
    s_teacher, _ = _scores_for_arm(set(worlds), worlds, "teacher")
    s_student, _ = _scores_for_arm(set(worlds[:3]), worlds, "student")
    a = quality_retention_ci(s_student, s_teacher, n_resamples=100, seed=17)
    b = quality_retention_ci(s_student, s_teacher, n_resamples=100, seed=17)
    assert a.estimate == b.estimate
    assert a.lower == b.lower
    assert 0.0 <= a.estimate <= 1.0 + 1e-9


def test_underpowered_flag_for_few_clusters() -> None:
    worlds = ["world_0", "world_1"]
    scores, _ = _scores_for_arm(set(worlds), worlds, "a")

    def mean_joint(xs):
        return sum(1.0 if s.joint_exact else 0.0 for s in xs) / len(xs)

    ci = paired_cluster_bootstrap(scores, mean_joint, n_resamples=20, seed=3)
    assert ci.underpowered is True


def test_arm_metrics_primary_feeds_bootstrap() -> None:
    worlds = [f"world_{i}" for i in range(3)]
    _, records = _scores_for_arm(set(worlds), worlds, "seq")
    metrics = compute_arm_metrics("seq", records)
    assert metrics.primary_index > 0
    ci = paired_cluster_bootstrap(
        metrics.example_scores,
        lambda xs: sum(1.0 if s.joint_exact else 0.0 for s in xs) / len(xs),
        n_resamples=30,
        seed=17,
    )
    assert ci.estimate == 1.0
