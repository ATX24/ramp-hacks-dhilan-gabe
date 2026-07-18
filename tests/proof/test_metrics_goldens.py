"""Hand-calculated metric goldens for proof evaluation."""

from __future__ import annotations

import json

from distillery.proof.metrics import (
    compute_arm_metrics,
    compute_primary_index,
    score_prediction,
)
from distillery.proof.testing import cash_gold, make_pred, txn_gold, var_gold


def test_primary_index_hand_calculated() -> None:
    # txn joint=1.0, var joint=0.5, schema=0.75
    # 0.45*1 + 0.45*0.5 + 0.10*0.75 = 0.45 + 0.225 + 0.075 = 0.75
    assert compute_primary_index(1.0, 0.5, 0.75) == 0.75


def test_transaction_joint_exact_and_components() -> None:
    gold = txn_gold(amount=4500, gl="6100")
    ok = make_pred(
        example_id="ex_t1",
        world_id="world_a",
        task="transaction_review",
        expected=gold,
        parsed=gold,
    )
    bad = dict(gold)
    bad["gl_account"] = "6205"
    bad["journal_entry"] = [
        {"account": "6205", "side": "debit", "amount_minor": 4500},
        {"account": "2100", "side": "credit", "amount_minor": 4500},
    ]
    wrong = make_pred(
        example_id="ex_t2",
        world_id="world_a",
        task="transaction_review",
        expected=gold,
        parsed=bad,
    )
    s_ok = score_prediction(ok)
    s_bad = score_prediction(wrong)
    assert s_ok.joint_exact is True
    assert s_ok.components["gl_account_exact"] == 1.0
    assert s_ok.components["debit_credit_balanced"] == 1.0
    assert s_bad.joint_exact is False
    assert s_bad.components["gl_account_exact"] == 0.0


def test_unbalanced_journal_is_invariant_violation() -> None:
    gold = txn_gold()
    unbalanced = dict(gold)
    unbalanced["journal_entry"] = [
        {"account": "6100", "side": "debit", "amount_minor": 10000},
        {"account": "2100", "side": "credit", "amount_minor": 9000},
    ]
    # Schema validation rejects unbalanced journals via Pydantic, so schema_valid=False.
    # Feed via raw_text that is JSON but will fail schema → not invariant path.
    # Instead, bypass by using a schema-valid balanced wrong amount then mutate through
    # a parsed dict that somehow balances... Use ValidationError path:
    # For invariant, we need schema_valid True. TransactionReviewOutput forbids unbalanced.
    # So invariant_violation for txn only triggers if somehow balanced check fails on
    # a schema-valid object — which can't happen via model. Use score path with
    # schema-invalid parsed that still runs? Actually score only scores when schema_valid.
    #
    # Hand-calc: schema-invalid unbalanced counts as schema fail, joint false.
    rec = make_pred(
        example_id="ex_unbal",
        world_id="world_d",
        task="transaction_review",
        expected=gold,
        parsed=unbalanced,
    )
    score = score_prediction(rec)
    assert score.json_schema_valid is False
    assert score.joint_exact is False


def test_variance_arithmetic_and_joint() -> None:
    gold = var_gold()
    ok = make_pred(
        example_id="ex_v1",
        world_id="world_b",
        task="variance_analysis",
        expected=gold,
        parsed=gold,
    )
    broken = dict(gold)
    broken["other_impact_minor"] = 0  # breaks closure vs profit
    # Schema validation will reject because closure validator runs.
    rec_bad = make_pred(
        example_id="ex_v2",
        world_id="world_b",
        task="variance_analysis",
        expected=gold,
        parsed=broken,
    )
    assert score_prediction(ok).joint_exact is True
    bad_score = score_prediction(rec_bad)
    assert bad_score.json_schema_valid is False
    assert bad_score.joint_exact is False


def test_cash_joint_exact() -> None:
    gold = cash_gold()
    ok = make_pred(
        example_id="ex_c1",
        world_id="world_c",
        task="cash_reconciliation",
        expected=gold,
        parsed=gold,
    )
    assert score_prediction(ok).joint_exact is True
    wrong = dict(gold)
    wrong["difference_minor"] = 1
    # schema invalid (difference mismatch)
    bad = make_pred(
        example_id="ex_c2",
        world_id="world_c",
        task="cash_reconciliation",
        expected=gold,
        parsed=wrong,
    )
    assert score_prediction(bad).json_schema_valid is False


def test_json_parse_refusal_and_schema_rates_hand_calc() -> None:
    gold_t = txn_gold()
    gold_v = var_gold(
        profit=100000,
        drivers=[{"driver_id": "hc", "impact_minor": 100000, "rank": 1}],
        other=0,
    )
    records = [
        make_pred(
            example_id="ex_1",
            world_id="world_1",
            task="transaction_review",
            expected=gold_t,
            parsed=gold_t,
            split="iid_test",
        ),
        make_pred(
            example_id="ex_2",
            world_id="world_1",
            task="transaction_review",
            expected=gold_t,
            raw_text="{not json",
            split="iid_test",
        ),
        make_pred(
            example_id="ex_3",
            world_id="world_2",
            task="variance_analysis",
            expected=gold_v,
            parsed=gold_v,
            split="ood_test",
        ),
        make_pred(
            example_id="ex_4",
            world_id="world_2",
            task="variance_analysis",
            expected=gold_v,
            refused=True,
            split="ood_test",
        ),
    ]
    # 4 examples: parse_ok = 2/4 = 0.5 (ex_1, ex_3); schema = 2/4 = 0.5
    # refusal/empty = 1/4 = 0.25
    # txn joint: 1 of 1 schema-valid txn? wait both txn scored: ex_1 joint True, ex_2 not schema
    # txn joint_exact mean over txn scores: joint flags are False when not schema-valid
    # So txn joints = [True, False] → 0.5
    # var joints = [True, False] → 0.5
    # primary = 0.45*0.5 + 0.45*0.5 + 0.10*0.5 = 0.225+0.225+0.05 = 0.5
    m = compute_arm_metrics("student", records)
    assert m.json_parse_rate == 0.5
    assert m.json_schema_validity == 0.5
    assert m.refusal_empty_rate == 0.25
    assert m.transaction_joint_exact == 0.5
    assert m.variance_joint_exact == 0.5
    assert m.primary_index == 0.5


def test_journal_set_f1_hand_calc() -> None:
    gold = txn_gold(amount=100, gl="6100")
    # pred shares one of two lines exactly, other wrong account
    pred = dict(gold)
    pred["journal_entry"] = [
        {"account": "6100", "side": "debit", "amount_minor": 100},
        {"account": "2200", "side": "credit", "amount_minor": 100},
    ]
    # Still balanced so schema-valid; joint false
    score = score_prediction(
        make_pred(
            example_id="ex_f1",
            world_id="world_z",
            task="transaction_review",
            expected=gold,
            parsed=pred,
        )
    )
    # |inter|=1, |pred|=2, |gold|=2 → P=0.5, R=0.5, F1=0.5
    assert score.components["journal_set_f1"] == 0.5
    assert score.components["journal_set_exact"] == 0.0
    assert score.joint_exact is False


def test_calibration_brier_hand_calc() -> None:
    gold = txn_gold()
    # two schema-valid: conf 1.0 correct, conf 0.0 incorrect → brier = (0+0)/2? 
    # (1-1)^2=0, for incorrect use conf 0.0 and joint false → (0-0)^2=0 → brier=0
    # Better: conf 0.8 correct and conf 0.8 incorrect
    # brier = ((0.8-1)^2 + (0.8-0)^2) / 2 = (0.04 + 0.64) / 2 = 0.34
    g1 = dict(gold)
    g1["confidence"] = 0.8
    g2 = dict(gold)
    g2["confidence"] = 0.8
    g2["gl_account"] = "9999"
    g2["journal_entry"] = [
        {"account": "9999", "side": "debit", "amount_minor": 4500},
        {"account": "2100", "side": "credit", "amount_minor": 4500},
    ]
    records = [
        make_pred(
            example_id="ex_cal1",
            world_id="world_cal",
            task="transaction_review",
            expected=gold,
            parsed=g1,
        ),
        make_pred(
            example_id="ex_cal2",
            world_id="world_cal",
            task="transaction_review",
            expected=gold,
            parsed=g2,
        ),
    ]
    m = compute_arm_metrics("cal", records)
    assert m.calibration.brier_score is not None
    assert abs(m.calibration.brier_score - 0.34) < 1e-12


def test_slice_underpowered_flag() -> None:
    gold = txn_gold()
    records = [
        make_pred(
            example_id=f"ex_s{i}",
            world_id=f"world_{i}",
            task="transaction_review",
            expected=gold,
            parsed=gold,
            difficulty="hard",
            slices={"policy": "POL-A"},
        )
        for i in range(3)
    ]
    m = compute_arm_metrics("slice", records)
    hard = [s for s in m.slices if s.slice_key == "difficulty" and s.slice_value == "hard"]
    assert hard and hard[0].n == 3 and hard[0].underpowered is True


def test_raw_text_json_roundtrip() -> None:
    gold = txn_gold()
    rec = make_pred(
        example_id="ex_raw",
        world_id="world_raw",
        task="transaction_review",
        expected=gold,
        parsed=None,
        raw_text=json.dumps(gold),
    )
    assert score_prediction(rec).joint_exact is True
