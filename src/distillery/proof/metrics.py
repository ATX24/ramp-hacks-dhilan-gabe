"""Deterministic task metrics, primary index, slices, and calibration."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from distillery.contracts.budgets import (
    PRIMARY_INDEX_WEIGHTS,
    PRIMARY_INDEX_WEIGHTS_V2,
    PrimaryIndexWeights,
    PrimaryIndexWeightsV2,
)
from distillery.contracts.tasks import (
    CashReconciliationOutput,
    MerchantTaggingOutput,
    TaskId,
    TransactionReviewOutput,
    VarianceAnalysisOutput,
)

JsonParseStatus = Literal["ok", "invalid_json", "empty", "refusal"]
RawTextProvenance = Literal["captured_model_output", "fixture_serialization"]


class PredictionRecord(BaseModel):
    """One immutable prediction row paired with gold expected output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    world_id: str
    group_id: str = ""
    task: str
    difficulty: str = "medium"
    split: str = "iid_test"
    template_family: str = ""
    arm_id: str = ""
    seed: int
    raw_text: str
    raw_text_provenance: RawTextProvenance
    parsed: dict[str, Any] | None = None
    refused: bool = False
    latency_ms: float | None = None
    output_tokens: int | None = None
    expected_output: dict[str, Any]
    # Optional slice dimensions (policy, GL, variance driver, renderer, etc.)
    slices: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _parsed_cache_matches_raw_text(self) -> PredictionRecord:
        """A parser cache may never override captured raw model output."""

        try:
            decoded = json.loads(self.raw_text)
        except json.JSONDecodeError:
            return self
        if self.parsed is not None and decoded != self.parsed:
            raise ValueError("parsed cache does not match raw_text JSON")
        return self


@dataclass(frozen=True)
class ExampleScore:
    example_id: str
    world_id: str
    seed: int
    task: str
    split: str
    difficulty: str
    template_family: str
    slices: dict[str, str]
    json_parse_ok: bool
    json_schema_valid: bool
    refused_or_empty: bool
    joint_exact: bool
    confidence: float | None
    components: dict[str, float]
    invariant_violation: bool = False


@dataclass
class CalibrationMetrics:
    brier_score: float | None
    adaptive_ece: float | None
    risk_coverage: list[dict[str, float]]
    n: int


@dataclass
class SliceReport:
    slice_key: str
    slice_value: str
    n: int
    joint_exact: float | None
    json_schema_validity: float | None
    underpowered: bool


@dataclass
class ArmMetrics:
    arm_id: str
    n: int
    json_parse_rate: float
    json_schema_validity: float
    refusal_empty_rate: float
    transaction_joint_exact: float | None
    variance_joint_exact: float | None
    cash_joint_exact: float | None
    primary_index: float
    components: dict[str, float]
    task_metrics: dict[str, dict[str, float | None]]
    calibration: CalibrationMetrics
    slices: list[SliceReport]
    merchant_joint_exact: float | None = None
    primary_index_v2: float | None = None
    seeds: tuple[int, ...] = ()
    seed_metrics: dict[int, dict[str, float | None]] = field(default_factory=dict)
    example_scores: list[ExampleScore] = field(default_factory=list)
    critical_invariant_violations: int = 0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _set_prf(pred: set[Any], gold: set[Any]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0 if gold else 1.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0
    inter = len(pred & gold)
    precision = inter / len(pred)
    recall = inter / len(gold)
    f1 = _f1(precision, recall)
    assert f1 is not None
    return precision, recall, f1


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true:
        return None
    labels = sorted(set(y_true) | set(y_pred))
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _f1(precision, recall)
        f1s.append(0.0 if f1 is None else f1)
    return sum(f1s) / len(f1s) if f1s else None


def _multilabel_macro_f1(
    y_true: list[set[str]],
    y_pred: list[set[str]],
) -> float | None:
    """Macro-average binary F1 independently across exception labels."""

    if not y_true:
        return None
    labels = sorted(
        set().union(*y_true, *y_pred)
        if y_true or y_pred
        else set()
    )
    if not labels:
        return 1.0
    f1s: list[float] = []
    for label in labels:
        tp = sum(
            1
            for truth, pred in zip(y_true, y_pred, strict=True)
            if label in truth and label in pred
        )
        fp = sum(
            1
            for truth, pred in zip(y_true, y_pred, strict=True)
            if label not in truth and label in pred
        )
        fn = sum(
            1
            for truth, pred in zip(y_true, y_pred, strict=True)
            if label in truth and label not in pred
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(_f1(precision, recall) or 0.0)
    return sum(f1s) / len(f1s)


def classify_raw_text(raw_text: str | None, refused: bool) -> JsonParseStatus:
    if refused:
        return "refusal"
    if raw_text is None:
        return "empty"
    text = raw_text.strip()
    if not text:
        return "empty"
    lowered = text.lower()
    if lowered in {"i refuse", "refuse", "n/a", "none"} or lowered.startswith("i cannot"):
        return "refusal"
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json"
    return "ok"


def parse_prediction(
    record: PredictionRecord,
) -> tuple[dict[str, Any] | None, JsonParseStatus, bool]:
    """Return (parsed_dict, status, schema_valid)."""
    if record.refused:
        return None, "refusal", False
    status = classify_raw_text(record.raw_text, refused=False)
    if status != "ok":
        return None, status, False
    decoded = json.loads(record.raw_text)
    if not isinstance(decoded, dict):
        return None, status, False
    schema_valid = _schema_valid(record.task, decoded)
    parsed = decoded
    return parsed, status, schema_valid


def _schema_valid(task: str, parsed: dict[str, Any]) -> bool:
    try:
        if task == TaskId.TRANSACTION_REVIEW.value:
            TransactionReviewOutput.model_validate(parsed)
            return True
        if task == TaskId.VARIANCE_ANALYSIS.value:
            VarianceAnalysisOutput.model_validate(parsed)
            return True
        if task == TaskId.CASH_RECONCILIATION.value:
            CashReconciliationOutput.model_validate(parsed)
            return True
        if task == TaskId.MERCHANT_TAGGING.value:
            MerchantTaggingOutput.model_validate(parsed)
            return True
    except ValidationError:
        return False
    # Unknown task: require dict with matching task/schema_version fields.
    return (
        isinstance(parsed.get("task"), str)
        and isinstance(parsed.get("schema_version"), str)
        and parsed.get("task") == task
    )


def _journal_line_set(entry: list[dict[str, Any]] | tuple[Any, ...]) -> set[tuple[str, str, int]]:
    out: set[tuple[str, str, int]] = set()
    for line in entry:
        if isinstance(line, dict):
            out.add((str(line["account"]), str(line["side"]), int(line["amount_minor"])))
        else:
            out.add((str(line.account), str(line.side), int(line.amount_minor)))
    return out


def _evidence_set(items: list[dict[str, Any]] | tuple[Any, ...]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for item in items:
        if isinstance(item, dict):
            out.add((str(item["source_id"]), str(item["field"]), str(item["value"])))
        else:
            out.add((str(item.source_id), str(item.field), str(item.value)))
    return out


def _score_transaction(
    gold: dict[str, Any], pred: dict[str, Any]
) -> tuple[dict[str, float], bool, bool]:
    components: dict[str, float] = {}
    gl_exact = float(gold.get("gl_account") == pred.get("gl_account"))
    components["gl_account_exact"] = gl_exact

    gold_lines = _journal_line_set(gold.get("journal_entry") or [])
    pred_lines = _journal_line_set(pred.get("journal_entry") or [])
    p, r, f1 = _set_prf(pred_lines, gold_lines)
    components["journal_set_exact"] = float(gold_lines == pred_lines)
    components["journal_set_f1"] = f1
    components["journal_set_precision"] = p
    components["journal_set_recall"] = r

    pred_debits = sum(
        int(x["amount_minor"])
        for x in (pred.get("journal_entry") or [])
        if isinstance(x, dict) and x.get("side") == "debit"
    )
    pred_credits = sum(
        int(x["amount_minor"])
        for x in (pred.get("journal_entry") or [])
        if isinstance(x, dict) and x.get("side") == "credit"
    )
    balanced = pred_debits == pred_credits and bool(pred.get("journal_entry"))
    components["debit_credit_balanced"] = float(balanced)

    gold_amounts = {int(x["amount_minor"]) for x in (gold.get("journal_entry") or [])}
    pred_amounts = {int(x["amount_minor"]) for x in (pred.get("journal_entry") or [])}
    components["amount_exact"] = float(gold_amounts == pred_amounts and bool(gold_amounts))

    components["policy_action_exact"] = float(
        gold.get("policy_action") == pred.get("policy_action")
    )

    gold_rules = set(gold.get("rule_ids") or [])
    pred_rules = set(pred.get("rule_ids") or [])
    _, _, rule_f1 = _set_prf(pred_rules, gold_rules)
    components["rule_id_set_f1"] = rule_f1
    components["rule_id_set_exact"] = float(gold_rules == pred_rules)

    gold_ev = _evidence_set(gold.get("evidence") or [])
    pred_ev = _evidence_set(pred.get("evidence") or [])
    ep, er, ef1 = _set_prf(pred_ev, gold_ev)
    components["evidence_precision"] = ep
    components["evidence_recall"] = er
    components["evidence_f1"] = ef1

    joint = (
        gl_exact == 1.0
        and components["journal_set_exact"] == 1.0
        and components["policy_action_exact"] == 1.0
        and components["rule_id_set_exact"] == 1.0
        and gold_ev == pred_ev
        and balanced
    )
    components["joint_exact"] = float(joint)
    invariant_violation = not balanced
    return components, joint, invariant_violation


def _score_variance(
    gold: dict[str, Any], pred: dict[str, Any]
) -> tuple[dict[str, float], bool, bool]:
    components: dict[str, float] = {}
    g_profit = int(gold["profit_impact_minor"])
    p_profit = int(pred.get("profit_impact_minor", 0)) if "profit_impact_minor" in pred else None
    profit_exact = p_profit is not None and p_profit == g_profit
    components["profit_impact_exact"] = float(profit_exact)
    components["profit_impact_mae"] = (
        float(abs(p_profit - g_profit)) if p_profit is not None else float("nan")
    )
    components["direction_exact"] = float(gold.get("direction") == pred.get("direction"))

    gold_drivers = list(gold.get("top_drivers") or [])
    pred_drivers = list(pred.get("top_drivers") or [])
    gold_ids = [d["driver_id"] for d in gold_drivers]
    pred_ids = [d["driver_id"] for d in pred_drivers if isinstance(d, dict)]
    k = len(gold_ids)
    if k == 0:
        components["driver_id_precision_at_k"] = 1.0 if not pred_ids else 0.0
    else:
        hit = len(set(pred_ids[:k]) & set(gold_ids))
        components["driver_id_precision_at_k"] = hit / k

    # Rank agreement: fraction of gold ranks whose driver_id matches pred at that rank.
    rank_hits = 0
    rank_total = 0
    pred_by_rank = {
        int(d["rank"]): d["driver_id"] for d in pred_drivers if isinstance(d, dict) and "rank" in d
    }
    for d in gold_drivers:
        rank_total += 1
        if pred_by_rank.get(int(d["rank"])) == d["driver_id"]:
            rank_hits += 1
    components["driver_rank_agreement"] = rank_hits / rank_total if rank_total else 1.0

    gold_impacts = {d["driver_id"]: int(d["impact_minor"]) for d in gold_drivers}
    pred_impacts = {
        d["driver_id"]: int(d["impact_minor"])
        for d in pred_drivers
        if isinstance(d, dict) and "driver_id" in d and "impact_minor" in d
    }
    shared = set(gold_impacts) & set(pred_impacts)
    if shared:
        components["driver_impact_mae"] = sum(
            abs(gold_impacts[i] - pred_impacts[i]) for i in shared
        ) / len(shared)
    else:
        components["driver_impact_mae"] = float("nan") if gold_impacts else 0.0

    try:
        driver_sum = sum(int(d["impact_minor"]) for d in pred_drivers if isinstance(d, dict))
        other = int(pred.get("other_impact_minor", 0))
        p_closed = (
            p_profit is not None and driver_sum + other == p_profit
        )
    except (TypeError, ValueError, KeyError):
        p_closed = False
    components["arithmetic_closure"] = float(p_closed)

    expected_dir = "favorable" if g_profit >= 0 else "unfavorable"
    direction_ok = pred.get("direction") == expected_dir if p_profit is None else (
        pred.get("direction") == ("favorable" if p_profit >= 0 else "unfavorable")
    )
    # Invariant: predicted direction must match predicted profit sign when present.
    invariant_violation = False
    if p_profit is not None:
        want = "favorable" if p_profit >= 0 else "unfavorable"
        if pred.get("direction") != want:
            invariant_violation = True
    if not p_closed:
        invariant_violation = True

    gold_rules = set(gold.get("rule_ids") or [])
    pred_rules = set(pred.get("rule_ids") or [])
    components["rule_ids_exact"] = float(gold_rules == pred_rules)
    gold_ev = set(gold.get("evidence_ids") or [])
    pred_ev = set(pred.get("evidence_ids") or [])
    components["evidence_ids_exact"] = float(gold_ev == pred_ev)

    joint = (
        profit_exact
        and components["direction_exact"] == 1.0
        and gold_ids == pred_ids
        and all(
            isinstance(d, dict)
            and int(d.get("impact_minor", 0)) == int(gold_drivers[i]["impact_minor"])
            and int(d.get("rank", -1)) == int(gold_drivers[i]["rank"])
            for i, d in enumerate(pred_drivers)
            if i < len(gold_drivers)
        )
        and len(pred_drivers) == len(gold_drivers)
        and int(pred.get("other_impact_minor", 0)) == int(gold.get("other_impact_minor", 0))
        and gold_rules == pred_rules
        and gold_ev == pred_ev
        and p_closed
        and direction_ok
    )
    components["joint_exact"] = float(joint)
    _ = expected_dir  # documented for readers; joint uses gold direction equality
    return components, joint, invariant_violation


def _match_group_key(group: dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    return (
        frozenset(str(x) for x in (group.get("book_ids") or [])),
        frozenset(str(x) for x in (group.get("bank_ids") or [])),
    )


def _matched_edges(groups: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(book_id), str(bank_id))
        for group in groups
        if isinstance(group, dict)
        for book_id in (group.get("book_ids") or [])
        for bank_id in (group.get("bank_ids") or [])
    }


def _score_cash(
    gold: dict[str, Any], pred: dict[str, Any]
) -> tuple[dict[str, float], bool, bool]:
    components: dict[str, float] = {}
    gold_groups = {_match_group_key(g) for g in (gold.get("matched_groups") or [])}
    pred_groups = {
        _match_group_key(g) for g in (pred.get("matched_groups") or []) if isinstance(g, dict)
    }
    _, _, group_f1 = _set_prf(pred_groups, gold_groups)
    components["matched_group_f1"] = group_f1
    components["matched_group_exact"] = float(gold_groups == pred_groups)
    gold_edges = _matched_edges(list(gold.get("matched_groups") or []))
    pred_edges = _matched_edges(list(pred.get("matched_groups") or []))
    edge_precision, edge_recall, edge_f1 = _set_prf(pred_edges, gold_edges)
    components["matched_edge_precision"] = edge_precision
    components["matched_edge_recall"] = edge_recall
    components["matched_edge_f1"] = edge_f1
    components["matched_edge_exact"] = float(gold_edges == pred_edges)

    gold_exc_types = [e["type"] for e in (gold.get("exceptions") or [])]
    pred_exc_types = [
        e["type"] for e in (pred.get("exceptions") or []) if isinstance(e, dict) and "type" in e
    ]
    # Per-example multiclass contribution via exact multiset equality proxy for aggregation.
    components["exception_type_exact"] = float(
        sorted(gold_exc_types) == sorted(pred_exc_types)
    )

    components["adjusted_book_exact"] = float(
        gold.get("adjusted_book_balance_minor") == pred.get("adjusted_book_balance_minor")
    )
    components["adjusted_bank_exact"] = float(
        gold.get("adjusted_bank_balance_minor") == pred.get("adjusted_bank_balance_minor")
    )
    components["difference_exact"] = float(
        gold.get("difference_minor") == pred.get("difference_minor")
    )

    try:
        book = int(pred["adjusted_book_balance_minor"])
        bank = int(pred["adjusted_bank_balance_minor"])
        diff = int(pred["difference_minor"])
        invariant_violation = diff != (book - bank)
    except (KeyError, TypeError, ValueError):
        invariant_violation = True

    joint = (
        components["matched_group_exact"] == 1.0
        and components["exception_type_exact"] == 1.0
        and components["adjusted_book_exact"] == 1.0
        and components["adjusted_bank_exact"] == 1.0
        and components["difference_exact"] == 1.0
        and gold.get("status") == pred.get("status")
        and not invariant_violation
    )
    components["joint_exact"] = float(joint)
    return components, joint, invariant_violation


def score_prediction(record: PredictionRecord) -> ExampleScore:
    parsed, status, schema_valid = parse_prediction(record)
    refused_or_empty = status in {"refusal", "empty"}
    json_parse_ok = status == "ok"
    components: dict[str, float] = {}
    joint = False
    invariant_violation = False
    confidence: float | None = None

    if parsed is not None and schema_valid:
        conf = parsed.get("confidence")
        confidence = float(conf) if isinstance(conf, (int, float)) else None
        if record.task == TaskId.TRANSACTION_REVIEW.value:
            components, joint, invariant_violation = _score_transaction(
                record.expected_output, parsed
            )
        elif record.task == TaskId.VARIANCE_ANALYSIS.value:
            components, joint, invariant_violation = _score_variance(
                record.expected_output, parsed
            )
        elif record.task == TaskId.CASH_RECONCILIATION.value:
            components, joint, invariant_violation = _score_cash(record.expected_output, parsed)
        elif record.task == TaskId.MERCHANT_TAGGING.value:
            components, joint, invariant_violation = _score_merchant(
                record.expected_output, parsed
            )
        else:
            joint = parsed == record.expected_output
            components["joint_exact"] = float(joint)
    elif parsed is not None:
        # Parsed JSON but schema-invalid: still expose confidence if present.
        conf = parsed.get("confidence")
        confidence = float(conf) if isinstance(conf, (int, float)) else None
        components["joint_exact"] = 0.0

    return ExampleScore(
        example_id=record.example_id,
        world_id=record.world_id,
        seed=record.seed,
        task=record.task,
        split=record.split,
        difficulty=record.difficulty,
        template_family=record.template_family,
        slices=dict(record.slices),
        json_parse_ok=json_parse_ok,
        json_schema_valid=schema_valid,
        refused_or_empty=refused_or_empty,
        joint_exact=joint,
        confidence=confidence,
        components=components,
        invariant_violation=invariant_violation,
    )


def compute_primary_index(
    transaction_joint_exact: float | None,
    variance_joint_exact: float | None,
    json_schema_validity: float,
    weights: PrimaryIndexWeights = PRIMARY_INDEX_WEIGHTS,
) -> float:
    """Prespecified finance-proof.v1 primary index. Missing tasks contribute 0."""
    txn = 0.0 if transaction_joint_exact is None else transaction_joint_exact
    var = 0.0 if variance_joint_exact is None else variance_joint_exact
    return (
        weights.transaction_joint_exact * txn
        + weights.variance_joint_exact * var
        + weights.json_schema_validity * json_schema_validity
    )


def compute_primary_index_v2(
    transaction_joint_exact: float | None,
    variance_joint_exact: float | None,
    merchant_joint_exact: float | None,
    json_schema_validity: float,
    weights: PrimaryIndexWeightsV2 = PRIMARY_INDEX_WEIGHTS_V2,
) -> float:
    """Prespecified finance-proof.v2 primary index (A/B/C + schema)."""
    txn = 0.0 if transaction_joint_exact is None else transaction_joint_exact
    var = 0.0 if variance_joint_exact is None else variance_joint_exact
    merch = 0.0 if merchant_joint_exact is None else merchant_joint_exact
    return (
        weights.transaction_joint_exact * txn
        + weights.variance_joint_exact * var
        + weights.merchant_joint_exact * merch
        + weights.json_schema_validity * json_schema_validity
    )


def _score_merchant(
    gold: dict[str, Any], pred: dict[str, Any]
) -> tuple[dict[str, float], bool, bool]:
    components: dict[str, float] = {}
    merchant_exact = float(
        gold.get("merchant_id") == pred.get("merchant_id")
        and gold.get("merchant_name") == pred.get("merchant_name")
    )
    components["merchant_exact"] = merchant_exact
    components["merchant_id_exact"] = float(gold.get("merchant_id") == pred.get("merchant_id"))
    components["merchant_name_exact"] = float(
        gold.get("merchant_name") == pred.get("merchant_name")
    )
    components["category_exact"] = float(
        gold.get("spend_category") == pred.get("spend_category")
    )
    gold_tags = set(gold.get("tags") or [])
    pred_tags = (
        set(pred.get("tags") or [])
        if isinstance(pred.get("tags"), (list, tuple))
        else set()
    )
    _, _, tag_f1 = _set_prf(pred_tags, gold_tags)
    components["tag_set_f1"] = tag_f1
    components["tag_set_exact"] = float(gold_tags == pred_tags)
    joint = (
        merchant_exact == 1.0
        and components["category_exact"] == 1.0
        and components["tag_set_exact"] == 1.0
    )
    components["joint_exact"] = float(joint)
    # Invariant: predicted tags must be sorted unique when present.
    pred_tag_list = pred.get("tags")
    invariant_violation = False
    if isinstance(pred_tag_list, list):
        if pred_tag_list != sorted(pred_tag_list) or len(pred_tag_list) != len(set(pred_tag_list)):
            invariant_violation = True
    return components, joint, invariant_violation


def _adaptive_ece(confidences: list[float], correct: list[float], n_bins: int = 10) -> float | None:
    n = len(confidences)
    if n == 0:
        return None
    # Equal-count quantile bins with boundaries moved right so equal-confidence
    # ties always remain in one bin.
    order = sorted(range(n), key=lambda i: (confidences[i], i))
    bins = max(1, min(n_bins, n))
    cuts = [0]
    for b in range(1, bins):
        cut = (b * n) // bins
        while (
            cut < n
            and cut > 0
            and confidences[order[cut - 1]] == confidences[order[cut]]
        ):
            cut += 1
        if cut < n and cut > cuts[-1]:
            cuts.append(cut)
    cuts.append(n)

    ece = 0.0
    for start, end in zip(cuts, cuts[1:], strict=False):
        idxs = order[start:end]
        bin_conf = sum(confidences[i] for i in idxs) / len(idxs)
        bin_acc = sum(correct[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(bin_acc - bin_conf)
    return ece


def _risk_coverage(confidences: list[float], correct: list[float]) -> list[dict[str, float]]:
    if not confidences:
        return []
    thresholds = sorted(set(confidences), reverse=True)
    out: list[dict[str, float]] = []
    n = len(confidences)
    for t in thresholds:
        keep = [c for conf, c in zip(confidences, correct, strict=True) if conf >= t]
        coverage = len(keep) / n
        risk = 1.0 - (sum(keep) / len(keep)) if keep else 1.0
        out.append({"threshold": t, "coverage": coverage, "risk": risk})
    return out


def compute_calibration(scores: list[ExampleScore]) -> CalibrationMetrics:
    pairs = [
        (s.confidence, 1.0 if s.joint_exact else 0.0)
        for s in scores
        if s.confidence is not None and s.json_schema_valid
    ]
    if not pairs:
        return CalibrationMetrics(brier_score=None, adaptive_ece=None, risk_coverage=[], n=0)
    confidences = [p[0] for p in pairs]
    correct = [p[1] for p in pairs]
    brier = sum((c - y) ** 2 for c, y in zip(confidences, correct, strict=True)) / len(pairs)
    return CalibrationMetrics(
        brier_score=brier,
        adaptive_ece=_adaptive_ece(confidences, correct),
        risk_coverage=_risk_coverage(confidences, correct),
        n=len(pairs),
    )


def _aggregate_components(scores: list[ExampleScore]) -> dict[str, float | None]:
    keys: set[str] = set()
    for s in scores:
        keys.update(s.components)
    out: dict[str, float | None] = {}
    for key in sorted(keys):
        vals = [
            s.components[key]
            for s in scores
            if key in s.components and not math.isnan(s.components[key])
        ]
        out[key] = _mean(vals)
    return out


def _slice_reports(scores: list[ExampleScore], min_n: int = 30) -> list[SliceReport]:
    buckets: dict[tuple[str, str], list[ExampleScore]] = defaultdict(list)
    for s in scores:
        buckets[("difficulty", s.difficulty)].append(s)
        buckets[("split", s.split)].append(s)
        if s.template_family:
            buckets[("template_family", s.template_family)].append(s)
        for k, v in s.slices.items():
            buckets[(k, v)].append(s)
    reports: list[SliceReport] = []
    for (key, value), group in sorted(buckets.items()):
        joints = [float(s.joint_exact) for s in group]
        schemas = [float(s.json_schema_valid) for s in group]
        reports.append(
            SliceReport(
                slice_key=key,
                slice_value=value,
                n=len(group),
                joint_exact=_mean(joints),
                json_schema_validity=_mean(schemas),
                underpowered=len(group) < min_n,
            )
        )
    return reports


def compute_arm_metrics(arm_id: str, records: list[PredictionRecord]) -> ArmMetrics:
    seen: set[tuple[int, str]] = set()
    for record in records:
        key = (record.seed, record.example_id)
        if key in seen:
            raise ValueError(
                "duplicate prediction record for "
                f"seed={record.seed}, example_id={record.example_id}"
            )
        seen.add(key)
    scores = [score_prediction(r) for r in records]
    n = len(scores)
    if n == 0:
        return ArmMetrics(
            arm_id=arm_id,
            n=0,
            json_parse_rate=0.0,
            json_schema_validity=0.0,
            refusal_empty_rate=0.0,
            transaction_joint_exact=None,
            variance_joint_exact=None,
            cash_joint_exact=None,
            merchant_joint_exact=None,
            primary_index=0.0,
            primary_index_v2=None,
            components={},
            task_metrics={},
            calibration=CalibrationMetrics(None, None, [], 0),
            slices=[],
            seeds=(),
            seed_metrics={},
            example_scores=[],
            critical_invariant_violations=0,
        )

    json_parse_rate = sum(1 for s in scores if s.json_parse_ok) / n
    json_schema_validity = sum(1 for s in scores if s.json_schema_valid) / n
    refusal_empty_rate = sum(1 for s in scores if s.refused_or_empty) / n

    txn = [s for s in scores if s.task == TaskId.TRANSACTION_REVIEW.value]
    var = [s for s in scores if s.task == TaskId.VARIANCE_ANALYSIS.value]
    cash = [s for s in scores if s.task == TaskId.CASH_RECONCILIATION.value]
    merchant = [s for s in scores if s.task == TaskId.MERCHANT_TAGGING.value]

    txn_joint = _mean([float(s.joint_exact) for s in txn])
    var_joint = _mean([float(s.joint_exact) for s in var])
    cash_joint = _mean([float(s.joint_exact) for s in cash])
    merchant_joint = _mean([float(s.joint_exact) for s in merchant])

    # Macro-F1 for policy actions across transaction examples with schema-valid preds.
    policy_true: list[str] = []
    policy_pred: list[str] = []
    for rec, score in zip(records, scores, strict=True):
        if rec.task != TaskId.TRANSACTION_REVIEW.value or not score.json_schema_valid:
            continue
        parsed, _, _ = parse_prediction(rec)
        if parsed is None:
            continue
        policy_true.append(str(rec.expected_output.get("policy_action")))
        policy_pred.append(str(parsed.get("policy_action")))
    policy_macro = _macro_f1(policy_true, policy_pred)

    cash_exc_true: list[set[str]] = []
    cash_exc_pred: list[set[str]] = []
    for rec, _score in zip(records, scores, strict=True):
        if rec.task != TaskId.CASH_RECONCILIATION.value:
            continue
        parsed, _, _ = parse_prediction(rec)
        gtypes = {
            str(e["type"])
            for e in (rec.expected_output.get("exceptions") or [])
        }
        ptypes = {
            str(e["type"])
            for e in ((parsed or {}).get("exceptions") or [])
            if isinstance(e, dict) and "type" in e
        }
        cash_exc_true.append(gtypes)
        cash_exc_pred.append(ptypes)
    exception_macro = _multilabel_macro_f1(cash_exc_true, cash_exc_pred)

    category_true: list[str] = []
    category_pred: list[str] = []
    tag_true: list[set[str]] = []
    tag_pred: list[set[str]] = []
    for rec, score in zip(records, scores, strict=True):
        if rec.task != TaskId.MERCHANT_TAGGING.value or not score.json_schema_valid:
            continue
        parsed, _, _ = parse_prediction(rec)
        if parsed is None:
            continue
        category_true.append(str(rec.expected_output.get("spend_category")))
        category_pred.append(str(parsed.get("spend_category")))
        tag_true.append({str(tag) for tag in (rec.expected_output.get("tags") or [])})
        raw_tags = parsed.get("tags") or []
        tag_pred.append(
            {str(tag) for tag in raw_tags}
            if isinstance(raw_tags, (list, tuple))
            else set()
        )
    category_macro = _macro_f1(category_true, category_pred)
    tag_macro = _multilabel_macro_f1(tag_true, tag_pred)

    task_metrics: dict[str, dict[str, float | None]] = {
        TaskId.TRANSACTION_REVIEW.value: {
            **_aggregate_components(txn),
            "policy_action_macro_f1": policy_macro,
            "n": float(len(txn)),
        },
        TaskId.VARIANCE_ANALYSIS.value: {
            **_aggregate_components(var),
            "n": float(len(var)),
        },
        TaskId.CASH_RECONCILIATION.value: {
            **_aggregate_components(cash),
            "exception_type_macro_f1": exception_macro,
            "n": float(len(cash)),
        },
        TaskId.MERCHANT_TAGGING.value: {
            **_aggregate_components(merchant),
            "category_macro_f1": category_macro,
            "tag_macro_f1": tag_macro,
            "n": float(len(merchant)),
        },
    }

    primary = compute_primary_index(txn_joint, var_joint, json_schema_validity)
    primary_v2 = compute_primary_index_v2(
        txn_joint, var_joint, merchant_joint, json_schema_validity
    )
    components = {
        "transaction_joint_exact": txn_joint if txn_joint is not None else float("nan"),
        "variance_joint_exact": var_joint if var_joint is not None else float("nan"),
        "cash_joint_exact": cash_joint if cash_joint is not None else float("nan"),
        "merchant_joint_exact": (
            merchant_joint if merchant_joint is not None else float("nan")
        ),
        "json_schema_validity": json_schema_validity,
        "json_parse_rate": json_parse_rate,
        "refusal_empty_rate": refusal_empty_rate,
        "primary_index": primary,
        "primary_index_v2": primary_v2,
    }
    seed_metrics: dict[int, dict[str, float | None]] = {}
    for seed in sorted({score.seed for score in scores}):
        seed_scores = [score for score in scores if score.seed == seed]
        seed_txn = [
            score
            for score in seed_scores
            if score.task == TaskId.TRANSACTION_REVIEW.value
        ]
        seed_var = [
            score
            for score in seed_scores
            if score.task == TaskId.VARIANCE_ANALYSIS.value
        ]
        seed_schema = sum(
            1.0 if score.json_schema_valid else 0.0
            for score in seed_scores
        ) / len(seed_scores)
        seed_txn_joint = _mean(
            [float(score.joint_exact) for score in seed_txn]
        )
        seed_var_joint = _mean(
            [float(score.joint_exact) for score in seed_var]
        )
        seed_ood = iid_ood_primary(seed_scores).get("ood_primary_index")
        seed_metrics[seed] = {
            "transaction_joint_exact": seed_txn_joint,
            "variance_joint_exact": seed_var_joint,
            "json_schema_validity": seed_schema,
            "primary_index": compute_primary_index(
                seed_txn_joint,
                seed_var_joint,
                seed_schema,
            ),
            "ood_primary_index": seed_ood,
        }

    return ArmMetrics(
        arm_id=arm_id,
        n=n,
        json_parse_rate=json_parse_rate,
        json_schema_validity=json_schema_validity,
        refusal_empty_rate=refusal_empty_rate,
        transaction_joint_exact=txn_joint,
        variance_joint_exact=var_joint,
        cash_joint_exact=cash_joint,
        merchant_joint_exact=merchant_joint,
        primary_index=primary,
        primary_index_v2=primary_v2,
        components=components,
        task_metrics=task_metrics,
        calibration=compute_calibration(scores),
        slices=_slice_reports(scores),
        seeds=tuple(sorted(seed_metrics)),
        seed_metrics=seed_metrics,
        example_scores=scores,
        critical_invariant_violations=sum(1 for s in scores if s.invariant_violation),
    )


def iid_ood_primary(
    scores: list[ExampleScore],
) -> dict[str, float | None]:
    """Primary-index components split by IID vs OOD."""
    out: dict[str, float | None] = {}
    for split_name, key in (("iid_test", "iid"), ("ood_test", "ood"), ("test", "test")):
        group = [s for s in scores if s.split == split_name]
        if not group:
            out[f"{key}_primary_index"] = None
            continue
        txn = [s for s in group if s.task == TaskId.TRANSACTION_REVIEW.value]
        var = [s for s in group if s.task == TaskId.VARIANCE_ANALYSIS.value]
        schema = sum(1 for s in group if s.json_schema_valid) / len(group)
        out[f"{key}_primary_index"] = compute_primary_index(
            _mean([float(s.joint_exact) for s in txn]),
            _mean([float(s.joint_exact) for s in var]),
            schema,
        )
        out[f"{key}_json_schema_validity"] = schema
        out[f"{key}_transaction_joint_exact"] = _mean([float(s.joint_exact) for s in txn])
        out[f"{key}_variance_joint_exact"] = _mean([float(s.joint_exact) for s in var])
    return out
