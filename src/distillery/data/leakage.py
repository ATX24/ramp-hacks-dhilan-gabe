"""Semantic duplicate and cross-group/split isolation checks."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import FinanceTaskEnvelope, SplitName, TaskId
from distillery.data.validate import find_input_hygiene_errors

DEFAULT_NUM_PERM = 64
DEFAULT_NGRAM = 1
DEFAULT_JACCARD_THRESHOLD = 0.85

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_OPAQUE_ID_RE = re.compile(
    r"\b(?:world|grp|ent|txn|vnd|src|bok|bnk|ex)_[0-9a-f]{8,}\b",
    re.IGNORECASE,
)
_RULE_ID_RE = re.compile(r"\b(?:POL|VAR)-[A-Z0-9-]+-[A-F0-9]{8}\b")
_IDENTITY_KEYS = {
    "entity_id",
    "event_ids",
    "id",
    "rule_id",
    "source_id",
    "txn_id",
    "vendor_id",
    "vendor_ids",
    "world_id",
    "group_id",
}
_NON_SEMANTIC_KEYS = {
    "case_nonce",
    "prompt",
    "template_family",
}


@dataclass(frozen=True)
class LeakFinding:
    kind: str
    left_id: str
    right_id: str
    detail: str


@dataclass
class LeakageReport:
    ok: bool
    findings: list[LeakFinding] = field(default_factory=list)
    exact_duplicate_groups: int = 0
    near_duplicate_pairs: int = 0
    cross_split_id_leaks: int = 0
    cross_split_template_leaks: int = 0
    normalized_hashes_checked: int = 0

    @property
    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "cross_split_id_leaks": self.cross_split_id_leaks,
            "cross_split_template_leaks": self.cross_split_template_leaks,
            "normalized_hashes_checked": self.normalized_hashes_checked,
            "by_kind": self.by_kind,
            "findings": [
                {
                    "kind": finding.kind,
                    "left_id": finding.left_id,
                    "right_id": finding.right_id,
                    "detail": finding.detail,
                }
                for finding in self.findings
            ],
        }


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", stripped).strip()


def normalized_content_hash(
    example: FinanceTaskEnvelope | dict[str, Any],
) -> str:
    """Hash semantic input while discarding all identity/provenance/nonce fields."""
    payload = _payload(example)
    material = {
        "task": payload.get("task"),
        "difficulty": payload.get("difficulty"),
        "input": _semantic_normalize(payload.get("input") or {}),
    }
    return content_sha256(material)


def example_fingerprint(
    example: FinanceTaskEnvelope | dict[str, Any],
) -> str:
    """Compact semantic tokens; IDs/provenance never improve similarity."""
    payload = _payload(example)
    finance_input = payload.get("input") or {}
    task = str(payload.get("task"))
    tokens = [f"task_{task}", f"difficulty_{payload.get('difficulty')}"]
    if task == TaskId.TRANSACTION_REVIEW.value:
        tokens.extend(
            (
                f"amount_{finance_input.get('amount_minor')}",
                f"category_{finance_input.get('expense_category')}",
                f"currency_{finance_input.get('currency')}",
                f"date_{finance_input.get('date')}",
            )
        )
        tokens.extend(
            f"descriptor_{word}"
            for word in normalize_text(str(finance_input.get("descriptor", ""))).split()
        )
        tokens.extend(
            f"vendor_{word}"
            for word in normalize_text(str(finance_input.get("vendor", ""))).split()
        )
        tokens.append(
            "chart_"
            + content_sha256(
                [
                    account.get("name")
                    for account in finance_input.get("chart_of_accounts", [])
                    if isinstance(account, Mapping)
                ]
            )[:16]
        )
        tokens.append(
            "policy_"
            + content_sha256(
                [
                    _scrub_string(str(rule.get("text", "")))
                    for rule in finance_input.get("policy_rules", [])
                    if isinstance(rule, Mapping)
                ]
            )[:16]
        )
    elif task == TaskId.VARIANCE_ANALYSIS.value:
        tokens.extend(
            (
                f"actual_{finance_input.get('actual_minor')}",
                f"budget_{finance_input.get('budget_minor')}",
                f"period_{finance_input.get('period')}",
            )
        )
        for item in finance_input.get("driver_observations", []) + finance_input.get(
            "unallocated_line_items", []
        ):
            if not isinstance(item, Mapping):
                continue
            tokens.extend(
                (
                    f"driver_{item.get('driver_id')}",
                    f"kind_{item.get('kind')}",
                    f"pnl_{item.get('pnl_type')}",
                    f"driver_budget_{item.get('budget_minor')}",
                    f"driver_actual_{item.get('actual_minor')}",
                )
            )
    else:
        tokens.extend(
            (
                f"bank_balance_{finance_input.get('bank_balance_minor')}",
                f"book_balance_{finance_input.get('book_balance_minor')}",
                f"period_{finance_input.get('close_period')}",
            )
        )
        for collection in ("book_entries", "bank_events"):
            for item in finance_input.get(collection, []):
                if not isinstance(item, Mapping):
                    continue
                tokens.extend(
                    (
                        f"{collection}_amount_{item.get('amount_minor')}",
                        f"{collection}_date_{item.get('date')}",
                        f"{collection}_memo_{normalize_text(str(item.get('memo', '')))}",
                    )
                )
    return " ".join(tokens)


def shingles(text: str, n: int = DEFAULT_NGRAM) -> set[str]:
    tokens = normalize_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def minhash_signature(
    text: str,
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram: int = DEFAULT_NGRAM,
) -> tuple[int, ...]:
    """Dependency-free MinHash with deterministic universal hash streams."""
    grams = shingles(text, ngram)
    if not grams:
        return tuple(0 for _ in range(num_perm))
    prime = 18_446_744_073_709_551_557
    base_hashes = [
        int.from_bytes(
            hashlib.blake2b(gram.encode(), digest_size=8).digest(),
            "big",
        )
        for gram in grams
    ]
    signature: list[int] = []
    for permutation in range(num_perm):
        a = 2 * permutation + 1
        b = int.from_bytes(
            hashlib.blake2b(
                f"perm:{permutation}".encode(),
                digest_size=8,
            ).digest(),
            "big",
        )
        signature.append(min((a * value + b) % prime for value in base_hashes))
    return tuple(signature)


def estimated_jaccard(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    matches = sum(
        1 for left_value, right_value in zip(left, right, strict=True) if left_value == right_value
    )
    return matches / len(left)


def check_leakage(
    examples: Iterable[FinanceTaskEnvelope | dict[str, Any]],
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    check_near_duplicates: bool = True,
) -> LeakageReport:
    """Run every seal-time duplicate, hygiene, and isolation detector."""
    items = [
        example
        if isinstance(example, FinanceTaskEnvelope)
        else FinanceTaskEnvelope.model_validate(example)
        for example in examples
    ]
    findings: list[LeakFinding] = []

    for example in items:
        for error in find_input_hygiene_errors(
            example.input,
            expected_output=example.expected_output,
        ):
            kind = error.split(":", maxsplit=1)[0]
            findings.append(
                LeakFinding(
                    kind=kind,
                    left_id=example.example_id,
                    right_id=example.example_id,
                    detail=error,
                )
            )

    exact_groups = _find_exact_duplicates(items, findings)
    _find_identity_reuse(items, findings)
    _find_semantic_overlaps(items, findings)
    _find_template_regime_overlap(items, findings)

    near_pairs = 0
    if check_near_duplicates and len(items) > 1:
        near_pairs, near_findings = _minhash_near_duplicates(
            items,
            jaccard_threshold=jaccard_threshold,
            num_perm=num_perm,
        )
        findings.extend(near_findings)

    cross_id = sum(
        1
        for finding in findings
        if finding.kind
        in {
            "world_id_reuse",
            "group_id_overlap",
            "entity_id_overlap",
            "source_id_overlap",
            "vendor_id_overlap",
        }
        and "cross_split=true" in finding.detail
    )
    cross_template = sum(
        1
        for finding in findings
        if finding.kind in {"template_family_overlap", "template_regime_overlap"}
    )
    return LeakageReport(
        ok=not findings,
        findings=findings,
        exact_duplicate_groups=exact_groups,
        near_duplicate_pairs=near_pairs,
        cross_split_id_leaks=cross_id,
        cross_split_template_leaks=cross_template,
        normalized_hashes_checked=len(items),
    )


def _find_exact_duplicates(
    items: list[FinanceTaskEnvelope],
    findings: list[LeakFinding],
) -> int:
    by_hash: dict[str, list[FinanceTaskEnvelope]] = defaultdict(list)
    for example in items:
        by_hash[normalized_content_hash(example)].append(example)
    duplicate_groups = 0
    for digest, examples in by_hash.items():
        if len(examples) < 2:
            continue
        duplicate_groups += 1
        first = examples[0]
        for duplicate in examples[1:]:
            findings.append(
                LeakFinding(
                    "exact_normalized_duplicate",
                    first.example_id,
                    duplicate.example_id,
                    f"semantic_hash={digest}",
                )
            )
    return duplicate_groups


def _find_identity_reuse(
    items: list[FinanceTaskEnvelope],
    findings: list[LeakFinding],
) -> None:
    _detect_overlap(
        items,
        findings,
        kind="world_id_reuse",
        values=lambda example: [example.world_id],
        allow_same_group=False,
    )
    _detect_overlap(
        items,
        findings,
        kind="group_id_overlap",
        values=lambda example: [example.group_id],
        allow_same_group=True,
    )
    _detect_overlap(
        items,
        findings,
        kind="entity_id_overlap",
        values=lambda example: [str(example.input.get("entity_id", ""))],
        allow_same_group=True,
    )
    _detect_overlap(
        items,
        findings,
        kind="vendor_id_overlap",
        values=lambda example: [str(example.input.get("vendor_id", ""))],
        allow_same_group=True,
    )
    _detect_overlap(
        items,
        findings,
        kind="source_id_overlap",
        values=_source_ids,
        allow_same_group=True,
    )


def _find_semantic_overlaps(
    items: list[FinanceTaskEnvelope],
    findings: list[LeakFinding],
) -> None:
    detectors = (
        (
            "vendor_name_overlap",
            lambda example: [normalize_text(str(example.input.get("vendor", "")))],
        ),
        ("merchant_name_overlap", _merchant_names),
        ("descriptor_family_overlap", _descriptor_families),
        ("period_overlap", _periods),
        ("policy_text_overlap", _policy_texts),
        ("coa_description_overlap", _coa_descriptions),
        ("numeric_case_overlap", lambda example: [_numeric_case_signature(example)]),
        (
            "template_family_overlap",
            lambda example: [example.provenance.template_family],
        ),
    )
    for kind, getter in detectors:
        _detect_overlap(
            items,
            findings,
            kind=kind,
            values=getter,
            allow_same_group=True,
        )


def _find_template_regime_overlap(
    items: list[FinanceTaskEnvelope],
    findings: list[LeakFinding],
) -> None:
    owners: dict[str, list[FinanceTaskEnvelope]] = defaultdict(list)
    for example in items:
        base = example.provenance.template_family.split("__", maxsplit=1)[0]
        owners[base].append(example)
    for base, examples in owners.items():
        has_ood = any(example.provenance.split == SplitName.OOD_TEST for example in examples)
        has_non_ood = any(example.provenance.split != SplitName.OOD_TEST for example in examples)
        if has_ood and has_non_ood:
            findings.append(
                LeakFinding(
                    "template_regime_overlap",
                    examples[0].example_id,
                    examples[-1].example_id,
                    f"base_family={base}",
                )
            )


def _detect_overlap(
    items: list[FinanceTaskEnvelope],
    findings: list[LeakFinding],
    *,
    kind: str,
    values,
    allow_same_group: bool,
) -> None:
    owners: dict[str, list[FinanceTaskEnvelope]] = defaultdict(list)
    for example in items:
        for raw_value in values(example):
            value = str(raw_value).strip()
            if value:
                owners[value].append(example)
    for value, examples in owners.items():
        first = examples[0]
        for other in examples[1:]:
            same_group = first.group_id == other.group_id
            if allow_same_group and same_group:
                continue
            cross_split = first.provenance.split != other.provenance.split
            findings.append(
                LeakFinding(
                    kind,
                    first.example_id,
                    other.example_id,
                    (f"value={value[:96]!r} cross_split={str(cross_split).casefold()}"),
                )
            )


def _minhash_near_duplicates(
    items: list[FinanceTaskEnvelope],
    *,
    jaccard_threshold: float,
    num_perm: int,
    band_size: int = 4,
) -> tuple[int, list[LeakFinding]]:
    findings: list[LeakFinding] = []
    near_pairs = 0
    by_task: dict[TaskId, list[FinanceTaskEnvelope]] = defaultdict(list)
    for example in items:
        by_task[example.task].append(example)
    for task_items in by_task.values():
        records = [
            (
                example,
                minhash_signature(
                    example_fingerprint(example),
                    num_perm=num_perm,
                ),
            )
            for example in task_items
        ]
        bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for index, (_example, signature) in enumerate(records):
            for start in range(0, num_perm, band_size):
                bands[(start, signature[start : start + band_size])].append(index)
        candidates: set[tuple[int, int]] = set()
        for indexes in bands.values():
            for left_index, left in enumerate(indexes):
                for right in indexes[left_index + 1 :]:
                    candidates.add((min(left, right), max(left, right)))
        for left_index, right_index in candidates:
            left, left_signature = records[left_index]
            right, right_signature = records[right_index]
            score = estimated_jaccard(left_signature, right_signature)
            if score < jaccard_threshold:
                continue
            near_pairs += 1
            cross = left.provenance.split != right.provenance.split
            kind = "cross_split_near_duplicate" if cross else "same_split_near_duplicate"
            findings.append(
                LeakFinding(
                    kind,
                    left.example_id,
                    right.example_id,
                    f"estimated_jaccard={score:.3f}",
                )
            )
    return near_pairs, findings


def _semantic_normalize(value: Any, key: str = "") -> Any:
    if key in _NON_SEMANTIC_KEYS:
        return None
    if key in _IDENTITY_KEYS:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return ["<id>" for _ in value]
        return "<id>"
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, Mapping):
        return {
            child_key: _semantic_normalize(value[child_key], child_key)
            for child_key in sorted(value)
            if child_key not in _NON_SEMANTIC_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_semantic_normalize(item, key) for item in value]
    return value


def _scrub_string(value: str) -> str:
    without_ids = _OPAQUE_ID_RE.sub("<id>", value)
    without_rules = _RULE_ID_RE.sub("<rule>", without_ids)
    return normalize_text(without_rules)


def _payload(
    example: FinanceTaskEnvelope | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(example, FinanceTaskEnvelope):
        return example.model_dump(mode="json")
    return example


def _source_ids(example: FinanceTaskEnvelope) -> list[str]:
    finance_input = example.input
    values: list[str] = []
    if finance_input.get("txn_id"):
        values.append(str(finance_input["txn_id"]))
    for item in finance_input.get("driver_observations", []):
        if isinstance(item, Mapping) and item.get("source_id"):
            values.append(str(item["source_id"]))
    for item in finance_input.get("unallocated_line_items", []):
        if isinstance(item, Mapping) and item.get("source_id"):
            values.append(str(item["source_id"]))
    for collection in ("book_entries", "bank_events"):
        for item in finance_input.get(collection, []):
            if isinstance(item, Mapping) and item.get("id"):
                values.append(str(item["id"]))
    return values


def _merchant_names(example: FinanceTaskEnvelope) -> list[str]:
    merchant = example.input.get("merchant")
    return [normalize_text(str(merchant))] if merchant else []


def _descriptor_families(example: FinanceTaskEnvelope) -> list[str]:
    descriptor = example.input.get("descriptor")
    vendor = example.input.get("vendor")
    if not descriptor:
        return []
    descriptor_tokens = normalize_text(str(descriptor)).split()
    vendor_tokens = set(normalize_text(str(vendor or "")).split())
    family = " ".join(
        token for token in descriptor_tokens if token not in vendor_tokens and not token.isdigit()
    )
    return [family] if family else []


def _periods(example: FinanceTaskEnvelope) -> list[str]:
    values = []
    for key in ("period", "close_period"):
        if example.input.get(key):
            values.append(str(example.input[key]))
    return values


def _policy_texts(example: FinanceTaskEnvelope) -> list[str]:
    return [
        normalize_text(str(rule["text"]))
        for rule in example.input.get("policy_rules", [])
        if isinstance(rule, Mapping) and rule.get("text")
    ]


def _coa_descriptions(example: FinanceTaskEnvelope) -> list[str]:
    return [
        normalize_text(str(account["name"]))
        for account in example.input.get("chart_of_accounts", [])
        if isinstance(account, Mapping) and account.get("name")
    ]


def _numeric_case_signature(example: FinanceTaskEnvelope) -> str:
    finance_input = example.input
    if example.task == TaskId.TRANSACTION_REVIEW:
        material = {
            "amount": finance_input.get("amount_minor"),
            "category": finance_input.get("expense_category"),
            "cost_center": finance_input.get("cost_center"),
            "date": finance_input.get("date"),
            "vendor": finance_input.get("vendor"),
        }
    elif example.task == TaskId.VARIANCE_ANALYSIS:
        material = {
            "actual": finance_input.get("actual_minor"),
            "budget": finance_input.get("budget_minor"),
            "period": finance_input.get("period"),
            "drivers": [
                {
                    "actual": item.get("actual_minor"),
                    "budget": item.get("budget_minor"),
                    "driver": item.get("driver_id"),
                    "pnl_type": item.get("pnl_type"),
                }
                for item in finance_input.get("driver_observations", [])
                if isinstance(item, Mapping)
            ],
        }
    else:
        material = {
            "bank": finance_input.get("bank_balance_minor"),
            "bank_events": [
                (item.get("amount_minor"), item.get("date"))
                for item in finance_input.get("bank_events", [])
                if isinstance(item, Mapping)
            ],
            "book": finance_input.get("book_balance_minor"),
            "period": finance_input.get("close_period"),
            "book_entries": [
                (item.get("amount_minor"), item.get("date"))
                for item in finance_input.get("book_entries", [])
                if isinstance(item, Mapping)
            ],
        }
    return content_sha256(material)
