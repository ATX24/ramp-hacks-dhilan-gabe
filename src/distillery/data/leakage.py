"""Normalized exact-hash and practical MinHash near-duplicate / leakage checks."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import FinanceTaskEnvelope, SplitName, TaskId

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Practical defaults: fast enough for 5k examples, sensitive to near-copies.
DEFAULT_NUM_PERM = 64
DEFAULT_NGRAM = 3
DEFAULT_JACCARD_THRESHOLD = 0.85


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "cross_split_id_leaks": self.cross_split_id_leaks,
            "cross_split_template_leaks": self.cross_split_template_leaks,
            "normalized_hashes_checked": self.normalized_hashes_checked,
            "findings": [
                {
                    "kind": f.kind,
                    "left_id": f.left_id,
                    "right_id": f.right_id,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", stripped).strip()


def normalized_content_hash(example: FinanceTaskEnvelope | dict[str, Any]) -> str:
    """Hash of task-relevant normalized fields (ignores ids that differ by construction)."""
    if isinstance(example, FinanceTaskEnvelope):
        payload = example.model_dump(mode="json")
    else:
        payload = example
    input_obj = payload.get("input") or {}
    # Drop ephemeral ids that are unique by generator design.
    scrubbed_input = {
        k: v
        for k, v in input_obj.items()
        if k
        not in {
            "txn_id",
            "entity_id",
            "template_family",
            "prompt",
        }
    }
    material = {
        "task": payload.get("task"),
        "difficulty": payload.get("difficulty"),
        "input": _normalize_obj(scrubbed_input),
        "expected_output": _normalize_obj(payload.get("expected_output") or {}),
    }
    return content_sha256(material)


def _normalize_obj(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, dict):
        return {k: _normalize_obj(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_obj(v) for v in value]
    return value


def shingles(text: str, n: int = DEFAULT_NGRAM) -> set[str]:
    """Whitespace token n-grams (practical near-dup signal on compact fingerprints)."""
    tokens = normalize_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def example_fingerprint(example: FinanceTaskEnvelope | dict[str, Any]) -> str:
    """Compact identity string for MinHash near-duplicate detection."""
    if isinstance(example, FinanceTaskEnvelope):
        payload = example.model_dump(mode="json")
    else:
        payload = example
    inp = payload.get("input") or {}
    out = payload.get("expected_output") or {}
    parts = [
        str(payload.get("task")),
        str(payload.get("difficulty")),
        str((payload.get("provenance") or {}).get("template_family")),
        str(inp.get("case_nonce", "")),
        str(inp.get("descriptor", "")),
        str(inp.get("vendor", "")),
        str(inp.get("amount_minor", "")),
        str(inp.get("period", "")),
        str(inp.get("budget_minor", "")),
        str(inp.get("actual_minor", "")),
        str(inp.get("book_balance_minor", "")),
        str(inp.get("bank_balance_minor", "")),
        str(out.get("gl_account", "")),
        str(out.get("policy_action", "")),
        str(out.get("profit_impact_minor", "")),
        str(out.get("direction", "")),
        str(out.get("difference_minor", "")),
        str(out.get("status", "")),
    ]
    drivers = out.get("top_drivers") or []
    if isinstance(drivers, list):
        parts.append(
            "|".join(
                f"{d.get('driver_id')}:{d.get('impact_minor')}"
                for d in drivers
                if isinstance(d, dict)
            )
        )
    return " ".join(normalize_text(p) for p in parts if p)


def _example_text(example: FinanceTaskEnvelope | dict[str, Any]) -> str:
    return example_fingerprint(example)


def minhash_signature(
    text: str,
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram: int = DEFAULT_NGRAM,
) -> tuple[int, ...]:
    """Practical MinHash using independent hash streams (no external deps)."""
    grams = shingles(text, ngram)
    if not grams:
        return tuple(0 for _ in range(num_perm))
    sig: list[int] = []
    for i in range(num_perm):
        best = 2**64 - 1
        for g in grams:
            digest = hashlib.blake2b(
                f"{i}:{g}".encode(),
                digest_size=8,
            ).digest()
            val = int.from_bytes(digest, "big")
            if val < best:
                best = val
        sig.append(best)
    return tuple(sig)


def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    agree = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
    return agree / len(sig_a)


def check_leakage(
    examples: Iterable[FinanceTaskEnvelope | dict[str, Any]],
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    check_near_duplicates: bool = True,
) -> LeakageReport:
    """Reject exact normalized duplicates and cross-split identity/template leakage."""
    items: list[FinanceTaskEnvelope] = []
    for ex in examples:
        if isinstance(ex, FinanceTaskEnvelope):
            items.append(ex)
        else:
            items.append(FinanceTaskEnvelope.model_validate(ex))

    findings: list[LeakFinding] = []
    by_norm: dict[str, list[str]] = defaultdict(list)
    for ex in items:
        h = normalized_content_hash(ex)
        by_norm[h].append(ex.example_id)

    exact_groups = 0
    for h, ids in by_norm.items():
        if len(ids) > 1:
            exact_groups += 1
            for other in ids[1:]:
                findings.append(
                    LeakFinding(
                        kind="exact_normalized_duplicate",
                        left_id=ids[0],
                        right_id=other,
                        detail=f"normalized_hash={h}",
                    )
                )

    # Cross-split isolation on identity keys.
    id_fields = (
        ("world_id", lambda e: e.world_id),
        ("group_id", lambda e: e.group_id),
        ("entity_id", lambda e: str((e.input or {}).get("entity_id", ""))),
        ("txn_id", lambda e: str((e.input or {}).get("txn_id", ""))),
        (
            "close_period+entity",
            lambda e: (
                f"{(e.input or {}).get('close_period', '')}"
                f"|{(e.input or {}).get('entity_id', '')}"
            ),
        ),
    )
    cross_id = 0
    for field_name, getter in id_fields:
        owners: dict[str, set[SplitName]] = defaultdict(set)
        id_to_example: dict[str, str] = {}
        for ex in items:
            key = getter(ex)
            if not key or key == "|":
                continue
            owners[key].add(ex.provenance.split)
            id_to_example.setdefault(key, ex.example_id)
        for key, splits in owners.items():
            if len(splits) > 1:
                cross_id += 1
                split_list = ",".join(sorted(s.value for s in splits))
                findings.append(
                    LeakFinding(
                        kind="cross_split_id",
                        left_id=id_to_example[key],
                        right_id=key,
                        detail=f"{field_name} spans splits={split_list}",
                    )
                )

    # Template families must not cross train/val/IID vs OOD improperly.
    # Strong rule: no template_family appears in both OOD and any non-OOD split.
    template_owners: dict[str, set[SplitName]] = defaultdict(set)
    template_example: dict[str, str] = {}
    for ex in items:
        fam = ex.provenance.template_family
        template_owners[fam].add(ex.provenance.split)
        template_example.setdefault(fam, ex.example_id)
    cross_tmpl = 0
    for fam, splits in template_owners.items():
        has_ood = SplitName.OOD_TEST in splits
        has_in = bool(splits - {SplitName.OOD_TEST})
        if has_ood and has_in:
            cross_tmpl += 1
            findings.append(
                LeakFinding(
                    kind="cross_split_template",
                    left_id=template_example[fam],
                    right_id=fam,
                    detail=(
                        "template spans OOD and in-distribution splits="
                        f"{{{','.join(sorted(s.value for s in splits))}}}"
                    ),
                )
            )

    near_pairs = 0
    if check_near_duplicates and len(items) >= 2:
        near_pairs, near_findings = _minhash_near_duplicates(
            items,
            jaccard_threshold=jaccard_threshold,
            num_perm=num_perm,
        )
        findings.extend(near_findings)

    ok = not findings
    return LeakageReport(
        ok=ok,
        findings=findings,
        exact_duplicate_groups=exact_groups,
        near_duplicate_pairs=near_pairs,
        cross_split_id_leaks=cross_id,
        cross_split_template_leaks=cross_tmpl,
        normalized_hashes_checked=len(items),
    )


def _minhash_near_duplicates(
    items: list[FinanceTaskEnvelope],
    *,
    jaccard_threshold: float,
    num_perm: int,
    band_size: int = 4,
) -> tuple[int, list[LeakFinding]]:
    """LSH-banded MinHash; only candidate pairs in shared bands are scored."""
    by_task: dict[TaskId, list[FinanceTaskEnvelope]] = defaultdict(list)
    for ex in items:
        by_task[ex.task].append(ex)

    findings: list[LeakFinding] = []
    near_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()

    for task_items in by_task.values():
        records = [
            (
                ex.example_id,
                ex.provenance.split,
                minhash_signature(_example_text(ex), num_perm=num_perm),
            )
            for ex in task_items
        ]
        bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for idx, (_eid, _split, sig) in enumerate(records):
            for band_i in range(0, num_perm, band_size):
                key = (band_i, tuple(sig[band_i : band_i + band_size]))
                bands[key].append(idx)

        candidates: set[tuple[int, int]] = set()
        for idxs in bands.values():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if i > j:
                        i, j = j, i
                    candidates.add((i, j))

        for i, j in candidates:
            left_id, left_split, left_sig = records[i]
            right_id, right_split, right_sig = records[j]
            pair_key = (left_id, right_id) if left_id < right_id else (right_id, left_id)
            if pair_key in seen_pairs:
                continue
            score = estimated_jaccard(left_sig, right_sig)
            if score < jaccard_threshold:
                continue
            seen_pairs.add(pair_key)
            near_pairs += 1
            cross = left_split != right_split
            findings.append(
                LeakFinding(
                    kind=(
                        "cross_split_near_duplicate" if cross else "near_duplicate"
                    ),
                    left_id=left_id,
                    right_id=right_id,
                    detail=(
                        f"jaccard≈{score:.3f} "
                        f"splits={left_split.value}/{right_split.value}"
                    ),
                )
            )

    return near_pairs, findings
