"""Corpus generation with the locked mixtures: 45/45/10 tasks, 30/40/30 difficulty,
group-disjoint splits, OOD from a shifted regime. Deterministic given the seed plan."""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts.dataset import Example, OracleRef, Provenance, canonical_json, hash_examples
from . import oracle as O
from .renderers import render_transaction_review, render_variance_analysis, render_cash_reconciliation
from .world import build_world, GENERATOR_REVISION

# smoke corpus: 320 train / 80 val / 160 test (split 80 iid / 80 ood)
SMOKE_PLAN = {"train": 320, "validation": 80, "test_iid": 80, "test_ood": 80}
FULL_PLAN = {"train": 3200, "validation": 400, "test_iid": 800, "test_ood": 800}

TASK_MIX = [("transaction_review", 0.45), ("variance_analysis", 0.45), ("cash_reconciliation", 0.10)]
DIFF_MIX = [("easy", 0.30), ("medium", 0.40), ("hard", 0.30)]


def _alloc(n: int, mix: list[tuple[str, float]]) -> dict[str, int]:
    counts = {k: int(n * f) for k, f in mix}
    counts[mix[0][0]] += n - sum(counts.values())
    return counts


def generate_split(split: str, n: int, world_offset: int) -> tuple[list[Example], int]:
    regime = "ood" if split == "test_ood" else "iid"
    task_counts = _alloc(n, TASK_MIX)
    examples: list[Example] = []
    widx = world_offset
    diff_cycle = ["easy", "medium", "medium", "hard", "easy", "medium", "hard", "hard", "easy", "medium"]

    made = {k: 0 for k in task_counts}
    i = 0
    while any(made[t] < task_counts[t] for t in made):
        w = build_world(widx, regime=regime)
        widx += 1
        for task in ("transaction_review", "variance_analysis", "cash_reconciliation"):
            if made[task] >= task_counts[task]:
                continue
            if task == "transaction_review":
                txn = w.txns[i % len(w.txns)]
                inp, exp = render_transaction_review(w, txn), O.transaction_review_expected(w, txn)
            elif task == "variance_analysis":
                inp, exp = render_variance_analysis(w), O.variance_analysis_expected(w)
            else:
                inp, exp = render_cash_reconciliation(w), O.cash_reconciliation_expected(w)
            difficulty = diff_cycle[(i + made[task]) % len(diff_cycle)]
            examples.append(Example(
                example_id=f"ex_{split}_{len(examples):05d}",
                world_id=w.world_id, group_id=w.group_id,
                task=task, difficulty=difficulty, input=inp, expected_output=exp,
                oracle=OracleRef(generator_revision=GENERATOR_REVISION,
                                 latent_state_hash=O.latent_state_hash(w)),
                provenance=Provenance(split=split, template_family=w.template_family,
                                      label_source="oracle"),
            ))
            made[task] += 1
        i += 1
    return examples, widx


def generate_corpus(plan: dict[str, int], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_meta = {"splits": {}, "generator_revision": GENERATOR_REVISION}
    offset = 0
    world_ids: dict[str, set] = {}
    for split, n in plan.items():
        exs, offset = generate_split(split, n, offset)
        # OOD worlds use a different regime namespace so offsets can't collide,
        # but keep the invariant explicit:
        world_ids[split] = {e.world_id for e in exs}
        path = out_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for e in exs:
                f.write(canonical_json(e.model_dump()) + "\n")
        all_meta["splits"][split] = {
            "count": len(exs), "sha256": hash_examples(exs),
            "by_task": _count(exs, lambda e: e.task),
            "by_difficulty": _count(exs, lambda e: e.difficulty),
        }
    # leakage check: no world_id crosses splits
    seen: dict[str, str] = {}
    for split, ids in world_ids.items():
        for wid in ids:
            if wid in seen:
                raise AssertionError(f"DATA_LEAKAGE_DETECTED: {wid} in {seen[wid]} and {split}")
            seen[wid] = split
    (out_dir / "metadata.json").write_text(json.dumps(all_meta, indent=2))
    return all_meta


def _count(exs, key):
    d: dict[str, int] = {}
    for e in exs:
        d[key(e)] = d.get(key(e), 0) + 1
    return d


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    plan = SMOKE_PLAN if which == "smoke" else FULL_PLAN
    meta = generate_corpus(plan, Path(sys.argv[2] if len(sys.argv) > 2 else f"data/{which}"))
    print(json.dumps(meta, indent=2))
