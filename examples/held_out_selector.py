"""Fixed-seed held-out example selector for demos.

Selects deterministic demonstration cases from held-out splits only
(``iid_test``, ``ood_test``, or ``test``). Never returns train or validation
examples, so demo prompts cannot leak gold training labels into the pitch.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Screen seed from the locked proof protocol. Demo selection uses the same seed
# so judges can reproduce which held-out cases appear on stage.
DEFAULT_DEMO_SEED = 17
HELD_OUT_SPLITS = frozenset({"iid_test", "ood_test", "test"})
PRIMARY_TASKS = ("transaction_review", "variance_analysis")
BACKUP_TASK = "cash_reconciliation"


@dataclass(frozen=True)
class HeldOutSelection:
    seed: int
    tasks: tuple[str, ...]
    per_task: int
    example_ids: tuple[str, ...]
    examples: tuple[dict[str, Any], ...]

    def as_public_view(self) -> list[dict[str, Any]]:
        """Return inputs only; strip expected_output / oracle gold from display."""
        public: list[dict[str, Any]] = []
        for example in self.examples:
            public.append(
                {
                    "example_id": example["example_id"],
                    "task": example["task"],
                    "difficulty": example.get("difficulty"),
                    "split": example.get("provenance", {}).get("split"),
                    "world_id": example.get("world_id"),
                    "group_id": example.get("group_id"),
                    "input": example.get("input", {}),
                }
            )
        return public


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def _split_of(example: dict[str, Any]) -> str | None:
    provenance = example.get("provenance")
    if isinstance(provenance, dict):
        split = provenance.get("split")
        if isinstance(split, str):
            return split
    return None


def filter_held_out(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    held: list[dict[str, Any]] = []
    for example in examples:
        split = _split_of(example)
        if split in HELD_OUT_SPLITS:
            held.append(example)
    return held


def _demo_rank(example: dict[str, Any]) -> tuple[int, int, str]:
    """Lower is better for on-stage demos: prefer valid positives over negatives."""
    tags = example.get("case_tags") or []
    tag_set = set(tags) if isinstance(tags, list) else set()
    negative = 1 if tag_set.intersection({"invalid_json", "negative_case", "unbalanced_journal"}) else 0
    valid = 0 if "valid" in tag_set else 1
    return (negative, valid, str(example.get("example_id", "")))


def select_held_out(
    examples: Sequence[dict[str, Any]],
    *,
    seed: int = DEFAULT_DEMO_SEED,
    tasks: Sequence[str] = PRIMARY_TASKS,
    per_task: int = 1,
    include_backup: bool = False,
    prefer_valid: bool = True,
) -> HeldOutSelection:
    """Deterministically pick ``per_task`` held-out examples for each task.

    Raises ``ValueError`` if a requested task has no held-out coverage.
    When ``prefer_valid`` is true, valid positive cases are selected before
    negative/invalid fixtures (still held-out only; still seeded).
    """
    if per_task < 1:
        raise ValueError("per_task must be >= 1")

    task_list = list(tasks)
    if include_backup and BACKUP_TASK not in task_list:
        task_list.append(BACKUP_TASK)

    held = filter_held_out(examples)
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in task_list}
    for example in held:
        task = example.get("task")
        if task in by_task:
            by_task[task].append(example)

    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for task in task_list:
        pool = list(by_task[task])
        if not pool:
            raise ValueError(
                f"no held-out examples for task={task!r} "
                f"(allowed splits: {sorted(HELD_OUT_SPLITS)})"
            )
        pool.sort(key=lambda row: str(row.get("example_id", "")))
        rng.shuffle(pool)
        if prefer_valid:
            # Stable sort: negatives sink; shuffle order breaks ties within a rank.
            pool.sort(key=_demo_rank)
        if len(pool) < per_task:
            raise ValueError(
                f"need {per_task} held-out examples for task={task!r}, found {len(pool)}"
            )
        chosen.extend(pool[:per_task])

    example_ids = tuple(str(row["example_id"]) for row in chosen)
    return HeldOutSelection(
        seed=seed,
        tasks=tuple(task_list),
        per_task=per_task,
        example_ids=example_ids,
        examples=tuple(chosen),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/finance_world_v1/golden.jsonl"),
        help="JSONL dataset path (default: golden fixture)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument(
        "--include-backup",
        action="store_true",
        help="Also select cash_reconciliation when available",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Emit inputs only (strip expected_output / oracle)",
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.dataset)
    selection = select_held_out(
        rows,
        seed=args.seed,
        per_task=args.per_task,
        include_backup=args.include_backup,
    )
    payload: dict[str, Any] = {
        "seed": selection.seed,
        "tasks": list(selection.tasks),
        "per_task": selection.per_task,
        "example_ids": list(selection.example_ids),
        "claim": (
            "Held-out selector only; synthetic fixture demo. "
            "Real benchmark results are pending."
        ),
    }
    if args.public_only:
        payload["examples"] = selection.as_public_view()
    else:
        payload["examples"] = list(selection.examples)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
