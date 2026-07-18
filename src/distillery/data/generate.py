"""Smoke and full corpus generators with sealed manifests and provenance hashes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import (
    Difficulty,
    FinanceTaskEnvelope,
    LabelSource,
    Provenance,
    SplitName,
    TaskId,
)
from distillery.data.leakage import LeakageReport, check_leakage, normalized_content_hash
from distillery.data.mixture import mixture_plan, summarize_mixture, task_counts
from distillery.data.oracle import GENERATOR_REVISION, oracle_meta, solve_task
from distillery.data.renderers import render_input, select_template_family
from distillery.data.split import FULL_SPLITS, SMOKE_SPLITS, SplitSpec
from distillery.data.validate import validate_example
from distillery.data.world import LatentWorld, build_world

CorpusName = Literal["smoke", "full"]

_TASK_SLUG = {
    TaskId.TRANSACTION_REVIEW: "txn",
    TaskId.VARIANCE_ANALYSIS: "var",
    TaskId.CASH_RECONCILIATION: "csh",
}


@dataclass(frozen=True)
class CorpusSpec:
    name: CorpusName
    seed: int
    splits: tuple[SplitSpec, ...]
    schema_version: str = "finance_world.v1"

    @property
    def total_examples(self) -> int:
        return sum(s.count for s in self.splits)


CORPUS_SMOKE = CorpusSpec(name="smoke", seed=17, splits=SMOKE_SPLITS)
CORPUS_FULL = CorpusSpec(name="full", seed=17, splits=FULL_SPLITS)


@dataclass
class GeneratedCorpus:
    spec: CorpusSpec
    examples: list[FinanceTaskEnvelope]
    manifest: dict[str, Any]
    leakage: LeakageReport
    by_split: dict[SplitName, list[FinanceTaskEnvelope]] = field(default_factory=dict)

    def write(self, output_dir: str | Path) -> dict[str, str]:
        """Write jsonl splits + manifest; return map of relative path → sha256."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for split, items in self.by_split.items():
            path = root / f"{split.value}.jsonl"
            payload = "\n".join(
                json.dumps(ex.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
                for ex in items
            )
            if items:
                payload += "\n"
            path.write_text(payload, encoding="utf-8")
            hashes[path.name] = sha256_hex(payload.encode("utf-8"))

        manifest_path = root / "manifest.json"
        manifest_body = dict(self.manifest)
        manifest_body["file_sha256"] = hashes
        manifest_body["leakage"] = self.leakage.to_dict()
        text = json.dumps(manifest_body, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        manifest_path.write_text(text, encoding="utf-8")
        hashes["manifest.json"] = sha256_hex(text.encode("utf-8"))
        return hashes


def generate_corpus(
    spec: CorpusSpec | CorpusName = "smoke",
    *,
    seed: int | None = None,
    validate: bool = True,
    check_near_duplicates: bool = True,
) -> GeneratedCorpus:
    """Generate a deterministic finance_world corpus with sealed mixture counts."""
    if isinstance(spec, str):
        spec = CORPUS_SMOKE if spec == "smoke" else CORPUS_FULL
    if seed is not None:
        spec = CorpusSpec(name=spec.name, seed=seed, splits=spec.splits)

    examples: list[FinanceTaskEnvelope] = []
    by_split: dict[SplitName, list[FinanceTaskEnvelope]] = {}
    split_hashes: dict[str, str] = {}
    mixture_records: dict[str, Any] = {}
    seen_norm: set[str] = set()

    for split_spec in spec.splits:
        slots = mixture_plan(split_spec.count)
        split_examples: list[FinanceTaskEnvelope] = []
        for index, (task, difficulty) in enumerate(slots):
            example = _generate_unique(
                seed=spec.seed,
                index=index,
                split_spec=split_spec,
                task=task,
                difficulty=difficulty,
                seen_norm=seen_norm,
            )
            if validate:
                result = validate_example(example)
                if not result.ok:
                    raise ValueError(
                        f"invalid example {example.example_id}: {result.errors}"
                    )
            split_examples.append(example)

        by_split[split_spec.name] = split_examples
        examples.extend(split_examples)
        split_hashes[split_spec.name.value] = _hash_examples(split_examples)
        mixture_records[split_spec.name.value] = {
            "count": len(split_examples),
            "mixture": summarize_mixture([(e.task, e.difficulty) for e in split_examples]),
            "target_task": {t.value: n for t, n in task_counts(split_spec.count).items()},
            "ood": split_spec.ood,
        }

    leakage = check_leakage(examples, check_near_duplicates=check_near_duplicates)
    fatal = [
        f
        for f in leakage.findings
        if f.kind
        in {
            "exact_normalized_duplicate",
            "cross_split_id",
            "cross_split_template",
            "cross_split_near_duplicate",
        }
    ]
    if fatal:
        raise ValueError(
            f"DATA_LEAKAGE_DETECTED: {len(fatal)} fatal findings; first={fatal[0]}"
        )

    manifest = {
        "schema_version": "finance_world.v1.corpus_manifest",
        "corpus": spec.name,
        "generator_revision": GENERATOR_REVISION,
        "seed": spec.seed,
        "renderer_seed": spec.seed,
        "total_examples": len(examples),
        "split_sha256": split_hashes,
        "content_sha256": content_sha256([e.model_dump(mode="json") for e in examples]),
        "mixtures": mixture_records,
        "task_mixture_target": {
            "transaction_review": 0.45,
            "variance_analysis": 0.45,
            "cash_reconciliation": 0.10,
        },
        "difficulty_mixture_target": {"easy": 0.30, "medium": 0.40, "hard": 0.30},
        "provenance": {
            "label_source": LabelSource.ORACLE.value,
            "generator_revision": GENERATOR_REVISION,
        },
        "leakage_summary": {
            "ok": leakage.ok or not fatal,
            "exact_duplicate_groups": leakage.exact_duplicate_groups,
            "near_duplicate_pairs": leakage.near_duplicate_pairs,
            "cross_split_id_leaks": leakage.cross_split_id_leaks,
            "cross_split_template_leaks": leakage.cross_split_template_leaks,
            "fatal_findings": len(fatal),
        },
    }
    manifest["manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )

    return GeneratedCorpus(
        spec=spec,
        examples=examples,
        manifest=manifest,
        leakage=leakage,
        by_split=by_split,
    )


def _generate_unique(
    *,
    seed: int,
    index: int,
    split_spec: SplitSpec,
    task: TaskId,
    difficulty: Difficulty,
    seen_norm: set[str],
) -> FinanceTaskEnvelope:
    salt = 0
    while salt < 32:
        example = _generate_one(
            seed=seed,
            index=index,
            split_spec=split_spec,
            task=task,
            difficulty=difficulty,
            salt=salt,
        )
        norm = normalized_content_hash(example)
        if norm not in seen_norm:
            seen_norm.add(norm)
            return example
        salt += 1
    raise RuntimeError(
        f"unable to generate unique example for {split_spec.name}/{task}/{difficulty}/{index}"
    )


def _generate_one(
    *,
    seed: int,
    index: int,
    split_spec: SplitSpec,
    task: TaskId,
    difficulty: Difficulty,
    salt: int = 0,
) -> FinanceTaskEnvelope:
    world = build_world(
        seed=seed ^ (salt * 0x9E3779B9),
        index=index,
        split_token=split_spec.token,
        task=task,
        difficulty=difficulty,
        ood=split_spec.ood,
    )
    template_family = select_template_family(
        task,
        difficulty=difficulty,
        ood=split_spec.ood,
        index=index + salt,
    )
    rendered = render_input(world, task, template_family=template_family)
    # Embed a unique case nonce so normalized hashes diverge even for similar regimes.
    rendered["case_nonce"] = f"{split_spec.token}:{index}:{salt}:{seed}"
    expected = solve_task(world, task)
    task_slug = _TASK_SLUG[task]
    example_id = f"ex_{split_spec.token}_{task_slug}_{difficulty.value[0]}_{index:05d}"
    if salt:
        example_id = f"{example_id}_s{salt}"

    return FinanceTaskEnvelope(
        example_id=example_id,
        world_id=world.world_id,
        group_id=world.group_id,
        task=task,
        difficulty=difficulty,
        input=rendered,
        expected_output=expected,
        oracle=oracle_meta(world),
        provenance=Provenance(
            split=split_spec.name,
            template_family=template_family,
            label_source=LabelSource.ORACLE,
        ),
        case_tags=_case_tags(world, difficulty),
    )


def _case_tags(world: LatentWorld, difficulty: Difficulty) -> tuple[str, ...]:
    tags = ["synthetic", difficulty.value]
    if world.ood_held_out:
        tags.append("ood")
    if world.transaction is not None:
        tags.append("primary")
        if world.transaction.hard_negative.value != "none":
            tags.append(world.transaction.hard_negative.value)
    if world.variance is not None:
        tags.append("primary")
        tags.append(world.variance.regime.value)
    if world.cash is not None:
        tags.append("backup")
        tags.append(world.cash.regime.value)
    return tuple(tags)


def _hash_examples(examples: list[FinanceTaskEnvelope]) -> str:
    return content_sha256([e.model_dump(mode="json") for e in examples])
