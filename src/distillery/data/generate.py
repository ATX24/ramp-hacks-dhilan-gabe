"""Smoke/full corpus generation with deterministic sealing and provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from distillery.contracts.dataset import FinanceWorldVersion, TaskDifficultyCounts
from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import (
    SCHEMA_VERSION_FINANCE_WORLD,
    SCHEMA_VERSION_FINANCE_WORLD_V2,
    Difficulty,
    FinanceTaskEnvelope,
    LabelSource,
    Provenance,
    SplitName,
    TaskId,
)
from distillery.data.leakage import LeakageReport, check_leakage, normalized_content_hash
from distillery.data.mixture import (
    TASK_MIXTURE,
    TASK_MIXTURE_V2,
    TASK_ORDER,
    TASK_ORDER_V2,
    mixture_plan,
    summarize_mixture,
    task_counts,
)
from distillery.data.oracle import (
    GENERATOR_REVISION,
    GENERATOR_REVISION_V1,
    GENERATOR_REVISION_V2,
    oracle_meta,
    solve_task,
)
from distillery.data.renderers import render_input, select_template_family
from distillery.data.split import (
    FULL_SPLITS,
    FULL_SPLITS_V2,
    SMOKE_SPLITS,
    SMOKE_SPLITS_V2,
    SplitSpec,
)
from distillery.data.validate import validate_example
from distillery.data.world import build_world

CorpusName = Literal["smoke", "full", "smoke_v2", "full_v2"]


@dataclass(frozen=True)
class CorpusSpec:
    name: CorpusName
    seed: int
    splits: tuple[SplitSpec, ...]
    schema_version: FinanceWorldVersion = SCHEMA_VERSION_FINANCE_WORLD
    generator_revision: str = GENERATOR_REVISION_V1
    task_mixture: Mapping[TaskId, float] = field(default_factory=lambda: dict(TASK_MIXTURE))
    task_order: tuple[TaskId, ...] = TASK_ORDER
    corpus_manifest_schema: str = "finance_world.v1.corpus_manifest"

    def __post_init__(self) -> None:
        if len(set(self.task_order)) != len(self.task_order):
            raise ValueError("task_order must contain each task exactly once")
        if set(self.task_mixture) != set(self.task_order):
            raise ValueError("task_mixture and task_order must contain the same tasks")
        TaskDifficultyCounts(
            by_task={task: 0 for task in self.task_order},
            by_difficulty={difficulty: 0 for difficulty in Difficulty},
        ).require_finance_world(self.schema_version)

    @property
    def total_examples(self) -> int:
        return sum(split.count for split in self.splits)


CORPUS_SMOKE = CorpusSpec(name="smoke", seed=17, splits=SMOKE_SPLITS)
CORPUS_FULL = CorpusSpec(name="full", seed=17, splits=FULL_SPLITS)
CORPUS_SMOKE_V2 = CorpusSpec(
    name="smoke_v2",
    seed=17,
    splits=SMOKE_SPLITS_V2,
    schema_version=SCHEMA_VERSION_FINANCE_WORLD_V2,
    generator_revision=GENERATOR_REVISION_V2,
    task_mixture=dict(TASK_MIXTURE_V2),
    task_order=TASK_ORDER_V2,
    corpus_manifest_schema="finance_world.v2.corpus_manifest",
)
CORPUS_FULL_V2 = CorpusSpec(
    name="full_v2",
    seed=17,
    splits=FULL_SPLITS_V2,
    schema_version=SCHEMA_VERSION_FINANCE_WORLD_V2,
    generator_revision=GENERATOR_REVISION_V2,
    task_mixture=dict(TASK_MIXTURE_V2),
    task_order=TASK_ORDER_V2,
    corpus_manifest_schema="finance_world.v2.corpus_manifest",
)

_CORPUS_BY_NAME: dict[str, CorpusSpec] = {
    "smoke": CORPUS_SMOKE,
    "full": CORPUS_FULL,
    "smoke_v2": CORPUS_SMOKE_V2,
    "full_v2": CORPUS_FULL_V2,
}


@dataclass
class GeneratedCorpus:
    spec: CorpusSpec
    examples: list[FinanceTaskEnvelope]
    manifest: dict[str, Any]
    leakage: LeakageReport
    by_split: dict[SplitName, list[FinanceTaskEnvelope]] = field(default_factory=dict)

    def write(self, output_dir: str | Path) -> dict[str, str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for split, examples in self.by_split.items():
            path = root / f"{split.value}.jsonl"
            payload = "\n".join(
                json.dumps(
                    example.model_dump(mode="json"),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                for example in examples
            )
            if examples:
                payload += "\n"
            path.write_text(payload, encoding="utf-8")
            hashes[path.name] = sha256_hex(payload.encode())

        manifest_path = root / "manifest.json"
        manifest_body = {
            **self.manifest,
            "file_sha256": hashes,
            "leakage": self.leakage.to_dict(),
        }
        text = (
            json.dumps(
                manifest_body,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        manifest_path.write_text(text, encoding="utf-8")
        hashes["manifest.json"] = sha256_hex(text.encode())
        return hashes


def generate_corpus(
    spec: CorpusSpec | CorpusName = "smoke",
    *,
    seed: int | None = None,
    validate: bool = True,
    check_near_duplicates: bool = True,
) -> GeneratedCorpus:
    """Generate, validate, shuffle, leak-check, and seal a corpus."""
    if isinstance(spec, str):
        if spec not in _CORPUS_BY_NAME:
            raise ValueError(f"unknown corpus {spec!r}")
        spec = _CORPUS_BY_NAME[spec]
    if seed is not None:
        spec = CorpusSpec(
            name=spec.name,
            seed=seed,
            splits=spec.splits,
            schema_version=spec.schema_version,
            generator_revision=spec.generator_revision,
            task_mixture=dict(spec.task_mixture),
            task_order=spec.task_order,
            corpus_manifest_schema=spec.corpus_manifest_schema,
        )

    examples: list[FinanceTaskEnvelope] = []
    by_split: dict[SplitName, list[FinanceTaskEnvelope]] = {}
    split_hashes: dict[str, str] = {}
    order_hashes: dict[str, str] = {}
    mixture_records: dict[str, Any] = {}
    seen_semantic_hashes: set[str] = set()

    for split_spec in spec.splits:
        generated: list[FinanceTaskEnvelope] = []
        plan = mixture_plan(
            split_spec.count,
            mixture=spec.task_mixture,
            order=spec.task_order,
        )
        for index, (task, difficulty) in enumerate(plan):
            example = _generate_unique(
                seed=spec.seed,
                index=index,
                split_spec=split_spec,
                task=task,
                difficulty=difficulty,
                schema_version=spec.schema_version,
                generator_revision=spec.generator_revision,
                seen_semantic_hashes=seen_semantic_hashes,
            )
            if validate:
                result = validate_example(example)
                if not result.ok:
                    raise ValueError(f"invalid example {example.example_id}: {result.errors}")
            generated.append(example)

        split_examples = _deterministic_shuffle(
            generated,
            seed=spec.seed,
            domain=split_spec.token,
        )
        by_split[split_spec.name] = split_examples
        examples.extend(split_examples)
        split_hashes[split_spec.name.value] = _hash_examples(split_examples)
        order_hashes[split_spec.name.value] = content_sha256(
            [example.example_id for example in split_examples]
        )
        mixture_records[split_spec.name.value] = {
            "count": len(split_examples),
            "mixture": summarize_mixture(
                [(example.task, example.difficulty) for example in split_examples],
                order=spec.task_order,
            ),
            "target_task": {
                task.value: count
                for task, count in task_counts(
                    split_spec.count,
                    mixture=spec.task_mixture,
                    order=spec.task_order,
                ).items()
            },
            "ood": split_spec.ood,
        }

    leakage = check_leakage(
        examples,
        check_near_duplicates=check_near_duplicates,
    )
    if leakage.findings:
        first = leakage.findings[0]
        raise ValueError(f"DATA_LEAKAGE_DETECTED: {len(leakage.findings)} findings; first={first}")

    manifest: dict[str, Any] = {
        "schema_version": spec.corpus_manifest_schema,
        "envelope_schema_version": spec.schema_version,
        "corpus": spec.name,
        "generator_revision": spec.generator_revision,
        "seed": spec.seed,
        "renderer_seed": spec.seed,
        "total_examples": len(examples),
        "split_sha256": split_hashes,
        "order_sha256": order_hashes,
        "content_sha256": content_sha256([example.model_dump(mode="json") for example in examples]),
        "mixtures": mixture_records,
        "task_mixture_target": {
            task.value: weight for task, weight in spec.task_mixture.items()
        },
        "difficulty_mixture_target": {
            "easy": 0.30,
            "medium": 0.40,
            "hard": 0.30,
        },
        "provenance": {
            "label_source": LabelSource.ORACLE.value,
            "generator_revision": spec.generator_revision,
        },
        "leakage_summary": {
            "ok": leakage.ok,
            "findings": len(leakage.findings),
            "exact_duplicate_groups": leakage.exact_duplicate_groups,
            "near_duplicate_pairs": leakage.near_duplicate_pairs,
            "cross_split_id_leaks": leakage.cross_split_id_leaks,
            "cross_split_template_leaks": (leakage.cross_split_template_leaks),
        },
    }
    if spec.schema_version == SCHEMA_VERSION_FINANCE_WORLD_V2:
        merchant_total = sum(
            1 for example in examples if example.task == TaskId.MERCHANT_TAGGING
        )
        manifest["merchant_tagging_examples"] = merchant_total
        manifest["min_full_merchant_examples"] = 1000 if spec.name == "full_v2" else 0
    manifest["manifest_sha256"] = content_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
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
    schema_version: str,
    generator_revision: str,
    seen_semantic_hashes: set[str],
) -> FinanceTaskEnvelope:
    for salt in range(64):
        example = _generate_one(
            seed=seed,
            index=index,
            split_spec=split_spec,
            task=task,
            difficulty=difficulty,
            schema_version=schema_version,
            generator_revision=generator_revision,
            salt=salt,
        )
        semantic_hash = normalized_content_hash(example)
        if semantic_hash not in seen_semantic_hashes:
            seen_semantic_hashes.add(semantic_hash)
            return example
    raise RuntimeError(
        f"could not generate unique semantic case for {split_spec.name}/{task}/{difficulty}/{index}"
    )


def _generate_one(
    *,
    seed: int,
    index: int,
    split_spec: SplitSpec,
    task: TaskId,
    difficulty: Difficulty,
    schema_version: str,
    generator_revision: str,
    salt: int = 0,
) -> FinanceTaskEnvelope:
    world = build_world(
        seed=seed,
        index=index,
        split_token=split_spec.token,
        task=task,
        difficulty=difficulty,
        ood=split_spec.ood,
        salt=salt,
    )
    template_family = select_template_family(
        task,
        difficulty=difficulty,
        ood=split_spec.ood,
        index=index + salt,
        family_key=world.group_id,
    )
    rendered = render_input(
        world,
        task,
        template_family=template_family,
    )
    expected = solve_task(world, task)
    example_key = f"{seed}|{split_spec.token}|{index}|{salt}|{task.value}|{difficulty.value}"
    example_id = f"ex_{hashlib.sha256(example_key.encode()).hexdigest()[:18]}"
    return FinanceTaskEnvelope(
        schema_version=schema_version,  # type: ignore[arg-type]
        example_id=example_id,
        world_id=world.world_id,
        group_id=world.group_id,
        task=task,
        difficulty=difficulty,
        input=rendered,
        expected_output=expected,
        oracle=oracle_meta(world, generator_revision=generator_revision),
        provenance=Provenance(
            split=split_spec.name,
            template_family=template_family,
            label_source=LabelSource.ORACLE,
        ),
        case_tags=("synthetic",),
    )


def _deterministic_shuffle(
    examples: list[FinanceTaskEnvelope],
    *,
    seed: int,
    domain: str,
) -> list[FinanceTaskEnvelope]:
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"{seed}|{domain}|order|{example.example_id}".encode()
        ).digest(),
    )


def _hash_examples(examples: Sequence[FinanceTaskEnvelope]) -> str:
    return content_sha256([example.model_dump(mode="json") for example in examples])


__all__ = [
    "CORPUS_FULL",
    "CORPUS_FULL_V2",
    "CORPUS_SMOKE",
    "CORPUS_SMOKE_V2",
    "CorpusSpec",
    "GeneratedCorpus",
    "GENERATOR_REVISION",
    "generate_corpus",
]
