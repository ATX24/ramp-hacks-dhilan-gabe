"""Smoke/full corpus generation with deterministic sealing and provenance."""

from __future__ import annotations

import hashlib
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
from distillery.data.world import build_world

CorpusName = Literal["smoke", "full"]


@dataclass(frozen=True)
class CorpusSpec:
    name: CorpusName
    seed: int
    splits: tuple[SplitSpec, ...]
    schema_version: str = "finance_world.v1"

    @property
    def total_examples(self) -> int:
        return sum(split.count for split in self.splits)


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
        if spec not in {"smoke", "full"}:
            raise ValueError(f"unknown corpus {spec!r}")
        spec = CORPUS_SMOKE if spec == "smoke" else CORPUS_FULL
    if seed is not None:
        spec = CorpusSpec(name=spec.name, seed=seed, splits=spec.splits)

    examples: list[FinanceTaskEnvelope] = []
    by_split: dict[SplitName, list[FinanceTaskEnvelope]] = {}
    split_hashes: dict[str, str] = {}
    order_hashes: dict[str, str] = {}
    mixture_records: dict[str, Any] = {}
    seen_semantic_hashes: set[str] = set()

    for split_spec in spec.splits:
        generated: list[FinanceTaskEnvelope] = []
        for index, (task, difficulty) in enumerate(mixture_plan(split_spec.count)):
            example = _generate_unique(
                seed=spec.seed,
                index=index,
                split_spec=split_spec,
                task=task,
                difficulty=difficulty,
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
                [(example.task, example.difficulty) for example in split_examples]
            ),
            "target_task": {
                task.value: count for task, count in task_counts(split_spec.count).items()
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
        "schema_version": "finance_world.v2.corpus_manifest",
        "envelope_schema_version": spec.schema_version,
        "corpus": spec.name,
        "generator_revision": GENERATOR_REVISION,
        "seed": spec.seed,
        "renderer_seed": spec.seed,
        "total_examples": len(examples),
        "split_sha256": split_hashes,
        "order_sha256": order_hashes,
        "content_sha256": content_sha256([example.model_dump(mode="json") for example in examples]),
        "mixtures": mixture_records,
        "task_mixture_target": {
            "transaction_review": 0.45,
            "variance_analysis": 0.45,
            "cash_reconciliation": 0.10,
        },
        "difficulty_mixture_target": {
            "easy": 0.30,
            "medium": 0.40,
            "hard": 0.30,
        },
        "provenance": {
            "label_source": LabelSource.ORACLE.value,
            "generator_revision": GENERATOR_REVISION,
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
    seen_semantic_hashes: set[str],
) -> FinanceTaskEnvelope:
    for salt in range(64):
        example = _generate_one(
            seed=seed,
            index=index,
            split_spec=split_spec,
            task=task,
            difficulty=difficulty,
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


def _hash_examples(examples: list[FinanceTaskEnvelope]) -> str:
    return content_sha256([example.model_dump(mode="json") for example in examples])
