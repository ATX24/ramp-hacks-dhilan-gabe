"""Finance-world.v2 latent-oracle targets for the adapted 72B fallback."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import (
    FinanceTaskEnvelope,
    LabelSource,
    SplitName,
    TaskId,
)
from distillery.data.generate import GeneratedCorpus, generate_corpus
from distillery.data.oracle import GENERATOR_REVISION_V2
from experiments.qwen72b_fallback.evidence import (
    PREFIXED_SHA256_PATTERN,
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
    sha256_bytes,
)

ALL_TASKS = frozenset(
    {
        TaskId.TRANSACTION_REVIEW,
        TaskId.VARIANCE_ANALYSIS,
        TaskId.MERCHANT_TAGGING,
        TaskId.CASH_RECONCILIATION,
    }
)


class FinanceWorldTargetRecord(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.finance_target.v1"] = (
        "distillery.qwen72b_fallback.finance_target.v1"
    )
    source: Literal[VerificationSource.FINANCE_WORLD_V2] = VerificationSource.FINANCE_WORLD_V2
    envelope_schema_version: Literal["finance_world.v2"] = "finance_world.v2"
    generator_revision: Literal["finance_world.v2"] = GENERATOR_REVISION_V2
    latent_state_hash: str = Field(pattern=PREFIXED_SHA256_PATTERN)
    envelope_sha256: str = Field(pattern=SHA256_PATTERN)
    example_id: str
    task: TaskId
    prompt_text: str
    target_text: str
    envelope: FinanceTaskEnvelope

    @model_validator(mode="after")
    def _verify_envelope_binding(self) -> FinanceWorldTargetRecord:
        if self.envelope.schema_version != "finance_world.v2":
            raise ValueError("72B target must use a finance_world.v2 envelope")
        if self.envelope.oracle.generator_revision != GENERATOR_REVISION_V2:
            raise ValueError("72B target has the wrong finance_world generator revision")
        if self.envelope.oracle.latent_state_hash != self.latent_state_hash:
            raise ValueError("72B target latent_state_hash differs from its envelope")
        if self.envelope.provenance.label_source is not LabelSource.ORACLE:
            raise ValueError("72B oracle target must come from the executable latent oracle")
        if self.envelope.task is not self.task or self.envelope.example_id != self.example_id:
            raise ValueError("72B target identity differs from its envelope")
        if "synthetic" not in self.envelope.case_tags:
            raise ValueError("72B target envelope must be explicitly synthetic")
        envelope_payload = self.envelope.model_dump(mode="json")
        if content_sha256(envelope_payload) != self.envelope_sha256:
            raise ValueError("72B target envelope_sha256 mismatch")
        expected_prompt = json.dumps(
            envelope_payload["input"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        expected_target = json.dumps(
            envelope_payload["expected_output"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if self.prompt_text != expected_prompt or self.target_text != expected_target:
            raise ValueError("72B target text is not the canonical oracle envelope rendering")
        return self


class FinanceWorldCorpusEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.finance_corpus.v1"] = (
        "distillery.qwen72b_fallback.finance_corpus.v1"
    )
    source: Literal[VerificationSource.FINANCE_WORLD_V2] = VerificationSource.FINANCE_WORLD_V2
    generator_revision: Literal["finance_world.v2"] = GENERATOR_REVISION_V2
    source_corpus: Literal["smoke_v2", "full_v2"]
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_content_sha256: str = Field(pattern=SHA256_PATTERN)
    records: tuple[FinanceWorldTargetRecord, ...] = Field(min_length=1)
    record_set_sha256: str = Field(pattern=SHA256_PATTERN)
    task_counts: dict[TaskId, int]

    @model_validator(mode="after")
    def _verify_record_set(self) -> FinanceWorldCorpusEvidence:
        if len({record.example_id for record in self.records}) != len(self.records):
            raise ValueError("finance-world target evidence contains duplicate example IDs")
        actual_counts = Counter(record.task for record in self.records)
        if dict(actual_counts) != self.task_counts:
            raise ValueError("finance-world target task counts differ from records")
        if set(actual_counts) != ALL_TASKS or any(actual_counts[task] < 1 for task in ALL_TASKS):
            raise ValueError("finance-world target evidence must cover all four tasks")
        expected_set_hash = content_sha256(
            [record.model_dump(mode="json") for record in self.records]
        )
        if self.record_set_sha256 != expected_set_hash:
            raise ValueError("finance-world target record_set_sha256 mismatch")
        return self


def _target_from_envelope(envelope: FinanceTaskEnvelope) -> FinanceWorldTargetRecord:
    payload = envelope.model_dump(mode="json")
    return FinanceWorldTargetRecord.seal(
        latent_state_hash=envelope.oracle.latent_state_hash,
        envelope_sha256=content_sha256(payload),
        example_id=envelope.example_id,
        task=envelope.task,
        prompt_text=json.dumps(
            payload["input"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        target_text=json.dumps(
            payload["expected_output"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        envelope=envelope,
    )


def _select_balanced(
    corpus: GeneratedCorpus,
    *,
    per_task: int | None,
) -> tuple[FinanceTaskEnvelope, ...]:
    train = corpus.by_split[SplitName.TRAIN]
    if per_task is None:
        if {example.task for example in train} != ALL_TASKS:
            raise ValueError("finance_world.v2 train split does not cover all four tasks")
        return tuple(sorted(train, key=lambda example: example.example_id))
    selected: list[FinanceTaskEnvelope] = []
    for task in sorted(ALL_TASKS, key=lambda item: item.value):
        candidates = sorted(
            (example for example in train if example.task is task),
            key=lambda example: example.example_id,
        )
        if len(candidates) < per_task:
            raise ValueError(
                f"finance_world.v2 train split has {len(candidates)} {task.value} "
                f"examples; need {per_task}"
            )
        selected.extend(candidates[:per_task])
    return tuple(sorted(selected, key=lambda example: example.example_id))


def build_finance_world_targets(
    *,
    source_corpus: Literal["smoke_v2", "full_v2"],
    per_task: int | None,
) -> FinanceWorldCorpusEvidence:
    corpus = generate_corpus(source_corpus)
    envelopes = _select_balanced(corpus, per_task=per_task)
    records = tuple(_target_from_envelope(envelope) for envelope in envelopes)
    counts = dict(Counter(record.task for record in records))
    return FinanceWorldCorpusEvidence.seal(
        source_corpus=source_corpus,
        source_manifest_sha256=str(corpus.manifest["manifest_sha256"]),
        source_content_sha256=str(corpus.manifest["content_sha256"]),
        records=records,
        record_set_sha256=content_sha256([record.model_dump(mode="json") for record in records]),
        task_counts=counts,
    )


def rehearsal_corpus() -> FinanceWorldCorpusEvidence:
    """Twenty-four real latent-oracle records: six for each finance_world.v2 task."""
    return build_finance_world_targets(source_corpus="smoke_v2", per_task=6)


def write_corpus_channel(
    evidence: FinanceWorldCorpusEvidence,
    destination: Path,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / "train.jsonl"
    manifest_path = destination / "finance_world_evidence.json"
    records_bytes = (
        "\n".join(
            json.dumps(record.model_dump(mode="json"), sort_keys=True)
            for record in evidence.records
        )
        + "\n"
    ).encode("utf-8")
    manifest_bytes = (
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    records_path.write_bytes(records_bytes)
    manifest_path.write_bytes(manifest_bytes)
    return {
        records_path.name: sha256_bytes(records_bytes),
        manifest_path.name: sha256_bytes(manifest_bytes),
    }


def load_corpus_channel(path: Path) -> FinanceWorldCorpusEvidence:
    evidence = FinanceWorldCorpusEvidence.model_validate_json(path.read_bytes())
    return evidence
