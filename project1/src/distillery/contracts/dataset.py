"""Canonical task envelope: finance_world.v1."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskId = Literal["transaction_review", "variance_analysis", "cash_reconciliation", "merchant_tagging"]
Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["train", "validation", "test_iid", "test_ood"]
LabelSource = Literal["oracle", "teacher", "imported"]

SCHEMA_VERSION = "finance_world.v1"


class OracleRef(BaseModel):
    generator_revision: str
    latent_state_hash: str


class Provenance(BaseModel):
    split: Split
    template_family: str
    label_source: LabelSource
    teacher_model: str | None = None
    teacher_params: dict[str, Any] | None = None


class Example(BaseModel):
    schema_version: str = SCHEMA_VERSION
    example_id: str
    world_id: str
    group_id: str
    task: TaskId
    difficulty: Difficulty
    input: dict[str, Any]
    expected_output: dict[str, Any]
    response: str | None = None  # training response text (JSON), if materialized
    oracle: OracleRef
    provenance: Provenance


class DatasetMeta(BaseModel):
    dataset_id: str
    schema_version: str = SCHEMA_VERSION
    sha256: str
    split_sha256: dict[str, str]
    counts_by_task: dict[str, int]
    counts_by_split: dict[str, int]
    counts_by_label_source: dict[str, int]
    counts_by_difficulty: dict[str, int]
    uri: str
    created_at: str


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def hash_examples(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(canonical_json(ex.model_dump()).encode())
        h.update(b"\n")
    return "sha256:" + h.hexdigest()
