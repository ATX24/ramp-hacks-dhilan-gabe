"""The four durable resources: Dataset, Distillation, Model, Deployment.

Dataset and Model are immutable once created. Distillation is asynchronous.
Deployment is callable and carries the alias + fallback chain.
"""
from __future__ import annotations

import enum
import hashlib
import json
import secrets
import time
from typing import Any

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time()):x}{secrets.token_hex(5)}"


class Record(BaseModel):
    """One trace: input descriptor -> structured normalization."""

    input: str
    output: dict[str, Any] | None = None  # None until a teacher labels it


class Dataset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ds"))
    records: list[Record]
    holdout_fraction: float = 0.2
    seed: int = 1337
    created_at: float = Field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        blob = json.dumps([r.model_dump() for r in self.records], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def split(self) -> tuple[list[Record], list[Record]]:
        """Deterministic train/holdout split. The holdout is frozen at creation:
        same dataset + seed always yields the same holdout, so the proof gate
        can never be gamed by re-rolling the eval set."""
        import random

        idx = list(range(len(self.records)))
        random.Random(self.seed).shuffle(idx)
        cut = max(1, int(len(idx) * self.holdout_fraction))
        holdout = {i for i in idx[:cut]}
        train = [r for i, r in enumerate(self.records) if i not in holdout]
        held = [r for i, r in enumerate(self.records) if i in holdout]
        return train, held


class DistillationStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    READY = "READY"
    BLOCKED = "BLOCKED"  # candidate failed the proof gate — not an error, not deployed
    FAILED = "FAILED"


class Target(BaseModel):
    quality: float = 0.92  # min field accuracy on frozen holdout
    cost: float = 0.10  # max cost ratio vs teacher


class Distillation(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dist"))
    teacher: str
    student_base: str
    dataset_id: str
    recipe: str
    target: Target
    status: DistillationStatus = DistillationStatus.QUEUED
    model_id: str | None = None
    report: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)


class Model(BaseModel):
    """Immutable trained artifact + its proof report."""

    id: str = Field(default_factory=lambda: new_id("model"))
    base: str
    backend_ref: str  # backend-specific handle (Bedrock custom model ARN, or mock ref)
    distillation_id: str
    report: dict[str, Any]
    created_at: float = Field(default_factory=time.time)


class Deployment(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dep"))
    alias: str  # OpenAI-compatible model alias, stable across promotions
    model_id: str
    fallback: str  # teacher model to fall back to on invalid output
    history: list[str] = Field(default_factory=list)  # prior model_ids, for rollback
    created_at: float = Field(default_factory=time.time)
