"""Sealed run manifest: distillery.run.v1. SHA-256 addressed, immutable once submitted."""
from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from .dataset import canonical_json

RunState = Literal[
    "QUEUED", "STARTING", "SYNTHESIZING", "TRAINING",
    "EVALUATING", "FINALIZING", "SUCCEEDED", "FAILED", "CANCELLED",
]

# Monotonic transition table; append-only.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "QUEUED": {"STARTING", "FAILED", "CANCELLED"},
    "STARTING": {"SYNTHESIZING", "TRAINING", "FAILED", "CANCELLED"},
    "SYNTHESIZING": {"TRAINING", "FAILED", "CANCELLED"},
    "TRAINING": {"EVALUATING", "FAILED", "CANCELLED"},
    "EVALUATING": {"FINALIZING", "FAILED", "CANCELLED"},
    "FINALIZING": {"SUCCEEDED", "FAILED", "CANCELLED"},
    "SUCCEEDED": set(), "FAILED": set(), "CANCELLED": set(),
}


class ModelPin(BaseModel):
    id: str
    revision: str  # commit SHA or provider-pinned model id (e.g. dated Anthropic model id)
    access: Literal["white_box", "api_black_box"]
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None


class DatasetRef(BaseModel):
    dataset_id: str
    uri: str
    sha256: str
    split_sha256: dict[str, str]


class TrainingConfig(BaseModel):
    seed: int = 17
    max_steps: int = 200
    token_budget: int = 0
    max_length: int = 768
    completion_cap: int = 256
    micro_batch: int = 1
    grad_accum: int = 16
    lr: float = 1e-4
    qlora: dict[str, Any] = Field(default_factory=lambda: {"r": 16, "alpha": 32, "dropout": 0.05})
    logit: dict[str, Any] = Field(default_factory=lambda: {"T": 2.0, "kd_weight": 0.7, "ce_weight": 0.3, "vocab_chunk": 4096})


class RuntimeConfig(BaseModel):
    backend: Literal["local", "sagemaker"] = "local"
    region: str = "us-east-1"
    instance_type: str = "ml.g5.xlarge"
    image_digest: str | None = None
    max_runtime_seconds: int = 10800


class CostConfig(BaseModel):
    max_run_usd: float = 25.0
    estimate_low_usd: float = 0.0
    estimate_high_usd: float = 0.0


class RunManifest(BaseModel):
    schema_version: str = "distillery.run.v1"
    run_id: str
    created_at: str
    dataset: DatasetRef
    models: dict[str, ModelPin]  # keys: teacher, student
    recipe: dict[str, Any]       # requested / resolved / resolver_reasons
    arm: str                     # e.g. sequence_kd, logit_kd, ce_ablation, oracle_sft
    training: TrainingConfig
    proof_protocol: dict[str, Any] = Field(default_factory=lambda: {"id": "finance-proof.v1", "sha256": None})
    runtime: RuntimeConfig
    cost: CostConfig
    output_prefix: str

    def seal_hash(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self.model_dump()).encode()).hexdigest()
