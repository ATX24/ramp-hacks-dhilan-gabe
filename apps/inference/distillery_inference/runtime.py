"""Inference runtimes. Production loads pinned local weights; tests use FakeRuntime."""

from __future__ import annotations

import importlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from distillery_inference.bundle import ArtifactManifest, LoadedBundle
from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.prompts import build_messages, render_chat_prompt
from distillery_inference.schemas import TASK_SCHEMA_VERSIONS, FinanceTaskId


@dataclass(frozen=True, slots=True)
class GenerationResult:
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    model_id: str
    artifact_id: str


class InferenceRuntime(Protocol):
    def ready(self) -> bool: ...

    def loaded_model_id(self) -> str | None: ...

    def available_model_ids(self) -> list[str]: ...

    def ensure_artifact(self, artifact: ArtifactManifest) -> None: ...

    def generate(
        self,
        *,
        artifact: ArtifactManifest,
        task: FinanceTaskId,
        example_input: dict[str, Any],
        max_prompt_tokens: int,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        timeout_s: float,
    ) -> GenerationResult: ...


@dataclass
class FakeRuntime:
    """Deterministic in-process runtime for tests. Never loads real model weights."""

    bundle: LoadedBundle
    outputs_by_model: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    fail_models: set[str] = field(default_factory=set)
    timeout_models: set[str] = field(default_factory=set)
    token_overflow_models: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _loaded_model_id: str | None = None
    _switch_count: int = 0
    network_calls: int = 0

    def ready(self) -> bool:
        return True

    def loaded_model_id(self) -> str | None:
        with self._lock:
            return self._loaded_model_id

    def available_model_ids(self) -> list[str]:
        return sorted(self.bundle.artifacts_by_model)

    def ensure_artifact(self, artifact: ArtifactManifest) -> None:
        with self._lock:
            if self._loaded_model_id != artifact.model_id:
                self._switch_count += 1
                self._loaded_model_id = artifact.model_id

    @property
    def switch_count(self) -> int:
        with self._lock:
            return self._switch_count

    def generate(
        self,
        *,
        artifact: ArtifactManifest,
        task: FinanceTaskId,
        example_input: dict[str, Any],
        max_prompt_tokens: int,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        timeout_s: float,
    ) -> GenerationResult:
        del temperature, top_p, seed
        with self._lock:
            self.ensure_artifact(artifact)
            if artifact.model_id in self.fail_models:
                raise InferenceError(
                    InferenceErrorCode.ADAPTER_SWITCH_FAILED,
                    f"Fake runtime refused model {artifact.model_id}",
                    http_status=500,
                    retryable=True,
                )
            if artifact.model_id in self.timeout_models or self.delay_s > timeout_s:
                time.sleep(min(max(self.delay_s, 0.0), timeout_s + 0.01))
                raise InferenceError(
                    InferenceErrorCode.TIMEOUT,
                    f"Inference timed out after {timeout_s:.1f}s",
                    http_status=504,
                    retryable=True,
                )
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            prompt = render_chat_prompt(
                build_messages(task=task, example_input=example_input)
            )
            prompt_tokens = max(1, len(prompt.split()))
            if (
                artifact.model_id in self.token_overflow_models
                or prompt_tokens > max_prompt_tokens
            ):
                raise InferenceError(
                    InferenceErrorCode.TOKEN_LIMIT_EXCEEDED,
                    (
                        f"Prompt tokens {prompt_tokens} exceed limit "
                        f"{max_prompt_tokens}"
                    ),
                    http_status=413,
                    details={
                        "prompt_tokens": prompt_tokens,
                        "max_prompt_tokens": max_prompt_tokens,
                    },
                )
            raw = self.outputs_by_model.get(artifact.model_id)
            if raw is None:
                raw = json.dumps(
                    _default_structured(task=task, example_input=example_input),
                    sort_keys=True,
                )
            completion_tokens = max(1, len(raw.split()))
            if completion_tokens > max_completion_tokens:
                raise InferenceError(
                    InferenceErrorCode.TOKEN_LIMIT_EXCEEDED,
                    (
                        f"Completion tokens {completion_tokens} exceed limit "
                        f"{max_completion_tokens}"
                    ),
                    http_status=413,
                    details={
                        "completion_tokens": completion_tokens,
                        "max_completion_tokens": max_completion_tokens,
                    },
                )
            return GenerationResult(
                raw_output=raw,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_id=artifact.model_id,
                artifact_id=artifact.artifact_id,
            )


def _default_structured(
    *,
    task: FinanceTaskId,
    example_input: dict[str, Any],
) -> dict[str, Any]:
    schema_version = TASK_SCHEMA_VERSIONS[task]
    if task == "transaction_review":
        amount = int(example_input.get("amount_minor", 0))
        account = "6400"
        candidates = example_input.get("gl_candidates")
        if isinstance(candidates, list) and candidates:
            account = str(candidates[0])
        return {
            "task": task,
            "schema_version": schema_version,
            "gl_account": account,
            "journal_entry": [
                {"account": account, "amount_minor": amount, "side": "debit"},
                {"account": "2100", "amount_minor": amount, "side": "credit"},
            ],
            "policy_action": "review",
            "rule_ids": ["POL-FAKE-001"],
            "evidence": [
                {"field": "amount_minor", "source_id": "txn", "value": str(amount)}
            ],
            "confidence": 0.5,
        }
    if task == "variance_analysis":
        return {
            "task": task,
            "schema_version": schema_version,
            "profit_impact_minor": 0,
            "direction": "favorable",
            "top_drivers": [],
            "other_impact_minor": 0,
            "rule_ids": ["VAR-FAKE-001"],
            "evidence_ids": [],
            "confidence": 0.5,
        }
    return {
        "task": task,
        "schema_version": schema_version,
        "status": "balanced",
        "matched_groups": [],
        "exceptions": [],
        "adjusted_book_balance_minor": 0,
        "adjusted_bank_balance_minor": 0,
        "difference_minor": 0,
        "confidence": 0.5,
    }


def build_runtime(
    *,
    bundle: LoadedBundle,
    backend: str,
) -> InferenceRuntime:
    normalized = backend.strip().lower()
    if normalized == "fake":
        return FakeRuntime(bundle=bundle)
    if normalized == "torch":
        module = importlib.import_module("distillery_inference.torch_backend")
        return module.TorchBackend(bundle)
    raise InferenceError(
        InferenceErrorCode.INTERNAL_ERROR,
        f"unsupported inference runtime backend: {backend}",
        http_status=500,
    )
