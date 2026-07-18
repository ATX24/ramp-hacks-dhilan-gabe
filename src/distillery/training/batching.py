"""Deterministic batching and sampler-order hashing (no model I/O)."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256


class MixtureSpec(BaseModel):
    """Task/difficulty mixture with deterministic largest-remainder rounding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_weights: dict[str, float]
    difficulty_weights: dict[str, float]
    rounding_method: Literal["largest_remainder_stable_order"] = (
        "largest_remainder_stable_order"
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> MixtureSpec:
        for name, weights in (
            ("task_weights", self.task_weights),
            ("difficulty_weights", self.difficulty_weights),
        ):
            if not weights:
                raise ValueError(f"{name} must be nonempty")
            invalid = [
                key
                for key, value in weights.items()
                if not key
                or not math.isfinite(value)
                or value <= 0.0
            ]
            if invalid:
                raise ValueError(f"{name} has invalid entries: {invalid}")
        return self

    def normalized_task_weights(self) -> dict[str, float]:
        total = sum(self.task_weights.values())
        if total <= 0.0:
            raise ValueError("task_weights must sum to a positive value")
        return {k: v / total for k, v in self.task_weights.items()}

    def normalized_difficulty_weights(self) -> dict[str, float]:
        total = sum(self.difficulty_weights.values())
        if total <= 0.0:
            raise ValueError("difficulty_weights must sum to a positive value")
        return {k: v / total for k, v in self.difficulty_weights.items()}


class SamplerExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    task: str
    difficulty: str
    completion_tokens: int = Field(ge=1)
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=1)
    completion_token_source: Literal["student_tokenizer"]
    completion_tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_token_counts(self) -> SamplerExample:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                "total_tokens must equal prompt_tokens + completion_tokens"
            )
        return self


class BatchPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order: tuple[str, ...]
    batches: tuple[tuple[str, ...], ...]
    sampler_order_hash: str
    seed: int
    microbatch_size: int
    mixture: dict[str, float]
    difficulty_mixture: dict[str, float]
    rounding_method: str
    expected_task_counts: dict[str, int]
    expected_difficulty_counts: dict[str, dict[str, int]]


DEFAULT_FINANCE_MIXTURE = MixtureSpec(
    task_weights={
        "transaction_review": 0.45,
        "variance_analysis": 0.45,
        "cash_reconciliation": 0.10,
    },
    difficulty_weights={"easy": 0.30, "medium": 0.40, "hard": 0.30},
)

# finance_world.v2 / finance-proof.v2 sampler mixture (does not replace DEFAULT).
FINANCE_MIXTURE_V2 = MixtureSpec(
    task_weights={
        "transaction_review": 0.35,
        "variance_analysis": 0.35,
        "merchant_tagging": 0.20,
        "cash_reconciliation": 0.10,
    },
    difficulty_weights={"easy": 0.30, "medium": 0.40, "hard": 0.30},
)


def sampler_order_hash(
    example_ids: Sequence[str],
    *,
    seed: int,
    mixture: Mapping[str, float] | None = None,
    microbatch_size: int = 1,
    extras: Mapping[str, Any] | None = None,
) -> str:
    """Content-addressed hash of the deterministic sample order and knobs."""
    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size < 1
    ):
        raise ValueError("microbatch_size must be an integer >= 1")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("sampler order contains duplicate example ids")
    payload = {
        "example_ids": list(example_ids),
        "seed": seed,
        "mixture": dict(mixture or {}),
        "microbatch_size": microbatch_size,
        "extras": dict(extras or {}),
    }
    return content_sha256(payload)


def deterministic_shuffle(
    example_ids: Sequence[str],
    *,
    seed: int,
) -> list[str]:
    """Seeded shuffle that does not mutate the input sequence."""
    items = list(example_ids)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def largest_remainder_counts(
    total: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    """
    Allocate ``total`` with Hamilton/largest-remainder rounding.

    Ties are broken by the declared mapping order, then key. This makes smoke
    and full corpus rounding stable and auditable.
    """
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if not weights:
        raise ValueError("weights must be nonempty")
    keys = list(weights)
    if any(
        not key
        or not math.isfinite(value)
        or value <= 0.0
        for key, value in weights.items()
    ):
        raise ValueError("weights must have nonempty keys and finite positive values")
    weight_total = sum(weights.values())
    quotas = {key: total * weights[key] / weight_total for key in keys}
    counts = {key: math.floor(quotas[key]) for key in keys}
    remaining = total - sum(counts.values())
    order_index = {key: index for index, key in enumerate(keys)}
    remainder_order = sorted(
        keys,
        key=lambda key: (
            -(quotas[key] - counts[key]),
            order_index[key],
            key,
        ),
    )
    for key in remainder_order[:remaining]:
        counts[key] += 1
    return counts


def validate_mixture_distribution(
    examples: Sequence[SamplerExample],
    *,
    mixture: MixtureSpec,
    require_difficulty_lra: bool = True,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Require exact rounded task weights and per-task difficulty counts.

    When ``require_difficulty_lra`` is false, task LRA is still enforced, but
    per-task difficulty counts are taken from the observed examples. Emergency
    smoke corpora use joint cell allocation that can diverge from independent
    difficulty LRA at small per-task N.
    """
    if not examples:
        raise ValueError("sampler requires at least one example")
    ids = [example.example_id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("sampler example_id values must be unique")

    known_tasks = set(mixture.task_weights)
    known_difficulties = set(mixture.difficulty_weights)
    unknown_tasks = sorted({example.task for example in examples} - known_tasks)
    unknown_difficulties = sorted(
        {example.difficulty for example in examples} - known_difficulties
    )
    if unknown_tasks:
        raise ValueError(
            f"unknown tasks {unknown_tasks}; declare them explicitly in task_weights"
        )
    if unknown_difficulties:
        raise ValueError(
            "unknown difficulties "
            f"{unknown_difficulties}; declare them explicitly in difficulty_weights"
        )
    if any(example.completion_token_source != "student_tokenizer" for example in examples):
        raise ValueError("all completion counts must come from the student tokenizer")

    expected_tasks = largest_remainder_counts(
        len(examples), mixture.normalized_task_weights()
    )
    actual_tasks = {task: 0 for task in mixture.task_weights}
    for example in examples:
        actual_tasks[example.task] += 1
    if actual_tasks != expected_tasks:
        raise ValueError(
            "task mixture does not match deterministic largest-remainder allocation: "
            f"expected={expected_tasks} actual={actual_tasks}"
        )

    expected_difficulties: dict[str, dict[str, int]] = {}
    for task, task_count in expected_tasks.items():
        expected = largest_remainder_counts(
            task_count, mixture.normalized_difficulty_weights()
        )
        actual = {difficulty: 0 for difficulty in mixture.difficulty_weights}
        for example in examples:
            if example.task == task:
                actual[example.difficulty] += 1
        if require_difficulty_lra:
            if actual != expected:
                raise ValueError(
                    "difficulty mixture does not match deterministic largest-remainder "
                    f"allocation for task {task}: expected={expected} actual={actual}"
                )
            expected_difficulties[task] = expected
        else:
            expected_difficulties[task] = actual
    return expected_tasks, expected_difficulties


def _smooth_schedule(counts: Mapping[str, int], *, seed: int) -> list[str]:
    """Build a prefix-balanced deterministic schedule with exact final counts."""
    total = sum(counts.values())
    if total == 0:
        return []
    keys = [key for key, count in counts.items() if count > 0]
    rng = random.Random(seed)
    tie_order = keys[:]
    rng.shuffle(tie_order)
    tie_rank = {key: index for index, key in enumerate(tie_order)}
    emitted = {key: 0 for key in keys}
    schedule: list[str] = []
    for position in range(total):
        candidates = [key for key in keys if emitted[key] < counts[key]]
        chosen = max(
            candidates,
            key=lambda key: (
                ((position + 1) * counts[key] / total) - emitted[key],
                -tie_rank[key],
            ),
        )
        schedule.append(chosen)
        emitted[chosen] += 1
    return schedule


def build_mixture_aware_order(
    examples: Sequence[SamplerExample],
    *,
    seed: int,
    mixture: MixtureSpec | None = None,
    require_difficulty_lra: bool = True,
) -> list[str]:
    """
    Build a deterministic order that respects task mixture proportions.

    Counts must already match largest-remainder task/difficulty allocations.
    The order interleaves both strata while preserving each example exactly once.
    """
    spec = mixture or DEFAULT_FINANCE_MIXTURE
    expected_tasks, expected_difficulties = validate_mixture_distribution(
        examples,
        mixture=spec,
        require_difficulty_lra=require_difficulty_lra,
    )
    by_stratum: dict[tuple[str, str], list[str]] = {}
    for ex in examples:
        by_stratum.setdefault((ex.task, ex.difficulty), []).append(ex.example_id)
    for stratum, ids in by_stratum.items():
        stratum_seed = (
            seed * 1_000_003 + sum(ord(char) for char in ":".join(stratum))
        ) % (2**32)
        by_stratum[stratum] = deterministic_shuffle(ids, seed=stratum_seed)

    task_schedule = _smooth_schedule(expected_tasks, seed=seed ^ 0xC0FFEE)
    difficulty_schedules = {
        task: _smooth_schedule(
            expected_difficulties[task],
            seed=(seed * 65_537 + sum(ord(char) for char in task)) % (2**32),
        )
        for task in expected_tasks
    }
    difficulty_cursors = {task: 0 for task in expected_tasks}
    stratum_cursors = {stratum: 0 for stratum in by_stratum}
    order: list[str] = []
    for task in task_schedule:
        difficulty_index = difficulty_cursors[task]
        difficulty = difficulty_schedules[task][difficulty_index]
        difficulty_cursors[task] += 1
        stratum = (task, difficulty)
        item_index = stratum_cursors[stratum]
        order.append(by_stratum[stratum][item_index])
        stratum_cursors[stratum] += 1

    return order


def chunk_batches(
    ordered_ids: Sequence[str],
    *,
    microbatch_size: int,
) -> list[tuple[str, ...]]:
    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size < 1
    ):
        raise ValueError("microbatch_size must be an integer >= 1")
    batches: list[tuple[str, ...]] = []
    for i in range(0, len(ordered_ids), microbatch_size):
        batches.append(tuple(ordered_ids[i : i + microbatch_size]))
    return batches


def plan_batches(
    examples: Sequence[SamplerExample],
    *,
    seed: int,
    microbatch_size: int = 1,
    mixture: MixtureSpec | None = None,
    require_difficulty_lra: bool = True,
) -> BatchPlan:
    """End-to-end deterministic batch plan with sampler_order_hash."""
    spec = mixture or DEFAULT_FINANCE_MIXTURE
    expected_tasks, expected_difficulties = validate_mixture_distribution(
        examples,
        mixture=spec,
        require_difficulty_lra=require_difficulty_lra,
    )
    order = build_mixture_aware_order(
        examples,
        seed=seed,
        mixture=spec,
        require_difficulty_lra=require_difficulty_lra,
    )
    batches = chunk_batches(order, microbatch_size=microbatch_size)
    order_hash = sampler_order_hash(
        order,
        seed=seed,
        mixture=spec.normalized_task_weights(),
        microbatch_size=microbatch_size,
        extras={
            "difficulty_mixture": spec.normalized_difficulty_weights(),
            "rounding_method": spec.rounding_method,
            "expected_task_counts": expected_tasks,
            "expected_difficulty_counts": expected_difficulties,
            "completion_tokens": {
                example.example_id: example.completion_tokens for example in examples
            },
            "prompt_tokens": {
                example.example_id: example.prompt_tokens for example in examples
            },
            "total_tokens": {
                example.example_id: example.total_tokens for example in examples
            },
            "completion_token_sources": {
                example.example_id: example.completion_token_source
                for example in examples
            },
            "completion_tokenizer_sha256": {
                example.example_id: example.completion_tokenizer_sha256
                for example in examples
            },
            "record_sha256": {
                example.example_id: example.record_sha256 for example in examples
            },
        },
    )
    return BatchPlan(
        order=tuple(order),
        batches=tuple(batches),
        sampler_order_hash=order_hash,
        seed=seed,
        microbatch_size=microbatch_size,
        mixture=spec.normalized_task_weights(),
        difficulty_mixture=spec.normalized_difficulty_weights(),
        rounding_method=spec.rounding_method,
        expected_task_counts=expected_tasks,
        expected_difficulty_counts=expected_difficulties,
    )


def completion_token_budget_filter(
    examples: Sequence[SamplerExample],
    *,
    max_completion_tokens: int,
) -> tuple[tuple[SamplerExample, ...], tuple[SamplerExample, ...]]:
    """Split examples into kept/discarded under a shared completion-token cap."""
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ValueError("max_completion_tokens must be an integer >= 1")
    kept: list[SamplerExample] = []
    discarded: list[SamplerExample] = []
    for ex in examples:
        if ex.completion_tokens <= max_completion_tokens:
            kept.append(ex)
        else:
            discarded.append(ex)
    return tuple(kept), tuple(discarded)
