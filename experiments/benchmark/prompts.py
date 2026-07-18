"""Deterministic finance-task prompts with hard no-leakage gates.

Uses smoke train/validation envelopes only. Never includes expected_output,
oracle, labels, or other answer fields in the model-visible payload.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from distillery.contracts.tasks import SplitName, TaskId
from distillery.data.generate import CORPUS_SMOKE, generate_corpus
from distillery.data.validate import FORBIDDEN_INPUT_KEYS, find_input_hygiene_errors

SYSTEM_PREAMBLE = (
    "You are TinyFable, a Distillery finance-generalist model. "
    "Respond with a single JSON object only. No markdown fences."
)

FORBIDDEN_PROMPT_KEYS = frozenset(FORBIDDEN_INPUT_KEYS) | frozenset(
    {
        "oracle",
        "response_text",
        "canonical_response_text",
        "teacher_response",
        "completion",
    }
)

TASK_SCHEMA_VERSIONS: dict[str, str] = {
    TaskId.TRANSACTION_REVIEW.value: "transaction_review.v1",
    TaskId.VARIANCE_ANALYSIS.value: "variance_analysis.v1",
    TaskId.CASH_RECONCILIATION.value: "cash_reconciliation.v1",
}

ALLOWED_SPLITS = frozenset({SplitName.TRAIN, SplitName.VALIDATION})
TaskName = Literal[
    "transaction_review",
    "variance_analysis",
    "cash_reconciliation",
]


@dataclass(frozen=True, slots=True)
class BenchmarkPrompt:
    example_id: str
    task: TaskName
    difficulty: str
    split: str
    prompt_text: str
    messages: tuple[dict[str, str], ...]


def assert_no_answer_leakage(payload: Mapping[str, Any]) -> None:
    """Fail loud if any forbidden answer/oracle field is present."""
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if str(key).casefold() in {k.casefold() for k in FORBIDDEN_PROMPT_KEYS}:
                    raise ValueError(f"refusing to expose forbidden prompt field {key!r}")
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_messages(*, task: str, example_input: Mapping[str, Any]) -> list[dict[str, str]]:
    clean_input = _jsonable(example_input)
    if not isinstance(clean_input, dict):
        raise ValueError("example_input must be an object")
    hygiene = find_input_hygiene_errors(clean_input)
    if hygiene:
        raise ValueError(f"input hygiene failed: {hygiene}")
    schema_version = TASK_SCHEMA_VERSIONS[task]
    user_payload = {
        "task": task,
        "schema_version": schema_version,
        "input": clean_input,
        "instructions": (
            f"Emit JSON for {task} with schema_version={schema_version}. "
            "Respond with a single JSON object only."
        ),
    }
    assert_no_answer_leakage(user_payload)
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":")),
        },
    ]


def render_plain_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def build_benchmark_prompts(
    *,
    warmups: int = 20,
    timed: int = 200,
    seed: int = 17,
) -> tuple[list[BenchmarkPrompt], list[BenchmarkPrompt]]:
    """Return (warmup_prompts, timed_prompts) from smoke train/validation only."""
    if warmups < 0 or timed < 1:
        raise ValueError("warmups must be >= 0 and timed must be >= 1")
    corpus = generate_corpus(CORPUS_SMOKE, seed=seed, validate=True)
    pool: list[BenchmarkPrompt] = []
    for split in (SplitName.TRAIN, SplitName.VALIDATION):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"unexpected split requested: {split}")
        for envelope in corpus.by_split[split]:
            if envelope.provenance.split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"refusing non-train/validation example {envelope.example_id} "
                    f"with split={envelope.provenance.split}"
                )
            # Explicitly drop expected_output / provenance before prompt build.
            example_input = _jsonable(envelope.input)
            if not isinstance(example_input, dict):
                raise ValueError(f"example {envelope.example_id} input is not an object")
            messages = build_messages(task=envelope.task.value, example_input=example_input)
            prompt_text = render_plain_prompt(messages)
            assert_no_answer_leakage({"prompt_text": prompt_text, "input": example_input})
            pool.append(
                BenchmarkPrompt(
                    example_id=str(envelope.example_id),
                    task=envelope.task.value,  # type: ignore[arg-type]
                    difficulty=envelope.difficulty.value,
                    split=split.value,
                    prompt_text=prompt_text,
                    messages=tuple(messages),
                )
            )
    if not pool:
        raise ValueError("benchmark prompt pool is empty")
    # Deterministic round-robin by task so all three finance tasks appear.
    by_task: dict[str, list[BenchmarkPrompt]] = {
        TaskId.TRANSACTION_REVIEW.value: [],
        TaskId.VARIANCE_ANALYSIS.value: [],
        TaskId.CASH_RECONCILIATION.value: [],
    }
    for prompt in pool:
        by_task[prompt.task].append(prompt)
    ordered: list[BenchmarkPrompt] = []
    task_order = (
        TaskId.TRANSACTION_REVIEW.value,
        TaskId.VARIANCE_ANALYSIS.value,
        TaskId.CASH_RECONCILIATION.value,
    )
    indices = {task: 0 for task in task_order}
    while len(ordered) < warmups + timed:
        progressed = False
        for task in task_order:
            bucket = by_task[task]
            if not bucket:
                continue
            ordered.append(bucket[indices[task] % len(bucket)])
            indices[task] += 1
            progressed = True
            if len(ordered) >= warmups + timed:
                break
        if not progressed:
            raise ValueError("unable to sample prompts across finance tasks")
    return ordered[:warmups], ordered[warmups : warmups + timed]
