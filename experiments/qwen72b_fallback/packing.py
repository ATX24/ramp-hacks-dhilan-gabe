"""Packed completion-only sequence construction (prompt masked)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PackingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PackedSequence:
    input_ids: list[int]
    labels: list[int]
    completion_mask: list[float]
    prompt_token_count: int
    completion_token_count: int
    truncated: bool


def pack_completion_only(
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    max_length: int,
    ignore_index: int = -100,
) -> PackedSequence:
    if max_length < 2:
        raise PackingError("max_length must be >= 2")
    if not prompt_ids:
        raise PackingError("prompt_ids must be nonempty")
    if not completion_ids:
        raise PackingError("completion_ids must be nonempty")

    truncated = False
    keep_completion = completion_ids
    keep_prompt = prompt_ids
    if len(keep_prompt) + len(keep_completion) > max_length:
        truncated = True
        budget = max_length - len(keep_completion)
        if budget < 1:
            keep_completion = keep_completion[-(max_length - 1) :]
            keep_prompt = prompt_ids[-1:]
        else:
            keep_prompt = keep_prompt[-budget:]

    input_ids = keep_prompt + keep_completion
    labels = [ignore_index] * len(keep_prompt) + list(keep_completion)
    completion_mask = [0.0] * len(keep_prompt) + [1.0] * len(keep_completion)
    if len(input_ids) > max_length:
        raise PackingError("packed sequence exceeded max_length after truncation")
    if sum(completion_mask) < 1:
        raise PackingError("completion-only packing produced empty completion mask")
    return PackedSequence(
        input_ids=input_ids,
        labels=labels,
        completion_mask=completion_mask,
        prompt_token_count=len(keep_prompt),
        completion_token_count=len(keep_completion),
        truncated=truncated,
    )


def pack_batch(
    pairs: list[tuple[list[int], list[int]]],
    *,
    max_length: int,
) -> dict[str, Any]:
    packed = [
        pack_completion_only(prompt, completion, max_length=max_length)
        for prompt, completion in pairs
    ]
    return {
        "input_ids": [row.input_ids for row in packed],
        "labels": [row.labels for row in packed],
        "completion_mask": [row.completion_mask for row in packed],
        "truncated": [row.truncated for row in packed],
    }
