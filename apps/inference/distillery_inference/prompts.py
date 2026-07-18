"""Deterministic prompt construction for finance Demo tasks."""

from __future__ import annotations

import json
from typing import Any

from distillery_inference.schemas import TASK_REQUIRED_FIELDS, TASK_SCHEMA_VERSIONS, FinanceTaskId

SYSTEM_PREAMBLE = (
    "You are TinyFable, a Distillery finance-generalist model. "
    "Respond with a single JSON object only. No markdown fences."
)


def build_messages(
    *,
    task: FinanceTaskId,
    example_input: dict[str, Any],
) -> list[dict[str, str]]:
    schema_version = TASK_SCHEMA_VERSIONS[task]
    required = ", ".join(TASK_REQUIRED_FIELDS[task])
    user_payload = {
        "task": task,
        "schema_version": schema_version,
        "required_fields": list(TASK_REQUIRED_FIELDS[task]),
        "input": example_input,
        "instructions": (
            f"Emit JSON for {task} with schema_version={schema_version}. "
            f"Required fields: {required}."
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":")),
        },
    ]


def render_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Fallback plain prompt when a chat template is unavailable."""
    parts: list[str] = []
    for message in messages:
        parts.append(f"{message['role'].upper()}: {message['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)
