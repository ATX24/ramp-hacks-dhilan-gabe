"""Task-specific structured-output validation aligned with the web Demo contract."""

from __future__ import annotations

import json
from typing import Any

from distillery_inference.schemas import (
    TASK_REQUIRED_FIELDS,
    TASK_SCHEMA_VERSIONS,
    FinanceTaskId,
    ValidationState,
)


def validate_structured_output(
    task: FinanceTaskId,
    output: dict[str, Any],
) -> tuple[ValidationState, str | None]:
    required = TASK_REQUIRED_FIELDS[task]
    missing = [field for field in required if field not in output]
    if missing:
        return "invalid", f"Missing required fields: {', '.join(missing)}"
    if output.get("task") != task:
        return "invalid", f"Output task {output.get('task')!r} does not match {task!r}"
    expected_version = TASK_SCHEMA_VERSIONS[task]
    if output.get("schema_version") != expected_version:
        return (
            "invalid",
            f"Unexpected schema_version {output.get('schema_version')!r}; "
            f"expected {expected_version!r}",
        )
    return "valid", None


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract a single JSON object from model text. Fail loud on ambiguity."""
    text = raw.strip()
    if not text:
        raise ValueError("empty model output")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model output does not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model output JSON must be an object")
    return payload
