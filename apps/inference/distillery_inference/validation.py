"""Task-specific structured-output validation aligned with the web Demo contract."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from distillery_inference.schemas import (
    TASK_OUTPUT_TYPES,
    FinanceTaskId,
    ValidationState,
)


def validate_structured_output(
    task: FinanceTaskId,
    output: dict[str, Any],
) -> tuple[ValidationState, str | None]:
    output_type = TASK_OUTPUT_TYPES[task]
    try:
        output_type.model_validate(output)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        first = errors[0] if errors else {"msg": "unknown schema violation", "loc": ()}
        location = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
        return "invalid", f"{location}: {first.get('msg', 'schema violation')}"
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
