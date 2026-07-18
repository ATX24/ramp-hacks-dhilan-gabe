"""Strict task-schema validation for teacher text and tool responses."""

from __future__ import annotations

import json
import re
from typing import Any, NoReturn

from distillery.contracts.hashing import canonical_json_bytes
from distillery.contracts.tasks import TaskId
from distillery.data.validate import validate_output
from distillery.teachers.errors import TeacherErrorCode, raise_teacher_error

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        _raise_malformed()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            _raise_malformed()
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            _raise_malformed()
    if not isinstance(value, dict):
        _raise_malformed()
    return value


def require_valid_task_response(*, task: TaskId, response: str | dict[str, Any]) -> str:
    payload = extract_json_object(response) if isinstance(response, str) else response
    validation = validate_output(task, payload)
    if not validation.ok:
        raise_teacher_error(
            TeacherErrorCode.SCHEMA_REJECTED,
            "teacher response failed the executable task schema",
            details={
                "task": task.value,
                "errors": list(validation.errors),
                "checks": list(validation.checks),
            },
        )
    return canonical_json_bytes(payload).decode("utf-8")


def _raise_malformed() -> NoReturn:
    raise_teacher_error(
        TeacherErrorCode.MALFORMED_JSON,
        "teacher response is not a JSON object",
    )


__all__ = ["extract_json_object", "require_valid_task_response"]
