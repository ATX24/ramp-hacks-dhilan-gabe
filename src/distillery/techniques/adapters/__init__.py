"""Technique runtime adapters (built-in and external)."""

from __future__ import annotations

from distillery.techniques.adapters.builtin import (
    BuiltinLogitAdapter,
    BuiltinSequenceAdapter,
)
from distillery.techniques.adapters.external import ExternalContainerAdapter

__all__ = [
    "BuiltinLogitAdapter",
    "BuiltinSequenceAdapter",
    "ExternalContainerAdapter",
]
