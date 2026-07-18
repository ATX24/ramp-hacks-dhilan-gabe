"""One small interface shared by teacher adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from distillery.teachers.types import TeacherRecipe, TeacherRequest, TeacherResult


@runtime_checkable
class TeacherGenerator(Protocol):
    @property
    def provider_name(self) -> str:
        """Stable adapter identity."""

    def supports_recipe(self, recipe: TeacherRecipe | None) -> bool:
        """Whether the adapter supports this recipe or evaluation-only request."""

    def generate(self, request: TeacherRequest) -> TeacherResult:
        """Generate once under fail-closed policy and resource gates."""


__all__ = ["TeacherGenerator"]
